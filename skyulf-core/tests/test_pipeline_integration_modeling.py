"""End-to-end pipeline integration tests: preprocessing -> split -> model -> evaluate.

These tests exercise the full node chain a real user would build against
``tests/data/pipeline_dataset.csv``: impute -> encode -> scale -> train/test
split -> fit a model -> predict -> evaluate. Unlike the unit tests in
``test_split.py``/``test_modeling_all.py`` (which test each node in
isolation), these tests verify the nodes compose correctly and that no
data leakage or row loss creeps in across the whole chain.
"""

from typing import Any, cast

import numpy as np
import pandas as pd
import pytest
from tests.utils.dataset_loader import load_sample_dataset

from skyulf.data.dataset import SplitDataset
from skyulf.modeling.base import StatefulEstimator
from skyulf.modeling.classification import (
    LogisticRegressionApplier,
    LogisticRegressionCalculator,
)
from skyulf.modeling.regression import LinearRegressionApplier, LinearRegressionCalculator
from skyulf.preprocessing.base import StatefulTransformer
from skyulf.preprocessing.encoding.one_hot import OneHotEncoderApplier, OneHotEncoderCalculator
from skyulf.preprocessing.imputation.simple import SimpleImputerApplier, SimpleImputerCalculator
from skyulf.preprocessing.scaling.standard import StandardScalerApplier, StandardScalerCalculator
from skyulf.preprocessing.split import DataSplitter

NUMERIC_FEATURES = ["age", "income", "tenure_months"]
CATEGORICAL_FEATURES = ["city", "plan_type"]
TARGETS = ["churned", "monthly_spend"]


def _preprocess_split(random_state: int = 42) -> tuple[SplitDataset, dict[str, Any]]:
    """Build a leakage-safe preprocessed train/test split of ``pipeline_dataset``.

    Chain: 80/20 train/test split -> impute (numeric mean, categorical mode) ->
    one-hot encode -> standard-scale numeric features. Every ``StatefulTransformer``
    fits its parameters on ``dataset.train`` only (per ``StatefulTransformer.
    _fit_transform_inner``) and then applies those same fitted parameters to
    both train and test, so test-set statistics can never leak into training.

    Returns the final ``SplitDataset`` (train/test still carry both target
    columns) plus a dict of the fitted artifacts from each step, so tests can
    assert on the exact leakage-free values that were used.
    """
    df = load_sample_dataset("pipeline_dataset")
    subset = df[NUMERIC_FEATURES + CATEGORICAL_FEATURES + TARGETS].copy()

    splitter = DataSplitter(test_size=0.2, random_state=random_state)
    split = splitter.split(subset)

    fitted: dict[str, Any] = {}

    numeric_imputer = StatefulTransformer(
        SimpleImputerCalculator(), SimpleImputerApplier(), node_id="numeric_imputer"
    )
    split = numeric_imputer.fit_transform(split, {"strategy": "mean", "columns": ["age", "income"]})
    split = cast(SplitDataset, split)
    fitted["numeric_imputer"] = numeric_imputer.params

    categorical_imputer = StatefulTransformer(
        SimpleImputerCalculator(), SimpleImputerApplier(), node_id="categorical_imputer"
    )
    split = cast(
        SplitDataset,
        categorical_imputer.fit_transform(
            split, {"strategy": "most_frequent", "columns": ["city"]}
        ),
    )
    fitted["categorical_imputer"] = categorical_imputer.params

    encoder = StatefulTransformer(
        OneHotEncoderCalculator(), OneHotEncoderApplier(), node_id="encoder"
    )
    split = cast(SplitDataset, encoder.fit_transform(split, {"columns": CATEGORICAL_FEATURES}))
    fitted["encoder"] = encoder.params

    scaler = StatefulTransformer(
        StandardScalerCalculator(), StandardScalerApplier(), node_id="scaler"
    )
    split = cast(SplitDataset, scaler.fit_transform(split, {"columns": NUMERIC_FEATURES}))
    fitted["scaler"] = scaler.params

    return split, fitted


class TestClassificationEndToEnd:
    """Full pipeline predicting the binary ``churned`` target."""

    def test_fit_predict_evaluate(self) -> None:
        """Preprocess -> split -> fit LogisticRegression -> predict -> evaluate.

        Verifies the whole chain runs without shape/type errors, that
        ``predict`` returns exactly one label per test row, and that all
        predicted labels are valid binary classes.
        """
        split, _ = _preprocess_split()
        assert isinstance(split.train, pd.DataFrame)
        assert isinstance(split.test, pd.DataFrame)
        # Drop the unrelated regression target so only the classification
        # target remains in the frame the estimator sees.
        train = split.train.drop(columns=["monthly_spend"])
        test = split.test.drop(columns=["monthly_spend"])
        dataset = SplitDataset(train=train, test=test, validation=None)

        estimator = StatefulEstimator(
            calculator=LogisticRegressionCalculator(),
            applier=LogisticRegressionApplier(),
            node_id="lr_churn",
        )
        predictions = estimator.fit_predict(dataset, target_column="churned", config={})

        assert len(predictions["test"]) == len(test)
        assert set(predictions["test"].unique()).issubset({0, 1})

        report = estimator.evaluate(dataset, target_column="churned")
        assert report["problem_type"] == "classification"
        accuracy = report["splits"]["test"].metrics["accuracy"]
        assert 0.0 <= accuracy <= 1.0
        f1 = report["splits"]["test"].metrics["f1"]
        assert 0.0 <= f1 <= 1.0


class TestRegressionEndToEnd:
    """Full pipeline predicting the continuous ``monthly_spend`` target."""

    def test_fit_predict_evaluate(self) -> None:
        """Preprocess -> split -> fit LinearRegression -> predict -> evaluate.

        Verifies the whole chain runs end to end, that ``predict`` returns
        exactly one finite float per test row, and that the reported error
        metric (RMSE) is finite and non-negative.
        """
        split, _ = _preprocess_split()
        assert isinstance(split.train, pd.DataFrame)
        assert isinstance(split.test, pd.DataFrame)
        # Drop the unrelated classification target for the regression run.
        train = split.train.drop(columns=["churned"])
        test = split.test.drop(columns=["churned"])
        dataset = SplitDataset(train=train, test=test, validation=None)

        estimator = StatefulEstimator(
            calculator=LinearRegressionCalculator(),
            applier=LinearRegressionApplier(),
            node_id="lr_spend",
        )
        predictions = estimator.fit_predict(dataset, target_column="monthly_spend", config={})

        assert len(predictions["test"]) == len(test)
        assert np.all(np.isfinite(predictions["test"]))

        report = estimator.evaluate(dataset, target_column="monthly_spend")
        assert report["problem_type"] == "regression"
        rmse = report["splits"]["test"].metrics["rmse"]
        assert np.isfinite(rmse)
        assert rmse >= 0.0


class TestSplitLeakagePrevention:
    """The most important check: preprocessing params must come from TRAIN only."""

    def test_imputer_and_scaler_params_reflect_train_statistics_only(self) -> None:
        """Imputer fill values and scaler mean/scale must equal TRAIN statistics.

        This is the critical data-leakage guard for the whole chain: if these
        preprocessing nodes fit on the full dataset (or on test) instead of
        train, the fitted mean/mode values below would not match a
        manually-computed train-only statistic and the pipeline would be
        silently peeking at test data during training.
        """
        raw = load_sample_dataset("pipeline_dataset")
        subset = raw[NUMERIC_FEATURES + CATEGORICAL_FEATURES + TARGETS].copy()

        # Reproduce the same train/test partition the pipeline itself uses,
        # so we can compute independent "ground truth" train-only statistics.
        splitter = DataSplitter(test_size=0.2, random_state=42)
        raw_split = splitter.split(subset)
        assert isinstance(raw_split.train, pd.DataFrame)
        assert isinstance(raw_split.test, pd.DataFrame)

        expected_age_mean = raw_split.train["age"].mean()
        expected_income_mean = raw_split.train["income"].mean()
        expected_city_mode = raw_split.train["city"].mode().iloc[0]

        # Sanity check: full-dataset / test-only statistics differ from the
        # train-only ones, otherwise this test couldn't actually detect leakage.
        full_age_mean = subset["age"].mean()
        test_age_mean = raw_split.test["age"].mean()
        assert expected_age_mean != pytest.approx(
            full_age_mean
        ) or expected_age_mean != pytest.approx(test_age_mean)

        _, fitted = _preprocess_split(random_state=42)

        numeric_fill_values = fitted["numeric_imputer"]["fill_values"]
        assert numeric_fill_values["age"] == pytest.approx(expected_age_mean)
        assert numeric_fill_values["income"] == pytest.approx(expected_income_mean)

        categorical_fill_values = fitted["categorical_imputer"]["fill_values"]
        assert categorical_fill_values["city"] == expected_city_mode

        # StandardScaler must have been fit on train-only (already-imputed)
        # numeric columns, i.e. its mean must match the imputed train mean,
        # not a mean computed over train+test.
        scaler_params = fitted["scaler"]
        scaler_cols = scaler_params["columns"]
        age_idx = scaler_cols.index("age")
        assert scaler_params["mean"][age_idx] == pytest.approx(expected_age_mean)

    def test_test_split_missing_values_filled_from_train_stats_not_test_stats(self) -> None:
        """Test-set NaNs must be filled using TRAIN means, never test-only means.

        Directly inspects a row that was NaN in the test split before
        imputation and confirms the filled value equals the train mean. The
        check runs *before* the scaler step (which the full chain applies
        afterwards) so the raw imputed value can be compared directly against
        the train-only mean without the scaling transform in the way.
        """
        raw = load_sample_dataset("pipeline_dataset")
        subset = raw[NUMERIC_FEATURES + CATEGORICAL_FEATURES + TARGETS].copy()
        splitter = DataSplitter(test_size=0.2, random_state=42)
        raw_split = splitter.split(subset)
        assert isinstance(raw_split.train, pd.DataFrame)
        assert isinstance(raw_split.test, pd.DataFrame)

        test_nan_mask = raw_split.test["age"].isna()
        assert test_nan_mask.any(), "Expected at least one NaN 'age' value in the test split"

        expected_age_mean = raw_split.train["age"].mean()
        full_dataset_age_mean = subset["age"].mean()
        # Guard against a false-positive pass: if train/full means coincided,
        # this test couldn't distinguish "fit on train" from "fit on everything".
        assert expected_age_mean != pytest.approx(full_dataset_age_mean)

        numeric_imputer = StatefulTransformer(
            SimpleImputerCalculator(), SimpleImputerApplier(), node_id="numeric_imputer"
        )
        imputed_split = cast(
            SplitDataset,
            numeric_imputer.fit_transform(
                raw_split, {"strategy": "mean", "columns": ["age", "income"]}
            ),
        )
        assert isinstance(imputed_split.test, pd.DataFrame)
        filled_ages = imputed_split.test.loc[test_nan_mask.index[test_nan_mask], "age"]
        np.testing.assert_allclose(filled_ages.to_numpy(), expected_age_mean)


class TestMissingValueRowPreservation:
    """Confirms the pipeline never silently drops rows that contained NaNs."""

    def test_row_counts_preserved_through_preprocessing_and_prediction(self) -> None:
        """Row counts must be identical before/after preprocessing and at predict time.

        ``pipeline_dataset.csv`` has NaNs in ``age``/``income``/``city``. The
        imputer is expected to *fill* those values, not drop the rows. This
        test confirms train/test row counts are unchanged by the
        impute -> encode -> scale chain, and that the model's predictions
        array has exactly one entry per input row (no rows lost along the way).
        """
        df = load_sample_dataset("pipeline_dataset")
        subset = df[NUMERIC_FEATURES + CATEGORICAL_FEATURES + TARGETS].copy()
        assert subset.isna().any().any(), "Fixture must contain NaNs to exercise this check"

        splitter = DataSplitter(test_size=0.2, random_state=42)
        raw_split = splitter.split(subset)
        expected_train_rows = len(raw_split.train)
        expected_test_rows = len(raw_split.test)

        split, _ = _preprocess_split(random_state=42)
        assert isinstance(split.train, pd.DataFrame)
        assert isinstance(split.test, pd.DataFrame)
        # Row counts must survive imputation/encoding/scaling untouched.
        assert len(split.train) == expected_train_rows
        assert len(split.test) == expected_test_rows
        # No NaNs should remain in the imputed numeric/categorical features.
        assert not split.train[["age", "income"]].isna().any().any()
        assert not split.test[["age", "income"]].isna().any().any()

        train = split.train.drop(columns=["monthly_spend"])
        test = split.test.drop(columns=["monthly_spend"])
        dataset = SplitDataset(train=train, test=test, validation=None)

        estimator = StatefulEstimator(
            calculator=LogisticRegressionCalculator(),
            applier=LogisticRegressionApplier(),
            node_id="lr_churn_rowcheck",
        )
        predictions = estimator.fit_predict(dataset, target_column="churned", config={})

        assert len(predictions["train"]) == expected_train_rows
        assert len(predictions["test"]) == expected_test_rows
