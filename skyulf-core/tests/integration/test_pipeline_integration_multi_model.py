"""Multi-model comparison integration tests over one preprocessed dataset.

Runs the SAME preprocessed train/test split of ``tests/data/pipeline_dataset.csv``
through several different classification and regression model nodes to catch
model-specific edge cases (e.g. a model that chokes on scaled-but-still-skewed
features, or one whose ``predict_proba`` output shape differs) that a
single-model integration test would miss. The model sweep itself is
data-driven via :class:`TestCaseLoader` (``tests/test_cases/modeling/*.json``)
since the scenario table is genuinely uniform: model name -> class paths to
exercise with identical preprocessing/split/assertions.
"""

import importlib
from typing import Any

import numpy as np
import pandas as pd
import pytest
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from tests.utils.dataset_loader import load_sample_dataset
from tests.utils.test_case_loader import TestCaseLoader

from skyulf.modeling.ensemble import (
    StackingRegressorApplier,
    StackingRegressorCalculator,
    VotingClassifierApplier,
    VotingClassifierCalculator,
)

_NUMERIC_COLS = ["age", "income", "tenure_months", "lat", "lon"]
_CATEGORICAL_COLS = ["city", "plan_type"]
_NON_FEATURE_COLS = ["customer_id", "signup_date"]
_CLASSIFICATION_TARGET = "churned"
_REGRESSION_TARGET = "monthly_spend"

_clf_params, _clf_scenarios, _clf_ids = TestCaseLoader("modeling/multi_model_smoke").load_with_ids()
_reg_params, _reg_scenarios, _reg_ids = TestCaseLoader(
    "modeling/multi_model_regression_smoke"
).load_with_ids()


def _import_from_path(dotted_path: str) -> Any:
    """Import a class given its fully-qualified dotted path.

    Args:
        dotted_path: e.g. ``"skyulf.modeling.classification.LogisticRegressionCalculator"``.

    Returns:
        The resolved class object.
    """
    module_path, class_name = dotted_path.rsplit(".", 1)
    module = importlib.import_module(module_path)
    return getattr(module, class_name)


def _preprocess(df: pd.DataFrame) -> pd.DataFrame:
    """Impute, encode, scale and drop non-feature columns.

    Numeric columns (`age`, `income`, `lat`, `lon`) are median-imputed;
    `city` nulls become an explicit "missing" category; `city`/`plan_type`
    are one-hot encoded; numeric columns are standard-scaled; `customer_id`
    and `signup_date` are dropped as non-feature columns. Both targets
    (`churned`, `monthly_spend`) are preserved untouched so the same frame
    can feed both the classification and regression sweeps.
    """
    out = df.drop(columns=_NON_FEATURE_COLS).copy()

    for col in _NUMERIC_COLS:
        out[col] = out[col].fillna(out[col].median())

    out["city"] = out["city"].fillna("missing")

    out = pd.get_dummies(out, columns=_CATEGORICAL_COLS)

    scaler = StandardScaler()
    out[_NUMERIC_COLS] = scaler.fit_transform(out[_NUMERIC_COLS])

    return out


@pytest.fixture(scope="module")
def preprocessed_split() -> tuple[pd.DataFrame, pd.DataFrame]:
    """Train/test split of the fully preprocessed `pipeline_dataset` sample.

    Shared across every model in the sweep so any pass/fail difference is
    attributable to the model, not to preprocessing or split variance.
    """
    df = load_sample_dataset("pipeline_dataset")
    processed = _preprocess(df)
    train_df, test_df = train_test_split(
        processed, test_size=0.25, random_state=42, stratify=processed[_CLASSIFICATION_TARGET]
    )
    return train_df, test_df


def _feature_columns(df: pd.DataFrame) -> list[str]:
    """All columns except the two known targets."""
    return [c for c in df.columns if c not in (_CLASSIFICATION_TARGET, _REGRESSION_TARGET)]


# ===========================================================================
# CLASSIFICATION MODEL SWEEP
# ===========================================================================


class TestClassificationModelSweep:
    """Same preprocessed split, swept across every classifier node."""

    @pytest.mark.parametrize(_clf_params, _clf_scenarios, ids=_clf_ids)
    def test_predict_returns_valid_binary_labels(
        self,
        preprocessed_split: tuple[pd.DataFrame, pd.DataFrame],
        model_key: str,
        calculator_path: str,
        applier_path: str,
        config: dict[str, Any],
    ) -> None:
        """`.predict()` returns 0/1 labels of the correct length for each classifier."""
        train_df, test_df = preprocessed_split
        feature_cols = _feature_columns(train_df)
        X_train, y_train = train_df[feature_cols], train_df[_CLASSIFICATION_TARGET]
        X_test = test_df[feature_cols]

        calculator = _import_from_path(calculator_path)()
        applier = _import_from_path(applier_path)()

        model = calculator.fit(X_train, y_train, config)
        preds = applier.predict(X_test, model)

        assert len(preds) == len(X_test), model_key
        assert set(pd.Series(preds).unique()).issubset({0, 1}), model_key

    @pytest.mark.parametrize(_clf_params, _clf_scenarios, ids=_clf_ids)
    def test_predict_proba_is_well_formed_when_available(
        self,
        preprocessed_split: tuple[pd.DataFrame, pd.DataFrame],
        model_key: str,
        calculator_path: str,
        applier_path: str,
        config: dict[str, Any],
    ) -> None:
        """Binary `predict_proba` rows are in [0, 1] and sum to ~1, when exposed."""
        train_df, test_df = preprocessed_split
        feature_cols = _feature_columns(train_df)
        X_train, y_train = train_df[feature_cols], train_df[_CLASSIFICATION_TARGET]
        X_test = test_df[feature_cols]

        calculator = _import_from_path(calculator_path)()
        applier = _import_from_path(applier_path)()

        model = calculator.fit(X_train, y_train, config)

        if not hasattr(applier, "predict_proba"):
            pytest.skip(f"{model_key} applier does not expose predict_proba")

        proba = applier.predict_proba(X_test, model)
        if proba is None:
            pytest.skip(f"{model_key} model does not support predict_proba")

        assert proba.shape == (len(X_test), 2), model_key
        assert ((proba.to_numpy() >= 0.0) & (proba.to_numpy() <= 1.0)).all(), model_key
        np.testing.assert_allclose(proba.sum(axis=1).to_numpy(), 1.0, atol=1e-6)


# ===========================================================================
# REGRESSION MODEL SWEEP
# ===========================================================================


class TestRegressionModelSweep:
    """Same preprocessed split, swept across every regressor node."""

    @pytest.mark.parametrize(_reg_params, _reg_scenarios, ids=_reg_ids)
    def test_predict_returns_finite_values_of_correct_length(
        self,
        preprocessed_split: tuple[pd.DataFrame, pd.DataFrame],
        model_key: str,
        calculator_path: str,
        applier_path: str,
        config: dict[str, Any],
    ) -> None:
        """`.predict()` returns finite floats of the correct length for each regressor."""
        train_df, test_df = preprocessed_split
        feature_cols = _feature_columns(train_df)
        X_train, y_train = train_df[feature_cols], train_df[_REGRESSION_TARGET]
        X_test = test_df[feature_cols]

        calculator = _import_from_path(calculator_path)()
        applier = _import_from_path(applier_path)()

        model = calculator.fit(X_train, y_train, config)
        preds = applier.predict(X_test, model)

        assert len(preds) == len(X_test), model_key
        assert np.isfinite(pd.Series(preds).to_numpy()).all(), model_key

    @pytest.mark.parametrize(_reg_params, _reg_scenarios, ids=_reg_ids)
    def test_predictions_beat_a_generous_mae_bar(
        self,
        preprocessed_split: tuple[pd.DataFrame, pd.DataFrame],
        model_key: str,
        calculator_path: str,
        applier_path: str,
        config: dict[str, Any],
    ) -> None:
        """MAE stays well below the naive "always predict train mean" baseline.

        The bar is intentionally generous (naive-baseline MAE, not some fixed
        absolute threshold) so it stays robust across model types/random
        initializations while still catching a genuinely broken predictor
        (e.g. one that returns near-constant or wildly-scaled output).
        """
        train_df, test_df = preprocessed_split
        feature_cols = _feature_columns(train_df)
        X_train, y_train = train_df[feature_cols], train_df[_REGRESSION_TARGET]
        X_test, y_test = test_df[feature_cols], test_df[_REGRESSION_TARGET]

        calculator = _import_from_path(calculator_path)()
        applier = _import_from_path(applier_path)()

        model = calculator.fit(X_train, y_train, config)
        preds = pd.Series(applier.predict(X_test, model)).to_numpy()

        naive_mae = float(np.abs(y_test.to_numpy() - y_train.mean()).mean())
        model_mae = float(np.abs(y_test.to_numpy() - preds).mean())

        assert model_mae < naive_mae, (model_key, model_mae, naive_mae)


# ===========================================================================
# ENSEMBLE MODELS
# ===========================================================================


class TestEnsembleModels:
    """Voting/Stacking meta-estimators wrapping 2+ base learners."""

    def test_voting_classifier_predicts_valid_labels(
        self, preprocessed_split: tuple[pd.DataFrame, pd.DataFrame]
    ) -> None:
        """VotingClassifier (soft vote over 3 base classifiers) predicts valid 0/1 labels."""
        train_df, test_df = preprocessed_split
        feature_cols = _feature_columns(train_df)
        X_train, y_train = train_df[feature_cols], train_df[_CLASSIFICATION_TARGET]
        X_test = test_df[feature_cols]

        calculator = VotingClassifierCalculator()
        applier = VotingClassifierApplier()
        config = {
            "params": {
                "base_estimators": ["logistic_regression", "random_forest", "decision_tree"],
                "voting": "soft",
            }
        }

        model = calculator.fit(X_train, y_train, config)
        preds = applier.predict(X_test, model)

        assert len(preds) == len(X_test)
        assert set(pd.Series(preds).unique()).issubset({0, 1})

        proba = applier.predict_proba(X_test, model)
        assert proba is not None
        np.testing.assert_allclose(proba.sum(axis=1).to_numpy(), 1.0, atol=1e-6)

    def test_stacking_regressor_predicts_finite_values(
        self, preprocessed_split: tuple[pd.DataFrame, pd.DataFrame]
    ) -> None:
        """StackingRegressor (2 base learners + ridge final estimator) predicts finite values."""
        train_df, test_df = preprocessed_split
        feature_cols = _feature_columns(train_df)
        X_train, y_train = train_df[feature_cols], train_df[_REGRESSION_TARGET]
        X_test = test_df[feature_cols]

        calculator = StackingRegressorCalculator()
        applier = StackingRegressorApplier()
        config = {
            "params": {
                "base_estimators": ["linear_regression", "random_forest"],
                "final_estimator": "ridge",
                "cv": 3,
            }
        }

        model = calculator.fit(X_train, y_train, config)
        preds = applier.predict(X_test, model)

        assert len(preds) == len(X_test)
        assert np.isfinite(pd.Series(preds).to_numpy()).all()
