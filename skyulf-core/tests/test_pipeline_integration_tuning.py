"""Integration tests for hyperparameter tuning wrapped around a real, preprocessed pipeline.

Unlike ``tests/test_tuning.py`` / ``tests/test_tuning_engine.py`` (which mostly
exercise ``TuningCalculator`` in isolation against tiny synthetic arrays),
these tests run the full flow — impute -> scale -> one-hot encode -> train/test
split -> tuned model fit -> best-params retrieval -> predict on held-out data
— against the 300-row ``pipeline_dataset.csv`` fixture (see
``tests/utils/dataset_loader.py``), for random, grid, and (when available)
Optuna-backed ("bayes") search strategies.
"""

from typing import Any, Dict

import numpy as np
import pandas as pd
import pytest
from sklearn.metrics import get_scorer
from sklearn.model_selection import KFold, train_test_split
from tests.utils.dataset_loader import load_sample_dataset

from skyulf.modeling._tuning.engine import TuningCalculator
from skyulf.modeling._tuning.schemas import TuningConfig
from skyulf.modeling.classification import RandomForestClassifierCalculator
from skyulf.modeling.regression import RidgeRegressionCalculator
from skyulf.preprocessing.encoding.one_hot import (
    OneHotEncoderApplier,
    OneHotEncoderCalculator,
)
from skyulf.preprocessing.imputation.simple import (
    SimpleImputerApplier,
    SimpleImputerCalculator,
)
from skyulf.preprocessing.scaling.standard import (
    StandardScalerApplier,
    StandardScalerCalculator,
)


def _fit_apply(
    calculator: Any, applier: Any, df: pd.DataFrame, config: Dict[str, Any]
) -> pd.DataFrame:
    """Run ``calculator.fit`` then ``applier.apply`` and return the transformed frame."""
    params = calculator.fit(df, config)
    return applier.apply(df, dict(params))


def _preprocess(df: pd.DataFrame) -> pd.DataFrame:
    """Impute numeric/categorical NaNs, scale numerics, and one-hot encode ``plan_type``.

    Produces a fully-numeric, NaN-free frame suitable for feeding directly into
    a scikit-learn estimator (which ``TuningCalculator`` wraps), mirroring how
    a real pipeline would prepare data before a modeling node.
    """
    imputed = _fit_apply(
        SimpleImputerCalculator(),
        SimpleImputerApplier(),
        df,
        {"strategy": "mean", "columns": ["age", "income"]},
    )
    scaled = _fit_apply(
        StandardScalerCalculator(),
        StandardScalerApplier(),
        imputed,
        {"columns": ["age", "income", "tenure_months"]},
    )
    encoded = _fit_apply(
        OneHotEncoderCalculator(),
        OneHotEncoderApplier(),
        scaled,
        {"columns": ["plan_type"]},
    )
    return encoded


def _classification_xy() -> tuple:
    """Return preprocessed (X, y) for the ``churned`` classification target."""
    df = _preprocess(load_sample_dataset("pipeline_dataset"))
    feature_cols = ["age", "income", "tenure_months"] + [
        c for c in df.columns if c.startswith("plan_type_")
    ]
    return df[feature_cols], df["churned"]


def _regression_xy() -> tuple:
    """Return preprocessed (X, y) for the ``monthly_spend`` regression target."""
    df = _preprocess(load_sample_dataset("pipeline_dataset"))
    feature_cols = ["age", "income", "tenure_months"] + [
        c for c in df.columns if c.startswith("plan_type_")
    ]
    return df[feature_cols], df["monthly_spend"]


class TestRandomSearchClassification:
    """Random search tuning of a RandomForestClassifier for the ``churned`` target."""

    def test_random_search_returns_valid_best_params_and_finite_test_score(self) -> None:
        """Random search should pick params from the declared space and yield a usable model.

        Verifies: best-params keys match the search space, the refit-on-full-
        training-data model can predict on a held-out test split, and the
        resulting test accuracy is a finite value in [0, 1].
        """
        X, y = _classification_xy()
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=0.25, random_state=42, stratify=y
        )

        tuner = TuningCalculator(RandomForestClassifierCalculator())
        config = TuningConfig(
            strategy="random",
            metric="accuracy",
            search_space={"n_estimators": [10, 20], "max_depth": [3, 5]},
            n_trials=4,
            cv_folds=3,
            random_state=42,
        )
        model, result = tuner.fit(X_train, y_train, config=config.__dict__)

        # Best params must come from the declared search space.
        assert set(result.best_params.keys()) <= {"n_estimators", "max_depth"}
        assert result.best_params["n_estimators"] in [10, 20]
        assert result.best_params["max_depth"] in [3, 5]
        assert np.isfinite(result.best_score)

        # Refit-on-full-train-data model must be immediately usable for prediction.
        assert hasattr(model, "predict")
        preds = model.predict(X_test)
        assert len(preds) == len(X_test)

        test_accuracy = (preds == y_test.to_numpy()).mean()
        assert np.isfinite(test_accuracy)
        assert 0.0 <= test_accuracy <= 1.0


class TestGridSearchRegression:
    """Grid search tuning of a RidgeRegression for the ``monthly_spend`` target."""

    def _run(self) -> tuple:
        X, y = _regression_xy()
        tuner = TuningCalculator(RidgeRegressionCalculator())
        config = TuningConfig(
            strategy="grid",
            metric="r2",
            search_space={"alpha": [0.01, 0.1, 1.0, 10.0]},
            cv_folds=4,
            random_state=42,
        )
        model, result = tuner.fit(X, y, config=config.__dict__)
        return X, y, model, result

    def test_grid_search_is_deterministic_across_runs(self) -> None:
        """Same search space + same data + same seed must yield identical best params."""
        _, _, _, result_1 = self._run()
        _, _, _, result_2 = self._run()

        assert result_1.best_params == result_2.best_params
        assert result_1.best_score == pytest.approx(result_2.best_score)

    def test_grid_search_actually_picks_the_best_scoring_combination(self) -> None:
        """The tuner's reported best trial must be the true max across all grid points.

        Cross-checks the tuner's own ``result.trials`` (internal consistency:
        best_score must equal max(trial scores)) AND independently recomputes
        each alpha's mean CV r2 with a manually-built ``KFold`` matching the
        tuner's default CV settings (k_fold, shuffle=True, random_state=42),
        to guard against the tuner silently picking a non-optimal candidate.
        """
        X, y, _, result = self._run()

        # 1) Internal consistency: reported best must equal the max of all trials.
        assert len(result.trials) == 4
        trial_scores = {t["params"]["alpha"]: t["score"] for t in result.trials}
        best_alpha_by_trials = max(trial_scores, key=lambda a: trial_scores[a])
        assert result.best_params["alpha"] == best_alpha_by_trials
        assert result.best_score == pytest.approx(max(trial_scores.values()))

        # 2) Independent recomputation using a manually-built matching KFold CV.
        from sklearn.linear_model import Ridge

        X_arr = X.to_numpy()
        y_arr = y.to_numpy()
        scorer = get_scorer("r2")
        manual_scores: Dict[float, float] = {}
        for alpha in [0.01, 0.1, 1.0, 10.0]:
            cv = KFold(n_splits=4, shuffle=True, random_state=42)
            fold_scores = []
            for train_idx, val_idx in cv.split(X_arr, y_arr):
                model = Ridge(alpha=alpha)
                model.fit(X_arr[train_idx], y_arr[train_idx])
                fold_scores.append(scorer(model, X_arr[val_idx], y_arr[val_idx]))
            manual_scores[alpha] = float(np.mean(fold_scores))

        best_alpha_manual = max(manual_scores, key=lambda a: manual_scores[a])
        assert result.best_params["alpha"] == best_alpha_manual
        # The tuner's reported score for the winning alpha should closely match
        # our independent recomputation of the same CV procedure.
        assert result.best_score == pytest.approx(manual_scores[best_alpha_manual], abs=1e-9)
        # And the winner must be >= every other grid point's manually-recomputed score.
        for score in manual_scores.values():
            assert manual_scores[best_alpha_manual] >= score - 1e-12


class TestBayesSearchSmoke:
    """Smoke test for the Optuna-backed ("bayes") tuning strategy."""

    def test_optuna_strategy_runs_and_returns_params_within_bounds(self) -> None:
        """A tiny-budget Optuna search should run without error and stay within bounds.

        If Optuna isn't installed, this is skipped via ``importorskip`` rather
        than failing, since it is an optional dependency for this strategy.
        """
        pytest.importorskip("optuna")
        X, y = _classification_xy()

        tuner = TuningCalculator(RandomForestClassifierCalculator())
        config = TuningConfig(
            strategy="optuna",
            metric="accuracy",
            search_space={"n_estimators": [10, 20, 30], "max_depth": [2, 3, 4]},
            n_trials=6,
            cv_folds=3,
            random_state=42,
        )
        model, result = tuner.fit(X, y, config=config.__dict__)

        assert hasattr(model, "predict")
        assert result.n_trials > 0
        assert result.best_params["n_estimators"] in [10, 20, 30]
        assert result.best_params["max_depth"] in [2, 3, 4]
        assert np.isfinite(result.best_score)


class TestDegenerateSearchSpace:
    """Edge cases: a search space with no real variation should still work."""

    def test_single_valued_search_space_returns_valid_single_result(self) -> None:
        """A search space where every hyperparameter has only one possible value.

        With no actual choice to make, the tuner must still complete cleanly
        and report that single combination as the (only) best result, rather
        than crashing on an empty/degenerate grid.
        """
        X, y = _classification_xy()

        tuner = TuningCalculator(RandomForestClassifierCalculator())
        config = TuningConfig(
            strategy="grid",
            metric="accuracy",
            search_space={"n_estimators": [15], "max_depth": [3]},
            cv_folds=3,
            random_state=42,
        )
        model, result = tuner.fit(X, y, config=config.__dict__)

        assert len(result.trials) == 1
        assert result.best_params == {"n_estimators": 15, "max_depth": 3}
        assert np.isfinite(result.best_score)
        assert hasattr(model, "predict")

    def test_random_search_with_n_iter_one_returns_valid_single_result(self) -> None:
        """``n_trials=1`` random search over a real space must still produce one valid result."""
        X, y = _regression_xy()

        tuner = TuningCalculator(RidgeRegressionCalculator())
        config = TuningConfig(
            strategy="random",
            metric="r2",
            search_space={"alpha": [0.1, 1.0, 10.0]},
            n_trials=1,
            cv_folds=3,
            random_state=42,
        )
        model, result = tuner.fit(X, y, config=config.__dict__)

        assert len(result.trials) == 1
        assert result.best_params["alpha"] in [0.1, 1.0, 10.0]
        assert np.isfinite(result.best_score)
        assert hasattr(model, "predict")
