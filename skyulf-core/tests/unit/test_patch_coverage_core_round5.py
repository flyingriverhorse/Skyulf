"""Round-5 patch-coverage tests for skyulf-core defensive branches (Codecov follow-up).

Closes the partial branches and missing line flagged by the fifth Codecov
patch report: tuning-engine log-callback/fold-error fallback directions,
the empty-candidate tuning failure, the SHAP predicted-class except path,
the categorical moderate-PSI suggestion, and every balance-recommendation
direction.
"""

from types import SimpleNamespace
from typing import Any, cast
from unittest.mock import MagicMock

import numpy as np
import pandas as pd
import pytest
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import KFold

from skyulf.modeling._explainability.shap_explanation import _predicted_class_index
from skyulf.modeling._tuning.engine import TuningCalculator
from skyulf.modeling._tuning.schemas import TuningConfig, TuningResult
from skyulf.profiling._analyzer.recommendations import RecommendationsMixin
from skyulf.profiling.drift import DriftCalculator


def _raiser(exc: Exception):
    def fn(*args, **kwargs):
        raise exc

    return fn


def _tuner(problem_type: str = "classification") -> TuningCalculator:
    return TuningCalculator(cast("Any", SimpleNamespace(problem_type=problem_type)))


class TestTuningEngineCallbackAndErrorBranches:
    def test_threshold_metric_fallback_without_log_callback(self):
        callable_, name = _tuner()._resolve_threshold_metric("roc_auc", None)
        assert name == "balanced_accuracy"
        assert callable_(np.array([0, 1]), np.array([0, 1])) == pytest.approx(1.0)

    def test_threshold_search_failure_without_log_callback(self):
        result = TuningResult(best_params={}, best_score=0.0, n_trials=1, trials=[])
        # One-element tuple: the (X, y) unpack inside the try raises.
        _tuner()._tune_decision_thresholds(
            SimpleNamespace(predict_proba=MagicMock(), classes_=[0, 1]),
            result,
            TuningConfig(),
            cast("Any", ("not-a-pair",)),
            None,
        )
        assert result.decision_thresholds is None

    def test_fold_failure_without_fold_errors_list(self):
        tuner = _tuner()
        df = pd.DataFrame({"f": [1.0, 2.0, 3.0, 4.0]})
        y = pd.Series([0, 1, 0, 1])
        # SimpleNamespace has no default_params -> the dict expansion inside
        # the try raises, exercising the except with fold_errors=None.
        score = tuner._fit_and_score_candidate_fold(
            0,
            0,
            {},
            object,
            KFold(n_splits=2),
            df,
            y,
            df.to_numpy(),
            y.to_numpy(),
            [0, 1],
            [2, 3],
            "accuracy",
            None,
            fold_errors=None,
        )
        assert score == -float("inf")

    def test_all_trials_failed_with_empty_fold_errors(self, monkeypatch):
        from skyulf.modeling._tuning import engine as engine_mod
        from skyulf.modeling.classification import LogisticRegressionCalculator

        tuner = TuningCalculator(LogisticRegressionCalculator())
        # Zero candidates -> best_params stays None without a single fold
        # error being collected, exercising the detail-less failure message.
        monkeypatch.setattr(
            engine_mod.TuningCalculator, "_generate_search_candidates", lambda self, config: []
        )
        X = pd.DataFrame({"a": [1.0, 2.0, 3.0, 4.0], "b": [4.0, 3.0, 2.0, 1.0]})
        y = pd.Series([0, 1, 0, 1])
        config = TuningConfig(strategy="grid", search_space={"max_iter": [2]}, cv_folds=2)
        with pytest.raises(ValueError) as excinfo:
            tuner._run_grid_or_random_search(
                X, y, config, LogisticRegression, KFold(n_splits=2), "accuracy", None, None
            )
        assert "All trials failed" in str(excinfo.value)
        assert "First trial error" not in str(excinfo.value)


class TestShapPredictedClassIndexFallback:
    def test_predict_failure_falls_back_to_class_zero(self):
        model = SimpleNamespace(classes_=[0, 1, 2], predict=_raiser(RuntimeError("no predict")))
        idx = _predicted_class_index(model, pd.DataFrame({"a": [1.0, 2.0]}), 3)
        assert idx.tolist() == [0, 0]


class TestCategoricalDriftSuggestions:
    def test_moderate_psi_branch(self):
        critical = DriftCalculator._categorical_drift_suggestions(True, 0.3)
        assert any("Critical" in m for m in critical)

        moderate = DriftCalculator._categorical_drift_suggestions(True, 0.15)
        assert any("Moderate" in m for m in moderate)

        assert DriftCalculator._categorical_drift_suggestions(False, 0.3) == []


class TestBalanceRecommendationDirections:
    def test_balanced_imbalanced_and_neutral_ratios(self):
        mixin = object.__new__(RecommendationsMixin)

        balanced = mixin._build_balance_recommendation("target", 0.9)
        assert balanced[0].reason == "Balanced Target"

        imbalanced = mixin._build_balance_recommendation("target", 0.1)
        assert imbalanced[0].reason == "Imbalanced Target"

        assert mixin._build_balance_recommendation("target", 0.5) == []

    def test_skewness_threshold_both_directions(self):
        mixin = object.__new__(RecommendationsMixin)
        skewed = SimpleNamespace(numeric_stats=SimpleNamespace(skewness=2.0))
        mild = SimpleNamespace(numeric_stats=SimpleNamespace(skewness=0.5))

        recs = mixin._skewness_recommendations("col", cast("Any", skewed))
        assert len(recs) == 1

        assert mixin._skewness_recommendations("col", cast("Any", mild)) == []
