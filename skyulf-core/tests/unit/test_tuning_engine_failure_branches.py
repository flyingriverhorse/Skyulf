"""Failure/edge-branch coverage for the tuning engine (Codecov patch follow-up).

Covers the optuna lazy-loader fallback chain and legacy module views,
param helpers (``_strip_model_prefix``/``_collect_trials``/nested
``set_params`` routing), the validation guard, threshold-tuning gates and
fallbacks, and ConvergenceWarning aggregation during ``fit()``.
"""

import logging
import sys
import types
from types import SimpleNamespace
from typing import Any, cast
from unittest.mock import MagicMock

import numpy as np
import pandas as pd
import pytest
from sklearn.datasets import make_classification

from skyulf.modeling._tuning import engine as engine_mod
from skyulf.modeling._tuning.engine import TuningCalculator
from skyulf.modeling._tuning.schemas import TuningConfig, TuningResult


class TestModuleGetattrLegacyViews:
    def test_legacy_names_reflect_optuna_state(self, monkeypatch):
        fresh = engine_mod._OptunaLoadState()
        fresh.attempted = True
        fresh.has_optuna = True
        fresh.optuna_module = "optuna-sentinel"
        fresh.search_cv = "search-cv-sentinel"
        monkeypatch.setattr(engine_mod, "_optuna_state", fresh)

        assert engine_mod.HAS_OPTUNA is True
        assert engine_mod.OptunaSearchCV == "search-cv-sentinel"
        assert engine_mod.optuna == "optuna-sentinel"
        assert engine_mod._optuna_load_attempted is True

    def test_unknown_name_raises_attribute_error(self):
        with pytest.raises(AttributeError):
            engine_mod.NO_SUCH_ATTRIBUTE  # noqa: B018 - bare access exercises module __getattr__


class TestEnsureOptunaLoadedFallbacks:
    def test_missing_optuna_package_marks_unavailable(self, monkeypatch):
        monkeypatch.setattr(engine_mod, "_optuna_state", engine_mod._OptunaLoadState())
        # `import optuna` treats a None sys.modules entry as a failed import.
        monkeypatch.setitem(sys.modules, "optuna", None)

        assert engine_mod._ensure_optuna_loaded() is False
        assert engine_mod._optuna_state.attempted is True
        assert engine_mod._optuna_state.has_optuna is False

    def test_integration_fallback_chain_warns_when_no_searchcv(self, monkeypatch, caplog):
        monkeypatch.setattr(engine_mod, "_optuna_state", engine_mod._OptunaLoadState())
        # Bar the first two integration import paths; the third
        # (optuna_integration) must also fail to reach the warning branch.
        monkeypatch.setitem(
            sys.modules, "optuna.integration", types.ModuleType("optuna.integration")
        )
        monkeypatch.setitem(
            sys.modules,
            "optuna.integration.sklearn",
            types.ModuleType("optuna.integration.sklearn"),
        )
        monkeypatch.setitem(
            sys.modules, "optuna_integration", types.ModuleType("optuna_integration")
        )
        # Earlier optuna tests in the suite may have cached the real submodule.
        monkeypatch.setitem(
            sys.modules,
            "optuna_integration.sklearn",
            types.ModuleType("optuna_integration.sklearn"),
        )

        with caplog.at_level(logging.WARNING):
            assert engine_mod._ensure_optuna_loaded() is False
        assert "OptunaSearchCV not found" in caplog.text
        # Memoized: a second call reuses the state without re-importing.
        assert engine_mod._ensure_optuna_loaded() is False


class TestParamHelpers:
    def test_strip_model_prefix_non_dict_passthrough(self):
        assert TuningCalculator._strip_model_prefix([("C", 1.0)]) == [("C", 1.0)]

    def test_strip_model_prefix_removes_pipeline_prefixes(self):
        stripped = TuningCalculator._strip_model_prefix(
            {"model__estimator__C": 1.0, "model__n_estimators": 10, "plain": 3}
        )
        assert stripped == {"C": 1.0, "n_estimators": 10, "plain": 3}

    def test_collect_trials_from_cv_results(self):
        searcher = SimpleNamespace(
            cv_results_={
                "params": [{"C": 1}, {"C": 2}],
                "mean_test_score": [0.8, 0.9],
            }
        )
        trials = TuningCalculator._collect_trials(searcher, TuningConfig(strategy="grid"))
        assert trials == [
            {"params": {"C": 1}, "score": 0.8},
            {"params": {"C": 2}, "score": 0.9},
        ]

    def test_collect_trials_without_results_is_empty(self):
        assert TuningCalculator._collect_trials(object(), TuningConfig(strategy="grid")) == []

    def test_instantiate_model_routes_nested_params_via_set_params(self):
        class _NestedModel:
            def __init__(self, n: int = 1) -> None:
                self.n = n

            def set_params(self, **kwargs):
                self.applied = kwargs
                return self

        model = TuningCalculator._instantiate_model(_NestedModel, {"n": 2, "sub__param": 5})
        assert model.n == 2
        assert model.applied == {"sub__param": 5}

    def test_validate_no_nan_inf_ignores_non_ndarray(self):
        # Plain lists bypass the check entirely (only ndarrays are scanned).
        TuningCalculator._validate_no_nan_inf([1.0, float("nan")], "nan", "inf", "obj-nan")

    def test_is_multiclass_target_branches(self):
        assert TuningCalculator._is_multiclass_target(np.array([0, 1, 2])) is True
        assert TuningCalculator._is_multiclass_target(np.array([0, 1])) is False
        assert TuningCalculator._is_multiclass_target(pd.Series(["a", "b", "c"])) is True
        assert TuningCalculator._is_multiclass_target([0, 1, 2]) is False

    def test_build_tuning_config_passthrough_and_key_filtering(self):
        cfg = TuningConfig(strategy="grid")
        assert TuningCalculator._build_tuning_config(cfg) is cfg

        built = TuningCalculator._build_tuning_config({"strategy": "grid", "not_a_field": 1})
        assert built.strategy == "grid"
        assert built.n_trials == 10


class TestThresholdMetricResolution:
    def test_probability_only_metric_falls_back_to_balanced_accuracy(self):
        tuner = TuningCalculator(MagicMock())
        logs: list[str] = []
        _, name = tuner._resolve_threshold_metric("roc_auc", logs.append)
        assert name == "balanced_accuracy"
        assert logs and "balanced_accuracy" in logs[0]

    def test_hard_label_metric_resolves_directly(self):
        tuner = TuningCalculator(MagicMock())
        callable_, name = tuner._resolve_threshold_metric("f1", None, pos_label="yes")
        assert name == "f1"
        assert callable_(np.array(["yes", "no"]), np.array(["yes", "no"])) == pytest.approx(1.0)


class TestDecisionThresholdGates:
    @staticmethod
    def _tuner(problem_type: str) -> TuningCalculator:
        return TuningCalculator(cast("Any", SimpleNamespace(problem_type=problem_type)))

    @staticmethod
    def _result() -> TuningResult:
        return TuningResult(best_params={}, best_score=0.0, n_trials=1, trials=[])

    def test_skip_gates_log_and_leave_thresholds_unset(self):
        config = TuningConfig()
        payload = (object(), object())

        logs: list[str] = []
        self._tuner("regression")._tune_decision_thresholds(
            object(), self._result(), config, payload, logs.append
        )
        assert "not a classification model" in logs[-1]

        self._tuner("classification")._tune_decision_thresholds(
            object(), self._result(), config, None, logs.append
        )
        assert "no validation split" in logs[-1]

        self._tuner("classification")._tune_decision_thresholds(
            SimpleNamespace(), self._result(), config, payload, logs.append
        )
        assert "predict_proba" in logs[-1]

        self._tuner("classification")._tune_decision_thresholds(
            SimpleNamespace(predict_proba=MagicMock()), self._result(), config, payload, logs.append
        )
        assert "classes_" in logs[-1]

        result = self._result()
        self._tuner("classification")._tune_decision_thresholds(
            SimpleNamespace(predict_proba=MagicMock(), classes_=[0, 1, 2]),
            result,
            config,
            payload,
            logs.append,
        )
        assert "binary" in logs[-1]
        assert result.decision_thresholds is None

    def test_search_failure_is_swallowed_and_logged(self):
        model = SimpleNamespace(predict_proba=MagicMock(), classes_=[0, 1])
        result = self._result()
        logs: list[str] = []
        # One-element tuple: the (X, y) unpack inside the try raises.
        self._tuner("classification")._tune_decision_thresholds(
            model, result, TuningConfig(), cast("Any", ("not-a-pair",)), logs.append
        )
        assert "Decision-threshold tuning failed" in logs[-1]
        assert result.decision_thresholds is None

    def test_successful_search_stores_thresholds_and_logs(self):
        def proba(X):
            p1 = np.clip(np.asarray(X, dtype=float)[:, 0] / 10.0, 0.0, 1.0)
            return np.column_stack([1.0 - p1, p1])

        model = SimpleNamespace(predict_proba=proba, classes_=[0, 1])
        X_val = pd.DataFrame({"f": [1.0, 2.0, 8.0, 9.0]})
        y_val = pd.Series([0, 0, 1, 1])
        result = self._result()
        logs: list[str] = []
        self._tuner("classification")._tune_decision_thresholds(
            model, result, TuningConfig(metric="f1"), (X_val, y_val), logs.append
        )
        assert result.decision_thresholds
        assert result.decision_threshold_metric == "f1"
        assert "selected threshold" in logs[-1]


class TestFitConvergenceAggregation:
    def test_convergence_warnings_surface_via_log_callback(self):
        from skyulf.modeling.classification import LogisticRegressionCalculator

        X_raw, y_raw = make_classification(
            n_samples=60, n_features=5, class_sep=0.2, random_state=42
        )
        # Unscaled features + tiny max_iter force ConvergenceWarning on every fit.
        X = pd.DataFrame(X_raw * 1000.0, columns=cast("Any", [f"f{i}" for i in range(5)]))
        y = pd.Series(y_raw)

        tuner = TuningCalculator(LogisticRegressionCalculator())
        logs: list[str] = []
        model = tuner.fit(
            X,
            y,
            {
                "strategy": "grid",
                "search_space": {"max_iter": [2, 3]},
                "cv_folds": 2,
                "metric": "accuracy",
            },
            log_callback=logs.append,
        )
        assert model is not None
        assert any("fully converge" in msg for msg in logs)
