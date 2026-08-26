"""Tests for skyulf.modeling._boosting_progress iteration adapters."""

import types
from typing import Any, cast

import numpy as np
import pandas as pd
import pytest

from skyulf.modeling._boosting_progress import (
    LightGBMIterationAdapter,
    XgboostIterationAdapter,
    direction_for_xgb_metric,
)


def _lgbm_env(iteration: int, total: int, results: list[tuple[str, str, float, bool]]) -> Any:
    return types.SimpleNamespace(
        iteration=iteration,
        begin_iteration=0,
        end_iteration=total,
        evaluation_result_list=results,
    )


class TestDirectionHeuristic:
    def test_losses_minimize(self):
        for metric in ("logloss", "rmse", "mae", "error", "poisson-nloglik"):
            assert direction_for_xgb_metric(metric) == "minimize"

    def test_auc_family_maximizes(self):
        for metric in ("auc", "aucpr", "map"):
            assert direction_for_xgb_metric(metric) == "maximize"


class TestXgboostAdapter:
    def test_reports_one_based_iteration_with_direction(self):
        assert XgboostIterationAdapter is not None
        seen: list[tuple] = []
        adapter = XgboostIterationAdapter(lambda *args: seen.append(args), total=10)
        evals_log = {"validation_0": {"logloss": [0.5, 0.45, 0.4123]}}

        stop = adapter.after_iteration(None, 4, evals_log)

        assert stop is False
        assert seen == [(5, 10, pytest.approx(0.4123), "logloss", "minimize")]

    def test_auc_metric_maximizes(self):
        assert XgboostIterationAdapter is not None
        seen: list[tuple] = []
        adapter = XgboostIterationAdapter(lambda *args: seen.append(args), total=3)
        adapter.after_iteration(None, 0, {"validation_0": {"auc": [0.8]}})
        assert seen[0][3:] == ("auc", "maximize")

    def test_empty_evals_log_no_callback(self):
        assert XgboostIterationAdapter is not None
        seen: list[tuple] = []
        adapter = XgboostIterationAdapter(lambda *args: seen.append(args), total=3)
        stop = adapter.after_iteration(None, 1, {})
        assert stop is False
        assert seen == []

    def test_callback_exception_swallowed(self):
        assert XgboostIterationAdapter is not None

        def boom(*_: Any) -> None:
            raise RuntimeError("downstream blew up")

        adapter = XgboostIterationAdapter(boom, total=2)
        stop = adapter.after_iteration(None, 0, {"validation_0": {"rmse": [1.5]}})
        assert stop is False


class TestLightGBMAdapter:
    def test_reports_one_based_iteration_using_is_higher_better(self):
        seen: list[tuple] = []
        adapter = LightGBMIterationAdapter(lambda *args: seen.append(args))
        env = _lgbm_env(6, 20, [("valid_0", "binary_logloss", 0.31, False)])

        adapter(env)

        assert seen == [(7, 20, pytest.approx(0.31), "binary_logloss", "minimize")]

    def test_higher_better_metric_maximizes(self):
        seen: list[tuple] = []
        adapter = LightGBMIterationAdapter(lambda *args: seen.append(args))
        adapter(_lgbm_env(0, 5, [("valid_0", "auc", 0.9, True)]))
        assert seen[0][3:] == ("auc", "maximize")

    def test_empty_results_no_callback(self):
        seen: list[tuple] = []
        adapter = LightGBMIterationAdapter(lambda *args: seen.append(args))
        adapter(_lgbm_env(0, 5, []))
        assert seen == []

    def test_callback_exception_swallowed(self):
        def boom(*_: Any) -> None:
            raise RuntimeError("downstream blew up")

        adapter = LightGBMIterationAdapter(boom)
        adapter(_lgbm_env(0, 2, [("valid_0", "l2", 1.5, False)]))


def _binary_xy(n: int = 200) -> tuple[np.ndarray, np.ndarray]:
    rng = np.random.default_rng(42)
    X = rng.normal(size=(n, 4))
    y = ((X[:, 0] + X[:, 1] * 0.5 + rng.normal(scale=0.25, size=n)) > 0).astype(int)
    return X, y


@pytest.mark.parametrize("node_id", ["xgboost_classifier", "lgbm_classifier"])
class TestRealBoostingFitEmitsIterations:
    """Real fits through the registered calculators: callback fires once per
    iteration and the trained model is identical with or without it."""

    def test_fit_emits_per_iteration_and_model_unchanged(self, node_id: str):
        from skyulf import NodeRegistry

        X, y = _binary_xy()
        config = {"params": {"n_estimators": 12}}
        calc = NodeRegistry.get_calculator(node_id)()

        plain = calc.fit(pd.DataFrame(X), pd.Series(y), config)

        seen: list[tuple] = []
        with_cb = calc.fit(
            pd.DataFrame(X),
            pd.Series(y),
            config,
            iteration_callback=lambda *args: seen.append(args),
        )

        assert len(seen) == 12
        assert [c for c, *_ in seen] == list(range(1, 13))
        assert [t for _, t, *_ in seen] == [12] * 12
        assert all(isinstance(s, float) for _, _, s, *_ in seen)
        assert all(m for *_, m, _d in seen)
        assert np.allclose(
            plain.predict_proba(X) if hasattr(plain, "predict_proba") else plain.predict(X),
            with_cb.predict_proba(X) if hasattr(with_cb, "predict_proba") else with_cb.predict(X),
        )

    def test_no_callback_means_plain_fit(self, node_id: str):
        from skyulf import NodeRegistry

        X, y = _binary_xy(120)
        calc = NodeRegistry.get_calculator(node_id)()
        model = calc.fit(
            pd.DataFrame(X),
            pd.Series(y),
            {"params": {"n_estimators": 5}},
            iteration_callback=None,
        )
        assert model is not None


class _FitSpy:
    """Minimal booster-shaped model for the refit wiring test."""

    def __init__(self) -> None:
        self.fit_kwargs: dict[str, Any] = {}

    def fit(self, X, y, **kwargs):
        self.fit_kwargs = kwargs
        for callback in kwargs.get("callbacks", []):
            for i in range(3):
                after = getattr(callback, "after_iteration", None)
                if after is not None:
                    after(
                        self,
                        i,
                        {"validation_0": {"logloss": [0.5 - j * 0.1 for j in range(i + 1)]}},
                    )
                elif callable(callback):
                    callback(_lgbm_env(i, 3, [("valid_0", "logloss", 0.5 - i * 0.1, False)]))
        return self


class _FakeBoostingCalculator:
    default_params: dict[str, Any] = {}
    model_class = _FitSpy

    def _boosting_fit_kwargs(self, model, X, y, iteration_callback):
        if iteration_callback is None:
            return {}
        return {
            "eval_set": [(X, y)],
            "callbacks": [LightGBMIterationAdapter(iteration_callback)],
        }


class TestRefitForwardsIterationCallback:
    """TuningCalculator._refit_best_model must apply the calculator's boosting
    fit kwargs when an iteration_callback is supplied — and stay plain without."""

    def _refit(self, iteration_callback):
        from skyulf.modeling._tuning.engine import TuningCalculator
        from skyulf.modeling._tuning.schemas import TuningConfig, TuningResult
        from skyulf.modeling.base import BaseModelCalculator

        tuner = TuningCalculator(cast("BaseModelCalculator", _FakeBoostingCalculator()))
        result = TuningResult(
            best_params={}, best_score=0.9, n_trials=1, trials=[], scoring_metric="accuracy"
        )
        X = np.zeros((4, 2))
        y = np.array([0, 1, 0, 1])
        return tuner._refit_best_model(
            result, TuningConfig(), X, y, None, iteration_callback=iteration_callback
        )

    def test_refit_fires_iteration_callback(self):
        calls: list[tuple] = []
        model = self._refit(lambda *args: calls.append(args))

        assert "eval_set" in model.fit_kwargs
        assert len(calls) == 3
        assert [c for c, *_ in calls] == [1, 2, 3]

    def test_refit_without_callback_stays_plain(self):
        model = self._refit(None)

        assert "callbacks" not in model.fit_kwargs
        assert "eval_set" not in model.fit_kwargs


class TestEndToEndStreaming:
    """Real boosting calculators streaming through tuning and plain fits —
    covers the refit hook invocation, callback detach, and the regression
    XGB adapter path that the fake-calculator tests cannot reach."""

    def test_xgb_tuning_refit_streams_iterations_and_detaches_callbacks(self):
        pytest.importorskip("xgboost")
        from sklearn.datasets import make_classification

        from skyulf.modeling._tuning.engine import TuningCalculator
        from skyulf.modeling._tuning.schemas import TuningConfig
        from skyulf.modeling.classification import XGBClassifierCalculator

        X_arr, y_arr = make_classification(n_samples=120, n_features=4, random_state=0)
        points: list[tuple] = []
        model, _result = TuningCalculator(XGBClassifierCalculator()).fit(
            pd.DataFrame(X_arr),
            pd.Series(y_arr, name="target"),
            config=TuningConfig(
                strategy="grid",
                metric="roc_auc",
                search_space={"max_depth": [2]},
                cv_folds=2,
            ),
            iteration_callback=lambda *args: points.append(args),
        )

        assert points, "boosting fits inside tuning must stream iteration points"
        assert getattr(model, "callbacks", None) is None, (
            "XGBoost callbacks must be detached from the saved artifact"
        )

    def test_xgb_regression_fit_streams_iterations(self):
        pytest.importorskip("xgboost")
        from sklearn.datasets import make_regression

        from skyulf.modeling.regression import XGBRegressorCalculator

        X_arr, y_arr = make_regression(n_samples=120, n_features=4, random_state=0)
        points: list[tuple] = []
        XGBRegressorCalculator().fit(
            pd.DataFrame(X_arr),
            pd.Series(y_arr, name="target"),
            {"n_estimators": 12},
            iteration_callback=lambda *args: points.append(args),
        )

        assert points, "a plain XGB regression fit must stream iteration points"
        assert [p[0] for p in points] == list(range(1, 13))

    def test_optuna_trial_failures_are_forwarded_to_log_callback(self, monkeypatch):
        pytest.importorskip("optuna")

        from sklearn.datasets import make_classification

        from skyulf.modeling._tuning.engine import TuningCalculator
        from skyulf.modeling._tuning.fold_pipeline import FoldAwareModelStep
        from skyulf.modeling._tuning.schemas import TuningConfig
        from skyulf.modeling.classification import LogisticRegressionCalculator

        def _explode(self, X, y=None):
            raise RuntimeError("synthetic fold failure")

        monkeypatch.setattr(FoldAwareModelStep, "fit", _explode)

        class _PassthroughAdapter:
            """Never runs in practice — the exploded step.fit raises first —
            but its presence switches tune() onto the wrapped searcher path."""

            def fit_transform(self, X, y):
                return X, y

            def transform(self, X, y):
                return X, y

        X_arr, y_arr = make_classification(n_samples=90, n_features=4, random_state=1)
        logs: list[str] = []
        with pytest.raises(ValueError, match="All trials failed"):
            TuningCalculator(LogisticRegressionCalculator()).fit(
                pd.DataFrame(X_arr),
                pd.Series(y_arr, name="target"),
                config=TuningConfig(
                    strategy="optuna",
                    metric="accuracy",
                    n_trials=2,
                    search_space={"C": [0.5, 1.0]},
                    cv_folds=3,
                ),
                log_callback=logs.append,
                preprocessing=_PassthroughAdapter(),
            )

        assert any("failed" in message for message in logs), logs
