"""End-to-end regressions for fold-isolated preprocessing in the core pipeline."""

from typing import Any

import numpy as np
import pandas as pd
import polars as pl
import pytest

from skyulf.data.dataset import SplitDataset
from skyulf.modeling.base import StatefulEstimator, extract_xy
from skyulf.pipeline import SkyulfPipeline
from skyulf.preprocessing.pipeline import FeatureEngineer
from skyulf.preprocessing.scaling.standard import StandardScalerCalculator


def _tuning_config(steps: list[dict[str, Any]]) -> dict[str, Any]:
    """Build a small real ridge search whose folds are observable through preprocessing."""
    return {
        "preprocessing": steps,
        "modeling": {
            "type": "hyperparameter_tuner",
            "base_model": {"type": "ridge_regression"},
            "strategy": "grid",
            "metric": "r2",
            "search_space": {"alpha": [0.1, 1.0]},
            "cv_folds": 3,
            "random_state": 42,
        },
    }


def _step(name: str, transformer: str, **params: Any) -> dict[str, Any]:
    """Keep configurations explicit while omitting unrelated defaults."""
    return {"name": name, "transformer": transformer, "params": params}


def _regression_frame(engine: str = "pandas") -> Any:
    """Provide stable row identities and a nonconstant real regression target."""
    frame = pd.DataFrame(
        {
            "row_id": np.arange(24),
            "value": np.arange(1, 25, dtype=float),
            "target": np.arange(1, 25, dtype=float) ** 2 + 3.0,
        }
    )
    return frame if engine == "pandas" else pl.from_pandas(frame)


@pytest.mark.parametrize("engine", ["pandas", "polars"])
@pytest.mark.parametrize("input_kind", ["raw_split", "already_split", "no_split"])
def test_core_tuning_refits_scaler_on_each_inner_training_fold(
    monkeypatch: pytest.MonkeyPatch, engine: str, input_kind: str
) -> None:
    """Candidate preprocessing fits must exclude their validation rows and the outer test."""
    observed: list[list[int]] = []
    real_fit = StandardScalerCalculator.fit

    def record_fit(self: Any, data: Any, config: dict[str, Any]) -> Any:
        """Observe real scaler inputs without replacing its learned statistics."""
        frame = data[0] if isinstance(data, tuple) else data
        observed.append(frame["row_id"].to_list())
        return real_fit(self, data, config)

    monkeypatch.setattr(StandardScalerCalculator, "fit", record_fit)
    scale = _step("scale", "StandardScaler", columns=["value"])
    split = _step("split", "TrainTestSplitter", test_size=0.25, shuffle=False)
    data = _regression_frame(engine)
    expected_rows = set(range(24 if input_kind == "no_split" else 18))
    if input_kind == "raw_split":
        steps = [
            _step("target", "feature_target_split", target_column="target"),
            _step(
                "square",
                "SimpleTransformation",
                transformations=[{"column": "value", "method": "square"}],
            ),
            split,
            scale,
        ]
    elif input_kind == "already_split":
        data = SplitDataset(train=data[:18], test=data[18:])
        steps = [scale, split]
    else:
        steps = [scale]
    pipeline = SkyulfPipeline(_tuning_config(steps))

    metrics = pipeline.fit(data, target_column="target")

    assert len(observed) == 7
    assert all(len(rows) == len(expected_rows) * 2 // 3 for rows in observed[:-1])
    assert all(set(rows) < expected_rows for rows in observed[:-1])
    assert set(observed[-1]) == expected_rows
    assert "modeling_error" not in metrics


def test_core_tuning_predictions_use_final_fitted_preprocessor_once() -> None:
    """Serving must compose prefix math with the final fold adapter's scaler exactly once."""
    data = _regression_frame()
    pipeline = SkyulfPipeline(
        _tuning_config(
            [
                _step(
                    "square",
                    "SimpleTransformation",
                    transformations=[{"column": "value", "method": "square"}],
                ),
                _step("split", "TrainTestSplitter", test_size=0.25, shuffle=False),
                _step("scale", "StandardScaler", columns=["value"]),
            ]
        )
    )
    pipeline.fit(data, target_column="target")
    inference = pd.DataFrame({"row_id": [90, 91], "value": [3.0, 25.0]})
    squared_train = np.arange(1, 19, dtype=float) ** 2
    expected_features = np.column_stack(
        [
            [90, 91],
            (np.array([9.0, 625.0]) - squared_train.mean()) / squared_train.std(),
        ]
    )
    assert pipeline.model_estimator is not None
    assert pipeline.model_estimator.model is not None
    model, _result = pipeline.model_estimator.model

    predictions = pipeline.predict(inference)

    np.testing.assert_allclose(predictions, model.predict(expected_features))


@pytest.mark.parametrize("operation", ["target_encoding", "drop_missing_rows"])
def test_core_tuning_evaluates_the_final_training_representation(
    monkeypatch: pytest.MonkeyPatch, operation: str
) -> None:
    """Final training evaluation must retain OOF features and row drops from the actual refit."""
    fitted_payloads: list[Any] = []
    evaluated_payloads: list[Any] = []
    real_fold_fit = FeatureEngineer.fit_transform
    real_evaluate = StatefulEstimator.evaluate

    def record_fold(self: Any, data: Any, *args: Any, **kwargs: Any) -> Any:
        """Record each real fold representation, including the final full-training refit."""
        result = real_fold_fit(self, data, *args, **kwargs)
        if isinstance(data, tuple):
            fitted_payloads.append(result[0])
        return result

    def record_evaluation(self: Any, dataset: SplitDataset, target_column: str) -> Any:
        """Observe the evaluation payload while retaining real metric computation."""
        evaluated_payloads.append(extract_xy(dataset.train, target_column))
        return real_evaluate(self, dataset=dataset, target_column=target_column)

    monkeypatch.setattr(FeatureEngineer, "fit_transform", record_fold)
    monkeypatch.setattr(StatefulEstimator, "evaluate", record_evaluation)
    data = _regression_frame()
    if operation == "target_encoding":
        data["city"] = [f"city_{index % 6}" for index in range(len(data))]
        steps = [
            _step(
                "encode",
                "TargetEncoder",
                columns=["city"],
                cv=2,
                random_state=7,
                target_type="continuous",
            )
        ]
    else:
        data.loc[1, "value"] = np.nan
        steps = [
            _step("drop", "DropMissingRows", subset=["value"]),
            _step("impute", "SimpleImputer", columns=["value"], strategy="mean"),
        ]
    steps.append(_step("scale", "StandardScaler", columns=["value"]))
    pipeline = SkyulfPipeline(_tuning_config(steps))
    dataset = SplitDataset(train=data.iloc[:18], test=data.iloc[18:])

    metrics = pipeline.fit(dataset, target_column="target")

    assert fitted_payloads
    pd.testing.assert_frame_equal(evaluated_payloads[-1][0], fitted_payloads[-1][0])
    pd.testing.assert_series_equal(evaluated_payloads[-1][1], fitted_payloads[-1][1])
    assert "modeling_error" not in metrics


@pytest.mark.parametrize("engine", ["pandas", "polars"])
def test_core_holdout_tuning_never_fits_preprocessing_on_validation(
    monkeypatch: pytest.MonkeyPatch, engine: str
) -> None:
    """An explicit holdout must drive candidate scoring without entering any fitting payload."""
    seen: list[list[int]] = []
    real_fit = StandardScalerCalculator.fit

    def record_fit(self: Any, data: Any, config: dict[str, Any]) -> Any:
        """Observe each real scaler fit by immutable row identifiers."""
        frame = data[0] if isinstance(data, tuple) else data
        seen.append(frame["row_id"].to_list())
        return real_fit(self, data, config)

    monkeypatch.setattr(StandardScalerCalculator, "fit", record_fit)
    frame = _regression_frame(engine)
    dataset = SplitDataset(train=frame[:12], validation=frame[12:18], test=frame[18:])
    pipeline = SkyulfPipeline(_tuning_config([_step("scale", "StandardScaler", columns=["value"])]))

    metrics = pipeline.fit(dataset, target_column="target")

    assert len(seen) == 3
    assert all(set(rows) == set(range(12)) for rows in seen)
    assert "modeling_error" not in metrics


@pytest.mark.parametrize("engine", ["pandas", "polars"])
@pytest.mark.parametrize("explicit_time_column", [True, False])
def test_core_time_series_tuning_drops_sort_column_at_serving(
    engine: str, explicit_time_column: bool
) -> None:
    """The CV sort key must stay excluded during final evaluation and raw-data prediction."""
    frame = _regression_frame()
    frame["time"] = pd.date_range("2024-01-01", periods=len(frame))[::-1]
    if engine == "polars":
        frame = pl.from_pandas(frame)
    dataset = SplitDataset(train=frame[:18], test=frame[18:])
    config = _tuning_config([_step("scale", "StandardScaler", columns=["value"])])
    config["modeling"]["cv_type"] = "time_series_split"
    if explicit_time_column:
        config["modeling"]["cv_time_column"] = "time"
    pipeline = SkyulfPipeline(config)

    metrics = pipeline.fit(dataset, target_column="target")
    inference = pd.DataFrame(
        {
            "row_id": [90, 91],
            "value": [3.0, 25.0],
            "time": pd.to_datetime(["2025-01-01", "2025-01-02"]),
        }
    )
    predictions = pipeline.predict(inference)
    expected = np.column_stack([[90, 91], (np.array([3.0, 25.0]) - 9.5) / np.arange(1, 19).std()])
    assert pipeline.model_estimator is not None
    assert pipeline.model_estimator.model is not None
    model, _result = pipeline.model_estimator.model

    assert "modeling_error" not in metrics
    np.testing.assert_allclose(predictions, model.predict(expected))


def test_core_tuning_threshold_uses_final_transformed_validation() -> None:
    """Threshold tuning must receive the final encoded feature space after fold refitting."""
    frame = pd.DataFrame(
        {"value": np.arange(40, dtype=float), "city": ["low", "high"] * 20, "target": [0, 1] * 20}
    )
    config = _tuning_config(
        [
            _step("scale", "StandardScaler", columns=["value"]),
            _step("encode", "OneHotEncoder", columns=["city"]),
        ]
    )
    config["modeling"].update(
        base_model={"type": "logistic_regression"},
        metric="f1",
        search_space={"C": [1.0]},
        tune_threshold=True,
    )
    pipeline = SkyulfPipeline(config)
    dataset = SplitDataset(
        train=frame.iloc[:24], validation=frame.iloc[24:32], test=frame.iloc[32:]
    )

    metrics = pipeline.fit(dataset, target_column="target")
    assert pipeline.model_estimator is not None
    assert pipeline.model_estimator.model is not None
    _model, result = pipeline.model_estimator.model
    predictions = pipeline.predict(frame.iloc[32:].drop(columns="target"))

    assert result.decision_thresholds is not None
    assert result.decision_threshold_metric == "f1"
    assert predictions.to_list() == [0, 1] * 4
    assert "modeling_error" not in metrics


@pytest.mark.parametrize("strategy", ["random", "halving_grid", "halving_random", "optuna"])
def test_core_tuning_search_strategies_refit_the_pipeline_inside_folds(
    monkeypatch: pytest.MonkeyPatch, strategy: str
) -> None:
    """Searcher-owned folds must clone the core adapter without leaking outer held-out rows."""
    if strategy == "optuna":
        pytest.importorskip("optuna")
    seen: list[set[int]] = []
    real_fit = StandardScalerCalculator.fit

    def record_fit(self: Any, data: Any, config: dict[str, Any]) -> Any:
        """Retain real fitting while observing folds cloned by each search strategy."""
        frame = data[0] if isinstance(data, tuple) else data
        seen.append(set(frame["row_id"].to_list()))
        return real_fit(self, data, config)

    monkeypatch.setattr(StandardScalerCalculator, "fit", record_fit)
    frame = _regression_frame()
    dataset = SplitDataset(train=frame.iloc[:18], test=frame.iloc[18:])
    config = _tuning_config([_step("scale", "StandardScaler", columns=["value"])])
    config["modeling"].update(strategy=strategy, n_trials=2)
    if strategy.startswith("halving"):
        config["modeling"]["strategy_params"] = {
            "min_resources": 12,
            "max_resources": 18,
            "factor": 2,
        }
    pipeline = SkyulfPipeline(config)

    metrics = pipeline.fit(dataset, target_column="target")
    predictions = pipeline.predict(frame.iloc[18:].drop(columns="target"))

    assert any(rows < set(range(18)) for rows in seen)
    assert all(rows <= set(range(18)) for rows in seen)
    assert seen[-1] == set(range(18))
    assert np.isfinite(predictions).all()
    assert "modeling_error" not in metrics
