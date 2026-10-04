"""Weights follow explicit row provenance through cleaning and fold fitting."""

from typing import Any

import numpy as np
import pandas as pd
import polars as pl
import pytest
from sklearn.linear_model import Ridge

from skyulf.modeling._sample_weights import SampleWeightError
from skyulf.modeling._tuning.engine import TuningCalculator, TuningConfig
from skyulf.modeling.regression import RidgeRegressionCalculator
from skyulf.preprocessing._weight_policy import apply_weighted
from skyulf.preprocessing.fold_adapter import FeatureEngineerFoldAdapter
from skyulf.preprocessing.pipeline import FeatureEngineer


@pytest.mark.parametrize("engine", ["pandas", "polars"])
@pytest.mark.parametrize("kind", ["series", "list"])
def test_transform_normalizes_weights_before_selecting_rows(engine, kind):
    """Standalone transform must slice weights by position regardless of their container."""
    from skyulf.data.dataset import SplitDataset
    from skyulf.preprocessing.base import StatefulTransformer
    from skyulf.registry import NodeRegistry

    X = pd.DataFrame({"row": [1, 2, 3], "value": [1.0, np.nan, 3.0]})
    y = pd.Series([100, 200, 300])
    if engine == "polars":
        X, y = pl.from_pandas(X), pl.Series("target", y)
    weights = pd.Series([10.0, 20.0, 30.0], index=[2, 1, 0]) if kind == "series" else [10, 20, 30]
    transformer = StatefulTransformer(
        NodeRegistry.get_calculator("DropMissingRows")(),
        NodeRegistry.get_applier("DropMissingRows")(),
        "drop",
    )
    transformer.fit_transform((X, y), {"subset": ["value"]})
    result = transformer.transform(
        SplitDataset(train=(X, y), test=(X, y), train_sample_weight=weights)
    )
    assert isinstance(result, SplitDataset)
    np.testing.assert_array_equal(np.asarray(result.train[0]["row"]), [1, 3])
    np.testing.assert_array_equal(np.asarray(result.train[1]), [100, 300])
    np.testing.assert_array_equal(result.train_sample_weight, [10, 30])


@pytest.mark.parametrize("engine", ["pandas", "polars"])
@pytest.mark.parametrize(
    "name,params",
    [
        ("DropMissingRows", {"subset": ["value"]}),
        ("Deduplicate", {"subset": ["value"]}),
        ("ManualBounds", {"bounds": {"row": {"lower": 1, "upper": 4}}}),
        ("LagFeatures", {"columns": ["row"], "lags": [1], "sort_by": "row", "drop_na": True}),
        ("RollingAggregate", {"columns": ["row"], "window": 2, "sort_by": "row"}),
    ],
)
def test_weighted_selection_uses_positions(engine, name, params):
    """Repeated labels cannot expand, sort or drop another observation's weight."""
    X = pd.DataFrame(
        {"row": [4.0, 1.0, 3.0, 2.0, 0.0], "value": [1.0, 2.0, 2.0, np.nan, 3.0]}, index=[7] * 5
    )
    y = pd.Series(X.row.to_numpy() * 10, index=X.index)
    weights = X.row.to_numpy() + 1
    if engine == "polars":
        X, y = pl.from_pandas(X), pl.Series("target", y.to_numpy())
    engineer = FeatureEngineer([{"name": "select", "transformer": name, "params": params}])
    (out, target), _ = engineer.fit_transform((X, y), sample_weight=weights)
    np.testing.assert_array_equal(engineer.train_sample_weight_, np.asarray(out["row"]) + 1)
    np.testing.assert_array_equal(np.asarray(target), np.asarray(out["row"]) * 10)


@pytest.mark.parametrize("strategy", ["grid", "random", "halving_grid", "halving_random", "optuna"])
@pytest.mark.parametrize("policy", ["k_fold", "group_k_fold", "time_series_split", "nested_cv"])
def test_row_drop_weights_reach_every_candidate_and_refit(monkeypatch, strategy, policy):
    """Training-only duplicate removal preserves each fit's actual weight vector."""
    if strategy == "optuna":
        pytest.importorskip("optuna")
    ids = np.tile(np.arange(16, dtype=float), 3)
    X = pd.DataFrame({"row": ids}, index=[4] * len(ids))
    if policy == "group_k_fold":
        X["group"] = ids % 4
    if policy == "time_series_split":
        X["time"] = pd.Timestamp("2025-01-01") + pd.to_timedelta(np.arange(len(ids)), unit="D")
    y = pd.Series(ids / 3, index=X.index)
    calls = []
    original = Ridge.fit

    def record(self, X, y, sample_weight=None):
        """Inspect actual model inputs, so a successful search cannot hide misalignment."""
        assert sample_weight is not None
        np.testing.assert_array_equal(sample_weight, np.asarray(X)[:, 0] + 1)
        assert len(np.unique(np.asarray(X)[:, 0])) == len(X)
        calls.append(len(X))
        return original(self, X, y, sample_weight=sample_weight)

    monkeypatch.setattr(Ridge, "fit", record)
    preprocessing = FeatureEngineerFoldAdapter(
        [{"name": "dedup", "transformer": "Deduplicate", "params": {"subset": ["row"]}}], "target"
    )
    config = TuningConfig(
        strategy=strategy,
        search_space={"alpha": [1.0]},
        cv_type=policy,
        cv_group_column="group" if policy == "group_k_fold" else None,
        cv_time_column="time" if policy == "time_series_split" else None,
        cv_inner_folds=2,
        cv_folds=2,
        n_trials=1,
        n_jobs=1,
        metric="mse",
    )
    TuningCalculator(RidgeRegressionCalculator()).fit(
        X,
        y,
        config,
        sample_weight=ids + 1,
        preprocessing=preprocessing,
    )
    assert calls and calls[-1] == 16


@pytest.mark.parametrize("mapping", [[-1], [3], [0.5], [True], [[0]], [0, 1]])
def test_custom_mapping_rejects_invalid_provenance(mapping):
    """A declared custom contract cannot smuggle malformed row positions into weights."""

    class Custom:
        """Return a deliberately invalid row mapping."""

        def apply_with_row_mapping(self, data, params):
            """Pair one output observation with the supplied invalid provenance."""
            return data.iloc[:1], mapping

    with pytest.raises(SampleWeightError, match="Row mapping"):
        apply_weighted(Custom(), pd.DataFrame({"x": [1, 2, 3]}), {}, np.ones(3))


@pytest.mark.parametrize("strategy", ["grid", "random", "halving_grid", "halving_random", "optuna"])
def test_sorted_validation_scores_paired_targets(strategy):
    """A temporal permutation must move scoring targets with validation features."""
    if strategy == "optuna":
        pytest.importorskip("optuna")
    ids = np.random.default_rng(7).permutation(80).astype(float)
    X = pd.DataFrame({"row": ids}, index=[5] * len(ids))
    y = pd.Series(ids * 3, index=X.index)
    preprocessing = FeatureEngineerFoldAdapter(
        [
            {
                "name": "rolling",
                "transformer": "RollingAggregate",
                "params": {"columns": ["row"], "window": 1, "sort_by": "row"},
            }
        ],
        "target",
    )
    config = TuningConfig(
        strategy=strategy,
        search_space={"alpha": [0.0]},
        cv_folds=2,
        n_trials=1,
        n_jobs=1,
        metric="mse",
    )
    _, result = TuningCalculator(RidgeRegressionCalculator()).fit(
        X,
        y,
        config,
        sample_weight=ids + 1,
        preprocessing=preprocessing,
    )
    assert abs(result.best_score) < 1e-10


@pytest.mark.parametrize("engine", ["pandas", "polars"])
@pytest.mark.parametrize(
    "name,params",
    [
        ("IQR", {"multiplier": 1.5}),
        ("ZScore", {"threshold": 1.5}),
        ("Winsorize", {"lower_percentile": 10.0, "upper_percentile": 90.0}),
        ("EllipticEnvelope", {"contamination": 0.2, "random_state": 7}),
        ("ManualBounds", {"bounds": {"value": {"lower": -2.0, "upper": 2.0}}}),
    ],
)
def test_each_outlier_node_transports_actual_weights(engine, name, params):
    """Each node's learned mask or clipping preserves observation ownership."""
    values = np.random.default_rng(7).normal(size=30)
    values[-1] = 100.0
    X = pd.DataFrame({"row": np.arange(30), "value": values}, index=[2] * 30)
    y = pd.Series(np.arange(30) * 10, index=X.index)
    if engine == "polars":
        X, y = pl.from_pandas(X), pl.Series("target", y.to_numpy())
    step = {"name": "outliers", "transformer": name, "params": {"columns": ["value"], **params}}
    engineer = FeatureEngineer([step])
    (out, target), _ = engineer.fit_transform((X, y), sample_weight=np.arange(30) + 1)
    rows = np.asarray(out["row"])
    np.testing.assert_array_equal(engineer.train_sample_weight_, rows + 1)
    np.testing.assert_array_equal(np.asarray(target), rows * 10)
    assert len(out) == 30 if name == "Winsorize" else len(out) < 30


def project_filter(frame):
    """Keep a concrete subset while letting the adapter own target alignment."""
    return frame["row"] % 2 == 0


def project_column(frame):
    """Compute one additional feature without accessing weight values."""
    assert list(frame.columns) == ["row"]
    return frame["row"] * 2


def project_learn(frame, y):
    """Learn only from actual feature and target columns."""
    assert list(frame.columns) == ["row"]
    return {"offset": float(np.mean(y))}


def project_apply(frame, state):
    """Reuse learned state on each existing observation."""
    return frame["row"] + state["offset"]


@pytest.mark.parametrize("engine", ["pandas", "polars"])
@pytest.mark.parametrize("kind", ["filter", "column", "fitted"])
def test_project_function_weight_contract(engine, kind):
    """Project code receives no weights as features and keeps explicit row ownership."""
    from skyulf.preprocessing.function_steps import column_step, filter_step, fitted_step

    steps = {
        "filter": filter_step("filter", project_filter, columns=["row"]),
        "column": column_step("column", project_column, output="extra"),
        "fitted": fitted_step("fitted", project_learn, project_apply, output="extra"),
    }
    X = pd.DataFrame({"row": np.arange(8)}, index=[9] * 8)
    y = pd.Series(np.arange(8) * 10, index=X.index)
    if engine == "polars":
        X, y = pl.from_pandas(X), pl.Series("target", y.to_numpy())
    engineer = FeatureEngineer([steps[kind]])
    (out, target), _ = engineer.fit_transform((X, y), sample_weight=np.arange(8) + 1)
    rows = np.asarray(out["row"])
    np.testing.assert_array_equal(engineer.train_sample_weight_, rows + 1)
    np.testing.assert_array_equal(np.asarray(target), rows * 10)
    assert len(out) == (4 if kind == "filter" else 8)


def test_custom_mapping_rebuilds_target_from_original_positions():
    """Custom output labels cannot desynchronize weights from the original labels."""

    class Custom:
        """Select and repeat rows using an explicit audited positional contract."""

        def apply_with_row_mapping(self, data, params):
            """Deliberately return bad labels to prove the orchestrator owns alignment."""
            return (data[0].iloc[[2, 0, 2]], pd.Series([-1] * 3)), [2, 0, 2]

    (out, target), weights = apply_weighted(
        Custom(),
        (pd.DataFrame({"row": [0, 1, 2]}), pd.Series([10, 20, 30])),
        {},
        np.array([1.0, 2.0, 3.0]),
    )
    assert list(out.row) == [2, 0, 2]
    assert list(target) == [30, 10, 30]
    np.testing.assert_array_equal(weights, [3, 1, 3])


def test_custom_training_representation_cannot_bypass_mapping():
    """A custom training hook cannot reorder observations behind an admitted applier."""
    from skyulf.preprocessing.base import StatefulTransformer

    class Calculator:
        """Expose both training entrypoints to reproduce the bypass risk."""

        def fit(self, df, config):
            """Return a stateless artifact."""
            return {}

        def fit_transform_train(self, data, config):
            """Fail if an unaudited hook is executed at all."""
            raise AssertionError("must reject before learning")

    class Applier:
        """Expose an otherwise valid custom positional mapping."""

        def apply(self, df, params):
            """Preserve the supplied data."""
            return df

        def apply_with_row_mapping(self, data, params):
            """Return explicit identity positions."""
            return data, np.arange(len(data))

    transformer = StatefulTransformer(Calculator(), Applier(), "custom")
    with pytest.raises(SampleWeightError, match="fit_transform_train"):
        transformer.fit_transform_weighted(pd.DataFrame({"x": [1, 2]}), {}, np.ones(2))


def test_weighted_sampler_transform_keeps_existing_splits():
    """Standalone transform must not sample an already fitted training split again."""
    from skyulf.data.dataset import SplitDataset
    from skyulf.preprocessing.base import StatefulTransformer
    from skyulf.preprocessing.resampling import OversamplingApplier, OversamplingCalculator

    X = pd.DataFrame({"x": [0.0, 1.0, 2.0, 3.0, 4.0, 5.0]})
    y = pd.Series([0, 0, 0, 0, 1, 1])
    dataset = SplitDataset(train=(X, y), test=(X, y), train_sample_weight=np.arange(6) + 1.0)
    transformer = StatefulTransformer(
        OversamplingCalculator(),
        OversamplingApplier(),
        "sample",
        apply_on_test=False,
        apply_on_validation=False,
    )
    fitted = transformer.fit_transform(dataset, {"method": "random_over", "random_state": 7})
    transformed = transformer.transform(fitted)
    assert transformed is fitted
    assert isinstance(transformed, SplitDataset)
    assert len(transformed.train[0]) == 8
    assert len(transformed.test[0]) == 6


@pytest.mark.parametrize("policy", ["k_fold", "group_k_fold", "time_series_split", "nested_cv"])
def test_ordinary_cv_transports_row_drop_weights(monkeypatch, policy):
    """Fixed-model inner and outer folds carry the retained training rows' weights."""
    from skyulf.modeling.cross_validation import perform_cross_validation
    from skyulf.modeling.regression import RidgeRegressionApplier

    ids = np.tile(np.arange(16, dtype=float), 3)
    X = pd.DataFrame({"row": ids}, index=[4] * len(ids))
    options: dict[str, Any] = {}
    if policy == "group_k_fold":
        X["group"] = ids % 4
        options["group_column"] = "group"
    if policy == "time_series_split":
        X["time"] = pd.Timestamp("2025-01-01") + pd.to_timedelta(np.arange(len(ids)), unit="D")
        options["time_column"] = "time"
    calls = []
    original = Ridge.fit

    def record(self, X, y, sample_weight=None):
        """Inspect every fitted estimator after its fold's preprocessing."""
        assert sample_weight is not None
        rows = np.asarray(X)[:, 0]
        np.testing.assert_array_equal(sample_weight, rows + 1)
        assert len(np.unique(rows)) == len(rows)
        calls.append(len(rows))
        return original(self, X, y, sample_weight=sample_weight)

    monkeypatch.setattr(Ridge, "fit", record)
    preprocessing = FeatureEngineerFoldAdapter(
        [{"name": "dedup", "transformer": "Deduplicate", "params": {"subset": ["row"]}}], "target"
    )
    result = perform_cross_validation(
        RidgeRegressionCalculator(),
        RidgeRegressionApplier(),
        X,
        pd.Series(ids / 3),
        {},
        n_folds=2,
        inner_folds=2,
        cv_type=policy,
        sample_weight=ids + 1,
        preprocessing=preprocessing,
        **options,
    )
    assert len(result["folds"]) == 2 and len(calls) >= 2


def test_merged_weighted_adapter_rejects_custom_row_mapping(monkeypatch):
    """Branch merging cannot equate row counts with identical row provenance."""
    from skyulf.preprocessing.fold_adapter import MergedBranchFoldAdapter
    from skyulf.registry import NodeRegistry

    class Calculator:
        """Custom no-op learner."""

        def fit(self, df, config):
            """Produce no learned state."""
            return {}

    class Applier:
        """Custom row permutation that requires explicit provenance."""

        def apply(self, df, params):
            """Reverse both payload components."""
            return df[0].iloc[::-1], df[1].iloc[::-1]

        def apply_with_row_mapping(self, df, params):
            """Return the source order alongside the permuted payload."""
            return self.apply(df, params), np.arange(len(df[0]))[::-1]

    monkeypatch.setitem(NodeRegistry._calculators, "CustomMapping", Calculator)
    monkeypatch.setitem(NodeRegistry._appliers, "CustomMapping", Applier)
    adapter = MergedBranchFoldAdapter(
        [[{"name": "custom", "transformer": "CustomMapping"}]], "first_wins", "target"
    )
    with pytest.raises(SampleWeightError, match="unsupported"):
        adapter.fit_transform(
            pd.DataFrame({"row": [0, 1]}), pd.Series([0, 1]), sample_weight=np.ones(2)
        )
