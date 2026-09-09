"""Behavioral re-audit of feature, temporal, geo, split, and row-changing nodes."""

from copy import deepcopy
from datetime import datetime

import numpy as np
import pandas as pd
import polars as pl
import pytest

from skyulf.data.dataset import SplitDataset
from skyulf.pipeline import SkyulfPipeline
from skyulf.preprocessing.base import StatefulTransformer
from skyulf.preprocessing.pipeline import FeatureEngineer
from skyulf.registry import NodeRegistry

ENGINES = ("pandas", "polars")
FEATURE_ALIASES = ("FeatureGeneration", "FeatureGenerationNode", "FeatureMath")
ROOT_AUDIT_IDS = frozenset(
    {
        *FEATURE_ALIASES,
        "PolynomialFeatures",
        "PolynomialFeaturesNode",
        "FeatureInteraction",
        "DateFeatures",
        "LagFeatures",
        "RollingAggregate",
        "GeoDistance",
        "H3Index",
        "Oversampling",
        "Undersampling",
        "DatasetProfile",
        "DataSnapshot",
        "TrainTestSplitter",
        "Split",
        "feature_target_split",
        "DropMissingRows",
        "Deduplicate",
    }
)


def _frame(values, engine):
    """Use native frames so neither engine's implementation is bypassed."""
    frame = pd.DataFrame(values)
    return frame if engine == "pandas" else pl.from_pandas(frame)


def _pandas(value):
    """Normalize assertions without changing the input engine under test."""
    return value.to_pandas() if hasattr(value, "to_pandas") else value


def _target(values, engine):
    """Keep target values and feature rows in the same native engine."""
    return pd.Series(values, name="target") if engine == "pandas" else pl.Series("target", values)


def _step(node_id, params):
    """Build the public linear preprocessing configuration."""
    return {"name": node_id, "transformer": node_id, "params": params}


def _pair(node_id):
    """Resolve real registered components, including legacy aliases."""
    return NodeRegistry.get_calculator(node_id)(), NodeRegistry.get_applier(node_id)()


ROW_LOCAL_CASES = (
    [
        (
            node_id,
            {
                "operations": [
                    {
                        "operation_type": "arithmetic",
                        "method": method,
                        "input_columns": ["x", "z"],
                        "output_column": "derived",
                    }
                ]
            },
            "derived",
        )
        for node_id in FEATURE_ALIASES
        for method in ("add", "subtract", "multiply", "divide")
    ]
    + [
        (
            node_id,
            {
                "operations": [
                    {
                        "operation_type": "ratio",
                        "input_columns": ["x"],
                        "secondary_columns": ["z"],
                        "output_column": "derived",
                    }
                ]
            },
            "derived",
        )
        for node_id in FEATURE_ALIASES
    ]
    + [
        (
            node_id,
            {
                "operations": [
                    {
                        "operation_type": "similarity",
                        "input_columns": ["text"],
                        "secondary_columns": ["other"],
                        "output_column": "derived",
                    }
                ]
            },
            "derived",
        )
        for node_id in FEATURE_ALIASES
    ]
    + [
        (node_id, {"columns": ["x", "z"], "degree": 2}, "poly_x_pow_2")
        for node_id in ("PolynomialFeatures", "PolynomialFeaturesNode")
    ]
    + [
        ("FeatureInteraction", {"columns": ["x", "z"]}, "x_x_z"),
        (
            "DateFeatures",
            {"columns": ["date"], "features": ["year", "weekofyear", "is_weekend"]},
            "date_year",
        ),
    ]
    + [
        (
            "GeoDistance",
            {
                "lat1_col": "x",
                "lon1_col": "z",
                "lat2_col": "lat",
                "lon2_col": "lon",
                "method": method,
                "unit": unit,
            },
            "geo_distance_km",
        )
        for method in ("haversine", "euclidean")
        for unit in ("km", "mi")
    ]
)


@pytest.mark.parametrize("engine", ENGINES)
@pytest.mark.parametrize("node_id,params,generated_column", ROW_LOCAL_CASES)
def test_fixed_features_replay_independently_of_companion_rows(
    engine, node_id, params, generated_column
):
    """A fixed transformation must not relearn from other inference rows."""
    data = _frame(
        {
            "x": [2.0, 6.0, 20.0],
            "z": [4.0, 3.0, 10.0],
            "lat": [3.0, 7.0, 21.0],
            "lon": [5.0, 4.0, 11.0],
            "text": ["red green", "blue", "orange"],
            "other": ["green red", "blue", "pink"],
            "date": [datetime(2024, 1, 1), datetime(2024, 6, 2), datetime(2025, 1, 1)],
        },
        engine,
    )
    calculator, applier = _pair(node_id)
    artifact = calculator.fit(data, deepcopy(params))
    before = deepcopy(artifact)
    single = _pandas(applier.apply(data.head(1), artifact)).reset_index(drop=True)
    batch = _pandas(applier.apply(data, artifact)).head(1).reset_index(drop=True)
    assert generated_column in single.columns
    pd.testing.assert_frame_equal(single, batch, check_dtype=False)
    assert artifact == before


@pytest.mark.parametrize("engine", ENGINES)
@pytest.mark.parametrize("node_id", FEATURE_ALIASES)
@pytest.mark.parametrize("method", ("mean", "sum", "count", "min", "max", "std", "median"))
def test_group_aggregations_replay_training_statistics_only(engine, node_id, method):
    """Seen, unseen, and null keys must not aggregate held-out values."""
    params = {
        "operations": [
            {
                "operation_type": "group_agg",
                "method": method,
                "input_columns": ["group"],
                "secondary_columns": ["x"],
                "output_column": "stat",
            }
        ]
    }
    train = _frame({"group": ["a", "a", "b", None], "x": [2.0, 4.0, 8.0, 10.0]}, engine)
    test = _frame({"group": ["a", "new", None], "x": [9000.0, -9000.0, 99999.0]}, engine)
    calculator, applier = _pair(node_id)
    transformer = StatefulTransformer(calculator, applier, node_id)
    result = transformer.fit_transform(
        SplitDataset(train=train, test=test, validation=test), params
    )
    assert isinstance(result, SplitDataset)
    expected = pd.Series([2.0, 4.0]).agg(method)
    values = _pandas(result.test)["stat"]
    assert values.iloc[0] == pytest.approx(expected)
    assert pd.isna(values.iloc[1])
    null_expected = pd.Series([10.0]).agg(method)
    assert pd.isna(values.iloc[2]) if pd.isna(null_expected) else values.iloc[2] == null_expected
    pd.testing.assert_frame_equal(_pandas(result.test), _pandas(result.validation))
    replay = _pandas(transformer.transform(test))
    pd.testing.assert_frame_equal(replay, _pandas(result.test))


@pytest.mark.parametrize("engine", ENGINES)
@pytest.mark.parametrize("node_id", ("LagFeatures", "RollingAggregate"))
def test_temporal_nodes_keep_target_alignment_and_split_history_separate(engine, node_id):
    """Sorting must permute y identically without borrowing another split's history."""
    config = {
        "columns": ["value"],
        "sort_by": "time",
        "group_by": ["group"],
        "lags": [1],
        "window": 2,
        "aggregations": ["mean"],
        "min_periods": 1,
    }
    train = _frame({"time": [3, 1, 2], "value": [30.0, 10.0, 20.0], "group": ["a"] * 3}, engine)
    test = _frame({"time": [5, 4], "value": [500.0, 400.0], "group": ["a"] * 2}, engine)
    calculator, applier = _pair(node_id)
    transformer = StatefulTransformer(calculator, applier, node_id)
    result = transformer.fit_transform(
        SplitDataset(
            train=(train, _target([30, 10, 20], engine)), test=(test, _target([500, 400], engine))
        ),
        config,
    )
    assert isinstance(result, SplitDataset)
    assert isinstance(result.train, tuple)
    assert isinstance(result.test, tuple)
    train_x, train_y = map(_pandas, result.train)
    test_x, test_y = map(_pandas, result.test)
    assert train_x["value"].tolist() == train_y.tolist() == [10, 20, 30]
    assert test_x["value"].tolist() == test_y.tolist() == [400, 500]
    if node_id == "LagFeatures":
        assert pd.isna(test_x["value_lag_1"].iloc[0])
        assert test_x["value_lag_1"].iloc[1] == 400
    else:
        assert test_x["value_roll_mean_2"].tolist() == [400, 450]


@pytest.mark.parametrize("engine", ENGINES)
@pytest.mark.parametrize("node_id", ("DatasetProfile", "DataSnapshot"))
def test_inspection_artifacts_do_not_change_heldout_features(engine, node_id):
    """Read-only diagnostics may summarize train but cannot transform model inputs."""
    train = _frame({"x": [1.0, 2.0, 3.0]}, engine)
    test = _frame({"x": [9999.0, None]}, engine)
    calculator, applier = _pair(node_id)
    transformer = StatefulTransformer(calculator, applier, node_id)
    result = transformer.fit_transform(SplitDataset(train, test), {"n_rows": 1})
    assert isinstance(result, SplitDataset)
    assert result.test is test
    if node_id == "DatasetProfile":
        assert transformer.params["profile"]["rows"] == 3
    else:
        assert transformer.params["snapshot"] == [{"x": 1.0}]


@pytest.mark.parametrize("engine", ENGINES)
@pytest.mark.parametrize(
    "node_id,method", (("Oversampling", "random_over"), ("Undersampling", "random_under_sampling"))
)
def test_linear_engineer_resamples_train_only_and_skips_inference(engine, node_id, method):
    """Class balancing must not fabricate or delete held-out or requested prediction rows."""
    train = _frame({"row_id": list(range(12)), "x": list(range(12))}, engine)
    y = _target([0] * 9 + [1] * 3, engine)
    test = _frame({"row_id": [100, 101, 102], "x": [1, 2, 3]}, engine)
    test_y = _target([0, 0, 1], engine)
    dataset = SplitDataset((train, y), (test, test_y), (test, test_y))
    engineer = FeatureEngineer([_step(node_id, {"method": method, "random_state": 7})])
    result, _ = engineer.fit_transform(dataset)
    assert len(result.train[0]) == (18 if node_id == "Oversampling" else 6)
    assert result.test is dataset.test and result.validation is dataset.validation
    assert engineer.transform(test) is test


@pytest.mark.parametrize("engine", ENGINES)
@pytest.mark.parametrize("split_id", ("TrainTestSplitter", "Split"))
@pytest.mark.parametrize("target_first", (False, True))
@pytest.mark.parametrize("stratify", (False, True))
def test_both_split_orders_preserve_disjoint_row_identity(engine, split_id, target_first, stratify):
    """X/y separation is not a row boundary, and every real partition keeps exact row identities."""
    data = _frame({"row_id": list(range(40)), "target": [i % 2 for i in range(40)]}, engine)
    split = _step(
        split_id,
        {"test_size": 0.2, "validation_size": 0.2, "random_state": 17, "stratify": stratify},
    )
    target = _step("feature_target_split", {"target_column": "target"})
    steps = [target, split] if target_first else [split, target]
    result, _ = FeatureEngineer(steps).fit_transform(data, target_column="target")
    members = []
    for payload in (result.train, result.test, result.validation):
        frame, y = map(_pandas, payload)
        assert "target" not in frame.columns
        ids = frame["row_id"].tolist()
        assert y.tolist() == [i % 2 for i in ids]
        members.append(set(ids))
    assert set.union(*members) == set(range(40))
    assert all(not left & right for i, left in enumerate(members) for right in members[i + 1 :])


@pytest.mark.parametrize("engine", ENGINES)
@pytest.mark.parametrize("node_id", ("DropMissingRows", "Deduplicate"))
@pytest.mark.parametrize("target_kind", ("native", "numpy", "list"))
def test_row_cleaning_accepts_engine_neutral_targets(engine, node_id, target_kind):
    """Supported neutral y containers must be filtered by the same surviving row positions."""
    data = _frame({"x": [1.0, 1.0, None, 3.0]}, engine)
    values = [10, 11, 12, 13]
    y = (
        _target(values, engine)
        if target_kind == "native"
        else (np.asarray(values) if target_kind == "numpy" else values)
    )
    calculator, applier = _pair(node_id)
    params = calculator.fit((data, y), {"subset": ["x"]})
    frame, target = applier.apply((data, y), params)
    expected = [10, 11, 13] if node_id == "DropMissingRows" else [10, 12, 13]
    assert len(frame) == len(expected)
    assert np.asarray(_pandas(target)).tolist() == expected


@pytest.mark.parametrize("engine", ENGINES)
def test_h3_real_dependency_replay_is_row_local(engine):
    """When the optional geo extra is installed, real H3 replay must preserve row locality."""
    pytest.importorskip("h3")
    data = _frame({"lat": [54.6872, None, 41.0082], "lon": [25.2797, None, 28.9784]}, engine)
    calculator, applier = _pair("H3Index")
    params = calculator.fit(data, {"lat_col": "lat", "lon_col": "lon"})
    batch = _pandas(applier.apply(data, params))
    single = _pandas(applier.apply(data.head(1), params))
    assert batch["h3_index"].iloc[0] == single["h3_index"].iloc[0]
    assert pd.isna(batch["h3_index"].iloc[1])


@pytest.mark.parametrize("engine", ENGINES)
def test_public_unsplit_training_accepts_feature_target_split(engine):
    """Explicit X/y separation must not break the documented no-holdout core fitting path."""
    data = _frame({"x": list(range(12)), "target": [2 * i + 1 for i in range(12)]}, engine)
    pipeline = SkyulfPipeline(
        {
            "preprocessing": [_step("feature_target_split", {"target_column": "target"})],
            "modeling": {"type": "linear_regression", "hyperparameters": {}},
        }
    )
    pipeline.fit(data, target_column="target")
    assert pipeline.is_fitted()


@pytest.mark.parametrize("engine", ENGINES)
@pytest.mark.parametrize("node_id", FEATURE_ALIASES)
def test_datetime_week_null_does_not_remove_features_for_valid_rows(engine, node_id):
    """A missing date cannot suppress calendar output for every other row in a batch."""
    data = _frame({"date": [datetime(2024, 1, 1), None]}, engine)
    config = {
        "operations": [
            {
                "operation_type": "datetime_extract",
                "input_columns": ["date"],
                "datetime_features": ["week", "month"],
            }
        ]
    }
    calculator, applier = _pair(node_id)
    params = calculator.fit(data, config)
    result = _pandas(applier.apply(data, params))
    assert "date_week" in result.columns and "date_month" in result.columns
    assert result["date_week"].iloc[0] == 1
    assert pd.isna(result["date_week"].iloc[1])


def test_audit_scope_contains_twenty_real_preprocessing_ids():
    """Aliases and optional nodes must stay visible in the audit's coverage denominator."""
    assert len(ROOT_AUDIT_IDS) == 20
    assert set(NodeRegistry.list_transformers()) >= ROOT_AUDIT_IDS
