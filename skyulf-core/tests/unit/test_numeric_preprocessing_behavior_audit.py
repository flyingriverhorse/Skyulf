"""Behavioral audit of numeric preprocessing state and inference boundaries."""

from copy import deepcopy
from typing import Any

import numpy as np
import pandas as pd
import polars as pl
import pytest

from skyulf.data.dataset import SplitDataset
from skyulf.preprocessing.base import StatefulTransformer
from skyulf.preprocessing.pipeline import FeatureEngineer
from skyulf.registry import NodeRegistry

_NUMERIC_MODES = [
    ("StandardScaler", {"columns": ["x"]}),
    ("StandardScaler", {"columns": ["x"], "with_mean": False}),
    ("StandardScaler", {"columns": ["x"], "with_std": False}),
    ("StandardScaler", {"columns": ["x"], "with_mean": False, "with_std": False}),
    ("MinMaxScaler", {"columns": ["x"], "feature_range": [-2, 2]}),
    ("MaxAbsScaler", {"columns": ["x"]}),
    ("RobustScaler", {"columns": ["x"], "quantile_range": [10, 90]}),
    ("RobustScaler", {"columns": ["x"], "with_centering": False}),
    ("RobustScaler", {"columns": ["x"], "with_scaling": False}),
    ("SimpleImputer", {"columns": ["x"], "strategy": "mean"}),
    ("SimpleImputer", {"columns": ["x"], "strategy": "median"}),
    ("SimpleImputer", {"columns": ["x"], "strategy": "mode"}),
    ("SimpleImputer", {"columns": ["x"], "strategy": "constant", "fill_value": -1}),
    ("KNNImputer", {"columns": ["x", "z"], "n_neighbors": 2}),
    ("IterativeImputer", {"columns": ["x", "z"], "random_state": 0}),
    ("IterativeImputer", {"columns": ["x", "z"], "estimator": "extra_trees"}),
    ("IQR", {"columns": ["x"], "multiplier": 1.5}),
    ("ZScore", {"columns": ["x"], "threshold": 2}),
    ("Winsorize", {"columns": ["x"], "lower_percentile": 10, "upper_percentile": 90}),
    ("EllipticEnvelope", {"columns": ["x"], "contamination": 0.1}),
    ("ManualBounds", {"bounds": {"x": {"lower": 0, "upper": 9}}}),
    ("CorrelationThreshold", {"columns": ["x", "z"], "correlation_method": "spearman"}),
    ("VarianceThreshold", {"columns": []}),
    ("UnivariateSelection", {"columns": ["x", "z"], "method": "SelectKBest", "k": 1}),
    ("UnivariateSelection", {"columns": ["x", "z"], "score_func": "chi2", "k": 1}),
    ("ModelBasedSelection", {"columns": ["x", "z"], "estimator": "logistic_regression"}),
    ("ModelBasedSelection", {"columns": ["x", "z"], "method": "rfe", "k": 1}),
    ("feature_selection", {"columns": [], "method": "variance"}),
    ("SimpleTransformation", {"transformations": [{"column": "x", "method": "sqrt"}]}),
    ("GeneralTransformation", {"transformations": [{"column": "x", "method": "square"}]}),
    ("GeneralTransformation", {"transformations": [{"column": "x", "method": "box-cox"}]}),
    ("PowerTransformer", {"columns": ["x"], "method": "yeo-johnson", "standardize": False}),
    ("PowerTransformer", {"columns": ["x"], "method": "box-cox"}),
    ("GeneralBinning", {"columns": ["x"], "strategy": "equal_width", "n_bins": 3}),
    ("GeneralBinning", {"columns": ["x"], "strategy": "equal_frequency", "n_bins": 3}),
    ("GeneralBinning", {"columns": ["x"], "strategy": "kmeans", "n_bins": 3}),
    ("GeneralBinning", {"columns": ["x"], "strategy": "custom", "custom_bins": {"x": [0, 4, 9]}}),
    ("CustomBinning", {"columns": ["x"], "bins": [0, 4, 9]}),
    ("KBinsDiscretizer", {"columns": ["x"], "strategy": "uniform", "n_bins": 3}),
    ("KBinsDiscretizer", {"columns": ["x"], "strategy": "quantile", "n_bins": 3}),
    ("KBinsDiscretizer", {"columns": ["x"], "strategy": "kmeans", "n_bins": 3}),
    ("AliasReplacement", {"columns": ["category"], "alias_type": "boolean"}),
    ("InvalidValueReplacement", {"columns": ["x"], "rule": "negative"}),
    ("ValueReplacement", {"columns": ["category"], "mapping": {"yes": "Y", "no": "N"}}),
    ("Casting", {"columns": ["category"], "target_type": "category"}),
    ("Casting", {"columns": ["x"], "target_type": "int", "coerce_on_error": True}),
    ("DropMissingColumns", {"missing_threshold": 50}),
    ("DropMissingColumns", {"columns": ["constant"], "missing_threshold": 0}),
    ("MissingIndicator", {"columns": []}),
    ("MissingIndicator", {"columns": ["x"]}),
]


def _frame(values: dict[str, list[Any]], engine: str) -> Any:
    """Build the same logical records for each supported dataframe engine."""
    frame = pd.DataFrame(values)
    return pl.from_pandas(frame) if engine == "polars" else frame


def _labels(values: list[int], engine: str) -> Any:
    """Keep target names and engines consistent through the split wrapper."""
    return pl.Series("target", values) if engine == "polars" else pd.Series(values, name="target")


def _assert_frame_equal(actual: Any, expected: Any, engine: str) -> None:
    """Compare values and schema without hiding engine changes or dropped rows."""
    if engine == "pandas":
        pd.testing.assert_frame_equal(actual, expected)
    else:
        assert actual.schema == expected.schema
        assert actual.equals(expected)


@pytest.mark.parametrize("node_type,config", _NUMERIC_MODES)
@pytest.mark.parametrize("engine", ["pandas", "polars"])
def test_numeric_modes_isolate_training_and_replay_state(node_type, config, engine):
    """Changing held-out records and replaying other batches must not refit training state."""
    train = _frame(
        {
            "x": [1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0],
            "z": [5.0, 1.0, 7.0, 2.0, 8.0, 3.0, 9.0, 4.0],
            "constant": [7.0] * 8,
            "missing": [None, None, None, None, 1.0, 2.0, 3.0, 4.0],
            "category": ["yes", "no", "YES", "NO", "true", "false", "yes", "no"],
        },
        engine,
    )
    original = deepcopy(train)
    labels = _labels([0, 0, 0, 0, 1, 1, 1, 1], engine)
    held_a = _frame(
        {
            "x": [3.0, 4.0],
            "z": [2.0, 3.0],
            "constant": [7.0, 7.0],
            "missing": [1.0, 2.0],
            "category": ["yes", "no"],
        },
        engine,
    )
    held_b = _frame(
        {
            "x": [10000.0, None],
            "z": [-9000.0, 700.0],
            "constant": [99.0, 5.0],
            "missing": [None, None],
            "category": ["unseen", "another"],
        },
        engine,
    )
    training_outputs = []
    probe_outputs = []
    for held in (held_a, held_b):
        transformer = StatefulTransformer(
            NodeRegistry.get_calculator(node_type)(),
            NodeRegistry.get_applier(node_type)(),
            node_type,
        )
        dataset = SplitDataset(
            train=(deepcopy(train), deepcopy(labels)),
            test=(deepcopy(held), _labels([0, 1], engine)),
            validation=(deepcopy(held), _labels([1, 0], engine)),
        )
        output = transformer.fit_transform(dataset, deepcopy(config))
        training_outputs.append(output.train[0])
        first_probe = transformer.transform(deepcopy(held_a))
        transformer.transform(deepcopy(held_b))
        repeated_probe = transformer.transform(deepcopy(held_a))
        _assert_frame_equal(repeated_probe, first_probe, engine)
        probe_outputs.append(first_probe)
    _assert_frame_equal(training_outputs[0], training_outputs[1], engine)
    _assert_frame_equal(probe_outputs[0], probe_outputs[1], engine)
    _assert_frame_equal(train, original, engine)


@pytest.mark.parametrize("engine", ["pandas", "polars"])
@pytest.mark.parametrize("strategy", ["mean", "median"])
@pytest.mark.parametrize(
    "training,mean,median",
    [([7.0, 7.0, None, 7.0], 7.0, 7.0), ([0.0, 1.0, None, 1.0], 2 / 3, 1.0), ([7.0], 7.0, 7.0)],
    ids=["constant", "binary", "singleton"],
)
def test_explicit_imputation_includes_low_cardinality_numeric_columns(
    engine, strategy, training, mean, median
):
    """An explicit feature selection must not be discarded by automatic binary/constant filters."""
    frame = _frame({"x": training}, engine)
    probe = _frame({"x": [None, 4.0]}, engine)
    params = NodeRegistry.get_calculator("SimpleImputer")().fit(
        frame, {"columns": ["x"], "strategy": strategy}
    )
    result = NodeRegistry.get_applier("SimpleImputer")().apply(probe, params)

    assert result["x"].to_list() == pytest.approx([mean if strategy == "mean" else median, 4.0])


@pytest.mark.parametrize("engine", ["pandas", "polars"])
@pytest.mark.parametrize(
    "node_type,config",
    [
        ("CustomBinning", {"columns": ["x"], "bins": [0, 10, 20, 30]}),
        (
            "GeneralBinning",
            {"columns": ["x"], "strategy": "custom", "custom_bins": {"x": [0, 10, 20, 30]}},
        ),
        ("KBinsDiscretizer", {"columns": ["x"], "strategy": "uniform", "n_bins": 3}),
    ],
)
def test_ordinal_bins_use_fitted_edges_for_singleton_and_reordered_batches(
    engine, node_type, config
):
    """Ordinal bin numbers must describe fitted intervals rather than category encounter order."""
    train = _frame({"x": [0.0, 10.0, 20.0, 30.0]}, engine)
    artifact = NodeRegistry.get_calculator(node_type)().fit(train, config)
    applier = NodeRegistry.get_applier(node_type)()
    alone = applier.apply(_frame({"x": [15.0]}, engine), artifact)
    batch = applier.apply(_frame({"x": [25.0, 15.0, 5.0]}, engine), artifact)

    assert alone["x_binned"].to_list() == [1]
    assert batch["x_binned"].to_list() == [2, 1, 0]


@pytest.mark.parametrize("engine", ["pandas", "polars"])
@pytest.mark.parametrize("node_type", ["PowerTransformer", "GeneralTransformation"])
@pytest.mark.parametrize(
    "method,companion", [("box-cox", -1.0), ("box-cox", np.inf), ("yeo-johnson", np.inf)]
)
def test_invalid_power_companion_cannot_disable_valid_row_transform(
    engine, node_type, method, companion
):
    """A logged batch failure must not silently feed untransformed valid values to a fitted model."""
    train = _frame({"x": [1.0, 2.0, 4.0, 10.0, 30.0]}, engine)
    config = (
        {"columns": ["x"], "method": method}
        if node_type == "PowerTransformer"
        else {"transformations": [{"column": "x", "method": method}]}
    )
    artifact = NodeRegistry.get_calculator(node_type)().fit(train, config)
    applier = NodeRegistry.get_applier(node_type)()
    alone = applier.apply(_frame({"x": [3.0]}, engine), artifact)
    batch = applier.apply(_frame({"x": [3.0, companion]}, engine), artifact)

    assert alone["x"].to_list()[0] != 3.0
    assert batch["x"].to_list()[0] == pytest.approx(alone["x"].to_list()[0])


@pytest.mark.parametrize("engine", ["pandas", "polars"])
@pytest.mark.parametrize(
    "node_type,config",
    [
        ("IQR", {"columns": ["x"]}),
        ("ZScore", {"columns": ["x"], "threshold": 2}),
        ("EllipticEnvelope", {"columns": ["x"], "contamination": 0.1}),
        ("ManualBounds", {"bounds": {"x": {"lower": 0, "upper": 9}}}),
    ],
)
def test_outlier_inference_keeps_one_output_per_requested_row(engine, node_type, config):
    """A serving transform must not silently remove inputs when prediction results have no row map."""
    train = _frame({"x": [0.0, 1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0]}, engine)
    engineer = FeatureEngineer([{"name": "outlier", "transformer": node_type, "params": config}])
    engineer.fit_transform(train)
    requested = _frame({"x": [3.0, 1000.0, None]}, engine)

    result = engineer.transform(requested)

    assert len(result) == len(requested)


@pytest.mark.parametrize("engine", ["pandas", "polars"])
def test_categorical_cast_preserves_known_numeric_value_across_batch_dtypes(engine):
    """A fractional companion must not turn a known integral categorical value into missing data."""
    train = _frame({"category": [1, 2]}, engine)
    artifact = NodeRegistry.get_calculator("Casting")().fit(
        train, {"columns": ["category"], "target_type": "category"}
    )
    applier = NodeRegistry.get_applier("Casting")()
    alone = applier.apply(_frame({"category": [1]}, engine), artifact)
    batch = applier.apply(_frame({"category": [1.0, 2.5]}, engine), artifact)

    assert pd.notna(alone["category"].to_list()[0])
    assert pd.notna(batch["category"].to_list()[0])
    assert str(alone["category"].to_list()[0]) == str(batch["category"].to_list()[0])
