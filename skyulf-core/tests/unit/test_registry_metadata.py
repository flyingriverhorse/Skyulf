"""Core model metadata must be discoverable without backend setup."""

from copy import deepcopy

import pandas as pd
import polars as pl
import pytest

from skyulf.modeling.classification import (
    LogisticRegressionCalculator,
    RandomForestClassifierCalculator,
)
from skyulf.registry import NodeRegistry


def test_registry():
    """Missing dynamic models or parameters must fail instead of only printing a warning."""
    assert hasattr(LogisticRegressionCalculator, "__node_meta__")
    assert hasattr(RandomForestClassifierCalculator, "__node_meta__")
    metadata = NodeRegistry.get_all_metadata()
    assert "logistic_regression" in metadata
    assert "random_forest_classifier" in metadata
    assert "max_iter" in (metadata["logistic_regression"].get("params") or {})


@pytest.mark.parametrize("engine", ["pandas", "polars"])
@pytest.mark.parametrize(
    "node_id,parameter,value,config,data,expected,ignored",
    [
        (
            "AliasReplacement",
            "alias_type",
            "punctuation",
            {"columns": ["x"]},
            {"x": ["Yes!", "No?"]},
            {"x": ["Yes", "No"]},
            {"domain"},
        ),
        (
            "Casting",
            "column_types",
            {"x": "float"},
            {},
            {"x": ["1.5", "2.5"]},
            {"x": [1.5, 2.5]},
            {"type_map"},
        ),
        (
            "CorrelationThreshold",
            "correlation_method",
            "spearman",
            {},
            {"x": [1, 2, 3, 4], "y": [1, 2, 4, 100]},
            {"x": [1, 2, 3, 4]},
            {"method"},
        ),
        (
            "KBinsDiscretizer",
            "label_format",
            "range",
            {"columns": ["x"], "n_bins": 2, "strategy": "uniform"},
            {"x": [0, 1, 2, 3]},
            {
                "x": [0, 1, 2, 3],
                "x_binned": ["[0.0, 1.5]", "[0.0, 1.5]", "[1.5, 3.0]", "[1.5, 3.0]"],
            },
            {"encode"},
        ),
        (
            "SimpleTransformation",
            "transformations",
            [{"column": "x", "method": "square"}],
            {},
            {"x": [1, 2, 3, 4]},
            {"x": [1, 4, 9, 16]},
            {"func", "columns"},
        ),
        (
            "feature_selection",
            "threshold",
            1.0,
            {"method": "variance"},
            {"small": [0.0, 0.1, 0.2], "large": [0.0, 10.0, 20.0]},
            {"large": [0.0, 10.0, 20.0]},
            set(),
        ),
    ],
    ids=["alias", "casting", "correlation", "bins", "math", "selection-threshold"],
)
def test_advertised_preprocessing_settings_control_fit_apply(
    engine, node_id, parameter, value, config, data, expected, ignored
):
    """Advertised inputs must configure real output instead of silently doing nothing."""
    defaults = deepcopy(NodeRegistry.get_all_metadata()[node_id]["params"])
    assert parameter in defaults
    assert not ignored.intersection(defaults)
    frame = pd.DataFrame(data) if engine == "pandas" else pl.DataFrame(data)
    params = {**defaults, **config, parameter: value}
    fitted = NodeRegistry.get_calculator(node_id)().fit(frame, params)
    result = NodeRegistry.get_applier(node_id)().apply(frame, fitted)
    if engine == "polars":
        result = result.to_pandas()
    pd.testing.assert_frame_equal(
        result, pd.DataFrame(expected), check_dtype=False, check_categorical=False
    )
