"""Column-removing nodes must retain samples when no features remain."""

from typing import Any

import pandas as pd
import polars as pl
import pytest

from skyulf.engines import PolarsEngine, SkyulfPolarsWrapper
from skyulf.engines.sklearn_bridge import SklearnBridge
from skyulf.registry import NodeRegistry


def _input_pair(engine: str, values: list[Any], retained: bool) -> tuple[Any, Any]:
    """Build aligned samples on each supported local input boundary."""
    data = {"feature": values}
    if retained:
        data["row_id"] = list(range(len(values)))
    labels = [index % 2 for index in range(len(values))]
    if engine == "pandas":
        return pd.DataFrame(data), pd.Series(labels, name="target", dtype="int64")
    frame = pl.DataFrame(data)
    X = PolarsEngine.wrap(frame) if engine == "wrapped-polars" else frame
    return X, pl.Series("target", labels, dtype=pl.Int64)


@pytest.mark.parametrize("engine", ["pandas", "polars", "wrapped-polars"])
@pytest.mark.parametrize("retained", [False, True], ids=["no-features-left", "retained-feature"])
@pytest.mark.parametrize("rows", [4, 0], ids=["nonempty", "empty-holdout"])
@pytest.mark.parametrize(
    "node_id, values, config",
    [
        (
            "DummyEncoder",
            ["same"] * 4,
            {"columns": ["feature"], "drop_first": True},
        ),
        ("DummyEncoder", [None] * 4, {"columns": ["feature"]}),
        ("DropMissingColumns", [1.0] * 4, {"columns": ["feature"]}),
        ("DropMissingColumns", [None] * 4, {"missing_threshold": 50}),
        ("VarianceThreshold", [1.0] * 4, {"columns": ["feature"]}),
        (
            "UnivariateSelection",
            [1.0] * 4,
            {"columns": ["feature"], "k": 0, "score_func": "chi2"},
        ),
        (
            "ModelBasedSelection",
            [1.0] * 4,
            {"columns": ["feature"], "estimator": "logistic_regression", "threshold": 10},
        ),
    ],
    ids=[
        "dummy-constant",
        "dummy-all-null",
        "drop-explicit",
        "drop-all-null",
        "variance",
        "univariate",
        "model-based",
    ],
)
def test_column_removal_preserves_sample_count(engine, retained, rows, node_id, values, config):
    """A fitted column transform must never turn feature removal into sample removal."""
    train, train_y = _input_pair(engine, values, retained)
    artifact = NodeRegistry.get_calculator(node_id)().fit((train, train_y), config)
    X, y = _input_pair(engine, values[:rows], retained)

    output, output_y = NodeRegistry.get_applier(node_id)().apply((X, y), artifact)

    assert output.shape == (rows, int(retained))
    assert list(output.columns) == (["row_id"] if retained else [])
    assert output_y is y
    assert len(output_y) == rows
    assert isinstance(output, SkyulfPolarsWrapper) == (engine == "wrapped-polars")
    converted, converted_y = SklearnBridge.to_sklearn((output, output_y))
    assert converted.shape == (rows, int(retained))
    assert converted_y.tolist() == y.to_list()
    if retained:
        assert output["row_id"].to_list() == list(range(rows))
    assert X.shape == (rows, 1 + int(retained))


@pytest.mark.parametrize("engine", ["polars", "wrapped-polars"])
@pytest.mark.parametrize(
    "node_id", ["VarianceThreshold", "UnivariateSelection", "ModelBasedSelection"]
)
def test_disabled_selector_drop_preserves_original_features(engine, node_id):
    """An empty selected set is only destructive when column removal is enabled."""
    X, y = _input_pair(engine, [1.0, 1.0], False)
    artifact = {"candidate_columns": ["feature"], "selected_columns": [], "drop_columns": False}

    output, output_y = NodeRegistry.get_applier(node_id)().apply((X, y), artifact)

    assert output.shape == (2, 1)
    assert output["feature"].to_list() == [1.0, 1.0]
    assert output_y is y
