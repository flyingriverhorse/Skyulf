"""GroupImputer and ClipValues: the two nodes that replace hand-written project code.

Both came from a real project (corporate lead scoring) whose cleaning step
filled gaps with per-industry means and clipped inputs to fixed business
ranges. Written by hand, that code computed group means on whatever batch was
being scored; these nodes learn once on training rows and reuse the result.
"""

import json

import numpy as np
import pandas as pd
import polars as pl
import pytest

from skyulf.preprocessing import FeatureEngineer
from skyulf.registry import NodeRegistry

ENGINES = ["pandas", "polars"]


def _native(frame: pd.DataFrame, engine: str):
    """Return the frame in the requested engine."""
    return pl.from_pandas(frame) if engine == "polars" else frame


def _pandas(frame):
    """Compare results independently of engine."""
    return frame.to_pandas() if isinstance(frame, pl.DataFrame) else frame


def _fit_apply(node: str, config: dict, train, score):
    """Fit on training rows, apply to scoring rows, return (artifact, result)."""
    artifact = NodeRegistry.get_calculator(node)().fit(train, config)
    return artifact, NodeRegistry.get_applier(node)().apply(score, artifact)


_TRAIN = pd.DataFrame(
    {
        "industry": ["A", "A", "A", "B", "B", "C", None],
        "employees": [10.0, 20.0, np.nan, 100.0, 300.0, np.nan, 7.0],
        "region": ["north", "north", "south", "east", "east", None, "west"],
    }
)
# Scoring batch has its own (very different) values: the node must ignore them.
_SCORE = pd.DataFrame(
    {
        "industry": ["A", "B", "C", "NEW", None, "A"],
        "employees": [np.nan, np.nan, np.nan, np.nan, np.nan, 9999.0],
        "region": [None, None, None, None, None, "north"],
    },
    index=[5, 3, 8, 1, 0, 9],
)


@pytest.mark.parametrize("engine", ENGINES)
def test_group_imputer_uses_training_group_means_with_global_fallback(engine):
    """Gaps get their training group's mean; unseen, null or all-missing groups get the global mean."""
    config = {"columns": ["employees"], "group_by": "industry", "strategy": "mean"}
    artifact, result = _fit_apply(
        "GroupImputer", config, _native(_TRAIN, engine), _native(_SCORE, engine)
    )
    result = _pandas(result)
    overall = (10 + 20 + 100 + 300 + 7) / 5
    # A=15, B=200, C has no observed value -> global, NEW/None -> global, 9999 kept.
    assert result["employees"].tolist() == [15.0, 200.0, overall, overall, overall, 9999.0]
    assert result["industry"].tolist()[:4] == ["A", "B", "C", "NEW"]
    if engine == "pandas":
        assert result.index.tolist() == [5, 3, 8, 1, 0, 9]
    json.dumps(artifact, allow_nan=False)  # artifact must be saveable as plain JSON


@pytest.mark.parametrize("engine", ENGINES)
def test_group_imputer_most_frequent_fills_text_and_breaks_ties_by_smallest(engine):
    """Mode per group works on text; a tie picks the smallest value, matching SimpleImputer."""
    config = {"columns": ["region"], "group_by": "industry", "strategy": "most_frequent"}
    _, result = _fit_apply("GroupImputer", config, _native(_TRAIN, engine), _native(_SCORE, engine))
    # A: north x2 -> north. B: east. C: no value -> global tie (east x2, north x2) -> east.
    assert _pandas(result)["region"].tolist() == ["north", "east", "east", "east", "east", "north"]


@pytest.mark.parametrize("engine", ENGINES)
def test_group_imputer_median_and_numeric_group_keys(engine):
    """Numeric group keys keep their type in the artifact and still match at apply time."""
    train = pd.DataFrame({"size": [1, 1, 1, 2, 2], "value": [1.0, 2.0, 30.0, 5.0, np.nan]})
    score = pd.DataFrame({"size": [1, 2, 3], "value": [np.nan, np.nan, np.nan]})
    config = {"columns": ["value"], "group_by": "size", "strategy": "median"}
    _, result = _fit_apply("GroupImputer", config, _native(train, engine), _native(score, engine))
    assert _pandas(result)["value"].tolist() == [2.0, 5.0, 3.5]


@pytest.mark.parametrize("engine", ENGINES)
def test_group_imputer_fills_nullable_integers_as_float(engine):
    """A mean cannot be stored in an integer column, so nullable integers become float."""
    train = pd.DataFrame({"g": ["a", "a", "b"], "n": pd.array([1, 2, None], dtype="Int64")})
    config = {"columns": ["n"], "group_by": "g", "strategy": "mean"}
    _, result = _fit_apply("GroupImputer", config, _native(train, engine), _native(train, engine))
    assert _pandas(result)["n"].tolist() == [1.0, 2.0, 1.5]


def test_group_imputer_treats_polars_nan_as_missing():
    """Polars keeps NaN apart from null; both are gaps to fill and NaN must not poison the mean."""
    train = pl.DataFrame({"g": ["a", "a", "a"], "x": [1.0, float("nan"), None]})
    config = {"columns": ["x"], "group_by": "g", "strategy": "mean"}
    _, result = _fit_apply("GroupImputer", config, train, train)
    assert result["x"].to_list() == [1.0, 1.0, 1.0]


@pytest.mark.parametrize(
    ("config", "message"),
    [
        ({"columns": ["employees"], "strategy": "mean"}, "group_by"),
        ({"columns": ["employees"], "group_by": "missing", "strategy": "mean"}, "missing"),
        ({"columns": ["industry"], "group_by": "industry", "strategy": "most_frequent"}, "itself"),
        ({"columns": ["region"], "group_by": "industry", "strategy": "mean"}, "numeric"),
        ({"columns": ["employees"], "group_by": "industry", "strategy": "constant"}, "strategy"),
    ],
)
def test_group_imputer_rejects_ambiguous_configs(config, message):
    """Configuration mistakes fail at fit time with a message naming the problem."""
    with pytest.raises(ValueError, match=message):
        NodeRegistry.get_calculator("GroupImputer")().fit(_TRAIN, config)


def test_group_imputer_requires_group_column_when_scoring():
    """Silently using the global value for every row would hide a broken scoring query."""
    config = {"columns": ["employees"], "group_by": "industry", "strategy": "mean"}
    artifact = NodeRegistry.get_calculator("GroupImputer")().fit(_TRAIN, config)
    with pytest.raises(ValueError, match="industry"):
        NodeRegistry.get_applier("GroupImputer")().apply(_SCORE.drop(columns="industry"), artifact)


def test_group_imputer_auto_selects_numeric_columns_except_group_and_target():
    """With no columns, mean fills every numeric input but never the group key or target."""
    train = _TRAIN.assign(size=[1, 1, 2, 2, 3, 3, 3], label=[0.0, np.nan, 1, 1, 0, 1, 0])
    config = {"group_by": "size", "strategy": "mean", "target_column": "label"}
    artifact = NodeRegistry.get_calculator("GroupImputer")().fit(train, config)
    assert artifact["columns"] == ["employees"]


_CLIP_DATA = pd.DataFrame(
    {
        "employees": [-5.0, 50.0, 11000.0, np.nan],
        "year": pd.array([1850, 1990, 2030, None], dtype="Int64"),
        "name": ["a", "b", "c", "d"],
    },
    index=[4, 2, 7, 1],
)
_CLIP_BOUNDS = {"employees": {"lower": 0, "upper": 500}, "year": {"lower": 1900}}


@pytest.mark.parametrize("engine", ENGINES)
def test_clip_values_limits_to_fixed_bounds_and_keeps_every_row(engine):
    """Values outside a bound are set to it; nulls, other columns and row order stay unchanged."""
    _, result = _fit_apply(
        "ClipValues",
        {"bounds": _CLIP_BOUNDS},
        _native(_CLIP_DATA, engine),
        _native(_CLIP_DATA, engine),
    )
    result = _pandas(result)
    assert result["employees"].tolist()[:3] == [0.0, 50.0, 500.0]
    assert pd.isna(result["employees"].iloc[3])
    assert result["year"].tolist()[:3] == [1900, 1990, 2030]
    assert pd.isna(result["year"].iloc[3])
    assert result["name"].tolist() == ["a", "b", "c", "d"]
    if engine == "pandas":
        assert result.index.tolist() == [4, 2, 7, 1]


@pytest.mark.parametrize(
    ("bounds", "message"),
    [
        ({"employees": {"lower": 10, "upper": 1}}, "lower"),
        ({"employees": {}}, "lower or upper"),
        ({"name": {"lower": 0}}, "numeric"),
        ({"missing": {"lower": 0}}, "missing"),
        ({"employees": {"lower": float("nan")}}, "finite"),
    ],
)
def test_clip_values_rejects_invalid_bounds(bounds, message):
    """A typo in the bounds must fail early instead of clipping the wrong way."""
    with pytest.raises(ValueError, match=message):
        NodeRegistry.get_calculator("ClipValues")().fit(_CLIP_DATA, {"bounds": bounds})


@pytest.mark.parametrize("engine", ENGINES)
def test_new_nodes_run_at_inference_through_the_pipeline(engine):
    """Unlike row-dropping outlier nodes, both steps must also run on scoring rows."""
    steps = [
        {
            "name": "fill",
            "transformer": "GroupImputer",
            "params": {"columns": ["employees"], "group_by": "industry", "strategy": "mean"},
        },
        {
            "name": "clip",
            "transformer": "ClipValues",
            "params": {"bounds": {"employees": {"upper": 150}}},
        },
    ]
    engineer = FeatureEngineer(steps)
    engineer.fit_transform(_native(_TRAIN, engine))
    result = _pandas(engineer.transform(_native(_SCORE, engine)))
    assert result["employees"].tolist()[:2] == [15.0, 150.0]
    assert result["employees"].iloc[-1] == 150.0
