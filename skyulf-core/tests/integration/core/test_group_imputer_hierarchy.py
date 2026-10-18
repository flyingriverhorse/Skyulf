"""Ordered group imputation learns prefix fallbacks once and retains legacy replay."""

import json
from copy import deepcopy
from typing import Any

import numpy as np
import pandas as pd
import polars as pl
import pytest
from polars.testing import assert_frame_equal as assert_polars_equal

from skyulf import SkyulfPipeline
from skyulf.inference.fitted_pipeline import load_pipeline, predict_pipeline, save_pipeline
from skyulf.preprocessing.imputation.group import GroupImputerApplier, GroupImputerCalculator
from skyulf.preprocessing.inference_context import get_inference_capability


def _frame(frame, engine):
    """Preserve exact native scalar values and pandas labels."""
    return pl.from_pandas(frame) if engine == "polars" else frame


def _config(fallback="hierarchical", strategy="mean"):
    """Use an explicit ordered multi-key contract with a single value to fill."""
    return {
        "columns": ["value"],
        "group_by": ["region", "industry"],
        "strategy": strategy,
        "fallback": fallback,
    }


def _train():
    """Give full, regional and global levels different learned values."""
    return pd.DataFrame(
        {
            "region": ["north", "north", "north", "south", None],
            "industry": ["a", "a", "b", "a", "a"],
            "value": [2.0, 4.0, 12.0, 22.0, 40.0],
        }
    )


@pytest.mark.parametrize("engine", ["pandas", "polars"])
@pytest.mark.parametrize(
    "fallback,expected",
    [
        ("hierarchical", [3.0, 6.0, 16.0, 6.0, 16.0, 99.0]),
        ("global", [3.0, 16.0, 16.0, 16.0, 16.0, 99.0]),
    ],
)
def test_multi_key_imputer_uses_ordered_prefixes_and_global_fallback(engine, fallback, expected):
    """Missing key components skip only that level and unseen groups never use request values."""
    config = _config(fallback)
    state = json.loads(json.dumps(GroupImputerCalculator().fit(_frame(_train(), engine), config)))
    saved = deepcopy(state)
    query = pd.DataFrame(
        {
            "region": ["north", "north", "new", "north", None, "north"],
            "industry": ["a", "new", "a", None, "a", "a"],
            "value": [None, None, None, None, None, 99.0],
        },
        index=[9, 2, 9, -1, 3, 4],
    )
    native = _frame(query, engine)
    result = GroupImputerApplier().apply(native, state)
    np.testing.assert_allclose(result["value"].to_numpy(), expected)
    for positions in ([0], [2, 3], [5, 1, 0], []):
        subset = native[positions] if engine == "polars" else native.iloc[positions]
        want = result[positions] if engine == "polars" else result.iloc[positions]
        got = GroupImputerApplier().apply(subset, state)
        if engine == "polars":
            assert_polars_equal(got, want)
        else:
            pd.testing.assert_frame_equal(got, want)
    capability = get_inference_capability("GroupImputer", config, state, engine=engine)
    assert capability is not None
    assert capability.context == "row"
    assert capability.execution_kind == ("python_batch" if engine == "pandas" else "local")
    assert capability.codec_version is None
    assert state == saved


@pytest.mark.parametrize("engine", ["pandas", "polars"])
@pytest.mark.parametrize("strategy,expected", [("median", 4.0), ("most_frequent", 2.0)])
def test_hierarchy_uses_each_configured_training_statistic(engine, strategy, expected):
    """Fallback levels must learn medians or modes directly rather than averages of child statistics."""
    config = _config(strategy=strategy)
    state = GroupImputerCalculator().fit(_frame(_train(), engine), config)
    query = _frame(
        pd.DataFrame({"region": ["north"], "industry": ["new"], "value": [np.nan]}), engine
    )
    assert GroupImputerApplier().apply(query, state)["value"].to_list() == [expected]


@pytest.mark.parametrize("engine", ["pandas", "polars"])
def test_new_single_key_list_keeps_exact_large_integer_group_and_mode_values(engine):
    """The opt-in tuple-key representation must retain integers beyond float precision."""
    big = 2**53 + 1
    train = pd.DataFrame(
        {
            "region": pd.Series([big, big + 1], dtype="Int64"),
            "value": pd.Series([big + 7, big + 9], dtype="Int64"),
        }
    )
    config = {"columns": ["value"], "group_by": ["region"], "strategy": "most_frequent"}
    state = GroupImputerCalculator().fit(_frame(train, engine), config)
    query = train.copy()
    query["value"] = pd.Series([None, None], dtype="Int64")
    result = GroupImputerApplier().apply(_frame(query, engine), state)
    assert result["value"].to_list() == [big + 7, big + 9]


@pytest.mark.parametrize("engine", ["pandas", "polars"])
def test_modal_float_fills_preserve_dtype_for_full_singleton_and_empty_batches(engine):
    """A float-valued mode must not turn nonempty pandas batches into object columns."""
    config = _config(strategy="most_frequent")
    state = GroupImputerCalculator().fit(_frame(_train(), engine), config)
    query = pd.DataFrame(
        {"region": ["north", "new"], "industry": ["a", "a"], "value": [np.nan, 7.0]}
    )
    result = GroupImputerApplier().apply(_frame(query, engine), state)
    expected = query.assign(value=[2.0, 7.0])
    for positions in ([0, 1], [0], []):
        got = GroupImputerApplier().apply(_frame(query.iloc[positions], engine), state)
        want = _frame(expected.iloc[positions], engine)
        if engine == "polars":
            assert_polars_equal(got, want)
        else:
            pd.testing.assert_frame_equal(got, want)
    assert result["value"].to_list() == [2.0, 7.0]


@pytest.mark.parametrize(
    "group_by,fallback",
    [
        ([], "global"),
        (["region", "region"], "global"),
        (["region", 1], "global"),
        (["missing"], "global"),
        (["value"], "global"),
        (["region"], "bad"),
    ],
)
def test_invalid_multi_group_configuration_fails_before_learning(group_by, fallback):
    """Malformed keys and fallback choices must not publish partial or ambiguous learned state."""
    with pytest.raises(ValueError, match="group|fallback"):
        GroupImputerCalculator().fit(
            _train(), {**_config(), "group_by": group_by, "fallback": fallback}
        )


@pytest.mark.parametrize(
    "mutation", ["version", "prefix", "width", "duplicate", "fallback", "config_order"]
)
def test_hierarchy_admission_rejects_malformed_state_or_mismatched_config(mutation):
    """The existing python-batch route must inspect every new level and bind it to its fit config."""
    config = _config()
    state: dict[str, Any] = dict(GroupImputerCalculator().fit(_train(), config))
    if mutation == "version":
        state["group_key_version"] = True
    elif mutation == "prefix":
        state["levels"][1]["group_by"] = ["industry"]
    elif mutation == "width":
        state["levels"][0]["group_values"]["value"][0][0].pop()
    elif mutation == "duplicate":
        state["levels"][0]["group_values"]["value"].append(
            state["levels"][0]["group_values"]["value"][0]
        )
    elif mutation == "fallback":
        config["fallback"] = "global"
    else:
        config["group_by"] = ["industry", "region"]
    with pytest.raises(ValueError):
        validated = GroupImputerApplier.validate_fitted_state(state)
        GroupImputerApplier.resolve_fitted_config(config, validated)
    assert get_inference_capability("GroupImputer", config, state, engine="pandas") is None


@pytest.mark.parametrize("engine", ["pandas", "polars"])
def test_three_level_hierarchy_keeps_tuple_boundaries_and_missing_prefix_rules(engine):
    """Delimited key strings must not collide and a null leading key must use the global value."""
    train = pd.DataFrame(
        {
            "region": ["a|b", "a", "a", None],
            "industry": ["c", "b|c", "other", "b|c"],
            "site": ["x", "x", "x", "x"],
            "value": [2.0, 10.0, 30.0, 50.0],
        }
    )
    config = {**_config(), "group_by": ["region", "industry", "site"]}
    state = GroupImputerCalculator().fit(_frame(train, engine), config)
    query = pd.DataFrame(
        {
            "region": ["a|b", "a", "a", "a", None],
            "industry": ["c", "b|c", "b|c", None, "b|c"],
            "site": ["x", "x", None, "x", "x"],
            "value": [np.nan] * 5,
        }
    )
    result = GroupImputerApplier().apply(_frame(query, engine), state)
    assert result["value"].to_list() == [2.0, 10.0, 10.0, 20.0, 23.0]


def test_tuple_keys_keep_typed_boolean_numeric_text_and_literal_missing_distinct():
    """Canonical keys must preserve type identity even when pandas equality would merge groups."""
    train = pd.DataFrame(
        {
            "region": pd.Series([True, 1, "1", "nan", None], dtype=object),
            "value": [2.0, 4.0, 8.0, 16.0, 32.0],
        }
    )
    state = GroupImputerCalculator().fit(
        train, {"columns": ["value"], "group_by": ["region"], "strategy": "mean"}
    )
    result = GroupImputerApplier().apply(train.assign(value=np.nan), state)
    assert result["value"].to_list() == [2.0, 4.0, 8.0, 16.0, 12.4]


@pytest.mark.parametrize("engine", ["pandas", "polars"])
def test_hierarchical_imputer_survives_real_model_save_load_and_cross_engine_replay(
    engine, tmp_path
):
    """Persisted models must keep training prefix statistics across engines and scoring chunks."""
    frame = pd.DataFrame(
        {
            "region": [1, 1, 1, 2, 2, 2],
            "industry": [1, 1, 2, 1, 1, 2],
            "value": [2.0, 4.0, np.nan, 10.0, 12.0, np.nan],
            "target": [1.0, 2.0, 3.0, 4.0, 5.0, 6.0],
        }
    )
    pipeline = SkyulfPipeline(
        {
            "preprocessing": [{"name": "fill", "transformer": "GroupImputer", "params": _config()}],
            "modeling": {"type": "decision_tree_regressor", "params": {"random_state": 42}},
        }
    )
    pipeline.fit(_frame(frame, engine), target_column="target")
    save_pipeline(pipeline, tmp_path / "groups")
    restored = load_pipeline(tmp_path / "groups")
    query = pd.DataFrame(
        {"region": [1, 3, 2], "industry": [9, 1, 1], "value": [np.nan, np.nan, np.nan]}
    )
    expected = predict_pipeline(query, restored)["prediction"].to_numpy()
    for apply_engine in ("pandas", "polars"):
        for positions in ([0, 1, 2], [2, 0], [1]):
            got = predict_pipeline(_frame(query.iloc[positions], apply_engine), restored)
            np.testing.assert_array_equal(got["prediction"].to_numpy(), expected[positions])
    assert (tmp_path / "groups").exists()
