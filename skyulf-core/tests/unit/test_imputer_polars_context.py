"""Reviewed local imputer declarations retain saved-state ownership and native boundaries."""

import pickle
from typing import Any

import pandas as pd
import pytest

pl = pytest.importorskip("polars")

from polars.testing import assert_frame_equal as assert_polars_frame_equal

from skyulf.core.capabilities import UnsupportedExecutionError, require_capability
from skyulf.preprocessing.inference_context import get_inference_capability
from skyulf.registry import NodeRegistry


def _frame(engine, values, dtype="Int64"):
    """Keep nullable storage and named columns stable across fit and empty inference requests."""
    result = pd.DataFrame({"g": ["a", "a", "b", None], "x": pd.Series(values, dtype=dtype)})
    return pl.from_pandas(result) if engine == "polars" else result


def _assert_frame(actual, expected):
    """Require exact values, null masks, row order and dtype for each supported container."""
    if isinstance(actual, pl.DataFrame):
        assert_polars_frame_equal(actual, expected, check_exact=True)
    else:
        pd.testing.assert_frame_equal(actual, expected, check_exact=True)


def _assert_replay(node, state, sample):
    """Replay saved values across request sizes without mutating the caller or learned artifact."""
    saved = pickle.dumps(state)
    before = sample.clone() if isinstance(sample, pl.DataFrame) else sample.copy(deep=True)
    applier = NodeRegistry.get_applier(node)()
    full = applier.apply(sample, state)
    _assert_frame(applier.apply(sample, state), full)
    for start in range(len(sample)):
        request = (
            sample.slice(start, 1)
            if isinstance(sample, pl.DataFrame)
            else sample.iloc[start : start + 1]
        )
        expected = (
            full.slice(start, 1) if isinstance(full, pl.DataFrame) else full.iloc[start : start + 1]
        )
        _assert_frame(applier.apply(request, state), expected)
    reverse = sample.reverse() if isinstance(sample, pl.DataFrame) else sample.iloc[::-1]
    expected_reverse = full.reverse() if isinstance(full, pl.DataFrame) else full.iloc[::-1]
    _assert_frame(applier.apply(reverse, state), expected_reverse)
    _assert_frame(applier.apply(sample.head(0), state), full.head(0))
    _assert_frame(sample, before)
    assert pickle.dumps(state) == saved
    return full


@pytest.mark.parametrize("engine", ["pandas", "polars"])
@pytest.mark.parametrize(
    "node,strategy",
    [
        ("SimpleImputer", "mean"),
        ("SimpleImputer", "constant"),
        ("SimpleImputer", "most_frequent"),
        ("SimpleImputer", "mode"),
        ("GroupImputer", "mean"),
        ("GroupImputer", "most_frequent"),
        ("GroupImputer", "mode"),
    ],
)
def test_real_imputer_state_declares_local_polars_and_replays(node, strategy, engine, monkeypatch):
    """Group null/unseen keys use saved fallback and local metadata never needs fit or apply."""
    train = _frame(engine, [1, 3, None, 9])
    before = train.clone() if engine == "polars" else train.copy(deep=True)
    config = {"columns": ["x"], "strategy": strategy}
    if node == "GroupImputer":
        config["group_by"] = "g"
    if strategy == "constant":
        config["fill_value"] = 7
    calculator = NodeRegistry.get_calculator(node)
    owner: Any = NodeRegistry.get_applier(node)
    state = calculator().fit(train, config)
    _assert_frame(train, before)
    saved = pickle.dumps(state)
    saved_config = pickle.dumps(config)

    def forbidden(*args, **kwargs):
        """Metadata inspection must never execute transformation or learn from inference data."""
        raise AssertionError("Context query invoked fit or apply.")

    monkeypatch.setattr(calculator, "fit", forbidden)
    with monkeypatch.context() as query:
        query.setattr(owner, "apply", forbidden)
        for declared_engine, kind in [("pandas", "python_batch"), ("polars", "local")]:
            capability = get_inference_capability(node, config, state, engine=declared_engine)
            assert capability is not None
            assert (capability.context, capability.row_effect, capability.execution_kind) == (
                "row",
                "preserve",
                kind,
            )
            expected_codec = (
                1 if node == "SimpleImputer" and strategy in ("mean", "constant") else None
            )
            assert capability.codec_version == expected_codec
        normalized = owner.resolve_fitted_config(config, owner.validate_fitted_state(state))
        assert normalized["strategy"] == ("most_frequent" if strategy == "mode" else strategy)
        for kind in ("native", "python_batch"):
            with pytest.raises(UnsupportedExecutionError):
                require_capability(node, "apply", "polars", config=normalized, execution_kind=kind)
    sample = _frame(engine, [None, None, None, 11])
    if engine == "polars":
        sample = sample.with_columns(pl.Series("g", ["a", "b", "new", None]))
    else:
        sample["g"] = ["a", "b", "new", None]
    full = _assert_replay(node, state, sample)
    expected_first = (
        state["group_values"]["x"][0][1] if node == "GroupImputer" else state["fill_values"]["x"]
    )
    assert full["x"].to_list() == [
        expected_first,
        state["fill_values"]["x"],
        state["fill_values"]["x"],
        11,
    ]
    assert pickle.dumps(state) == saved
    assert pickle.dumps(config) == saved_config


@pytest.mark.parametrize("engine", ["pandas", "polars"])
@pytest.mark.parametrize("node", ["SimpleImputer", "GroupImputer"])
@pytest.mark.parametrize("strategy", ["most_frequent", "mode"])
@pytest.mark.parametrize(
    "dtype,values",
    [
        ("string", ["a", "a", None, "b"]),
        ("boolean", [True, True, None, False]),
    ],
)
def test_modal_imputer_scalar_types_replay_without_new_promotions(
    engine, node, strategy, dtype, values
):
    """Scalar mode declarations must retain string and nullable boolean values and empty schemas."""
    train = _frame(engine, values, dtype)
    config = {"columns": ["x"], "strategy": strategy}
    if node == "GroupImputer":
        config["group_by"] = "g"
    state = NodeRegistry.get_calculator(node)().fit(train, config)
    capability = get_inference_capability(node, config, state, engine="polars")
    assert capability is not None and capability.execution_kind == "local"
    full = _assert_replay(node, state, _frame(engine, [None, values[0], None, None], dtype))
    assert full["x"].to_list() == [values[0]] * 4


@pytest.mark.parametrize("strategy", ["mean", "constant", "most_frequent", "mode"])
def test_simple_restores_missing_fitted_column_with_existing_row_anchor(strategy):
    """A fitted column is restored per existing row and remains empty when its anchor is empty."""
    config = {"columns": ["x"], "strategy": strategy}
    if strategy == "constant":
        config["fill_value"] = 7
    state = NodeRegistry.get_calculator("SimpleImputer")().fit(
        pl.DataFrame({"x": [1.0, 3.0]}), config
    )
    sample = pl.DataFrame({"keep": [10, 20]})
    full = _assert_replay("SimpleImputer", state, sample)
    assert full["x"].to_list() == [state["fill_values"]["x"]] * 2


@pytest.mark.parametrize("engine", ["pandas", "polars"])
@pytest.mark.parametrize(
    "change",
    [
        {"strategy": "median"},
        {"columns": ["g"]},
        {"fill_value": 7},
        {"unknown_option": True},
    ],
)
def test_simple_mode_alias_retains_fitted_recipe_binding(engine, change):
    """Accepting a spelling alias must not hide a changed strategy, column or unsupported option."""
    config = {"columns": ["x"], "strategy": "mode"}
    state = NodeRegistry.get_calculator("SimpleImputer")().fit(
        _frame(engine, [1, 3, None, 9]), config
    )
    assert (
        get_inference_capability("SimpleImputer", {**config, **change}, state, engine=engine)
        is None
    )


@pytest.mark.parametrize("node", ["SimpleImputer", "GroupImputer"])
def test_imputer_invalid_state_config_and_identity_remain_unknown(node):
    """Declarations retain existing artifact checks and cannot be borrowed by modified appliers."""
    config = {"columns": ["x"], "strategy": "mean"}
    if node == "GroupImputer":
        config["group_by"] = "g"
    state = NodeRegistry.get_calculator(node)().fit(_frame("polars", [1, 3, None, 9]), config)
    malformed = {**state, "fill_values": {}}
    assert get_inference_capability(node, config, malformed, engine="polars") is None
    assert (
        get_inference_capability(node, {**config, "columns": ["missing"]}, state, engine="polars")
        is None
    )
    applier = NodeRegistry.get_applier(node)()
    applier.apply = lambda *args, **kwargs: None
    assert get_inference_capability(node, config, state, engine="polars", applier=applier) is None


@pytest.mark.parametrize("engine", ["pandas", "polars"])
@pytest.mark.parametrize(
    "node,strategy",
    [
        ("GroupImputer", "mean"),
        ("GroupImputer", "most_frequent"),
        ("SimpleImputer", "most_frequent"),
        ("SimpleImputer", "mode"),
    ],
)
def test_missing_global_statistic_retains_existing_state_boundary(engine, node, strategy):
    """Missing group fallbacks remain valid while a pandas all-null modal NaN stays unreviewed."""
    sample = _frame(engine, [None] * 4, "Float64")
    config = {"columns": ["x"], "strategy": strategy}
    if node == "GroupImputer":
        config["group_by"] = "g"
    state = NodeRegistry.get_calculator(node)().fit(sample, config)
    capability = get_inference_capability(node, config, state, engine="polars")
    assert (capability is None) == (node == "SimpleImputer" and engine == "pandas")
    full = _assert_replay(node, state, sample)
    missing = full["x"].is_null() if engine == "polars" else full["x"].isna()
    assert missing.all()
