"""Saved group medians have row context without admitting independent workers."""

import pickle
from copy import deepcopy

import pandas as pd
import pytest

pl = pytest.importorskip("polars")

from polars.testing import assert_frame_equal as assert_polars_equal

from skyulf.core.capabilities import UnsupportedExecutionError, require_capability
from skyulf.preprocessing.imputation.group import GroupImputerApplier, GroupImputerCalculator
from skyulf.preprocessing.inference_context import get_inference_capability

CONFIG = {"columns": ["x"], "group_by": "g", "strategy": "median"}


def _fit(engine):
    """Learn a fractional group median and a different global fallback."""
    frame = pd.DataFrame({"g": ["a", "a", "b", "b"], "x": [1.0, 4.0, 9.0, None]})
    return GroupImputerCalculator().fit(
        pl.from_pandas(frame) if engine == "polars" else frame, CONFIG
    )


@pytest.mark.parametrize("engine", ["pandas", "polars"])
def test_group_median_inspection_is_local_row_and_never_executes(engine, monkeypatch):
    """Context describes saved lookups while the existing worker and Spark gates stay closed."""
    state = pickle.loads(pickle.dumps(_fit(engine)))
    before = pickle.dumps(state)

    def forbidden(*args, **kwargs):
        """Metadata must not recalculate or apply training statistics."""
        raise AssertionError("Unexpected fit/apply")

    monkeypatch.setattr(GroupImputerCalculator, "fit", forbidden)
    monkeypatch.setattr(GroupImputerApplier, "apply", forbidden)
    for native in ("pandas", "polars"):
        capability = get_inference_capability("GroupImputer", CONFIG, state, engine=native)
        assert capability is not None
        assert (capability.context, capability.row_effect, capability.execution_kind) == (
            "row",
            "preserve",
            "local",
        )
        assert capability.codec_version is None
        with pytest.raises(UnsupportedExecutionError):
            require_capability(
                "GroupImputer", "apply", native, config=CONFIG, execution_kind="python_batch"
            )
    assert get_inference_capability("GroupImputer", CONFIG, state, engine="spark") is None
    assert pickle.dumps(state) == before


@pytest.mark.parametrize("engine", ["pandas", "polars"])
@pytest.mark.parametrize("dtype", ["Float32", "Float64", "Int64"])
def test_group_median_replays_fixed_fallbacks_at_every_request_size(engine, dtype):
    """Null or unseen group rows must use saved medians with exact per-request dtype parity."""
    state = _fit(engine)
    assert state["fill_values"]["x"] == 4.0
    assert state["group_values"]["x"] == [["a", 2.5], ["b", 9.0]]
    before = pickle.dumps(state)
    frame = pd.DataFrame({"g": ["a", "b", "new", None], "x": pd.Series([None] * 4, dtype=dtype)})
    frame.index = [7, 2, 2, 0]
    if engine == "polars":
        frame = pl.from_pandas(frame)
    original = deepcopy(frame)
    applier = GroupImputerApplier()
    full = applier.apply(frame, state)
    assert full["x"].to_list() == [2.5, 9.0, 4.0, 4.0]
    for positions in ([0], [1, 2], [3], [3, 2, 1, 0], []):
        request = frame[positions] if engine == "polars" else frame.iloc[positions]
        expected = full[positions] if engine == "polars" else full.iloc[positions]
        if engine == "polars":
            assert_polars_equal(applier.apply(request, state), expected, check_exact=True)
        else:
            pd.testing.assert_frame_equal(applier.apply(request, state), expected, check_exact=True)
    if engine == "polars":
        assert isinstance(frame, pl.DataFrame)
        assert isinstance(original, pl.DataFrame)
        assert_polars_equal(frame, original, check_exact=True)
    else:
        pd.testing.assert_frame_equal(frame, original, check_exact=True)
    assert pickle.dumps(state) == before


@pytest.mark.parametrize(
    "field,value",
    [
        ("fill_values", {"x": "2.5"}),
        ("fill_values", {"x": True}),
        ("fill_values", {"x": float("nan")}),
        ("group_values", {"x": [["a", "2.5"]]}),
        ("group_values", {"x": [["a", 2.5], ["a", 4.0]]}),
        ("group_by", "x"),
        ("columns", ["x", "x"]),
        ("strategy", "unreviewed"),
    ],
)
def test_group_median_invalid_state_abstains(field, value):
    """Numeric median declarations must not certify malformed or nonnumeric saved lookups."""
    state = deepcopy(_fit("pandas"))
    state[field] = value
    assert get_inference_capability("GroupImputer", CONFIG, state, engine="pandas") is None


@pytest.mark.parametrize(
    "change", [{"strategy": "mean"}, {"columns": ["other"]}, {"group_by": "other"}, {"extra": True}]
)
def test_group_median_mismatched_config_abstains(change):
    """The context declaration must remain bound to the exact saved recipe."""
    assert (
        get_inference_capability(
            "GroupImputer", {**CONFIG, **change}, _fit("pandas"), engine="polars"
        )
        is None
    )


@pytest.mark.parametrize("engine", ["pandas", "polars"])
def test_group_median_all_null_statistics_are_valid_saved_noops(engine):
    """A missing global median must not be fabricated from inference rows."""
    frame = pd.DataFrame({"g": ["a", None], "x": pd.Series([None, None], dtype="Float64")})
    if engine == "polars":
        frame = pl.from_pandas(frame)
    state = GroupImputerCalculator().fit(frame, CONFIG)
    assert state["fill_values"] == {"x": None} and state["group_values"] == {"x": []}
    assert get_inference_capability("GroupImputer", CONFIG, dict(state), engine=engine) is not None
    output = GroupImputerApplier().apply(frame, state)
    assert output["x"].null_count() == 2 if engine == "polars" else output["x"].isna().all()
