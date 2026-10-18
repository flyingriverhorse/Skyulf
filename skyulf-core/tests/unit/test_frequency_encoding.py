"""Learned category frequencies preserve missing values and batch-independent replay."""

import copy

import numpy as np
import pandas as pd
import polars as pl
import pytest

from skyulf.registry import NodeRegistry


def _frame(engine, values):
    """Create frames with a retained numeric column and duplicate pandas indexes."""
    data = {"cat": values, "number": list(range(len(values)))}
    return pl.DataFrame(data) if engine == "polars" else pd.DataFrame(data, index=[7] * len(values))


@pytest.mark.parametrize("fit_engine", ["pandas", "polars"])
@pytest.mark.parametrize("apply_engine", ["pandas", "polars"])
@pytest.mark.parametrize("mode", ["frequency", "count"])
def test_frequency_counts_missing_unseen_chunks_and_empty(fit_engine, apply_engine, mode):
    """Saved counts include missing rows and cannot depend on inference batch composition."""
    calc = NodeRegistry.get_calculator("FrequencyEncoder")()
    applier = NodeRegistry.get_applier("FrequencyEncoder")()
    state = calc.fit(
        _frame(fit_engine, ["a", "a", None, "nan"]), {"columns": ["cat"], "mode": mode}
    )
    original = copy.deepcopy(state)
    frame = _frame(apply_engine, [None, "a", "new", "nan"])
    output = applier.apply(frame, state)
    expected = np.array([1, 2, 0, 1]) / (4 if mode == "frequency" else 1)
    np.testing.assert_allclose(output["cat"].to_numpy(), expected)
    for index in range(len(frame)):
        chunk = frame.slice(index, 1) if apply_engine == "polars" else frame.iloc[index : index + 1]
        np.testing.assert_allclose(
            applier.apply(chunk, state)["cat"].to_numpy(), expected[index : index + 1]
        )
    empty = frame.head(0)
    result = applier.apply(empty, state)
    assert str(result["cat"].dtype).lower() == ("float64" if mode == "frequency" else "int64")
    assert list(output.columns) == ["cat", "number"]
    assert state == original
    assert applier.validate_inference_state(state) is state


@pytest.mark.parametrize("engine", ["pandas", "polars"])
def test_frequency_auto_excludes_target_and_explicit_empty_is_noop(engine):
    """Automatic selection must preserve labels while an empty explicit list learns nothing."""
    calc = NodeRegistry.get_calculator("FrequencyEncoder")()
    frame = _frame(engine, ["a", "b"])
    assert calc.fit(frame, {"target_column": "cat"}) == {}
    assert calc.fit(frame, {"columns": []}) == {}
    with pytest.raises(ValueError, match="training rows"):
        calc.fit(frame.head(0), {"columns": ["cat"]})
    with pytest.raises(ValueError, match="mode"):
        calc.fit(frame, {"columns": ["cat"], "mode": "percentage"})


@pytest.mark.parametrize(
    "mutate",
    [
        lambda s: s.update(category_key_version=2),
        lambda s: s.update(n_samples=True),
        lambda s: s.update(columns=["cat", "cat"]),
        lambda s: s.update(mode="invalid"),
        lambda s: s["mappings"]["cat"].update(a=0),
        lambda s: s["mappings"].update(extra={"a": 2}),
        lambda s: s.update(extra=True),
    ],
)
def test_frequency_rejects_corrupt_saved_state(mutate):
    """Admission must reject corrupt learned statistics rather than normalize them."""
    state = NodeRegistry.get_calculator("FrequencyEncoder")().fit(_frame("pandas", ["a", "a"]), {})
    mutate(state)
    with pytest.raises(ValueError):
        NodeRegistry.get_applier("FrequencyEncoder")().validate_inference_state(state)


@pytest.mark.parametrize("engine", ["pandas", "polars"])
def test_frequency_all_missing_and_paired_target(engine):
    """All-null training is a learned category and a named paired label remains untouched."""
    calc = NodeRegistry.get_calculator("FrequencyEncoder")()
    frame = _frame(engine, [None, None])
    state = calc.fit(frame, {"columns": ["cat"]})
    output = NodeRegistry.get_applier("FrequencyEncoder")().apply(frame, state)
    np.testing.assert_array_equal(output["cat"].to_numpy(), [1.0, 1.0])
    y = pl.Series("cat", [0, 1]) if engine == "polars" else pd.Series([0, 1], name="cat")
    assert calc.fit((frame, y), {"columns": ["cat"]}) == {}
