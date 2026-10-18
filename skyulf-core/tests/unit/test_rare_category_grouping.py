"""Rare grouping learns text membership and preserves a stable nullable string schema."""

import numpy as np
import pandas as pd
import polars as pl
import pytest

from skyulf.pipeline.seal import artifact_digest
from skyulf.registry import NodeRegistry


def _frame(engine, values):
    """Keep native nulls and duplicate pandas indexes visible to the applier."""
    return (
        pl.DataFrame({"cat": values})
        if engine == "polars"
        else pd.DataFrame({"cat": values}, index=[8] * len(values))
    )


@pytest.mark.parametrize("fit_engine", ["pandas", "polars"])
@pytest.mark.parametrize("apply_engine", ["pandas", "polars"])
def test_rare_thresholds_missing_unseen_and_empty(fit_engine, apply_engine):
    """Inclusive conjunctive thresholds learn null membership and replay on either engine."""
    calc = NodeRegistry.get_calculator("RareCategoryGrouping")()
    applier = NodeRegistry.get_applier("RareCategoryGrouping")()
    state = calc.fit(
        _frame(fit_engine, ["a", "a", None, None, "b", "nan"]), {"min_count": 2, "min_share": 1 / 3}
    )
    frame = _frame(apply_engine, ["a", None, "b", "new", "nan"])
    out = applier.apply(frame, state)
    values = out["cat"].to_list() if apply_engine == "polars" else out["cat"].tolist()
    assert values[0] == "a" and pd.isna(values[1])
    assert values[2:] == ["Other"] * 3
    assert str(applier.apply(frame.head(0), state)["cat"].dtype).lower() == "string"
    assert applier.validate_inference_state(state) is state
    strict = calc.fit(_frame(fit_engine, ["a", "a", "b"]), {"min_count": 2, "min_share": 0.8})
    assert applier.apply(_frame(apply_engine, ["a", None]), strict)["cat"].to_list() == [
        "Other",
        "Other",
    ]


@pytest.mark.parametrize("engine", ["pandas", "polars"])
@pytest.mark.parametrize(
    "config",
    [
        {"min_count": 0},
        {"min_count": True},
        {"min_count": 2.5},
        {"min_share": -0.1},
        {"min_share": 1.1},
        {"min_share": float("nan")},
        {"min_share": True},
        {"replacement": ""},
        {"replacement": None},
    ],
)
def test_rare_rejects_invalid_options(engine, config):
    """Malformed grouping recipes must fail before producing a fitted artifact."""
    with pytest.raises(ValueError, match="RareCategoryGrouping"):
        NodeRegistry.get_calculator("RareCategoryGrouping")().fit(_frame(engine, ["a"]), config)


@pytest.mark.parametrize("engine", ["pandas", "polars"])
def test_rare_rejects_nontext_at_fit_and_apply(engine):
    """Numeric categories require explicit Casting instead of silent string coercion."""
    calc = NodeRegistry.get_calculator("RareCategoryGrouping")()
    applier = NodeRegistry.get_applier("RareCategoryGrouping")()
    numeric = _frame(engine, [1, 2])
    with pytest.raises(ValueError, match="cat.*text.*Casting"):
        calc.fit(numeric, {"columns": ["cat"]})
    state = calc.fit(_frame(engine, ["a", "a"]), {})
    with pytest.raises(ValueError, match="cat.*text.*Casting"):
        applier.apply(numeric, state)
    assert calc.fit(numeric, {"columns": []}) == {}


@pytest.mark.parametrize("kind", ["object", "string", "category", "polars", "enum"])
def test_rare_nulls_categorical_types_and_chunks(kind):
    """Native text category dtypes and every missing scalar share fixed learned membership."""
    raw = ["a", "a", None, None, "b"]
    if kind in ("polars", "enum"):
        dtype = pl.Categorical if kind == "polars" else pl.Enum(["a", "b"])
        frame = pl.DataFrame({"cat": pl.Series(raw, dtype=dtype)})
    else:
        frame = pd.DataFrame({"cat": pd.Series(["a", "a", pd.NA, np.nan, "b"], dtype=kind)})
    calc = NodeRegistry.get_calculator("RareCategoryGrouping")()
    applier = NodeRegistry.get_applier("RareCategoryGrouping")()
    state = calc.fit(frame, {"columns": ["cat"]})
    digest = artifact_digest(state)
    expected = applier.apply(frame, state)
    for i in range(len(frame)):
        chunk = frame[i : i + 1] if isinstance(frame, pl.DataFrame) else frame.iloc[[i]]
        actual = applier.apply(chunk, state)["cat"][0 if isinstance(frame, pl.DataFrame) else i]
        wanted = expected["cat"][i]
        assert pd.isna(actual) if pd.isna(wanted) else actual == wanted
    assert artifact_digest(state) == digest


@pytest.mark.parametrize(
    "mutate",
    [
        lambda s: s.update(category_key_version=True),
        lambda s: s.update(replacement=""),
        lambda s: s.update(columns=["cat", "cat"]),
        lambda s: s["kept_categories"].update(extra=[]),
        lambda s: s["kept_categories"].update(cat=["a", "a"]),
        lambda s: s["kept_categories"].update(cat=[1]),
        lambda s: s.update(extra=True),
    ],
)
def test_rare_rejects_corrupt_saved_membership(mutate):
    """Unexpected keys, duplicates and unsupported versions cannot acquire local admission."""
    state = NodeRegistry.get_calculator("RareCategoryGrouping")().fit(
        _frame("pandas", ["a", "a"]), {}
    )
    mutate(state)
    with pytest.raises(ValueError):
        NodeRegistry.get_applier("RareCategoryGrouping")().validate_inference_state(state)


@pytest.mark.parametrize("engine", ["pandas", "polars"])
def test_rare_all_missing_empty_training_and_target_exclusion(engine):
    """Selected null-only input learns membership while zero rows cannot define shares."""
    calc = NodeRegistry.get_calculator("RareCategoryGrouping")()
    frame = _frame(engine, [None, None])
    state = calc.fit(frame, {"columns": ["cat"]})
    result = NodeRegistry.get_applier("RareCategoryGrouping")().apply(frame, state)
    assert all(pd.isna(v) for v in result["cat"])
    assert calc.fit(frame, {"columns": ["cat"], "target_column": "cat"}) == {}
    with pytest.raises(ValueError, match="training rows"):
        calc.fit(frame.head(0), {"columns": ["cat"]})
