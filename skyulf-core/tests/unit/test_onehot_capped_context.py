"""Saved category caps describe local row context without widening worker admission."""

import pickle
from copy import deepcopy

import numpy as np
import pandas as pd
import polars as pl
import pytest
from polars.testing import assert_frame_equal as assert_polars_equal

from skyulf.core.capabilities import (
    ExecutionCapability,
    UnsupportedExecutionError,
    require_capability,
)
from skyulf.preprocessing.inference_context import get_inference_capability
from skyulf.registry import NodeRegistry


def _fitted(engine, options=None):
    """Keep rare leading categories and a second column with fewer categories than the cap."""
    values = ["a", "b", "b", "b", "c", "d", "d", "e", "f", "f", "f", "f", None]
    frame = pd.DataFrame({"category": values, "other": ["u", "v"] * 6 + ["u"], "x": range(13)})
    frame.index = [7, 2, 2, 0] + list(range(9))
    if engine == "polars":
        frame = pl.from_pandas(frame)
    config = {"columns": ["category", "other"], **(options or {})}
    state = NodeRegistry.get_calculator("OneHotEncoder")().fit(frame, config)
    return frame, config, state


def _take(frame, positions):
    """Preserve native schemas and duplicate pandas labels for positional partitions."""
    return frame.iloc[positions] if isinstance(frame, pd.DataFrame) else frame[positions]


def _equal(actual, expected):
    """A row promise includes values, dtypes, column order and pandas row labels."""
    if isinstance(actual, pd.DataFrame):
        pd.testing.assert_frame_equal(actual, expected, check_exact=True)
    else:
        assert_polars_equal(actual, expected, check_exact=True)


@pytest.mark.parametrize("engine", ["pandas", "polars"])
@pytest.mark.parametrize(
    "options",
    [
        {},
        {"max_categories": 20},
        {"max_categories": 7},
        {"max_categories": 3},
        {"max_categories": 1},
        {"max_categories": 3, "drop_first": True},
        {"max_categories": 1, "drop_first": True},
        {"max_categories": 3, "drop_original": False},
    ],
)
def test_capped_onehot_replays_exact_native_rows(engine, options):
    """Fitted caps and rare-category mappings must survive empty, singleton and reordered requests."""
    frame, config, state = _fitted(engine, options)
    original = deepcopy(frame)
    saved = pickle.dumps(state)
    assert get_inference_capability(
        "OneHotEncoder", config, state, engine=engine
    ) == ExecutionCapability(engine, "apply", "local", "preserve", "row")
    applier = NodeRegistry.get_applier("OneHotEncoder")()
    target = np.arange(len(frame))
    full, output_target = applier.apply((frame, target), state)
    assert output_target is target
    for positions in ([0], [1], [12], [12, 1, 0, 1], [], list(range(12, -1, -1))):
        _equal(applier.apply(_take(frame, positions), state), _take(full, positions))
    _equal(frame, original)
    assert pickle.dumps(state) == saved


@pytest.mark.parametrize("engine", ["pandas", "polars"])
@pytest.mark.parametrize("policy", ["ignore", "error"])
def test_capped_onehot_preserves_unseen_category_policy(engine, policy):
    """An unknown category must keep the saved ignore/error policy even beside a rare bucket."""
    _, config, state = _fitted(engine, {"max_categories": 3, "handle_unknown": policy})
    sample = pd.DataFrame({"category": ["unseen"], "other": ["u"], "x": [1]})
    if engine == "polars":
        sample = pl.from_pandas(sample)
    assert get_inference_capability("OneHotEncoder", config, state, engine=engine) is not None
    applier = NodeRegistry.get_applier("OneHotEncoder")()
    if policy == "error":
        with pytest.raises(ValueError, match="unknown categories"):
            applier.apply(sample, state)
    else:
        output = applier.apply(sample, state)
        assert (
            sum(
                output[name].to_list()[0]
                for name in state["feature_names"]
                if name.startswith("category_")
            )
            == 0
        )


@pytest.mark.parametrize("engine", ["pandas", "polars"])
@pytest.mark.parametrize("cap", [1, None])
def test_onehot_dropping_every_output_preserves_rows(engine, cap):
    """A zero-feature result must retain input height even without a companion column."""
    frame = pd.DataFrame({"category": ["a"] * 3}, index=[7, 2, 2])
    if engine == "polars":
        frame = pl.from_pandas(frame)
    config = {"columns": ["category"], "max_categories": cap, "drop_first": True}
    state = NodeRegistry.get_calculator("OneHotEncoder")().fit(frame, config)
    assert get_inference_capability("OneHotEncoder", config, state, engine=engine) is not None
    applier = NodeRegistry.get_applier("OneHotEncoder")()
    full = applier.apply(frame, state)
    assert full.shape == (3, 0)
    for positions in ([0], [2, 0], []):
        _equal(applier.apply(_take(frame, positions), state), _take(full, positions))


@pytest.mark.parametrize("engine", ["pandas", "polars"])
def test_onehot_numpy_caps_and_nonfinite_categories_remain_unknown(engine):
    """This extension must retain explicit scalar and learned-category review boundaries."""
    _, config, state = _fitted(engine, {"max_categories": np.int64(3)})
    assert get_inference_capability("OneHotEncoder", config, state, engine=engine) is None
    frame = pd.DataFrame({"category": [1.0, 2.0, np.nan]})
    if engine == "polars":
        frame = pl.from_pandas(frame, nan_to_null=False)
    config = {"columns": ["category"], "max_categories": 3}
    state = NodeRegistry.get_calculator("OneHotEncoder")().fit(frame, config)
    assert get_inference_capability("OneHotEncoder", config, state, engine=engine) is None


@pytest.mark.parametrize(
    "field,value",
    [
        ("_infrequent_indices", [None, None]),
        ("_infrequent_indices", [np.array([0, 0, 2, 3, 4]), None]),
        ("_infrequent_indices", [np.array([-1, 0, 2, 3, 4]), None]),
        ("_infrequent_indices", [np.array([0, 2, 3, 4, 7]), None]),
        ("_infrequent_indices", [np.array([0.0, 2.0, 3.0, 4.0, 6.0]), None]),
        ("_infrequent_indices", [np.array([[0, 2, 3, 4, 6]]), None]),
        ("_default_to_infrequent_mappings", [None, None]),
        ("_default_to_infrequent_mappings", [np.array([0, 1, 2, 3, 4, 5, 6]), None]),
        ("_default_to_infrequent_mappings", [np.array([2, 0, 2, 2, 2, 1, 2], dtype=float), None]),
        ("_default_to_infrequent_mappings", [np.array([2, 0, 2, 2, 2, 1, 2]), np.array([0, 1])]),
        ("_default_to_infrequent_mappings", []),
        ("_infrequent_enabled", False),
        ("_n_features_outs", [7, 2]),
        ("drop_idx_", np.array([0, 0], dtype=object)),
        ("_drop_idx_after_grouping", np.array([1, 0], dtype=object)),
        ("min_frequency", 2),
        ("max_categories", True),
        ("max_categories", 0),
        ("max_categories", 3.0),
    ],
)
def test_capped_onehot_rejects_malformed_fitted_grouping(field, value):
    """Local metadata must not certify inconsistent mappings, indices, widths or unsupported options."""
    _, config, state = _fitted("pandas", {"max_categories": 3, "drop_first": True})
    assert get_inference_capability("OneHotEncoder", config, state, engine="pandas") is not None
    setattr(state["encoder_object"], field, value)
    assert get_inference_capability("OneHotEncoder", config, state, engine="pandas") is None


@pytest.mark.parametrize("change", ["cap", "names", "extra", "columns", "callback", "applier"])
def test_capped_onehot_retains_recipe_and_identity_checks(change):
    """A correct mapping cannot authorize changed recipes, names or executable callbacks."""
    _, config, state = _fitted("pandas", {"max_categories": 3})
    applier = NodeRegistry.get_applier("OneHotEncoder")()
    if change == "cap":
        config["max_categories"] = 4
    elif change == "names":
        state["feature_names"] = ["changed"]
    elif change == "extra":
        state["encoder_object"].unused = True
    elif change == "columns":
        config["columns"] = ["other", "category"]
    elif change == "callback":
        state["encoder_object"].feature_name_combiner = lambda *args: "changed"
    else:
        applier.apply = lambda *args: None
    assert (
        get_inference_capability("OneHotEncoder", config, state, engine="pandas", applier=applier)
        is None
    )


@pytest.mark.parametrize("cap", [3, None])
@pytest.mark.parametrize("field", ["drop_idx_", "_drop_idx_after_grouping"])
@pytest.mark.parametrize("dtype", [float, bool])
def test_onehot_rejects_noninteger_drop_indices(cap, field, dtype):
    """Metadata must abstain before sklearn interprets float or boolean category positions."""
    _, config, state = _fitted("pandas", {"max_categories": cap, "drop_first": True})
    encoder = state["encoder_object"]
    setattr(encoder, field, getattr(encoder, field).astype(dtype))
    assert get_inference_capability("OneHotEncoder", config, state, engine="pandas") is None


@pytest.mark.parametrize("engine", ["pandas", "polars"])
def test_capped_onehot_inspection_is_pure_and_does_not_admit_workers(engine, monkeypatch):
    """Inspecting fitted categories must execute neither learning nor transforms or grant workers."""
    _, config, state = _fitted(engine)
    saved = pickle.dumps(state)

    def forbidden(*args, **kwargs):
        """Fail if a diagnostic starts executing data transformations."""
        raise AssertionError("Unexpected execution")

    monkeypatch.setattr(NodeRegistry.get_calculator("OneHotEncoder"), "fit", forbidden)
    monkeypatch.setattr(NodeRegistry.get_applier("OneHotEncoder"), "apply", forbidden)
    monkeypatch.setattr(type(state["encoder_object"]), "transform", forbidden)
    assert get_inference_capability("OneHotEncoder", config, state, engine=engine) is not None
    for worker, kind in (
        ("pandas", "python_batch"),
        ("polars", "python_batch"),
        ("spark", "native"),
    ):
        with pytest.raises(UnsupportedExecutionError):
            require_capability("OneHotEncoder", "apply", worker, config=config, execution_kind=kind)
    assert pickle.dumps(state) == saved
