"""Calendar and polynomial saved-state context uses existing local apply paths."""

import pickle
from copy import deepcopy

import numpy as np
import pandas as pd
import pytest

pl = pytest.importorskip("polars")

from polars.testing import assert_frame_equal as assert_polars_frame_equal

from skyulf.core.capabilities import UnsupportedExecutionError, require_capability
from skyulf.preprocessing.inference_context import get_inference_capability
from skyulf.registry import NodeRegistry


def _frame(values, engine):
    """Keep native schema and nontrivial pandas labels for inference ordering checks."""
    frame = pd.DataFrame(values)
    frame.index = [9, 2, 9, -1][: len(frame)]
    return pl.from_pandas(frame) if engine == "polars" else frame


def _assert_equal(actual, expected):
    """Keep dtype, values and row/column ordering checks exact."""
    if isinstance(actual, pd.DataFrame):
        pd.testing.assert_frame_equal(actual, expected, check_exact=True)
    else:
        assert_polars_frame_equal(actual, expected, check_exact=True)


def _slice(frame, positions):
    """Select row positions without discarding pandas index labels."""
    return frame.iloc[positions] if isinstance(frame, pd.DataFrame) else frame[positions]


def _poison_fit(monkeypatch, node):
    """Reject any attempt to call the node calculator after capturing its state."""

    def forbidden(*args, **kwargs):
        """Expose accidental inference-time selection or fitting."""
        raise AssertionError("Node calculator must not refit during inference.")

    monkeypatch.setattr(NodeRegistry.get_calculator(node), "fit", forbidden)


def _check_saved_apply(node, state, sample, engine, monkeypatch, *, empty=True):
    """Probe actual apply with the genuine saved state and no permitted calculator fit."""
    _poison_fit(monkeypatch, node)
    before = deepcopy(sample)
    state_bytes = pickle.dumps(state)
    applier = NodeRegistry.get_applier(node)()
    assert applier.validate_inference_state(state) is state
    capability = get_inference_capability(node, {}, state, engine=engine)
    assert capability is not None
    assert (capability.execution_kind, capability.context, capability.row_effect) == (
        "local",
        "row",
        "preserve",
    )
    full = applier.apply(sample, state)
    selections = [[0, 1], [2, 3], [0], [1], [2], [3], [3, 2, 1, 0]]
    for positions in selections + ([[]] if empty else []):
        _assert_equal(applier.apply(_slice(sample, positions), state), _slice(full, positions))
    _assert_equal(sample, before)
    assert pickle.dumps(state) == state_bytes
    return full


@pytest.mark.parametrize("engine", ["pandas", "polars"])
def test_date_context_replays_mixed_formats_offsets_nulls_and_empty(engine, monkeypatch):
    """UTC calendar values must not be inferred from neighboring date formats."""
    train = _frame({"date": ["2020-01-01", "2020-02-01"]}, engine)
    config = {
        "columns": ["date"],
        "features": ["year", "month", "day", "hour"],
        "drop_original": True,
    }
    state = NodeRegistry.get_calculator("DateFeatures")().fit(train, config)
    sample = _frame(
        {"keep": [4, 3, 2, 1], "date": ["2024-03-31T00:30:00+02:00", "1/2/2025", "invalid", None]},
        engine,
    )
    result = _check_saved_apply("DateFeatures", state, sample, engine, monkeypatch)
    assert list(result.columns) == ["keep", "date_year", "date_month", "date_day", "date_hour"]
    expected = {
        "date_year": [2024, 2025, None, None],
        "date_month": [3, 1, None, None],
        "date_day": [30, 2, None, None],
        "date_hour": [22, 0, None, None],
    }
    for name, values in expected.items():
        assert [None if pd.isna(value) else value for value in result[name].to_list()] == values
    assert state["timezone"] == "UTC"


@pytest.mark.parametrize("engine", ["pandas", "polars"])
@pytest.mark.parametrize("alias", ["PolynomialFeatures", "PolynomialFeaturesNode"])
def test_polynomial_aliases_replay_saved_column_order_and_powers(alias, engine, monkeypatch):
    """Unseen values and changed frame order must keep the saved polynomial term ordering."""
    train = _frame({"a": [1.0, 2.0], "b": [3.0, 5.0]}, engine)
    state = NodeRegistry.get_calculator(alias)().fit(train, {"columns": ["b", "a"], "degree": 2})
    sample = _frame(
        {"a": [2.0, 0.0, -3.0, 8.0], "keep": [1, 2, 3, 4], "b": [-1.0, 4.0, 0.0, 2.0]}, engine
    )
    result = _check_saved_apply(alias, state, sample, engine, monkeypatch, empty=False)
    assert list(result.columns) == ["a", "keep", "b", "poly_b_pow_2", "poly_b_a", "poly_a_pow_2"]
    assert result["poly_b_pow_2"].to_list() == [1.0, 16.0, 0.0, 4.0]
    assert result["poly_b_a"].to_list() == [-2.0, 0.0, 0.0, 16.0]
    assert result["poly_a_pow_2"].to_list() == [4.0, 0.0, 9.0, 64.0]


@pytest.mark.parametrize("node", ["DateFeatures", "PolynomialFeatures", "PolynomialFeaturesNode"])
def test_feature_metadata_is_pure_and_grants_no_worker_execution(node, monkeypatch):
    """Local context inspection neither runs transformations nor enables Spark or workers."""
    frame = pd.DataFrame({"x": ["2024-01-01"] if node == "DateFeatures" else [2.0]})
    state = NodeRegistry.get_calculator(node)().fit(frame, {"columns": ["x"]})
    _poison_fit(monkeypatch, node)
    applier = NodeRegistry.get_applier(node)

    def forbidden(*args, **kwargs):
        """Make hidden metadata execution fail visibly."""
        raise AssertionError("Inspection must not apply.")

    monkeypatch.setattr(applier, "apply", forbidden)
    saved = pickle.dumps(state)
    assert get_inference_capability(node, {}, state, engine="pandas") is not None
    assert get_inference_capability(node, {}, state, engine="spark") is None
    with pytest.raises(UnsupportedExecutionError):
        require_capability(node, "apply", "pandas", config={}, execution_kind="python_batch")
    assert pickle.dumps(state) == saved


@pytest.mark.parametrize("engine", ["pandas", "polars"])
@pytest.mark.parametrize("source", ["native", "epoch"])
def test_date_native_timezone_and_explicit_epoch_keep_calendar_values(engine, source, monkeypatch):
    """Saved timestamp units and native offsets remain stable across row arrangements."""
    values = pd.to_datetime(
        pd.Series(
            [
                "2024-01-01T00:30:00+02:00",
                "2024-06-01T00:30:00+02:00",
                None,
                "2025-01-01T00:30:00+02:00",
            ]
        )
    )
    config = {"columns": ["date"], "features": ["year", "month", "day", "hour"]}
    if source == "epoch":
        values = [1704061800, 1717194600, None, 1735684200]
        config["epoch_unit"] = "s"
    sample = _frame({"date": values}, engine)
    state = NodeRegistry.get_calculator("DateFeatures")().fit(sample, config)
    result = _check_saved_apply("DateFeatures", state, sample, engine, monkeypatch)
    assert [None if pd.isna(value) else value for value in result["date_year"].to_list()] == [
        2023,
        2024,
        None,
        2024,
    ]
    assert [None if pd.isna(value) else value for value in result["date_hour"].to_list()] == [
        22,
        22,
        None,
        22,
    ]


@pytest.mark.parametrize("engine", ["pandas", "polars"])
def test_legacy_date_state_keeps_values_without_a_new_context_promise(engine):
    """Old local-calendar artifacts stay usable without claiming consistent mixed-offset parsing."""
    state = {"type": "date_features", "columns": ["date"], "features": ["year", "hour"]}
    saved = deepcopy(state)
    applier = NodeRegistry.get_applier("DateFeatures")()
    assert applier.validate_inference_state(state) is state
    assert get_inference_capability("DateFeatures", {}, state, engine=engine) is None
    result = applier.apply(_frame({"date": ["2024-01-01T00:30:00+02:00"]}, engine), state)
    assert result["date_year"].to_list() == ([2024] if engine == "pandas" else [2023])
    assert state == saved


@pytest.mark.parametrize("engine", ["pandas", "polars"])
def test_date_generated_collision_preserves_existing_overwrite_behavior(engine):
    """A context promise does not add collision protection that calendar apply never provided."""
    sample = _frame({"date": ["2024-01-01"], "date_year": [7]}, engine)
    state = NodeRegistry.get_calculator("DateFeatures")().fit(
        sample, {"columns": ["date"], "features": ["year"]}
    )
    assert get_inference_capability("DateFeatures", {}, state, engine=engine) is not None
    result = NodeRegistry.get_applier("DateFeatures")().apply(sample, state)
    assert result["date_year"].to_list() == [2024]
    assert sample["date_year"].to_list() == [7]


@pytest.mark.parametrize("engine", ["pandas", "polars"])
def test_polynomial_numpy_configuration_keeps_interactions_bias_and_input_terms(
    engine, monkeypatch
):
    """Genuine NumPy settings and a degree interval must retain the fitted emitted terms."""
    sample = _frame({"a": [2.0, 0.0, -3.0, 8.0], "b": [-1.0, 4.0, 0.0, 2.0]}, engine)
    config = {
        "columns": ["b", "a"],
        "degree": np.array([1, 2]),
        "interaction_only": np.bool_(True),
        "include_bias": np.bool_(True),
        "include_input_features": np.bool_(True),
        "output_prefix": "terms",
    }
    state = NodeRegistry.get_calculator("PolynomialFeatures")().fit(sample, config)
    result = _check_saved_apply(
        "PolynomialFeatures", state, sample, engine, monkeypatch, empty=False
    )
    assert list(result.columns) == ["a", "b", "terms_1", "terms_b", "terms_a", "terms_b_a"]
    assert result["terms_1"].to_list() == [1.0] * 4
    assert result["terms_b_a"].to_list() == [-2.0, 0.0, 0.0, 16.0]


@pytest.mark.parametrize("engine", ["pandas", "polars"])
def test_polynomial_empty_null_and_new_name_collision_keep_apply_errors(engine, monkeypatch):
    """Context inspection cannot relax sklearn input validation or generated-name protection."""
    state = NodeRegistry.get_calculator("PolynomialFeatures")().fit(
        _frame({"x": [1.0, 2.0]}, engine), {"columns": ["x"]}
    )
    _poison_fit(monkeypatch, "PolynomialFeatures")
    applier = NodeRegistry.get_applier("PolynomialFeatures")()
    assert get_inference_capability("PolynomialFeatures", {}, state, engine=engine) is not None
    bad_frames = [
        _slice(_frame({"x": [1.0]}, engine), []),
        _frame({"x": [np.nan]}, engine),
        _frame({"x": [1.0], "poly_x_pow_2": [7.0]}, engine),
    ]
    for sample in bad_frames:
        before = deepcopy(sample)
        with pytest.raises(ValueError):
            applier.apply(sample, state)
        _assert_equal(sample, before)
    assert state["columns"] == ["x"]


def test_polynomial_name_metadata_can_repeat_when_actual_outputs_are_unambiguous():
    """Unused raw sklearn names can legitimately overlap between input and interaction terms."""
    sample = pd.DataFrame({"a b": [1.0, 2.0], "a": [2.0, 3.0], "b": [3.0, 4.0]})
    state = NodeRegistry.get_calculator("PolynomialFeatures")().fit(
        sample, {"columns": list(sample.columns)}
    )
    applier = NodeRegistry.get_applier("PolynomialFeatures")()
    assert state["feature_names"].count("a b") == 2
    assert applier.validate_inference_state(state) is state
    result = applier.apply(sample, state)
    assert result["poly_a_b"].to_list() == [6.0, 12.0]


@pytest.mark.parametrize("node", ["DateFeatures", "PolynomialFeatures"])
def test_feature_noop_state_and_absent_columns_keep_input(node, monkeypatch):
    """Empty saved selections remain real no-ops without creating synthetic fitted objects."""
    sample = pd.DataFrame({"keep": [1, 2, 3, 4]})
    state = NodeRegistry.get_calculator(node)().fit(sample, {})
    result = _check_saved_apply(node, state, sample, "pandas", monkeypatch)
    _assert_equal(result, sample)
    assert not state.get("columns")


@pytest.mark.parametrize("engine", ["pandas", "polars"])
def test_polynomial_missing_feature_rebuilds_only_present_terms(engine):
    """Saved raw name metadata must not be mistaken for apply's filtered output schema."""
    train = _frame({"a": [1.0, 2.0], "b": [3.0, 4.0]}, engine)
    state = NodeRegistry.get_calculator("PolynomialFeatures")().fit(train, {"columns": ["b", "a"]})
    assert get_inference_capability("PolynomialFeatures", {}, state, engine=engine) is not None
    result = NodeRegistry.get_applier("PolynomialFeatures")().apply(
        _frame({"a": [3.0, -2.0]}, engine), state
    )
    assert list(result.columns) == ["a", "poly_a_pow_2"]
    assert result["poly_a_pow_2"].to_list() == [9.0, 4.0]


def test_polynomial_metadata_never_fits_a_hidden_estimator(monkeypatch):
    """Inspecting config-only state must not fit even sklearn's combinatorial helper."""
    from sklearn.preprocessing import PolynomialFeatures

    state = NodeRegistry.get_calculator("PolynomialFeatures")().fit(
        pd.DataFrame({"x": [1.0]}), {"columns": ["x"], "degree": np.int64(0), "include_bias": True}
    )

    def forbidden(*args, **kwargs):
        """Expose estimator fitting during metadata queries."""
        raise AssertionError("Inspection must not fit sklearn.")

    monkeypatch.setattr(PolynomialFeatures, "fit", forbidden)
    assert get_inference_capability("PolynomialFeatures", {}, state, engine="pandas") is not None
    assert state["feature_names"] == ["1"]


@pytest.mark.parametrize(
    ("node", "field", "value"),
    [
        ("DateFeatures", "columns", [123]),
        ("DateFeatures", "features", ["unsupported"]),
        ("DateFeatures", "drop_original", 1),
        ("DateFeatures", "timezone", []),
        ("DateFeatures", "epoch_unit", "minutes"),
        ("PolynomialFeatures", "columns", "x"),
        ("PolynomialFeatures", "degree", True),
        ("PolynomialFeatures", "degree", -1),
        ("PolynomialFeatures", "degree", [2, 1]),
        ("PolynomialFeatures", "degree", np.array([[1, 2]])),
        ("PolynomialFeatures", "degree", [1.0, 2.0]),
        ("PolynomialFeatures", "degree", 0),
        ("PolynomialFeatures", "interaction_only", 1),
        ("PolynomialFeatures", "output_prefix", {}),
        ("PolynomialFeatures", "feature_names", [None]),
    ],
)
def test_feature_context_rejects_malformed_saved_fields_without_mutation(node, field, value):
    """Invalid saved configuration must fail before it acquires a local context promise."""
    sample = (
        pd.DataFrame({"date": ["2024-01-01"]})
        if node == "DateFeatures"
        else pd.DataFrame({"x": [1.0]})
    )
    state = NodeRegistry.get_calculator(node)().fit(sample, {"columns": list(sample.columns)})
    state[field] = value
    before = pickle.dumps(state)
    with pytest.raises(ValueError):
        get_inference_capability(node, {}, state, engine="pandas")
    assert pickle.dumps(state) == before


@pytest.mark.parametrize("engine", ["pandas", "polars"])
@pytest.mark.parametrize("mode", ["tuple_columns", "numpy_feature", "duplicate_features"])
def test_date_real_fit_retains_native_sequences_and_feature_values(engine, mode, monkeypatch):
    """Local inspection must retain accepted fit containers without hiding native apply errors."""
    columns = ("date",) if mode == "tuple_columns" else ["date"]
    features = [np.str_("year")] if mode == "numpy_feature" else ["year"]
    if mode == "duplicate_features":
        features = ["year", "year"]
    sample = _frame({"date": ["2024-01-01", "2025-02-03", "invalid", None]}, engine)
    state = NodeRegistry.get_calculator("DateFeatures")().fit(
        sample, {"columns": columns, "features": features}
    )
    applier = NodeRegistry.get_applier("DateFeatures")()
    if engine == "polars" and mode == "duplicate_features":
        assert applier.validate_inference_state(state) is state
        _poison_fit(monkeypatch, "DateFeatures")
        with pytest.raises(pl.exceptions.ComputeError, match="duplicate"):
            applier.apply(sample, state)
    else:
        direct = applier.apply(sample, state)
        result = _check_saved_apply("DateFeatures", state, sample, engine, monkeypatch)
        _assert_equal(result, direct)
    assert type(state["columns"]) is type(columns)
    assert type(state["features"][0]) is type(features[0])


@pytest.mark.parametrize("engine", ["pandas", "polars"])
@pytest.mark.parametrize("field", ["output_prefix", "columns"])
def test_polynomial_real_fit_keeps_numpy_string_configuration(engine, field, monkeypatch):
    """String-compatible NumPy values accepted by fit and apply must remain inspectable."""
    sample = _frame({"x": [1.0, -2.0, 0.0, 3.0]}, engine)
    config = {
        "columns": [np.str_("x")] if field == "columns" else ["x"],
        "output_prefix": np.str_("terms"),
    }
    state = NodeRegistry.get_calculator("PolynomialFeatures")().fit(sample, config)
    applier = NodeRegistry.get_applier("PolynomialFeatures")()
    direct = applier.apply(sample, state)
    result = _check_saved_apply(
        "PolynomialFeatures", state, sample, engine, monkeypatch, empty=False
    )
    _assert_equal(result, direct)
    assert result["terms_x_pow_2"].to_list() == [1.0, 4.0, 0.0, 9.0]
