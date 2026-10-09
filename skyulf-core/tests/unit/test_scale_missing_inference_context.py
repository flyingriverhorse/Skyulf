"""Saved local scaler and missing-value states describe apply without refitting."""

import copy
import pickle
from decimal import Decimal
from itertools import product

import numpy as np
import pandas as pd
import pytest

pl = pytest.importorskip("polars")

from polars.testing import assert_frame_equal as assert_polars_frame_equal

from skyulf.core.capabilities import UnsupportedExecutionError, require_capability
from skyulf.data.dataset import SplitDataset
from skyulf.inference.local_pipeline import load_local_pipeline, save_local_pipeline
from skyulf.inference.preprocessing_probe import probe_fitted_preprocessing
from skyulf.pipeline import SkyulfPipeline
from skyulf.preprocessing.inference_context import get_inference_capability
from skyulf.registry import NodeRegistry

NODES = ("MaxAbsScaler", "RobustScaler", "DropMissingColumns", "MissingIndicator")
ENGINES = ("pandas", "polars")


def _frame(engine, values, *, index=None):
    """Retain a nontrivial pandas index while providing equal local-engine values."""
    frame = pd.DataFrame(values, index=index)
    return pl.from_pandas(frame) if engine == "polars" else frame


def _assert_frame(actual, expected):
    """Check values, dtype, column order and index on the active engine."""
    if isinstance(actual, pd.DataFrame):
        pd.testing.assert_frame_equal(actual, expected, check_exact=True)
    else:
        assert_polars_frame_equal(actual, expected, check_exact=True)


def _slice(frame, positions):
    """Select the same ordered rows without resetting pandas labels."""
    return frame.iloc[positions] if isinstance(frame, pd.DataFrame) else frame[positions]


def _poison_fit(monkeypatch, node):
    """Make accidental fitting fail once the genuine artifact has been captured."""

    def fail(*args, **kwargs):
        """Reject any attempt to relearn statistics from inference rows."""
        raise AssertionError("Inference must not fit.")

    monkeypatch.setattr(NodeRegistry.get_calculator(node), "fit", fail)


def _assert_saved_apply(node, state, request, expected, engine):
    """Verify every row arrangement against explicit saved-parameter expectations."""
    before = request.copy(deep=True) if engine == "pandas" else request.clone()
    serialized = pickle.dumps(state)
    applier = NodeRegistry.get_applier(node)()
    assert applier.validate_inference_state(state) is state
    capability = get_inference_capability(node, {}, state, engine=engine)
    assert capability is not None
    assert (capability.context, capability.row_effect, capability.execution_kind) == (
        "row",
        "preserve",
        "local",
    )
    for positions in ([0, 1, 2, 3], [0, 1], [2, 3], [0], [1], [2], [3], [3, 2, 1, 0], []):
        output = applier.apply(_slice(request, positions), state)
        _assert_frame(output, _slice(expected, positions))
    _assert_frame(request, before)
    assert pickle.dumps(state) == serialized


@pytest.mark.parametrize("engine", ENGINES)
def test_maxabs_saved_statistics_cover_null_zero_and_constant_columns(engine, monkeypatch):
    """All-null fitted statistics and zero scales remain valid without batch relearning."""
    train = _frame(
        engine,
        {
            "x": [-4.0, -2.0, 0.0, 2.0, 4.0],
            "constant": [5.0] * 5,
            "zero": [0.0] * 5,
            "null": [np.nan] * 5,
        },
    )
    state = NodeRegistry.get_calculator("MaxAbsScaler")().fit(
        train, {"columns": list(train.columns)}
    )
    np.testing.assert_equal(state["scale"], [4.0, 5.0, 1.0, np.nan])
    np.testing.assert_equal(state["max_abs"], [4.0, 5.0, 0.0, np.nan])
    _poison_fit(monkeypatch, "MaxAbsScaler")
    request = _frame(
        engine,
        {
            "constant": [5.0, 10.0, 0.0, np.nan],
            "keep": ["b", "a", "c", "d"],
            "x": [8.0, -4.0, np.nan, 0.0],
            "zero": [0.0, 2.0, -2.0, 0.0],
            "null": [1.0, np.nan, 2.0, 3.0],
        },
        index=[8, 2, 8, -1],
    )
    expected = _frame(
        engine,
        {
            "constant": [1.0, 2.0, 0.0, np.nan],
            "keep": ["b", "a", "c", "d"],
            "x": [2.0, -1.0, np.nan, 0.0],
            "zero": [0.0, 2.0, -2.0, 0.0],
            "null": [np.nan] * 4,
        },
        index=[8, 2, 8, -1],
    )
    if engine == "polars":
        expected = expected.with_columns(pl.Series("null", [np.nan, None, np.nan, np.nan]))
    _assert_saved_apply("MaxAbsScaler", state, request, expected, engine)
    assert state["columns"] == ["x", "constant", "zero", "null"]


@pytest.mark.parametrize("engine", ENGINES)
@pytest.mark.parametrize(("centering", "scaling"), list(product((False, True), repeat=2)))
@pytest.mark.parametrize(
    ("quantiles", "spread"), [((0.0, 100.0), 8.0), ((25.0, 75.0), 4.0), ((50.0, 50.0), 1.0)]
)
def test_robust_saved_flags_quantiles_and_nulls(
    engine, centering, scaling, quantiles, spread, monkeypatch
):
    """Flag combinations and equal quantiles apply the learned arrays on every row shape."""
    train = _frame(
        engine,
        {
            "x": [-4.0, -2.0, 0.0, 2.0, 4.0],
            "constant": [5.0] * 5,
            "zero": [0.0] * 5,
            "null": [np.nan] * 5,
        },
    )
    config = {
        "columns": list(train.columns),
        "with_centering": centering,
        "with_scaling": scaling,
        "quantile_range": quantiles,
    }
    state = NodeRegistry.get_calculator("RobustScaler")().fit(train, config)
    np.testing.assert_equal(state["center"], [0.0, 5.0, 0.0, np.nan] if centering else None)
    np.testing.assert_equal(state["scale"], [spread, 1.0, 1.0, np.nan] if scaling else None)
    _poison_fit(monkeypatch, "RobustScaler")
    request = _frame(
        engine,
        {
            "constant": [5.0, 7.0, 0.0, np.nan],
            "keep": ["b", "a", "c", "d"],
            "x": [8.0, -4.0, np.nan, 0.0],
            "zero": [0.0, 2.0, -2.0, 0.0],
            "null": [1.0, np.nan, 2.0, 3.0],
        },
        index=[8, 2, 8, -1],
    )
    expected = _frame(
        engine,
        {
            "constant": [0.0, 2.0, -5.0, np.nan] if centering else [5.0, 7.0, 0.0, np.nan],
            "keep": ["b", "a", "c", "d"],
            "x": [8.0 / spread, -4.0 / spread, np.nan, 0.0]
            if scaling
            else [8.0, -4.0, np.nan, 0.0],
            "zero": [0.0, 2.0, -2.0, 0.0],
            "null": [np.nan] * 4 if centering or scaling else [1.0, np.nan, 2.0, 3.0],
        },
        index=[8, 2, 8, -1],
    )
    if engine == "polars" and (centering or scaling):
        expected = expected.with_columns(pl.Series("null", [np.nan, None, np.nan, np.nan]))
    _assert_saved_apply("RobustScaler", state, request, expected, engine)
    assert state["quantile_range"] == quantiles


@pytest.mark.parametrize("engine", ENGINES)
def test_missing_column_drop_keeps_the_training_selection(engine, monkeypatch):
    """Request missingness cannot change the learned drop set or target protection."""
    train = _frame(
        engine,
        {
            "drop": [np.nan, np.nan, 1.0, 2.0],
            "keep": [1.0] * 4,
            "explicit": [3] * 4,
            "target": [np.nan] * 4,
        },
    )
    config = {"missing_threshold": 50, "columns": ["explicit"], "target_column": "target"}
    state = NodeRegistry.get_calculator("DropMissingColumns")().fit(train, config)
    assert set(state["columns_to_drop"]) == {"drop", "explicit"}
    _poison_fit(monkeypatch, "DropMissingColumns")
    request = _frame(
        engine,
        {"keep": [np.nan] * 4, "drop": [1.0] * 4, "target": [1, 2, 3, 4], "new": ["a"] * 4},
        index=[8, 2, 8, -1],
    )
    expected = _frame(
        engine,
        {"keep": [np.nan] * 4, "target": [1, 2, 3, 4], "new": ["a"] * 4},
        index=[8, 2, 8, -1],
    )
    _assert_saved_apply("DropMissingColumns", state, request, expected, engine)
    assert state["threshold"] == 50


@pytest.mark.parametrize("engine", ENGINES)
@pytest.mark.parametrize("suffix", ["_missing", "__flag"])
def test_missing_indicator_keeps_learned_columns_and_flag_dtype(engine, suffix, monkeypatch):
    """New missing values never create unlearned flag columns or reorder existing inputs."""
    train = _frame(engine, {"x": [1.0, np.nan], "absent": [np.nan, 1.0], "keep": [1, 2]})
    state = NodeRegistry.get_calculator("MissingIndicator")().fit(train, {"flag_suffix": suffix})
    assert state["columns"] == ["x", "absent"]
    _poison_fit(monkeypatch, "MissingIndicator")
    request = _frame(
        engine, {"keep": [np.nan] * 4, "x": [1.0, np.nan, np.nan, 2.0]}, index=[8, 2, 8, -1]
    )
    expected = _frame(
        engine,
        {"keep": [np.nan] * 4, "x": [1.0, np.nan, np.nan, 2.0], f"x{suffix}": [0, 1, 1, 0]},
        index=[8, 2, 8, -1],
    )
    _assert_saved_apply("MissingIndicator", state, request, expected, engine)
    assert state["flag_suffix"] == suffix


@pytest.mark.parametrize("engine", ENGINES)
@pytest.mark.parametrize("node", NODES)
def test_genuine_noop_artifacts_remain_local(node, engine, monkeypatch):
    """Empty learned selections have a context promise without fabricating statistics."""
    request = _frame(engine, {"x": [1, 2, 3, 4]}, index=[8, 2, 8, -1])
    config = {"columns": []} if node.endswith("Scaler") else {}
    state = NodeRegistry.get_calculator(node)().fit(request, config)
    _poison_fit(monkeypatch, node)
    _assert_saved_apply(node, state, request, request, engine)
    assert not state if node.endswith("Scaler") else state["type"]


@pytest.mark.parametrize("engine", ENGINES)
@pytest.mark.parametrize("node", NODES)
def test_context_query_is_pure_and_does_not_admit_workers(node, engine, monkeypatch):
    """Metadata inspection executes neither fit nor apply and cannot grant worker execution."""
    state = NodeRegistry.get_calculator(node)().fit(_frame(engine, {"x": [1.0, 2.0]}), {})
    _poison_fit(monkeypatch, node)
    applier = NodeRegistry.get_applier(node)

    def fail(*args, **kwargs):
        """Make accidental metadata execution visible."""
        raise AssertionError("Metadata must not apply.")

    monkeypatch.setattr(applier, "apply", fail)
    before = pickle.dumps(state)
    capability = get_inference_capability(node, {}, state, engine=engine)
    assert capability is not None and capability.context == "row"
    assert get_inference_capability(node, {}, state, engine="spark") is None
    with pytest.raises(UnsupportedExecutionError):
        require_capability(node, "apply", engine, config={}, execution_kind="python_batch")
    assert pickle.dumps(state) == before


@pytest.mark.parametrize("node", NODES)
@pytest.mark.parametrize("corruption", ["missing", "extra", "type", "column_type", "duplicates"])
def test_malformed_saved_shape_is_rejected(node, corruption):
    """A damaged artifact must fail validation before receiving a context promise."""
    state = NodeRegistry.get_calculator(node)().fit(pd.DataFrame({"x": [1.0, np.nan, 3.0]}), {})
    key = "columns_to_drop" if node == "DropMissingColumns" else "columns"
    if corruption == "missing":
        state.pop(key)
    elif corruption == "extra":
        state["unknown_behavior"] = True
    elif corruption == "type":
        state["type"] = "different_node"
    else:
        state[key] = ["x", "x"] if corruption == "duplicates" else "x"
    with pytest.raises(ValueError):
        get_inference_capability(node, {}, state, engine="pandas")
    assert "type" in state


@pytest.mark.parametrize(
    ("node", "field"),
    [
        ("MaxAbsScaler", "scale"),
        ("MaxAbsScaler", "max_abs"),
        ("RobustScaler", "center"),
        ("RobustScaler", "scale"),
    ],
)
@pytest.mark.parametrize("value", [None, [], [1.0, 2.0], ["1"], [True], [np.inf], (1.0,)])
def test_malformed_saved_numeric_arrays_are_rejected(node, field, value):
    """Array shape and numeric checks prevent late indexing errors and silent coercion."""
    state = NodeRegistry.get_calculator(node)().fit(pd.DataFrame({"x": [1.0, 2.0]}), {})
    state[field] = value
    before = copy.deepcopy(state)
    with pytest.raises(ValueError):
        get_inference_capability(node, {}, state, engine="pandas")
    assert state == before


@pytest.mark.parametrize(
    ("field", "value"),
    [
        ("with_centering", 1),
        ("with_scaling", "yes"),
        ("with_centering", False),
        ("with_scaling", False),
        ("quantile_range", [-1, 75]),
        ("quantile_range", [75, 25]),
        ("quantile_range", [0, 101]),
        ("quantile_range", [0]),
        ("quantile_range", [0, np.inf]),
        ("quantile_range", [False, 75]),
    ],
)
def test_robust_rejects_inconsistent_flags_and_quantiles(field, value):
    """Flags must agree with required saved arrays and quantiles must describe a valid fit."""
    state = NodeRegistry.get_calculator("RobustScaler")().fit(pd.DataFrame({"x": [1.0, 2.0]}), {})
    state[field] = value
    with pytest.raises(ValueError):
        get_inference_capability("RobustScaler", {}, state, engine="pandas")
    assert state[field] == value


@pytest.mark.parametrize("node", ["MaxAbsScaler", "RobustScaler"])
def test_scaler_nullable_and_decimal_columns_keep_existing_conversion(node, monkeypatch):
    """Saved local hooks must accept fitted Decimal/nullable inputs without changing conversion."""
    train = pd.DataFrame(
        {"x": [Decimal("-2"), Decimal("0"), Decimal("2")], "n": pd.array([0, 2, 4], dtype="Int64")}
    )
    state = NodeRegistry.get_calculator(node)().fit(train, {})
    _poison_fit(monkeypatch, node)
    request = pd.DataFrame(
        {
            "x": [Decimal("4"), None, Decimal("-2"), Decimal("0")],
            "n": pd.array([8, None, 4, 0], dtype="Int64"),
        },
        index=[8, 2, 8, -1],
    )
    expected = pd.DataFrame(
        {
            "x": [2.0, np.nan, -1.0, 0.0],
            "n": [2.0, np.nan, 1.0, 0.0] if node == "MaxAbsScaler" else [3.0, np.nan, 1.0, -1.0],
        },
        index=request.index,
    )
    _assert_saved_apply(node, state, request, expected, "pandas")
    assert expected.dtypes.tolist() == [np.dtype("float64"), np.dtype("float64")]


@pytest.mark.parametrize(
    ("node", "field"),
    [("MaxAbsScaler", "scale"), ("MaxAbsScaler", "max_abs"), ("RobustScaler", "scale")],
)
def test_scaler_rejects_negative_magnitudes(node, field):
    """Negative magnitudes cannot represent the statistics these calculators learn."""
    state = NodeRegistry.get_calculator(node)().fit(pd.DataFrame({"x": [1.0, 2.0]}), {})
    state[field] = [-1.0]
    with pytest.raises(ValueError):
        get_inference_capability(node, {}, state, engine="pandas")
    assert state[field] == [-1.0]


@pytest.mark.parametrize("quantiles", [{0: "low", 100: "high"}, iter([0, 100])])
def test_robust_rejects_non_array_quantile_state_without_consuming_it(quantiles):
    """Validation must not execute or consume iterators embedded in a damaged artifact."""
    state = NodeRegistry.get_calculator("RobustScaler")().fit(pd.DataFrame({"x": [1.0, 2.0]}), {})
    state["quantile_range"] = quantiles
    with pytest.raises(ValueError):
        get_inference_capability("RobustScaler", {}, state, engine="pandas")
    assert list(quantiles) == [0, 100]


@pytest.mark.parametrize("suffix", [None, "", 1, {"suffix": "bad"}])
def test_indicator_rejects_suffix_outside_the_saved_string_contract(suffix):
    """Malformed or arbitrary suffix objects cannot gain an inspected context promise."""
    state = {"type": "missing_indicator", "columns": ["x"], "flag_suffix": suffix}
    with pytest.raises(ValueError, match="suffix"):
        get_inference_capability("MissingIndicator", {}, state, engine="pandas")
    assert state["flag_suffix"] is suffix


@pytest.mark.parametrize("engine", ENGINES)
def test_robust_disabled_flags_retain_existing_engine_dtype(engine, monkeypatch):
    """Local inspection must preserve the established no-arithmetic dtype behavior."""
    request = _frame(engine, {"x": [1, 2, 3, 4]}, index=[8, 2, 8, -1])
    state = NodeRegistry.get_calculator("RobustScaler")().fit(
        request, {"columns": ["x"], "with_centering": False, "with_scaling": False}
    )
    _poison_fit(monkeypatch, "RobustScaler")
    expected = request.astype("float64") if engine == "pandas" else request
    _assert_saved_apply("RobustScaler", state, request, expected, engine)
    assert state["center"] is None and state["scale"] is None


@pytest.mark.parametrize("engine", ENGINES)
@pytest.mark.parametrize("node", ["MaxAbsScaler", "RobustScaler"])
def test_scaler_uses_original_statistic_position_for_partial_columns(node, engine, monkeypatch):
    """A missing first feature must not shift the saved statistic for a surviving feature."""
    train = _frame(engine, {"absent": [-4.0, 0.0, 4.0], "x": [-2.0, 0.0, 2.0]})
    state = NodeRegistry.get_calculator(node)().fit(train, {"columns": ["absent", "x"]})
    _poison_fit(monkeypatch, node)
    request = _frame(
        engine, {"new": [1, 2, 3, 4], "x": [4.0, -2.0, np.nan, 0.0]}, index=[8, 2, 8, -1]
    )
    expected = _frame(
        engine, {"new": [1, 2, 3, 4], "x": [2.0, -1.0, np.nan, 0.0]}, index=[8, 2, 8, -1]
    )
    _assert_saved_apply(node, state, request, expected, engine)
    assert state["columns"] == ["absent", "x"]


@pytest.mark.parametrize("threshold", [None, "invalid", "50", 0, -1, {"ignored": True}])
def test_drop_threshold_remains_uninterpreted_after_fit(threshold):
    """Reporting-only fit metadata must not affect saved column-drop context validation."""
    request = pd.DataFrame({"x": [1.0, 2.0]})
    state = NodeRegistry.get_calculator("DropMissingColumns")().fit(
        request, {"missing_threshold": threshold}
    )
    applier = NodeRegistry.get_applier("DropMissingColumns")()
    assert applier.validate_inference_state(state) is state
    _assert_frame(applier.apply(request, state), request)
    assert state["threshold"] is threshold


def test_indicator_polars_distinguishes_nan_and_null_but_flags_both(monkeypatch):
    """Float NaN and null both produce flags without coercing original input values."""
    request = pl.DataFrame({"x": [1.0, np.nan, None, 2.0]})
    state = NodeRegistry.get_calculator("MissingIndicator")().fit(request, {})
    _poison_fit(monkeypatch, "MissingIndicator")
    expected = pl.DataFrame({"x": [1.0, np.nan, None, 2.0], "x_missing": [0, 1, 1, 0]})
    _assert_saved_apply("MissingIndicator", state, request, expected, "polars")
    assert state["columns"] == ["x"]


@pytest.mark.parametrize("flag", [np.bool_(True), np.bool_(False)])
def test_robust_accepts_numpy_boolean_flags_saved_by_real_fit(flag):
    """NumPy boolean configuration accepted during fitting must remain inspectable."""
    request = pd.DataFrame({"x": [1.0, 2.0, 3.0]})
    state = NodeRegistry.get_calculator("RobustScaler")().fit(request, {"with_centering": flag})
    applier = NodeRegistry.get_applier("RobustScaler")()
    assert applier.validate_inference_state(state) is state
    expected = pd.DataFrame({"x": [-1.0, 0.0, 1.0] if flag else [1.0, 2.0, 3.0]})
    _assert_frame(applier.apply(request, state), expected)
    assert state["with_centering"] is flag


@pytest.mark.parametrize("node,context", [("MaxAbsScaler", "row"), ("StandardScaler", "row")])
def test_polars_nonbinary_scale_replays_exactly_across_saved_requests(
    tmp_path, monkeypatch, node, context
):
    """Saved native scaling must use identical arithmetic for batches and singleton requests."""
    train = pl.DataFrame({"x": [-5.0, 5.0], "target": [-10.0, 10.0]})
    pipeline = SkyulfPipeline(
        {
            "preprocessing": [{"name": "scale", "transformer": node, "params": {"columns": ["x"]}}],
            "modeling": {"type": "linear_regression"},
        }
    )
    pipeline.fit(SplitDataset(train=train, test=train.head(0)), target_column="target")
    save_local_pipeline(pipeline, tmp_path / "scale")
    artifact = load_local_pipeline(tmp_path / "scale")
    _poison_fit(monkeypatch, node)
    sample = pl.DataFrame({"x": [None, 2.0, -10.0, 200.0, 6.0]})
    state = artifact.pipeline.feature_engineer.fitted_steps[0]["artifact"]
    assert state["scale"] == [5.0]
    applier = NodeRegistry.get_applier(node)()
    full = applier.apply(sample, state)
    singleton = pl.concat([applier.apply(sample.slice(i, 1), state) for i in range(len(sample))])
    assert_polars_frame_equal(full, singleton, check_exact=True)
    result = probe_fitted_preprocessing(artifact, sample, chunk_sizes=(2,))
    step = result["steps"][0]
    assert step["context"] == context and step["state_validation"] == "node_owned"
    check = next(item for item in step["checks"] if item["name"] == "chunks:1")
    assert check["status"] == "passed"
    assert step["status"] == "passed", step
    assert result["admission"] == "diagnostic_only"


@pytest.mark.parametrize(
    "node,options",
    [
        ("MaxAbsScaler", {}),
        ("RobustScaler", {"with_centering": False}),
    ],
)
@pytest.mark.parametrize("dtype", [pl.Float32, pl.Float64, pl.Int64, pl.UInt64])
def test_polars_scalers_preserve_bulk_values_and_exact_partitions(
    node, options, dtype, monkeypatch
):
    """Native Series arithmetic must fix partition rounding without changing bulk schemas or masks."""
    train = pl.DataFrame({"x": [-5.0, 5.0]})
    state = NodeRegistry.get_calculator(node)().fit(train, {"columns": ["x"], **options})
    assert state["scale"] == [5.0]
    values: list[float | None] = [None, 2, 200, 6, 0]
    if dtype.is_float():
        values += [-0.0, float("nan"), float("inf"), -float("inf")]
    sample = pl.DataFrame({"x": pl.Series(values, dtype=dtype), "keep": range(len(values))})
    before = sample.clone()
    serialized = pickle.dumps(state)
    _poison_fit(monkeypatch, node)
    applier = NodeRegistry.get_applier(node)()
    column = pl.col("x").cast(pl.Float64) if node == "RobustScaler" else pl.col("x")
    expected = sample.with_columns((column / 5.0).alias("x"))
    actual = applier.apply(sample, state)
    assert_polars_frame_equal(actual, expected, check_exact=True)
    for size in (1, 2, 3):
        for start in range(0, len(sample), size):
            chunk = applier.apply(sample.slice(start, size), state)
            assert_polars_frame_equal(chunk, expected.slice(start, size), check_exact=True)
    assert_polars_frame_equal(applier.apply(sample.head(0), state), expected.head(0))
    assert_polars_frame_equal(applier.apply(sample.reverse(), state), expected.reverse())
    if dtype.is_float():
        assert np.signbit(actual["x"][5]) and np.signbit(expected["x"][5])
    assert_polars_frame_equal(sample, before)
    assert pickle.dumps(state) == serialized


@pytest.mark.parametrize("scale", [np.float64(5), np.int64(5), np.float32(5)])
def test_maxabs_numpy_statistics_keep_native_promotion(scale):
    """NumPy scalar dtype metadata must survive instead of narrowing Float32 requests."""
    sample = pl.DataFrame({"x": pl.Series([2, 6, 10], dtype=pl.Float32)})
    state = {"type": "maxabs_scaler", "columns": ["x"], "scale": [scale], "max_abs": [scale]}
    applier = NodeRegistry.get_applier("MaxAbsScaler")()
    assert applier.validate_inference_state(state) is state
    expected = sample.with_columns((pl.col("x") / scale).alias("x"))
    assert_polars_frame_equal(applier.apply(sample, state), expected, check_exact=True)
    singleton = pl.concat([applier.apply(sample.slice(i, 1), state) for i in range(len(sample))])
    assert_polars_frame_equal(singleton, expected, check_exact=True)


@pytest.mark.parametrize("node", ["MaxAbsScaler", "RobustScaler"])
def test_polars_scalers_retain_lazy_inputs(node):
    """Repairing eager partition replay must not collect or reject existing lazy inputs."""
    sample = pl.DataFrame({"x": [2.0, 6.0, 10.0]})
    state = NodeRegistry.get_calculator(node)().fit(sample, {"columns": ["x"]})
    applier = NodeRegistry.get_applier(node)()
    result = applier.apply(sample.lazy(), state)
    assert isinstance(result, pl.LazyFrame)
    assert_polars_frame_equal(result.collect(), applier.apply(sample, state), check_exact=True)


def test_maxabs_decimal_keeps_native_expression_boundary():
    """Decimal division must retain Polars' native result without a new cast policy."""
    sample = pl.DataFrame({"x": [Decimal("2"), Decimal("6"), None]})
    state = {"type": "maxabs_scaler", "columns": ["x"], "scale": [5.0], "max_abs": [5.0]}
    applier = NodeRegistry.get_applier("MaxAbsScaler")()
    for request in (sample, sample.slice(1, 1), sample.head(0)):
        expected = request.with_columns((pl.col("x") / 5.0).alias("x"))
        assert_polars_frame_equal(applier.apply(request, state), expected, check_exact=True)


@pytest.mark.parametrize("dtype", [pl.Float32, pl.Float64, pl.Int8, pl.Int64, pl.UInt16, pl.UInt64])
@pytest.mark.parametrize("stat_dtype", [np.float32, np.float64, np.int16, np.int64])
@pytest.mark.parametrize("with_mean", [False, True])
def test_standard_polars_preserves_dtype_and_exact_request_partitions(dtype, stat_dtype, with_mean):
    """Saved NumPy statistic types must survive consistent division on empty, lazy and split input."""
    values: list[float | None] = [None, 2, 6, 100, 0]
    if dtype.is_float():
        values += [-0.0, float("nan"), float("inf"), -float("inf")]
    sample = pl.DataFrame({"x": pl.Series(values, dtype=dtype), "keep": range(len(values))})
    state = {
        "columns": ["missing", "x"],
        "mean": np.array([100, 0], dtype=stat_dtype),
        "scale": np.array([100, 5], dtype=stat_dtype),
        "with_mean": with_mean,
        "with_std": True,
    }
    before, saved = sample.clone(), pickle.dumps(state)
    applier = NodeRegistry.get_applier("StandardScaler")()
    full = applier.apply(sample, state)
    centered = pl.col("x") - state["mean"][1] if with_mean else pl.col("x")
    legacy = sample.select((centered / state["scale"][1]).alias("x"))
    assert full["x"].dtype == legacy["x"].dtype
    expected = np.divide(
        sample.select(centered).to_series().to_numpy(),
        state["scale"][1:],
        dtype=str(legacy["x"].dtype).lower(),
    )
    np.testing.assert_array_equal(full["x"].to_numpy(), expected)
    assert full["x"].is_null().equals(sample["x"].is_null())
    for size in (1, 2, 3):
        parts = [applier.apply(sample.slice(i, size), state) for i in range(0, len(sample), size)]
        assert_polars_frame_equal(pl.concat(parts), full, check_exact=True)
    assert_polars_frame_equal(applier.apply(sample.head(0), state), full.head(0), check_exact=True)
    lazy = applier.apply(sample.lazy(), state)
    assert isinstance(lazy, pl.LazyFrame)
    assert_polars_frame_equal(lazy.collect(), full, check_exact=True)
    assert_polars_frame_equal(
        applier.apply(sample.reverse(), state), full.reverse(), check_exact=True
    )
    if dtype.is_float():
        assert np.signbit(full["x"][5]) and full["x"].is_nan()[6]
    assert_polars_frame_equal(sample, before)
    assert pickle.dumps(state) == saved


@pytest.mark.parametrize("with_mean,with_std", list(product([False, True], repeat=2)))
def test_standard_polars_zero_scale_and_disabled_flags_preserve_native_output(with_mean, with_std):
    """Zero-scale guards and disabled options must retain existing Float32 promotion and values."""
    sample = pl.DataFrame({"x": pl.Series([None, 2, 6, -0.0], dtype=pl.Float32)})
    state = {
        "columns": ["x"],
        "mean": [2.0],
        "scale": [0.0],
        "with_mean": with_mean,
        "with_std": with_std,
    }
    expression = pl.col("x")
    if with_mean:
        expression = expression - np.float64(2)
    if with_std:
        expression = expression / 1.0
    expected = sample.with_columns(expression)
    applier = NodeRegistry.get_applier("StandardScaler")()
    assert_polars_frame_equal(applier.apply(sample, state), expected, check_exact=True)
    assert_polars_frame_equal(
        applier.apply(sample.head(0), state), expected.head(0), check_exact=True
    )


@pytest.mark.parametrize(
    "dtype,values",
    [
        (pl.Decimal(scale=1), [Decimal("2"), Decimal("6"), None]),
        (pl.Int128, [2, 6, None]),
        (pl.Boolean, [True, False, None]),
    ],
)
def test_standard_polars_retains_nonprimitive_expression_boundary(dtype, values):
    """Types outside NumPy's primitive bridge must keep the existing Polars expression semantics."""
    sample = pl.DataFrame({"x": pl.Series(values, dtype=dtype)})
    state = {"columns": ["x"], "mean": None, "scale": [5.0], "with_mean": False}
    applier = NodeRegistry.get_applier("StandardScaler")()
    for request in (sample, sample.slice(1, 1), sample.head(0)):
        expected = request.with_columns((pl.col("x") / np.float64(5)).alias("x"))
        assert_polars_frame_equal(applier.apply(request, state), expected, check_exact=True)


@pytest.mark.parametrize("value", [0.0, -0.0, 1e-320, 1.0, None])
def test_standard_polars_tiny_scale_uses_exact_native_division(value):
    """Reciprocal overflow must not turn finite quotients or signed zeros into infinities or NaNs."""
    sample = pl.DataFrame({"x": [value, 1e-320]})
    state = {"columns": ["x"], "mean": None, "scale": [1e-320], "with_mean": False}
    before, saved = sample.clone(), pickle.dumps(state)
    applier = NodeRegistry.get_applier("StandardScaler")()
    with np.errstate(over="ignore", invalid="ignore"):
        expected = np.divide(sample["x"].to_numpy(), 1e-320)
        full = applier.apply(sample, state)
        singleton = pl.concat(
            [applier.apply(sample.slice(i, 1), state) for i in range(len(sample))]
        )
        empty = applier.apply(sample.head(0), state)
    np.testing.assert_array_equal(full["x"].to_numpy(), expected)
    np.testing.assert_array_equal(np.signbit(full["x"].to_numpy()), np.signbit(expected))
    assert full["x"].is_null().equals(sample["x"].is_null())
    assert_polars_frame_equal(full, singleton, check_exact=True)
    assert_polars_frame_equal(empty, full.head(0), check_exact=True)
    assert_polars_frame_equal(sample, before)
    assert pickle.dumps(state) == saved


@pytest.mark.parametrize("with_mean,with_std", list(product([False, True], repeat=2)))
def test_standard_polars_context_is_local_without_worker_admission(
    with_mean, with_std, monkeypatch
):
    """Reviewed Polars context must reuse saved validation without granting native or worker execution."""
    flags = {"with_mean": with_mean, "with_std": with_std}
    state = NodeRegistry.get_calculator("StandardScaler")().fit(
        pl.DataFrame({"x": [-5.0, 5.0]}), {"columns": ["x"], **flags}
    )
    saved = pickle.dumps(state)
    _poison_fit(monkeypatch, "StandardScaler")
    for engine, kind in [("polars", "local"), ("pandas", "python_batch"), ("spark", "native")]:
        capability = get_inference_capability("StandardScaler", flags, state, engine=engine)
        assert capability is not None
        assert (capability.context, capability.row_effect, capability.execution_kind) == (
            "row",
            "preserve",
            kind,
        )
    for kind in ("python_batch", "native"):
        with pytest.raises(UnsupportedExecutionError):
            require_capability(
                "StandardScaler", "apply", "polars", config=flags, execution_kind=kind
            )
    malformed = {**state, "type": "invalid_scaler"}
    assert get_inference_capability("StandardScaler", flags, malformed, engine="polars") is None
    applier = NodeRegistry.get_applier("StandardScaler")()
    applier.apply = lambda *args, **kwargs: None
    assert (
        get_inference_capability("StandardScaler", flags, state, engine="polars", applier=applier)
        is None
    )
    assert pickle.dumps(state) == saved


@pytest.mark.parametrize(
    "node,center", [("MaxAbsScaler", False), ("RobustScaler", False), ("RobustScaler", True)]
)
@pytest.mark.parametrize("dtype", [pl.Float32, pl.Float64, pl.Int8])
@pytest.mark.parametrize("stat_dtype", [float, np.float32, np.float64, np.int16])
def test_scaler_lazy_columns_bind_native_dtype_and_statistics(node, center, dtype, stat_dtype):
    """Each lazy column must keep its own saved scale and native bulk values across request sizes."""
    values: list[float | None] = [None, 2, 6, 0]
    if dtype.is_float():
        values += [-0.0, float("nan"), float("inf")]
    sample = pl.DataFrame({"x": pl.Series(values, dtype=dtype), "z": [6.0] * len(values)})
    scales = [stat_dtype(100), stat_dtype(5), stat_dtype(3)]
    centers = [100.0, 2.0, 0.0]
    state = {"columns": ["missing", "x", "z"], "scale": scales}
    if node == "MaxAbsScaler":
        state.update(type="maxabs_scaler", max_abs=scales)
    else:
        state.update(
            type="robust_scaler",
            center=centers if center else None,
            with_centering=center,
            with_scaling=True,
            quantile_range=(25.0, 75.0),
        )
    expressions = []
    for i, name in enumerate(["x", "z"], start=1):
        column = pl.col(name).cast(pl.Float64) if node == "RobustScaler" else pl.col(name)
        if center:
            column = column - centers[i]
        expressions.append((column / scales[i]).alias(name))
    expected = sample.with_columns(expressions)
    before, saved = sample.clone(), pickle.dumps(state)
    applier = NodeRegistry.get_applier(node)()
    for request in (sample, sample.lazy()):
        actual = applier.apply(request, state)
        if isinstance(request, pl.LazyFrame):
            assert isinstance(actual, pl.LazyFrame)
            actual = actual.collect()
        assert_polars_frame_equal(actual, expected, check_exact=True)
    for size in (1, 2):
        parts = [
            applier.apply(sample.slice(i, size).lazy(), state).collect()
            for i in range(0, len(sample), size)
        ]
        assert_polars_frame_equal(pl.concat(parts), expected, check_exact=True)
    assert_polars_frame_equal(
        applier.apply(sample.head(0).lazy(), state).collect(), expected.head(0)
    )
    assert_polars_frame_equal(sample, before)
    assert pickle.dumps(state) == saved


@pytest.mark.parametrize("node", ["MaxAbsScaler", "RobustScaler"])
@pytest.mark.parametrize("scale", [0.0, 1e-320])
def test_scaler_lazy_zero_and_tiny_scales_keep_legacy_bulk_semantics(node, scale):
    """Partition repair must preserve legacy reciprocal overflow, zero guards and signed zeros."""
    sample = pl.DataFrame({"x": [None, 0.0, -0.0, 1e-320, 1.0, 6.0]})
    state = {"columns": ["x"], "scale": [np.float64(scale)], "with_centering": False}
    denominator = np.float64(scale) if scale != 0 else 1.0
    expected = sample.with_columns((pl.col("x") / denominator).alias("x"))
    applier = NodeRegistry.get_applier(node)()
    full = applier.apply(sample.lazy(), state).collect()
    singleton = pl.concat(
        [applier.apply(sample.slice(i, 1).lazy(), state).collect() for i in range(len(sample))]
    )
    assert_polars_frame_equal(full, expected, check_exact=True)
    assert_polars_frame_equal(singleton, expected, check_exact=True)
    np.testing.assert_array_equal(
        np.signbit(full["x"].to_numpy()), np.signbit(expected["x"].to_numpy())
    )
