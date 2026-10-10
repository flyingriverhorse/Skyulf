"""Fitted inference metadata describes context without running transformations."""

import numpy as np
import pandas as pd
import polars as pl
import pytest

from skyulf.core.capabilities import (
    ExecutionCapability,
    UnsupportedExecutionError,
    require_capability,
)
from skyulf.inference.project_code import load_project_module
from skyulf.preprocessing import column_step, filter_step, fitted_step
from skyulf.preprocessing.function_steps import ColumnFunctionApplier
from skyulf.preprocessing.inference_context import get_inference_capability as _capability
from skyulf.registry import NodeRegistry


def values(frame):
    """Return a row-local column for real custom-step fitting."""
    return frame["x"] * 2


def learn(frame, target):
    """Save a scalar that can be reused without neighboring inference rows."""
    return {"offset": float(frame["x"].mean())}


def apply_saved(frame, state):
    """Apply a fitted scalar without learning from the request."""
    return frame["x"] - state["offset"]


def keep(frame):
    """Provide a real boolean filter for saving the declaration."""
    return frame["x"] > 0


def _custom_step(kind, **kwargs):
    """Build the same public APIs used by project recipes."""
    if kind == "ColumnFunction":
        return column_step("custom", values, output="result", **kwargs)
    if kind == "FittedFunction":
        return fitted_step("custom", learn, apply_saved, output="result", **kwargs)
    return filter_step("custom", keep, columns=["x"], **kwargs)


def _frame(engine):
    """Supply matching training values to both local engines."""
    frame = pd.DataFrame({"x": [1.0, 2.0, 4.0], "group": ["a", "a", "b"]})
    return pl.from_pandas(frame) if engine == "polars" else frame


@pytest.mark.parametrize("node", ["missing_context_node"])
def test_undeclared_nodes_stay_unknown(node):
    """Unreviewed behavior must not become row-local merely because it is registered."""
    assert _capability(node, {}, {}, engine="pandas") is None


@pytest.mark.parametrize("engine", ["pandas", "polars"])
@pytest.mark.parametrize("kind", ["ColumnFunction", "FittedFunction", "RowFilterFunction"])
def test_custom_default_stays_unknown_after_fit(kind, engine):
    """Existing custom recipes retain their saved payload and gain no implicit promise."""
    step = _custom_step(kind)
    state = NodeRegistry.get_calculator(kind)().fit(_frame(engine), step["params"])
    assert "inference_context" not in step["params"]
    assert "inference_context" not in state
    assert _capability(kind, step["params"], state, engine=engine) is None


@pytest.mark.parametrize("engine", ["pandas", "polars"])
@pytest.mark.parametrize("context", ["row", "group", "window", "global"])
@pytest.mark.parametrize("kind", ["ColumnFunction", "FittedFunction", "RowFilterFunction"])
def test_custom_declaration_survives_fit(kind, context, engine):
    """A user declaration must describe the saved apply without granting worker execution."""
    step = _custom_step(kind, inference_context=context)
    state = NodeRegistry.get_calculator(kind)().fit(_frame(engine), step["params"])
    capability = _capability(kind, step["params"], state, engine=engine)
    assert state["inference_context"] == context
    assert capability is not None
    assert (capability.engine, capability.operation, capability.execution_kind) == (
        engine,
        "apply",
        "local",
    )
    assert capability.context == context
    assert capability.row_effect == ("filter" if kind == "RowFilterFunction" else "preserve")
    with pytest.raises(UnsupportedExecutionError):
        require_capability(
            kind, "apply", engine, config=step["params"], execution_kind="python_batch"
        )


@pytest.mark.parametrize("kind", ["ColumnFunction", "FittedFunction", "RowFilterFunction"])
@pytest.mark.parametrize("context", ["batch", "", True, ["row"]])
def test_custom_builders_reject_invalid_context(kind, context):
    """Malformed promises must fail before recipes are saved."""
    with pytest.raises(ValueError, match="context"):
        _custom_step(kind, inference_context=context)


@pytest.mark.parametrize("engine", ["pandas", "polars"])
@pytest.mark.parametrize(
    ("node", "config", "context", "row_effect"),
    [
        ("LagFeatures", {"columns": ["x"], "lags": [1]}, "window", "preserve"),
        ("LagFeatures", {"columns": ["x"], "lags": [1], "drop_na": True}, "window", "filter"),
        ("RollingAggregate", {"columns": ["x"], "window": 2}, "window", "preserve"),
        ("Deduplicate", {"subset": ["x"], "keep": "last"}, "global", "filter"),
    ],
)
def test_context_comes_from_saved_applier_without_apply(node, config, context, row_effect, engine):
    """Temporal and duplicate dependencies must remain visible before any rows are executed."""
    state = NodeRegistry.get_calculator(node)().fit(_frame(engine), config)
    applier = NodeRegistry.get_applier(node)()
    capability = _capability(node, config, state, engine=engine, applier=applier)
    assert capability is not None
    assert (capability.context, capability.row_effect, capability.execution_kind) == (
        context,
        row_effect,
        "local",
    )


@pytest.mark.parametrize("engine", ["pandas", "polars"])
def test_carry_history_still_requires_a_window(engine):
    """Saving training history cannot make independently evaluated chunks row-local."""
    config = {"columns": ["x"], "lags": [1], "sort_by": "x", "history_mode": "carry"}
    state = NodeRegistry.get_calculator("LagFeatures")().fit(_frame(engine), config)
    capability = _capability("LagFeatures", config, state, engine=engine)
    assert state["history_mode"] == "carry"
    assert capability is not None and capability.context == "window"


@pytest.mark.parametrize(
    ("node", "config"),
    [
        ("SimpleImputer", {"columns": ["x"]}),
        ("StandardScaler", {"columns": ["x"]}),
        ("MinMaxScaler", {"columns": ["x"]}),
        ("GroupImputer", {"columns": ["x"], "group_by": "group", "strategy": "mean"}),
        ("OneHotEncoder", {"columns": ["group"], "max_categories": None}),
        ("FeatureInteraction", {"columns": ["x"]}),
        ("ClipValues", {"bounds": {"x": {"lower": 0, "upper": 5}}}),
    ],
)
def test_existing_declarations_resolve_fitted_defaults(node, config):
    """Existing state validators and default normalization remain the declaration owners."""
    state = NodeRegistry.get_calculator(node)().fit(_frame("pandas"), config)
    capability = _capability(node, config, state, engine="pandas")
    assert capability is not None
    assert (capability.context, capability.row_effect, capability.execution_kind) == (
        "row",
        "preserve",
        "python_batch",
    )
    polars_capability = _capability(node, config, state, engine="polars")
    assert polars_capability is not None
    assert (polars_capability.context, polars_capability.execution_kind) == ("row", "local")


def test_changed_saved_applier_is_unknown():
    """Registry metadata must not be borrowed by a different saved implementation."""
    config = {"columns": ["x"]}
    state = NodeRegistry.get_calculator("StandardScaler")().fit(_frame("pandas"), config)
    unrelated = NodeRegistry.get_applier("LagFeatures")()
    assert _capability("StandardScaler", config, state, engine="pandas", applier=unrelated) is None


def test_metadata_query_does_not_resolve_custom_callback():
    """Reporting saved promises must not import or execute project functions."""
    state = {"function": "missing.project:explode", "inference_context": "group"}
    capability = _capability("ColumnFunction", state, state, engine="pandas")
    assert capability is not None and capability.context == "group"


@pytest.mark.parametrize(
    "node",
    [
        "ColumnFunction",
        "FittedFunction",
        "RowFilterFunction",
        "LagFeatures",
        "RollingAggregate",
        "Deduplicate",
    ],
)
def test_local_hooks_do_not_declare_spark_execution(node):
    """Local context metadata must not imply remote engine support."""
    state = {"inference_context": "row"}
    assert _capability(node, {}, state, engine="spark") is None


def test_explicit_none_keeps_custom_payload_unchanged():
    """Forwarding an optional unset keyword must preserve existing saved recipe identity."""
    assert _custom_step("ColumnFunction", inference_context=None) == _custom_step("ColumnFunction")


def test_invalid_saved_custom_declaration_is_rejected():
    """Corrupted metadata must not become a row-local declaration after loading."""
    with pytest.raises(ValueError, match="context"):
        _capability("ColumnFunction", {}, {"inference_context": "automatic"}, engine="pandas")


@pytest.fixture
def isolated_registry(monkeypatch):
    """Keep test-only applier hooks from leaking into other registry consumers."""
    for name in ("_calculators", "_appliers", "_metadata"):
        monkeypatch.setattr(NodeRegistry, name, dict(getattr(NodeRegistry, name)))


def test_custom_class_hook_never_executes_apply(isolated_registry):
    """A saved class can describe its context without executing user transformations."""

    class Applier:
        """Declare group context while making accidental apply calls visible."""

        @staticmethod
        def inference_capability(state, *, engine):
            """Describe the saved grouping requirement."""
            return ExecutionCapability(engine, "apply", "local", "preserve", state["context"])

        def apply(self, data, params):
            """Fail if metadata inspection tries to transform data."""
            raise AssertionError("Metadata inspection called apply.")

    NodeRegistry.register("context_hook_probe", Applier)(object)
    capability = _capability(
        "context_hook_probe", {}, {"context": "group"}, engine="pandas", applier=Applier()
    )
    assert capability is not None and capability.context == "group"


def test_saved_instance_override_stays_unknown():
    """A replacement apply method must not borrow the registered class's declaration."""
    applier = NodeRegistry.get_applier("ColumnFunction")()
    applier.apply = lambda data, params: data
    assert (
        _capability(
            "ColumnFunction", {}, {"inference_context": "row"}, engine="pandas", applier=applier
        )
        is None
    )


def test_subclass_does_not_inherit_context_hook(isolated_registry):
    """A changed apply body must explicitly own its context promise."""

    class ChangedApplier(ColumnFunctionApplier):
        """Represent a custom implementation derived from a declaring node."""

    NodeRegistry.register("changed_context_hook", ChangedApplier)(object)
    assert (
        _capability("changed_context_hook", {}, {"inference_context": "row"}, engine="pandas")
        is None
    )


def test_valid_local_state_outside_worker_subset_stays_unknown():
    """Strict worker validators must not mislabel ordinary local state as corrupt."""
    config = {"columns": []}
    state = NodeRegistry.get_calculator("OneHotEncoder")().fit(_frame("pandas"), config)
    assert _capability("OneHotEncoder", config, state, engine="pandas") is None


def test_noncallable_explicit_hook_is_rejected(isolated_registry):
    """Malformed explicit hooks must not silently fall back to a different declaration."""

    class Applier:
        """Hold invalid metadata rather than a hook."""

        inference_capability = "row"

    NodeRegistry.register("invalid_context_hook", Applier)(object)
    with pytest.raises(ValueError, match="inference_capability"):
        _capability("invalid_context_hook", {}, {}, engine="pandas")


def test_unregistered_captured_applier_reports_its_own_context():
    """Fresh-process replay must not need to run a recipe builder to inspect saved classes."""
    module = load_project_module("""
from skyulf.core.capabilities import ExecutionCapability

class Calculator:
    def fit(self, data, config):
        raise AssertionError("Do not fit during metadata inspection.")

class Applier:
    @staticmethod
    def inference_capability(state, *, engine):
        return ExecutionCapability(engine, "apply", "local", "preserve", "row")

    def apply(self, data, state):
        raise AssertionError("Do not apply during metadata inspection.")
""")
    identity = f"{module.__name__}.Calculator.Applier"
    with pytest.raises(ValueError, match="not found"):
        NodeRegistry.get_applier(identity)
    capability = _capability(identity, {}, {}, engine="pandas", applier=module.Applier())
    assert capability is not None and capability.context == "row"
    with pytest.raises(ValueError, match="not found"):
        NodeRegistry.get_applier(identity)


@pytest.mark.parametrize(
    ("module", "identity"),
    [
        ("plain_module", "plain_module.Calculator.Applier"),
        ("_skyulf_project_short", "_skyulf_project_short.Calculator.Applier"),
        ("_skyulf_project_" + "a" * 64, "_skyulf_project_" + "b" * 64 + ".Calculator.Applier"),
        ("_skyulf_project_" + "a" * 64, "_skyulf_project_" + "a" * 64 + ".Calculator.Other"),
    ],
)
def test_unregistered_saved_identity_must_match_captured_module(module, identity):
    """A saved class cannot borrow a source-specific identity from another implementation."""
    applier = type(
        "Applier",
        (),
        {
            "__module__": module,
            "inference_capability": staticmethod(
                lambda state, *, engine: ExecutionCapability(
                    engine, "apply", "local", "preserve", "row"
                )
            ),
        },
    )
    assert _capability(identity, {}, {}, engine="pandas", applier=applier()) is None


def test_unregistered_captured_applier_without_hook_stays_unknown():
    """Fresh-process lookup must not require a calculator registration for undeclared classes."""
    module = "_skyulf_project_" + "c" * 64 + ".custom"
    applier = type("Applier", (), {"__module__": module})
    assert (
        _capability(f"{module}.Calculator.Applier", {}, {}, engine="pandas", applier=applier())
        is None
    )


@pytest.mark.parametrize("raw", [None, [], {"type": "other"}, {"type": "test", "extra": 1}])
def test_local_state_fields_rejects_corrupt_artifacts(raw):
    """Malformed local state cannot silently gain a context declaration."""
    from skyulf.preprocessing._fitted_validation import local_state_fields

    with pytest.raises(ValueError, match="state"):
        local_state_fields(raw, "test", {"type"}, allow_empty=True)


def test_local_state_fields_distinguishes_noop_from_missing_state():
    """Only owners whose fit emits an empty no-op may accept that saved artifact."""
    from skyulf.preprocessing._fitted_validation import local_state_fields

    assert local_state_fields({"type": "test"}, "test", {"type"}) is True
    assert local_state_fields({}, "test", {"type"}, allow_empty=True) is False
    with pytest.raises(ValueError, match="state"):
        local_state_fields({}, "test", {"type"})


@pytest.mark.parametrize("engine", ["pandas", "polars"])
def test_missing_indicator_context_rejects_saved_output_collision(engine):
    """A corrupted selection must not promise valid apply when its own flag overwrites input."""
    frame = pd.DataFrame({"x": [1.0, None], "x_missing": [2.0, None]})
    if engine == "polars":
        frame = pl.from_pandas(frame)
    config = {"columns": ["x_missing"]}
    state = NodeRegistry.get_calculator("MissingIndicator")().fit(frame, config)
    assert _capability("MissingIndicator", config, state, engine=engine) is not None
    state["columns"] = ["x", "x_missing"]
    with pytest.raises(ValueError, match="collid|existing"):
        _capability("MissingIndicator", config, state, engine=engine)


@pytest.mark.parametrize("engine", ["pandas", "polars"])
@pytest.mark.parametrize(
    "node", ["CustomBinning", "CorrelationThreshold", "UnivariateSelection", "VarianceThreshold"]
)
@pytest.mark.parametrize("enabled", [False, True])
def test_local_context_accepts_real_fitted_numpy_flags(node, engine, enabled):
    """NumPy booleans accepted by fitting must survive metadata inspection unchanged."""
    frame = pd.DataFrame(
        {"x": [0.0, 1.0, 2.0, 3.0], "copy": [0.0, 2.0, 4.0, 6.0], "constant": [1.0] * 4}
    )
    if engine == "polars":
        frame = pl.from_pandas(frame)
    flag = np.bool_(enabled)
    field = "drop_original" if node == "CustomBinning" else "drop_columns"
    config = {
        "columns": ["x", "copy"],
        field: flag,
        "bins": [0.0, 2.0, 4.0],
        "label_format": "range",
        "allow_missing_target": True,
    }
    state = NodeRegistry.get_calculator(node)().fit(frame, config)
    applier = NodeRegistry.get_applier(node)()
    output = applier.apply(frame, state)
    assert len(output) == len(frame)
    capability = _capability(node, config, state, engine=engine)
    assert capability is not None and capability.context == "row"
    assert state[field] is flag
    state[field] = np.array([True, False])
    with pytest.raises(ValueError, match="boolean|scalar"):
        _capability(node, config, state, engine=engine)
