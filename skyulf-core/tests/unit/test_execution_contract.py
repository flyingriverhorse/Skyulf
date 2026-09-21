"""Execution options must reject ambiguous identities and implicit coercion."""

from dataclasses import FrozenInstanceError

import pytest


def test_execution_options_preserve_explicit_engine():
    """Selecting a runtime must not change the process's current dataframe engine."""
    from skyulf.core.execution import ExecutionOptions
    from skyulf.engines import get_engine

    before = get_engine().name
    options = ExecutionOptions(engine="spark", python_batch_rows=17)
    assert options.engine == "spark"
    assert options.python_batch_rows == 17
    assert get_engine().name == before


@pytest.mark.parametrize("engine", ["cloud", "databricks", "Spark", "", None])
def test_execution_options_reject_unknown_engine(engine):
    """A platform or misspelled engine must not silently select a fallback."""
    from skyulf.core.execution import ExecutionOptions

    with pytest.raises(ValueError, match="engine"):
        ExecutionOptions(engine=engine)


@pytest.mark.parametrize("field", ["state_max_bytes", "python_batch_rows", "model_max_bytes"])
@pytest.mark.parametrize("value", [0, -1, True, 1.5, "12"])
def test_execution_options_require_positive_integer_limits(field, value):
    """Invalid resource limits must fail before a distributed job is submitted."""
    from skyulf.core.execution import ExecutionOptions

    with pytest.raises(ValueError, match=field):
        ExecutionOptions.from_config({"engine": "spark", field: value})


def test_execution_config_rejects_typo():
    """A misspelled batch limit must not be ignored by permissive node config parsing."""
    from skyulf.core.execution import ExecutionOptions

    with pytest.raises(TypeError, match="python_batch_row"):
        ExecutionOptions.from_config({"engine": "spark", "python_batch_row": 17})


def test_execution_config_does_not_mutate_input():
    """Parsing must not consume configuration shared with another execution."""
    from skyulf.core.execution import ExecutionOptions

    config = {"engine": "polars", "python_batch_rows": 17}
    options = ExecutionOptions.from_config(config)
    assert config == {"engine": "polars", "python_batch_rows": 17}
    assert options.engine == "polars"


@pytest.mark.parametrize("keys", [(), ("id", "id"), ("",), ("  ",), (1,), ["id"], "id"])
def test_frame_spec_rejects_ambiguous_keys(keys):
    """Row matching requires a nonempty immutable tuple of distinct column names."""
    from skyulf.core.execution import FrameSpec

    with pytest.raises(ValueError, match="row_keys"):
        FrameSpec(row_keys=keys)


@pytest.mark.parametrize("target", ["id", "", "  ", 1])
def test_frame_spec_rejects_invalid_target(target):
    """The label column must not also identify an observation."""
    from skyulf.core.execution import FrameSpec

    with pytest.raises(ValueError, match="target"):
        FrameSpec(row_keys=("id",), target=target)


def test_contracts_are_immutable():
    """A concurrent caller must not be able to mutate another run's options."""
    from skyulf.core.execution import ExecutionOptions, FrameSpec

    spec = FrameSpec(row_keys=("entity", "event"), target="label")
    options = ExecutionOptions(engine="pandas")
    for instance, field, value in [(spec, "target", "other"), (options, "engine", "spark")]:
        with pytest.raises(FrozenInstanceError):
            setattr(instance, field, value)
    assert spec.row_keys == ("entity", "event")
