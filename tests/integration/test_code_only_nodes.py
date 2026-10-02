"""Code-only Core nodes must never run from a server-side graph.

They call a Python function named in their parameters, so accepting them over
HTTP would let a request choose which server function runs.
"""

from __future__ import annotations

import pandas as pd
import pytest
from pydantic import ValidationError

from backend.data.catalog import FileSystemCatalog
from backend.ml_pipeline._execution.engine import PipelineEngine
from backend.ml_pipeline._execution.schemas import NodeConfig
from backend.ml_pipeline._internal._code_only_nodes import code_only_step_types
from backend.ml_pipeline._internal._schemas import NodeConfigModel
from backend.ml_pipeline.api import _build_node_registry
from backend.ml_pipeline.artifacts.local import LocalArtifactStore

_HOSTILE = {"function": "os:system", "output": ["x"], "replace": False, "params": {}}


def test_code_only_types_are_the_three_function_nodes():
    """The guard must cover exactly the function-calling nodes, nothing canvas users need."""
    assert code_only_step_types() == {"ColumnFunction", "FittedFunction", "RowFilterFunction"}


@pytest.mark.parametrize("step_type", ["ColumnFunction", "FittedFunction", "RowFilterFunction"])
def test_api_rejects_code_only_node(step_type):
    """A submitted graph cannot name a server function to call."""
    with pytest.raises(ValidationError, match="project code"):
        NodeConfigModel(node_id="n", step_type=step_type, params=_HOSTILE)


def test_api_rejects_code_only_step_nested_in_feature_engineering():
    """Wrapping the node inside a feature_engineering step list must not bypass the guard."""
    params = {"steps": [{"name": "s", "transformer": "ColumnFunction", "params": _HOSTILE}]}
    with pytest.raises(ValidationError, match="ColumnFunction"):
        NodeConfigModel(node_id="n", step_type="feature_engineering", params=params)


@pytest.mark.parametrize("runner", ["_run_transformer", "_run_feature_engineering"])
def test_engine_rejects_code_only_steps_without_calling_them(tmp_path, monkeypatch, runner):
    """Configs that skip API validation (saved or queued jobs) are blocked before resolution."""
    import skyulf.preprocessing.function_steps as function_steps

    def fail(_ref):
        """Fail if the engine reaches function resolution."""
        raise AssertionError("function resolved")

    monkeypatch.setattr(function_steps, "resolve_function", fail)
    engine = PipelineEngine(LocalArtifactStore(str(tmp_path)), FileSystemCatalog())
    monkeypatch.setattr(engine, "_get_input", lambda *_args: pd.DataFrame({"a": [1, 2]}))
    monkeypatch.setattr(engine, "_execution_target_column", lambda _node: None)
    step = {"name": "s", "transformer": "RowFilterFunction", "params": _HOSTILE}
    if runner == "_run_transformer":
        node = NodeConfig(node_id="n", step_type="RowFilterFunction", params=_HOSTILE, inputs=["x"])
    else:
        node = NodeConfig(
            node_id="n", step_type="feature_engineering", params={"steps": [step]}, inputs=["x"]
        )
    with pytest.raises(ValueError, match="RowFilterFunction"):
        getattr(engine, runner)(node)


def test_registry_endpoint_hides_code_only_nodes():
    """The canvas registry must not advertise nodes the canvas cannot submit."""
    ids = {item.id for item in _build_node_registry()}
    assert ids.isdisjoint(code_only_step_types())
    assert "SimpleImputer" in ids
