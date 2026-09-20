"""Notebook readers should see useful configuration instead of canvas identifiers."""

import pytest

from backend.ml_pipeline._internal._routers import notebook_export as ne
from backend.ml_pipeline._internal._routers._notebook_builders import _NodeIn, _PipelineIn


@pytest.mark.parametrize("mode", ["full", "compact"])
@pytest.mark.parametrize("branched", [False, True])
def test_export_uses_readable_configuration_without_canvas_ids(mode, branched):
    """Internal identifiers and one-use step variables should not clutter runnable exports."""
    ids = [f"node-a03a9f8d-b0d5-48aa-822b-858ee888ba2{i}" for i in range(4)]
    nodes = [
        _NodeIn(node_id=ids[0], step_type="data_loader"),
        _NodeIn(
            node_id=ids[1], step_type="StandardScaler", params={"columns": ["x"]}, inputs=[ids[0]]
        ),
        _NodeIn(
            node_id=ids[2],
            step_type="training",
            params={"algorithm": "logistic_regression"},
            inputs=[ids[1]],
        ),
    ]
    if branched:
        nodes.append(
            _NodeIn(
                node_id=ids[3],
                step_type="training",
                params={"algorithm": "logistic_regression"},
                inputs=[ids[1]],
            )
        )
    builder = ne._build_full_notebook if mode == "full" else ne._build_compact_notebook
    notebook = builder(_PipelineIn(nodes=nodes), "data", "data.csv")
    source = "".join("".join(cell["source"]) for cell in notebook["cells"])
    assert all(node_id not in source for node_id in ids)
    assert "_step01" not in source
    assert "_steps.append" not in source
    assert "StandardScaler" in source
    assert "'columns': ['x']" in source
