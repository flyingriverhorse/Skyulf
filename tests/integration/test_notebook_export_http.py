"""Validate the downloadable notebook boundary and unsupported graph diagnostics."""

from unittest.mock import AsyncMock

import httpx
import pytest
from fastapi import FastAPI

from backend.ml_pipeline._internal._routers import notebook_export as ne
from backend.ml_pipeline._internal._routers._notebook_builders import _NodeIn, _PipelineIn


@pytest.fixture
def export_app(monkeypatch):
    """Replace only dataset metadata lookups; use the real endpoint and builders."""
    monkeypatch.setattr(ne, "_lookup_dataset_name", AsyncMock(return_value="data.csv"))
    monkeypatch.setattr(ne, "_lookup_dataset_file_path", AsyncMock(return_value=None))
    app = FastAPI()
    app.include_router(ne.router)
    app.dependency_overrides[ne.get_async_session] = lambda: None
    return app


@pytest.mark.asyncio
@pytest.mark.parametrize("mode", ["full", "compact"])
async def test_unsupported_model_returns_actionable_client_error(export_app, mode):
    """An unsupported model must explain the export limitation instead of returning 500."""
    config = {
        "nodes": [
            {"node_id": "load", "step_type": "data_loader"},
            {
                "node_id": "model",
                "step_type": "training",
                "inputs": ["load"],
                "params": {"model_type": "voting_classifier", "target_column": "target"},
            },
        ]
    }
    async with httpx.AsyncClient(
        transport=httpx.ASGITransport(app=export_app), base_url="http://test"
    ) as client:
        response = await client.post(f"/pipeline/probe/export-notebook?mode={mode}", json=config)
    assert response.status_code == 400
    assert "does not yet support structural models" in response.json()["detail"]


@pytest.mark.asyncio
@pytest.mark.parametrize("mode", ["full", "compact"])
async def test_download_is_a_compilable_notebook(export_app, mode):
    """The HTTP download must preserve the format and safe target literals."""
    config = {
        "nodes": [
            {"node_id": "load", "step_type": "data_loader", "params": {"path": "data.csv"}},
            {
                "node_id": "target",
                "step_type": "feature_target_split",
                "params": {"target_column": 'label"name'},
                "inputs": ["load"],
            },
            {
                "node_id": "model",
                "step_type": "training",
                "params": {"algorithm": "logistic_regression"},
                "inputs": ["target"],
            },
        ]
    }
    async with httpx.AsyncClient(
        transport=httpx.ASGITransport(app=export_app), base_url="http://test"
    ) as client:
        response = await client.post(f"/pipeline/probe/export-notebook?mode={mode}", json=config)
    assert response.status_code == 200
    assert response.headers["content-type"] == "application/x-ipynb+json"
    assert ".ipynb" in response.headers["content-disposition"]
    notebook = response.json()
    assert all("id" in cell for cell in notebook["cells"])
    assert len({cell["id"] for cell in notebook["cells"]}) == len(notebook["cells"])
    for cell in notebook["cells"]:
        if cell["cell_type"] == "code":
            compile("".join(cell["source"]), "download.ipynb", "exec")
    assert notebook["nbformat"] == 4


@pytest.mark.asyncio
@pytest.mark.parametrize(
    "kind", ["merge", "multiple_loaders", "cycle", "missing_input", "post_model"]
)
async def test_unsupported_graph_returns_explicit_client_error(export_app, kind):
    """Unsupported topology must not be silently flattened into a different pipeline."""
    nodes = [
        {"node_id": "load", "step_type": "data_loader"},
        {"node_id": "a", "step_type": "StandardScaler", "inputs": ["load"]},
        {"node_id": "b", "step_type": "MinMaxScaler", "inputs": ["load"]},
        {
            "node_id": "model",
            "step_type": "training",
            "params": {"algorithm": "logistic_regression"},
            "inputs": ["a", "b"],
        },
    ]
    if kind == "multiple_loaders":
        nodes[2] = {"node_id": "b", "step_type": "data_loader"}
    elif kind == "cycle":
        nodes[1]["inputs"] = ["b"]
        nodes[2]["inputs"] = ["a"]
    elif kind == "missing_input":
        nodes[1]["inputs"] = ["missing"]
    elif kind == "post_model":
        nodes[1] = {"node_id": "a", "step_type": "training", "inputs": ["load"]}
        nodes[2]["inputs"] = ["a"]
        nodes[3]["inputs"] = ["load"]
    async with httpx.AsyncClient(
        transport=httpx.ASGITransport(app=export_app), base_url="http://test"
    ) as client:
        response = await client.post("/pipeline/probe/export-notebook", json={"nodes": nodes})
    assert response.status_code == 400
    assert "export" in response.json()["detail"].lower()


@pytest.mark.parametrize("mode", ["full", "compact"])
def test_disconnected_transform_is_not_applied_to_model(mode):
    """A disconnected canvas experiment must not alter the selected model's chain."""
    cfg = _PipelineIn(
        nodes=[
            _NodeIn(node_id="load", step_type="data_loader"),
            _NodeIn(
                node_id="model",
                step_type="training",
                params={"algorithm": "logistic_regression"},
                inputs=["load"],
            ),
            _NodeIn(node_id="unconnected", step_type="StandardScaler"),
        ]
    )
    builder = ne._build_full_notebook if mode == "full" else ne._build_compact_notebook
    notebook = builder(cfg, "probe", "data.csv")
    code = "".join(
        "".join(cell["source"]) for cell in notebook["cells"] if cell["cell_type"] == "code"
    )
    assert "StandardScaler" not in code
