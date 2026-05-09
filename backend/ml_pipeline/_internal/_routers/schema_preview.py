"""`POST /schema-preview` — pre-execution schema prediction (C7 Phase C+D).

Walks the pipeline topology without running it and returns the predicted
output schema for every node plus any broken column references found in
node ``params``.

Frontend consumers paint:
- "↳ N cols" badges on each node's output handle (from ``predicted_schemas``)
- red borders on nodes listed in ``broken_references``

Loaders are unseeded for now — Phase C-follow-up will accept dataset
metadata to seed loader schemas from the catalog without loading the
underlying data.
"""

from __future__ import annotations

import logging
from typing import Any, Dict, List, Optional

from fastapi import APIRouter, Depends
from pydantic import BaseModel
from sqlalchemy import select
from sqlalchemy.ext.asyncio import AsyncSession

from backend.database.engine import get_async_session
from backend.database.models import DataSource
from backend.ml_pipeline._execution._schema_graph import (
    predict_schemas,
    schemas_to_dict,
)
from backend.ml_pipeline._execution._schema_validator import find_broken_references
from backend.ml_pipeline._execution.schemas import (
    NodeConfig,
    PipelineConfig,
    coerce_step_type,
)
from backend.ml_pipeline._internal._schemas import PipelineConfigModel
from skyulf.preprocessing import SkyulfSchema

logger = logging.getLogger(__name__)

router = APIRouter(tags=["ML Pipeline"])


class SchemaPreviewResponse(BaseModel):
    """Response payload for `POST /schema-preview`."""

    pipeline_id: str
    # Per-node predicted output schema. Value is `None` when the node's
    # Calculator does not implement `infer_output_schema`, when an
    # upstream prediction was unknown, or when prediction failed.
    predicted_schemas: Dict[str, Optional[Dict[str, Any]]]
    # Broken column references (typo / deleted upstream column / etc.)
    # detected in node `params`. Each entry: {node_id, field, column,
    # upstream_node_id}. Empty list when nothing to flag.
    broken_references: List[Dict[str, Any]]


def _to_pipeline_config(model: PipelineConfigModel) -> PipelineConfig:
    """Coerce the API model into the dataclass the schema graph expects."""
    nodes = [
        NodeConfig(
            node_id=n.node_id,
            step_type=coerce_step_type(n.step_type),
            params=dict(n.params),
            inputs=list(n.inputs),
        )
        for n in model.nodes
    ]
    return PipelineConfig(
        pipeline_id=model.pipeline_id,
        nodes=nodes,
        metadata=dict(model.metadata),
    )


async def _seed_loader_schemas(
    config: PipelineConfig,
    session: AsyncSession,
) -> Dict[str, SkyulfSchema]:
    """Seed initial schemas for data_loader nodes from the dataset catalog.

    Looks up each loader's ``dataset_id`` param in the DataSource table by
    integer ``id`` (the value the canvas stores) and builds a SkyulfSchema
    from the ingested ``source_metadata["schema"]`` map.
    Nodes whose dataset is not found (or not yet ingested) are silently
    skipped — their schema remains None and prediction cascades to None for
    any downstream node.
    """
    seeds: Dict[str, SkyulfSchema] = {}
    for node in config.nodes:
        raw = node.step_type
        step = raw.value if hasattr(raw, "value") else str(raw)
        if step != "data_loader":
            continue
        dataset_id = node.params.get("dataset_id")
        if not dataset_id:
            continue
        try:
            row = await session.scalar(select(DataSource).where(DataSource.id == int(dataset_id)))
        except (ValueError, TypeError):
            # dataset_id is not an integer — try source_id UUID fallback
            row = await session.scalar(
                select(DataSource).where(DataSource.source_id == str(dataset_id))
            )
        if row is None or not row.source_metadata:
            continue
        schema_meta: Dict[str, str] = row.source_metadata.get("schema", {})
        if not schema_meta:
            continue
        seeds[node.node_id] = SkyulfSchema.from_columns(list(schema_meta.keys()), dict(schema_meta))
    return seeds


@router.post("/schema-preview", response_model=SchemaPreviewResponse)
async def schema_preview(
    config: PipelineConfigModel,
    session: AsyncSession = Depends(get_async_session),
) -> SchemaPreviewResponse:
    """Predict each node's output schema and surface broken column refs.

    Seeds data_loader nodes from the dataset catalog so downstream predictions
    propagate correctly. Pure read — no execution, no DB writes.
    """
    pipeline_config = _to_pipeline_config(config)
    initial_schemas = await _seed_loader_schemas(pipeline_config, session)
    predicted = predict_schemas(pipeline_config, initial_schemas if initial_schemas else None)
    broken = find_broken_references(pipeline_config, predicted)
    return SchemaPreviewResponse(
        pipeline_id=pipeline_config.pipeline_id,
        predicted_schemas=schemas_to_dict(predicted),
        broken_references=broken,
    )
