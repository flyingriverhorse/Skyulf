from typing import Any, cast

from fastapi import HTTPException

from backend.data_ingestion.service import DataIngestionService
from backend.ml_pipeline.constants import StepType
from backend.utils.file_utils import extract_file_path_from_source


def _extract_node_fields(node: Any) -> tuple[dict[str, Any], str, list[Any]]:
    """Extract (params, step_type, inputs) from a node, handling both dicts and Pydantic models."""
    if isinstance(node, dict):
        return node.get("params", {}), node.get("step_type", ""), node.get("inputs", [])
    return getattr(node, "params", {}), getattr(node, "step_type", ""), getattr(node, "inputs", [])


def _is_loader_node(step_type: str, inputs: list[Any], params: dict[str, Any]) -> bool:
    """Return True when this node is a data loader (explicit or implicit) with a dataset_id to resolve."""
    is_data_loader = step_type == StepType.DATA_LOADER
    is_implicit_loader = (
        step_type == StepType.FEATURE_ENGINEERING and not inputs and "dataset_id" in params
    )
    return (is_data_loader or is_implicit_loader) and "dataset_id" in params


async def _resolve_dataset_source(raw_id: Any, ingestion_service: DataIngestionService) -> Any:
    """Look up a dataset source by its raw ID (UUID str) or, failing that, as an int."""
    ds = await ingestion_service.get_source(raw_id)
    # If not found and it looks like an int, try converting
    if not ds and str(raw_id).isdigit():
        ds = await ingestion_service.get_source(int(raw_id))
    return ds


async def _resolve_and_apply_dataset_path(
    raw_id: Any, params: dict[str, Any], ingestion_service: DataIngestionService
) -> dict[str, Any]:
    """Resolve a dataset_id to a file path, write it into ``params``, and return S3 storage options if any."""
    ds = await _resolve_dataset_source(raw_id, ingestion_service)
    if not ds:
        raise HTTPException(status_code=404, detail=f"Dataset {raw_id} not found")

    ds_dict = {
        "connection_info": ds.config,
        "file_path": ds.config.get("file_path") if ds.config else None,
    }
    path = extract_file_path_from_source(ds_dict)
    if not path:
        raise HTTPException(status_code=400, detail=f"Could not resolve path for dataset {raw_id}")

    params["path"] = str(path)
    # Keep the original ID for reference, but path is what matters for execution
    # params["dataset_id"] = str(path) # Don't overwrite ID with path, engine uses path if present

    # Extract credentials if S3
    if str(path).startswith("s3://") and ds.config and "storage_options" in ds.config:
        return cast(dict[str, Any], ds.config["storage_options"])
    return {}


async def resolve_pipeline_nodes(
    nodes: list[Any], ingestion_service: DataIngestionService
) -> dict[str, Any]:
    """
    Resolves dataset IDs to paths in the nodes and returns any S3 storage options found.
    Modifies the nodes in-place.
    """
    resolved_s3_options: dict[str, Any] = {}

    for node in nodes:
        # Handle both Pydantic models and dicts/objects
        # We need to be able to modify params
        params, step_type, inputs = _extract_node_fields(node)

        if not _is_loader_node(step_type, inputs, params):
            continue

        try:
            raw_id = params["dataset_id"]
            s3_options = await _resolve_and_apply_dataset_path(raw_id, params, ingestion_service)
            if s3_options:
                resolved_s3_options = s3_options
        except HTTPException:
            # Re-raise known HTTP errors (404 dataset not found, 400 bad path)
            # so the API surface returns the correct status instead of silently
            # allowing execution to proceed with an unresolved dataset.
            raise
        except Exception as e:
            # Non-HTTP errors (e.g. DB connection failure) — log and continue;
            # the engine will surface a clearer error if `path` remains unset.
            import logging

            logging.getLogger(__name__).warning(
                f"Failed to resolve dataset {params.get('dataset_id')}: {e}"
            )

    return resolved_s3_options
