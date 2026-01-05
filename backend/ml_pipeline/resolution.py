from typing import List, Dict, Any, Optional, cast
from fastapi import HTTPException
from backend.data_ingestion.service import DataIngestionService
from backend.utils.file_utils import extract_file_path_from_source
from backend.ml_pipeline.constants import StepType

async def resolve_pipeline_nodes(nodes: List[Any], ingestion_service: DataIngestionService) -> Dict[str, Any]:
    """
    Resolves dataset IDs to paths in the nodes and returns any S3 storage options found.
    Modifies the nodes in-place.
    """
    resolved_s3_options: Dict[str, Any] = {}
    
    for node in nodes:
        # Handle both Pydantic models and dicts/objects
        # We need to be able to modify params
        if isinstance(node, dict):
            params = node.get("params", {})
            step_type = node.get("step_type", "")
            inputs = node.get("inputs", [])
        else:
            params = getattr(node, "params", {})
            step_type = getattr(node, "step_type", "")
            inputs = getattr(node, "inputs", [])

        # Logic to identify data loader
        is_data_loader = step_type == StepType.DATA_LOADER
        is_implicit_loader = (
            step_type == StepType.FEATURE_ENGINEERING
            and not inputs
            and "dataset_id" in params
        )

        if (is_data_loader or is_implicit_loader) and "dataset_id" in params:
            try:
                raw_id = params["dataset_id"]
                # Try to resolve source by ID (int) or UUID (str)
                ds = await ingestion_service.get_source(raw_id)
                
                # If not found and it looks like an int, try converting
                if not ds and str(raw_id).isdigit():
                     ds = await ingestion_service.get_source(int(raw_id))

                if ds:
                    ds_dict = {
                        "connection_info": ds.config,
                        "file_path": ds.config.get("file_path") if ds.config else None,
                    }
                    path = extract_file_path_from_source(ds_dict)
                    if path:
                        params["path"] = str(path)
                        # Keep the original ID for reference, but path is what matters for execution
                        # params["dataset_id"] = str(path) # Don't overwrite ID with path, engine uses path if present
                        
                        # Extract credentials if S3
                        if str(path).startswith("s3://") and ds.config and "storage_options" in ds.config:
                            resolved_s3_options = cast(Dict[str, Any], ds.config["storage_options"])
                    else:
                        raise HTTPException(
                            status_code=400,
                            detail=f"Could not resolve path for dataset {raw_id}",
                        )
                else:
                    raise HTTPException(
                        status_code=404, detail=f"Dataset {raw_id} not found"
                    )
            except Exception as e:
                # Log error but don't crash if resolution fails (might be already a path)
                # But if it was an ID that needed resolution, this will likely fail later
                import logging
                logging.getLogger(__name__).warning(f"Failed to resolve dataset {params.get('dataset_id')}: {e}")
                pass 
                
    return resolved_s3_options
