import asyncio
import logging
from typing import Any, Dict, List

from core.feature_engineering.modeling.shared import celery_app
from core.database.models import get_database_session
from core.feature_engineering.execution.engine import run_full_dataset_execution

logger = logging.getLogger(__name__)

async def _run_full_execution_workflow(
    dataset_source_id: str,
    execution_order: List[str],
    node_map: Dict[str, Dict[str, Any]],
    pipeline_id: str,
    applied_steps: List[str],
    preview_total_rows: int,
) -> Dict[str, Any]:
    
    async with get_database_session() as session:
         signal, rows = await run_full_dataset_execution(
            session=session,
            dataset_source_id=dataset_source_id,
            execution_order=execution_order,
            node_map=node_map,
            pipeline_id=pipeline_id,
            applied_steps=applied_steps,
            preview_total_rows=preview_total_rows,
        )
         # Return the signal as a dictionary so it can be serialized by Celery
         return signal.dict()

@celery_app.task(
    bind=True, 
    name="core.feature_engineering.execution.tasks.run_full_dataset_execution_task",
    queue="mlops-training"
)
def run_full_dataset_execution_task(
    self,
    dataset_source_id: str,
    execution_order: List[str],
    node_map: Dict[str, Dict[str, Any]],
    pipeline_id: str,
    applied_steps: List[str],
    preview_total_rows: int,
) -> Dict[str, Any]:
    
    logger.info(f"Starting full execution task. Pipeline ID: {pipeline_id}")
    logger.info(f"Execution Order: {execution_order}")
    
    # Fallback print for immediate visibility in console
    print(f"--- Starting full execution task. Pipeline ID: {pipeline_id} ---")
    print(f"--- Execution Order: {execution_order} ---")
    
    return asyncio.run(_run_full_execution_workflow(
        dataset_source_id,
        execution_order,
        node_map,
        pipeline_id,
        applied_steps,
        preview_total_rows
    ))
