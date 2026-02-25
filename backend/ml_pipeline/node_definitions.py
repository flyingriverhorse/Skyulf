from typing import Any, Dict, List
import logging
from pydantic import BaseModel
from backend.ml_pipeline.constants import StepType

logger = logging.getLogger(__name__)

# Try to import skyulf registry for dynamic node discovery
try:
    from skyulf.registry import NodeRegistry as SkyulfRegistry

    SKYULF_AVAILABLE = True
except ImportError:
    SKYULF_AVAILABLE = False
    logger.warning("Skyulf core registry not available. Using static definitions only.")

if SKYULF_AVAILABLE:
    try:
        import skyulf.preprocessing
    except ImportError as exc:
        logger.warning(f"Could not import skyulf preprocessing nodes: {exc}")

    try:
        import skyulf.modeling
    except ImportError as exc:
        logger.warning(f"Could not import skyulf modeling nodes: {exc}")


class RegistryItem(BaseModel):
    id: str
    name: str
    category: str
    description: str
    params: Dict[str, Any] = {}
    tags: List[str] = []


class NodeRegistry:
    @staticmethod
    def get_dynamic_nodes() -> List[RegistryItem]:
        """Fetch nodes dynamically from skyulf-core registry."""
        items = []
        if SKYULF_AVAILABLE:
            for node_id, metadata in SkyulfRegistry._metadata.items():
                 # Convert skyulf metadata to RegistryItem
                 # We assume metadata dict matches the structure or we map fields
                 # Skyulf meta: id, name, category, description, params
                 items.append(RegistryItem(**metadata))
        return items

    @staticmethod
    def get_all_nodes() -> List[RegistryItem]:
        static_nodes = [
            # --- Data Loading ---
            RegistryItem(
                id=StepType.DATA_LOADER,
                name="Data Loader",
                category="Data Source",
                description="Loads data from a source.",
                params={"source_id": "string", "date_range": "optional[dict]"},
            ),
        ]

        # Merge with dynamic nodes (favoring dynamic if ID exists)
        dynamic_nodes = NodeRegistry.get_dynamic_nodes()
        dynamic_ids = {node.id for node in dynamic_nodes}
        
        # Filter out static nodes that are overwritten by dynamic ones
        final_nodes = [n for n in static_nodes if n.id not in dynamic_ids]
        
        return final_nodes + dynamic_nodes
