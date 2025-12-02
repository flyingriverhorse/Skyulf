from typing import Any, Dict, Optional
from pydantic import BaseModel

class RecommendationRequest(BaseModel):
    dataset_source_id: str
    target_node_id: Optional[str] = None
    sample_size: int = 10000
    graph: Optional[Dict[str, Any]] = None


class SkewnessRecommendationRequest(RecommendationRequest):
    transformations: Optional[str] = None
