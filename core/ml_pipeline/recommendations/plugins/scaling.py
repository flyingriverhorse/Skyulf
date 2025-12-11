from typing import List
from ..schemas import AnalysisProfile, Recommendation, RecommendationType, ColumnType
from .base import BaseAdvisorPlugin

class ScalingAdvisor(BaseAdvisorPlugin):
    def analyze(self, profile: AnalysisProfile) -> List[Recommendation]:
        recs = []
        
        cols_to_scale = []
        
        for col_name, col_profile in profile.columns.items():
            if col_profile.column_type == ColumnType.NUMERIC:
                # Check if scaling is needed
                # Heuristic: If range > 10 or std > 5, suggest scaling
                if col_profile.min_value is not None and col_profile.max_value is not None:
                    data_range = col_profile.max_value - col_profile.min_value
                    if data_range > 10 or (col_profile.std_value and col_profile.std_value > 3):
                        cols_to_scale.append(col_name)
        
        if cols_to_scale:
            recs.append(Recommendation(
                rule_id="standard_scaling",
                type=RecommendationType.SCALING,
                target_columns=cols_to_scale,
                description=f"Scale numeric columns {cols_to_scale} to unit variance.",
                suggested_node_type="StandardScaler",
                suggested_params={"columns": cols_to_scale},
                confidence=0.7,
                reasoning="Features have different scales/ranges, which can affect model performance."
            ))
            
        return recs
