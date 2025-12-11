from typing import List
from ..schemas import AnalysisProfile, Recommendation, RecommendationType, ColumnType
from .base import BaseAdvisorPlugin

class TransformationAdvisor(BaseAdvisorPlugin):
    def analyze(self, profile: AnalysisProfile) -> List[Recommendation]:
        recs = []
        
        yeo_johnson_candidates = []
        box_cox_candidates = []
        
        for col_name, col_profile in profile.columns.items():
            if col_profile.column_type == ColumnType.NUMERIC:
                # Check for skewness
                # Threshold of 1.0 is commonly used to indicate moderate to high skewness
                if col_profile.skewness is not None and abs(col_profile.skewness) > 1.0:
                    # Check if strictly positive for Box-Cox
                    if col_profile.min_value is not None and col_profile.min_value > 0:
                        box_cox_candidates.append(col_name)
                    else:
                        yeo_johnson_candidates.append(col_name)
        
        if box_cox_candidates:
            recs.append(Recommendation(
                rule_id="power_transform_box_cox",
                type=RecommendationType.TRANSFORMATION,
                target_columns=box_cox_candidates,
                description=f"Apply Box-Cox transformation to skewed positive columns {box_cox_candidates}.",
                suggested_node_type="PowerTransformer",
                suggested_params={"method": "box-cox", "columns": box_cox_candidates},
                confidence=0.7,
                reasoning="Column has high skewness and strictly positive values, making it a good candidate for Box-Cox normalization."
            ))

        if yeo_johnson_candidates:
            recs.append(Recommendation(
                rule_id="power_transform_yeo_johnson",
                type=RecommendationType.TRANSFORMATION,
                target_columns=yeo_johnson_candidates,
                description=f"Apply Yeo-Johnson transformation to skewed columns {yeo_johnson_candidates}.",
                suggested_node_type="PowerTransformer",
                suggested_params={"method": "yeo-johnson", "columns": yeo_johnson_candidates},
                confidence=0.7,
                reasoning="Column has high skewness. Yeo-Johnson transformation improves normality and handles negative values."
            ))
            
        return recs
