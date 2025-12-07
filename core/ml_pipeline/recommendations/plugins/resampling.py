from typing import List
from ..schemas import AnalysisProfile, Recommendation, RecommendationType, ColumnType
from .base import BaseAdvisorPlugin

class ResamplingAdvisor(BaseAdvisorPlugin):
    def analyze(self, profile: AnalysisProfile) -> List[Recommendation]:
        recs = []
        
        # We need to identify the target column to check for imbalance.
        # However, AnalysisProfile currently doesn't explicitly tag the target column.
        # We might need to infer it or update the profile schema.
        # For now, let's assume we can't reliably detect the target without more info.
        # But if we see a categorical column with very few unique values (2-10) and high imbalance, we can flag it.
        
        for col_name, col_profile in profile.columns.items():
            if col_profile.column_type == ColumnType.CATEGORICAL or col_profile.column_type == ColumnType.BOOLEAN:
                # Check for class imbalance if we have value counts
                if col_profile.top_values and col_profile.unique_count and 2 <= col_profile.unique_count <= 10:
                    total_count = sum(col_profile.top_values.values())
                    if total_count > 0:
                        counts = list(col_profile.top_values.values())
                        min_count = min(counts)
                        max_count = max(counts)
                        
                        # Imbalance ratio
                        ratio = max_count / min_count if min_count > 0 else float('inf')
                        
                        if ratio > 5.0: # Heuristic threshold for imbalance
                            recs.append(Recommendation(
                                rule_id="resampling_imbalance",
                                type=RecommendationType.RESAMPLING,
                                target_columns=[col_name],
                                description=f"Potential class imbalance detected in '{col_name}' (Ratio {ratio:.1f}:1). Consider resampling.",
                                suggested_node_type="ResamplingNode",
                                suggested_params={
                                    "target_column": col_name,
                                    "method": "smote", # Default suggestion
                                    "sampling_strategy": "auto"
                                },
                                confidence=0.6,
                                reasoning=f"The distribution of classes in '{col_name}' is highly imbalanced, which may bias the model."
                            ))
        
        return recs
