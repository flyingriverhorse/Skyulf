from typing import List
from ..schemas import AnalysisProfile, Recommendation, RecommendationType, ColumnType
from .base import BaseAdvisorPlugin

class FeatureGenerationAdvisor(BaseAdvisorPlugin):
    def analyze(self, profile: AnalysisProfile) -> List[Recommendation]:
        recs = []
        
        # 1. Date Extraction Recommendation
        datetime_cols = [
            name for name, p in profile.columns.items() 
            if p.column_type == ColumnType.DATETIME
        ]
        
        if datetime_cols:
            recs.append(Recommendation(
                rule_id="feature_gen_date_extract",
                type=RecommendationType.FEATURE_GENERATION,
                target_columns=datetime_cols,
                description=f"Extract date features (year, month, day, etc.) from {datetime_cols}.",
                suggested_node_type="FeatureGenerationNode",
                suggested_params={
                    "operations": [
                        {
                            "operation_type": "datetime_extract",
                            "input_columns": [col],
                            "datetime_features": ["year", "month", "day", "dayofweek", "hour"],
                            "output_prefix": col
                        } for col in datetime_cols
                    ]
                },
                confidence=0.8,
                reasoning="Datetime columns are often more useful when broken down into components like year, month, day, etc."
            ))

        # 2. Interaction/Ratio Recommendation (Heuristic)
        # If we have two numeric columns with similar ranges or names, maybe a ratio is useful?
        # This is harder to guess without domain knowledge.
        # But we can suggest it generally if there are many numeric columns.
        numeric_cols = [
            name for name, p in profile.columns.items() 
            if p.column_type == ColumnType.NUMERIC
        ]
        
        if len(numeric_cols) >= 2:
             recs.append(Recommendation(
                rule_id="feature_gen_interactions",
                type=RecommendationType.FEATURE_GENERATION,
                target_columns=numeric_cols[:2], # Just hint at the first few
                description="Consider creating interaction features (ratios, sums, products) between numeric columns.",
                suggested_node_type="FeatureGenerationNode",
                suggested_params={
                    "operations": [
                        {
                            "operation_type": "arithmetic",
                            "method": "divide", # Ratio is common
                            "input_columns": [numeric_cols[0]],
                            "secondary_columns": [numeric_cols[1]],
                            "output_column": f"{numeric_cols[0]}_div_{numeric_cols[1]}"
                        }
                    ]
                },
                confidence=0.3, # Low confidence as we don't know if it makes sense
                reasoning="Interactions between features can sometimes capture relationships that individual features miss."
            ))

        return recs
