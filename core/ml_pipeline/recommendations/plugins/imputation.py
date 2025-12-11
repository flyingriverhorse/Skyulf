from typing import List
from ..schemas import AnalysisProfile, Recommendation, RecommendationType, ColumnType
from .base import BaseAdvisorPlugin

class ImputationAdvisor(BaseAdvisorPlugin):
    def analyze(self, profile: AnalysisProfile) -> List[Recommendation]:
        recs = []
        
        # Group columns by recommended strategy
        numeric_missing = []
        categorical_missing = []
        
        for col_name, col_profile in profile.columns.items():
            # Only consider columns that are NOT high missing (handled by CleaningAdvisor)
            # But wait, if CleaningAdvisor suggests dropping, we shouldn't suggest imputing.
            # However, plugins run independently. The user chooses which recommendation to apply.
            # So we can still suggest imputation, but maybe with lower confidence if missing is high?
            # Or we just stick to "if missing > 0, suggest imputation".
            # Let's keep it simple: If missing > 0, suggest imputation.
            
            if col_profile.missing_count > 0:
                if col_profile.column_type == ColumnType.NUMERIC:
                    numeric_missing.append(col_name)
                else:
                    categorical_missing.append(col_name)
        
        # 1. Numeric Missing -> Mean/Median Imputation
        if numeric_missing:
            # Ideally we check skewness per column, but for batch recommendation we can group them
            # or create separate recommendations. Let's group for simplicity.
            recs.append(Recommendation(
                rule_id="numeric_imputation",
                type=RecommendationType.IMPUTATION,
                target_columns=numeric_missing,
                description=f"Impute missing values in numeric columns {numeric_missing}.",
                suggested_node_type="SimpleImputer",
                suggested_params={"strategy": "mean", "columns": numeric_missing},
                confidence=0.8,
                reasoning="Numeric columns contain missing values."
            ))
            
        # 3. Categorical Missing -> Most Frequent
        if categorical_missing:
            recs.append(Recommendation(
                rule_id="categorical_imputation",
                type=RecommendationType.IMPUTATION,
                target_columns=categorical_missing,
                description=f"Impute missing values in categorical columns {categorical_missing}.",
                suggested_node_type="SimpleImputer",
                suggested_params={"strategy": "most_frequent", "columns": categorical_missing},
                confidence=0.8,
                reasoning="Categorical columns contain missing values."
            ))
            
        return recs
