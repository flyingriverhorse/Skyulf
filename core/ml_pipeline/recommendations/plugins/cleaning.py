from typing import List
from ..schemas import AnalysisProfile, Recommendation, RecommendationType, ColumnType
from .base import BaseAdvisorPlugin

class CleaningAdvisor(BaseAdvisorPlugin):
    def analyze(self, profile: AnalysisProfile) -> List[Recommendation]:
        recs = []
        
        high_missing = [] # > 40% missing
        constant_cols = [] # 1 unique value
        
        for col_name, col_profile in profile.columns.items():
            if col_profile.missing_ratio > 0.4:
                high_missing.append(col_name)
            
            if col_profile.unique_count == 1:
                constant_cols.append(col_name)
        
        # 1. High Missing -> Drop
        if high_missing:
            recs.append(Recommendation(
                rule_id="high_missing_drop",
                type=RecommendationType.FEATURE_SELECTION,
                target_columns=high_missing,
                description=f"Columns {high_missing} have >40% missing values.",
                suggested_node_type="DropColumns",
                suggested_params={"columns": high_missing},
                confidence=0.9,
                reasoning="High missing rate usually indicates poor data quality."
            ))
            
        # 2. Constant Columns -> Drop
        if constant_cols:
            recs.append(Recommendation(
                rule_id="constant_drop",
                type=RecommendationType.FEATURE_SELECTION,
                target_columns=constant_cols,
                description=f"Columns {constant_cols} have only one unique value.",
                suggested_node_type="DropColumns",
                suggested_params={"columns": constant_cols},
                confidence=0.95,
                reasoning="Constant columns provide no information to the model."
            ))
            
        # 3. Duplicate Rows -> Drop Duplicates
        if profile.duplicate_row_count > 0:
            recs.append(Recommendation(
                rule_id="duplicate_rows_drop",
                type=RecommendationType.CLEANING,
                target_columns=[], # Applies to whole dataset
                description=f"Dataset contains {profile.duplicate_row_count} duplicate rows.",
                suggested_node_type="DropDuplicates",
                suggested_params={},
                confidence=0.85,
                reasoning="Duplicate rows can bias the model and should usually be removed."
            ))
            
        return recs
