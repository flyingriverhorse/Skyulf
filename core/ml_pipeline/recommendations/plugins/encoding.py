from typing import List
from ..schemas import AnalysisProfile, Recommendation, RecommendationType, ColumnType
from .base import BaseAdvisorPlugin

class EncodingAdvisor(BaseAdvisorPlugin):
    def analyze(self, profile: AnalysisProfile) -> List[Recommendation]:
        recs = []
        
        one_hot_candidates = []
        ordinal_candidates = []
        drop_candidates = []
        
        for col_name, col_profile in profile.columns.items():
            if col_profile.column_type in [ColumnType.CATEGORICAL, ColumnType.TEXT]:
                unique_count = col_profile.unique_count
                row_count = profile.row_count
                unique_ratio = unique_count / row_count if row_count > 0 else 0
                avg_len = col_profile.avg_text_length or 0
                
                # 1. High Cardinality / ID-like -> Drop
                if unique_ratio > 0.9 and unique_count > 50:
                    drop_candidates.append(col_name)
                    continue
                    
                # 2. Free Text -> Drop (for now, until we have NLP)
                if avg_len > 50:
                    drop_candidates.append(col_name)
                    continue
                    
                # 3. Low Cardinality -> OneHot
                if unique_count <= 10:
                    one_hot_candidates.append(col_name)
                else:
                    # 4. Medium Cardinality -> Ordinal (or Label)
                    ordinal_candidates.append(col_name)
        
        if drop_candidates:
            recs.append(Recommendation(
                rule_id="high_cardinality_drop",
                type=RecommendationType.FEATURE_SELECTION,
                target_columns=drop_candidates,
                description=f"Columns {drop_candidates} have high cardinality or are free text.",
                suggested_node_type="DropColumns",
                suggested_params={"columns": drop_candidates},
                confidence=0.85,
                reasoning="High cardinality or free text columns require specialized processing or should be dropped."
            ))
            
        if one_hot_candidates:
            recs.append(Recommendation(
                rule_id="one_hot_encoding",
                type=RecommendationType.ENCODING,
                target_columns=one_hot_candidates,
                description=f"One-Hot Encode categorical columns {one_hot_candidates}.",
                suggested_node_type="OneHotEncoder",
                suggested_params={"columns": one_hot_candidates},
                confidence=0.9,
                reasoning="Low cardinality categorical features are best handled with One-Hot Encoding."
            ))
            
        if ordinal_candidates:
            recs.append(Recommendation(
                rule_id="ordinal_encoding",
                type=RecommendationType.ENCODING,
                target_columns=ordinal_candidates,
                description=f"Ordinal Encode categorical columns {ordinal_candidates}.",
                suggested_node_type="OrdinalEncoder",
                suggested_params={"columns": ordinal_candidates},
                confidence=0.8,
                reasoning="Medium cardinality categorical features are better handled with Ordinal Encoding to avoid dimensionality explosion."
            ))
            
        return recs
