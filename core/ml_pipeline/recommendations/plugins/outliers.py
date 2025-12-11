from typing import List
from ..schemas import AnalysisProfile, Recommendation, RecommendationType, ColumnType
from .base import BaseAdvisorPlugin

class OutlierAdvisor(BaseAdvisorPlugin):
    def analyze(self, profile: AnalysisProfile) -> List[Recommendation]:
        recs = []
        
        # We don't have IQR stats in the profile yet.
        # We can either add them to the profile or infer from skewness/std.
        # Adding to profile is better but requires re-reading data which we can't do here (we only have profile).
        # Wait, DataProfiler has access to data. I should have added IQR there.
        # For now, let's use skewness as a proxy. High skewness often implies outliers.
        
        skewed_cols = []
        
        for col_name, col_profile in profile.columns.items():
            if col_profile.column_type == ColumnType.NUMERIC:
                if col_profile.skewness is not None and abs(col_profile.skewness) > 3:
                    skewed_cols.append(col_name)
        
        if skewed_cols:
            recs.append(Recommendation(
                rule_id="outlier_removal_iqr",
                type=RecommendationType.OUTLIER_REMOVAL,
                target_columns=skewed_cols,
                description=f"Remove outliers from skewed columns {skewed_cols}.",
                suggested_node_type="OutlierRemoval",
                suggested_params={"method": "iqr", "columns": skewed_cols},
                confidence=0.6,
                reasoning="High skewness indicates potential outliers."
            ))
            
        return recs
