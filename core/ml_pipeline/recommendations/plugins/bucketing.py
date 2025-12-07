from typing import List
from ..schemas import AnalysisProfile, Recommendation, RecommendationType, ColumnType
from .base import BaseAdvisorPlugin

class BucketingAdvisor(BaseAdvisorPlugin):
    def analyze(self, profile: AnalysisProfile) -> List[Recommendation]:
        recs = []
        
        for col_name, col_profile in profile.columns.items():
            if col_profile.column_type == ColumnType.NUMERIC:
                # Check for high cardinality but not continuous? Or maybe skewed?
                # Binning is good for:
                # 1. Skewed data (Quantile binning)
                # 2. Non-linear relationships (though trees handle this, linear models need binning)
                # 3. Outliers (Binning caps them)
                
                if col_profile.skewness is not None and abs(col_profile.skewness) > 1.0:
                     recs.append(Recommendation(
                        rule_id="bucketing_skewed",
                        type=RecommendationType.TRANSFORMATION,
                        target_columns=[col_name],
                        description=f"Bin skewed column '{col_name}' using quantiles.",
                        suggested_node_type="BinningNode",
                        suggested_params={
                            "strategy": "equal_frequency", # Quantile binning
                            "n_bins": 5,
                            "columns": [col_name]
                        },
                        confidence=0.6,
                        reasoning=f"Column '{col_name}' is skewed. Quantile binning can help handle skewness and outliers."
                    ))
                
                # If unique count is somewhat low for a numeric (e.g. 10-50), maybe it's ordinal or should be binned?
                if col_profile.unique_count and 10 < col_profile.unique_count < 50:
                     recs.append(Recommendation(
                        rule_id="bucketing_low_cardinality",
                        type=RecommendationType.TRANSFORMATION,
                        target_columns=[col_name],
                        description=f"Bin column '{col_name}' with few unique values.",
                        suggested_node_type="BinningNode",
                        suggested_params={
                            "strategy": "equal_width",
                            "n_bins": 5,
                            "columns": [col_name]
                        },
                        confidence=0.5,
                        reasoning=f"Column '{col_name}' has relatively few unique numeric values. Binning might reduce noise."
                    ))

        return recs
