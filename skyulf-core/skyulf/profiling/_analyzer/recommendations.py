"""Heuristic data-cleaning recommendations derived from column profiles."""

from typing import Dict, List, Optional

from ..schemas import Alert, ColumnProfile, Recommendation
from ._utils import _AnalyzerState


class RecommendationsMixin(_AnalyzerState):
    """Recommendation helpers for :class:`EDAAnalyzer`."""

    def _generate_recommendations(
        self,
        profiles: Dict[str, ColumnProfile],
        alerts: List[Alert],
        target_col: Optional[str],
    ) -> List[Recommendation]:
        recs: List[Recommendation] = []

        for col, profile in profiles.items():
            if profile.missing_percentage > 50:
                recs.append(
                    Recommendation(
                        column=col,
                        action="Drop",
                        reason=f"High missing values ({profile.missing_percentage:.1f}%)",
                        suggestion=f"Drop column '{col}' as it contains mostly nulls.",
                    )
                )
            elif profile.missing_percentage > 0:
                method = "Median" if profile.dtype == "Numeric" else "Mode"
                recs.append(
                    Recommendation(
                        column=col,
                        action="Impute",
                        reason=f"Missing values ({profile.missing_percentage:.1f}%)",
                        suggestion=f"Impute '{col}' using {method}.",
                    )
                )

        for col, profile in profiles.items():
            if (
                profile.numeric_stats
                and profile.numeric_stats.skewness
                and abs(profile.numeric_stats.skewness) > 1.5
            ):
                recs.append(
                    Recommendation(
                        column=col,
                        action="Transform",
                        reason=f"High skewness ({profile.numeric_stats.skewness:.2f})",
                        suggestion=f"Apply Log or Box-Cox transformation to '{col}'.",
                    )
                )

        for col, profile in profiles.items():
            if (
                profile.categorical_stats
                and profile.dtype == "Categorical"
                and profile.categorical_stats.unique_count > 50
            ):
                recs.append(
                    Recommendation(
                        column=col,
                        action="Encode",
                        reason=f"High cardinality ({profile.categorical_stats.unique_count})",
                        suggestion=f"Use Target Encoding or Hashing for '{col}' instead of One-Hot.",
                    )
                )

        for col, profile in profiles.items():
            if profile.is_constant:
                recs.append(
                    Recommendation(
                        column=col,
                        action="Drop",
                        reason="Constant value",
                        suggestion=f"Drop '{col}' as it has zero variance.",
                    )
                )

        for col, profile in profiles.items():
            if profile.is_unique and profile.dtype in [
                "Categorical",
                "Text",
                "Numeric",
            ]:
                recs.append(
                    Recommendation(
                        column=col,
                        action="Drop",
                        reason="Likely ID column",
                        suggestion=f"Drop '{col}' as it appears to be a unique identifier.",
                    )
                )

        # Positive reinforcement when nothing critical surfaced.
        critical_issues = [r for r in recs if r.action in ["Drop", "Impute"]]
        if not critical_issues:
            recs.append(
                Recommendation(
                    column=None,
                    action="Keep",
                    reason="Clean Dataset",
                    suggestion="No missing values or constant columns found. Data is ready for modeling!",
                )
            )

        if target_col and target_col in profiles:
            target_profile = profiles[target_col]
            if (
                target_profile.dtype == "Categorical"
                and target_profile.categorical_stats
            ):
                counts = [
                    item["count"] for item in target_profile.categorical_stats.top_k
                ]
                if counts:
                    min_c = min(counts)
                    max_c = max(counts)
                    ratio = min_c / max_c if max_c > 0 else 0
                    if ratio > 0.8:
                        recs.append(
                            Recommendation(
                                column=target_col,
                                action="Info",
                                reason="Balanced Target",
                                suggestion=f"Target classes are well balanced (Ratio: {ratio:.2f}).",
                            )
                        )
                    elif ratio < 0.2:
                        recs.append(
                            Recommendation(
                                column=target_col,
                                action="Resample",
                                reason="Imbalanced Target",
                                suggestion=(
                                    f"Target is imbalanced (Ratio: {ratio:.2f}). "
                                    "Consider SMOTE or Class Weights."
                                ),
                            )
                        )

        return recs
