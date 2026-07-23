"""Heuristic data-cleaning recommendations derived from column profiles."""

from ..schemas import Alert, ColumnProfile, Recommendation
from ._utils import _AnalyzerState


class RecommendationsMixin(_AnalyzerState):
    """Recommendation helpers for :class:`EDAAnalyzer`."""

    def _generate_recommendations(
        self,
        profiles: dict[str, ColumnProfile],
        alerts: list[Alert],
        target_col: str | None,
    ) -> list[Recommendation]:
        recs: list[Recommendation] = []

        for col, profile in profiles.items():
            recs.extend(self._missing_value_recommendations(col, profile))
        for col, profile in profiles.items():
            recs.extend(self._skewness_recommendations(col, profile))
        for col, profile in profiles.items():
            recs.extend(self._cardinality_recommendations(col, profile))
        for col, profile in profiles.items():
            recs.extend(self._constant_column_recommendations(col, profile))
        for col, profile in profiles.items():
            recs.extend(self._id_column_recommendations(col, profile))

        recs.extend(self._clean_dataset_recommendation(recs))
        recs.extend(self._target_balance_recommendations(profiles, target_col))

        return recs

    def _missing_value_recommendations(
        self, col: str, profile: ColumnProfile
    ) -> list[Recommendation]:
        """Recommend dropping or imputing a column based on its missing-value ratio."""
        if profile.missing_percentage > 50:
            return [
                Recommendation(
                    column=col,
                    action="Drop",
                    reason=f"High missing values ({profile.missing_percentage:.1f}%)",
                    suggestion=f"Drop column '{col}' as it contains mostly nulls.",
                )
            ]
        if profile.missing_percentage > 0:
            method = "Median" if profile.dtype == "Numeric" else "Mode"
            return [
                Recommendation(
                    column=col,
                    action="Impute",
                    reason=f"Missing values ({profile.missing_percentage:.1f}%)",
                    suggestion=f"Impute '{col}' using {method}.",
                )
            ]
        return []

    def _skewness_recommendations(self, col: str, profile: ColumnProfile) -> list[Recommendation]:
        """Recommend a transformation for highly skewed numeric columns."""
        if (
            profile.numeric_stats
            and profile.numeric_stats.skewness
            and abs(profile.numeric_stats.skewness) > 1.5
        ):
            return [
                Recommendation(
                    column=col,
                    action="Transform",
                    reason=f"High skewness ({profile.numeric_stats.skewness:.2f})",
                    suggestion=f"Apply Log or Box-Cox transformation to '{col}'.",
                )
            ]
        return []

    def _cardinality_recommendations(
        self, col: str, profile: ColumnProfile
    ) -> list[Recommendation]:
        """Recommend alternative encodings for high-cardinality categorical columns."""
        if (
            profile.categorical_stats
            and profile.dtype == "Categorical"
            and profile.categorical_stats.unique_count > 50
        ):
            return [
                Recommendation(
                    column=col,
                    action="Encode",
                    reason=f"High cardinality ({profile.categorical_stats.unique_count})",
                    suggestion=f"Use Target Encoding or Hashing for '{col}' instead of One-Hot.",
                )
            ]
        return []

    def _constant_column_recommendations(
        self, col: str, profile: ColumnProfile
    ) -> list[Recommendation]:
        """Recommend dropping zero-variance (constant) columns."""
        if profile.is_constant:
            return [
                Recommendation(
                    column=col,
                    action="Drop",
                    reason="Constant value",
                    suggestion=f"Drop '{col}' as it has zero variance.",
                )
            ]
        return []

    def _id_column_recommendations(self, col: str, profile: ColumnProfile) -> list[Recommendation]:
        """Recommend dropping columns that look like unique identifiers."""
        if profile.is_unique and profile.dtype in ["Categorical", "Text", "Numeric"]:
            return [
                Recommendation(
                    column=col,
                    action="Drop",
                    reason="Likely ID column",
                    suggestion=f"Drop '{col}' as it appears to be a unique identifier.",
                )
            ]
        return []

    def _clean_dataset_recommendation(self, recs: list[Recommendation]) -> list[Recommendation]:
        """Provide positive reinforcement when no critical issues were found."""
        critical_issues = [r for r in recs if r.action in ["Drop", "Impute"]]
        if not critical_issues:
            return [
                Recommendation(
                    column=None,
                    action="Keep",
                    reason="Clean Dataset",
                    suggestion="No missing values or constant columns found. Data is ready for modeling!",
                )
            ]
        return []

    def _target_balance_recommendations(
        self, profiles: dict[str, ColumnProfile], target_col: str | None
    ) -> list[Recommendation]:
        """Recommend resampling or note balance for a categorical target column."""
        if not target_col or target_col not in profiles:
            return []
        target_profile = profiles[target_col]
        if target_profile.dtype != "Categorical" or not target_profile.categorical_stats:
            return []

        counts = self._target_class_counts(target_col, target_profile)
        if not counts:
            return []

        min_c = min(counts)
        max_c = max(counts)
        ratio = min_c / max_c if max_c > 0 else 0
        return self._build_balance_recommendation(target_col, ratio)

    @staticmethod
    def _build_balance_recommendation(target_col: str, ratio: float) -> list[Recommendation]:
        """Build the balanced/imbalanced recommendation for the given class ratio, or [] if neither applies."""
        if ratio > 0.8:
            return [
                Recommendation(
                    column=target_col,
                    action="Info",
                    reason="Balanced Target",
                    suggestion=f"Target classes are well balanced (Ratio: {ratio:.2f}).",
                )
            ]
        if ratio < 0.2:
            return [
                Recommendation(
                    column=target_col,
                    action="Resample",
                    reason="Imbalanced Target",
                    suggestion=(
                        f"Target is imbalanced (Ratio: {ratio:.2f}). Consider SMOTE or Class Weights."
                    ),
                )
            ]
        return []

    def _target_class_counts(self, target_col: str, target_profile: ColumnProfile) -> list[int]:
        """Return per-class counts for the target column.

        Uses a full group-by over the target column rather than the
        already-truncated ``categorical_stats.top_k`` (capped to the 10 most
        frequent classes upstream), so classes outside the top 10 are not
        silently ignored when computing the imbalance ratio. Falls back to
        ``top_k`` if the target's cardinality is too high for a full count to
        be meaningful (e.g. an ID-like column mistakenly typed as target).
        """
        cat_stats = target_profile.categorical_stats
        unique_count = cat_stats.unique_count if cat_stats else 0
        # Guard against accidentally high-cardinality "targets" (e.g. an ID
        # column): a full group-by over thousands of distinct values isn't a
        # meaningful class-imbalance signal, so fall back to top_k.
        if unique_count == 0 or unique_count > 1000 or cat_stats is None:
            return [item["count"] for item in cat_stats.top_k] if cat_stats else []

        counts_df = (
            self.lazy_df.select(target_col)
            .drop_nulls()
            .group_by(target_col)
            .len(name="count")
            .collect()
        )
        return counts_df["count"].to_list()  # ty: ignore[not-subscriptable]
