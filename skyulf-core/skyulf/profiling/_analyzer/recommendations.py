"""Heuristic data-cleaning recommendations derived from column profiles."""

from ..schemas import Alert, ColumnProfile, Recommendation
from ._utils import _INT_DTYPES, _AnalyzerState

# |skew| above this triggers domain-aware transform advice.
SKEWNESS_TRANSFORM_THRESHOLD = 1.5
# Class-ratio bands: above the upper band the target is balanced, below the
# lower band it is imbalanced enough to recommend resampling.
BALANCED_RATIO_UPPER = 0.8
IMBALANCED_RATIO_LOWER = 0.2


class RecommendationsMixin(_AnalyzerState):
    """Recommendation helpers for :class:`EDAAnalyzer`."""

    def _generate_recommendations(
        self,
        profiles: dict[str, ColumnProfile],
        alerts: list[Alert],
        target_col: str | None,
        task_type: str | None = None,
    ) -> list[Recommendation]:
        """Collect actionable advice before deciding whether a clean message applies."""
        recs: list[Recommendation] = []
        task_type = self._resolve_target_task_type(target_col, task_type)

        for col, profile in profiles.items():
            recs.extend(self._missing_value_recommendations(col, profile))
        recs.extend(self._value_preparation_recommendations(profiles, target_col, task_type))
        for col, profile in profiles.items():
            recs.extend(self._constant_column_recommendations(col, profile))
        for col, profile in profiles.items():
            recs.extend(self._id_column_recommendations(col, profile))

        recs.extend(self._target_balance_recommendations(profiles, target_col, task_type))
        recs.extend(self._clean_dataset_recommendation(recs))

        return recs

    def _value_preparation_recommendations(
        self, profiles: dict[str, ColumnProfile], target_col: str | None, task_type: str | None
    ) -> list[Recommendation]:
        """Keep feature encoding and numeric transforms away from target class labels."""
        recs = []
        for col, profile in profiles.items():
            if col != target_col or task_type != "Classification":
                recs.extend(self._skewness_recommendations(col, profile))
            if col != target_col:
                recs.extend(self._cardinality_recommendations(col, profile))
        return recs

    def _resolve_target_task_type(
        self, target_col: str | None, task_type: str | None
    ) -> str | None:
        """Infer binary integer targets independently of frame size, honoring explicit tasks.

        Only a supplied target gets this override; ordinary feature typing
        and fractional numeric measurements retain their existing semantics.
        """
        if task_type or not target_col or target_col not in self.columns:
            return task_type
        target = self.df[target_col]
        if target.dtype in _INT_DTYPES and target.drop_nulls().n_unique() == 2:
            return "Classification"
        # EDAAnalyzer resolves this Protocol declaration to ColumnMixin's
        # concrete string-returning implementation; Pylint cannot follow that
        # multiple-inheritance path (see the matching RulesMixin exemption).
        semantic_type = self._get_semantic_type(target)  # pylint: disable=assignment-from-no-return
        if semantic_type in ("Categorical", "Boolean"):
            return "Classification"
        return "Regression" if semantic_type == "Numeric" else None

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
        """Select transform advice using the numeric domain and skew direction.

        Log/Box-Cox advice requires positive observations and right skew.
        Yeo-Johnson supports non-positive values and is also the fallback
        when the profile cannot establish a strictly positive domain.
        """
        stats = profile.numeric_stats
        if (
            stats is None
            or stats.skewness is None
            or abs(stats.skewness) <= SKEWNESS_TRANSFORM_THRESHOLD
        ):
            return []

        positive_right_skew = (
            stats.skewness > 0
            and stats.min is not None
            and stats.min > 0
            and not stats.zeros_count
            and not stats.negatives_count
        )
        method = "Log or Box-Cox" if positive_right_skew else "Yeo-Johnson"
        return [
            Recommendation(
                column=col,
                action="Transform",
                reason=f"High skewness ({stats.skewness:.2f})",
                suggestion=f"Consider {method} transformation for '{col}'.",
            )
        ]

    def _cardinality_recommendations(
        self, col: str, profile: ColumnProfile
    ) -> list[Recommendation]:
        """Recommend categorical encoding, conditionally for repeated integer codes."""
        if profile.dtype not in ("Numeric", "Categorical"):
            return []
        unique_count = (
            profile.categorical_stats.unique_count
            if profile.dtype == "Categorical" and profile.categorical_stats
            else self._integer_code_cardinality(col)
        )
        if unique_count > 50:
            suggestion = f"Use Target Encoding or Hashing for '{col}' instead of One-Hot."
            if profile.dtype == "Numeric":
                suggestion = (
                    f"If '{col}' contains category codes, consider Target Encoding or Hashing "
                    "instead of One-Hot. Keep numeric measurements numeric."
                )
            return [
                Recommendation(
                    column=col,
                    action="Encode",
                    reason=f"High cardinality ({unique_count})",
                    suggestion=suggestion,
                )
            ]
        return []

    def _integer_code_cardinality(self, col: str) -> int:
        """Count repeated integer values without asserting they are categorical."""
        if col not in self.columns or not self.df[col].dtype.is_integer():
            return 0
        observed = self.df[col].drop_nulls()
        unique_count = observed.n_unique()
        return unique_count if unique_count < len(observed) else 0

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
        """Report no recommended preparation only after all actionable advice is known."""
        if not any(r.action in {"Drop", "Impute", "Transform", "Encode", "Resample"} for r in recs):
            return [
                Recommendation(
                    column=None,
                    action="Keep",
                    reason="Clean Dataset",
                    suggestion="No data preparation changes were recommended by these checks.",
                )
            ]
        return []

    def _target_balance_recommendations(
        self,
        profiles: dict[str, ColumnProfile],
        target_col: str | None,
        task_type: str | None = None,
    ) -> list[Recommendation]:
        """Recommend resampling or note balance for classification target labels."""
        if not target_col or target_col not in profiles or task_type == "Regression":
            return []
        target_profile = profiles[target_col]
        if task_type != "Classification" and target_profile.dtype not in ("Categorical", "Boolean"):
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
        if ratio > BALANCED_RATIO_UPPER:
            return [
                Recommendation(
                    column=target_col,
                    action="Info",
                    reason="Balanced Target",
                    suggestion=f"Target classes are well balanced (Ratio: {ratio:.2f}).",
                )
            ]
        if ratio < IMBALANCED_RATIO_LOWER:
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
        unique_count = (
            cat_stats.unique_count if cat_stats else self.df[target_col].drop_nulls().n_unique()
        )
        # Guard against accidentally high-cardinality "targets" (e.g. an ID
        # column): a full group-by over thousands of distinct values isn't a
        # meaningful class-imbalance signal, so fall back to top_k.
        if unique_count == 0 or unique_count > 1000:
            return [item["count"] for item in cat_stats.top_k] if cat_stats else []

        counts_df = (
            self.lazy_df.select(target_col)
            .rename({target_col: "value"})
            .drop_nulls()
            .group_by("value")
            .len(name="count")
            .collect()
        )
        return counts_df["count"].to_list()  # ty: ignore[not-subscriptable]
