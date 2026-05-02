"""Target-feature association: correlations, eta², box-plot interactions."""

from typing import Dict, List

import numpy as np
import polars as pl

from ..schemas import BoxPlotStats, CategoryBoxPlot, TargetInteraction
from ._utils import SCIPY_AVAILABLE, _AnalyzerState, _collect


class TargetMixin(_AnalyzerState):
    """Target-relationship helpers for :class:`EDAAnalyzer`."""

    def _calculate_target_correlations(
        self, target_col: str, numeric_cols: List[str]
    ) -> Dict[str, float]:
        try:
            features = [c for c in numeric_cols if c != target_col]
            if not features:
                return {}

            exprs = [pl.corr(col, target_col).alias(col) for col in features]

            import warnings

            # corrcoef on constant columns emits a divide-by-zero RuntimeWarning.
            with warnings.catch_warnings():
                warnings.simplefilter("ignore", RuntimeWarning)
                result = _collect(self.lazy_df.select(exprs))  # type: ignore[attr-defined]

            corrs = {}
            for col in features:
                val = result[col][0]
                if val is not None and not np.isnan(val):
                    corrs[col] = float(val)

            return dict(
                sorted(corrs.items(), key=lambda item: abs(item[1]), reverse=True)
            )

        except Exception as e:
            print(f"Error calculating target correlations: {e}")
            return {}

    def _calculate_categorical_target_associations(
        self, target_col: str, numeric_cols: List[str]
    ) -> Dict[str, float]:
        """Correlation Ratio (η) between a categorical target and numeric features.

        η² = SS_between / SS_total. We return η so the magnitude is comparable
        to a Pearson |r|.
        """
        try:
            associations = {}
            features = [c for c in numeric_cols if c != target_col]

            for col in features:
                global_mean = self.df[col].mean()  # type: ignore[attr-defined]
                ss_total = self.df.select(  # type: ignore[attr-defined]
                    ((pl.col(col) - global_mean) ** 2).sum()
                ).item()

                if ss_total == 0:
                    associations[col] = 0.0
                    continue

                groups = self.df.group_by(target_col).agg(  # type: ignore[attr-defined]
                    [pl.len().alias("n"), pl.col(col).mean().alias("mean")]
                )

                ss_between = 0.0
                for row in groups.iter_rows(named=True):
                    n = row["n"]
                    mean_group = row["mean"]
                    if mean_group is not None:
                        ss_between += n * ((mean_group - global_mean) ** 2)

                eta_squared = ss_between / ss_total
                associations[col] = float(np.sqrt(eta_squared))

            return dict(
                sorted(associations.items(), key=lambda item: item[1], reverse=True)
            )

        except Exception as e:
            print(f"Error calculating categorical target associations: {e}")
            return {}

    def _calculate_target_interactions(
        self, target_col: str, features: List[str], is_target_numeric: bool
    ) -> List[TargetInteraction]:
        """Per-feature box-plot stats vs target, plus ANOVA p-value when SciPy is available."""
        interactions = []
        try:
            features_to_process = features[:20]

            for feature in features_to_process:
                if is_target_numeric:
                    group_col = feature
                    value_col = target_col
                else:
                    group_col = target_col
                    value_col = feature

                # Skip high-cardinality grouping — not useful as a box plot.
                if self.df[group_col].n_unique() > 20:  # type: ignore[attr-defined]
                    continue

                stats_df = _collect(
                    self.lazy_df.group_by(group_col).agg(  # type: ignore[attr-defined]
                        [
                            pl.col(value_col)
                            .cast(pl.Float64, strict=False)
                            .min()
                            .alias("min"),
                            pl.col(value_col)
                            .cast(pl.Float64, strict=False)
                            .quantile(0.25)
                            .alias("q1"),
                            pl.col(value_col)
                            .cast(pl.Float64, strict=False)
                            .median()
                            .alias("median"),
                            pl.col(value_col)
                            .cast(pl.Float64, strict=False)
                            .quantile(0.75)
                            .alias("q3"),
                            pl.col(value_col)
                            .cast(pl.Float64, strict=False)
                            .max()
                            .alias("max"),
                        ]
                    )
                )

                category_plots = []
                for row in stats_df.iter_rows(named=True):
                    if row[group_col] is None or row["min"] is None:
                        continue
                    category_plots.append(
                        CategoryBoxPlot(
                            name=str(row[group_col]),
                            stats=BoxPlotStats(
                                min=float(row["min"]),
                                q1=float(row["q1"]),
                                median=float(row["median"]),
                                q3=float(row["q3"]),
                                max=float(row["max"]),
                            ),
                        )
                    )

                p_value = None
                if SCIPY_AVAILABLE and len(category_plots) > 1:
                    try:
                        from scipy.stats import f_oneway

                        anova_data = _collect(
                            self.lazy_df.select(  # type: ignore[attr-defined]
                                [pl.col(group_col), pl.col(value_col)]
                            )
                            .group_by(group_col)
                            .agg(pl.col(value_col))
                        )

                        groups_data = []
                        for row in anova_data.iter_rows(named=True):
                            if (
                                row[group_col] is not None
                                and row[value_col] is not None
                            ):
                                vals = [v for v in row[value_col] if v is not None]
                                if len(vals) > 1:
                                    groups_data.append(vals)

                        if len(groups_data) > 1:
                            _f_stat, p_val = f_oneway(*groups_data)
                            if not np.isnan(p_val):
                                p_value = float(p_val)
                    except Exception as e:
                        print(f"ANOVA failed for {feature}: {e}")

                if category_plots:
                    interactions.append(
                        TargetInteraction(
                            feature=feature,
                            plot_type="boxplot",
                            data=category_plots,
                            p_value=p_value,
                        )
                    )

            return interactions

        except Exception as e:
            print(f"Error calculating target interactions: {e}")
            return []
