"""Multivariate analyses: PCA, KMeans clustering, Isolation Forest outliers."""

from typing import Any, List, Optional, Tuple, cast

import numpy as np
import polars as pl

from ..schemas import (
    ClusteringAnalysis,
    ClusteringPoint,
    ClusterStats,
    OutlierAnalysis,
    OutlierPoint,
    PCAComponent,
    PCAPoint,
)
from ._utils import SKLEARN_AVAILABLE, _AnalyzerState


class MultivariateMixin(_AnalyzerState):
    """Multivariate helpers for :class:`EDAAnalyzer`."""

    def _prepare_matrix_sample(
        self,
        numeric_cols: List[str],
        target_col: Optional[str] = None,
        limit: int = 5000,
    ) -> Tuple[Optional[np.ndarray], Optional[pl.DataFrame], Optional[Any]]:
        """Sample → impute (mean) → scale → return ``(X_scaled, sample_df, scaler)``.

        ``seed=42`` is hard-coded so PCA and Clustering see the same subset
        when called separately on the same analyzer instance.
        """
        if not SKLEARN_AVAILABLE:
            return None, None, None
        try:
            from sklearn.impute import SimpleImputer
            from sklearn.preprocessing import StandardScaler

            cols_to_fetch = list(numeric_cols)
            if (
                target_col
                and target_col in self.columns  # type: ignore[attr-defined]
                and target_col not in cols_to_fetch
            ):
                cols_to_fetch.append(target_col)

            if self.row_count > limit:  # type: ignore[attr-defined]
                sample_df = self.df.select(cols_to_fetch).sample(  # type: ignore[attr-defined]
                    n=limit, with_replacement=False, seed=42
                )
            else:
                sample_df = self.df.select(cols_to_fetch)  # type: ignore[attr-defined]

            if sample_df.height < 5:
                return None, None, None

            X_df = sample_df.select(numeric_cols)

            try:
                # Polars fill_null is faster than sklearn's imputer; fall back if it errors.
                # Trailing fill_null(0) covers all-null columns where mean is also null.
                X_df = X_df.fill_null(strategy="mean").fill_null(0)
                X = X_df.to_numpy()
                if not np.isfinite(X).all():
                    X = np.nan_to_num(X, nan=0.0, posinf=0.0, neginf=0.0)
            except Exception:
                X = X_df.to_pandas().values
                imputer = SimpleImputer(strategy="mean")
                X = imputer.fit_transform(X)
                X = np.nan_to_num(X, nan=0.0)

            scaler = StandardScaler()
            X_scaled = scaler.fit_transform(X)

            return X_scaled, sample_df, scaler

        except Exception as e:
            print(f"Error preparing matrix sample: {e}")
            return None, None, None

    def _calculate_pca(
        self, numeric_cols: List[str], target_col: Optional[str] = None
    ) -> Tuple[Optional[List[PCAPoint]], Optional[List[PCAComponent]]]:
        """3-component PCA projection + per-component top loadings."""
        try:
            from sklearn.decomposition import PCA

            X_scaled, sample_df, _ = self._prepare_matrix_sample(
                numeric_cols, target_col=target_col, limit=5000
            )

            if X_scaled is None or sample_df is None:
                return None, None

            pca = PCA(n_components=3)
            X_pca = pca.fit_transform(X_scaled)

            components_list = []
            if hasattr(pca, "components_"):
                feature_names = numeric_cols
                for i, comp in enumerate(pca.components_):
                    weights = {
                        feature_names[j]: float(comp[j])
                        for j in range(len(feature_names))
                    }
                    top_features = dict(
                        sorted(
                            weights.items(), key=lambda item: abs(item[1]), reverse=True
                        )[:5]
                    )
                    components_list.append(
                        PCAComponent(
                            component=f"PC{i+1}",
                            explained_variance_ratio=float(
                                pca.explained_variance_ratio_[i]
                            ),
                            top_features=top_features,
                        )
                    )

            X_pca = np.asarray(X_pca)
            if len(X_pca.shape) == 1:
                X_pca = X_pca.reshape(-1, 1)
            # Pad if PCA returned fewer components (e.g. low-rank input).
            if X_pca.shape[1] < 3:
                padding = np.zeros((X_pca.shape[0], 3 - X_pca.shape[1]))
                X_pca = np.hstack([X_pca, padding])

            labels = None
            if target_col and target_col in self.columns:  # type: ignore[attr-defined]
                labels = sample_df[target_col].to_list()

            points = []
            for i in range(len(X_pca)):
                label_val = str(labels[i]) if labels else None
                points.append(
                    PCAPoint(
                        x=float(X_pca[i, 0]),
                        y=float(X_pca[i, 1]),
                        z=float(X_pca[i, 2]) if X_pca.shape[1] > 2 else None,
                        label=label_val,
                    )
                )

            return points, components_list

        except Exception as e:
            print(f"Error calculating PCA: {e}")
            return None, None

    def _perform_clustering(
        self, numeric_cols: List[str], target_col: Optional[str] = None
    ) -> Optional[ClusteringAnalysis]:
        """KMeans (k=3) post-hoc segmentation, projected to 2D via PCA for plotting."""
        try:
            from sklearn.cluster import KMeans
            from sklearn.decomposition import PCA

            X_scaled, sample_df, scaler = self._prepare_matrix_sample(
                numeric_cols, target_col=target_col, limit=5000
            )

            if X_scaled is None or sample_df is None or scaler is None:
                return None

            # k=3 for generic Low/Medium/High discovery in EDA.
            n_clusters = 3
            kmeans = KMeans(n_clusters=n_clusters, random_state=42, n_init=10)
            labels = kmeans.fit_predict(X_scaled)

            clusters_stats = []
            unique_labels, counts = np.unique(labels, return_counts=True)
            total_points = len(labels)

            centers_scaled = kmeans.cluster_centers_
            centers_original = scaler.inverse_transform(centers_scaled)

            feature_names = numeric_cols
            for i, label in enumerate(unique_labels):
                center_dict = {
                    col: float(val)
                    for col, val in zip(feature_names, centers_original[i])
                }
                clusters_stats.append(
                    ClusterStats(
                        cluster_id=int(label),
                        size=int(counts[i]),
                        percentage=float(counts[i] / total_points * 100),
                        center=center_dict,
                    )
                )

            pca = PCA(n_components=2)
            X_pca = pca.fit_transform(X_scaled)
            X_pca = np.asarray(X_pca)
            if len(X_pca.shape) == 1:
                X_pca = X_pca.reshape(-1, 1)
            if X_pca.shape[1] < 2:
                padding = np.zeros((X_pca.shape[0], 2 - X_pca.shape[1]))
                X_pca = np.hstack([X_pca, padding])

            original_labels = None
            if target_col and target_col in self.columns:  # type: ignore[attr-defined]
                original_labels = sample_df[target_col].to_list()

            points = []
            for i in range(len(X_pca)):
                label_val = str(original_labels[i]) if original_labels else None
                points.append(
                    ClusteringPoint(
                        x=float(X_pca[i, 0]),
                        y=float(X_pca[i, 1]),
                        cluster=int(labels[i]),
                        label=label_val,
                    )
                )

            return ClusteringAnalysis(
                method="KMeans",
                n_clusters=n_clusters,
                inertia=float(cast(Any, kmeans.inertia_)),
                clusters=clusters_stats,
                points=points,
            )

        except Exception as e:
            print(f"Error in clustering analysis: {e}")
            return None

    def _detect_outliers(self, numeric_cols: List[str]) -> Optional[OutlierAnalysis]:
        """Isolation-Forest outlier detection with per-feature deviation explanations."""
        try:
            from sklearn.ensemble import IsolationForest
            from sklearn.impute import SimpleImputer

            limit = 50000
            df_numeric = self.df.select(numeric_cols).head(limit)  # type: ignore[attr-defined]

            X = df_numeric.to_pandas().values
            imputer = SimpleImputer(strategy="mean")
            X = imputer.fit_transform(X)

            clf = IsolationForest(random_state=42, contamination=0.05, n_jobs=-1)
            clf.fit(X)

            preds = clf.predict(X)
            scores = clf.decision_function(X)  # lower = more anomalous

            outlier_indices = np.where(preds == -1)[0]
            total_outliers = len(outlier_indices)
            if total_outliers == 0:
                return None

            scored_indices = list(zip(range(len(scores)), scores))
            scored_indices.sort(key=lambda x: x[1])

            top_k = 20
            top_outliers = []
            medians = df_numeric.median().row(0, named=True)

            for idx, score in scored_indices[:top_k]:
                if preds[idx] != -1:
                    continue

                row_values = df_numeric.row(idx, named=True)

                explanation = [
                    {
                        "feature": col,
                        "value": val,
                        "median": medians.get(col, 0),
                        "diff_pct": (
                            abs((val - medians.get(col, 0)) / medians.get(col, 1)) * 100
                            if medians.get(col, 0) != 0
                            else 0
                        ),
                    }
                    for col, val in row_values.items()
                    if val is not None and medians.get(col) is not None
                ]
                # Only surface deviations large enough to be interpretable.
                explanation = [
                    e for e in explanation if cast(float, e["diff_pct"]) > 50
                ]
                explanation.sort(key=lambda x: x["diff_pct"], reverse=True)

                top_outliers.append(
                    OutlierPoint(
                        index=int(idx),
                        values=row_values,
                        score=float(score),
                        explanation=explanation[:3],
                    )
                )

            return OutlierAnalysis(
                method="IsolationForest",
                total_outliers=total_outliers,
                outlier_percentage=(total_outliers / len(X)) * 100,
                top_outliers=top_outliers,
            )

        except Exception as e:
            print(f"Error in outlier detection: {e}")
            return None
