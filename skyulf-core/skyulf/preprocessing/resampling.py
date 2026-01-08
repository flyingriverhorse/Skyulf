import logging
from typing import Any, Dict, Tuple, Union

import numpy as np
import pandas as pd
from imblearn.combine import SMOTETomek
from imblearn.over_sampling import ADASYN, SMOTE, SVMSMOTE, BorderlineSMOTE, KMeansSMOTE
from imblearn.under_sampling import (
    EditedNearestNeighbours,
    NearMiss,
    RandomUnderSampler,
    TomekLinks,
)

from ..registry import NodeRegistry
from ..core.meta.decorators import node_meta
from ..utils import pack_pipeline_output, unpack_pipeline_input
from .base import BaseApplier, BaseCalculator
from ..engines import SkyulfDataFrame, get_engine

logger = logging.getLogger(__name__)

# --- Oversampling (SMOTE variants) ---


class OversamplingApplier(BaseApplier):
    def apply(
        self,
        df: Union[pd.DataFrame, SkyulfDataFrame, Tuple[Any, ...], Any],
        params: Dict[str, Any],
    ) -> Union[pd.DataFrame, SkyulfDataFrame, Tuple[Any, ...]]:
        X, y, is_tuple = unpack_pipeline_input(df)
        engine = get_engine(X)
        was_polars = engine.name == "polars"
        target_col = params.get("target_column")

        # Resampling requires y. If not provided in tuple, try to extract from dataframe using target_column
        if y is None:
            if target_col and target_col in X.columns:
                if was_polars:
                    import polars as pl
                    y = X.select(target_col).to_series()
                    X = X.drop(target_col)
                else:
                    y = X[target_col]
                    X = X.drop(columns=[target_col])
            else:
                # Cannot resample without target
                return pack_pipeline_output(X, y, is_tuple)
        
        # Convert to Pandas for imblearn
        if was_polars:
            import polars as pl
            X_pd = X.to_pandas()
            y_pd = y.to_pandas() if hasattr(y, "to_pandas") else y
        else:
            X_pd = X
            y_pd = y

        # Check for non-numeric columns
        non_numeric_cols = X_pd.select_dtypes(exclude=[np.number]).columns
        if len(non_numeric_cols) > 0:
            raise ValueError(
                f"Resampling requires all features to be numeric. Found non-numeric columns: {list(non_numeric_cols)}. "
                "Please use an Encoder node (e.g., OneHotEncoder, OrdinalEncoder) before Resampling."
            )

        method = params.get("method", "smote")
        strategy = params.get("sampling_strategy", "auto")
        random_state = params.get("random_state", 42)
        k_neighbors = params.get("k_neighbors", 5)
        # n_jobs = params.get('n_jobs', -1)

        # Additional params
        m_neighbors = params.get("m_neighbors", 10)
        kind = params.get("kind", "borderline-1")
        out_step = params.get("out_step", 0.5)
        cluster_balance_threshold = params.get("cluster_balance_threshold", 0.1)
        density_exponent = params.get("density_exponent", "auto")

        sampler = None
        if method == "smote":
            sampler = SMOTE(
                sampling_strategy=strategy,
                random_state=random_state,
                k_neighbors=k_neighbors,
            )
        elif method == "adasyn":
            sampler = ADASYN(
                sampling_strategy=strategy,
                random_state=random_state,
                n_neighbors=k_neighbors,
            )
        elif method == "borderline_smote":
            sampler = BorderlineSMOTE(
                sampling_strategy=strategy,
                random_state=random_state,
                k_neighbors=k_neighbors,
                m_neighbors=m_neighbors,
                kind=kind,
            )
        elif method == "svm_smote":
            sampler = SVMSMOTE(
                sampling_strategy=strategy,
                random_state=random_state,
                k_neighbors=k_neighbors,
                m_neighbors=m_neighbors,
                out_step=out_step,
            )
        elif method == "kmeans_smote":
            sampler = KMeansSMOTE(
                sampling_strategy=strategy,
                random_state=random_state,
                k_neighbors=k_neighbors,
                cluster_balance_threshold=cluster_balance_threshold,
                density_exponent=density_exponent,
            )
        elif method == "smote_tomek":
            sampler = SMOTETomek(sampling_strategy=strategy, random_state=random_state)

        if not sampler:
            return pack_pipeline_output(X, y, is_tuple)

        X_res, y_res = sampler.fit_resample(X_pd, y_pd)

        # Ensure we return DataFrame/Series with correct names/columns
        if not isinstance(X_res, pd.DataFrame):
            X_res = pd.DataFrame(X_res, columns=X_pd.columns)
        if not isinstance(y_res, pd.Series):
            y_res = pd.Series(y_res, name=y_pd.name if y_pd is not None else target_col)

        # Convert back if needed
        if was_polars:
            X_res = pl.from_pandas(X_res)
            y_res = pl.from_pandas(y_res)

        return pack_pipeline_output(X_res, y_res, is_tuple)


@NodeRegistry.register("Oversampling", OversamplingApplier)
@node_meta(
    id="Oversampling",
    name="Oversampling",
    category="Preprocessing",
    description="Resample dataset to balance classes by oversampling minority class.",
    params={"method": "smote", "target_column": "target", "sampling_strategy": "auto"}
)
class OversamplingCalculator(BaseCalculator):
    def fit(
        self,
        df: Union[pd.DataFrame, SkyulfDataFrame, Tuple[Any, ...], Any],
        config: Dict[str, Any],
    ) -> Dict[str, Any]:
        # Config: {'method': 'smote', 'target_column': 'target', 'sampling_strategy': 'auto', ...}
        return {
            "type": "oversampling",
            "method": config.get("method", "smote"),
            "target_column": config.get("target_column"),
            "sampling_strategy": config.get("sampling_strategy", "auto"),
            "random_state": config.get("random_state", 42),
            "k_neighbors": config.get("k_neighbors", 5),
            "m_neighbors": config.get("m_neighbors", 10),
            "kind": config.get("kind", "borderline-1"),
            "svm_estimator": config.get("svm_estimator", None),
            "out_step": config.get("out_step", 0.5),
            "kmeans_estimator": config.get("kmeans_estimator", None),
            "cluster_balance_threshold": config.get("cluster_balance_threshold", 0.1),
            "density_exponent": config.get("density_exponent", "auto"),
            "n_jobs": config.get("n_jobs", -1),
        }


# --- Undersampling ---


class UndersamplingApplier(BaseApplier):
    def apply(
        self,
        df: Union[pd.DataFrame, SkyulfDataFrame, Tuple[Any, ...], Any],
        params: Dict[str, Any],
    ) -> Union[pd.DataFrame, SkyulfDataFrame, Tuple[Any, ...]]:
        X, y, is_tuple = unpack_pipeline_input(df)
        engine = get_engine(X)
        was_polars = engine.name == "polars"
        target_col = params.get("target_column")

        # Resampling requires y. If not provided in tuple, try to extract from dataframe using target_column
        if y is None:
            if target_col and target_col in X.columns:
                if was_polars:
                    import polars as pl
                    y = X.select(target_col).to_series()
                    X = X.drop(target_col)
                else:
                    y = X[target_col]
                    X = X.drop(columns=[target_col])
            else:
                # Cannot resample without target
                return pack_pipeline_output(X, y, is_tuple)

        # Convert to Pandas for imblearn
        if was_polars:
            import polars as pl
            X_pd = X.to_pandas()
            y_pd = y.to_pandas() if hasattr(y, "to_pandas") else y
        else:
            X_pd = X
            y_pd = y

        # Check for non-numeric columns
        non_numeric_cols = X_pd.select_dtypes(exclude=[np.number]).columns
        if len(non_numeric_cols) > 0:
            raise ValueError(
                f"Resampling requires all features to be numeric. Found non-numeric columns: {list(non_numeric_cols)}. "
                "Please use an Encoder node (e.g., OneHotEncoder, OrdinalEncoder) before Resampling."
            )

        method = params.get("method", "random_under_sampling")
        strategy = params.get("sampling_strategy", "auto")
        random_state = params.get("random_state", 42)
        replacement = params.get("replacement", False)
        # n_jobs = params.get('n_jobs', -1)

        sampler = None
        if method == "random_under_sampling":
            sampler = RandomUnderSampler(
                sampling_strategy=strategy,
                random_state=random_state,
                replacement=replacement,
            )
        elif method == "nearmiss":
            version = params.get("version", 1)
            sampler = NearMiss(sampling_strategy=strategy, version=version)
        elif method == "tomek_links":
            sampler = TomekLinks(sampling_strategy=strategy)
        elif method == "edited_nearest_neighbours":
            n_neighbors = params.get("n_neighbors", 3)
            kind_sel = params.get("kind_sel", "all")
            sampler = EditedNearestNeighbours(
                sampling_strategy=strategy, n_neighbors=n_neighbors, kind_sel=kind_sel
            )

        if not sampler:
            return pack_pipeline_output(X, y, is_tuple)

        X_res, y_res = sampler.fit_resample(X_pd, y_pd)

        # Ensure we return DataFrame/Series with correct names/columns
        if not isinstance(X_res, pd.DataFrame):
            X_res = pd.DataFrame(X_res, columns=X_pd.columns)
        if not isinstance(y_res, pd.Series):
            y_res = pd.Series(y_res, name=y_pd.name if y_pd is not None else target_col)

        # Convert back if needed
        if was_polars:
            X_res = pl.from_pandas(X_res)
            y_res = pl.from_pandas(y_res)

        return pack_pipeline_output(X_res, y_res, is_tuple)


@NodeRegistry.register("Undersampling", UndersamplingApplier)
@node_meta(
    id="Undersampling",
    name="Undersampling",
    category="Preprocessing",
    description="Resample dataset to balance classes by undersampling majority class.",
    params={"method": "random_under_sampling", "target_column": "target", "sampling_strategy": "auto"}
)
class UndersamplingCalculator(BaseCalculator):
    def fit(
        self,
        df: Union[pd.DataFrame, SkyulfDataFrame, Tuple[Any, ...], Any],
        config: Dict[str, Any],
    ) -> Dict[str, Any]:
        return {
            "type": "undersampling",
            "method": config.get("method", "random_under_sampling"),
            "target_column": config.get("target_column"),
            "sampling_strategy": config.get("sampling_strategy", "auto"),
            "random_state": config.get("random_state", 42),
            "replacement": config.get("replacement", False),
            "version": config.get("version", 1),  # For NearMiss
            "n_neighbors": config.get("n_neighbors", 3),  # For EditedNearestNeighbours
            "kind_sel": config.get("kind_sel", "all"),  # For EditedNearestNeighbours
            "n_jobs": config.get("n_jobs", -1),
        }
