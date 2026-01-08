import logging
from typing import Any, Dict, Tuple, Union

import numpy as np
import pandas as pd
from sklearn.covariance import EllipticEnvelope

from ..utils import (
    detect_numeric_columns,
    pack_pipeline_output,
    resolve_columns,
    unpack_pipeline_input,
)
from .base import BaseApplier, BaseCalculator
from ..core.meta.decorators import node_meta
from ..registry import NodeRegistry
from ..engines import SkyulfDataFrame, get_engine
from ..engines.pandas_engine import SkyulfPandasWrapper

logger = logging.getLogger(__name__)

# --- IQR Filter ---


class IQRApplier(BaseApplier):
    def apply(
        self,
        df: Union[pd.DataFrame, SkyulfDataFrame, Tuple[Any, ...]],
        params: Dict[str, Any],
    ) -> Union[pd.DataFrame, SkyulfDataFrame, Tuple[Any, ...]]:
        X, y, is_tuple = unpack_pipeline_input(df)
        engine = get_engine(X)

        bounds = params.get("bounds", {})
        if not bounds:
            return pack_pipeline_output(X, y, is_tuple)

        # Polars Path
        if engine.name == "polars":
            import polars as pl

            X_pl: Any = X
            
            mask = pl.lit(True)
            
            for col, bound in bounds.items():
                if col not in X_pl.columns:
                    continue
                
                lower = bound["lower"]
                upper = bound["upper"]
                
                # Keep values within bounds or Null
                col_mask = (pl.col(col) >= lower) & (pl.col(col) <= upper)
                col_mask = col_mask | pl.col(col).is_null()
                
                mask = mask & col_mask
            
            # Evaluate mask to get a boolean Series for filtering both X and y
            mask_series = X_pl.select(mask.alias("mask")).get_column("mask")
            X_filtered = X_pl.filter(mask_series)
            
            y_filtered = None
            if y is not None:
                if isinstance(y, (pl.Series, pl.DataFrame)):
                    y_filtered = y.filter(mask_series)
                else:
                    y_filtered = y
                
            return pack_pipeline_output(X_filtered, y_filtered, is_tuple)

        # Pandas Path
        X_pd: Any = X.to_pandas() if hasattr(X, "to_pandas") else X
        
        # Determine index from the converted or original DF
        mask = pd.Series(True, index=X_pd.index)

        for col, bound in bounds.items():
            if col not in X_pd.columns:
                continue

            lower = bound["lower"]
            upper = bound["upper"]

            series = pd.to_numeric(X_pd[col], errors="coerce")

            # Keep values within bounds or NaN
            col_mask = (series >= lower) & (series <= upper)
            col_mask = col_mask | series.isna()

            mask = mask & col_mask

        X_filtered = X_pd[mask]

        if y is not None:
            y_pd: Any = y
            y_filtered = y_pd[mask]
            return pack_pipeline_output(X_filtered, y_filtered, is_tuple)

        return pack_pipeline_output(X_filtered, y, is_tuple)


@NodeRegistry.register("IQR", IQRApplier)
@node_meta(
    id="IQR",
    name="IQR Outlier Removal",
    category="Preprocessing",
    description="Remove outliers using Interquartile Range.",
    params={"factor": 1.5, "columns": []}
)
class IQRCalculator(BaseCalculator):
    def fit(
        self,
        df: Union[pd.DataFrame, SkyulfDataFrame, Tuple[Any, ...]],
        config: Dict[str, Any],
    ) -> Dict[str, Any]:
        X, _, _ = unpack_pipeline_input(df)
        
        engine = get_engine(X)
        X_pd: Any = X.to_pandas() if hasattr(X, "to_pandas") else X
            
        # Config: {'multiplier': 1.5, 'columns': [...]}
        multiplier = config.get("multiplier", 1.5)

        cols = resolve_columns(X_pd, config, detect_numeric_columns)

        if not cols:
            return {}

        bounds = {}
        warnings = []
        for col in cols:
            series = pd.to_numeric(X_pd[col], errors="coerce").dropna()
            if series.empty:
                warnings.append(f"Column '{col}': Empty or non-numeric")
                continue

            q1 = series.quantile(0.25)
            q3 = series.quantile(0.75)
            iqr = q3 - q1

            lower = q1 - (multiplier * iqr)
            upper = q3 + (multiplier * iqr)

            bounds[col] = {"lower": lower, "upper": upper}

        return {
            "type": "iqr",
            "bounds": bounds,
            "multiplier": multiplier,
            "warnings": warnings,
        }


# --- Z-Score Filter ---


class ZScoreApplier(BaseApplier):
    def apply(
        self,
        df: Union[pd.DataFrame, SkyulfDataFrame, Tuple[Any, ...]],
        params: Dict[str, Any],
    ) -> Union[pd.DataFrame, SkyulfDataFrame, Tuple[Any, ...]]:
        X, y, is_tuple = unpack_pipeline_input(df)
        engine = get_engine(X)

        stats = params.get("stats", {})
        threshold = params.get("threshold", 3.0)

        if not stats:
            return pack_pipeline_output(X, y, is_tuple)

        # Polars Path
        if engine.name == "polars":
            import polars as pl
            
            X_pl: Any = X
            mask = pl.lit(True)
            
            for col, stat in stats.items():
                if col not in X_pl.columns:
                    continue
                
                mean = stat["mean"]
                std = stat["std"]
                
                if std == 0:
                    continue
                
                # z_score = (col - mean) / std
                # abs(z) <= threshold
                
                z_score = (pl.col(col) - mean) / std
                col_mask = z_score.abs() <= threshold
                col_mask = col_mask | pl.col(col).is_null()
                
                mask = mask & col_mask
            
            mask_series = X_pl.select(mask.alias("mask")).get_column("mask")
            X_filtered = X_pl.filter(mask_series)
            
            y_filtered = None
            if y is not None:
                if isinstance(y, (pl.Series, pl.DataFrame)):
                    y_filtered = y.filter(mask_series)
                else:
                    y_filtered = y
                
            return pack_pipeline_output(X_filtered, y_filtered, is_tuple)

        # Pandas Path
        X_pd: Any = X.to_pandas() if hasattr(X, "to_pandas") else X
        mask = pd.Series(True, index=X_pd.index)

        for col, stat in stats.items():
            if col not in X_pd.columns:
                continue

            mean = stat["mean"]
            std = stat["std"]

            if std == 0:
                continue

            series = pd.to_numeric(X_pd[col], errors="coerce")
            z_scores = (series - mean) / std

            # Keep if abs(z) <= threshold
            col_mask = z_scores.abs() <= threshold
            col_mask = col_mask | series.isna()

            mask = mask & col_mask

        X_filtered = X_pd[mask]

        if y is not None:
            y_pd: Any = y
            y_filtered = y_pd[mask]
            return pack_pipeline_output(X_filtered, y_filtered, is_tuple)

        return pack_pipeline_output(X_filtered, y, is_tuple)


@NodeRegistry.register("ZScore", ZScoreApplier)
@node_meta(
    id="ZScore",
    name="Z-Score Outlier Removal",
    category="Preprocessing",
    description="Remove outliers using Z-Score.",
    params={"threshold": 3.0, "columns": []}
)
class ZScoreCalculator(BaseCalculator):
    def fit(
        self,
        df: Union[pd.DataFrame, SkyulfDataFrame, Tuple[Any, ...]],
        config: Dict[str, Any],
    ) -> Dict[str, Any]:
        X, _, _ = unpack_pipeline_input(df)
        engine = get_engine(X)
        X_pd: Any = X.to_pandas() if hasattr(X, "to_pandas") else X

        # Config: {'threshold': 3.0, 'columns': [...]}
        threshold = config.get("threshold", 3.0)

        cols = resolve_columns(X_pd, config, detect_numeric_columns)

        if not cols:
            return {}

        stats = {}
        warnings = []
        for col in cols:
            series = pd.to_numeric(X_pd[col], errors="coerce").dropna()
            if series.empty:
                warnings.append(f"Column '{col}': Empty or non-numeric")
                continue

            mean = series.mean()
            std = series.std(ddof=0)

            if std == 0:
                warnings.append(f"Column '{col}': Zero variance (std=0)")
                continue

            stats[col] = {"mean": mean, "std": std}

        return {
            "type": "zscore",
            "stats": stats,
            "threshold": threshold,
            "warnings": warnings,
        }


# --- Winsorize ---


class WinsorizeApplier(BaseApplier):
    def apply(
        self,
        df: Union[pd.DataFrame, SkyulfDataFrame, Tuple[Any, ...]],
        params: Dict[str, Any],
    ) -> Union[pd.DataFrame, SkyulfDataFrame, Tuple[Any, ...]]:
        X, y, is_tuple = unpack_pipeline_input(df)
        engine = get_engine(X)

        bounds = params.get("bounds", {})
        if not bounds:
            return pack_pipeline_output(X, y, is_tuple)

        # Polars Path
        if engine.name == "polars":
            import polars as pl
            
            X_pl: Any = X
            exprs = []
            for col, bound in bounds.items():
                if col not in X_pl.columns:
                    continue
                
                lower = bound["lower"]
                upper = bound["upper"]
                
                # Clip
                # Ensure we cast to float if bounds are float to avoid truncation
                exprs.append(pl.col(col).cast(pl.Float64).clip(lower, upper).alias(col))
            
            X_out = X_pl.with_columns(exprs)
            return pack_pipeline_output(X_out, y, is_tuple)

        # Pandas Path
        X_pd: Any = X.to_pandas() if hasattr(X, "to_pandas") else X
        df_out = X_pd.copy()

        for col, bound in bounds.items():
            if col not in df_out.columns:
                continue

            lower = bound["lower"]
            upper = bound["upper"]

            # Clip values
            if pd.api.types.is_numeric_dtype(df_out[col]):
                df_out[col] = df_out[col].clip(lower=lower, upper=upper)

        return pack_pipeline_output(df_out, y, is_tuple)


@NodeRegistry.register("Winsorize", WinsorizeApplier)
@node_meta(
    id="Winsorize",
    name="Winsorization",
    category="Preprocessing",
    description="Limit extreme values in the data.",
    params={"limits": [0.05, 0.05], "columns": []}
)
class WinsorizeCalculator(BaseCalculator):
    def fit(
        self,
        df: Union[pd.DataFrame, SkyulfDataFrame, Tuple[Any, ...]],
        config: Dict[str, Any],
    ) -> Dict[str, Any]:
        X, _, _ = unpack_pipeline_input(df)
        engine = get_engine(X)
        X_pd: Any = X.to_pandas() if hasattr(X, "to_pandas") else X

        # Config: {'lower_percentile': 5.0, 'upper_percentile': 95.0, 'columns': [...]}
        lower_p = config.get("lower_percentile", 5.0)
        upper_p = config.get("upper_percentile", 95.0)

        cols = resolve_columns(X_pd, config, detect_numeric_columns)

        if not cols:
            return {}

        bounds = {}
        warnings = []
        for col in cols:
            series = pd.to_numeric(X_pd[col], errors="coerce").dropna()
            if series.empty:
                warnings.append(f"Column '{col}': Empty or non-numeric")
                continue

            lower_val = series.quantile(lower_p / 100.0)
            upper_val = series.quantile(upper_p / 100.0)

            bounds[col] = {"lower": lower_val, "upper": upper_val}

        return {
            "type": "winsorize",
            "bounds": bounds,
            "lower_percentile": lower_p,
            "upper_percentile": upper_p,
            "warnings": warnings,
        }


# --- Manual Bounds ---


class ManualBoundsApplier(BaseApplier):
    def apply(
        self,
        df: Union[pd.DataFrame, SkyulfDataFrame, Tuple[Any, ...]],
        params: Dict[str, Any],
    ) -> Union[pd.DataFrame, SkyulfDataFrame, Tuple[Any, ...]]:
        X, y, is_tuple = unpack_pipeline_input(df)
        engine = get_engine(X)

        bounds = params.get("bounds", {})
        if not bounds:
            return pack_pipeline_output(X, y, is_tuple)

        # Polars Path
        if engine.name == "polars":
            import polars as pl
            
            X_pl: Any = X
            mask = pl.lit(True)
            
            for col, bound in bounds.items():
                if col not in X_pl.columns:
                    continue
                
                lower = bound.get("lower")
                upper = bound.get("upper")
                
                col_mask = pl.lit(True)
                if lower is not None:
                    col_mask = col_mask & (pl.col(col) >= lower)
                if upper is not None:
                    col_mask = col_mask & (pl.col(col) <= upper)
                
                col_mask = col_mask | pl.col(col).is_null()
                mask = mask & col_mask
            
            mask_series = X_pl.select(mask.alias("mask")).get_column("mask")
            X_filtered = X_pl.filter(mask_series)
            
            y_filtered = None
            if y is not None:
                if isinstance(y, (pl.Series, pl.DataFrame)):
                    y_filtered = y.filter(mask_series)
                else:
                    y_filtered = y
                
            return pack_pipeline_output(X_filtered, y_filtered, is_tuple)

        # Pandas Path
        X_pd: Any = X.to_pandas() if hasattr(X, "to_pandas") else X
        mask = pd.Series(True, index=X_pd.index)

        for col, bound in bounds.items():
            if col not in X_pd.columns:
                continue

            lower = bound.get("lower")
            upper = bound.get("upper")

            series = pd.to_numeric(X_pd[col], errors="coerce")
            col_mask = pd.Series(True, index=X_pd.index)

            if lower is not None:
                col_mask = col_mask & (series >= lower)
            if upper is not None:
                col_mask = col_mask & (series <= upper)

            col_mask = col_mask | series.isna()
            mask = mask & col_mask

        X_filtered = X_pd[mask]

        if y is not None:
            y_pd: Any = y
            y_filtered = y_pd[mask]
            return pack_pipeline_output(X_filtered, y_filtered, is_tuple)

        return pack_pipeline_output(X_filtered, y, is_tuple)


@NodeRegistry.register("ManualBounds", ManualBoundsApplier)
@node_meta(
    id="ManualBounds",
    name="Manual Bounds",
    category="Preprocessing",
    description="Filter outliers by manually specifying lower and upper bounds for columns.",
    params={"bounds": {}}
)
class ManualBoundsCalculator(BaseCalculator):
    def fit(
        self,
        df: Union[pd.DataFrame, SkyulfDataFrame, Tuple[Any, ...]],
        config: Dict[str, Any],
    ) -> Dict[str, Any]:
        # Config: {'bounds': {'col1': {'lower': 0, 'upper': 100}, ...}}
        bounds = config.get("bounds", {})

        return {"type": "manual_bounds", "bounds": bounds}


# --- Elliptic Envelope ---


class EllipticEnvelopeApplier(BaseApplier):
    def apply(
        self,
        df: Union[pd.DataFrame, SkyulfDataFrame, Tuple[Any, ...]],
        params: Dict[str, Any],
    ) -> Union[pd.DataFrame, SkyulfDataFrame, Tuple[Any, ...]]:
        X, y, is_tuple = unpack_pipeline_input(df)
        engine = get_engine(X)

        models = params.get("models", {})
        if not models:
            return pack_pipeline_output(X, y, is_tuple)

        # Polars Path: Convert to Pandas because we use sklearn models
        if engine.name == "polars":
            # Polars conversion path
            X_pd = X.to_pandas()
            y_pd = None
            if y is not None:
                if hasattr(y, "to_pandas"):
                    y_pd = y.to_pandas()
                else:
                    y_pd = y
            
            # We will use X_pd for logic
        else:
            # Standard Pandas path
            # Explicitly alias to Any to avoid "is not indexable" errors from mypy on Union types
            X_pd = X.to_pandas() if hasattr(X, "to_pandas") else X
            y_pd = y

        X_pd_any: Any = X_pd
        
        mask = pd.Series(True, index=X_pd_any.index)

        for col, model in models.items():
            if col not in X_pd_any.columns:
                continue

            series = pd.to_numeric(X_pd_any[col], errors="coerce")
            # EllipticEnvelope.predict returns 1 for inliers, -1 for outliers
            # We need to handle NaNs separately as predict might fail or treat them weirdly

            # Only predict on non-NaN
            valid_idx = series.dropna().index
            if valid_idx.empty:
                continue

            try:
                preds = model.predict(series.loc[valid_idx].to_numpy().reshape(-1, 1))
                # 1 is inlier, -1 is outlier
                is_inlier = preds == 1

                # Create mask for this column
                col_mask = pd.Series(
                    False, index=X_pd_any.index
                )  # Default to outlier? Or inlier?
                # Usually we keep inliers. So default to False (outlier) unless proven inlier.
                # But we also keep NaNs usually? Or drop them?
                # Let's say we keep NaNs (handled by other steps)
                col_mask[series.isna()] = True

                col_mask.loc[valid_idx] = is_inlier

                mask = mask & col_mask
            except Exception as e:
                logger.warning(f"EllipticEnvelope predict failed for column {col}: {e}")
                pass

        X_filtered = X_pd_any[mask]

        if y_pd is not None:
            y_filtered = y_pd[mask]
            # If we started with Polars, we might want to convert back.
            if engine.name == "polars":
                import polars as pl
                return pack_pipeline_output(pl.from_pandas(X_filtered), pl.from_pandas(y_filtered) if y_filtered is not None else None, is_tuple)
            
            return pack_pipeline_output(X_filtered, y_filtered, is_tuple)

        if engine.name == "polars":
            import polars as pl
            return pack_pipeline_output(pl.from_pandas(X_filtered), y, is_tuple)

        return pack_pipeline_output(X_filtered, y, is_tuple)


@NodeRegistry.register("EllipticEnvelope", EllipticEnvelopeApplier)
@node_meta(
    id="EllipticEnvelope",
    name="Elliptic Envelope",
    category="Preprocessing",
    description="Detect outliers in a Gaussian distributed dataset.",
    params={"contamination": 0.01, "columns": []}
)
class EllipticEnvelopeCalculator(BaseCalculator):
    def fit(
        self,
        df: Union[pd.DataFrame, SkyulfDataFrame, Tuple[Any, ...]],
        config: Dict[str, Any],
    ) -> Dict[str, Any]:
        X, _, _ = unpack_pipeline_input(df)
        engine = get_engine(X)
        X_pd: Any = X.to_pandas() if hasattr(X, "to_pandas") else X

        # Config: {'contamination': 0.01, 'columns': [...]}
        contamination = config.get("contamination", 0.01)

        cols = resolve_columns(X_pd, config, detect_numeric_columns)

        if not cols:
            return {}

        models = {}
        warnings = []
        for col in cols:
            series = pd.to_numeric(X_pd[col], errors="coerce").dropna()
            if series.shape[0] < 5:
                warnings.append(f"Column '{col}': Too few samples ({series.shape[0]})")
                continue

            try:
                model = EllipticEnvelope(contamination=contamination)
                model.fit(series.to_numpy().reshape(-1, 1))
                models[col] = model
            except Exception as e:
                logger.warning(f"EllipticEnvelope fit failed for column {col}: {e}")
                warnings.append(f"Column '{col}': {str(e)}")
                pass

        return {
            "type": "elliptic_envelope",
            "models": models,
            "contamination": contamination,
            "warnings": warnings,
        }
