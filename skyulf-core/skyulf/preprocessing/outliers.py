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
from ..registry import NodeRegistry
from ..engines import SkyulfDataFrame, get_engine

logger = logging.getLogger(__name__)

# --- IQR Filter ---


class IQRApplier(BaseApplier):
    def apply(
        self,
        df: SkyulfDataFrame,
        params: Dict[str, Any],
    ) -> Union[SkyulfDataFrame, Tuple[SkyulfDataFrame, Any]]:
        X, y, is_tuple = unpack_pipeline_input(df)
        engine = get_engine(X)

        bounds = params.get("bounds", {})
        if not bounds:
            return pack_pipeline_output(X, y, is_tuple)

        # Polars Path
        if engine.name == "polars":
            import polars as pl
            
            mask = pl.lit(True)
            
            for col, bound in bounds.items():
                if col not in X.columns:
                    continue
                
                lower = bound["lower"]
                upper = bound["upper"]
                
                # Keep values within bounds or Null
                col_mask = (pl.col(col) >= lower) & (pl.col(col) <= upper)
                col_mask = col_mask | pl.col(col).is_null()
                
                mask = mask & col_mask
            
            X_filtered = X.filter(mask)
            
            if y is not None:
                if isinstance(y, pl.Series):
                    y_filtered = y.filter(mask)
                elif isinstance(y, pl.DataFrame):
                     y_filtered = y.filter(mask)
                else:
                    y_filtered = y
            else:
                y_filtered = None
                
            return pack_pipeline_output(X_filtered, y_filtered, is_tuple)

        # Pandas Path
        mask = pd.Series(True, index=X.index)

        for col, bound in bounds.items():
            if col not in X.columns:
                continue

            lower = bound["lower"]
            upper = bound["upper"]

            series = pd.to_numeric(X[col], errors="coerce")

            # Keep values within bounds or NaN
            col_mask = (series >= lower) & (series <= upper)
            col_mask = col_mask | series.isna()

            mask = mask & col_mask

        X_filtered = X[mask]

        if y is not None:
            y_filtered = y[mask]
            return pack_pipeline_output(X_filtered, y_filtered, is_tuple)

        return pack_pipeline_output(X_filtered, y, is_tuple)


@NodeRegistry.register("IQR", IQRApplier)
class IQRCalculator(BaseCalculator):
    def fit(
        self,
        df: SkyulfDataFrame,
        config: Dict[str, Any],
    ) -> Dict[str, Any]:
        X, _, _ = unpack_pipeline_input(df)
        
        engine = get_engine(X)
        if engine.name == "polars":
            X = X.to_pandas()

        # Config: {'multiplier': 1.5, 'columns': [...]}
        multiplier = config.get("multiplier", 1.5)

        cols = resolve_columns(X, config, detect_numeric_columns)

        if not cols:
            return {}

        bounds = {}
        warnings = []
        for col in cols:
            series = pd.to_numeric(X[col], errors="coerce").dropna()
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
        df: SkyulfDataFrame,
        params: Dict[str, Any],
    ) -> Union[SkyulfDataFrame, Tuple[SkyulfDataFrame, Any]]:
        X, y, is_tuple = unpack_pipeline_input(df)
        engine = get_engine(X)

        stats = params.get("stats", {})
        threshold = params.get("threshold", 3.0)

        if not stats:
            return pack_pipeline_output(X, y, is_tuple)

        # Polars Path
        if engine.name == "polars":
            import polars as pl
            
            mask = pl.lit(True)
            
            for col, stat in stats.items():
                if col not in X.columns:
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
            
            X_filtered = X.filter(mask)
            
            if y is not None:
                if isinstance(y, pl.Series):
                    y_filtered = y.filter(mask)
                elif isinstance(y, pl.DataFrame):
                     y_filtered = y.filter(mask)
                else:
                    y_filtered = y
            else:
                y_filtered = None
                
            return pack_pipeline_output(X_filtered, y_filtered, is_tuple)

        # Pandas Path
        mask = pd.Series(True, index=X.index)

        for col, stat in stats.items():
            if col not in X.columns:
                continue

            mean = stat["mean"]
            std = stat["std"]

            if std == 0:
                continue

            series = pd.to_numeric(X[col], errors="coerce")
            z_scores = (series - mean) / std

            # Keep if abs(z) <= threshold
            col_mask = z_scores.abs() <= threshold
            col_mask = col_mask | series.isna()

            mask = mask & col_mask

        X_filtered = X[mask]

        if y is not None:
            y_filtered = y[mask]
            return pack_pipeline_output(X_filtered, y_filtered, is_tuple)

        return pack_pipeline_output(X_filtered, y, is_tuple)


@NodeRegistry.register("ZScore", ZScoreApplier)
class ZScoreCalculator(BaseCalculator):
    def fit(
        self,
        df: SkyulfDataFrame,
        config: Dict[str, Any],
    ) -> Dict[str, Any]:
        X, _, _ = unpack_pipeline_input(df)
        
        engine = get_engine(X)
        if engine.name == "polars":
            X = X.to_pandas()

        # Config: {'threshold': 3.0, 'columns': [...]}
        threshold = config.get("threshold", 3.0)

        cols = resolve_columns(X, config, detect_numeric_columns)

        if not cols:
            return {}

        stats = {}
        warnings = []
        for col in cols:
            series = pd.to_numeric(X[col], errors="coerce").dropna()
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
        df: SkyulfDataFrame,
        params: Dict[str, Any],
    ) -> Union[SkyulfDataFrame, Tuple[SkyulfDataFrame, Any]]:
        X, y, is_tuple = unpack_pipeline_input(df)
        engine = get_engine(X)

        bounds = params.get("bounds", {})
        if not bounds:
            return pack_pipeline_output(X, y, is_tuple)

        # Polars Path
        if engine.name == "polars":
            import polars as pl
            
            exprs = []
            for col, bound in bounds.items():
                if col not in X.columns:
                    continue
                
                lower = bound["lower"]
                upper = bound["upper"]
                
                # Clip
                # Ensure we cast to float if bounds are float to avoid truncation
                exprs.append(pl.col(col).cast(pl.Float64).clip(lower, upper).alias(col))
            
            X_out = X.with_columns(exprs)
            return pack_pipeline_output(X_out, y, is_tuple)

        # Pandas Path
        df_out = X.copy()

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
class WinsorizeCalculator(BaseCalculator):
    def fit(
        self,
        df: SkyulfDataFrame,
        config: Dict[str, Any],
    ) -> Dict[str, Any]:
        X, _, _ = unpack_pipeline_input(df)
        
        engine = get_engine(X)
        if engine.name == "polars":
            X = X.to_pandas()

        # Config: {'lower_percentile': 5.0, 'upper_percentile': 95.0, 'columns': [...]}
        lower_p = config.get("lower_percentile", 5.0)
        upper_p = config.get("upper_percentile", 95.0)

        cols = resolve_columns(X, config, detect_numeric_columns)

        if not cols:
            return {}

        bounds = {}
        warnings = []
        for col in cols:
            series = pd.to_numeric(X[col], errors="coerce").dropna()
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
        df: SkyulfDataFrame,
        params: Dict[str, Any],
    ) -> Union[SkyulfDataFrame, Tuple[SkyulfDataFrame, Any]]:
        X, y, is_tuple = unpack_pipeline_input(df)
        engine = get_engine(X)

        bounds = params.get("bounds", {})
        if not bounds:
            return pack_pipeline_output(X, y, is_tuple)

        # Polars Path
        if engine.name == "polars":
            import polars as pl
            
            mask = pl.lit(True)
            
            for col, bound in bounds.items():
                if col not in X.columns:
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
            
            X_filtered = X.filter(mask)
            
            if y is not None:
                if isinstance(y, pl.Series):
                    y_filtered = y.filter(mask)
                elif isinstance(y, pl.DataFrame):
                     y_filtered = y.filter(mask)
                else:
                    y_filtered = y
            else:
                y_filtered = None
                
            return pack_pipeline_output(X_filtered, y_filtered, is_tuple)

        # Pandas Path
        mask = pd.Series(True, index=X.index)

        for col, bound in bounds.items():
            if col not in X.columns:
                continue

            lower = bound.get("lower")
            upper = bound.get("upper")

            series = pd.to_numeric(X[col], errors="coerce")
            col_mask = pd.Series(True, index=X.index)

            if lower is not None:
                col_mask = col_mask & (series >= lower)
            if upper is not None:
                col_mask = col_mask & (series <= upper)

            col_mask = col_mask | series.isna()
            mask = mask & col_mask

        X_filtered = X[mask]

        if y is not None:
            y_filtered = y[mask]
            return pack_pipeline_output(X_filtered, y_filtered, is_tuple)

        return pack_pipeline_output(X_filtered, y, is_tuple)


@NodeRegistry.register("ManualBounds", ManualBoundsApplier)
class ManualBoundsCalculator(BaseCalculator):
    def fit(
        self,
        df: SkyulfDataFrame,
        config: Dict[str, Any],
    ) -> Dict[str, Any]:
        # Config: {'bounds': {'col1': {'lower': 0, 'upper': 100}, ...}}
        bounds = config.get("bounds", {})

        return {"type": "manual_bounds", "bounds": bounds}


# --- Elliptic Envelope ---


class EllipticEnvelopeApplier(BaseApplier):
    def apply(
        self,
        df: SkyulfDataFrame,
        params: Dict[str, Any],
    ) -> Union[SkyulfDataFrame, Tuple[SkyulfDataFrame, Any]]:
        X, y, is_tuple = unpack_pipeline_input(df)
        engine = get_engine(X)

        models = params.get("models", {})
        if not models:
            return pack_pipeline_output(X, y, is_tuple)

        # Polars Path: Convert to Pandas because we use sklearn models
        if engine.name == "polars":
            X_pd = X.to_pandas()
            # We need to convert y too if we are going to filter it
            y_pd = None
            if y is not None:
                if hasattr(y, "to_pandas"):
                    y_pd = y.to_pandas()
                else:
                    y_pd = y
            
            # Use the pandas logic (recursive call or just copy paste)
            # Let's just convert and proceed with pandas logic below
            X = X_pd
            y = y_pd
            # Note: We will return Pandas DataFrame. 
            # If we want to return Polars, we should convert back.
            # But pack_pipeline_output handles it? No, pack_pipeline_output just packs what we give.
            # If input was Polars, output should ideally be Polars.
            # But for EllipticEnvelope, maybe it's fine to return Pandas?
            # Or we convert back.
            
            # Let's proceed with X as pandas, and at the end convert back if needed.
            # But wait, `X` variable is now overwritten.
            
        # Pandas Path (and Polars converted)
        mask = pd.Series(True, index=X.index)

        for col, model in models.items():
            if col not in X.columns:
                continue

            series = pd.to_numeric(X[col], errors="coerce")
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
                    False, index=X.index
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

        X_filtered = X[mask]

        if y is not None:
            y_filtered = y[mask]
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
class EllipticEnvelopeCalculator(BaseCalculator):
    def fit(
        self,
        df: SkyulfDataFrame,
        config: Dict[str, Any],
    ) -> Dict[str, Any]:
        X, _, _ = unpack_pipeline_input(df)
        
        engine = get_engine(X)
        if engine.name == "polars":
            X = X.to_pandas()

        # Config: {'contamination': 0.01, 'columns': [...]}
        contamination = config.get("contamination", 0.01)

        cols = resolve_columns(X, config, detect_numeric_columns)

        if not cols:
            return {}

        models = {}
        warnings = []
        for col in cols:
            series = pd.to_numeric(X[col], errors="coerce").dropna()
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
