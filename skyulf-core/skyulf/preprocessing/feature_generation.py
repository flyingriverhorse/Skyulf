import builtins
import math
from difflib import SequenceMatcher
from typing import Any, Dict, List, Optional, Tuple, Union

import pandas as pd
from sklearn.preprocessing import PolynomialFeatures

from ..registry import NodeRegistry
from ..core.meta.decorators import node_meta
from ..utils import detect_numeric_columns, pack_pipeline_output, unpack_pipeline_input
from .base import BaseApplier, BaseCalculator
from ..engines import SkyulfDataFrame, get_engine

# --- Optional Dependencies ---
try:
    from rapidfuzz import fuzz  # type: ignore

    _HAS_RAPIDFUZZ = True
except ImportError:
    fuzz = None
    _HAS_RAPIDFUZZ = False

# --- Constants ---
DEFAULT_EPSILON = 1e-9
FEATURE_MATH_ALLOWED_TYPES = {
    "arithmetic",
    "ratio",
    "similarity",
    "datetime_extract",
    "group_agg",
    "polynomial",
}
ALLOWED_DATETIME_FEATURES = {
    "year",
    "quarter",
    "month",
    "month_name",
    "week",
    "day",
    "day_name",
    "weekday",
    "is_weekend",
    "hour",
    "minute",
    "second",
    "season",
    "time_of_day",
}

# --- Helpers ---


def _coerce_float(value: Any) -> Optional[float]:
    if value is None:
        return None
    try:
        val = float(value)
        return val if not math.isnan(val) else None
    except (TypeError, ValueError):
        return None


def _safe_divide(
    numerator: pd.Series, denominator: pd.Series, epsilon: float
) -> pd.Series:
    adjusted = denominator.copy()
    adjusted = adjusted.replace({0: epsilon, -0.0: epsilon})
    adjusted = adjusted.fillna(epsilon)
    # Avoid division by zero or near-zero
    mask = adjusted.abs() < epsilon
    if mask.any():
        adjusted[mask] = epsilon
    return numerator / adjusted


def _compute_similarity_score(a: Any, b: Any, method: str) -> float:
    text_a = "" if pd.isna(a) else str(a)
    text_b = "" if pd.isna(b) else str(b)
    if not text_a and not text_b:
        return 100.0
    if not text_a or not text_b:
        return 0.0

    if _HAS_RAPIDFUZZ:
        if method == "token_sort_ratio":
            return float(fuzz.token_sort_ratio(text_a, text_b))
        if method == "token_set_ratio":
            return float(fuzz.token_set_ratio(text_a, text_b))
        return float(fuzz.ratio(text_a, text_b))

    # Fallback
    return SequenceMatcher(None, text_a, text_b).ratio() * 100.0


# --- Polynomial Features ---


class PolynomialFeaturesApplier(BaseApplier):
    def apply(
        self,
        df: Union[pd.DataFrame, SkyulfDataFrame, Tuple[Any, ...], Any],
        params: Dict[str, Any],
    ) -> Union[pd.DataFrame, SkyulfDataFrame, Tuple[Any, ...]]:
        X, y, is_tuple = unpack_pipeline_input(df)
        engine = get_engine(X)

        cols = params.get("columns", [])
        valid_cols = [c for c in cols if c in X.columns]

        if not valid_cols:
            return pack_pipeline_output(X, y, is_tuple)

        degree = params.get("degree", 2)
        interaction_only = params.get("interaction_only", False)
        include_bias = params.get("include_bias", False)
        include_input_features = params.get("include_input_features", False)
        output_prefix = params.get("output_prefix", "poly")

        # Polars Path
        if engine.name == "polars":
            import polars as pl
            X_pl: Any = X
            
            # For PolynomialFeatures, we use the pandas/sklearn implementation via conversion
            # to ensure full compatibility with complex degree/interaction logic.
            X_pd = X_pl.to_pandas()
            
            poly = PolynomialFeatures(
                degree=degree, interaction_only=interaction_only, include_bias=include_bias
            )
            poly.fit(X_pd[valid_cols])
            
            transformed = poly.transform(X_pd[valid_cols])
            feature_names = poly.get_feature_names_out(valid_cols)
            
            # Filter logic
            indices_to_keep = []
            for i, p in enumerate(poly.powers_):
                deg = sum(p)
                if deg == 1 and not include_input_features:
                    continue
                indices_to_keep.append(i)
                
            if not indices_to_keep:
                return pack_pipeline_output(X, y, is_tuple)
                
            transformed = transformed[:, indices_to_keep]
            feature_names = feature_names[indices_to_keep]
            
            new_names = []
            for name in feature_names:
                clean_name = name.replace(" ", "_").replace("^", "_pow_")
                new_names.append(f"{output_prefix}_{clean_name}")
            
            # Create Polars DataFrame
            df_poly = pl.DataFrame(transformed, schema=new_names)
            
            # Horizontal concat
            X_out = pl.concat([X_pl, df_poly], how="horizontal")
            
            return pack_pipeline_output(X_out, y, is_tuple)

        # Pandas Path
        poly = PolynomialFeatures(
            degree=degree, interaction_only=interaction_only, include_bias=include_bias
        )
        poly.fit(X[valid_cols])  # Cheap fit

        transformed = poly.transform(X[valid_cols])

        # Handle case where sklearn is configured to output pandas DataFrames
        if hasattr(transformed, "iloc"):
            transformed = transformed.values

        feature_names = poly.get_feature_names_out(valid_cols)

        # Filter out original features (degree 1) if not requested
        # We use powers_ to determine the degree of each output feature
        indices_to_keep = []
        for i, p in enumerate(poly.powers_):
            deg = sum(p)
            # If degree is 1 (linear term) and we don't want to include input features, skip it
            if deg == 1 and not include_input_features:
                continue
            indices_to_keep.append(i)

        if not indices_to_keep:
            return pack_pipeline_output(X, y, is_tuple)

        transformed = transformed[:, indices_to_keep]
        feature_names = feature_names[indices_to_keep]

        # Rename features to be more readable and avoid collisions
        # We convert sklearn's "col1 col2" format to "prefix_col1_col2"

        new_names = []
        for name in feature_names:
            # Clean up name
            clean_name = name.replace(" ", "_").replace("^", "_pow_")
            new_names.append(f"{output_prefix}_{clean_name}")

        df_poly = pd.DataFrame(transformed, columns=new_names, index=X.index)

        df_out = X.copy()
        # We no longer need to check include_input_features here as we filtered them above

        # Concatenate
        df_out = pd.concat([df_out, df_poly], axis=1)

        return pack_pipeline_output(df_out, y, is_tuple)


@NodeRegistry.register("PolynomialFeatures", PolynomialFeaturesApplier)
@NodeRegistry.register("PolynomialFeaturesNode", PolynomialFeaturesApplier)
@node_meta(
    id="PolynomialFeatures",
    name="Polynomial Features",
    category="Feature Engineering",
    description="Generate polynomial and interaction features.",
    params={"degree": 2, "interaction_only": False, "include_bias": False}
)
class PolynomialFeaturesCalculator(BaseCalculator):
    def fit(
        self,
        df: Union[pd.DataFrame, SkyulfDataFrame, Tuple[Any, ...], Any],
        config: Dict[str, Any],
    ) -> Dict[str, Any]:
        # Extract configuration parameters
        # We support standard polynomial features settings like degree and interaction_only

        X, _, _ = unpack_pipeline_input(df)
        engine = get_engine(X)

        cols = config.get("columns", [])
        auto_detect = config.get("auto_detect", False)

        if not cols and auto_detect:
            cols = detect_numeric_columns(X)

        cols = [c for c in cols if c in X.columns]

        if not cols:
            return {}

        degree = config.get("degree", 2)
        interaction_only = config.get("interaction_only", False)
        include_bias = config.get("include_bias", False)

        # We use sklearn to get feature names
        # Ensure X is compatible with sklearn (Pandas/Numpy)
        if engine.name == "polars":
            X_pl: Any = X
            X_fit = X_pl.select(cols).to_pandas()
        else:
            X_fit = X[cols]

        poly = PolynomialFeatures(
            degree=degree, interaction_only=interaction_only, include_bias=include_bias
        )
        poly.fit(X_fit)

        return {
            "type": "polynomial_features",
            "columns": cols,
            "degree": degree,
            "interaction_only": interaction_only,
            "include_bias": include_bias,
            "include_input_features": config.get("include_input_features", False),
            "output_prefix": config.get("output_prefix", "poly"),
            "feature_names": poly.get_feature_names_out(cols).tolist(),
        }


# --- Feature Generation ---


class FeatureGenerationApplier(BaseApplier):
    def apply(  # noqa: C901
        self,
        df: Union[pd.DataFrame, SkyulfDataFrame, Tuple[Any, ...], Any],
        params: Dict[str, Any],
    ) -> Union[pd.DataFrame, SkyulfDataFrame, Tuple[Any, ...]]:
        X, y, is_tuple = unpack_pipeline_input(df)
        engine = get_engine(X)

        operations = params.get("operations", [])
        epsilon = params.get("epsilon", DEFAULT_EPSILON)
        allow_overwrite = params.get("allow_overwrite", False)

        if not operations:
            return pack_pipeline_output(X, y, is_tuple)

        # Polars Path
        if engine.name == "polars":
            import polars as pl
            X_pl: Any = X
            
            X_out = X_pl
            
            for i, op in enumerate(operations):
                op_type = op.get("operation_type", "arithmetic")
                method = op.get("method")
                input_cols = op.get("input_columns", [])
                secondary_cols = op.get("secondary_columns", [])
                constants = op.get("constants", [])
                output_col = op.get("output_column")
                output_prefix = op.get("output_prefix")
                fillna = op.get("fillna")
                round_digits = op.get("round_digits")

                # Resolve output name
                if not output_col:
                    base = f"{op_type}_{i}"
                    if output_prefix:
                        base = f"{output_prefix}_{base}"
                    output_col = base

                if output_col in X_out.columns and not allow_overwrite:
                    j = 1
                    while f"{output_col}_{j}" in X_out.columns:
                        j += 1
                    output_col = f"{output_col}_{j}"

                try:
                    expr = None
                    
                    if op_type == "arithmetic":
                        valid_inputs = [c for c in input_cols if c in X_out.columns]
                        valid_secondary = [c for c in secondary_cols if c in X_out.columns]
                        all_cols = valid_inputs + valid_secondary
                        
                        if not all_cols and not constants:
                            continue
                            
                        fill_val = fillna if fillna is not None else 0
                        col_exprs = [pl.col(c).cast(pl.Float64).fill_null(fill_val) for c in all_cols]
                        const_vals = [float(c) for c in constants]
                        
                        if method == "add":
                            expr = pl.sum_horizontal(col_exprs) + sum(const_vals)
                            
                        elif method == "subtract":
                            if col_exprs:
                                expr = col_exprs[0]
                                others = col_exprs[1:]
                            else:
                                expr = pl.lit(0.0)
                                others = []
                                
                            for e in others:
                                expr = expr - e
                            for c in const_vals:
                                expr = expr - c
                                
                        elif method == "multiply":
                            expr = pl.lit(1.0)
                            for e in col_exprs:
                                expr = expr * e
                            for c in const_vals:
                                expr = expr * c
                                
                        elif method == "divide":
                            if col_exprs:
                                expr = col_exprs[0]
                                others = col_exprs[1:]
                            elif const_vals:
                                expr = pl.lit(const_vals[0])
                                others = []
                                const_vals = const_vals[1:]
                            else:
                                continue
                                
                            def safe_denom(d):
                                return pl.when(d.abs() < epsilon).then(epsilon).otherwise(d)
                                
                            for e in others:
                                expr = expr / safe_denom(e)
                            for c in const_vals:
                                c_val = c if abs(c) > epsilon else epsilon
                                expr = expr / c_val

                    elif op_type == "ratio":
                        nums = [pl.col(c).cast(pl.Float64).fill_null(0) for c in input_cols if c in X_out.columns]
                        dens = [pl.col(c).cast(pl.Float64).fill_null(0) for c in secondary_cols if c in X_out.columns]
                        
                        if not nums or not dens:
                            continue
                            
                        num_sum = pl.sum_horizontal(nums)
                        den_sum = pl.sum_horizontal(dens)
                        
                        expr = num_sum / pl.when(den_sum.abs() < epsilon).then(epsilon).otherwise(den_sum)

                    elif op_type == "similarity":
                        col_a = input_cols[0] if input_cols else None
                        col_b = secondary_cols[0] if secondary_cols else (input_cols[1] if len(input_cols) > 1 else None)
                        
                        if col_a and col_b and col_a in X_out.columns and col_b in X_out.columns:
                            def sim_func(struct_val):
                                a = struct_val.get("a")
                                b = struct_val.get("b")
                                return _compute_similarity_score(a, b, method)
                                
                            expr = pl.struct([pl.col(col_a).alias("a"), pl.col(col_b).alias("b")]).map_elements(sim_func, return_dtype=pl.Float64)

                    elif op_type == "datetime_extract":
                        valid_inputs = [c for c in input_cols if c in X_out.columns]
                        features = op.get("datetime_features", [])
                        
                        dt_exprs = []
                        for col in valid_inputs:
                            dtype = X_out.schema[col]
                            base_dt = pl.col(col)
                            if dtype == pl.String:
                                base_dt = pl.col(col).str.to_datetime(strict=False)
                            
                            for feat in features:
                                feat_name = f"{col}_{feat}"
                                val = None
                                if feat == "year": val = base_dt.dt.year()
                                elif feat == "month": val = base_dt.dt.month()
                                elif feat == "day": val = base_dt.dt.day()
                                elif feat == "hour": val = base_dt.dt.hour()
                                elif feat == "minute": val = base_dt.dt.minute()
                                elif feat == "second": val = base_dt.dt.second()
                                elif feat == "quarter": val = base_dt.dt.quarter()
                                elif feat == "dayofweek" or feat == "weekday": val = base_dt.dt.weekday() - 1
                                elif feat == "is_weekend": val = (base_dt.dt.weekday() >= 6).cast(pl.Int64)
                                
                                if val is not None:
                                    dt_exprs.append(val.alias(feat_name))
                        
                        if dt_exprs:
                            X_out = X_out.with_columns(dt_exprs)
                        continue

                    if expr is not None:
                        if round_digits is not None:
                            expr = expr.round(round_digits)
                        X_out = X_out.with_columns(expr.alias(output_col))

                except Exception:
                    pass
            
            return pack_pipeline_output(X_out, y, is_tuple)

        # Pandas Path
        df_out = X.copy()

        for i, op in enumerate(operations):
            op_type = op.get("operation_type", "arithmetic")
            method = op.get("method")
            input_cols = op.get("input_columns", [])
            secondary_cols = op.get("secondary_columns", [])
            constants = op.get("constants", [])
            output_col = op.get("output_column")
            output_prefix = op.get("output_prefix")
            fillna = op.get("fillna")
            round_digits = op.get("round_digits")

            # Resolve output name
            if not output_col:
                # Generate fallback
                base = f"{op_type}_{i}"
                if output_prefix:
                    base = f"{output_prefix}_{base}"
                output_col = base

            if output_col in df_out.columns and not allow_overwrite:
                # Avoid overwriting existing columns by appending a numeric suffix
                j = 1
                while f"{output_col}_{j}" in df_out.columns:
                    j += 1
                output_col = f"{output_col}_{j}"

            try:
                result = None

                if op_type == "arithmetic":
                    # Ensure inputs exist
                    valid_inputs = [c for c in input_cols if c in df_out.columns]
                    valid_secondary = [c for c in secondary_cols if c in df_out.columns]
                    all_cols = valid_inputs + valid_secondary
                    if not all_cols and not constants:
                        continue

                    series_list = [
                        pd.to_numeric(df_out[c], errors="coerce").fillna(
                            fillna if fillna is not None else 0
                        )
                        for c in all_cols
                    ]

                    if method == "add":
                        res: pd.Series = pd.Series(0.0, index=df_out.index)
                        for s in series_list:
                            res = res.add(s, fill_value=0)
                        for c in constants:
                            val = builtins.float(c)
                            res = res.add(val)
                        result = res
                    elif method == "subtract":
                        res_sub: pd.Series
                        if series_list:
                            res_sub = series_list[0].copy()
                            others = series_list[1:]
                        else:
                            res_sub = pd.Series(0.0, index=df_out.index)
                            others = []
                        for s in others:
                            res_sub = res_sub.subtract(s, fill_value=0)
                        for c in constants:
                            val = builtins.float(c)
                            res_sub = res_sub.sub(val)
                        result = res_sub
                    elif method == "multiply":
                        res_mul: pd.Series = pd.Series(1.0, index=df_out.index)
                        for s in series_list:
                            res_mul = res_mul.multiply(s, fill_value=1)
                        for c in constants:
                            val = builtins.float(c)
                            res_mul = res_mul.mul(val)
                        result = res_mul
                    elif method == "divide":
                        if series_list:
                            res = series_list[0].copy()
                            others = series_list[1:]
                        elif constants:
                            res = pd.Series(constants[0], index=df_out.index)
                            others = []
                            constants = constants[1:]
                        else:
                            continue

                        for s in others:
                            res = _safe_divide(res, s, epsilon)
                        for c in constants:
                            c_val = builtins.float(c)
                            # For division, we need to handle epsilon check
                            # res = res / (c_val if abs(c_val) > epsilon else epsilon)
                            denom = c_val if abs(c_val) > epsilon else epsilon
                            res = res.div(denom)
                        result = res

                elif op_type == "ratio":
                    # input_cols (numerator) / secondary_cols (denominator)
                    # Sum of numerators / Sum of denominators
                    nums = [
                        pd.to_numeric(df_out[c], errors="coerce").fillna(0)
                        for c in input_cols
                        if c in df_out.columns
                    ]
                    dens = [
                        pd.to_numeric(df_out[c], errors="coerce").fillna(0)
                        for c in secondary_cols
                        if c in df_out.columns
                    ]

                    if not nums or not dens:
                        continue

                    num_sum = pd.Series(0.0, index=df_out.index)
                    for s in nums:
                        num_sum = num_sum.add(s, fill_value=0)

                    den_sum = pd.Series(0.0, index=df_out.index)
                    for s in dens:
                        den_sum = den_sum.add(s, fill_value=0)

                    result = _safe_divide(num_sum, den_sum, epsilon)

                elif op_type == "similarity":
                    # input_cols[0] vs secondary_cols[0] (or input_cols[1])
                    col_a = input_cols[0] if input_cols else None
                    col_b = (
                        secondary_cols[0]
                        if secondary_cols
                        else (input_cols[1] if len(input_cols) > 1 else None)
                    )

                    if (
                        not col_a
                        or not col_b
                        or col_a not in df_out.columns
                        or col_b not in df_out.columns
                    ):
                        continue

                    result = df_out.apply(
                        lambda row: _compute_similarity_score(
                            row[col_a], row[col_b], method
                        ),
                        axis=1,
                    )

                elif op_type == "datetime_extract":
                    valid_inputs = [c for c in input_cols if c in df_out.columns]
                    features = op.get("datetime_features", [])

                    for col in valid_inputs:
                        try:
                            dt = pd.to_datetime(df_out[col], errors="coerce")
                            for feat in features:
                                feat_name = f"{col}_{feat}"
                                if feat == "year":
                                    val = dt.dt.year
                                elif feat == "month":
                                    val = dt.dt.month
                                elif feat == "day":
                                    val = dt.dt.day
                                elif feat == "hour":
                                    val = dt.dt.hour
                                elif feat == "minute":
                                    val = dt.dt.minute
                                elif feat == "second":
                                    val = dt.dt.second
                                elif feat == "quarter":
                                    val = dt.dt.quarter
                                elif feat == "dayofweek" or feat == "weekday":
                                    val = dt.dt.dayofweek
                                elif feat == "is_weekend":
                                    val = (dt.dt.dayofweek >= 5).astype(int)
                                else:
                                    continue

                                df_out[feat_name] = val
                        except Exception:
                            pass
                    # datetime_extract usually generates multiple columns, so we might not set "result"
                    # unless we want to return one specific thing. V1 generates multiple.
                    continue

                if result is not None:
                    if round_digits is not None:
                        result = result.round(round_digits)
                    df_out[output_col] = result

            except Exception:
                pass

        return pack_pipeline_output(df_out, y, is_tuple)


@NodeRegistry.register("FeatureGeneration", FeatureGenerationApplier)
@NodeRegistry.register("FeatureMath", FeatureGenerationApplier)
@NodeRegistry.register("FeatureGenerationNode", FeatureGenerationApplier)
@node_meta(
    id="FeatureGenerationNode",
    name="Feature Generation (Math)",
    category="Feature Engineering",
    description="Generate new features using mathematical operations.",
    params={"operations": []}
)
class FeatureGenerationCalculator(BaseCalculator):
    def fit(
        self,
        df: Union[pd.DataFrame, SkyulfDataFrame, Tuple[Any, ...], Any],
        config: Dict[str, Any],
    ) -> Dict[str, Any]:
        # Config:
        # operations: List[Dict]
        # epsilon: float
        # allow_overwrite: bool

        return {
            "type": "feature_generation",
            "operations": config.get("operations", []),
            "epsilon": config.get("epsilon", DEFAULT_EPSILON),
            "allow_overwrite": config.get("allow_overwrite", False),
        }
