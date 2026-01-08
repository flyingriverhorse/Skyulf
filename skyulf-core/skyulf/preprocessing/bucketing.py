from typing import Any, Dict, List, Literal, Tuple, Union

import numpy as np
import pandas as pd
from sklearn.preprocessing import KBinsDiscretizer

from ..core.meta.decorators import node_meta
from ..utils import (
    detect_numeric_columns,
    pack_pipeline_output,
    resolve_columns,
    unpack_pipeline_input,
)
from .base import BaseApplier, BaseCalculator
from ..registry import NodeRegistry
from ..engines import SkyulfDataFrame, get_engine
from ..engines.sklearn_bridge import SklearnBridge

# --- Base Binning Applier ---


class BaseBinningApplier(BaseApplier):
    """
    Base class for applying binning transformations.
    Expects 'bin_edges' in params: Dict[str, List[float]] mapping column names to bin edges.
    """

    def apply(  # noqa: C901
        self,
        df: Union[pd.DataFrame, SkyulfDataFrame, Tuple[Any, ...], Any],
        params: Dict[str, Any],
    ) -> Union[pd.DataFrame, SkyulfDataFrame, Tuple[Any, ...]]:
        X, y, is_tuple = unpack_pipeline_input(df)
        engine = get_engine(X)

        bin_edges_map = params.get("bin_edges", {})
        if not bin_edges_map:
            return pack_pipeline_output(X, y, is_tuple)

        output_suffix = params.get("output_suffix", "_binned")
        drop_original = params.get("drop_original", False)
        label_format = params.get(
            "label_format", "ordinal"
        )  # ordinal, range, bin_index
        missing_strategy = params.get("missing_strategy", "keep")  # keep, label
        missing_label = params.get("missing_label", "Missing")
        include_lowest = params.get("include_lowest", True)
        precision = params.get("precision", 3)
        custom_labels_map = params.get("custom_labels", {})

        # Polars Path
        if engine.name == "polars":
            import polars as pl
            X_pl: Any = X

            exprs = []
            cols_to_drop = []

            for col, edges in bin_edges_map.items():
                if col not in X_pl.columns:
                    continue
                
                if drop_original:
                    cols_to_drop.append(col)

                sorted_edges = sorted(list(set(edges)))
                if len(sorted_edges) < 2:
                    continue

                # Determine labels
                labels = None
                col_custom_labels = custom_labels_map.get(col)
                if col_custom_labels and len(col_custom_labels) == len(sorted_edges) - 1:
                    labels = col_custom_labels

                # Polars cut
                # breaks are the internal cut points
                breaks = sorted_edges[1:-1]
                
                # Polars cut
                cut_expr = pl.col(col).cut(
                    breaks=breaks,
                    labels=labels,
                    left_closed=False, # (a, b]
                    include_breaks=False
                )
                
                target_col_name = f"{col}{output_suffix}"
                
                if label_format in ["ordinal", "bin_index"] and not labels:
                    # We want integer indices.
                    # Polars cut returns Categorical. Cast to UInt32 gives the physical index.
                    exprs.append(cut_expr.cast(pl.UInt32).alias(target_col_name))
                else:
                    # Range or Custom Labels
                    exprs.append(cut_expr.alias(target_col_name))

            X_out = X_pl.with_columns(exprs)
            if drop_original:
                X_out = X_out.drop(cols_to_drop)
            
            return pack_pipeline_output(X_out, y, is_tuple)

        # Pandas Path
        df_out = X.copy()
        processed_cols = []

        for col, edges in bin_edges_map.items():
            if col not in df_out.columns:
                continue

            processed_cols.append(col)

            # Determine labels for pd.cut
            labels: Union[Literal[False], List[Any], None] = (
                False  # Default for ordinal (returns integers)
            )

            # Check for custom labels first
            col_custom_labels = custom_labels_map.get(col)
            if col_custom_labels and len(col_custom_labels) == len(edges) - 1:
                labels = col_custom_labels
            elif label_format == "range":
                labels = None  # Returns intervals
            elif label_format == "bin_index":
                labels = False  # Returns integers 0..n-1

            # Apply cut
            try:
                # Ensure edges are unique and sorted
                sorted_edges = sorted(list(set(edges)))
                if len(sorted_edges) < 2:
                    continue

                binned_series = pd.cut(
                    df_out[col],
                    bins=sorted_edges,
                    labels=labels,
                    include_lowest=include_lowest,
                )

                # Handle missing values
                if missing_strategy == "label":
                    # If categorical (range or custom labels), add category
                    if isinstance(binned_series.dtype, pd.CategoricalDtype):
                        if missing_label not in binned_series.cat.categories:
                            binned_series = binned_series.cat.add_categories(
                                [missing_label]
                            )
                        binned_series = binned_series.fillna(missing_label)
                    else:
                        # If numeric (ordinal/bin_index), we convert to object/str to support "Missing" label
                        binned_series = binned_series.astype(object).fillna(
                            missing_label
                        )

                # Format ranges if needed
                if label_format == "range" and labels is None:
                    # Convert intervals to string with precision
                    if isinstance(binned_series.dtype, pd.CategoricalDtype):
                        # It's a categorical of intervals
                        def format_interval(iv):
                            if pd.isna(iv) or isinstance(iv, str):
                                return iv

                            # Use the logical left edge if it's the first bin and include_lowest is True
                            left_val = iv.left
                            if (
                                include_lowest
                                and len(sorted_edges) > 0
                                and left_val < sorted_edges[0]
                            ):
                                left_val = sorted_edges[0]

                            l_val = round(left_val, precision)
                            r_val = round(iv.right, precision)
                            if include_lowest:
                                return f"[{l_val}, {r_val}]"
                            else:
                                return f"({l_val}, {r_val}]"

                        # We need to map the categories themselves
                        new_categories = [
                            format_interval(c) for c in binned_series.cat.categories
                        ]
                        binned_series = binned_series.cat.rename_categories(
                            new_categories
                        )
                        binned_series = binned_series.astype(str)
                    else:
                        binned_series = binned_series.astype(str)

                    if missing_strategy == "keep":
                        # Restore NaNs if they were converted to 'nan' string
                        binned_series = binned_series.replace("nan", np.nan)

                out_col = f"{col}{output_suffix}"
                df_out[out_col] = binned_series

            except Exception:
                # Log error or skip
                pass

        if drop_original:
            df_out = df_out.drop(columns=processed_cols)

        return pack_pipeline_output(df_out, y, is_tuple)


# --- General Binning Calculator ---


class GeneralBinningApplier(BaseBinningApplier):
    pass


@NodeRegistry.register("GeneralBinning", GeneralBinningApplier)
@node_meta(
    id="GeneralBinning",
    name="General Binning",
    category="Preprocessing",
    description="Bin continuous data into intervals.",
    params={"n_bins": 5, "strategy": "uniform", "columns": []}
)
class GeneralBinningCalculator(BaseCalculator):
    """
    Master calculator that handles mixed strategies and overrides.
    """

    def fit(  # noqa: C901
        self, df: Union[pd.DataFrame, SkyulfDataFrame, Tuple[Any, ...], Any], config: Dict[str, Any]
    ) -> Dict[str, Any]:
        X, _, _ = unpack_pipeline_input(df)
        
        # Ensure X is pandas for fitting logic
        engine = get_engine(X)
        if engine.name == "polars":
            X = X.to_pandas()

        columns = resolve_columns(X, config, detect_numeric_columns)

        global_strategy = config.get("strategy", "equal_width")
        column_strategies = config.get("column_strategies", {})

        # Global settings
        default_n_bins = config.get("n_bins", 5)
        n_bins_global = config.get("equal_width_bins", default_n_bins)
        q_bins_global = config.get("equal_frequency_bins", default_n_bins)
        duplicates_global = config.get("duplicates", "drop")

        valid_cols = [c for c in columns if c in X.columns]
        bin_edges_map = {}
        custom_labels_map = {}

        for col in valid_cols:
            # Determine strategy and params for this column
            override = column_strategies.get(col, {})
            strategy = override.get("strategy", global_strategy)

            try:
                series = X[col].dropna()
                if series.empty:
                    continue

                edges = None

                if strategy == "equal_width":
                    n_bins = override.get("equal_width_bins", n_bins_global)
                    _, edges = pd.cut(series, bins=n_bins, retbins=True)
                    # Clamp first edge to min if it was extended
                    if len(edges) > 0 and edges[0] < series.min():
                        edges[0] = series.min()

                elif strategy == "equal_frequency":
                    n_bins = override.get("equal_frequency_bins", q_bins_global)
                    duplicates = override.get("duplicates", duplicates_global)
                    _, edges = pd.qcut(
                        series, q=n_bins, retbins=True, duplicates=duplicates
                    )
                    # Clamp first edge to min if it was extended
                    if len(edges) > 0 and edges[0] < series.min():
                        edges[0] = series.min()

                elif strategy == "kmeans":
                    n_bins = override.get("n_bins", default_n_bins)
                    est = KBinsDiscretizer(
                        n_bins=n_bins, strategy="kmeans", encode="ordinal", quantile_method="averaged_inverted_cdf"
                    )
                    est.fit(series.values.reshape(-1, 1))  # type: ignore
                    edges = est.bin_edges_[0]

                elif strategy == "custom":
                    # Check override first, then global custom_bins
                    custom_bins = override.get("custom_bins")
                    if not custom_bins:
                        custom_bins = config.get("custom_bins", {}).get(col)

                    if custom_bins:
                        edges = np.array(sorted(custom_bins))

                    # Handle custom labels
                    labels = override.get("custom_labels")
                    if not labels:
                        labels = config.get("custom_labels", {}).get(col)
                    if labels:
                        custom_labels_map[col] = labels

                elif strategy == "kbins":
                    n_bins = override.get(
                        "kbins_n_bins", config.get("kbins_n_bins", default_n_bins)
                    )
                    k_strategy = override.get(
                        "kbins_strategy", config.get("kbins_strategy", "quantile")
                    )

                    # Map strategy names
                    sklearn_strategy = k_strategy
                    if k_strategy == "equal_width":
                        sklearn_strategy = "uniform"
                    elif k_strategy == "equal_frequency":
                        sklearn_strategy = "quantile"

                    est = KBinsDiscretizer(
                        n_bins=n_bins, strategy=sklearn_strategy, encode="ordinal", quantile_method="averaged_inverted_cdf"
                    )
                    est.fit(series.values.reshape(-1, 1))  # type: ignore
                    edges = est.bin_edges_[0]

                if edges is not None:
                    bin_edges_map[col] = edges.tolist()

            except Exception:
                continue

        return {
            "type": "general_binning",
            "bin_edges": bin_edges_map,
            "custom_labels": custom_labels_map,
            "output_suffix": config.get("output_suffix", "_binned"),
            "drop_original": config.get("drop_original", False),
            "label_format": config.get("label_format", "ordinal"),
            "missing_strategy": config.get("missing_strategy", "keep"),
            "missing_label": config.get("missing_label", "Missing"),
            "include_lowest": config.get("include_lowest", True),
            "precision": config.get("precision", 3),
        }


class CustomBinningApplier(GeneralBinningApplier):
    pass


@NodeRegistry.register("CustomBinning", CustomBinningApplier)
@node_meta(
    id="CustomBinning",
    name="Custom Binning",
    category="Preprocessing",
    description="Bin data using custom edges.",
    params={"bins": [], "columns": []}
)
class CustomBinningCalculator(BaseCalculator):
    """
    Calculator for CustomBinning node.
    Applies specific bin edges to selected columns.
    """

    def fit(
        self,
        df: Union[pd.DataFrame, SkyulfDataFrame, Tuple[Any, ...], Any],
        config: Dict[str, Any],
    ) -> Dict[str, Any]:
        X, _, _ = unpack_pipeline_input(df)
        
        # Ensure X is pandas for fitting logic
        engine = get_engine(X)
        if engine.name == "polars":
            X = X.to_pandas()

        columns = resolve_columns(X, config, detect_numeric_columns)
        bins = config.get("bins")

        bin_edges_map = {}
        if bins:
            sorted_bins = sorted(bins)
            for col in columns:
                if col in X.columns:
                    bin_edges_map[col] = sorted_bins

        return {
            "type": "general_binning",  # Use GeneralBinningApplier
            "bin_edges": bin_edges_map,
            "output_suffix": config.get("output_suffix", "_binned"),
            "drop_original": config.get("drop_original", False),
            "label_format": config.get("label_format", "ordinal"),
            "missing_strategy": config.get("missing_strategy", "keep"),
            "missing_label": config.get("missing_label", "Missing"),
            "include_lowest": config.get("include_lowest", True),
            "precision": config.get("precision", 3),
        }


class KBinsDiscretizerApplier(GeneralBinningApplier):
    pass


@NodeRegistry.register("KBinsDiscretizer", KBinsDiscretizerApplier)
@node_meta(
    id="KBinsDiscretizer",
    name="K-Bins Discretizer",
    category="Preprocessing",
    description="Bin continuous data into intervals using sklearn KBinsDiscretizer.",
    params={"n_bins": 5, "encode": "ordinal", "strategy": "quantile", "columns": []}
)
class KBinsDiscretizerCalculator(GeneralBinningCalculator):
    """
    Calculator for KBinsDiscretizer node.
    Wraps GeneralBinningCalculator with kbins strategy.
    """

    def fit(
        self,
        df: Union[pd.DataFrame, SkyulfDataFrame, Tuple[Any, ...], Any],
        config: Dict[str, Any],
    ) -> Dict[str, Any]:
        new_config = config.copy()
        new_config["strategy"] = "kbins"

        # Map sklearn params to GeneralBinning params
        if "n_bins" in config:
            new_config["kbins_n_bins"] = config["n_bins"]

        # sklearn strategy: uniform, quantile, kmeans
        # GeneralBinning kbins_strategy: same
        if "strategy" in config and config["strategy"] != "kbins":
            new_config["kbins_strategy"] = config["strategy"]

        return super().fit(df, new_config)
