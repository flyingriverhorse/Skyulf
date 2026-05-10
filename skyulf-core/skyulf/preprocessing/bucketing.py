"""Binning / discretization nodes (`GeneralBinning`, `CustomBinning`, `KBinsDiscretizer`).

All Appliers share a single :class:`BaseBinningApplier` which routes through
:func:`apply_dual_engine`. Per-engine helpers live at module level so each one
stays at low CCN. Fits are sklearn / pandas-bound (`pd.cut`, `pd.qcut`,
`KBinsDiscretizer`), so they convert via ``to_pandas`` once at the top and do
not use :func:`fit_dual_engine`.
"""

from typing import Any, Dict, List, Literal, Tuple, Union, cast

import numpy as np
import pandas as pd
from sklearn.preprocessing import KBinsDiscretizer

from ..core.meta.decorators import node_meta
from ..utils import (
    detect_numeric_columns,
    resolve_columns,
    user_picked_no_columns,
)
from .base import BaseApplier, BaseCalculator, apply_method, fit_method
from .dispatcher import apply_dual_engine
from ._artifacts import GeneralBinningArtifact
from ..registry import NodeRegistry
from ..engines import EngineName, SkyulfDataFrame, get_engine


# -----------------------------------------------------------------------------
# Polars apply
# -----------------------------------------------------------------------------


def _polars_cut_expr(col: str, sorted_edges: List[float], labels: Any) -> Any:
    """Build a single polars ``cut`` expression for one column."""
    import polars as pl

    breaks = sorted_edges[1:-1]
    return pl.col(col).cut(breaks=breaks, labels=labels, left_closed=False, include_breaks=False)


def _polars_one_col_expr(
    col: str,
    edges: List[float],
    output_suffix: str,
    label_format: str,
    custom_labels_map: Dict[str, Any],
) -> Any:
    """Build the polars cut-expression for a single column, or ``None`` if degenerate."""
    import polars as pl

    sorted_edges = sorted(set(edges))
    if len(sorted_edges) < 2:
        return None

    col_custom_labels = custom_labels_map.get(col)
    labels = (
        col_custom_labels
        if col_custom_labels and len(col_custom_labels) == len(sorted_edges) - 1
        else None
    )

    cut_expr = _polars_cut_expr(col, sorted_edges, labels)
    target_col_name = f"{col}{output_suffix}"
    if label_format in ("ordinal", "bin_index") and not labels:
        # Polars cut returns Categorical; cast → physical index.
        return cut_expr.cast(pl.UInt32).alias(target_col_name)
    return cut_expr.alias(target_col_name)


def _build_polars_exprs(X: Any, params: Dict[str, Any]) -> Tuple[List[Any], List[str]]:
    """Build cut exprs and the list of original columns to drop."""
    bin_edges_map = params.get("bin_edges", {})
    output_suffix = params.get("output_suffix", "_binned")
    drop_original = params.get("drop_original", False)
    label_format = params.get("label_format", "ordinal")
    custom_labels_map = params.get("custom_labels", {})

    exprs: List[Any] = []
    cols_to_drop: List[str] = []
    for col, edges in bin_edges_map.items():
        if col not in X.columns:
            continue
        expr = _polars_one_col_expr(col, edges, output_suffix, label_format, custom_labels_map)
        if expr is not None:
            exprs.append(expr)
            if drop_original:
                cols_to_drop.append(col)
    return exprs, cols_to_drop


def _bucketing_apply_polars(X: Any, y: Any, params: Dict[str, Any]) -> Tuple[Any, Any]:
    if not params.get("bin_edges"):
        return X, y
    exprs, cols_to_drop = _build_polars_exprs(X, params)
    X_out = X.with_columns(exprs) if exprs else X
    if cols_to_drop:
        X_out = X_out.drop(cols_to_drop)
    return X_out, y


# -----------------------------------------------------------------------------
# Pandas apply — split per concern to keep CCN low
# -----------------------------------------------------------------------------


def _resolve_pandas_labels(
    label_format: str, edges: List[float], custom_labels: Any
) -> Union[Literal[False], List[Any], None]:
    """Pick the ``labels`` argument for :func:`pd.cut`."""
    if custom_labels and len(custom_labels) == len(edges) - 1:
        return custom_labels
    if label_format == "range":
        return None  # pd.cut returns intervals.
    # "ordinal" / "bin_index" → integer codes.
    return False


def _format_one_interval(
    iv: Any, sorted_edges: List[float], include_lowest: bool, precision: int
) -> Any:
    """Format a single :class:`pd.Interval` as ``[a, b]`` / ``(a, b]``."""
    if pd.isna(iv) or isinstance(iv, str):
        return iv
    left_val = iv.left
    if include_lowest and sorted_edges and left_val < sorted_edges[0]:
        left_val = sorted_edges[0]
    l_val = round(left_val, precision)
    r_val = round(iv.right, precision)
    bracket_l = "[" if include_lowest else "("
    return f"{bracket_l}{l_val}, {r_val}]"


def _format_intervals_to_strings(
    binned_series: pd.Series,
    sorted_edges: List[float],
    include_lowest: bool,
    precision: int,
    missing_strategy: str,
) -> pd.Series:
    """Convert an interval-categorical series to display strings."""
    if isinstance(binned_series.dtype, pd.CategoricalDtype):
        new_categories = [
            _format_one_interval(c, sorted_edges, include_lowest, precision)
            for c in binned_series.cat.categories
        ]
        binned_series = binned_series.cat.rename_categories(new_categories)
    binned_series = binned_series.astype(str)
    if missing_strategy == "keep":
        # ``astype(str)`` turns NaN into the literal "nan"; restore.
        binned_series = binned_series.replace("nan", np.nan)
    return binned_series


def _apply_missing_strategy(
    binned_series: pd.Series, missing_strategy: str, missing_label: str
) -> pd.Series:
    """Honour the ``label`` missing strategy by tagging NaNs with a category."""
    if missing_strategy != "label":
        return binned_series
    if isinstance(binned_series.dtype, pd.CategoricalDtype):
        if missing_label not in binned_series.cat.categories:
            binned_series = binned_series.cat.add_categories([missing_label])
        return binned_series.fillna(missing_label)
    # Numeric (ordinal/bin_index): widen to object so the label fits.
    return binned_series.astype(object).fillna(missing_label)


def _bin_one_column_pandas(
    series: pd.Series, edges: List[float], params: Dict[str, Any], custom_labels: Any
) -> pd.Series:
    """Build the binned series for one column. Raises on unrecoverable errors."""
    sorted_edges = sorted(set(edges))
    if len(sorted_edges) < 2:
        raise ValueError("not enough unique edges")

    label_format = params.get("label_format", "ordinal")
    include_lowest = params.get("include_lowest", True)
    precision = params.get("precision", 3)
    missing_strategy = params.get("missing_strategy", "keep")
    missing_label = params.get("missing_label", "Missing")

    labels = _resolve_pandas_labels(label_format, edges, custom_labels)
    binned = pd.cut(series, bins=sorted_edges, labels=labels, include_lowest=include_lowest)

    binned = _apply_missing_strategy(binned, missing_strategy, missing_label)
    if label_format == "range" and labels is None:
        binned = _format_intervals_to_strings(
            binned, sorted_edges, include_lowest, precision, missing_strategy
        )
    return binned


def _bucketing_apply_pandas(X: Any, y: Any, params: Dict[str, Any]) -> Tuple[Any, Any]:
    bin_edges_map = params.get("bin_edges", {})
    if not bin_edges_map:
        return X, y

    output_suffix = params.get("output_suffix", "_binned")
    drop_original = params.get("drop_original", False)
    custom_labels_map = params.get("custom_labels", {})

    df_out = X.copy()
    processed_cols: List[str] = []

    for col, edges in bin_edges_map.items():
        if col not in df_out.columns:
            continue
        processed_cols.append(col)
        try:
            binned_series = _bin_one_column_pandas(
                df_out[col], edges, params, custom_labels_map.get(col)
            )
            df_out[f"{col}{output_suffix}"] = binned_series
        except Exception:
            # Skip columns that fail (e.g. degenerate edges, dtype mismatch).
            continue

    if drop_original and processed_cols:
        df_out = df_out.drop(columns=processed_cols)
    return df_out, y


# --- Base Binning Applier ---


class BaseBinningApplier(BaseApplier):
    """Shared Applier for all binning Calculators.

    Expects ``bin_edges`` in params: ``Dict[str, List[float]]`` mapping column
    names to bin edges.
    """

    @apply_method
    def apply(self, X: Any, _y: Any, params: Dict[str, Any]) -> Any:
        return apply_dual_engine(X, params, _bucketing_apply_polars, _bucketing_apply_pandas)


# -----------------------------------------------------------------------------
# Fit helpers (per strategy) — pandas-only path, sklearn-bound for kbins/kmeans
# -----------------------------------------------------------------------------


def _fit_equal_width(series: pd.Series, n_bins: int) -> np.ndarray:
    _, edges = pd.cut(series, bins=n_bins, retbins=True)
    if len(edges) > 0 and edges[0] < series.min():
        edges[0] = series.min()
    return edges


def _fit_equal_frequency(series: pd.Series, n_bins: int, duplicates: str) -> np.ndarray:
    _, edges = pd.qcut(series, q=n_bins, labels=None, retbins=True, duplicates=duplicates)  # type: ignore[call-overload]
    if len(edges) > 0 and edges[0] < series.min():
        edges[0] = series.min()
    return edges


def _fit_kmeans(series: pd.Series, n_bins: int) -> np.ndarray:
    est = KBinsDiscretizer(n_bins=n_bins, strategy="kmeans", encode="ordinal")
    est.fit(np.asarray(series.values).reshape(-1, 1))
    return est.bin_edges_[0]


def _resolve_kbins_strategy(k_strategy: str) -> str:
    """Map UI-friendly aliases to sklearn's strategy names."""
    if k_strategy == "equal_width":
        return "uniform"
    if k_strategy == "equal_frequency":
        return "quantile"
    return k_strategy


def _fit_kbins(series: pd.Series, n_bins: int, k_strategy: str) -> np.ndarray:
    sklearn_strategy = _resolve_kbins_strategy(k_strategy)
    kwargs: Dict[str, Any] = {
        "n_bins": n_bins,
        "strategy": sklearn_strategy,
        "encode": "ordinal",
    }
    if sklearn_strategy == "quantile":
        kwargs["quantile_method"] = "averaged_inverted_cdf"
    est = KBinsDiscretizer(**kwargs)
    est.fit(np.asarray(series.values).reshape(-1, 1))
    return est.bin_edges_[0]


def _resolve_custom_edges(
    col: str, override: Dict[str, Any], config: Dict[str, Any]
) -> Tuple[Any, Any]:
    """Look up custom bins + labels from override-then-global config."""
    custom_bins = override.get("custom_bins") or config.get("custom_bins", {}).get(col)
    edges = np.array(sorted(custom_bins)) if custom_bins else None
    labels = override.get("custom_labels") or config.get("custom_labels", {}).get(col)
    return edges, labels


def _fit_one_column_edges(
    series: pd.Series,
    strategy: str,
    override: Dict[str, Any],
    config: Dict[str, Any],
    defaults: Dict[str, Any],
) -> Tuple[Any, Any]:
    """Dispatch one column to its strategy-specific fitter; returns ``(edges, labels)``."""
    default_n_bins = defaults["default_n_bins"]
    if strategy == "equal_width":
        return _fit_equal_width(series, override.get("equal_width_bins", defaults["n_bins"])), None
    if strategy == "equal_frequency":
        return (
            _fit_equal_frequency(
                series,
                override.get("equal_frequency_bins", defaults["q_bins"]),
                override.get("duplicates", defaults["duplicates"]),
            ),
            None,
        )
    if strategy == "kmeans":
        return _fit_kmeans(series, override.get("n_bins", default_n_bins)), None
    if strategy == "custom":
        return _resolve_custom_edges(str(series.name or ""), override, config)
    if strategy == "kbins":
        n_bins = override.get("kbins_n_bins", config.get("kbins_n_bins", default_n_bins))
        k_strategy = override.get("kbins_strategy", config.get("kbins_strategy", "quantile"))
        return _fit_kbins(series, n_bins, k_strategy), None
    return None, None


def _to_pandas_for_fit(X: Any) -> Any:
    """Always operate on a pandas frame in fit (sklearn-bound)."""
    engine = get_engine(X)
    if engine.name == EngineName.POLARS:
        return X.to_pandas()
    return X


def _passthrough_artifact_options(config: Dict[str, Any]) -> Dict[str, Any]:
    """Apply-time options that don't depend on the fit math."""
    return {
        "output_suffix": config.get("output_suffix", "_binned"),
        "drop_original": config.get("drop_original", False),
        "label_format": config.get("label_format", "ordinal"),
        "missing_strategy": config.get("missing_strategy", "keep"),
        "missing_label": config.get("missing_label", "Missing"),
        "include_lowest": config.get("include_lowest", True),
        "precision": config.get("precision", 3),
    }


# -----------------------------------------------------------------------------
# Calculators
# -----------------------------------------------------------------------------


class GeneralBinningApplier(BaseBinningApplier):
    pass


def _fit_one_column_into_maps(
    X: Any,
    col: str,
    config: Dict[str, Any],
    defaults: Dict[str, Any],
    bin_edges_map: Dict[str, List[float]],
    custom_labels_map: Dict[str, Any],
) -> None:
    """Compute edges/labels for one column and write them into the result maps."""
    column_strategies = config.get("column_strategies", {})
    global_strategy = config.get("strategy", "equal_width")
    override = column_strategies.get(col, {})
    strategy = override.get("strategy", global_strategy)
    try:
        series = X[col].dropna()
        if series.empty:
            return
        edges, labels = _fit_one_column_edges(series, strategy, override, config, defaults)
        if labels:
            custom_labels_map[col] = labels
        if edges is not None:
            bin_edges_map[col] = edges.tolist()
    except Exception:
        return


@NodeRegistry.register("GeneralBinning", GeneralBinningApplier)
@node_meta(
    id="GeneralBinning",
    name="General Binning",
    category="Preprocessing",
    description="Bin continuous data into intervals.",
    params={"n_bins": 5, "strategy": "uniform", "columns": []},
)
class GeneralBinningCalculator(BaseCalculator):
    """Master calculator that handles mixed strategies and per-column overrides."""

    @fit_method
    def fit(self, X: Any, _y: Any, config: Dict[str, Any]) -> GeneralBinningArtifact:
        if user_picked_no_columns(config):
            return cast(GeneralBinningArtifact, {})

        X = _to_pandas_for_fit(X)
        columns = resolve_columns(X, config, detect_numeric_columns)

        defaults = {
            "default_n_bins": config.get("n_bins", 5),
            "n_bins": config.get("equal_width_bins", config.get("n_bins", 5)),
            "q_bins": config.get("equal_frequency_bins", config.get("n_bins", 5)),
            "duplicates": config.get("duplicates", "drop"),
        }

        valid_cols = [c for c in columns if c in X.columns]
        bin_edges_map: Dict[str, List[float]] = {}
        custom_labels_map: Dict[str, Any] = {}

        for col in valid_cols:
            _fit_one_column_into_maps(X, col, config, defaults, bin_edges_map, custom_labels_map)

        artifact: Dict[str, Any] = {
            "type": "general_binning",
            "bin_edges": bin_edges_map,
            "custom_labels": custom_labels_map,
        }
        artifact.update(_passthrough_artifact_options(config))
        return cast(GeneralBinningArtifact, artifact)


class CustomBinningApplier(GeneralBinningApplier):
    pass


@NodeRegistry.register("CustomBinning", CustomBinningApplier)
@node_meta(
    id="CustomBinning",
    name="Custom Binning",
    category="Preprocessing",
    description="Bin data using custom edges.",
    params={"bins": [], "columns": []},
)
class CustomBinningCalculator(BaseCalculator):
    """Apply user-supplied bin edges to selected columns."""

    @fit_method
    def fit(self, X: Any, _y: Any, config: Dict[str, Any]) -> GeneralBinningArtifact:
        if user_picked_no_columns(config):
            return cast(GeneralBinningArtifact, {})

        X = _to_pandas_for_fit(X)
        columns = resolve_columns(X, config, detect_numeric_columns)
        bins = config.get("bins")

        bin_edges_map: Dict[str, List[float]] = {}
        if bins:
            sorted_bins = sorted(bins)
            for col in columns:
                if col in X.columns:
                    bin_edges_map[col] = sorted_bins

        artifact: Dict[str, Any] = {
            "type": "general_binning",  # Reuses GeneralBinningApplier.
            "bin_edges": bin_edges_map,
        }
        artifact.update(_passthrough_artifact_options(config))
        return cast(GeneralBinningArtifact, artifact)


class KBinsDiscretizerApplier(GeneralBinningApplier):
    pass


@NodeRegistry.register("KBinsDiscretizer", KBinsDiscretizerApplier)
@node_meta(
    id="KBinsDiscretizer",
    name="K-Bins Discretizer",
    category="Preprocessing",
    description="Bin continuous data into intervals using sklearn KBinsDiscretizer.",
    params={"n_bins": 5, "encode": "ordinal", "strategy": "quantile", "columns": []},
)
class KBinsDiscretizerCalculator(GeneralBinningCalculator):
    """Thin wrapper around :class:`GeneralBinningCalculator` with ``kbins`` strategy."""

    def fit(
        self,
        df: Union[pd.DataFrame, SkyulfDataFrame, Tuple[Any, ...], Any],
        config: Dict[str, Any],
    ) -> GeneralBinningArtifact:
        new_config = config.copy()
        new_config["strategy"] = "kbins"
        if "n_bins" in config:
            new_config["kbins_n_bins"] = config["n_bins"]
        if "strategy" in config and config["strategy"] != "kbins":
            new_config["kbins_strategy"] = config["strategy"]
        return super().fit(df, new_config)
