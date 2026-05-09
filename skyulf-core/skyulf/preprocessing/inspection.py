"""Inspection nodes (DatasetProfile, DataSnapshot).

Both nodes are read-only; the appliers are pure passthroughs. Only the fits
collect engine-specific summary statistics, so they route through
:func:`fit_dual_engine`.
"""

from typing import Any, Dict, Tuple, Union, cast

import pandas as pd

from ..registry import NodeRegistry
from ..core.meta.decorators import node_meta
from ..utils import detect_numeric_columns
from .base import BaseApplier, BaseCalculator, fit_method
from .dispatcher import fit_dual_engine
from ._artifacts import DatasetProfileArtifact, DataSnapshotArtifact
from ._schema import SkyulfSchema
from ..engines import SkyulfDataFrame


# -----------------------------------------------------------------------------
# DatasetProfile
# -----------------------------------------------------------------------------


def _extract_polars_numeric_stats(X: Any, numeric_cols: list) -> Dict[str, Dict[str, object]]:
    """Convert Polars ``describe()`` output into a per-column stats dict."""
    if not numeric_cols:
        return {}
    desc_df = X.select(numeric_cols).describe()
    stats: Dict[str, Dict[str, object]] = {col: {} for col in numeric_cols}
    for row in desc_df.to_dicts():
        # Polars < 0.19 uses "describe", newer versions use "statistic".
        metric = row.get("describe") or row.get("statistic")
        if not metric:
            continue
        for col in numeric_cols:
            if col in row:
                stats[col][metric] = row[col]
    return stats


def _profile_fit_polars(X: Any, _y: Any, _config: Dict[str, Any]) -> DatasetProfileArtifact:
    import polars as pl

    profile: Dict[str, Any] = {
        "rows": len(X),
        "columns": len(X.columns),
        "dtypes": {col: str(dtype) for col, dtype in zip(X.columns, X.dtypes)},
        "missing": {col: X[col].null_count() for col in X.columns},
    }
    numeric_cols = [
        col
        for col, dtype in zip(X.columns, X.dtypes)
        if dtype in (pl.Float64, pl.Float32, pl.Int64, pl.Int32)
    ]
    if numeric_cols:
        profile["numeric_stats"] = _extract_polars_numeric_stats(X, numeric_cols)
    return {"type": "dataset_profile", "profile": profile}


def _profile_fit_pandas(X: Any, _y: Any, _config: Dict[str, Any]) -> DatasetProfileArtifact:
    profile: Dict[str, Any] = {
        "rows": len(X),
        "columns": len(X.columns),
        "dtypes": X.dtypes.astype(str).to_dict(),
        "missing": X.isna().sum().to_dict(),
    }
    numeric_cols = detect_numeric_columns(X)
    if numeric_cols:
        profile["numeric_stats"] = X[numeric_cols].describe().to_dict()
    return {"type": "dataset_profile", "profile": profile}


class DatasetProfileApplier(BaseApplier):
    def apply(
        self,
        df: Union[pd.DataFrame, SkyulfDataFrame, Tuple[Any, ...], Any],
        params: Dict[str, Any],
    ) -> Any:
        # Inspection nodes do not modify data.
        return df


@NodeRegistry.register("DatasetProfile", DatasetProfileApplier)
@node_meta(
    id="DatasetProfile",
    name="Dataset Profile",
    category="Inspection",
    description="Generate a statistical profile of the dataset.",
    params={},
)
class DatasetProfileCalculator(BaseCalculator):
    def infer_output_schema(
        self, input_schema: SkyulfSchema, config: Dict[str, Any]
    ) -> SkyulfSchema:
        # Inspection nodes are read-only; schema is unchanged.
        return input_schema

    @fit_method
    def fit(self, X: Any, _y: Any, config: Dict[str, Any]) -> DatasetProfileArtifact:
        return cast(
            DatasetProfileArtifact,
            fit_dual_engine(X, config, _profile_fit_polars, _profile_fit_pandas),
        )


# -----------------------------------------------------------------------------
# DataSnapshot
# -----------------------------------------------------------------------------


def _snapshot_fit_polars(X: Any, _y: Any, config: Dict[str, Any]) -> DataSnapshotArtifact:
    n = config.get("n_rows", 5)
    return {"type": "data_snapshot", "snapshot": X.head(n).to_dicts()}


def _snapshot_fit_pandas(X: Any, _y: Any, config: Dict[str, Any]) -> DataSnapshotArtifact:
    n = config.get("n_rows", 5)
    return {"type": "data_snapshot", "snapshot": X.head(n).to_dict(orient="records")}


class DataSnapshotApplier(BaseApplier):
    def apply(
        self,
        df: Union[pd.DataFrame, SkyulfDataFrame, Tuple[Any, ...], Any],
        params: Dict[str, Any],
    ) -> Any:
        return df


@NodeRegistry.register("DataSnapshot", DataSnapshotApplier)
@node_meta(
    id="DataSnapshot",
    name="Data Snapshot",
    category="Inspection",
    description="Take a snapshot of the first N rows of the dataset.",
    params={"n_rows": 5},
)
class DataSnapshotCalculator(BaseCalculator):
    def infer_output_schema(
        self, input_schema: SkyulfSchema, config: Dict[str, Any]
    ) -> SkyulfSchema:
        # Inspection nodes are read-only; schema is unchanged.
        return input_schema

    @fit_method
    def fit(self, X: Any, _y: Any, config: Dict[str, Any]) -> DataSnapshotArtifact:
        return cast(
            DataSnapshotArtifact,
            fit_dual_engine(X, config, _snapshot_fit_polars, _snapshot_fit_pandas),
        )
