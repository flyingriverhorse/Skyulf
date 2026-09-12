"""Inspection nodes (DatasetProfile, DataSnapshot).

Both nodes are read-only; the appliers are pure passthroughs. Only the fits
collect engine-specific summary statistics, so they route through
:func:`fit_dual_engine`.
"""

from typing import Any, cast

import pandas as pd

from ..core.meta.decorators import node_meta
from ..engines import SkyulfDataFrame
from ..registry import NodeRegistry
from ._artifacts import DatasetProfileArtifact, DataSnapshotArtifact
from ._helpers import auto_detect_numeric_columns, decimal_columns_to_float
from ._schema import SkyulfSchema
from .base import BaseApplier, BaseCalculator, fit_method
from .dispatcher import fit_dual_engine

# -----------------------------------------------------------------------------
# DatasetProfile
# -----------------------------------------------------------------------------


def _extract_polars_numeric_stats(X: Any, numeric_cols: list) -> dict[str, dict[str, object]]:
    """Convert Polars ``describe()`` output into a per-column stats dict."""
    if not numeric_cols:
        return {}
    desc_df = X.select(numeric_cols).describe()
    stats: dict[str, dict[str, object]] = {col: {} for col in numeric_cols}
    for row in desc_df.to_dicts():
        # Polars < 0.19 uses "describe", newer versions use "statistic".
        metric = row.get("describe") or row.get("statistic")
        if not metric:
            continue
        for col in numeric_cols:
            if col in row:
                stats[col][metric] = row[col]
    return stats


def _profile_fit_polars(X: Any, _y: Any, _config: dict[str, Any]) -> DatasetProfileArtifact:
    """Describe every supported numeric dtype, independent of observed cardinality."""
    profile: dict[str, Any] = {
        "rows": len(X),
        "columns": len(X.columns),
        "dtypes": {col: str(dtype) for col, dtype in zip(X.columns, X.dtypes, strict=True)},
        "missing": {col: X[col].null_count() for col in X.columns},
    }
    numeric_cols = auto_detect_numeric_columns(X)
    if numeric_cols:
        profile["numeric_stats"] = _extract_polars_numeric_stats(X, numeric_cols)
    return {"type": "dataset_profile", "profile": profile}


def _profile_fit_pandas(X: Any, _y: Any, _config: dict[str, Any]) -> DatasetProfileArtifact:
    """Describe numeric dtypes without filtering binary, constant or missing columns."""
    profile: dict[str, Any] = {
        "rows": len(X),
        "columns": len(X.columns),
        "dtypes": X.dtypes.astype(str).to_dict(),
        "missing": X.isna().sum().to_dict(),
    }
    # pandas includes timedeltas in select_dtypes("number"); they are temporal
    # columns, and Polars correctly leaves them outside numeric statistics.
    numeric_cols = auto_detect_numeric_columns(X.select_dtypes(exclude=["timedelta"]))
    if numeric_cols:
        profile["numeric_stats"] = (
            decimal_columns_to_float(X[numeric_cols], numeric_cols).describe().to_dict()
        )
    return {"type": "dataset_profile", "profile": profile}


class DatasetProfileApplier(BaseApplier):
    """Passthrough applier; the profile is produced entirely at fit time."""

    def apply(
        self,
        df: pd.DataFrame | SkyulfDataFrame | tuple[Any, ...] | Any,
        params: dict[str, Any],
    ) -> Any:
        """Return ``df`` unchanged, ignoring the fitted profile."""
        # Inspection nodes do not modify data.
        return df


@NodeRegistry.register("DatasetProfile", DatasetProfileApplier)
@node_meta(
    id="DatasetProfile",
    name="Dataset Profile",
    category="Inspection",
    description="Generate a statistical profile of the dataset.",
    params={},
    learns_from_data=False,
)
class DatasetProfileCalculator(BaseCalculator):
    """Summarise shape, dtypes, missingness and numeric statistics.

    Numeric statistics include supported integer, unsigned integer, float and
    Decimal columns, even when binary, constant, entirely missing or empty.
    Boolean, categorical and temporal columns retain their dtype/missingness
    metadata but are not treated as numeric features.
    """

    def infer_output_schema(
        self, input_schema: SkyulfSchema, config: dict[str, Any]
    ) -> SkyulfSchema:
        """Return ``input_schema`` untouched."""
        # Inspection nodes are read-only; schema is unchanged.
        return input_schema

    @fit_method
    def fit(self, X: Any, _y: Any, config: dict[str, Any]) -> DatasetProfileArtifact:  # pylint: disable=arguments-differ
        """Collect row/column counts, dtypes, missing counts and numeric stats.

        Numeric statistics come from Polars ``describe()`` or pandas
        ``describe()`` depending on the engine, so the per-column metric names
        are not identical across engines.
        """
        return cast(
            DatasetProfileArtifact,
            fit_dual_engine(
                X, config, {"polars": _profile_fit_polars, "pandas": _profile_fit_pandas}
            ),
        )


# -----------------------------------------------------------------------------
# DataSnapshot
# -----------------------------------------------------------------------------


def _snapshot_fit_polars(X: Any, _y: Any, config: dict[str, Any]) -> DataSnapshotArtifact:
    n = config.get("n_rows", 5)
    return {"type": "data_snapshot", "snapshot": X.head(n).to_dicts()}


def _snapshot_fit_pandas(X: Any, _y: Any, config: dict[str, Any]) -> DataSnapshotArtifact:
    n = config.get("n_rows", 5)
    return {"type": "data_snapshot", "snapshot": X.head(n).to_dict(orient="records")}


class DataSnapshotApplier(BaseApplier):
    """Passthrough applier; the snapshot is produced entirely at fit time."""

    def apply(
        self,
        df: pd.DataFrame | SkyulfDataFrame | tuple[Any, ...] | Any,
        params: dict[str, Any],
    ) -> Any:
        """Return ``df`` unchanged, ignoring the captured snapshot."""
        return df


@NodeRegistry.register("DataSnapshot", DataSnapshotApplier)
@node_meta(
    id="DataSnapshot",
    name="Data Snapshot",
    category="Inspection",
    description="Take a snapshot of the first N rows of the dataset.",
    params={"n_rows": 5},
    learns_from_data=False,
)
class DataSnapshotCalculator(BaseCalculator):
    """Capture the leading rows of the input as plain record dicts."""

    def infer_output_schema(
        self, input_schema: SkyulfSchema, config: dict[str, Any]
    ) -> SkyulfSchema:
        """Return ``input_schema`` untouched."""
        # Inspection nodes are read-only; schema is unchanged.
        return input_schema

    @fit_method
    def fit(self, X: Any, _y: Any, config: dict[str, Any]) -> DataSnapshotArtifact:  # pylint: disable=arguments-differ
        """Return the first ``config["n_rows"]`` rows (default 5) as dicts.

        Rows are serialised eagerly at fit time, so the snapshot is unaffected
        by later mutation of the source frame.
        """
        return cast(
            DataSnapshotArtifact,
            fit_dual_engine(
                X, config, {"polars": _snapshot_fit_polars, "pandas": _snapshot_fit_pandas}
            ),
        )
