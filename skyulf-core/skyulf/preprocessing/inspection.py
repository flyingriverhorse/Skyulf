from typing import Any, Dict, Tuple, Union

import pandas as pd

from ..registry import NodeRegistry
from ..core.meta.decorators import node_meta
from ..utils import detect_numeric_columns
from .base import BaseApplier, BaseCalculator, fit_method
from ._artifacts import DatasetProfileArtifact, DataSnapshotArtifact
from ._schema import SkyulfSchema
from ..engines import EngineName, SkyulfDataFrame, get_engine


class DatasetProfileApplier(BaseApplier):
    def apply(
        self,
        df: Union[pd.DataFrame, SkyulfDataFrame, Tuple[Any, ...], Any],
        params: Dict[str, Any],
    ) -> Any:
        # Inspection nodes do not modify data
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
    def fit(
        self,
        X: Any,
        _y: Any,
        config: Dict[str, Any],
    ) -> DatasetProfileArtifact:
        # Generate a lightweight dataset profile
        engine = get_engine(X)

        profile: Dict[str, Any] = {}

        if engine.name == EngineName.POLARS:
            import polars as pl

            profile["rows"] = len(X)
            profile["columns"] = len(X.columns)
            profile["dtypes"] = {col: str(dtype) for col, dtype in zip(X.columns, X.dtypes)}
            profile["missing"] = {col: X[col].null_count() for col in X.columns}

            # Numeric stats
            numeric_cols = [
                col
                for col, dtype in zip(X.columns, X.dtypes)
                if dtype in [pl.Float64, pl.Float32, pl.Int64, pl.Int32]
            ]
            if numeric_cols:
                # Polars describe returns a DataFrame
                desc_df = X.select(numeric_cols).describe()
                # Convert to list of dicts
                desc_dicts = desc_df.to_dicts()

                stats: dict[str, dict[str, object]] = {}
                # Initialize stats dicts
                for col in numeric_cols:
                    stats[col] = {}

                for row in desc_dicts:
                    # Polars < 0.19 uses "describe", newer uses "statistic"
                    metric = row.get("describe") or row.get("statistic")
                    if not metric:
                        continue

                    for col in numeric_cols:
                        if col in row:
                            stats[col][metric] = row[col]

                profile["numeric_stats"] = stats
        else:
            # Pandas logic
            profile["rows"] = len(X)
            profile["columns"] = len(X.columns)
            profile["dtypes"] = X.dtypes.astype(str).to_dict()
            profile["missing"] = X.isna().sum().to_dict()

            numeric_cols = detect_numeric_columns(X)
            if numeric_cols:
                desc = X[numeric_cols].describe().to_dict()
                profile["numeric_stats"] = desc

        return {"type": "dataset_profile", "profile": profile}


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
    def fit(
        self,
        X: Any,
        _y: Any,
        config: Dict[str, Any],
    ) -> DataSnapshotArtifact:
        engine = get_engine(X)

        n = config.get("n_rows", 5)

        if engine.name == EngineName.POLARS:
            snapshot = X.head(n).to_dicts()
        else:
            snapshot = X.head(n).to_dict(orient="records")

        return {"type": "data_snapshot", "snapshot": snapshot}
