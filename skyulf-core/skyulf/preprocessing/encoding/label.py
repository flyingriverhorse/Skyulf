"""Label Encoder node (Calculator + Applier)."""
import logging
from typing import Any, Dict, List, Tuple, Union

import numpy as np
import pandas as pd
from sklearn.preprocessing import LabelEncoder

from ...utils import (
    pack_pipeline_output,
    unpack_pipeline_input,
)
from ..base import BaseApplier, BaseCalculator
from ...core.meta.decorators import node_meta
from ...registry import NodeRegistry
from ...engines import EngineName, SkyulfDataFrame, get_engine

logger = logging.getLogger(__name__)


class LabelEncoderApplier(BaseApplier):
    def apply(
        self,
        df: Union[pd.DataFrame, SkyulfDataFrame, Tuple[Any, ...]],
        params: Dict[str, Any],
    ) -> Any:
        X, y, is_tuple = unpack_pipeline_input(df)
        engine = get_engine(X)

        encoders = params.get("encoders", {})
        cols = params.get("columns")

        if engine.name == EngineName.POLARS:
            import polars as pl

            X_pl: Any = X
            X_out = X_pl.clone()
            y_out = y.clone() if y is not None else None

            if cols:
                missing_code = params.get("missing_code", -1)
                exprs = []
                for col in cols:
                    if col in X_out.columns and col in encoders:
                        le = encoders[col]
                        mapping = {
                            str(k): int(v)
                            for k, v in zip(le.classes_, le.transform(le.classes_))
                        }
                        exprs.append(
                            pl.col(col)
                            .cast(pl.Utf8)
                            .replace(mapping, default=missing_code)
                            .cast(pl.Int64)
                            .alias(col)
                        )
                if exprs:
                    X_out = X_out.with_columns(exprs)

            if y_out is not None and "__target__" in encoders:
                le = encoders["__target__"]
                mapping = {
                    str(k): int(v) for k, v in zip(le.classes_, le.transform(le.classes_))
                }
                missing_code = params.get("missing_code", -1)
                y_out = (
                    y_out.cast(pl.Utf8).replace(mapping, default=missing_code).cast(pl.Int64)
                )

        else:
            X_out = X.copy()
            y_out = y.copy() if y is not None else None

            if cols:
                for col in cols:
                    if col in X_out.columns and col in encoders:
                        le = encoders[col]
                        mapping = dict(zip(le.classes_, le.transform(le.classes_)))
                        missing_code = params.get("missing_code", -1)
                        X_out[col] = X_out[col].astype(str).map(mapping).fillna(missing_code)

            if y_out is not None and "__target__" in encoders:
                le = encoders["__target__"]
                mapping = dict(zip(le.classes_, le.transform(le.classes_)))
                missing_code = params.get("missing_code", -1)
                y_out = y_out.astype(str).map(mapping).fillna(missing_code)

        return pack_pipeline_output(X_out, y_out, is_tuple)


@NodeRegistry.register("LabelEncoder", LabelEncoderApplier)
@node_meta(
    id="LabelEncoder",
    name="Label Encoder",
    category="Preprocessing",
    description="Encode target labels with value between 0 and n_classes-1.",
    params={"columns": [], "missing_code": -1},
)
class LabelEncoderCalculator(BaseCalculator):
    def fit(
        self,
        df: Union[pd.DataFrame, SkyulfDataFrame, Tuple[Any, ...]],
        config: Dict[str, Any],
    ) -> Dict[str, Any]:
        X, y, _ = unpack_pipeline_input(df)
        engine = get_engine(X)

        target_col = config.get("target_column")
        if y is None and target_col:
            if engine.name == EngineName.POLARS:
                X_pl: Any = X
                if target_col in X_pl.columns:
                    y = X_pl.get_column(target_col)
            else:
                if target_col in X.columns:
                    y = X[target_col]

        cols: List[str] | None = config.get("columns")
        encoders: Dict[str, Any] = {}
        classes_count: Dict[str, int] = {}

        if cols:
            if engine.name == EngineName.POLARS:
                import polars as pl

                X_pl_data: Any = X
                valid_cols = [c for c in cols if c in X_pl_data.columns]
                for col in valid_cols:
                    le = LabelEncoder()
                    col_data = (
                        X_pl_data.select(pl.col(col).cast(pl.Utf8)).to_series().to_numpy()
                    )
                    le.fit(col_data)
                    encoders[col] = le
                    classes_count[col] = len(le.classes_)
            else:
                valid_cols = [c for c in cols if c in X.columns]
                for col in valid_cols:
                    le = LabelEncoder()
                    le.fit(X[col].astype(str))
                    encoders[col] = le
                    classes_count[col] = len(le.classes_)

            if y is not None:
                y_name = getattr(y, "name", None)
                if y_name and y_name in cols:
                    le = LabelEncoder()
                    y_data = (
                        y.to_numpy().astype(str)
                        if hasattr(y, "to_numpy")
                        else np.array(y).astype(str)
                    )
                    le.fit(y_data)
                    encoders["__target__"] = le
                    classes_count["__target__"] = len(le.classes_)

        else:
            if y is not None:
                le = LabelEncoder()
                y_data = (
                    y.to_numpy().astype(str)
                    if hasattr(y, "to_numpy")
                    else np.array(y).astype(str)
                )
                le.fit(y_data)
                encoders["__target__"] = le
                classes_count["__target__"] = len(le.classes_)

        return {
            "type": "label_encoder",
            "encoders": encoders,
            "columns": cols,
            "classes_count": classes_count,
            "missing_code": config.get("missing_code", -1),
        }


__all__ = ["LabelEncoderApplier", "LabelEncoderCalculator"]
