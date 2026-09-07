"""Simple Transformation node (log, sqrt, square, etc.)."""

from typing import Any

import pandas as pd

from ...core.meta.decorators import node_meta
from ...engines import SkyulfDataFrame
from ...registry import NodeRegistry
from .._artifacts import SimpleTransformationArtifact
from .._schema import SkyulfSchema
from ..base import BaseApplier, BaseCalculator, apply_method
from ..dispatcher import apply_dual_engine
from ._ops import _PANDAS_OPS, _POLARS_OPS


class SimpleTransformationApplier(BaseApplier):
    """Apply the configured per-column math transformations (log, sqrt, ...)."""

    @apply_method
    def apply(self, X: Any, _y: Any, params: dict[str, Any]) -> Any:  # pylint: disable=arguments-differ
        """Run each configured transformation on the active engine; ``y`` passes through."""
        return apply_dual_engine(
            X, params, {"polars": self._apply_polars, "pandas": self._apply_pandas}
        )

    @staticmethod
    def _apply_polars(X: Any, _y: Any, params: dict[str, Any]) -> tuple[Any, Any]:
        transformations = params.get("transformations", [])
        if not transformations:
            return X, _y

        X_out = X
        for item in transformations:
            col = item.get("column")
            method = item.get("method")
            if col not in X_out.columns:
                continue
            op = _POLARS_OPS.get(method)
            if op is None:
                continue
            X_out = X_out.with_columns(op(item).alias(col))
        return X_out, _y

    @staticmethod
    def _apply_pandas(X: Any, _y: Any, params: dict[str, Any]) -> tuple[Any, Any]:
        transformations = params.get("transformations", [])
        if not transformations:
            return X, _y

        df_out = X.copy()
        for item in transformations:
            col = item.get("column")
            method = item.get("method")
            if col not in df_out.columns:
                continue
            op = _PANDAS_OPS.get(method)
            if op is None:
                continue
            df_out[col] = op(pd.to_numeric(df_out[col], errors="coerce"), item)
        return df_out, _y


@NodeRegistry.register("SimpleTransformation", SimpleTransformationApplier)
@node_meta(
    id="SimpleTransformation",
    name="Simple Transformation",
    category="Preprocessing",
    description="Apply simple mathematical transformations (log, sqrt, etc.).",
    params={"func": "log", "columns": []},
    learns_from_data=False,
)
class SimpleTransformationCalculator(BaseCalculator):
    """Package the configured transformations into an artifact without reading the data."""

    _FLOAT_METHODS = {
        "log",
        "sqrt",
        "square_root",
        "cube_root",
        "reciprocal",
        "exp",
        "exponential",
    }

    def infer_output_schema(
        self, input_schema: SkyulfSchema, config: dict[str, Any]
    ) -> SkyulfSchema:
        """Return updated dtypes for transformed numeric columns."""
        schema = input_schema
        touched: set[str] = set()
        for item in config.get("transformations", []):
            col = item.get("column")
            method = item.get("method")
            if col not in schema.columns or method is None:
                continue
            if method == "square":
                continue
            if method in self._FLOAT_METHODS:
                touched.add(col)

        for col in touched:
            schema = schema.with_dtype(col, "float64")
        return schema

    def fit(
        self,
        df: pd.DataFrame | SkyulfDataFrame | tuple[Any, ...] | Any,
        config: dict[str, Any],
    ) -> SimpleTransformationArtifact:
        """Return the ``transformations`` config as-is; nothing is fitted."""
        # Config: {'transformations': [{'column': 'col1', 'method': 'log'}, ...]}
        return {
            "type": "simple_transformation",
            "transformations": config.get("transformations", []),
        }
