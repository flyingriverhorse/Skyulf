"""Quantile transformation using the training distribution saved by sklearn."""

from typing import Any

import numpy as np
from sklearn.preprocessing import QuantileTransformer

from ...core.capabilities import ExecutionCapability
from ...core.meta.decorators import node_meta
from ...registry import NodeRegistry
from ...types import DEFAULT_RANDOM_STATE
from ...utils import user_picked_no_columns
from .._artifacts import QuantileTransformerArtifact
from .._schema import SkyulfSchema
from ..base import BaseApplier, BaseCalculator, apply_method, fit_method
from ..dispatcher import apply_dual_engine
from ._sklearn import (
    apply_scaler_pandas,
    apply_scaler_polars,
    scaler_fit_values,
    scaler_output_schema,
    validate_scaler_object,
)


def _validate_quantile_matrix(transformer: QuantileTransformer, width: int) -> None:
    """Bind the saved quantile matrix to its learned count and selected feature axes."""
    count = getattr(transformer, "n_quantiles_", None)
    quantiles = getattr(transformer, "quantiles_", None)
    if (
        not isinstance(count, (int, np.integer))
        or isinstance(count, (bool, np.bool_))
        or count < 1
        or not isinstance(quantiles, np.ndarray)
        or quantiles.shape != (count, width)
        or quantiles.dtype.kind != "f"
    ):
        raise ValueError("Fitted quantile statistics disagree with the feature axes.")
    if np.isinf(quantiles).any():
        raise ValueError("Fitted quantiles cannot contain infinite limits.")


def _validate_quantile_references(transformer: QuantileTransformer) -> None:
    """Require finite probability references aligned to the saved quantile matrix."""
    references = getattr(transformer, "references_", None)
    if (
        not isinstance(references, np.ndarray)
        or references.shape != (transformer.n_quantiles_,)
        or references.dtype.kind != "f"
        or not np.isfinite(references).all()
    ):
        raise ValueError("Fitted quantile references must align with finite quantile limits.")
    if transformer.output_distribution not in ("uniform", "normal"):
        raise ValueError("Fitted quantile output_distribution must be uniform or normal.")


class QuantileTransformerApplier(BaseApplier):
    """Replay the learned marginal distributions without fitting on inference requests."""

    @staticmethod
    def validate_inference_state(raw: dict) -> dict:
        """Inspect the native saved transformer and its learned numeric axes."""
        transformer = validate_scaler_object(raw, "quantile_transformer", QuantileTransformer)
        if transformer is not None:
            _validate_quantile_matrix(transformer, len(raw["columns"]))
            _validate_quantile_references(transformer)
        return raw

    @staticmethod
    def inference_capability(state: dict, *, engine: str) -> ExecutionCapability | None:
        """Describe local row-preserving replay without granting worker execution."""
        if engine not in ("pandas", "polars"):
            return None
        QuantileTransformerApplier.validate_inference_state(state)
        return ExecutionCapability(engine, "apply", "local", "preserve", "row")

    @apply_method
    def apply(self, X: Any, y: Any, params: dict[str, Any]) -> Any:
        """Transform saved columns on the request's engine and preserve paired labels."""
        return apply_dual_engine(
            (X, y) if y is not None else X,
            params,
            {"pandas": apply_scaler_pandas, "polars": apply_scaler_polars},
        )


@NodeRegistry.register("QuantileTransformer", QuantileTransformerApplier)
@node_meta(
    id="QuantileTransformer",
    name="Quantile Transformer",
    category="Preprocessing",
    description="Map numeric features to a uniform or normal marginal distribution.",
    params={
        "columns": [],
        "n_quantiles": 1000,
        "output_distribution": "uniform",
        "subsample": 10000,
        "random_state": DEFAULT_RANDOM_STATE,
    },
    learns_from_data=True,
)
class QuantileTransformerCalculator(BaseCalculator):
    """Fit sklearn quantiles on the selected numeric training features only."""

    def infer_output_schema(
        self, input_schema: SkyulfSchema, config: dict[str, Any]
    ) -> SkyulfSchema | None:
        """Keep the column set and promote selected features to float64."""
        return scaler_output_schema(input_schema, config)

    @fit_method
    def fit(self, X: Any, y: Any, config: dict[str, Any]) -> QuantileTransformerArtifact:
        """Learn deterministic quantiles; sklearn caps their count at the training row count."""
        if user_picked_no_columns(config):
            return {}
        values, columns = scaler_fit_values(X, config)
        if not columns:
            return {}
        for name in ("n_quantiles", "subsample"):
            if isinstance(config.get(name), (bool, np.bool_)):
                raise ValueError(
                    f"QuantileTransformer {name} must be a positive integer, not bool."
                )
        transformer = QuantileTransformer(
            n_quantiles=config.get("n_quantiles", 1000),
            output_distribution=config.get("output_distribution", "uniform"),
            subsample=config.get("subsample", 10000),
            random_state=config.get("random_state", DEFAULT_RANDOM_STATE),
        ).fit(values)
        return {
            "type": "quantile_transformer",
            "columns": columns,
            "transformer_object": transformer,
        }
