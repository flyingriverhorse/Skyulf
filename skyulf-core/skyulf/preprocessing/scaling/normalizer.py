"""Row normalization over the selected numeric feature vector."""

from typing import Any

from sklearn.preprocessing import Normalizer

from ...core.capabilities import ExecutionCapability
from ...core.meta.decorators import node_meta
from ...registry import NodeRegistry
from ...utils import user_picked_no_columns
from .._artifacts import NormalizerArtifact
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


class NormalizerApplier(BaseApplier):
    """Normalize each row with the saved norm and selected feature axes."""

    @staticmethod
    def validate_inference_state(raw: dict) -> dict:
        """Inspect the saved native normalizer without deriving request-dependent statistics."""
        transformer = validate_scaler_object(raw, "normalizer", Normalizer)
        if transformer is not None and transformer.norm not in ("l1", "l2", "max"):
            raise ValueError("Fitted Normalizer norm must be l1, l2 or max.")
        return raw

    @staticmethod
    def inference_capability(state: dict, *, engine: str) -> ExecutionCapability | None:
        """Describe local row normalization without granting worker execution."""
        if engine not in ("pandas", "polars"):
            return None
        NormalizerApplier.validate_inference_state(state)
        return ExecutionCapability(engine, "apply", "local", "preserve", "row")

    @apply_method
    def apply(self, X: Any, y: Any, params: dict[str, Any]) -> Any:
        """Normalize selected columns on the active engine while retaining paired labels."""
        return apply_dual_engine(
            (X, y) if y is not None else X,
            params,
            {"pandas": apply_scaler_pandas, "polars": apply_scaler_polars},
        )


@NodeRegistry.register("Normalizer", NormalizerApplier)
@node_meta(
    id="Normalizer",
    name="Normalizer",
    category="Preprocessing",
    description="Normalize each selected feature vector to unit norm.",
    params={"columns": [], "norm": "l2"},
    learns_from_data=False,
)
class NormalizerCalculator(BaseCalculator):
    """Validate the training matrix and save sklearn's norm and feature width."""

    def infer_output_schema(
        self, input_schema: SkyulfSchema, config: dict[str, Any]
    ) -> SkyulfSchema | None:
        """Keep all names and promote selected numeric features to float64."""
        return scaler_output_schema(input_schema, config)

    @fit_method
    def fit(self, X: Any, y: Any, config: dict[str, Any]) -> NormalizerArtifact:
        """Fit the native normalizer for consistent validation and saved feature axes."""
        if user_picked_no_columns(config):
            return {}
        values, columns = scaler_fit_values(X, config)
        if not columns:
            return {}
        transformer = Normalizer(norm=config.get("norm", "l2")).fit(values)
        return {"type": "normalizer", "columns": columns, "transformer_object": transformer}
