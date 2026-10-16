"""Power Transformer node (Box-Cox / Yeo-Johnson)."""

import logging
from typing import Any

import numpy as np
import pandas as pd
import polars as pl
from sklearn.preprocessing import PowerTransformer

from ...core.capabilities import ExecutionCapability
from ...core.meta.decorators import node_meta
from ...registry import NodeRegistry
from ...utils import detect_numeric_columns, user_picked_no_columns
from .._artifacts import PowerTransformerArtifact
from .._fitted_validation import local_state_fields
from .._helpers import (
    decimal_columns_to_float,
    promote_configured_columns_to_float64,
    resolve_columns_then_to_pandas,
)
from .._schema import SkyulfSchema
from ..base import BaseApplier, BaseCalculator, apply_method, fit_method
from ..dispatcher import apply_dual_engine
from ._power_common import build_pretrained_power_transformer, validate_power_parameters

logger = logging.getLogger(__name__)


def _power_transform_array(
    X_vals: np.ndarray, params: dict[str, Any], cols: list[str], valid_cols: list[str]
) -> np.ndarray:
    """Transform valid values independently, replacing invalid inputs/results with NaN."""
    values = np.array(X_vals, dtype=np.float64, copy=True)
    invalid = np.isinf(values)
    if params.get("method", "yeo-johnson") == "box-cox":
        invalid |= values <= 0
    values[invalid] = np.nan
    if np.isfinite(values).any():
        values = _transform_valid_power(values, params, cols, valid_cols)
    invalid |= np.isinf(values)
    values[invalid] = np.nan
    count = int(np.count_nonzero(invalid))
    if count:
        logger.warning("PowerTransformer replaced %d invalid value(s) with NaN", count)
    return values


def _transform_valid_power(
    values: np.ndarray, params: dict[str, Any], cols: list[str], valid_cols: list[str]
) -> np.ndarray:
    """Apply saved lambdas then scaler arithmetic without batch-wide finite-output checks."""
    col_indices = [cols.index(c) for c in valid_cols]
    lambdas_arr = np.array(params["lambdas"])[col_indices]
    pt = build_pretrained_power_transformer(
        method=params.get("method", "yeo-johnson"),
        standardize=False,
        lambdas_arr=lambdas_arr,
        scaler_params=None,
    )
    with np.errstate(over="ignore", invalid="ignore", divide="ignore"):
        transformed = np.asarray(pt.transform(values))
        if params.get("standardize", True):
            scaler = params["scaler_params"]
            transformed = (transformed - np.asarray(scaler["mean"])[col_indices]) / np.asarray(
                scaler["scale"]
            )[col_indices]
    return transformed


def _filter_power_columns(X_pd: pd.DataFrame, cols: list[str], method: str) -> list[str]:
    """Box-Cox requires strictly positive data — drop columns that violate it."""
    if not cols:
        return []
    if method == "box-cox":
        invalid = [c for c in cols if (X_pd[c] <= 0).any()]
        if invalid:
            logger.warning("Box-Cox excludes columns containing non-positive values: %s", invalid)
        return [c for c in cols if c not in invalid]
    return cols


def _fitted_columns_present(X: Any, cols: list[str]) -> list[str]:
    """Return the fitted columns that ``X`` actually has, warning about the rest.

    A lambda only means something for the column it was fitted on. When the
    apply-time frame lacks one, that column passes through untransformed while
    the node still reports success — train/serve skew, not a no-op — so it is
    named in the log rather than filtered out quietly. Shared by both engine
    paths so they cannot drift apart.
    """
    present = [c for c in cols if c in X.columns]
    missing = [c for c in cols if c not in X.columns]
    if missing:
        logger.warning(
            "PowerTransformer was fitted on column(s) %r that this frame does not have; "
            "they pass through untransformed",
            missing,
        )
    return present


def _extract_scaler_params(transformer: PowerTransformer, standardize: bool) -> dict[str, Any]:
    """Pull mean/scale arrays out of a fitted PowerTransformer's internal scaler."""
    if not standardize:
        return {}
    scaler = getattr(transformer, "_scaler", None)
    if scaler is None:
        return {}
    return {
        "mean": scaler.mean_.tolist() if scaler.mean_ is not None else None,
        "scale": scaler.scale_.tolist() if scaler.scale_ is not None else None,
    }


class PowerTransformerApplier(BaseApplier):
    """Apply the fitted Box-Cox / Yeo-Johnson transform to the selected columns."""

    @staticmethod
    def validate_inference_state(raw: dict) -> dict:
        """Inspect the saved lambdas and scaler without normalizing or relearning them."""
        if local_state_fields(
            raw,
            "power_transformer",
            {"type", "columns", "method", "standardize", "lambdas", "scaler_params"},
            allow_empty=True,
        ):
            columns = raw["columns"]
            if type(columns) is not list or any(not isinstance(col, str) for col in columns):
                raise ValueError("Fitted power columns must be a list of strings.")
            validate_power_parameters(raw, len(columns))
        return raw

    @staticmethod
    def inference_capability(state: dict, *, engine: str) -> ExecutionCapability | None:
        """Describe per-value invalid-to-NaN handling without executing the transform."""
        if engine not in ("pandas", "polars"):
            return None
        PowerTransformerApplier.validate_inference_state(state)
        return ExecutionCapability(engine, "apply", "local", "preserve", "row")

    @apply_method
    def apply(self, X: Any, _y: Any, params: dict[str, Any]) -> Any:  # pylint: disable=arguments-differ
        """Rebuild the transformer from stored lambdas and transform ``X``; ``y`` untouched.

        Invalid numeric values become NaN and are counted in a warning. Valid
        neighbours keep their fitted transform. Missing fitted columns are
        warned about; malformed state and unexpected implementation errors raise.
        """
        return apply_dual_engine(
            X, params, {"polars": self._apply_polars, "pandas": self._apply_pandas}
        )

    @staticmethod
    def _apply_polars(X: Any, _y: Any, params: dict[str, Any]) -> tuple[Any, Any]:
        cols = params.get("columns", [])
        if params.get("lambdas") is None:
            return X, _y
        valid_cols = _fitted_columns_present(X, cols)
        if not valid_cols:
            return X, _y

        X_vals = X.select(valid_cols).to_numpy()
        X_trans = _power_transform_array(X_vals, params, cols, valid_cols)
        series = [pl.Series(name, X_trans[:, i]) for i, name in enumerate(valid_cols)]
        return X.with_columns(series), _y

    @staticmethod
    def _apply_pandas(X: Any, _y: Any, params: dict[str, Any]) -> tuple[Any, Any]:
        cols = params.get("columns", [])
        if params.get("lambdas") is None:
            return X, _y
        valid_cols = _fitted_columns_present(X, cols)
        if not valid_cols:
            return X, _y

        subset = decimal_columns_to_float(X[valid_cols], valid_cols)
        X_vals = subset.to_numpy(dtype=np.float64, na_value=np.nan)
        X_trans = _power_transform_array(X_vals, params, cols, valid_cols)
        df_out = X.copy()
        # Replace selected columns so nullable integers and Decimal objects
        # can receive float results without incompatible-dtype assignment.
        df_out[valid_cols] = X_trans
        return df_out, _y


@NodeRegistry.register("PowerTransformer", PowerTransformerApplier)
@node_meta(
    id="PowerTransformer",
    name="Power Transformer",
    category="Preprocessing",
    description="Apply a power transform featurewise to make data more Gaussian-like.",
    params={"method": "yeo-johnson", "standardize": True, "columns": []},
    learns_from_data=True,
)
class PowerTransformerCalculator(BaseCalculator):
    """Fit one shared PowerTransformer across the selected columns."""

    def infer_output_schema(
        self, input_schema: SkyulfSchema, config: dict[str, Any]
    ) -> SkyulfSchema:
        """Return a schema with transformed columns promoted to ``float64``."""
        return promote_configured_columns_to_float64(input_schema, config)

    @fit_method
    def fit(self, X: Any, _y: Any, config: dict[str, Any]) -> PowerTransformerArtifact:  # pylint: disable=arguments-differ
        """Fit per-column lambdas, dropping columns Box-Cox cannot accept."""
        if user_picked_no_columns(config):
            return {}

        method = config.get("method", "yeo-johnson")
        standardize = config.get("standardize", True)
        # TODO(pandas-removal): PowerTransformer.fit accepts numpy directly,
        # but _filter_power_columns() uses Pandas column-wise boolean
        # indexing (`X_pd[c] <= 0`) to drop non-positive columns for box-cox.
        # That check is easy to port to numpy (X_np[:, i] <= 0), but do it
        # together with a matching rewrite of _filter_power_columns to take
        # an ndarray, so both stay in sync.
        X_pd, cols = resolve_columns_then_to_pandas(X, config, detect_numeric_columns)
        valid_cols = _filter_power_columns(X_pd, cols, method)
        if not valid_cols:
            return {}

        transformer = PowerTransformer(method=method, standardize=standardize)
        transformer.fit(X_pd[valid_cols])

        return {
            "type": "power_transformer",
            "lambdas": transformer.lambdas_.tolist(),
            "method": method,
            "standardize": standardize,
            "columns": valid_cols,
            "scaler_params": _extract_scaler_params(transformer, standardize),
        }
