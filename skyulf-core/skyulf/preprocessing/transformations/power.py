"""Power Transformer node (Box-Cox / Yeo-Johnson)."""

import logging
from typing import Any

import numpy as np
import pandas as pd
import polars as pl
from sklearn.preprocessing import PowerTransformer, StandardScaler

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
    select_rows_by_position,
)
from .._schema import SkyulfSchema
from ..base import BaseApplier, BaseCalculator, apply_method, fit_method
from ..dispatcher import apply_dual_engine
from ._power_common import build_pretrained_power_transformer, validate_power_parameters

logger = logging.getLogger(__name__)
_MAX_INVALID_TRAINING_PERCENT = 10


def _power_transform_array(
    X_vals: np.ndarray, params: dict[str, Any], cols: list[str], valid_cols: list[str]
) -> np.ndarray:
    """Transform independently, marking invalid values for the row filter."""
    values = np.array(X_vals, dtype=np.float64, copy=True)
    invalid = np.isinf(values)
    if params.get("method", "yeo-johnson") == "box-cox":
        invalid |= values <= 0
    values[invalid] = np.nan
    if np.isfinite(values).any():
        values = _transform_valid_power(values, params, cols, valid_cols)
    invalid |= ~np.isfinite(values)
    values[invalid] = np.nan
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


def _valid_training_values(X_pd: pd.DataFrame, cols: list[str], method: str) -> np.ndarray:
    """Exclude invalid rows before learning lambdas or standardization statistics."""
    values = decimal_columns_to_float(X_pd[cols], cols).to_numpy(dtype=np.float64, na_value=np.nan)
    valid = ~np.isinf(values).any(axis=1)
    if method == "box-cox":
        valid &= ~(values <= 0).any(axis=1)
    count = int(np.count_nonzero(~valid))
    if count:
        logger.warning("PowerTransformer fit excluded %d invalid row(s)", count)
    _check_training_exclusion(len(values), int(np.count_nonzero(valid)))
    if not valid.any():
        raise ValueError("PowerTransformer has no valid training rows for the selected columns.")
    values = values[valid]
    _require_power_observations(values, cols)
    return values


def _check_training_exclusion(input_rows: int, output_rows: int) -> None:
    """Stop training above the fixed invalid-row limit using the original row denominator."""
    invalid_rows = input_rows - output_rows
    if invalid_rows * 100 > input_rows * _MAX_INVALID_TRAINING_PERCENT:
        raise ValueError(
            f"PowerTransformer step would exclude {invalid_rows}/{input_rows} "
            f"({invalid_rows / input_rows:.2%}) invalid training rows, exceeding the fixed "
            f"{_MAX_INVALID_TRAINING_PERCENT}% limit. Use Yeo-Johnson for nonpositive "
            "Box-Cox inputs, or remove/repair invalid rows before this step. "
            "Missing NaN/null values do not count as invalid."
        )


def _require_power_observations(values: np.ndarray, cols: list[str]) -> None:
    """Require observed values per selected feature without deleting partially missing rows."""
    missing = [
        col
        for col, observed in zip(cols, np.isfinite(values).any(axis=0), strict=True)
        if not observed
    ]
    if missing:
        raise ValueError(
            f"PowerTransformer has no observed training values for columns: {missing}."
        )


def _valid_power_positions(values: np.ndarray, original: np.ndarray) -> np.ndarray:
    """Preserve missing inputs while excluding domain errors, infinities and new nonfinite values."""
    invalid = np.isinf(original) | (np.isfinite(original) & ~np.isfinite(values))
    valid = ~invalid.any(axis=1)
    count = int(np.count_nonzero(~valid))
    if count:
        logger.warning("PowerTransformer excluded %d invalid row(s)", count)
    return np.flatnonzero(valid)


def _validate_finite_power_state(state: dict, size: int) -> None:
    """Reject unusable learned parameters rather than treating every row as invalid."""
    validate_power_parameters(state, size)
    if not np.isfinite(state["lambdas"]).all():
        raise ValueError("Fitted PowerTransformer lambdas must be finite.")
    if state.get("standardize", True):
        scaler = state["scaler_params"]
        if not np.isfinite(scaler["mean"]).all() or not np.isfinite(scaler["scale"]).all():
            raise ValueError("Fitted PowerTransformer scaler statistics must be finite.")
        if not (np.asarray(scaler["scale"]) > 0).all():
            raise ValueError("Fitted PowerTransformer scale must be positive.")


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


def _fit_power_state(
    values: np.ndarray, cols: list[str], method: str, standardize: bool, input_rows: int
) -> PowerTransformerArtifact:
    """Fit lambdas first so overflowing transformed rows cannot contaminate the scaler."""
    transformer = PowerTransformer(method=method, standardize=False).fit(values)
    state: PowerTransformerArtifact = {
        "type": "power_transformer",
        "lambdas": transformer.lambdas_.tolist(),
        "method": method,
        "standardize": False,
        "columns": cols,
        "scaler_params": {},
    }
    transformed = _power_transform_array(values, dict(state), cols, cols)
    positions = _valid_power_positions(transformed, values)
    _check_training_exclusion(input_rows, len(positions))
    if not len(positions):
        raise ValueError("PowerTransformer has no valid training rows after transformation.")
    _require_power_observations(transformed[positions], cols)
    if standardize:
        scaler = StandardScaler().fit(transformed[positions])
        state["scaler_params"] = {"mean": scaler.mean_.tolist(), "scale": scaler.scale_.tolist()}
    state["standardize"] = standardize
    _validate_finite_power_state(dict(state), len(cols))
    if standardize:
        with np.errstate(over="ignore", invalid="ignore", divide="ignore"):
            standardized = (transformed - scaler.mean_) / scaler.scale_
        positions = _valid_power_positions(standardized, values)
        _check_training_exclusion(input_rows, len(positions))
    return state


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
            _validate_finite_power_state(raw, len(columns))
        return raw

    @staticmethod
    def inference_capability(state: dict, *, engine: str) -> ExecutionCapability | None:
        """Declare row-local filtering of invalid selected values and transformed outputs."""
        if engine not in ("pandas", "polars"):
            return None
        PowerTransformerApplier.validate_inference_state(state)
        return ExecutionCapability(
            engine, "apply", "local", "filter" if state else "preserve", "row"
        )

    @apply_method
    def apply(self, X: Any, _y: Any, params: dict[str, Any]) -> Any:  # pylint: disable=arguments-differ
        """Transform valid rows with stored lambdas and align ``y`` by position.

        Invalid numeric rows are excluded and counted in a warning. Missing
        values remain missing for later imputers. Valid
        neighbours keep their fitted transform. Missing fitted columns are
        warned about; malformed state and unexpected implementation errors raise.
        """
        if params.get("lambdas") is not None:
            _validate_finite_power_state(params, len(params["columns"]))
        return apply_dual_engine(
            (X, _y) if _y is not None else X,
            params,
            {"polars": self._apply_polars, "pandas": self._apply_pandas},
        )

    @staticmethod
    def _apply_polars(X: Any, _y: Any, params: dict[str, Any]) -> tuple[Any, Any]:
        cols = params.get("columns", [])
        if params.get("lambdas") is None:
            return X, _y
        valid_cols = _fitted_columns_present(X, cols)
        if not valid_cols:
            return X, _y

        X_vals = np.asarray(X.select(valid_cols).to_numpy(), dtype=np.float64)
        X_trans = _power_transform_array(X_vals, params, cols, valid_cols)
        series = [pl.Series(name, X_trans[:, i]) for i, name in enumerate(valid_cols)]
        positions = _valid_power_positions(X_trans, X_vals)
        return X.with_columns(series)[positions], select_rows_by_position(_y, positions)

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
        positions = _valid_power_positions(X_trans, X_vals)
        return df_out.iloc[positions], select_rows_by_position(_y, positions)


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
        """Fit selected columns on valid rows, excluding invalid rows from learned statistics."""
        if user_picked_no_columns(config):
            return {}

        method = config.get("method", "yeo-johnson")
        standardize = config.get("standardize", True)
        X_pd, cols = resolve_columns_then_to_pandas(X, config, detect_numeric_columns)
        if not cols:
            return {}
        values = _valid_training_values(X_pd, cols, method)
        return _fit_power_state(values, cols, method, standardize, len(X_pd))
