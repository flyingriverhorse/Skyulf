"""Polynomial-features node."""

from numbers import Integral
from typing import Any, cast

import numpy as np
import pandas as pd
import polars as pl
from sklearn.preprocessing import PolynomialFeatures

from ...core.capabilities import ExecutionCapability
from ...core.meta.decorators import node_meta
from ...registry import NodeRegistry
from ...utils import detect_numeric_columns
from .._artifacts import PolynomialFeaturesArtifact
from .._fitted_validation import local_boolean, local_state_fields
from .._helpers import select_then_to_numpy
from ..base import BaseApplier, BaseCalculator, apply_method, fit_method
from ..dispatcher import apply_dual_engine
from ._common import _validate_generated_names


def _validate_polynomial_degree(degree: Any, include_bias: bool) -> None:
    """Check saved sklearn degree bounds without constructing or fitting an estimator."""
    if isinstance(degree, Integral):
        bounds = (0, degree)
    elif type(degree) in (list, tuple, np.ndarray) and np.shape(degree) == (2,):
        bounds = degree
    else:
        raise ValueError("Fitted polynomial degree must be an integer or a pair of bounds.")
    if any(not isinstance(item, Integral) or isinstance(item, (bool, np.bool_)) for item in bounds):
        raise ValueError("Fitted polynomial degree bounds must be integers.")
    if not 0 <= bounds[0] <= bounds[1] or (bounds[1] == 0 and not include_bias):
        raise ValueError("Fitted polynomial degree bounds cannot produce an expansion.")


def _validate_polynomial_names(value: Any) -> None:
    """Accept fitted string names without coercing NumPy string scalars."""
    if type(value) is not list or any(not isinstance(name, str) for name in value):
        raise ValueError("Fitted polynomial names must be a list of strings.")


def _polynomial_compute(
    X_subset: Any, valid_cols: list[str], params: dict[str, Any]
) -> tuple[Any, list[str]] | None:
    """Run sklearn PolynomialFeatures + name normalisation; ``None`` ⇒ skip."""
    poly = PolynomialFeatures(
        degree=params.get("degree", 2),
        interaction_only=params.get("interaction_only", False),
        include_bias=params.get("include_bias", False),
    )
    # sklearn needs one row to derive its layout; only the feature count is fitted.
    values = X_subset if len(X_subset) else np.zeros((1, len(valid_cols)), dtype=X_subset.dtype)
    poly.fit(values)
    transformed = poly.transform(values)[: len(X_subset)]
    if hasattr(transformed, "to_numpy"):
        transformed = transformed.to_numpy()
    keep, new_names = _polynomial_names(poly, valid_cols, params)
    if not keep:
        return None
    return np.ascontiguousarray(transformed[:, keep]), new_names


def _polynomial_names(
    poly: PolynomialFeatures, columns: list[str], params: dict[str, Any]
) -> tuple[list[int], list[str]]:
    """Derive the exact emitted names without transforming data or changing stored metadata."""
    include_input = params.get("include_input_features", False)
    keep = [
        i for i, powers in enumerate(poly.powers_) if not (sum(powers) == 1 and not include_input)
    ]
    names = poly.get_feature_names_out(columns)[keep]
    prefix = params.get("output_prefix", "poly")
    return keep, [f"{prefix}_{name.replace(' ', '_').replace('^', '_pow_')}" for name in names]


def _polynomial_apply_polars(X: Any, _y: Any, params: dict[str, Any]) -> tuple[Any, Any]:
    valid_cols = [c for c in params.get("columns", []) if c in X.columns]
    if not valid_cols:
        return X, _y

    X_np, valid_cols = select_then_to_numpy(X, valid_cols)
    if not valid_cols:
        return X, _y

    result = _polynomial_compute(X_np, valid_cols, params)
    if result is None:
        return X, _y
    transformed, new_names = result
    _validate_generated_names(new_names, list(X.columns), "PolynomialFeatures")
    df_poly = pl.DataFrame(transformed, schema=new_names)
    return X.hstack(df_poly), _y


def _polynomial_apply_pandas(X: Any, _y: Any, params: dict[str, Any]) -> tuple[Any, Any]:
    valid_cols = [c for c in params.get("columns", []) if c in X.columns]
    if not valid_cols:
        return X, _y

    X_np, valid_cols = select_then_to_numpy(X, valid_cols)
    if not valid_cols:
        return X, _y

    result = _polynomial_compute(X_np, valid_cols, params)
    if result is None:
        return X, _y
    transformed, new_names = result
    _validate_generated_names(new_names, list(X.columns), "PolynomialFeatures")
    df_poly = pd.DataFrame(cast(Any, transformed), columns=cast(Any, new_names), index=X.index)
    return pd.concat(cast(Any, [X, df_poly]), axis=1), _y


class PolynomialFeaturesApplier(BaseApplier):
    """Append the polynomial expansion described by the artifact."""

    @staticmethod
    def validate_inference_state(raw: dict) -> dict:
        """Inspect saved expansion configuration without rebuilding or fitting its terms."""
        if not local_state_fields(
            raw,
            "polynomial_features",
            {
                "type",
                "columns",
                "degree",
                "interaction_only",
                "include_bias",
                "include_input_features",
                "output_prefix",
                "feature_names",
            },
            allow_empty=True,
        ):
            return raw
        _validate_polynomial_names(raw["columns"])
        if len(set(raw["columns"])) != len(raw["columns"]):
            raise ValueError("Fitted polynomial columns must be unique.")
        for flag in ("interaction_only", "include_bias", "include_input_features"):
            local_boolean(raw[flag], flag)
        _validate_polynomial_degree(raw["degree"], raw["include_bias"])
        if not isinstance(raw["output_prefix"], str):
            raise ValueError("Fitted polynomial prefix must be a string.")
        _validate_polynomial_names(raw["feature_names"])
        return raw

    @staticmethod
    def inference_capability(state: dict, *, engine: str) -> ExecutionCapability | None:
        """Describe local polynomial arithmetic without granting worker execution."""
        if engine not in ("pandas", "polars"):
            return None
        PolynomialFeaturesApplier.validate_inference_state(state)
        return ExecutionCapability(engine, "apply", "local", "preserve", "row")

    @apply_method
    def apply(self, X: Any, _y: Any, params: dict[str, Any]) -> Any:  # pylint: disable=arguments-differ
        """Recompute the expansion from ``X`` and append it as new columns.

        ``PolynomialFeatures`` is refit here rather than persisted, so the
        artifact only carries configuration. Artifact columns absent from ``X``
        are filtered out first, and the default ``include_input_features=False``
        drops the degree-1 terms so the originals are not duplicated. ``X`` is
        returned unchanged when nothing survives. Emitted names must be unique
        and must not collide with existing columns.
        """
        return apply_dual_engine(
            X, params, {"polars": _polynomial_apply_polars, "pandas": _polynomial_apply_pandas}
        )


@NodeRegistry.register("PolynomialFeatures", PolynomialFeaturesApplier)
@NodeRegistry.register("PolynomialFeaturesNode", PolynomialFeaturesApplier)
@node_meta(
    id="PolynomialFeatures",
    name="Polynomial Features",
    category="Feature Engineering",
    description="Generate polynomial and interaction features.",
    params={"degree": 2, "interaction_only": False, "include_bias": False},
    # Automatic column discovery inspects values; fixed selections are exempted.
    learns_from_data=True,
)
class PolynomialFeaturesCalculator(BaseCalculator):
    """Resolve the polynomial configuration and the names it will emit."""

    @fit_method
    def fit(self, X: Any, _y: Any, config: dict[str, Any]) -> PolynomialFeaturesArtifact:  # pylint: disable=arguments-differ
        """Record the expansion settings and the feature names they produce.

        ``columns`` falls back to auto-detected numeric columns when
        ``auto_detect`` is set. ``PolynomialFeatures`` is fitted here purely to
        derive ``feature_names`` — the expansion itself is recomputed at apply
        time. Returns an empty artifact, a no-op passthrough, when no column
        resolves. Ambiguous generated names raise ``ValueError`` before replay.
        """
        cols = list(config.get("columns", []))
        if not cols and config.get("auto_detect", False):
            # detect_numeric_columns dispatches natively on Polars frames too,
            # so this doesn't require converting the full frame first.
            cols = detect_numeric_columns(X)
        X_np, cols = select_then_to_numpy(X, cols)
        if not cols:
            return cast(PolynomialFeaturesArtifact, {})

        degree = config.get("degree", 2)
        interaction_only = config.get("interaction_only", False)
        include_bias = config.get("include_bias", False)

        poly = PolynomialFeatures(
            degree=degree, interaction_only=interaction_only, include_bias=include_bias
        )
        poly.fit(X_np)
        _, names = _polynomial_names(poly, cols, config)
        _validate_generated_names(names, list(X.columns), "PolynomialFeatures")
        return cast(
            PolynomialFeaturesArtifact,
            {
                "type": "polynomial_features",
                "columns": cols,
                "degree": degree,
                "interaction_only": interaction_only,
                "include_bias": include_bias,
                "include_input_features": config.get("include_input_features", False),
                "output_prefix": config.get("output_prefix", "poly"),
                "feature_names": poly.get_feature_names_out(cols).tolist(),
            },
        )
