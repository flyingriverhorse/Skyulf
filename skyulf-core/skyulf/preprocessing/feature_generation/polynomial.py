"""Polynomial-features node."""

from typing import Any, cast

import numpy as np
import pandas as pd
import polars as pl
from sklearn.preprocessing import PolynomialFeatures

from ...core.meta.decorators import node_meta
from ...registry import NodeRegistry
from ...utils import detect_numeric_columns
from .._artifacts import PolynomialFeaturesArtifact
from .._helpers import select_then_to_numpy
from ..base import BaseApplier, BaseCalculator, apply_method, fit_method
from ..dispatcher import apply_dual_engine
from ._common import _validate_generated_names


def _polynomial_compute(
    X_subset: Any, valid_cols: list[str], params: dict[str, Any]
) -> tuple[Any, list[str]] | None:
    """Run sklearn PolynomialFeatures + name normalisation; ``None`` ⇒ skip."""
    poly = PolynomialFeatures(
        degree=params.get("degree", 2),
        interaction_only=params.get("interaction_only", False),
        include_bias=params.get("include_bias", False),
    )
    poly.fit(X_subset)
    transformed = poly.transform(X_subset)
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
