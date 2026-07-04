"""Polynomial-features node."""

from typing import Any, Dict, List, Optional, Tuple, cast

import pandas as pd
from sklearn.preprocessing import PolynomialFeatures

from ...core.meta.decorators import node_meta
from ...registry import NodeRegistry
from ...utils import detect_numeric_columns
from .._artifacts import PolynomialFeaturesArtifact
from .._helpers import to_pandas
from ..base import BaseApplier, BaseCalculator, apply_method, fit_method
from ..dispatcher import apply_dual_engine


def _polynomial_compute(
    X_subset: pd.DataFrame, valid_cols: List[str], params: Dict[str, Any]
) -> Optional[Tuple[Any, List[str]]]:
    """Run sklearn PolynomialFeatures + name normalisation; ``None`` ⇒ skip."""
    poly = PolynomialFeatures(
        degree=params.get("degree", 2),
        interaction_only=params.get("interaction_only", False),
        include_bias=params.get("include_bias", False),
    )
    poly.fit(X_subset)
    transformed = poly.transform(X_subset)
    if hasattr(transformed, "values"):
        transformed = transformed.values
    feature_names = poly.get_feature_names_out(valid_cols)

    include_input = params.get("include_input_features", False)
    keep = [i for i, p in enumerate(poly.powers_) if not (sum(p) == 1 and not include_input)]
    if not keep:
        return None

    transformed = transformed[:, keep]
    feature_names = feature_names[keep]
    output_prefix = params.get("output_prefix", "poly")
    new_names = [
        f"{output_prefix}_{name.replace(' ', '_').replace('^', '_pow_')}" for name in feature_names
    ]
    return transformed, new_names


def _polynomial_apply_polars(X: Any, _y: Any, params: Dict[str, Any]) -> Tuple[Any, Any]:
    import polars as pl

    valid_cols = [c for c in params.get("columns", []) if c in X.columns]
    if not valid_cols:
        return X, _y

    result = _polynomial_compute(X.select(valid_cols).to_pandas(), valid_cols, params)
    if result is None:
        return X, _y
    transformed, new_names = result
    df_poly = pl.DataFrame(transformed, schema=new_names)
    return pl.concat([X, df_poly], how="horizontal"), _y


def _polynomial_apply_pandas(X: Any, _y: Any, params: Dict[str, Any]) -> Tuple[Any, Any]:
    valid_cols = [c for c in params.get("columns", []) if c in X.columns]
    if not valid_cols:
        return X, _y

    result = _polynomial_compute(X[valid_cols], valid_cols, params)
    if result is None:
        return X, _y
    transformed, new_names = result
    df_poly = pd.DataFrame(cast(Any, transformed), columns=cast(Any, new_names), index=X.index)
    return pd.concat(cast(Any, [X, df_poly]), axis=1), _y


class PolynomialFeaturesApplier(BaseApplier):
    @apply_method
    def apply(self, X: Any, _y: Any, params: Dict[str, Any]) -> Any:  # pylint: disable=arguments-differ
        return apply_dual_engine(X, params, _polynomial_apply_polars, _polynomial_apply_pandas)


@NodeRegistry.register("PolynomialFeatures", PolynomialFeaturesApplier)
@NodeRegistry.register("PolynomialFeaturesNode", PolynomialFeaturesApplier)
@node_meta(
    id="PolynomialFeatures",
    name="Polynomial Features",
    category="Feature Engineering",
    description="Generate polynomial and interaction features.",
    params={"degree": 2, "interaction_only": False, "include_bias": False},
)
class PolynomialFeaturesCalculator(BaseCalculator):
    @fit_method
    def fit(self, X: Any, _y: Any, config: Dict[str, Any]) -> PolynomialFeaturesArtifact:  # pylint: disable=arguments-differ
        X_pd = to_pandas(X)
        cols = list(config.get("columns", []))
        if not cols and config.get("auto_detect", False):
            cols = detect_numeric_columns(X_pd)
        cols = [c for c in cols if c in X_pd.columns]
        if not cols:
            return cast(PolynomialFeaturesArtifact, {})

        degree = config.get("degree", 2)
        interaction_only = config.get("interaction_only", False)
        include_bias = config.get("include_bias", False)

        poly = PolynomialFeatures(
            degree=degree, interaction_only=interaction_only, include_bias=include_bias
        )
        poly.fit(X_pd[cols])
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
