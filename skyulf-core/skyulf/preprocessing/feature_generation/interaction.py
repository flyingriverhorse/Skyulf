"""Feature-interaction node — automatic 2-way / 3-way / 4-way multiplicative interactions.

Unlike :mod:`.polynomial` (which also generates squared/cubed terms), this
node focuses purely on cross-products between distinct columns and names the
resulting columns with a regularization-friendly, deterministic scheme so the
same combination of inputs always produces the same output column name.
"""

from itertools import combinations, combinations_with_replacement
from typing import Any, cast

import pandas as pd

from ...core.artifacts import FeatureInteractionArtifact
from ...core.meta.decorators import node_meta
from ...registry import NodeRegistry
from .._helpers import to_pandas
from ..base import BaseApplier, BaseCalculator, apply_method, fit_method
from ..dispatcher import apply_dual_engine

# Separator used between column names in generated interaction names. Avoids
# ``*``/spaces/other characters that break patsy/statsmodels formula parsing
# or common ML naming conventions.
_NAME_SEP = "_x_"
_SUPPORTED_DEGREES = (2, 3, 4)
_BIAS_COLUMN = "interaction_bias"


def _interaction_name(columns: tuple[str, ...]) -> str:
    """Build a deterministic, regularization-friendly interaction column name.

    Columns are joined in sorted order with ``_x_`` so the same combination
    of inputs always yields the same name, regardless of the order the
    columns were configured/requested in.

    Args:
        columns: Column names participating in the interaction.

    Returns:
        A name such as ``"x1_x_x2"`` (2-way), ``"x1_x_x2_x_x3"`` (3-way), or
        ``"x1_x_x2_x_x3_x_x4"`` (4-way).
    """
    return _NAME_SEP.join(sorted(columns))


def _resolve_combinations(
    columns: list[str], degree: int, interaction_only: bool
) -> list[tuple[str, ...]]:
    """Resolve the sorted column combinations to multiply for ``degree``.

    Args:
        columns: Candidate numeric columns (already validated to exist).
        degree: 2 for pairwise, 3 for three-way, or 4 for four-way interactions.
        interaction_only: If ``True``, skip self-products (e.g. ``x1 * x1``)
            by using combinations without replacement.

    Returns:
        A sorted list of column-name tuples, one per generated interaction.
    """
    sorted_cols = sorted(columns)
    combo_fn = combinations if interaction_only else combinations_with_replacement
    return sorted(combo_fn(sorted_cols, degree))


def _multiply_columns_pandas(X: pd.DataFrame, combo: tuple[str, ...]) -> pd.Series:
    """Element-wise product of the columns in ``combo`` (pandas engine)."""
    result = X[combo[0]].astype(float)
    for col in combo[1:]:
        result = result * X[col].astype(float)
    return result


def _interaction_apply_pandas(X: Any, _y: Any, params: dict[str, Any]) -> tuple[Any, Any]:
    """Compute interaction columns and append them to a pandas DataFrame."""
    combos = [tuple(c) for c in params.get("combinations", [])]
    new_cols: dict[str, pd.Series] = {}
    for combo in combos:
        if not all(col in X.columns for col in combo):
            continue
        new_cols[_interaction_name(combo)] = _multiply_columns_pandas(X, combo)

    if params.get("include_bias", False) and _BIAS_COLUMN not in X.columns:
        new_cols[_BIAS_COLUMN] = pd.Series(1.0, index=X.index)

    if not new_cols:
        return X, _y
    return pd.concat([X, pd.DataFrame(new_cols, index=X.index)], axis=1), _y


def _interaction_apply_polars(X: Any, _y: Any, params: dict[str, Any]) -> tuple[Any, Any]:
    """Compute interaction columns and append them to a polars DataFrame."""
    import polars as pl

    combos = [tuple(c) for c in params.get("combinations", [])]
    exprs = []
    for combo in combos:
        if not all(col in X.columns for col in combo):
            continue
        expr = pl.col(combo[0]).cast(pl.Float64)
        for col in combo[1:]:
            expr = expr * pl.col(col).cast(pl.Float64)
        exprs.append(expr.alias(_interaction_name(combo)))

    if params.get("include_bias", False) and _BIAS_COLUMN not in X.columns:
        exprs.append(pl.lit(1.0).alias(_BIAS_COLUMN))

    if not exprs:
        return X, _y
    return X.with_columns(exprs), _y


class FeatureInteractionApplier(BaseApplier):
    @apply_method
    def apply(self, X: Any, _y: Any, params: dict[str, Any]) -> Any:  # pylint: disable=arguments-differ
        return apply_dual_engine(X, params, _interaction_apply_polars, _interaction_apply_pandas)


@NodeRegistry.register("FeatureInteraction", FeatureInteractionApplier)
@node_meta(
    id="FeatureInteraction",
    name="Feature Interaction",
    category="Feature Engineering",
    description=(
        "Generate 2-way/3-way/4-way multiplicative interaction features between "
        "numeric columns, using deterministic regularization-friendly names."
    ),
    params={"columns": [], "degree": 2, "interaction_only": True, "include_bias": False},
)
class FeatureInteractionCalculator(BaseCalculator):
    @fit_method
    def fit(self, X: Any, _y: Any, config: dict[str, Any]) -> FeatureInteractionArtifact:  # pylint: disable=arguments-differ
        X_pd = to_pandas(X)
        cols = list(config.get("columns", []))

        missing = [c for c in cols if c not in X_pd.columns]
        if missing:
            raise ValueError(f"FeatureInteraction: columns not found in data: {missing}")

        non_numeric = [c for c in cols if not pd.api.types.is_numeric_dtype(X_pd[c])]
        if non_numeric:
            raise ValueError(
                f"FeatureInteraction requires numeric columns; non-numeric: {non_numeric}"
            )

        degree = config.get("degree", 2)
        if degree not in _SUPPORTED_DEGREES:
            raise ValueError(f"FeatureInteraction only supports degree 2, 3 or 4, got {degree}")
        interaction_only = config.get("interaction_only", True)
        include_bias = config.get("include_bias", False)

        combos = (
            _resolve_combinations(cols, degree, interaction_only) if len(cols) >= degree else []
        )
        feature_names = [_interaction_name(c) for c in combos]
        if include_bias:
            feature_names.append(_BIAS_COLUMN)

        return cast(
            FeatureInteractionArtifact,
            {
                "type": "feature_interaction",
                "columns": sorted(cols),
                "degree": degree,
                "interaction_only": interaction_only,
                "include_bias": include_bias,
                "combinations": [list(c) for c in combos],
                "feature_names": feature_names,
            },
        )
