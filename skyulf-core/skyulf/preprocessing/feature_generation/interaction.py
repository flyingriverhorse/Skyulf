"""Feature-interaction node — automatic 2-way / 3-way / 4-way multiplicative interactions.

This node generates products of exactly the configured degree. By default,
only distinct columns participate; ``interaction_only=False`` also allows
repeated columns, including powers of a single input. Generated names are
deterministic so the same inputs always produce the same output column name.
"""

from collections.abc import Iterator
from itertools import combinations, combinations_with_replacement, zip_longest
from typing import Any, cast

import pandas as pd
import polars as pl

from ..._validation import raise_invalid_choice
from ...core.artifacts import FeatureInteractionArtifact
from ...core.capabilities import ExecutionCapability
from ...core.meta.decorators import node_meta
from ...core.portable_state import _normalize
from ...registry import NodeRegistry
from .._helpers import select_then_to_pandas
from ..base import BaseApplier, BaseCalculator, apply_method, fit_method
from ..dispatcher import apply_dual_engine
from ._common import _validate_generated_names

# Separator used between column names in generated interaction names. Avoids
# ``*``/spaces/other characters that break patsy/statsmodels formula parsing
# or common ML naming conventions.
_NAME_SEP = "_x_"
_SUPPORTED_DEGREES = (2, 3, 4)
_BIAS_COLUMN = "interaction_bias"
_DEFAULT_OPTIONS = {"degree": 2, "interaction_only": True, "include_bias": False}
_OPTIONS = {"columns", *_DEFAULT_OPTIONS}
_STATE_FIELDS = _OPTIONS | {"type", "combinations", "feature_names"}


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
    return sorted(_iter_combinations(columns, degree, interaction_only))


def _iter_combinations(
    columns: list[str], degree: int, interaction_only: bool
) -> Iterator[tuple[str, ...]]:
    """Share the product definition between fit and bounded saved-state inspection."""
    factory = combinations if interaction_only else combinations_with_replacement
    return factory(sorted(columns), degree)


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


def _build_interaction_exprs(X: Any, combos: list[tuple]) -> list:
    """Build polars expressions for each valid interaction combination, casting to Float64."""
    exprs = []
    for combo in combos:
        if not all(col in X.columns for col in combo):
            continue
        expr = pl.col(combo[0]).cast(pl.Float64)
        for col in combo[1:]:
            expr = expr * pl.col(col).cast(pl.Float64)
        exprs.append(expr.alias(_interaction_name(combo)))
    return exprs


def _interaction_apply_polars(X: Any, _y: Any, params: dict[str, Any]) -> tuple[Any, Any]:
    """Compute interaction columns and append them to a polars DataFrame."""
    combos = [tuple(c) for c in params.get("combinations", [])]
    exprs = _build_interaction_exprs(X, combos)

    if params.get("include_bias", False) and _BIAS_COLUMN not in X.columns:
        exprs.append(pl.repeat(1.0, pl.len()).alias(_BIAS_COLUMN))

    if not exprs:
        return X, _y
    return X.with_columns(exprs), _y


def _validate_interaction_columns(X_pd: pd.DataFrame, cols: list[str]) -> None:
    """Raise ValueError if any configured column is missing from X_pd or non-numeric."""
    missing = [c for c in cols if c not in X_pd.columns]
    if missing:
        raise ValueError(f"FeatureInteraction: columns not found in data: {missing}")

    non_numeric = [c for c in cols if not pd.api.types.is_numeric_dtype(X_pd[c])]
    if non_numeric:
        raise ValueError(f"FeatureInteraction requires numeric columns; non-numeric: {non_numeric}")


def _validate_interaction_degree(degree: int) -> None:
    """Raise ValueError if degree is not one of the supported interaction degrees."""
    if degree not in _SUPPORTED_DEGREES:
        raise_invalid_choice(degree, _SUPPORTED_DEGREES, "FeatureInteraction degree")


def _build_interaction_feature_names(
    combos: list[tuple[str, ...]], include_bias: bool
) -> list[str]:
    """Build output feature names for the given combinations, appending the bias column if set."""
    feature_names = [_interaction_name(c) for c in combos]
    if include_bias:
        feature_names.append(_BIAS_COLUMN)
    return feature_names


class FeatureInteractionApplier(BaseApplier):
    """Append the multiplicative interaction columns named by the artifact."""

    @staticmethod
    def validate_fitted_state(raw: dict) -> dict:
        """Inspect this node's supported saved state without fitting or applying data."""
        return interaction_state(raw)

    @staticmethod
    def resolve_fitted_config(raw: dict, state: dict) -> dict:
        """Bind inference configuration to this node's inspected fitted artifact."""
        return interaction_config(raw, state)

    @apply_method
    def apply(self, X: Any, _y: Any, params: dict[str, Any]) -> Any:  # pylint: disable=arguments-differ
        """Compute each artifact combination as a column product and append it.

        A combination whose columns are absent from ``X`` is skipped rather than
        raising, so the node survives an upstream column drop; ``include_bias``
        adds a constant 1.0 ``interaction_bias`` column per existing row,
        including zero rows when no input columns remain. When no column is
        generated the frame is returned unchanged. Generated product names must
        be unique and must not collide with existing columns. An existing bias
        column retains its established passthrough behavior.
        """
        combos = [
            tuple(combo)
            for combo in params.get("combinations", [])
            if all(col in X.columns for col in combo)
        ]
        names = _build_interaction_feature_names(
            combos, params.get("include_bias", False) and _BIAS_COLUMN not in X.columns
        )
        _validate_generated_names(names, list(X.columns), "FeatureInteraction")
        return apply_dual_engine(
            X, params, {"polars": _interaction_apply_polars, "pandas": _interaction_apply_pandas}
        )


@NodeRegistry.register(
    "FeatureInteraction",
    FeatureInteractionApplier,
    execution_capabilities=(
        ExecutionCapability("pandas", "apply", "python_batch", "preserve", "row"),
        ExecutionCapability("polars", "apply", "local", "preserve", "row"),
    ),
)
@node_meta(
    id="FeatureInteraction",
    name="Feature Interaction",
    category="Feature Engineering",
    description=(
        "Generate 2-way/3-way/4-way multiplicative interaction features between "
        "numeric columns, using deterministic regularization-friendly names."
    ),
    params={"columns": [], **_DEFAULT_OPTIONS},
    learns_from_data=False,
)
class FeatureInteractionCalculator(BaseCalculator):
    """Resolve which cross-products to generate, without computing them."""

    @fit_method
    def fit(self, X: Any, _y: Any, config: dict[str, Any]) -> FeatureInteractionArtifact:  # pylint: disable=arguments-differ
        """Validate the requested columns and degree, then enumerate the combinations.

        Columns are sorted into the artifact so the generated names depend only
        on the set of inputs, never on the order they were configured in.
        With ``interaction_only=True``, fewer columns than ``degree`` yields
        no products. With ``False``, one column is sufficient for self-products.
        An empty selection yields no products; bias generation is independent.

        Raises:
            ValueError: If a configured column is missing or non-numeric, or if
                ``degree`` falls outside the supported ``(2, 3, 4)``. Generated
                product names must also be unique and absent from the input.
        """
        cols = list(config.get("columns", []))
        X_pd = select_then_to_pandas(X, cols)

        _validate_interaction_columns(X_pd, cols)

        degree = config.get("degree", _DEFAULT_OPTIONS["degree"])
        _validate_interaction_degree(degree)
        interaction_only = config.get("interaction_only", _DEFAULT_OPTIONS["interaction_only"])
        include_bias = config.get("include_bias", _DEFAULT_OPTIONS["include_bias"])

        combos = _resolve_combinations(cols, degree, interaction_only)
        feature_names = _build_interaction_feature_names(combos, include_bias)
        generated = [
            name for name in feature_names if name != _BIAS_COLUMN or name not in X.columns
        ]
        _validate_generated_names(generated, list(X.columns), "FeatureInteraction")

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


def _columns(value: Any) -> list[str]:
    """Require distinct named inputs before comparing canonical product definitions."""
    if type(value) is not list or any(type(item) is not str or not item for item in value):
        raise ValueError("Interaction columns must be a list of nonempty strings.")
    if len(set(value)) != len(value):
        raise ValueError("Interaction columns must be unique.")
    return value


def _options(value: dict) -> dict:
    """Normalize input order while keeping degree and boolean switches type-exact."""
    if type(value["degree"]) is not int:
        raise ValueError("Interaction degree must be an integer from 2 through 4.")
    try:
        _validate_interaction_degree(value["degree"])
    except ValueError as exc:
        raise ValueError("Interaction degree must be an integer from 2 through 4.") from exc
    if any(type(value[key]) is not bool for key in ("interaction_only", "include_bias")):
        raise ValueError("Interaction switches must be booleans.")
    return {**value, "columns": sorted(_columns(value["columns"]))}


def interaction_state(raw: dict) -> dict:
    """Reject changed combinations, names, defaults and ignored state fields."""
    state = _normalize(raw)
    if type(state) is not dict or set(state) != _STATE_FIELDS:
        raise ValueError("Unexpected interaction state fields.")
    if state["type"] != "feature_interaction":
        raise ValueError("Wrong interaction artifact type.")
    options = _options({key: state[key] for key in _OPTIONS})
    if state["columns"] != options["columns"]:
        raise ValueError("Saved interaction columns must be canonically ordered.")
    _require_combinations(state, options)
    names = _build_interaction_feature_names(
        [tuple(combo) for combo in state["combinations"]], options["include_bias"]
    )
    if state["feature_names"] != names or len(set(names)) != len(names):
        raise ValueError("Interaction feature names disagree with the saved products.")
    return state


def _require_combinations(state: dict, options: dict) -> None:
    """Compare lazily so a malformed high-degree recipe cannot expand unbounded state."""
    actual = state["combinations"]
    if type(actual) is not list:
        raise ValueError("Interaction combinations must be a list.")
    expected = _iter_combinations(
        options["columns"], options["degree"], options["interaction_only"]
    )
    for saved, required in zip_longest(actual, expected):
        if required is None or saved != list(required):
            raise ValueError("Interaction combinations disagree with the configured products.")


def interaction_config(raw: dict, state: dict) -> dict:
    """Bind the configured columns and options to the inspected fitted state."""
    if type(raw) is not dict:
        raise ValueError("Interaction configuration must be a plain mapping.")
    params = _normalize(raw)
    params.pop("target_column", None)
    params.pop("_auto_columns", None)
    resolved = {"columns": [], **_DEFAULT_OPTIONS, **params}
    if set(resolved) != _OPTIONS:
        raise ValueError("Unexpected interaction configuration fields.")
    resolved = _options(resolved)
    if resolved != {key: state[key] for key in _OPTIONS}:
        raise ValueError("Interaction configuration disagrees with saved products.")
    return resolved
