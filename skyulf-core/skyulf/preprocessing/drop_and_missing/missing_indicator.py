"""Missing-indicator node (binary flags for missing values)."""

from typing import Any, cast

import polars as pl

from ...core.meta.decorators import node_meta
from ...registry import NodeRegistry
from .._artifacts import MissingIndicatorArtifact
from .._output_names import validate_generated_column_names
from .._schema import SkyulfSchema
from ..base import BaseApplier, BaseCalculator, apply_method
from ..dispatcher import apply_dual_engine, fit_dual_engine

_DEFAULT_FLAG_SUFFIX = "_missing"


def _validate_flag_names(X: Any, cols: list[str], suffix: str) -> None:
    """Protect retained inputs from the flags that present source columns emit."""
    validate_generated_column_names(
        X.columns,
        (f"{col}{suffix}" for col in cols if col in X.columns),
        node_name="MissingIndicator",
    )


def _missing_indicator_apply_polars(X: Any, y: Any, params: dict[str, Any]) -> tuple[Any, Any]:
    cols = params.get("columns", [])
    if not cols:
        return X, y
    suffix = params.get("flag_suffix") or _DEFAULT_FLAG_SUFFIX
    _validate_flag_names(X, cols, suffix)
    schema = X.schema
    exprs = []
    for c in cols:
        if c not in X.columns:
            continue
        missing = pl.col(c).is_null()
        if schema[c].is_float():
            # Polars stores float NaN distinctly from null; pandas' `isna()`
            # treats both as missing, so mirror that for parity.
            missing = missing | pl.col(c).is_nan()
        exprs.append(missing.cast(pl.Int64).alias(f"{c}{suffix}"))
    return (X.with_columns(exprs) if exprs else X), y


def _missing_indicator_apply_pandas(X: Any, y: Any, params: dict[str, Any]) -> tuple[Any, Any]:
    cols = params.get("columns", [])
    if not cols:
        return X, y
    suffix = params.get("flag_suffix") or _DEFAULT_FLAG_SUFFIX
    _validate_flag_names(X, cols, suffix)
    X_out = X.copy()
    for col in cols:
        if col in X.columns:
            # Keep the pandas result aligned with the explicit Polars Int64 cast
            # and the inferred output schema.
            X_out[f"{col}{suffix}"] = X[col].isna().astype("int64")
    return X_out, y


class MissingIndicatorApplier(BaseApplier):
    """Add a binary ``<col><flag_suffix>`` column beside each configured column.

    Additive, not replacing: the original columns survive. Both engines must
    flag the same cells, which needs care on polars — it stores float NaN
    distinctly from null where pandas' ``isna()`` treats both as missing, so
    float columns get an explicit NaN test. Columns absent from the frame are
    skipped. A generated flag name matching an existing column raises
    ``ValueError``; input values are never overwritten by flags.
    """

    @apply_method
    def apply(self, X: Any, _y: Any, params: dict[str, Any]) -> Any:  # pylint: disable=arguments-differ
        """Dispatch the flag creation to the pandas or polars path; ``y`` passes through."""
        return apply_dual_engine(
            X,
            params,
            {"polars": _missing_indicator_apply_polars, "pandas": _missing_indicator_apply_pandas},
        )


def _missing_cols_polars(X: Any) -> list:
    schema = X.schema
    exprs = []
    for c in X.columns:
        missing = pl.col(c).is_null()
        if schema[c].is_float():
            missing = missing | pl.col(c).is_nan()
        exprs.append(missing.any().alias(c))
    row = X.select(exprs).row(0) if X.width else ()
    return [c for c, has_missing in zip(X.columns, row, strict=True) if has_missing]


def _missing_cols_pandas(X: Any) -> list:
    return X.columns[X.isna().any()].tolist()


def _missing_indicator_fit_polars(
    X: Any, _y: Any, config: dict[str, Any]
) -> MissingIndicatorArtifact:
    explicit = config.get("columns")
    cols = [c for c in explicit if c in X.columns] if explicit else _missing_cols_polars(X)
    suffix = config.get("flag_suffix") or _DEFAULT_FLAG_SUFFIX
    _validate_flag_names(X, cols, suffix)
    return {
        "type": "missing_indicator",
        "columns": cols,
        "flag_suffix": suffix,
    }


def _missing_indicator_fit_pandas(
    X: Any, _y: Any, config: dict[str, Any]
) -> MissingIndicatorArtifact:
    explicit = config.get("columns")
    cols = [c for c in explicit if c in X.columns] if explicit else _missing_cols_pandas(X)
    suffix = config.get("flag_suffix") or _DEFAULT_FLAG_SUFFIX
    _validate_flag_names(X, cols, suffix)
    return {
        "type": "missing_indicator",
        "columns": cols,
        "flag_suffix": suffix,
    }


@NodeRegistry.register("MissingIndicator", MissingIndicatorApplier)
@node_meta(
    id="MissingIndicator",
    name="Missing Indicator",
    category="Feature Engineering",
    description="Create binary indicators for missing values.",
    params={"columns": [], "flag_suffix": _DEFAULT_FLAG_SUFFIX},
    learns_from_data=True,
)
class MissingIndicatorCalculator(BaseCalculator):
    """Choose which columns get a missing flag, learning the set from data when none is named.

    An explicit ``columns`` list is filtered down to names present on the
    frame; without one, every column that actually contains a missing value is
    flagged, so the artifact depends on the training data.
    """

    def infer_output_schema(
        self, input_schema: SkyulfSchema, config: dict[str, Any]
    ) -> SkyulfSchema | None:
        """Predict the added flag columns, or ``None`` when the set is data-dependent.

        With an explicit list the output is the input plus one ``int64``
        ``<col><flag_suffix>`` column for each source present in the input;
        otherwise which columns hold missing values decides, and callers
        must introspect at runtime.
        """
        # Adds one int64 (0/1) column "<col><flag_suffix>" per indicator
        # column. Only predictable when the user supplied an explicit column
        # list; otherwise the set depends on which columns actually contain
        # missing values.
        explicit = config.get("columns") or []
        if not explicit:
            return None
        suffix = config.get("flag_suffix") or _DEFAULT_FLAG_SUFFIX
        new_schema = input_schema
        for col in explicit:
            if col in input_schema:
                new_schema = new_schema.add(f"{col}{suffix}", "int64")
        return new_schema

    def fit(self, df: Any, config: dict[str, Any]) -> MissingIndicatorArtifact:
        """Compute the flagged column set on the frame's own engine and return it."""
        return cast(
            MissingIndicatorArtifact,
            fit_dual_engine(
                df,
                config,
                {"polars": _missing_indicator_fit_polars, "pandas": _missing_indicator_fit_pandas},
            ),
        )
