"""One-Hot Encoder node (Calculator + Applier)."""

import logging
from collections.abc import Mapping
from typing import Any, cast

import numpy as np
import pandas as pd
import polars as pl
from sklearn.preprocessing import OneHotEncoder

from ...core.capabilities import ExecutionCapability
from ...core.meta.decorators import node_meta
from ...core.portable_state import _normalize
from ...engines.sklearn_bridge import SklearnBridge
from ...registry import NodeRegistry
from ...utils import resolve_columns, user_picked_no_columns
from .._artifacts import OneHotArtifact
from .._fitted_validation import _columns, _fields, _scalar, fitted_columns
from .._output_names import validate_generated_column_names
from ..base import BaseApplier, BaseCalculator, apply_method, fit_method
from ..dispatcher import apply_dual_engine, fit_dual_engine
from ._common import _exclude_target_column, detect_categorical_columns

logger = logging.getLogger(__name__)

_MISSING_TOKEN = "__mlops_missing__"  # nosec B105 - sentinel value, not a credential
_ESCAPE_PREFIX = "__mlops_literal__:"
_DEFAULT_OPTIONS = {
    "drop_first": False,
    "max_categories": 20,
    "handle_unknown": "ignore",
    "prefix_separator": "_",
    "drop_original": True,
    "include_missing": False,
}


def _uses_missing_encoding(params: dict[str, Any]) -> bool:
    """Validate the fitted policy while leaving unversioned artifacts on legacy replay."""
    if not params or "missing_encoding_version" not in params:
        return False
    version = params["missing_encoding_version"]
    if type(version) is not int or version != 1:
        raise ValueError(
            f"Unsupported OneHotEncoder missing encoding version {version!r}; refit the encoder."
        )
    return True


def _escape_literal(value: Any) -> Any:
    """Keep literal missing tokens and escape-prefixed strings distinct from missing values."""
    if isinstance(value, str) and (value == _MISSING_TOKEN or value.startswith(_ESCAPE_PREFIX)):
        return _ESCAPE_PREFIX + value
    return value


def _prepare_missing_pandas(subset: pd.DataFrame, escape_literals: bool) -> pd.DataFrame:
    """Escape reserved text before replacing nulls without coercing numeric categories."""
    prepared = subset.astype(object)
    if escape_literals:
        for column in prepared.columns:
            prepared[column] = pd.Series(
                [_escape_literal(value) for value in prepared[column]],
                index=prepared.index,
                dtype=object,
            )
    return prepared.where(prepared.notna(), _MISSING_TOKEN)


def _escape_literal_expr(column: str) -> pl.Expr:
    """Express the string escape policy natively for Polars string and categorical columns."""
    values = pl.col(column).cast(pl.String)
    reserved = (values == _MISSING_TOKEN) | values.str.starts_with(_ESCAPE_PREFIX)
    return (
        pl.when(reserved)
        .then(pl.concat_str(pl.lit(_ESCAPE_PREFIX), values))
        .otherwise(values)
        .alias(column)
    )


def _prepare_missing_polars(subset: pl.DataFrame, escape_literals: bool) -> pl.DataFrame:
    """Escape only textual categories and retain the legacy fill policy for other dtypes."""
    if escape_literals:
        subset = subset.with_columns(
            [
                _escape_literal_expr(column)
                for column, dtype in subset.schema.items()
                if dtype.base_type() in (pl.String, pl.Categorical, pl.Enum)
            ]
        )
    return subset.fill_null(_MISSING_TOKEN)


# -----------------------------------------------------------------------------
# Apply
# -----------------------------------------------------------------------------


def _validate_apply_params(X: Any, params: dict[str, Any]) -> tuple[list[str], Any, Any]:
    """Resolve ``(valid_cols, encoder, feature_names)`` or sentinel values for early-out."""
    if not params or not params.get("columns"):
        return [], None, None
    cols = params["columns"]
    encoder = params.get("encoder_object")
    feature_names = params.get("feature_names")
    valid_cols = [c for c in cols if c in X.columns]
    if not valid_cols or not encoder:
        return [], None, None
    validate_generated_column_names(
        X.columns,
        cast(list[str], feature_names),
        dropped_columns=valid_cols if params.get("drop_original", True) else (),
        node_name="OneHotEncoder",
    )
    return valid_cols, encoder, feature_names


def _to_dense(encoded_array: Any) -> Any:
    """Densify sparse sklearn output and unwrap pandas wrappers."""
    if hasattr(encoded_array, "toarray"):
        return encoded_array.toarray()
    if hasattr(encoded_array, "to_numpy"):
        return encoded_array.to_numpy()
    return encoded_array


def _transform_partition(encoder: Any, values: Any, feature_names: list[str]) -> Any:
    """Preserve fitted output schema for an empty fit-only test partition."""
    if len(values) == 0:
        return np.empty((0, len(feature_names)), dtype=encoder.dtype)
    return _to_dense(encoder.transform(values))


def _onehot_apply_polars(X: Any, y: Any, params: dict[str, Any]) -> tuple[Any, Any]:
    escape_literals = _uses_missing_encoding(params)
    valid_cols, encoder, feature_names = _validate_apply_params(X, params)
    if not valid_cols:
        return X, y

    drop_original = params.get("drop_original", True)
    include_missing = params.get("include_missing", False)

    X_subset = X.select(valid_cols)
    if include_missing:
        X_subset = _prepare_missing_polars(X_subset, escape_literals)

    X_np, _ = SklearnBridge.to_sklearn(X_subset)
    encoded = _transform_partition(encoder, X_np, feature_names)

    encoded_df = pl.DataFrame(encoded, schema=feature_names)
    retained = X.drop(valid_cols) if drop_original else X
    X_out = retained.hstack(encoded_df) if retained.width else encoded_df
    return X_out, y


def _onehot_apply_pandas(X: Any, y: Any, params: dict[str, Any]) -> tuple[Any, Any]:
    escape_literals = _uses_missing_encoding(params)
    valid_cols, encoder, feature_names = _validate_apply_params(X, params)
    if not valid_cols:
        return X, y

    drop_original = params.get("drop_original", True)
    include_missing = params.get("include_missing", False)
    X_out = X.copy()
    X_subset = X_out[valid_cols]
    if include_missing:
        X_subset = _prepare_missing_pandas(X_subset, escape_literals)

    X_input = X_subset.to_numpy() if hasattr(X_subset, "to_numpy") else X_subset
    encoded = _transform_partition(encoder, X_input, feature_names)
    encoded_df = pd.DataFrame(encoded, columns=feature_names, index=X_out.index)
    if drop_original:
        X_out = X_out.drop(columns=valid_cols)
    X_out = pd.concat(cast(Any, [X_out, encoded_df]), axis=1)
    return X_out, y


class OneHotEncoderApplier(BaseApplier):
    """Expand each categorical column into its one-hot indicator columns.

    Parity here comes from both engines transforming through the *same* fitted
    sklearn encoder carried in the artifact, not from two parallel
    implementations. Nulls are filled with the sentinel token before
    transforming, which only lines up when ``include_missing`` was set at fit
    time too — the encoder knows that level solely if it saw it. Indicator
    columns are concatenated on and the originals dropped unless
    ``drop_original`` is false. Generated names must be unique and must not
    collide with retained input columns; conflicts raise ``ValueError``.
    New artifacts escape literal missing tokens and escape-prefixed strings;
    unversioned artifacts retain their original missing-value transformation.
    """

    @staticmethod
    def validate_fitted_state(raw: dict) -> dict:
        """Inspect this node's supported saved state without fitting or applying data."""
        return _onehot_state(raw)

    @staticmethod
    def resolve_fitted_config(raw: dict, state: dict) -> dict:
        """Bind inference configuration to this node's inspected fitted artifact."""
        return _onehot_config(fitted_columns(raw, state), state)

    @apply_method
    def apply(self, X: Any, y: Any, params: dict[str, Any]) -> Any:  # pylint: disable=arguments-differ
        """Dispatch to the engine-specific transform, forwarding ``(X, y)`` only when present."""
        return apply_dual_engine(
            (X, y) if y is not None else X,
            params,
            {"polars": _onehot_apply_polars, "pandas": _onehot_apply_pandas},
        )


# -----------------------------------------------------------------------------
# Fit
# -----------------------------------------------------------------------------


def _resolve_fit_options(config: dict[str, Any]) -> dict[str, Any]:
    """Pull the per-call options out of ``config`` once, with defaults."""
    options = {**_DEFAULT_OPTIONS, **config}
    return {
        "drop": "first" if options["drop_first"] else None,
        "max_categories": options["max_categories"],
        "handle_unknown": "ignore" if options["handle_unknown"] == "ignore" else "error",
        "prefix_separator": options["prefix_separator"],
        "drop_original": options["drop_original"],
        "include_missing": options["include_missing"],
    }


def _warn_degenerate_categories(encoder: OneHotEncoder, cols: list[str], drop: Any) -> None:
    """Log warnings for empty or single-category columns when relevant."""
    if not hasattr(encoder, "categories_"):
        return
    for i, col in enumerate(cols):
        n_cats = len(encoder.categories_[i])
        if n_cats == 0:
            logger.warning(
                f"OneHotEncoder: Column '{col}' has 0 categories "
                "(empty or all missing). It will be dropped."
            )
        elif drop == "first" and n_cats == 1:
            logger.warning(
                f"OneHotEncoder: Column '{col}' has only 1 category "
                f"('{encoder.categories_[i][0]}') and 'Drop First' is enabled. "
                "This results in 0 encoded features."
            )


def _fit_sklearn_onehot(X_subset: Any, opts: dict[str, Any], cols: list[str]) -> OneHotEncoder:
    """Run the sklearn ``OneHotEncoder.fit`` step on a prepared subset."""
    X_np, _ = SklearnBridge.to_sklearn(X_subset)
    encoder = OneHotEncoder(
        drop=opts["drop"],
        max_categories=opts["max_categories"],
        handle_unknown=opts["handle_unknown"],
        sparse_output=False,
        dtype=np.int8,
    )
    encoder.fit(X_np)
    _warn_degenerate_categories(encoder, cols, opts["drop"])
    return encoder


def _build_onehot_artifact(
    X: Any, encoder: OneHotEncoder, cols: list[str], opts: dict[str, Any]
) -> Mapping[str, Any]:
    """Validate the fitted feature names before returning the reusable artifact."""
    feature_names = encoder.get_feature_names_out(cols).tolist()
    validate_generated_column_names(
        X.columns,
        feature_names,
        dropped_columns=cols if opts["drop_original"] else (),
        node_name="OneHotEncoder",
    )
    artifact = {
        "type": "onehot",
        "columns": cols,
        "encoder_object": encoder,
        "feature_names": feature_names,
        "prefix_separator": opts["prefix_separator"],
        "drop_original": opts["drop_original"],
        "include_missing": opts["include_missing"],
    }
    if opts["include_missing"]:
        artifact["missing_encoding_version"] = 1
    return artifact


def _onehot_fit_polars(X: Any, y: Any, config: dict[str, Any]) -> Mapping[str, Any]:
    cols = resolve_columns(X, config, detect_categorical_columns)
    cols = _exclude_target_column(cols, config, "OneHotEncoder", y)
    if not cols:
        return {}

    opts = _resolve_fit_options(config)
    X_subset = X.select(cols)
    if opts["include_missing"]:
        X_subset = _prepare_missing_polars(X_subset, escape_literals=True)

    encoder = _fit_sklearn_onehot(X_subset, opts, cols)
    return _build_onehot_artifact(X, encoder, cols, opts)


def _onehot_fit_pandas(X: Any, y: Any, config: dict[str, Any]) -> Mapping[str, Any]:
    cols = resolve_columns(X, config, detect_categorical_columns)
    cols = _exclude_target_column(cols, config, "OneHotEncoder", y)
    if not cols:
        return {}

    opts = _resolve_fit_options(config)
    X_subset = X[cols]
    if opts["include_missing"]:
        X_subset = _prepare_missing_pandas(X_subset, escape_literals=True)

    encoder = _fit_sklearn_onehot(X_subset, opts, cols)
    return _build_onehot_artifact(X, encoder, cols, opts)


@NodeRegistry.register(
    "OneHotEncoder",
    OneHotEncoderApplier,
    execution_capabilities=(
        ExecutionCapability(
            "pandas",
            "apply",
            "python_batch",
            "preserve",
            "row",
            config_match=(("max_categories", None),),
        ),
        ExecutionCapability("pandas", "apply", "local", "preserve", "row"),
        ExecutionCapability("polars", "apply", "local", "preserve", "row"),
    ),
)
@node_meta(
    id="OneHotEncoder",
    name="One-Hot Encoder",
    category="Preprocessing",
    description="Encodes categorical features as a one-hot numeric array.",
    params={
        "handle_unknown": "ignore",
        "drop_first": False,
        "max_categories": 20,
        "columns": [],
        "include_missing": False,
    },
    learns_from_data=True,
)
class OneHotEncoderCalculator(BaseCalculator):
    """Fit a sklearn ``OneHotEncoder`` on the resolved categorical columns.

    The target column is excluded before fitting: one-hot replaces the column
    it encodes with several derived ones, which would break a downstream
    Feature/Target Split. An explicit ``columns: []``, or auto-detection
    finding no categorical columns, yields an empty artifact so the applier
    no-ops. Degenerate columns (zero categories, or one category with
    ``drop_first``) are warned about rather than silently producing nothing.
    """

    @fit_method
    def fit(self, X: Any, y: Any, config: dict[str, Any]) -> OneHotArtifact:  # pylint: disable=arguments-differ
        """Short-circuit an explicit empty column selection, else fit on the frame's own engine."""
        if user_picked_no_columns(config):
            return {}
        return cast(
            OneHotArtifact,
            fit_dual_engine(
                (X, y) if y is not None else X,
                config,
                {"polars": _onehot_fit_polars, "pandas": _onehot_fit_pandas},
            ),
        )


__all__ = ["OneHotEncoderApplier", "OneHotEncoderCalculator"]


def _check_encoder(encoder: Any, columns: list[str]) -> None:
    """Admit the exact dense fitted sklearn encoder without arbitrary callables."""
    if type(encoder) is not OneHotEncoder:
        raise ValueError("Expected exact OneHotEncoder.")
    expected = {
        "categories",
        "sparse_output",
        "dtype",
        "handle_unknown",
        "drop",
        "min_frequency",
        "max_categories",
        "feature_name_combiner",
        "_infrequent_enabled",
        "n_features_in_",
        "categories_",
        "_drop_idx_after_grouping",
        "drop_idx_",
        "_n_features_outs",
    }
    if vars(encoder).get("max_categories") is not None:
        expected.update({"_infrequent_indices", "_default_to_infrequent_mappings"})
    _fields(vars(encoder), expected)
    _encoder_options(encoder)
    if encoder.drop not in (None, "first") or encoder.n_features_in_ != len(columns):
        raise ValueError("Encoder feature count or dropped-category policy disagrees.")
    _encoder_categories(encoder, columns)


def _encoder_options(encoder: Any) -> None:
    """Restrict inspected state to deterministic dense encoding with an optional category cap."""
    if encoder.dtype is not np.int8 or encoder.feature_name_combiner != "concat":
        raise ValueError("Unsupported encoder dtype or feature-name callback.")
    if encoder.categories != "auto" or encoder.sparse_output is not False:
        raise ValueError("Encoder requires automatic categories and dense output.")
    _encoder_grouping_options(encoder)
    if encoder.handle_unknown not in ("ignore", "error"):
        raise ValueError("Unsupported encoder inference policy.")


def _encoder_grouping_options(encoder: Any) -> None:
    """Bind the grouping flag to the supported positive Python integer cap."""
    cap = encoder.max_categories
    if cap is not None and (type(cap) is not int or cap < 1):
        raise ValueError("Encoder category cap must be a positive integer or None.")
    if encoder.min_frequency is not None:
        raise ValueError("Frequency thresholds require separate context review.")
    if encoder._infrequent_enabled is not (cap is not None):
        raise ValueError("Encoder grouping flag disagrees with the category cap.")


def _encoder_categories(encoder: Any, columns: list[str]) -> None:
    """Validate category widths and dropped indices before calling known name generation."""
    if type(encoder.categories_) is not list or len(encoder.categories_) != len(columns):
        raise ValueError("Encoder categories must align with columns.")
    counts = []
    for categories in encoder.categories_:
        if type(categories) is not np.ndarray or categories.ndim != 1 or not len(categories):
            raise ValueError("Encoder categories must be nonempty vectors.")
        values = _normalize(categories.tolist())
        for value in values:
            _scalar(value)
        if len(set(values)) != len(values):
            raise ValueError("Encoder categories must be unique.")
        counts.append(len(values))
    widths, dropped = _encoder_grouping(encoder, counts)
    if encoder._n_features_outs != widths:
        raise ValueError("Encoder output widths disagree with fitted categories.")
    _encoder_drop_indices(encoder, dropped)


def _encoder_grouping(encoder: Any, counts: list[int]) -> tuple[list[int], list[int]]:
    """Bind saved rare-category indices and mappings to each capped output width."""
    cap = encoder.max_categories
    drop = int(encoder.drop == "first")
    if cap is None:
        return [count - drop for count in counts], [0] * len(counts)
    indices = encoder._infrequent_indices
    mappings = encoder._default_to_infrequent_mappings
    for values in (indices, mappings):
        if type(values) is not list or len(values) != len(counts):
            raise ValueError("Encoder grouping must align with columns.")
    dropped = [
        _encoder_mapping(indices[i], mappings[i], count, cap) for i, count in enumerate(counts)
    ]
    return [min(count, cap) - drop for count in counts], dropped


def _encoder_index_vector(value: Any, length: int) -> None:
    """Require a concrete integer vector before interpreting saved category positions."""
    if type(value) is not np.ndarray or value.ndim != 1 or value.dtype.kind not in "iu":
        raise ValueError("Encoder grouping positions must be integer vectors.")
    if len(value) != length:
        raise ValueError("Encoder grouping vector has the wrong length.")


def _encoder_mapping(indices: Any, mapping: Any, count: int, cap: int) -> int:
    """Check the canonical fitted mapping and return the first original output category."""
    if count < cap:
        if indices is not None or mapping is not None:
            raise ValueError("Unexpected infrequent-category grouping below the cap.")
        return 0
    _encoder_index_vector(indices, count - cap + 1)
    _encoder_index_vector(mapping, count)
    if (
        not np.array_equal(indices, np.unique(indices))
        or np.any(indices >= count)
        or np.any(indices < 0)
    ):
        raise ValueError("Infrequent-category indices must be sorted, unique and in range.")
    frequent = np.ones(count, dtype=bool)
    frequent[indices] = False
    expected = np.full(count, cap - 1, dtype=np.int64)
    expected[frequent] = np.arange(cap - 1)
    if not np.array_equal(mapping, expected):
        raise ValueError("Encoder grouping mapping disagrees with infrequent categories.")
    return int(np.flatnonzero(expected == 0)[0])


def _encoder_drop_indices(encoder: Any, dropped: list[int]) -> None:
    """Bind both sklearn dropped-index fields to the supported drop policy."""
    for indices, expected in (
        (encoder.drop_idx_, dropped),
        (encoder._drop_idx_after_grouping, [0] * len(dropped)),
    ):
        if encoder.drop is None:
            if indices is not None:
                raise ValueError("Unexpected dropped category indices.")
        elif (
            type(indices) is not np.ndarray
            or indices.ndim != 1
            or any(type(value) is not int for value in _normalize(indices.tolist()))
            or indices.tolist() != expected
        ):
            raise ValueError("Dropped indices must select the first fitted category.")


def _onehot_state(raw: dict) -> dict:
    """Keep the estimator in the certificate while validating every execution option."""
    _fields(
        raw,
        {
            "type",
            "columns",
            "encoder_object",
            "feature_names",
            "prefix_separator",
            "drop_original",
            "include_missing",
        },
    )
    scalar = _normalize({key: value for key, value in raw.items() if key != "encoder_object"})
    columns = _columns(scalar["columns"])
    _columns(scalar["feature_names"])
    if scalar["type"] != "onehot" or scalar["include_missing"] is not False:
        raise ValueError("Only observed-category one-hot artifacts are admitted.")
    if type(scalar["drop_original"]) is not bool or type(scalar["prefix_separator"]) is not str:
        raise ValueError("Invalid encoder output options.")
    encoder = raw["encoder_object"]
    _check_encoder(encoder, columns)
    if encoder.get_feature_names_out(columns).tolist() != scalar["feature_names"]:
        raise ValueError("Encoder feature names disagree with fitted categories.")
    return {**scalar, "encoder_object": encoder}


def _onehot_config(params: dict, state: dict) -> dict:
    """Check recipe options against the encoder that will actually transform rows."""
    resolved = {**_DEFAULT_OPTIONS, **params}
    _fields(
        resolved,
        {
            "columns",
            "drop_first",
            "max_categories",
            "handle_unknown",
            "prefix_separator",
            "drop_original",
            "include_missing",
        },
    )
    encoder = state["encoder_object"]
    expected = {
        "drop_first": encoder.drop == "first",
        "max_categories": encoder.max_categories,
        "handle_unknown": encoder.handle_unknown,
        **{key: state[key] for key in ("prefix_separator", "drop_original", "include_missing")},
    }
    for key, value in expected.items():
        if type(resolved[key]) is not type(value) or resolved[key] != value:
            raise ValueError(f"Configured encoder {key} disagrees with fitted state.")
    return resolved
