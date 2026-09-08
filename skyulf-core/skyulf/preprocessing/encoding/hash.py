"""Hash Encoder node (Calculator + Applier)."""

import hashlib
import logging
from typing import Any

import pandas as pd
import polars as pl

from ...core.meta.decorators import node_meta
from ...registry import NodeRegistry
from ...utils import resolve_columns, user_picked_no_columns
from .._artifacts import HashEncoderArtifact
from .._schema import SkyulfSchema
from ..base import BaseApplier, BaseCalculator, apply_method, fit_method
from ..dispatcher import apply_dual_engine
from ._common import _exclude_target_column, detect_categorical_columns

logger = logging.getLogger(__name__)


def _resolve_valid_cols(X: Any, params: dict[str, Any]) -> list[str]:
    """Filter requested columns down to those present in ``X``."""
    cols = params.get("columns", [])
    return [c for c in cols if c in X.columns]


def _uses_numeric_normalization(params: dict[str, Any]) -> bool:
    """Preserve legacy buckets and reject unknown numeric-normalization versions."""
    if "numeric_normalization_version" not in params:
        return False
    version = params["numeric_normalization_version"]
    if isinstance(version, bool) or not isinstance(version, int) or version != 1:
        raise ValueError(
            f"Unsupported HashEncoder numeric normalization version {version!r}; "
            "refit the encoder with a supported version."
        )
    return True


def _hash_apply_polars(X: Any, y: Any, params: dict[str, Any]) -> tuple[Any, Any]:
    """Polars apply path — same ``_stable_hash`` (blake2b) as the pandas path.

    Polars' native ``hash()`` used to bucket differently from pandas' blake2b,
    and deployment guarantees an engine crossing (Polars-trained pipelines
    always serve a pandas frame), so the divergence hit every production
    encoding (F-11). Hashing once per *unique* value keeps this cheap on
    low-cardinality categorical columns.
    """
    valid_cols = _resolve_valid_cols(X, params)
    if not valid_cols:
        return X, y

    n_features = params.get("n_features", 10)
    normalize_numeric = _uses_numeric_normalization(params)
    exprs = []
    for col in valid_cols:
        # fill_null("nan") mirrors pandas' astype(str) so missing values land
        # in the same bucket on both engines.
        str_col = pl.col(col).cast(pl.Utf8).fill_null("nan")
        if normalize_numeric and X.schema[col].is_float():
            str_col = str_col.str.replace(r"\.0$", "")
        unique_vals = X.select(str_col.alias(col)).to_series().unique().to_list()
        bucket_by_value = {v: _stable_hash(v) % n_features for v in unique_vals}
        exprs.append(str_col.replace_strict(bucket_by_value, default=None).alias(col))
    return X.with_columns(exprs), y


def _stable_hash(value: str) -> int:
    """Deterministic hash for strings, stable across processes/interpreters.

    Python's built-in ``hash()`` is salted per-process (``PYTHONHASHSEED``),
    so the same category would map to a different bucket every time the
    interpreter restarts (e.g. a new Celery worker or API server), silently
    corrupting encodings learned at fit time. ``blake2b`` is deterministic,
    fixing that cross-process instability. Both the pandas and polars apply
    paths use this function, so buckets are identical across engines (F-11) —
    artifacts remain portable even when fit and serve use different engines.
    """
    digest = hashlib.blake2b(value.encode("utf-8"), digest_size=8).digest()
    return int.from_bytes(digest, "little")


def _hash_apply_pandas(X: Any, y: Any, params: dict[str, Any]) -> tuple[Any, Any]:
    """Pandas apply path — uses a stable hash per value."""
    valid_cols = _resolve_valid_cols(X, params)
    if not valid_cols:
        return X, y

    n_features = params.get("n_features", 10)
    normalize_numeric = _uses_numeric_normalization(params)
    X_out = X.copy()
    for col in valid_cols:
        s = X_out[col].astype(str)
        # An unrelated fractional/null row can promote an integer column to
        # float. Normalize integral floats per value, retaining literal text.
        if normalize_numeric and pd.api.types.is_float_dtype(X_out[col]):
            s = s.str.replace(r"\.0$", "", regex=True)
        # Hash each *unique* value once, then vectorize the lookup via
        # `.map()` — cheaper than a per-row `.apply()` on high-row-count,
        # low-cardinality columns. Building the mapping from this column's
        # own `.unique()` guarantees every value is covered, so there's no
        # NaN-from-unmapped-key risk.
        bucket_by_value = {v: _stable_hash(v) % n_features for v in s.unique()}
        X_out[col] = s.map(bucket_by_value)
    return X_out, y


class HashEncoderApplier(BaseApplier):
    """Replace each categorical column's values in place with a bucket index.

    Buckets run over ``[0, n_features)``. No vocabulary is learned, so the same
    artifact handles values never seen at fit time — at the cost of collisions,
    which the hashing trick accepts by design. Both engines must agree on the
    bucket, which is why they share ``_stable_hash`` (blake2b) instead of using
    polars' native ``hash()``: deployment always crosses engines, so a
    divergence would corrupt every production encoding. Nulls are filled with
    the literal ``"nan"`` on polars to mirror pandas' ``astype(str)``, keeping
    them in the same bucket on both sides.
    """

    @apply_method
    def apply(self, X: Any, y: Any, params: dict[str, Any]) -> Any:  # pylint: disable=arguments-differ
        """Dispatch to the engine-specific hashing, forwarding ``(X, y)`` only when ``y`` exists."""
        return apply_dual_engine(
            (X, y) if y is not None else X,
            params,
            {"polars": _hash_apply_polars, "pandas": _hash_apply_pandas},
        )


@NodeRegistry.register("HashEncoder", HashEncoderApplier)
@node_meta(
    id="HashEncoder",
    name="Hash Encoder",
    category="Preprocessing",
    description="Encode categorical features using hashing.",
    params={"n_features": 8, "columns": []},
    learns_from_data=True,
)
class HashEncoderCalculator(BaseCalculator):
    """Resolve which columns to hash and how many buckets to use.

    No vocabulary is learned — hashing is stateless, so the artifact is just
    the column set plus ``n_features``. The target column is excluded because
    hash encoding overwrites the column it encodes. An explicit
    ``columns: []``, or auto-detection finding nothing, yields an empty
    artifact so the applier no-ops.
    """

    @fit_method
    def fit(self, X: Any, y: Any, config: dict[str, Any]) -> HashEncoderArtifact:  # pylint: disable=arguments-differ
        """Build the artifact, auto-detecting categorical columns when none are named."""
        if user_picked_no_columns(config):
            return {}

        cols = resolve_columns(X, config, detect_categorical_columns)
        cols = _exclude_target_column(cols, config, "HashEncoder", y)
        if not cols:
            return {}

        return {
            "type": "hash_encoder",
            "columns": cols,
            "n_features": config.get("n_features", 10),
            "numeric_normalization_version": 1,
        }

    def infer_output_schema(
        self,
        input_schema: SkyulfSchema,
        config: dict[str, Any],
    ) -> SkyulfSchema | None:
        """Pass the input schema through: bucket indices replace values in place."""
        # Hash encoder replaces values in source columns in place
        # (`pl.col(col)...alias(col)`). Schema is unchanged.
        return input_schema


__all__ = ["HashEncoderApplier", "HashEncoderCalculator"]
