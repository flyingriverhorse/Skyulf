"""Hashing Vectorizer node — stateless text-to-feature-column transform.

Wraps ``sklearn.feature_extraction.text.HashingVectorizer``.  Unlike
``CountVectorizer`` / ``TfidfVectorizer`` this node has **no vocabulary** — it
hashes tokens directly to column indices.  The Calculator records the column
list and ``n_features`` but does not require any training data to fit.

Output is **always dense**.  Column names use the indexed scheme
``{src}__hash__0 … {src}__hash__{n_features-1}``.
"""

import logging
from typing import Any

import pandas as pd
from sklearn.feature_extraction.text import HashingVectorizer

from ...core.meta.decorators import node_meta
from ...registry import NodeRegistry
from .._artifacts import HashingVectorizerArtifact
from ..base import BaseApplier, BaseCalculator, apply_method, fit_method
from ._common import (
    _sklearn_vectorizer_apply_pandas,
    _sklearn_vectorizer_apply_polars,
    _warn_large_output,
    apply_text_dual_engine,
    resolve_fit_text_valid_columns,
)

logger = logging.getLogger(__name__)

_DEFAULT_N_FEATURES = 1_024


# ── Apply ─────────────────────────────────────────────────────────────────────


def _hash_apply_pandas(X: pd.DataFrame, y: Any, params: dict[str, Any]) -> tuple[pd.DataFrame, Any]:
    """Transform text columns using the fitted ``HashingVectorizer``."""
    return _sklearn_vectorizer_apply_pandas(X, y, params)


class HashingVectorizerApplier(BaseApplier):
    """Attach ``n_features`` hashed-token columns to the frame.

    No vocabulary is consulted — tokens hash straight to column indices, so the
    same artifact works on text never seen at fit time and the output width is
    fixed by config. The trade-off is collisions, which the hashing trick
    accepts by design. The configured text columns are joined first, the dense
    result is concatenated on, and the sources survive unless ``drop_original``
    is set. The polars path runs natively and falls back to a pandas round-trip
    when a text column is not String dtype.
    """

    @apply_method
    def apply(self, X: Any, _y: Any, params: dict[str, Any]) -> Any:  # pylint: disable=arguments-differ
        """Dispatch through the text-specific dual-engine path; ``y`` passes through."""
        return apply_text_dual_engine(
            X, params, _hash_apply_pandas, _sklearn_vectorizer_apply_polars
        )


# ── Calculate ─────────────────────────────────────────────────────────────────


def _build_hashing_artifact(
    config: dict[str, Any], valid_cols: list[str]
) -> HashingVectorizerArtifact:
    """Build a fitted ``HashingVectorizer`` and its artifact dict from config."""
    n_features: int = int(config.get("n_features", _DEFAULT_N_FEATURES))
    norm: str | None = config.get("norm", "l2") or None
    if norm == "none":
        norm = None
    alternate_sign: bool = bool(config.get("alternate_sign", True))
    lowercase: bool = bool(config.get("lowercase", True))
    stop_words: str | None = config.get("stop_words") or None

    vectorizer = HashingVectorizer(
        n_features=n_features,
        norm=norm,
        alternate_sign=alternate_sign,
        lowercase=lowercase,
        stop_words=stop_words,
    )

    prefix = valid_cols[0] if len(valid_cols) == 1 else "_".join(valid_cols)
    output_columns = [f"{prefix}__hash__{i}" for i in range(n_features)]

    warn = _warn_large_output(n_features)
    if warn:
        logger.warning(warn)

    return {
        "type": "hashing_vectorizer",
        "columns": valid_cols,
        "output_columns": output_columns,
        "n_features": n_features,
        "norm": norm,
        "lowercase": lowercase,
        "stop_words": stop_words,
        "vectorizer_object": vectorizer,
        "drop_original": bool(config.get("drop_original", False)),
    }


@NodeRegistry.register("hashing_vectorizer", HashingVectorizerApplier)
@node_meta(
    id="hashing_vectorizer",
    name="Hashing Vectorizer",
    category="Text",
    description=(
        "Stateless text vectorizer using the hashing trick. "
        "No vocabulary is stored — tokens are hashed directly to column indices. "
        "Suitable for very large or streaming datasets."
    ),
    params={
        "columns": [],
        "n_features": _DEFAULT_N_FEATURES,
        "norm": "l2",
        "alternate_sign": True,
        "lowercase": True,
        "stop_words": None,
        "drop_original": False,
    },
    tags=["text", "nlp", "hashing", "vectorizer", "stateless"],
    learns_from_data=False,
)
class HashingVectorizerCalculator(BaseCalculator):
    """Record the column list and build an unfitted ``HashingVectorizer`` from config.

    Nothing is learned from the data (``learns_from_data=False``), so the artifact
    is fully reproducible from config and identical across runs — the property that
    makes it usable on streaming or very large corpora where building a vocabulary
    would not fit in memory.

    Normalization defaults to ``"l2"``; ``"l1"`` selects L1 normalization.
    The Canvas value ``"none"`` and Python ``None`` both disable normalization.
    """

    def infer_output_schema(self, input_schema: Any, config: dict[str, Any]) -> None:
        """Return ``None``: the output width is fixed, the source columns are not known."""
        # n_features is fixed by config, but we still return None because the
        # source column list may not be known without seeing data.
        return None

    @fit_method
    def fit(self, X: Any, _y: Any, config: dict[str, Any]) -> HashingVectorizerArtifact:  # pylint: disable=arguments-differ
        """Resolve the configured text columns and build the artifact without touching values.

        Uses the column-names-only resolver rather than the data-narrowing sibling,
        so a polars input is never converted to pandas just to fit. An empty or
        wholly-absent selection yields an empty artifact so the applier no-ops.
        """
        valid_cols = resolve_fit_text_valid_columns(X, config, _y)
        if valid_cols is None:
            return {}

        return _build_hashing_artifact(config, valid_cols)
