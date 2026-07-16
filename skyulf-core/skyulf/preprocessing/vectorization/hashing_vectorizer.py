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
    _warn_large_output,
    apply_text_pandas_only,
    resolve_fit_text_columns,
)

logger = logging.getLogger(__name__)

_DEFAULT_N_FEATURES = 1_024


# ── Apply ─────────────────────────────────────────────────────────────────────


def _hash_apply_pandas(X: pd.DataFrame, y: Any, params: dict[str, Any]) -> tuple[pd.DataFrame, Any]:
    """Transform text columns using the fitted ``HashingVectorizer``."""
    return _sklearn_vectorizer_apply_pandas(X, y, params)


class HashingVectorizerApplier(BaseApplier):
    @apply_method
    def apply(self, X: Any, _y: Any, params: dict[str, Any]) -> Any:  # pylint: disable=arguments-differ
        return apply_text_pandas_only(X, params, _hash_apply_pandas)


# ── Calculate ─────────────────────────────────────────────────────────────────


def _build_hashing_artifact(
    config: dict[str, Any], valid_cols: list[str]
) -> HashingVectorizerArtifact:
    """Build a fitted ``HashingVectorizer`` and its artifact dict from config."""
    n_features: int = int(config.get("n_features", _DEFAULT_N_FEATURES))
    norm: str | None = config.get("norm", "l2") or None
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
)
class HashingVectorizerCalculator(BaseCalculator):
    def infer_output_schema(self, input_schema: Any, config: dict[str, Any]) -> None:
        # n_features is fixed by config, but we still return None because the
        # source column list may not be known without seeing data.
        return None

    @fit_method
    def fit(self, X: Any, _y: Any, config: dict[str, Any]) -> HashingVectorizerArtifact:  # pylint: disable=arguments-differ
        resolved = resolve_fit_text_columns(X, config)
        if resolved is None:
            return {}
        X, valid_cols = resolved

        return _build_hashing_artifact(config, valid_cols)
