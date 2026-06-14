"""Hashing Vectorizer node — stateless text-to-feature-column transform.

Wraps ``sklearn.feature_extraction.text.HashingVectorizer``.  Unlike
``CountVectorizer`` / ``TfidfVectorizer`` this node has **no vocabulary** — it
hashes tokens directly to column indices.  The Calculator records the column
list and ``n_features`` but does not require any training data to fit.

Output is **always dense**.  Column names use the indexed scheme
``{src}__hash__0 … {src}__hash__{n_features-1}``.
"""

import logging
from typing import Any, Dict, List, Optional, Tuple

import pandas as pd
from sklearn.feature_extraction.text import HashingVectorizer

from ...core.meta.decorators import node_meta
from ...registry import NodeRegistry
from .._artifacts import HashingVectorizerArtifact
from ..base import BaseApplier, BaseCalculator, apply_method, fit_method
from ._common import _join_text_columns, _warn_large_output, apply_text_pandas_only

logger = logging.getLogger(__name__)

_DEFAULT_N_FEATURES = 1_024


# ── Apply ─────────────────────────────────────────────────────────────────────


def _hash_apply_pandas(X: pd.DataFrame, y: Any, params: Dict[str, Any]) -> Tuple[pd.DataFrame, Any]:
    cols: List[str] = params.get("columns", [])
    vectorizer: Optional[HashingVectorizer] = params.get("vectorizer_object")
    output_columns: List[str] = params.get("output_columns", [])
    drop_original: bool = params.get("drop_original", False)

    if not cols or vectorizer is None or not output_columns:
        return X, y

    valid_cols = [c for c in cols if c in X.columns]
    if not valid_cols:
        return X, y

    text = _join_text_columns(X, valid_cols)
    encoded = vectorizer.transform(text)
    dense = encoded.toarray() if hasattr(encoded, "toarray") else encoded

    encoded_df = pd.DataFrame(
        dense,
        columns=output_columns,  # ty: ignore[invalid-argument-type]
        index=X.index,
    )

    X_out = X.copy()
    if drop_original:
        X_out = X_out.drop(columns=valid_cols)
    return pd.concat([X_out, encoded_df], axis=1), y


class HashingVectorizerApplier(BaseApplier):
    @apply_method
    def apply(self, X: Any, _y: Any, params: Dict[str, Any]) -> Any:
        return apply_text_pandas_only(X, params, _hash_apply_pandas)


# ── Calculate ─────────────────────────────────────────────────────────────────


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
    def infer_output_schema(self, input_schema: Any, config: Dict[str, Any]) -> None:
        # n_features is fixed by config, but we still return None because the
        # source column list may not be known without seeing data.
        return None

    @fit_method
    def fit(self, X: Any, _y: Any, config: Dict[str, Any]) -> HashingVectorizerArtifact:
        cols: List[str] = config.get("columns", [])
        if not cols:
            return {}

        if hasattr(X, "to_pandas"):
            X = X.to_pandas()

        valid_cols = [c for c in cols if c in X.columns]
        if not valid_cols:
            return {}

        n_features: int = int(config.get("n_features", _DEFAULT_N_FEATURES))
        norm: Optional[str] = config.get("norm", "l2") or None
        alternate_sign: bool = bool(config.get("alternate_sign", True))
        lowercase: bool = bool(config.get("lowercase", True))
        stop_words: Optional[str] = config.get("stop_words") or None

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
