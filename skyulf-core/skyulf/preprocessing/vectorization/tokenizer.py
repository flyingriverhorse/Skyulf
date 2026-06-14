"""Tokenizer node — splits text into tokens (word / char / char_wb).

Stateless: the Calculator stores configuration only (no vocabulary is fitted).
The Applier rebuilds an sklearn analyzer and emits a space-joined token string
column ``{src}__tokens`` per source column, optionally with a token-count column.

The joined-token output is intentionally a plain string column so it can feed a
vectorizer node downstream or be inspected directly.
"""

import logging
from typing import Any, Dict, List, Tuple

import pandas as pd
from sklearn.feature_extraction.text import CountVectorizer

from ...core.meta.decorators import node_meta
from ...registry import NodeRegistry
from .._artifacts import TokenizerArtifact
from ..base import BaseApplier, BaseCalculator, apply_method, fit_method
from ._common import apply_text_pandas_only

logger = logging.getLogger(__name__)


def _build_analyzer(params: Dict[str, Any]):
    """Construct an sklearn token analyzer callable from node params."""
    analyzer: str = params.get("analyzer", "word")
    lowercase: bool = params.get("lowercase", True)
    stop_words = params.get("stop_words") or None
    ngram_min, ngram_max = params.get("ngram_range", [1, 1])

    vec = CountVectorizer(
        analyzer=analyzer,
        lowercase=lowercase,
        stop_words=stop_words,
        ngram_range=(ngram_min, ngram_max),
    )
    return vec.build_analyzer()


# ── Apply ─────────────────────────────────────────────────────────────────────


def _tokenizer_apply_pandas(
    X: pd.DataFrame, y: Any, params: Dict[str, Any]
) -> Tuple[pd.DataFrame, Any]:
    cols: List[str] = params.get("columns", [])
    drop_original: bool = params.get("drop_original", False)
    add_token_count: bool = params.get("add_token_count", False)

    valid_cols = [c for c in cols if c in X.columns]
    if not valid_cols:
        return X, y

    analyze = _build_analyzer(params)
    X_out = X.copy()

    for col in valid_cols:
        text = X_out[col].fillna("").astype(str)
        tokens = text.map(analyze)
        X_out[f"{col}__tokens"] = tokens.map(lambda toks: " ".join(toks))
        if add_token_count:
            X_out[f"{col}__token_count"] = tokens.map(len)

    if drop_original:
        X_out = X_out.drop(columns=valid_cols)
    return X_out, y


class TokenizerApplier(BaseApplier):
    @apply_method
    def apply(self, X: Any, _y: Any, params: Dict[str, Any]) -> Any:
        return apply_text_pandas_only(X, params, _tokenizer_apply_pandas)


# ── Calculate ─────────────────────────────────────────────────────────────────


@NodeRegistry.register("tokenizer", TokenizerApplier)
@node_meta(
    id="tokenizer",
    name="Tokenizer",
    category="Text",
    description=(
        "Split text columns into tokens (word, char, or char_wb). "
        "Outputs a space-joined token string column per source column, "
        "optionally with a token-count column. Stateless — no vocabulary fitted. "
        "Inspection / intermediate tool only: do NOT chain before a vectorizer "
        "(Count / TF-IDF / Hashing already tokenize internally)."
    ),
    params={
        "columns": [],
        "analyzer": "word",
        "lowercase": True,
        "stop_words": None,
        "ngram_range": [1, 1],
        "add_token_count": False,
        "drop_original": False,
    },
    tags=["text", "nlp", "tokenizer"],
)
class TokenizerCalculator(BaseCalculator):
    def infer_output_schema(self, input_schema: Any, config: Dict[str, Any]) -> None:
        return None

    @fit_method
    def fit(self, X: Any, _y: Any, config: Dict[str, Any]) -> TokenizerArtifact:
        cols: List[str] = config.get("columns", [])
        if not cols:
            return {}

        if hasattr(X, "to_pandas"):
            X = X.to_pandas()

        valid_cols = [c for c in cols if c in X.columns]
        if not valid_cols:
            return {}

        add_token_count = config.get("add_token_count", False)

        output_columns: List[str] = []
        for col in valid_cols:
            output_columns.append(f"{col}__tokens")
            if add_token_count:
                output_columns.append(f"{col}__token_count")

        ngram_min, ngram_max = config.get("ngram_range", [1, 1])

        return {
            "type": "tokenizer",
            "columns": valid_cols,
            "analyzer": config.get("analyzer", "word"),
            "lowercase": config.get("lowercase", True),
            "stop_words": config.get("stop_words") or None,
            "ngram_range": [ngram_min, ngram_max],
            "output_columns": output_columns,
            "add_token_count": add_token_count,
            "drop_original": config.get("drop_original", False),
        }
