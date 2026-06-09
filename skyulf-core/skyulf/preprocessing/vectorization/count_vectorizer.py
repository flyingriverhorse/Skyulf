"""CountVectorizer node — converts text to token-count feature columns.

Calculator fits a ``sklearn.feature_extraction.text.CountVectorizer`` on the
training corpus and records the vocabulary.  Applier transforms any text input
using the fitted vocabulary.

Output is **always dense** (``toarray()`` called on the sparse sklearn output).
For very large vocabularies a ``node_warnings`` entry is emitted instead of
raising an error.
"""

import logging
from typing import Any, Dict, List, Optional, Tuple

import pandas as pd
from sklearn.feature_extraction.text import CountVectorizer

from ...core.meta.decorators import node_meta
from ...registry import NodeRegistry
from ...utils import unpack_pipeline_input
from .._artifacts import CountVectorizerArtifact
from ..base import BaseApplier, BaseCalculator, apply_method, fit_method
from ._common import _join_text_columns, _warn_large_output, apply_text_pandas_only

logger = logging.getLogger(__name__)

_LARGE_VOCAB_THRESHOLD = 10_000


# ── Apply ─────────────────────────────────────────────────────────────────────


def _count_apply_pandas(
    X: pd.DataFrame, y: Any, params: Dict[str, Any]
) -> Tuple[pd.DataFrame, Any]:
    cols: List[str] = params.get("columns", [])
    vectorizer: Optional[CountVectorizer] = params.get("vectorizer_object")
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


class CountVectorizerApplier(BaseApplier):
    @apply_method
    def apply(self, X: Any, _y: Any, params: Dict[str, Any]) -> Any:
        return apply_text_pandas_only(X, params, _count_apply_pandas)


# ── Calculate ─────────────────────────────────────────────────────────────────


@NodeRegistry.register("count_vectorizer", CountVectorizerApplier)
@node_meta(
    id="count_vectorizer",
    name="Count Vectorizer",
    category="Text",
    description=(
        "Convert text columns to token-count feature columns (bag-of-words). "
        "Fits a vocabulary on training data; apply uses that vocabulary."
    ),
    params={
        "columns": [],
        "max_features": None,
        "min_df": 1,
        "max_df": 1.0,
        "ngram_range": [1, 1],
        "drop_original": False,
    },
    tags=["text", "nlp", "vectorizer", "bag-of-words"],
)
class CountVectorizerCalculator(BaseCalculator):
    def infer_output_schema(self, input_schema: Any, config: Dict[str, Any]) -> None:
        # Vocabulary size is data-dependent — return None to signal unknown.
        return None

    @fit_method
    def fit(self, X: Any, _y: Any, config: Dict[str, Any]) -> CountVectorizerArtifact:
        cols: List[str] = config.get("columns", [])
        if not cols:
            return {}

        # Always work in pandas
        if hasattr(X, "to_pandas"):
            X = X.to_pandas()

        valid_cols = [c for c in cols if c in X.columns]
        if not valid_cols:
            return {}

        max_features: Optional[int] = config.get("max_features") or None
        min_df: Any = config.get("min_df", 1)
        max_df: Any = config.get("max_df", 1.0)
        ngram_min, ngram_max = config.get("ngram_range", [1, 1])

        vectorizer = CountVectorizer(
            max_features=max_features,
            min_df=min_df,
            max_df=max_df,
            ngram_range=(ngram_min, ngram_max),
        )

        text = _join_text_columns(X, valid_cols)
        vectorizer.fit(text)

        feature_names: List[str] = vectorizer.get_feature_names_out().tolist()
        prefix = valid_cols[0] if len(valid_cols) == 1 else "_".join(valid_cols)
        output_columns = [f"{prefix}__count__{name}" for name in feature_names]

        warn = _warn_large_output(len(output_columns))
        if warn:
            logger.warning(warn)

        return {
            "type": "count_vectorizer",
            "columns": valid_cols,
            "output_columns": output_columns,
            "vocabulary": vectorizer.vocabulary_,
            "max_features": max_features,
            "vectorizer_object": vectorizer,
            "drop_original": bool(config.get("drop_original", False)),
        }
