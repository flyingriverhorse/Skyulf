"""TF-IDF Vectorizer node — converts text to TF-IDF weighted feature columns.

Wraps ``sklearn.feature_extraction.text.TfidfVectorizer``.  The fitted IDF
weights are stored in the artifact so they can be inspected or serialised.
Output is **always dense**.
"""

import logging
from typing import Any

import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer

from ...core.meta.decorators import node_meta
from ...registry import NodeRegistry
from .._artifacts import TfidfVectorizerArtifact
from ..base import BaseApplier, BaseCalculator, apply_method, fit_method
from ._common import _join_text_columns, _warn_large_output, apply_text_pandas_only

logger = logging.getLogger(__name__)


# ── Apply ─────────────────────────────────────────────────────────────────────


def _tfidf_apply_pandas(
    X: pd.DataFrame, y: Any, params: dict[str, Any]
) -> tuple[pd.DataFrame, Any]:
    cols: list[str] = params.get("columns", [])
    vectorizer: TfidfVectorizer | None = params.get("vectorizer_object")
    output_columns: list[str] = params.get("output_columns", [])
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


class TfidfVectorizerApplier(BaseApplier):
    @apply_method
    def apply(self, X: Any, _y: Any, params: dict[str, Any]) -> Any:  # pylint: disable=arguments-differ
        return apply_text_pandas_only(X, params, _tfidf_apply_pandas)


# ── Calculate ─────────────────────────────────────────────────────────────────


@NodeRegistry.register("tfidf_vectorizer", TfidfVectorizerApplier)
@node_meta(
    id="tfidf_vectorizer",
    name="TF-IDF Vectorizer",
    category="Text",
    description=(
        "Convert text columns to TF-IDF weighted feature columns. "
        "Penalises very common tokens and rewards rare-but-informative ones."
    ),
    params={
        "columns": [],
        "max_features": None,
        "min_df": 1,
        "max_df": 1.0,
        "ngram_range": [1, 1],
        "sublinear_tf": False,
        "lowercase": True,
        "stop_words": None,
        "drop_original": False,
    },
    tags=["text", "nlp", "tfidf", "vectorizer"],
)
class TfidfVectorizerCalculator(BaseCalculator):
    def infer_output_schema(self, input_schema: Any, config: dict[str, Any]) -> None:
        return None

    @fit_method
    def fit(self, X: Any, _y: Any, config: dict[str, Any]) -> TfidfVectorizerArtifact:  # pylint: disable=arguments-differ
        cols: list[str] = config.get("columns", [])
        if not cols:
            return {}

        if hasattr(X, "to_pandas"):
            X = X.to_pandas()

        valid_cols = [c for c in cols if c in X.columns]
        if not valid_cols:
            return {}

        max_features: int | None = config.get("max_features") or None
        min_df: Any = config.get("min_df", 1)
        max_df: Any = config.get("max_df", 1.0)
        ngram_min, ngram_max = config.get("ngram_range", [1, 1])
        sublinear_tf: bool = bool(config.get("sublinear_tf", False))
        lowercase: bool = bool(config.get("lowercase", True))
        stop_words: str | None = config.get("stop_words") or None

        vectorizer = TfidfVectorizer(
            max_features=max_features,
            min_df=min_df,
            max_df=max_df,
            ngram_range=(ngram_min, ngram_max),
            sublinear_tf=sublinear_tf,
            lowercase=lowercase,
            stop_words=stop_words,
        )

        text = _join_text_columns(X, valid_cols)
        vectorizer.fit(text)

        feature_names: list[str] = vectorizer.get_feature_names_out().tolist()
        prefix = valid_cols[0] if len(valid_cols) == 1 else "_".join(valid_cols)
        output_columns = [f"{prefix}__tfidf__{name}" for name in feature_names]

        warn = _warn_large_output(len(output_columns))
        if warn:
            logger.warning(warn)

        return {
            "type": "tfidf_vectorizer",
            "columns": valid_cols,
            "output_columns": output_columns,
            "vocabulary": vectorizer.vocabulary_,
            "idf": vectorizer.idf_.tolist(),
            "max_features": max_features,
            "lowercase": lowercase,
            "stop_words": stop_words,
            "vectorizer_object": vectorizer,
            "drop_original": bool(config.get("drop_original", False)),
        }
