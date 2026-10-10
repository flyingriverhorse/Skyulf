"""TF-IDF Vectorizer node — converts text to TF-IDF weighted feature columns.

Wraps ``sklearn.feature_extraction.text.TfidfVectorizer``.  The fitted IDF
weights are stored in the artifact so they can be inspected or serialised.
Output is **always dense**.
"""

import logging
from typing import Any

import numpy as np
import pandas as pd
from sklearn.feature_extraction.text import TfidfTransformer, TfidfVectorizer

from ...core.capabilities import ExecutionCapability
from ...core.meta.decorators import node_meta
from ...registry import NodeRegistry
from .._artifacts import TfidfVectorizerArtifact
from ..base import BaseApplier, BaseCalculator, apply_method, fit_method
from ._common import (
    _join_text_columns,
    _sklearn_vectorizer_apply_pandas,
    _sklearn_vectorizer_apply_polars,
    _validate_local_vectorizer,
    _validate_vectorizer_vocabulary,
    _vectorizer_uses_callbacks,
    _warn_large_output,
    apply_text_dual_engine,
    resolve_fit_text_columns,
    validate_text_output_names,
)

logger = logging.getLogger(__name__)


# ── Apply ─────────────────────────────────────────────────────────────────────


def _tfidf_apply_pandas(
    X: pd.DataFrame, y: Any, params: dict[str, Any]
) -> tuple[pd.DataFrame, Any]:
    """Transform text columns using the fitted ``TfidfVectorizer``."""
    return _sklearn_vectorizer_apply_pandas(X, y, params)


def _validate_saved_idf(raw: dict, vectorizer: TfidfVectorizer) -> None:
    """Inspect the stored dense IDF vector and its native learned transformer."""
    transformer = getattr(vectorizer, "_tfidf", None)
    if type(transformer) is not TfidfTransformer:
        raise ValueError("Fitted TF-IDF transformer is missing or has an unexpected type.")
    if type(raw["idf"]) not in (list, tuple, np.ndarray):
        raise ValueError("Fitted IDF values must be an ordered numeric sequence.")
    weights = np.asarray(raw["idf"])
    width = len(raw["output_columns"])
    if weights.shape != (width,) or weights.dtype.kind not in "fiu":
        raise ValueError("Fitted IDF values have the wrong shape or dtype.")
    if not np.isfinite(weights).all() or not (weights > 0).all():
        raise ValueError("Fitted IDF values must be finite and positive.")
    saved_weights = getattr(transformer, "idf_", None)
    if saved_weights is None or not np.array_equal(weights, saved_weights):
        raise ValueError("Fitted IDF values disagree with the saved transformer.")
    if getattr(transformer, "n_features_in_", None) != width:
        raise ValueError("Fitted TF-IDF transformer has the wrong feature width.")


class TfidfVectorizerApplier(BaseApplier):
    """Attach one TF-IDF-weighted column per vocabulary term to the frame.

    The configured text columns are joined into a single corpus string before
    transforming, the dense result is concatenated on, and the source columns
    survive unless ``drop_original`` is set. This node uses the text-specific
    dispatcher: the polars path runs natively, so only the text payload crosses
    into sklearn, and it falls back to a full pandas round-trip when a text
    column is not String dtype.
    """

    @staticmethod
    def validate_inference_state(raw: dict) -> dict:
        """Inspect learned vocabulary and IDF weights without recomputing corpus statistics."""
        vectorizer = _validate_local_vectorizer(
            raw, "tfidf_vectorizer", TfidfVectorizer, {"vocabulary", "max_features", "idf"}
        )
        if vectorizer is not None:
            _validate_vectorizer_vocabulary(raw, vectorizer)
            _validate_saved_idf(raw, vectorizer)
        return raw

    @staticmethod
    def inference_capability(state: dict, *, engine: str) -> ExecutionCapability | None:
        """Describe fixed document weighting while leaving arbitrary analyzers undeclared."""
        if engine not in ("pandas", "polars"):
            return None
        TfidfVectorizerApplier.validate_inference_state(state)
        if state and _vectorizer_uses_callbacks(state["vectorizer_object"]):
            return None
        return ExecutionCapability(engine, "apply", "local", "preserve", "row")

    @apply_method
    def apply(self, X: Any, _y: Any, params: dict[str, Any]) -> Any:  # pylint: disable=arguments-differ
        """Dispatch through the text-specific dual-engine path; ``y`` passes through."""
        return apply_text_dual_engine(
            X, params, _tfidf_apply_pandas, _sklearn_vectorizer_apply_polars
        )


# ── Calculate ─────────────────────────────────────────────────────────────────


def _build_tfidf_artifact(
    config: dict[str, Any], X: pd.DataFrame, valid_cols: list[str]
) -> TfidfVectorizerArtifact:
    """Fit a ``TfidfVectorizer`` on the resolved text columns and build its artifact dict."""
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
    try:
        vectorizer.fit(text)
    except ValueError as exc:
        if not str(exc).startswith("empty vocabulary;"):
            raise
        logger.warning(
            "TF-IDF Vectorizer found an empty vocabulary; input is unchanged. "
            "Check the text, stop words and n-gram settings."
        )
        return {}

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
    learns_from_data=True,
)
class TfidfVectorizerCalculator(BaseCalculator):
    """Fit a ``TfidfVectorizer`` vocabulary and IDF weights on the joined text.

    Both the vocabulary and the per-term IDF weights are learned from the
    training corpus and carried in the artifact, so the weights stay inspectable
    and serialisable. A very wide output is warned about rather than raising,
    since it is a memory problem, not an error.

    Schema inference inherits the base ``None`` result: the learned vocabulary
    determines the output columns, so their schema is only known at runtime.
    """

    @fit_method
    def fit(self, X: Any, _y: Any, config: dict[str, Any]) -> TfidfVectorizerArtifact:  # pylint: disable=arguments-differ
        """Narrow to the configured text columns and fit the vocabulary on their joined text.

        An empty or wholly-absent column selection yields an empty artifact so
        the applier no-ops. An empty vocabulary also yields an empty artifact
        with a warning; source columns are retained even with ``drop_original``.
        """
        resolved = resolve_fit_text_columns(X, config, _y)
        if resolved is None:
            return {}
        text_frame, valid_cols = resolved

        artifact = _build_tfidf_artifact(config, text_frame, valid_cols)
        validate_text_output_names(X, artifact, valid_cols)
        return artifact
