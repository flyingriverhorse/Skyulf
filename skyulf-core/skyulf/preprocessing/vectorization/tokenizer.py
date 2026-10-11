"""Tokenizer node — splits text into tokens (word / char / char_wb).

Stateless: the Calculator stores configuration only (no vocabulary is fitted).
The Applier rebuilds an sklearn analyzer and emits a space-joined token string
column ``{src}__tokens`` per source column, optionally with a token-count column.

The joined-token output is intentionally a plain string column so it can feed a
vectorizer node downstream or be inspected directly.
"""

import logging
from typing import Any

import pandas as pd
import polars as pl
from sklearn.feature_extraction.text import CountVectorizer

from ...core.capabilities import ExecutionCapability
from ...core.meta.decorators import node_meta
from ...registry import NodeRegistry
from .._artifacts import TokenizerArtifact
from .._fitted_validation import local_boolean, local_state_fields
from ..base import BaseApplier, BaseCalculator, apply_method, fit_method
from ._common import (
    _drop_and_concat,
    _drop_and_concat_polars,
    _text_series,
    _validate_text_layout,
    _validate_text_settings,
    apply_text_dual_engine,
    resolve_fit_text_valid_columns,
    validate_text_output_names,
)

logger = logging.getLogger(__name__)


def _build_analyzer(params: dict[str, Any]):
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
    X: pd.DataFrame, y: Any, params: dict[str, Any]
) -> tuple[pd.DataFrame, Any]:
    """Tokenize original sources before dropping them and attaching validated outputs."""
    cols: list[str] = params.get("columns", [])
    drop_original: bool = params.get("drop_original", False)
    add_token_count: bool = params.get("add_token_count", False)

    valid_cols = [c for c in cols if c in X.columns]
    if not valid_cols:
        return X, y

    analyze = _build_analyzer(params)
    outputs = pd.DataFrame(index=X.index)

    for col in valid_cols:
        text = _text_series(X[col])
        tokens = text.map(analyze)
        outputs[f"{col}__tokens"] = tokens.map(" ".join)  # ty: ignore[no-matching-overload]
        if add_token_count:
            outputs[f"{col}__token_count"] = tokens.map(len).astype("int64")

    return _drop_and_concat(X, outputs, valid_cols, drop_original), y


def _tokenizer_apply_polars(X: Any, params: dict[str, Any]) -> Any:
    """Native-Polars tokenizer apply.

    Returns ``None`` to fall back to pandas when a text column is not String
    dtype (``astype(str)`` parity).
    """
    cols: list[str] = params.get("columns", [])
    drop_original: bool = params.get("drop_original", False)
    add_token_count: bool = params.get("add_token_count", False)

    valid_cols = [c for c in cols if c in X.columns]
    if not valid_cols:
        return X

    analyze = _build_analyzer(params)
    new_cols = []
    for col in valid_cols:
        series = X.get_column(col)
        if series.dtype != pl.String:
            return None
        tokens = [analyze(text) for text in series.fill_null("").to_list()]
        new_cols.append(
            pl.Series(f"{col}__tokens", [" ".join(toks) for toks in tokens], dtype=pl.String)
        )
        if add_token_count:
            new_cols.append(
                pl.Series(f"{col}__token_count", [len(toks) for toks in tokens], dtype=pl.Int64)
            )

    return _drop_and_concat_polars(X, pl.DataFrame(new_cols), valid_cols, drop_original)


class TokenizerApplier(BaseApplier):
    """Emit one space-joined token string per source column, not one column per token.

    Keeping the tokens in a single ``{col}__tokens`` string is deliberate: it
    stays inspectable and can feed a downstream vectorizer, whereas exploding to
    columns would make the output width data-dependent. ``add_token_count`` adds
    a companion ``{col}__token_count`` integer column. Nulls are coerced to empty
    strings before tokenizing, and sources survive unless ``drop_original`` is
    set. The node is stateless — the sklearn analyzer is rebuilt from ``params``
    on every call. The polars path runs natively and falls back to pandas when a
    text column is not String dtype.
    """

    @staticmethod
    def validate_inference_state(raw: dict) -> dict:
        """Inspect saved analyzer configuration without constructing or executing it."""
        fields = {
            "type",
            "columns",
            "analyzer",
            "lowercase",
            "stop_words",
            "ngram_range",
            "output_columns",
            "add_token_count",
            "drop_original",
        }
        if not local_state_fields(raw, "tokenizer", fields, allow_empty=True):
            return raw
        _validate_text_layout(raw)
        _validate_text_settings(raw["lowercase"], raw["stop_words"], raw["ngram_range"])
        if raw["add_token_count"] is not None:
            local_boolean(raw["add_token_count"], "add_token_count")
        if not callable(raw["analyzer"]) and raw["analyzer"] not in ("word", "char", "char_wb"):
            raise ValueError("Fitted tokenizer analyzer is unknown.")
        expected = _build_tokenizer_artifact(raw, raw["columns"])["output_columns"]
        if list(raw["output_columns"]) != expected:
            raise ValueError("Fitted tokenizer output names disagree with its selected columns.")
        return raw

    @staticmethod
    def inference_capability(state: dict, *, engine: str) -> ExecutionCapability | None:
        """Describe per-document tokenization without promising arbitrary callback behavior."""
        if engine not in ("pandas", "polars"):
            return None
        TokenizerApplier.validate_inference_state(state)
        if state and callable(state["analyzer"]):
            return None
        return ExecutionCapability(engine, "apply", "local", "preserve", "row")

    @apply_method
    def apply(self, X: Any, _y: Any, params: dict[str, Any]) -> Any:  # pylint: disable=arguments-differ
        """Dispatch through the text-specific dual-engine path; ``y`` passes through."""
        return apply_text_dual_engine(X, params, _tokenizer_apply_pandas, _tokenizer_apply_polars)


# ── Calculate ─────────────────────────────────────────────────────────────────


def _build_tokenizer_artifact(config: dict[str, Any], valid_cols: list[str]) -> TokenizerArtifact:
    """Build the tokenizer artifact dict (config only — no vocabulary is fitted)."""
    add_token_count = config.get("add_token_count", False)

    output_columns: list[str] = []
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
    learns_from_data=False,
)
class TokenizerCalculator(BaseCalculator):
    """Record the resolved text columns and the tokenizer settings — nothing is fitted.

    ``learns_from_data=False``: the artifact is pure configuration, so it is
    reproducible across runs and safe to reuse on data the calculator never saw.

    Schema inference conservatively inherits the base ``None`` result. Fitting
    resolves source columns and excludes the target, including a separately
    supplied ``y``; runtime introspection determines which source and token
    columns remain after applying ``drop_original``.
    """

    @fit_method
    def fit(self, X: Any, _y: Any, config: dict[str, Any]) -> TokenizerArtifact:  # pylint: disable=arguments-differ
        """Resolve the configured text column names and record them with the tokenizer settings.

        Uses the column-names-only resolver rather than the data-narrowing
        sibling, so a polars input is never converted to pandas just to fit. An
        empty or wholly-absent selection yields an empty artifact so the applier
        no-ops.
        """
        valid_cols = resolve_fit_text_valid_columns(X, config, _y)
        if valid_cols is None:
            return {}

        artifact = _build_tokenizer_artifact(config, valid_cols)
        validate_text_output_names(X, artifact, valid_cols)
        return artifact
