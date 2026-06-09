"""SentenceEmbedder node — dense semantic embeddings via sentence-transformers.

Optional dependency: requires the ``sentence-transformers`` package (install via
``pip install skyulf[nlp]`` or ``pip install sentence-transformers``).  The import
is lazy so the rest of skyulf-core works without it installed.

The Calculator loads the model and records the embedding dimension; the Applier
encodes text into ``{src}__emb__{i}`` float columns.  Models are cached per
``model_name`` at module level so repeated applies don't reload weights.
"""

import logging
from typing import Any, Dict, List, Tuple

import pandas as pd

from ...core.meta.decorators import node_meta
from ...registry import NodeRegistry
from .._artifacts import SentenceEmbedderArtifact
from ..base import BaseApplier, BaseCalculator, apply_method, fit_method
from ._common import _join_text_columns, apply_text_pandas_only

logger = logging.getLogger(__name__)

_MODEL_CACHE: Dict[str, Any] = {}

_INSTALL_HINT = (
    "SentenceEmbedder requires the 'sentence-transformers' package. "
    "Install it with: pip install skyulf[nlp]  (or  pip install sentence-transformers)"
)


def _load_model(model_name: str) -> Any:
    """Lazily import sentence-transformers and return a cached model."""
    if model_name in _MODEL_CACHE:
        return _MODEL_CACHE[model_name]
    try:
        from sentence_transformers import SentenceTransformer  # ty: ignore[unresolved-import]
    except ImportError as exc:  # pragma: no cover - depends on optional extra
        raise ImportError(_INSTALL_HINT) from exc

    model = SentenceTransformer(model_name)
    _MODEL_CACHE[model_name] = model
    return model


# ── Apply ─────────────────────────────────────────────────────────────────────


def _embed_apply_pandas(
    X: pd.DataFrame, y: Any, params: Dict[str, Any]
) -> Tuple[pd.DataFrame, Any]:
    cols: List[str] = params.get("columns", [])
    output_columns: List[str] = params.get("output_columns", [])
    model_name: str = params.get("model_name", "all-MiniLM-L6-v2")
    normalize: bool = params.get("normalize", True)
    drop_original: bool = params.get("drop_original", False)

    valid_cols = [c for c in cols if c in X.columns]
    if not valid_cols or not output_columns:
        return X, y

    model = _load_model(model_name)
    text = _join_text_columns(X, valid_cols).tolist()
    embeddings = model.encode(text, normalize_embeddings=normalize, show_progress_bar=False)

    emb_df = pd.DataFrame(
        embeddings,
        columns=output_columns,  # ty: ignore[invalid-argument-type]
        index=X.index,
    )

    X_out = X.copy()
    if drop_original:
        X_out = X_out.drop(columns=valid_cols)
    return pd.concat([X_out, emb_df], axis=1), y


class SentenceEmbedderApplier(BaseApplier):
    @apply_method
    def apply(self, X: Any, _y: Any, params: Dict[str, Any]) -> Any:
        return apply_text_pandas_only(X, params, _embed_apply_pandas)


# ── Calculate ─────────────────────────────────────────────────────────────────


@NodeRegistry.register("sentence_embedder", SentenceEmbedderApplier)
@node_meta(
    id="sentence_embedder",
    name="Sentence Embedder",
    category="Text",
    description=(
        "Encode text columns into dense semantic embeddings using a "
        "sentence-transformers model (default all-MiniLM-L6-v2). Requires the "
        "optional 'sentence-transformers' package."
    ),
    params={
        "columns": [],
        "model_name": "all-MiniLM-L6-v2",
        "normalize": True,
        "drop_original": False,
    },
    tags=["text", "nlp", "embeddings", "transformers"],
)
class SentenceEmbedderCalculator(BaseCalculator):
    def infer_output_schema(self, input_schema: Any, config: Dict[str, Any]) -> None:
        return None

    @fit_method
    def fit(self, X: Any, _y: Any, config: Dict[str, Any]) -> SentenceEmbedderArtifact:
        cols: List[str] = config.get("columns", [])
        if not cols:
            return {}

        if hasattr(X, "to_pandas"):
            X = X.to_pandas()

        valid_cols = [c for c in cols if c in X.columns]
        if not valid_cols:
            return {}

        model_name: str = config.get("model_name") or "all-MiniLM-L6-v2"
        model = _load_model(model_name)
        embedding_dim = int(model.get_sentence_embedding_dimension())

        prefix = valid_cols[0] if len(valid_cols) == 1 else "_".join(valid_cols)
        output_columns = [f"{prefix}__emb__{i}" for i in range(embedding_dim)]

        return {
            "type": "sentence_embedder",
            "columns": valid_cols,
            "model_name": model_name,
            "embedding_dim": embedding_dim,
            "normalize": config.get("normalize", True),
            "output_columns": output_columns,
            "drop_original": config.get("drop_original", False),
        }
