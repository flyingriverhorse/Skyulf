"""SentenceEmbedder node — dense semantic embeddings via sentence-transformers.

Optional dependency: requires the ``sentence-transformers`` package (install via
``pip install skyulf[nlp]`` or ``pip install sentence-transformers``).  The import
is lazy so the rest of skyulf-core works without it installed.

The Calculator captures native PyTorch weights, tokenizer and configuration as
immutable bytes. The Applier reuses those bytes on CPU without downloading a
model. Legacy name-only artifacts retain their original external dependency.
Only load saved artifacts from a trusted producer: native snapshots use pickle.
"""

import hashlib
import logging
from concurrent.futures import Future
from importlib.metadata import version
from io import BytesIO
from threading import Lock
from typing import Any

import numpy as np
import pandas as pd
import polars as pl

from ...core.capabilities import ExecutionCapability
from ...core.meta.decorators import node_meta
from ...registry import NodeRegistry
from .._artifacts import SentenceEmbedderArtifact
from .._fitted_validation import local_scalar, local_state_fields
from ..base import BaseApplier, BaseCalculator, apply_method, fit_method
from ._common import (
    _drop_and_concat,
    _drop_and_concat_polars,
    _join_text_columns,
    _join_text_columns_polars,
    _validate_text_layout,
    apply_text_dual_engine,
    resolve_fit_text_valid_columns,
    validate_text_output_names,
)

logger = logging.getLogger(__name__)

_MODEL_CACHE: dict[str | bytes, Any] = {}
_MODEL_LOADS: dict[str | bytes, Future[Any]] = {}
_MODEL_CACHE_LOCK = Lock()
_MODEL_PACKAGES = (
    "sentence-transformers",
    "transformers",
    "torch",
    "tokenizers",
    "huggingface-hub",
    "safetensors",
)
_MAX_MODEL_BYTES = 256 * 1024 * 1024

_INSTALL_HINT = (
    "SentenceEmbedder requires the 'sentence-transformers' package. "
    "Install it with: pip install skyulf[nlp]  (or  pip install sentence-transformers)"
)


def _load_model(model_name: str | bytes) -> Any:
    """Share one in-process construction per key, including failures, without blocking other keys."""
    with _MODEL_CACHE_LOCK:
        if model_name in _MODEL_CACHE:
            return _MODEL_CACHE[model_name]
        pending = _MODEL_LOADS.get(model_name)
        owner = pending is None
        if owner:
            pending = Future()
            _MODEL_LOADS[model_name] = pending
    if not owner:
        return pending.result()
    try:
        model = _construct_model(model_name)
    except BaseException as exc:
        # A notified waiter may retry immediately while this owner is still unwinding.
        with _MODEL_CACHE_LOCK:
            _MODEL_LOADS.pop(model_name, None)
        # Wake every waiter even when loading is interrupted rather than raising Exception.
        pending.set_exception(exc)
        raise
    with _MODEL_CACHE_LOCK:
        _MODEL_CACHE[model_name] = model
        _MODEL_LOADS.pop(model_name, None)
    pending.set_result(model)
    return model


def _construct_model(model_name: str | bytes) -> Any:
    """Import the optional dependency and load model weights outside the cache lock."""
    try:
        # ty: ignore[unresolved-import]
        from sentence_transformers import (  # noqa: PLC0415 - optional nlp extra
            SentenceTransformer,
        )
    except ImportError as exc:  # pragma: no cover - depends on optional extra
        raise ImportError(_INSTALL_HINT) from exc

    if isinstance(model_name, bytes):
        import torch  # noqa: PLC0415 - optional nlp extra

        # Like pipeline.pkl, this is executable state from a trusted producer.
        model = torch.load(BytesIO(model_name), map_location="cpu", weights_only=False)
        if type(model) is not SentenceTransformer or model.get_backend() != "torch":
            raise ValueError("Saved sentence encoder must be a native PyTorch SentenceTransformer.")
        return model.eval()
    return SentenceTransformer(model_name)


def _snapshot_model(model: Any) -> dict[str, Any]:
    """Capture native weights, tokenizer and config without filesystem or Hub references."""
    import torch  # noqa: PLC0415 - optional nlp extra
    from sentence_transformers import SentenceTransformer  # noqa: PLC0415 - optional nlp extra

    if type(model) is not SentenceTransformer or model.get_backend() != "torch":
        raise ValueError("Sentence snapshots require a native PyTorch SentenceTransformer.")
    buffer = BytesIO()
    torch.save(model, buffer)
    payload = buffer.getvalue()
    if len(payload) > _MAX_MODEL_BYTES:
        raise ValueError("Sentence model snapshot exceeds the 256 MiB size limit.")
    return {
        "model_snapshot": payload,
        "model_sha256": hashlib.sha256(payload).hexdigest(),
        "model_requirements": tuple(f"{name}=={version(name)}" for name in _MODEL_PACKAGES),
    }


def _validate_snapshot(params: dict) -> None:
    """Check immutable bytes and exact optional packages before native deserialization."""
    payload = params["model_snapshot"]
    if type(payload) is not bytes or not 0 < len(payload) <= _MAX_MODEL_BYTES:
        raise ValueError("Sentence model snapshot must be nonempty bounded bytes.")
    if hashlib.sha256(payload).hexdigest() != params["model_sha256"]:
        raise ValueError("Sentence model snapshot checksum mismatch.")
    requirements = params["model_requirements"]
    if type(requirements) is not tuple or len(requirements) != len(_MODEL_PACKAGES):
        raise ValueError("Sentence model runtime requirements are incomplete.")
    for name, pin in zip(_MODEL_PACKAGES, requirements, strict=True):
        if pin != f"{name}=={version(name)}":
            raise ValueError(f"Sentence model runtime mismatch for {name}: saved {pin}.")


def _embedding_dimension(model: Any) -> int:
    """Return the model's embedding dimension across sentence-transformers versions.

    ``get_sentence_embedding_dimension`` was renamed to ``get_embedding_dimension``
    in newer releases; prefer the new name and fall back to the old one.
    """
    getter = (
        getattr(model, "get_embedding_dimension", None) or model.get_sentence_embedding_dimension
    )
    return int(getter())


# ── Apply ─────────────────────────────────────────────────────────────────────


def _encode_text(text: list[str], params: dict, output_width: int) -> np.ndarray:
    """Keep the fitted embedding width for empty partitions without loading model weights."""
    if not text:
        return np.empty((0, output_width), dtype=np.float32)
    if "model_snapshot" in params:
        _validate_snapshot(params)
    model = _load_model(params.get("model_snapshot", params.get("model_name", "all-MiniLM-L6-v2")))
    return model.encode(
        text, normalize_embeddings=params.get("normalize", True), show_progress_bar=False
    )


def _embed_apply_pandas(
    X: pd.DataFrame, y: Any, params: dict[str, Any]
) -> tuple[pd.DataFrame, Any]:
    """Encode the original text and attach embeddings without aliasing retained columns."""
    cols: list[str] = params.get("columns", [])
    output_columns: list[str] = params.get("output_columns", [])
    drop_original: bool = params.get("drop_original", False)

    valid_cols = [c for c in cols if c in X.columns]
    if not valid_cols or not output_columns:
        return X, y

    text = _join_text_columns(X, valid_cols).tolist()
    embeddings = _encode_text(text, params, len(output_columns))

    emb_df = pd.DataFrame(
        embeddings,
        columns=output_columns,  # ty: ignore[invalid-argument-type]
        index=X.index,
    )

    return _drop_and_concat(X, emb_df, valid_cols, drop_original), y


def _embed_apply_polars(X: Any, params: dict[str, Any]) -> Any:
    """Native-Polars embed apply.

    Returns ``None`` to fall back to pandas when a text column is not String
    dtype.
    """
    cols: list[str] = params.get("columns", [])
    output_columns: list[str] = params.get("output_columns", [])
    drop_original: bool = params.get("drop_original", False)

    valid_cols = [c for c in cols if c in X.columns]
    if not valid_cols or not output_columns:
        return X

    validate_text_output_names(X, params, valid_cols)
    text = _join_text_columns_polars(X, valid_cols)
    if text is None:
        return None

    embeddings = _encode_text(text.to_list(), params, len(output_columns))
    emb_frame = pl.from_numpy(np.asarray(embeddings), schema=output_columns)
    return _drop_and_concat_polars(X, emb_frame, valid_cols, drop_original)


class SentenceEmbedderApplier(BaseApplier):
    """Encode the joined text of the configured columns into ``embedding_dim`` float columns.

    All configured columns are concatenated into one corpus, so a multi-column
    selection yields a single embedding whose name prefix joins the source names.
    Width is set by the model, not by config, and is recorded in the artifact at
    fit time. New artifacts cache loaded CPU encoders by their immutable saved
    bytes, so models sharing a display name cannot select each other's weights.
    ``normalize=True`` (the default)
    yields unit-length vectors ready for cosine similarity. Because the optional
    ``sentence-transformers`` import is lazy, a missing extra surfaces here as an
    ``ImportError`` carrying the install hint rather than at skyulf import time.
    The polars path keeps the frame native around the encode, but the encoding
    itself always runs outside the dataframe engine.
    """

    @staticmethod
    def validate_inference_state(raw: dict) -> dict:
        """Inspect saved schema and asset identity without loading or running the encoder."""
        if type(raw) is not dict:
            raise ValueError("Local fitted state must be a dictionary.")
        fields = {
            "type",
            "columns",
            "model_name",
            "embedding_dim",
            "normalize",
            "output_columns",
            "drop_original",
        }
        assets = {"model_snapshot", "model_sha256", "model_requirements"}
        fields |= assets if assets.intersection(raw) else set()
        if not local_state_fields(raw, "sentence_embedder", fields, allow_empty=True):
            return raw
        local_scalar(raw["normalize"], "normalize")
        local_scalar(raw["drop_original"], "drop_original")
        _validate_text_layout({**raw, "drop_original": bool(raw["drop_original"])})
        if not isinstance(raw["model_name"], str) or not raw["model_name"]:
            raise ValueError("Fitted sentence model name must be nonempty text.")
        width = raw["embedding_dim"]
        if type(width) is not int or width <= 0 or len(raw["output_columns"]) != width:
            raise ValueError("Fitted sentence embedding width disagrees with its columns.")
        if "model_snapshot" in raw:
            _validate_snapshot(raw)
        return raw

    @staticmethod
    def inference_capability(state: dict, *, engine: str) -> ExecutionCapability | None:
        """Describe packaged row-local encoding; name-only legacy state stays unknown."""
        if engine not in ("pandas", "polars"):
            return None
        SentenceEmbedderApplier.validate_inference_state(state)
        if state and "model_snapshot" not in state:
            return None
        return ExecutionCapability(engine, "apply", "local", "preserve", "row")

    @apply_method
    def apply(self, X: Any, _y: Any, params: dict[str, Any]) -> Any:  # pylint: disable=arguments-differ
        """Dispatch through the text-specific dual-engine path; ``y`` passes through."""
        return apply_text_dual_engine(X, params, _embed_apply_pandas, _embed_apply_polars)


# ── Calculate ─────────────────────────────────────────────────────────────────


def _build_sentence_embedder_artifact(
    config: dict[str, Any], valid_cols: list[str]
) -> SentenceEmbedderArtifact:
    """Load the embedding model and build the sentence-embedder artifact dict."""
    model_name: str = config.get("model_name") or "all-MiniLM-L6-v2"
    model = _load_model(model_name)
    embedding_dim = int(_embedding_dimension(model))

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
        **_snapshot_model(model),
    }


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
    learns_from_data=False,
)
class SentenceEmbedderCalculator(BaseCalculator):
    """Resolve text columns and capture the pretrained encoder used by subsequent applies.

    ``learns_from_data=False`` — the weights come pretrained and are never
    adjusted here — but fitting is not free: it performs the lazy
    ``sentence-transformers`` import and, on a cold cache, downloads the model.
    Native PyTorch serialization captures its weights, tokenizer and settings in
    memory, with a SHA256 identity and exact optional dependency versions. The
    existing local pipeline/model-set size budgets still apply to these bytes.

    Schema inference inherits the base ``None`` result because the embedding
    width is only known after loading the model. Previewing the schema therefore
    does not load model weights and leaves the output unknown until runtime.
    """

    @fit_method
    def fit(self, X: Any, _y: Any, config: dict[str, Any]) -> SentenceEmbedderArtifact:  # pylint: disable=arguments-differ
        """Resolve the configured text column names and record the model's embedding width.

        Uses the column-names-only resolver rather than the data-narrowing
        sibling, so a polars input is never converted to pandas just to fit. An
        empty or wholly-absent selection yields an empty artifact so the applier
        no-ops — and, importantly, skips loading the model at all.
        """
        valid_cols = resolve_fit_text_valid_columns(X, config, _y)
        if valid_cols is None:
            return {}

        artifact = _build_sentence_embedder_artifact(config, valid_cols)
        validate_text_output_names(X, artifact, valid_cols)
        return artifact
