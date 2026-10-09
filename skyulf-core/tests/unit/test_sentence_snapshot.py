"""Inspect embedding snapshot contracts without importing the optional neural runtime."""

import hashlib

import pytest

from skyulf.preprocessing.vectorization import sentence_embedder as embedder


def _state():
    """Provide bounded transport bytes for structural checks that never deserialize."""
    return {
        "type": "sentence_embedder",
        "columns": ["text"],
        "model_name": "original",
        "embedding_dim": 1,
        "normalize": True,
        "output_columns": ["text__emb__0"],
        "drop_original": True,
        "model_snapshot": b"structural-test-only",
        "model_sha256": hashlib.sha256(b"structural-test-only").hexdigest(),
        "model_requirements": tuple(f"{name}==1.0" for name in embedder._MODEL_PACKAGES),
    }


@pytest.mark.parametrize("engine", ["pandas", "polars"])
def test_snapshot_context_checks_bytes_without_loading_model(monkeypatch, engine):
    """Metadata inspection must not run native model code or retrieve a Hub model."""
    monkeypatch.setattr(embedder, "version", lambda name: "1.0")

    def forbidden(*args, **kwargs):
        """Fail if a descriptive hook starts an encoder."""
        raise AssertionError("Metadata inspection loaded a model")

    monkeypatch.setattr(embedder, "_load_model", forbidden)
    state = _state()
    capability = embedder.SentenceEmbedderApplier.inference_capability(state, engine=engine)
    assert capability is not None
    assert capability.context == "row"
    assert capability.execution_kind == "local"
    assert embedder.SentenceEmbedderApplier.validate_inference_state(state) is state


@pytest.mark.parametrize(
    "field,value",
    [
        ("model_snapshot", bytearray(b"structural-test-only")),
        ("model_snapshot", b""),
        ("model_snapshot", b"changed"),
        ("model_requirements", ()),
        ("model_requirements", ("torch==2.0",) * 6),
        ("model_sha256", "0" * 64),
        ("embedding_dim", True),
        ("embedding_dim", 2),
        ("normalize", [True, False]),
        ("model_name", None),
    ],
)
def test_malformed_snapshot_fails_inspection(monkeypatch, field, value):
    """A corrupt or incomplete snapshot must not receive a row-context declaration."""
    monkeypatch.setattr(embedder, "version", lambda name: "1.0")
    state = {**_state(), field: value}
    with pytest.raises(ValueError):
        embedder.SentenceEmbedderApplier.validate_inference_state(state)


def test_legacy_name_only_state_has_no_portability_promise():
    """Old artifacts remain readable but depend on an external model name or directory."""
    state = {
        key: value
        for key, value in _state().items()
        if key not in {"model_snapshot", "model_sha256", "model_requirements"}
    }
    assert embedder.SentenceEmbedderApplier.inference_capability(state, engine="pandas") is None
    capability = embedder.SentenceEmbedderApplier.inference_capability({}, engine="pandas")
    assert capability is not None and capability.context == "row"


@pytest.mark.parametrize("flag", [None, 0, 1, "yes"])
def test_saved_scalar_flag_semantics_remain_inspectable(monkeypatch, flag):
    """Metadata must accept native truthy/falsy scalar options without rewriting them."""
    monkeypatch.setattr(embedder, "version", lambda name: "1.0")
    state = {**_state(), "normalize": flag, "drop_original": flag}
    assert embedder.SentenceEmbedderApplier.validate_inference_state(state) is state
