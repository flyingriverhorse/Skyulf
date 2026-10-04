"""Exercise the real optional encoder without downloading model weights."""

import json
from pathlib import Path

import numpy as np
import pandas as pd
import polars as pl
import pytest

sentence_transformers = pytest.importorskip("sentence_transformers")
encoder_modules = pytest.importorskip("sentence_transformers.sentence_transformer.modules")

from skyulf.preprocessing.vectorization.sentence_embedder import (
    SentenceEmbedderApplier,
    SentenceEmbedderCalculator,
)


@pytest.fixture
def local_encoder(tmp_path: Path) -> Path:
    """Save a real built-in model so compatibility checks need no Hub access."""
    model = sentence_transformers.SentenceTransformer(
        modules=[encoder_modules.BoW(vocab=["hello", "world", "test"])], device="cpu"
    )
    model.save_pretrained(str(tmp_path), create_model_card=False)
    return tmp_path


@pytest.mark.parametrize("engine", ["pandas", "polars"])
@pytest.mark.parametrize("normalize", [False, True])
def test_real_encoder_round_trip(local_encoder: Path, engine: str, normalize: bool, monkeypatch):
    """Dependency upgrades must preserve real encoding, row identity and normalization."""
    monkeypatch.setenv("SKYULF_ENGINE", engine)
    data = {"text": ["hello world", "test"], "keep": [20, 10]}
    frame = pd.DataFrame(data, index=[8, 3]) if engine == "pandas" else pl.DataFrame(data)
    artifact = SentenceEmbedderCalculator().fit(
        frame, {"columns": ["text"], "model_name": str(local_encoder), "normalize": normalize}
    )
    result = SentenceEmbedderApplier().apply(frame, artifact)
    embeddings = result[artifact["output_columns"]].to_numpy()
    assert embeddings.shape == (2, 3)
    assert embeddings.dtype == np.float32
    assert np.isfinite(embeddings).all()
    assert list(result["keep"]) == data["keep"]
    assert list(result["text"]) == data["text"]
    if isinstance(frame, pd.DataFrame):
        assert result.index.equals(frame.index)
    expected_norms = [1, 1] if normalize else [np.sqrt(2), 1]
    np.testing.assert_allclose(np.linalg.norm(embeddings, axis=1), expected_norms, rtol=1e-6)


def test_local_custom_module_rejected_before_import(local_encoder: Path):
    """A local model must not execute its Python module through the default loader."""
    marker = local_encoder / "module_imported.txt"
    (local_encoder / "modeling_security_probe.py").write_text(
        f"from pathlib import Path\nPath({str(marker)!r}).write_text('imported')\n"
        "raise RuntimeError('Untrusted model module executed')\n",
        encoding="utf-8",
    )
    (local_encoder / "modules.json").write_text(
        json.dumps([{"idx": 0, "name": "0", "path": "", "type": "modeling_security_probe.Probe"}]),
        encoding="utf-8",
    )
    with pytest.raises(ValueError, match="trust_remote_code"):
        SentenceEmbedderCalculator().fit(
            pd.DataFrame({"text": ["hello"]}),
            {"columns": ["text"], "model_name": str(local_encoder)},
        )
    assert not marker.exists()
