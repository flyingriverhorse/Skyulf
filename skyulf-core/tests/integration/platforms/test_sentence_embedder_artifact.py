"""Embedding assets must survive transport without the original model or Hub cache."""

import json
import os
import subprocess
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import polars as pl
import pytest

sentence_transformers = pytest.importorskip("sentence_transformers")
encoder_modules = pytest.importorskip("sentence_transformers.sentence_transformer.modules")
tokenizer_modules = pytest.importorskip(
    "sentence_transformers.sentence_transformer.modules.tokenizer"
)
torch = pytest.importorskip("torch")

from skyulf.data.dataset import SplitDataset
from skyulf.inference.local_pipeline import (
    load_local_pipeline,
    predict_local_pipeline,
    save_local_pipeline,
)
from skyulf.pipeline import SkyulfPipeline


def _pipeline(tmp_path, engine="pandas"):
    """Use a real native encoder; neither encoding nor persistence is mocked."""
    source = tmp_path / "source"
    tokenizer = tokenizer_modules.WhitespaceTokenizer(
        vocab=["[PAD]", "hello", "world", "test"], stop_words=[], do_lower_case=True
    )
    model = sentence_transformers.SentenceTransformer(
        modules=[
            encoder_modules.WordEmbeddings(tokenizer, torch.eye(4, dtype=torch.float32)),
            encoder_modules.Pooling(4, pooling_mode="mean"),
        ],
        device="cpu",
    )
    model.save_pretrained(str(source), create_model_card=False)
    frame = pd.DataFrame(
        {
            "text": ["hello", "world", "test", "hello world", "hello test", "world test"],
            "target": [1.0, 2.0, 3.0, 3.0, 4.0, 5.0],
        }
    )
    pipeline = SkyulfPipeline(
        {
            "preprocessing": [
                {
                    "name": "embed",
                    "transformer": "sentence_embedder",
                    "params": {
                        "columns": ["text"],
                        "model_name": str(source),
                        "drop_original": True,
                    },
                }
            ],
            "modeling": {"type": "linear_regression"},
        }
    )
    data = pl.from_pandas(frame) if engine == "polars" else frame
    pipeline.fit(SplitDataset(train=data[:5], test=data[5:]), target_column="target")
    sample = frame.drop(columns="target")
    return pipeline, source, pl.from_pandas(sample) if engine == "polars" else sample


def test_embedding_artifact_replays_in_fresh_offline_process(tmp_path):
    """The fitted model must carry its encoder after source removal and cache isolation."""
    pipeline, source, sample = _pipeline(tmp_path)
    expected = pipeline.predict(sample).tolist()
    save_local_pipeline(pipeline, tmp_path / "artifact")
    source.rename(tmp_path / "source-unavailable")
    script = """
import json, socket, sys
def blocked(*args, **kwargs):
    raise AssertionError('Offline replay attempted network access')
socket.socket.connect = blocked
socket.create_connection = blocked
import pandas as pd
from skyulf.inference.local_pipeline import load_local_pipeline, predict_local_pipeline
artifact = load_local_pipeline(sys.argv[1])
result = predict_local_pipeline(pd.DataFrame(json.loads(sys.argv[2])), artifact)
print('RESULT=' + json.dumps(result.to_dict(orient='list')))
"""
    env = {
        **os.environ,
        "HF_HUB_OFFLINE": "1",
        "TRANSFORMERS_OFFLINE": "1",
        "HF_HOME": str(tmp_path / "empty-cache"),
        "SENTENCE_TRANSFORMERS_HOME": str(tmp_path / "empty-cache"),
        "PYTHONPATH": str(Path(__file__).resolve().parents[3]),
    }
    result = subprocess.run(
        [
            sys.executable,
            "-c",
            script,
            str(tmp_path / "artifact"),
            json.dumps(sample.to_dict(orient="list")),
        ],
        env=env,
        cwd=tmp_path,
        capture_output=True,
        text=True,
        timeout=90,
    )
    assert result.returncode == 0, result.stdout + result.stderr
    actual = json.loads(
        next(line[7:] for line in result.stdout.splitlines() if line.startswith("RESULT="))
    )
    np.testing.assert_allclose(actual["prediction"], expected, atol=1e-9)


def test_embedding_dependency_pins_travel_with_local_manifest(tmp_path):
    """MLflow must install the optional runtime required by the saved embedding bytes."""
    pipeline, _, sample = _pipeline(tmp_path)
    save_local_pipeline(pipeline, tmp_path / "artifact")
    restored = load_local_pipeline(tmp_path / "artifact")
    pins = {pin.split("==")[0] for pin in restored.manifest.project_requirements}
    assert {"sentence-transformers", "transformers", "torch", "tokenizers"} <= pins
    np.testing.assert_allclose(
        predict_local_pipeline(sample, restored)["prediction"], pipeline.predict(sample)
    )


@pytest.mark.parametrize("engine", ["pandas", "polars"])
def test_saved_encoder_probe_reuses_exact_weights_after_cache_loss(tmp_path, monkeypatch, engine):
    """Original model changes and empty caches cannot alter the immutable fitted encoder."""
    from skyulf.inference.preprocessing_probe import probe_fitted_preprocessing
    from skyulf.preprocessing.vectorization import sentence_embedder as embedder

    pipeline, source, sample = _pipeline(tmp_path, engine)
    expected = pipeline.predict(sample)
    with torch.no_grad():
        embedder._MODEL_CACHE[str(source)][0].emb_layer.weight.zero_()
    save_local_pipeline(pipeline, tmp_path / "artifact")
    source.rename(tmp_path / "source-unavailable")
    monkeypatch.setattr(embedder, "_MODEL_CACHE", {})

    def forbidden(*args, **kwargs):
        """A saved encoder must not construct another pretrained model."""
        raise AssertionError("Saved apply tried to load the original model")

    monkeypatch.setattr(sentence_transformers.SentenceTransformer, "__init__", forbidden)
    artifact = load_local_pipeline(tmp_path / "artifact")
    report = probe_fitted_preprocessing(artifact, sample)
    assert report["status"] == "passed", report
    assert report["steps"][0]["context"] == "row"
    np.testing.assert_array_equal(predict_local_pipeline(sample, artifact)["prediction"], expected)
