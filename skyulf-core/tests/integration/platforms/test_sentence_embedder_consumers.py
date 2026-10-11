"""Native sentence snapshots must survive model-set and MLflow package transport."""

import json
import os
import subprocess
import sys
from pathlib import Path

import numpy as np
import pytest

pytest.importorskip("sentence_transformers")
torch = pytest.importorskip("torch")

from tests.integration.platforms.test_sentence_embedder_artifact import _pipeline

from skyulf.inference._manifest import ColumnSpec
from skyulf.inference.fitted_pipeline import load_pipeline, save_pipeline
from skyulf.inference.model_set import ComponentReference, save_model_set
from skyulf.preprocessing.vectorization import sentence_embedder as embedder


def _offline_replay(kind, package, sample, tmp_path):
    """Restore the moved package with empty caches and model constructors disabled."""
    script = """
import json, socket, sys
def forbidden(*args, **kwargs):
    raise AssertionError('Replay attempted an external model lookup')
socket.socket.connect = forbidden
socket.create_connection = forbidden
import pandas as pd
import sentence_transformers
sentence_transformers.SentenceTransformer.__init__ = forbidden
from skyulf.preprocessing.vectorization import sentence_embedder as embedder
embedder._MODEL_CACHE.clear()
embedder._MODEL_LOADS.clear()
sample = pd.DataFrame(json.loads(sys.argv[3]))
if sys.argv[1] == 'model_set':
    from skyulf.inference.model_set import load_model_set
    from skyulf.inference.model_set_scoring import predict_model_set
    result = predict_model_set(sample, load_model_set(sys.argv[2]))
else:
    import mlflow
    result = mlflow.pyfunc.load_model(sys.argv[2]).predict(sample)
print('RESULT=' + json.dumps(result.to_dict(orient='list')))
"""
    result = subprocess.run(
        [
            sys.executable,
            "-c",
            script,
            kind,
            str(package),
            json.dumps(sample.to_dict(orient="list")),
        ],
        env={
            **os.environ,
            "HF_HUB_OFFLINE": "1",
            "TRANSFORMERS_OFFLINE": "1",
            "HF_HOME": str(tmp_path / "empty-cache"),
            "SENTENCE_TRANSFORMERS_HOME": str(tmp_path / "empty-cache"),
            "PYTHONPATH": str(Path(__file__).resolve().parents[3]),
        },
        cwd=tmp_path,
        capture_output=True,
        text=True,
        timeout=90,
    )
    assert result.returncode == 0, result.stdout + result.stderr
    return json.loads(
        next(line[7:] for line in result.stdout.splitlines() if line.startswith("RESULT="))
    )


def _retire_sources(source, component, package, tmp_path):
    """Remove all original names and process caches before replaying the relocated package."""
    source.rename(tmp_path / "retired-encoder")
    component.rename(tmp_path / "retired-component")
    moved = package.rename(tmp_path / "relocated-package")
    embedder._MODEL_CACHE.clear()
    embedder._MODEL_LOADS.clear()
    return moved


def test_model_set_replays_exact_sentence_snapshot_offline(tmp_path):
    """A copied component must preserve fitted embeddings and caller record-key order."""
    pipeline, source, sample = _pipeline(tmp_path)
    expected = pipeline.predict(sample)
    original_model = embedder._MODEL_CACHE[str(source)]
    with torch.no_grad():
        for parameter in original_model.parameters():
            parameter.zero_()
    original_model.save_pretrained(str(source), create_model_card=False)
    assert np.count_nonzero(original_model.encode(sample["text"].tolist())) == 0
    component = tmp_path / "component"
    save_pipeline(pipeline, component)
    artifact = load_pipeline(component)
    package = save_model_set(
        tmp_path / "package",
        {
            "text": (
                ComponentReference(
                    name="catalog.schema.text",
                    version="1",
                    digest=artifact.manifest.pipeline_sha256,
                ),
                component,
            )
        },
        record_key_schema=(ColumnSpec(name="record_id", dtype="int64"),),
    )
    copied = package.directory / "components" / "text"
    assert (copied / "pipeline.pkl").read_bytes() == (component / "pipeline.pkl").read_bytes()
    assert (
        load_pipeline(copied).manifest.project_requirements
        == artifact.manifest.project_requirements
    )
    query = sample.assign(record_id=[17, 3, 21, 8, 13, 1])
    moved = _retire_sources(source, component, package.directory, tmp_path)
    actual = _offline_replay("model_set", moved, query, tmp_path)
    np.testing.assert_allclose(actual["text__prediction"], expected, rtol=0, atol=1e-9)
    assert actual["record_id"] == query["record_id"].tolist()


def test_local_pyfunc_replays_sentence_snapshot_and_dependency_pins(tmp_path):
    """MLflow must copy native encoder bytes and install their exact optional runtime."""
    mlflow = pytest.importorskip("mlflow")
    from skyulf.integrations.mlflow.models.pipeline_model import pipeline_model_save_options

    pipeline, source, sample = _pipeline(tmp_path)
    expected = pipeline.predict(sample)
    component = tmp_path / "component"
    save_pipeline(pipeline, component)
    artifact = load_pipeline(component)
    package = tmp_path / "package"
    environment = tmp_path / "environment"
    environment.mkdir()
    mlflow.pyfunc.save_model(
        path=str(package), **pipeline_model_save_options(artifact, component, environment)
    )
    pins = set((package / "requirements.txt").read_text(encoding="utf-8").splitlines())
    assert set(artifact.manifest.project_requirements) <= pins
    moved = _retire_sources(source, component, package, tmp_path)
    actual = _offline_replay("local_pipeline", moved, sample, tmp_path)
    np.testing.assert_allclose(actual["prediction"], expected, rtol=0, atol=1e-9)
