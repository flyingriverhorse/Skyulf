"""Purpose-based inference APIs own their implementations and replay fitted models."""

import importlib
import importlib.util
import os
import pickle
import subprocess
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import polars as pl
import pytest

from skyulf.data.dataset import SplitDataset
from skyulf.pipeline import SkyulfPipeline


@pytest.mark.parametrize(
    "canonical, legacy, symbols",
    [
        (
            "skyulf.inference.fitted_pipeline",
            "skyulf.inference.local_pipeline",
            [
                ("FittedPipelineArtifact", "LocalPipelineArtifact"),
                ("FittedPipelineManifest", "LocalPipelineManifest"),
                ("save_pipeline", "save_local_pipeline"),
                ("load_pipeline", "load_local_pipeline"),
                ("predict_pipeline", "predict_local_pipeline"),
                ("validate_pipeline_input", "validate_local_input"),
                ("require_pipeline_scope", "require_local_pipeline_scope"),
            ],
        ),
        (
            "skyulf.inference.pipeline_scoring",
            "skyulf.inference.local_scoring",
            [
                ("PipelinePrediction", "LocalPrediction"),
                ("score_pipeline", "score_local_pipeline"),
                ("score_pipeline_with_history", "score_local_pipeline_with_history"),
                ("pipeline_history_session", "local_history_session"),
            ],
        ),
        (
            "skyulf.inference.pipeline_evaluation",
            "skyulf.inference.local_evaluation",
            [("evaluate_holdout", "evaluate_local_holdout")],
        ),
        (
            "skyulf.integrations.mlflow.models.pipeline_model",
            "skyulf.integrations.mlflow.models.local_model",
            [
                ("SkyulfPipelinePythonModel", "SkyulfLocalPythonModel"),
                ("log_pipeline_model", "log_local_model"),
                ("pipeline_model_save_options", "local_model_save_options"),
            ],
        ),
        (
            "skyulf.integrations.mlflow.models.feature_model",
            "skyulf.integrations.mlflow.models.local_feature_model",
            [("log_feature_pipeline_model", "log_local_feature_model")],
        ),
    ],
)
def test_canonical_modules_own_public_api(canonical, legacy, symbols):
    """Saved classes resolve directly from the public implementation modules."""
    if ".mlflow." in canonical:
        pytest.importorskip("mlflow")
    assert importlib.util.find_spec(canonical) is not None
    implementation = importlib.import_module(canonical)
    assert importlib.util.find_spec(legacy) is None
    for name, old_name in symbols:
        value = getattr(implementation, name)
        assert not hasattr(implementation, old_name)
        assert value.__module__ == canonical
        assert value.__name__ == name
        restored = pickle.loads(pickle.dumps(value))
        assert restored is value


@pytest.mark.parametrize("engine", ["pandas", "polars"])
def test_pipeline_roundtrip_and_object_reload_in_fresh_process(engine, tmp_path):
    """A saved fit retains its engine, digest and predictions in a fresh process."""
    assert importlib.util.find_spec("skyulf.inference.fitted_pipeline") is not None
    fitted = importlib.import_module("skyulf.inference.fitted_pipeline")
    scoring = importlib.import_module("skyulf.inference.pipeline_scoring")
    evaluation = importlib.import_module("skyulf.inference.pipeline_evaluation")
    frame = pd.DataFrame({"x": [1.0, 2.0, 3.0, 4.0], "target": [3.0, 5.0, 7.0, 9.0]})
    native = pl.from_pandas(frame) if engine == "polars" else frame
    pipeline = SkyulfPipeline({"preprocessing": [], "modeling": {"type": "linear_regression"}})
    pipeline.fit(SplitDataset(train=native, test=native.head(0)), target_column="target")
    artifact_path = tmp_path / "fitted"
    fitted.save_pipeline(pipeline, artifact_path)
    artifact = fitted.load_pipeline(artifact_path)
    query = pd.DataFrame({"x": [5.0, 6.0]})
    result = scoring.score_pipeline_with_history(query, artifact)
    np.testing.assert_allclose(result.frame["prediction"], [11.0, 13.0])
    assert result.history is None
    assert evaluation.evaluate_holdout(artifact, native, target_column="target")["heldout_r2"] == 1
    assert artifact.manifest.execution_scope == "whole_frame_local"
    assert artifact.manifest.format_version == 2

    (tmp_path / "fitted.pkl").write_bytes(pickle.dumps((artifact, result)))
    script = """
import pickle
import sys
from pathlib import Path
import numpy as np
import pandas as pd
import polars as pl
from skyulf.inference.fitted_pipeline import FittedPipelineArtifact, load_pipeline
from skyulf.inference.pipeline_scoring import PipelinePrediction, score_pipeline
root = Path(sys.argv[1])
artifact, result = pickle.loads((root / 'fitted.pkl').read_bytes())
assert type(artifact) is FittedPipelineArtifact
assert type(result) is PipelinePrediction
restored = load_pipeline(root / 'fitted')
assert restored.manifest == artifact.manifest
assert restored.manifest.pipeline_sha256 == sys.argv[2]
query = pd.DataFrame({'x': [5.0, 6.0]})
for frame in (query, pl.from_pandas(query)):
    np.testing.assert_allclose(score_pipeline(frame, artifact)['prediction'], [11.0, 13.0])
    np.testing.assert_allclose(score_pipeline(frame, restored)['prediction'], [11.0, 13.0])
"""
    environment = dict(os.environ)
    environment["PYTHONPATH"] = str(Path(fitted.__file__).resolve().parents[2])
    process = subprocess.run(
        [sys.executable, "-c", script, str(tmp_path), artifact.manifest.pipeline_sha256],
        env=environment,
        capture_output=True,
        text=True,
        check=False,
        timeout=60,
    )
    assert process.returncode == 0, process.stderr


def test_canonical_mlflow_logger_validates_artifact_before_upload(monkeypatch):
    """Invalid pipeline artifacts fail before any tracking client is created."""
    pytest.importorskip("mlflow")
    assert importlib.util.find_spec("skyulf.integrations.mlflow.models.pipeline_model") is not None
    canonical = importlib.import_module("skyulf.integrations.mlflow.models.pipeline_model")

    def reject_artifact(path):
        """Stop before any external operation before any tracking operation."""
        raise ValueError("injected artifact rejection")

    monkeypatch.setattr(canonical, "load_pipeline", reject_artifact)
    with pytest.raises(ValueError, match="injected artifact rejection"):
        canonical.log_pipeline_model("fitted", run_id="run", artifact_path="model")


def test_registry_and_comparison_functions_have_canonical_owners():
    """Registry loading and comparison resolve from the documented modules."""
    from skyulf.integrations.mlflow.lifecycle import validation
    from skyulf.integrations.mlflow.registration import registry

    for name, old_name in [
        ("load_registered_pipeline", "load_registered_local_pipeline"),
        ("load_run_pipeline", "load_run_local_pipeline"),
    ]:
        assert hasattr(registry, name)
        assert not hasattr(registry, old_name)
        assert getattr(registry, name).__module__ == registry.__name__
    assert hasattr(validation, "compare_registered_pipeline_models")
    assert not hasattr(validation, "compare_registered_local_models")
    assert validation.compare_registered_pipeline_models.__module__ == validation.__name__


@pytest.mark.parametrize("engine", ["pandas", "polars"])
def test_mlflow_pipeline_package_replays_in_fresh_process(engine, tmp_path):
    """Packaged pandas and Polars fits load through canonical model paths after restart."""
    mlflow = pytest.importorskip("mlflow")
    from skyulf.inference import fitted_pipeline
    from skyulf.integrations.mlflow.models.pipeline_model import pipeline_model_save_options

    frame = pd.DataFrame({"x": [1.0, 2.0, 3.0, 4.0], "target": [3.0, 5.0, 7.0, 9.0]})
    native = pl.from_pandas(frame) if engine == "polars" else frame
    pipeline = SkyulfPipeline({"preprocessing": [], "modeling": {"type": "linear_regression"}})
    pipeline.fit(SplitDataset(train=native, test=native.head(0)), target_column="target")
    artifact_path = tmp_path / "fitted"
    fitted_pipeline.save_pipeline(pipeline, artifact_path)
    artifact = fitted_pipeline.load_pipeline(artifact_path)
    model_path = tmp_path / "model"
    mlflow.pyfunc.save_model(
        str(model_path),
        **pipeline_model_save_options(artifact, artifact_path, tmp_path),
    )
    script = """
import sys
import mlflow
import numpy as np
import pandas as pd
from skyulf.integrations.mlflow.models.pipeline_model import SkyulfPipelinePythonModel
model = mlflow.pyfunc.load_model(sys.argv[1])
assert type(model.unwrap_python_model()) is SkyulfPipelinePythonModel
query = pd.DataFrame({'x': [5.0, 6.0]})
np.testing.assert_allclose(model.predict(query)['prediction'], [11.0, 13.0])
"""
    environment = dict(os.environ)
    environment["PYTHONPATH"] = str(Path(fitted_pipeline.__file__).resolve().parents[2])
    process = subprocess.run(
        [sys.executable, "-c", script, str(model_path)],
        env=environment,
        capture_output=True,
        text=True,
        check=False,
        timeout=60,
    )
    assert process.returncode == 0, process.stderr
