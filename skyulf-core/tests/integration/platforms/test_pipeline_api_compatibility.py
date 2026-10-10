"""Purpose-based inference names preserve released imports and persisted models."""

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
def test_canonical_modules_preserve_legacy_identity(canonical, legacy, symbols):
    """Legacy imports must patch the same globals and restore released pickle classes."""
    if ".mlflow." in canonical:
        pytest.importorskip("mlflow")
    assert importlib.util.find_spec(canonical) is not None
    implementation = importlib.import_module(canonical)
    previous = importlib.import_module(legacy)
    assert implementation is previous
    for name, old_name in symbols:
        value = getattr(implementation, name)
        assert value is getattr(previous, old_name)
        assert value.__module__ == canonical
        assert value.__name__ == name
        restored = pickle.loads(f"c{legacy}\n{old_name}\n.".encode())
        assert restored is value


@pytest.mark.parametrize("engine", ["pandas", "polars"])
def test_pipeline_roundtrip_and_legacy_object_reload_in_fresh_process(engine, tmp_path):
    """Renaming a fitted artifact cannot change predictions, digests or old pickle loading."""
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

    # Protocol 0 exposes GLOBAL records so this fixture records released paths,
    # independently of the class's new import identity in the current process.
    payload = pickle.dumps((artifact, result), protocol=0)
    for current, previous in [
        ("fitted_pipeline\nFittedPipelineArtifact", "local_pipeline\nLocalPipelineArtifact"),
        ("fitted_pipeline\nFittedPipelineManifest", "local_pipeline\nLocalPipelineManifest"),
        ("pipeline_scoring\nPipelinePrediction", "local_scoring\nLocalPrediction"),
    ]:
        assert current.encode() in payload
        payload = payload.replace(current.encode(), previous.encode())
    (tmp_path / "legacy.pkl").write_bytes(payload)
    script = """
import pickle
import sys
from pathlib import Path
import numpy as np
import pandas as pd
import polars as pl
from skyulf.inference.fitted_pipeline import FittedPipelineArtifact, load_pipeline
from skyulf.inference.pipeline_scoring import PipelinePrediction, score_pipeline
from skyulf.inference.local_scoring import score_local_pipeline
root = Path(sys.argv[1])
artifact, result = pickle.loads((root / 'legacy.pkl').read_bytes())
assert type(artifact) is FittedPipelineArtifact
assert type(result) is PipelinePrediction
restored = load_pipeline(root / 'fitted')
assert restored.manifest == artifact.manifest
assert restored.manifest.pipeline_sha256 == sys.argv[2]
query = pd.DataFrame({'x': [5.0, 6.0]})
for frame in (query, pl.from_pandas(query)):
    for predict in (score_pipeline, score_local_pipeline):
        np.testing.assert_allclose(predict(frame, artifact)['prediction'], [11.0, 13.0])
        np.testing.assert_allclose(predict(frame, restored)['prediction'], [11.0, 13.0])
"""
    environment = dict(os.environ)
    environment["PYTHONPATH"] = str(Path(__file__).resolve().parents[3])
    process = subprocess.run(
        [sys.executable, "-c", script, str(tmp_path), artifact.manifest.pipeline_sha256],
        env=environment,
        capture_output=True,
        text=True,
        check=False,
        timeout=60,
    )
    assert process.returncode == 0, process.stderr


def test_legacy_mlflow_dependency_patch_reaches_canonical_logger(monkeypatch):
    """Existing dependency injection through old module paths must still affect logging."""
    pytest.importorskip("mlflow")
    assert importlib.util.find_spec("skyulf.integrations.mlflow.models.pipeline_model") is not None
    canonical = importlib.import_module("skyulf.integrations.mlflow.models.pipeline_model")
    legacy = importlib.import_module("skyulf.integrations.mlflow.local_model")

    def reject_artifact(path):
        """Stop before any external operation through the released dependency seam."""
        raise ValueError("injected artifact rejection")

    monkeypatch.setattr(legacy, "load_local_pipeline", reject_artifact)
    with pytest.raises(ValueError, match="injected artifact rejection"):
        canonical.log_pipeline_model("fitted", run_id="run", artifact_path="model")


def test_registry_and_comparison_aliases_preserve_public_calls():
    """Registry selection and comparison retain the released callable identities."""
    from skyulf.integrations.mlflow.lifecycle import validation
    from skyulf.integrations.mlflow.registration import registry

    for name, old_name in [
        ("load_registered_pipeline", "load_registered_local_pipeline"),
        ("load_run_pipeline", "load_run_local_pipeline"),
    ]:
        assert hasattr(registry, name)
        assert getattr(registry, name) is getattr(registry, old_name)
    assert hasattr(validation, "compare_registered_pipeline_models")
    assert (
        validation.compare_registered_pipeline_models is validation.compare_registered_local_models
    )
