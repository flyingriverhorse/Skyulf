"""Local artifact identities include the saved classification decision policy."""

import hashlib
import json
import pickle
from pathlib import Path
from types import SimpleNamespace
from typing import Any

import numpy as np
import pandas as pd
import polars as pl
import pytest
from sklearn.metrics import accuracy_score

from skyulf.data.dataset import SplitDataset
from skyulf.inference.fitted_pipeline import load_pipeline, predict_pipeline, save_pipeline
from skyulf.integrations.mlflow.registration.registry import load_local_package
from skyulf.pipeline import SkyulfPipeline


def _fit(engine: str, *, tune: bool = True, regression: bool = False):
    """Fit a model whose explicit threshold policy changes an observable decision."""
    frame = pd.DataFrame(
        {
            "x": [-3.0, -2.0, -1.0, 1.0, 2.0, 3.0],
            "target": [-6.0, -4.0, -2.0, 2.0, 4.0, 6.0]
            if regression
            else ["no", "no", "no", "yes", "yes", "yes"],
        }
    )
    native = pl.from_pandas(frame) if engine == "polars" else frame
    pipeline = SkyulfPipeline(
        {
            "preprocessing": [],
            "modeling": {"type": "linear_regression" if regression else "logistic_regression"},
        }
    )
    pipeline.fit(SplitDataset(train=native, test=native.head(0)), target_column="target")
    query = pd.DataFrame({"x": [-2.5, -0.25, 0.25, 2.5]})
    if tune:
        pipeline.optimize_thresholds(
            pl.from_pandas(query) if engine == "polars" else query,
            np.array(["no", "yes", "yes", "yes"]),
            accuracy_score,
            grid_points=5,
        )
    return pipeline, query


@pytest.fixture(params=["pandas", "polars"])
def fitted(request):
    """Exercise the same transport contract for both supported fit engines."""
    return _fit(request.param)


def _metadata(path: Path, **updates: Any) -> dict[str, Any]:
    """Change saved JSON while leaving the serialized pipeline bytes untouched."""
    file = path / "manifest.json"
    document = json.loads(file.read_text(encoding="utf-8"))
    document.update(updates)
    file.write_text(json.dumps(document), encoding="utf-8")
    return document


def _payload(path: Path, value: Any, *, version: int) -> None:
    """Write a self-consistent checksum so payload shape validation is actually reached."""
    payload = pickle.dumps(value, protocol=pickle.HIGHEST_PROTOCOL)
    (path / "pipeline.pkl").write_bytes(payload)
    _metadata(path, format_version=version, pipeline_sha256=hashlib.sha256(payload).hexdigest())


def _package(path: Path, digest: str, engine: str) -> SimpleNamespace:
    """Represent MLflow's parsed metadata for its actual shared package loader."""
    return SimpleNamespace(
        metadata={
            "skyulf_artifact_kind": "local_pipeline",
            "skyulf_execution_scope": "whole_frame_local",
            "local_pipeline_digest": digest,
            "skyulf_fitted_engine": engine,
        },
        flavors={"python_function": {"artifacts": {"local_pipeline": {"path": path.name}}}},
    )


def test_export_policy_changes_digest_without_mutating_fitted_pipeline(fitted, tmp_path):
    """Different decisions require distinct pinned identities without altering the trained model."""
    pipeline, query = fitted
    before = pickle.dumps(pipeline, protocol=pickle.HIGHEST_PROTOCOL)
    results, digests = [], []
    for policy in (False, True):
        path = tmp_path / str(policy)
        save_pipeline(pipeline, path, use_tuned_thresholds=policy)
        artifact = load_pipeline(path)
        assert artifact.manifest.format_version == 2
        assert artifact.manifest.use_tuned_thresholds is policy
        assert (
            artifact.manifest.pipeline_sha256
            == hashlib.sha256((path / "pipeline.pkl").read_bytes()).hexdigest()
        )
        digests.append(artifact.manifest.pipeline_sha256)
        results.append(predict_pipeline(query, artifact))
    assert pickle.dumps(pipeline, protocol=pickle.HIGHEST_PROTOCOL) == before
    assert results[0]["prediction"].to_list() == ["no", "no", "yes", "yes"]
    assert results[1]["prediction"].to_list() == ["no", "yes", "yes", "yes"]
    pd.testing.assert_frame_equal(
        results[0].filter(like="probability"), results[1].filter(like="probability")
    )
    assert digests[0] != digests[1]


@pytest.mark.parametrize("policy", [False, True])
@pytest.mark.parametrize("loader", ["local", "registry"])
def test_manifest_policy_flip_is_rejected_with_unchanged_payload(fitted, tmp_path, policy, loader):
    """Neither a local load nor pinned registry loading may accept metadata-only policy changes."""
    pipeline, _ = fitted
    path = tmp_path / "artifact"
    save_pipeline(pipeline, path, use_tuned_thresholds=policy)
    artifact = load_pipeline(path)
    digest = artifact.manifest.pipeline_sha256
    model = _package(path, digest, artifact.manifest.fitted_engine)
    _metadata(path, use_tuned_thresholds=not policy)
    with pytest.raises(ValueError, match="manifest.*policy|policy.*manifest"):
        if loader == "registry":
            load_local_package(tmp_path, model, digest)
        else:
            load_pipeline(path)


def test_registry_pin_rejects_a_valid_artifact_with_the_other_policy(fitted, tmp_path):
    """A valid package with different decision semantics must not satisfy an existing pin."""
    pipeline, _ = fitted
    original, replacement = tmp_path / "original", tmp_path / "replacement"
    save_pipeline(pipeline, original, use_tuned_thresholds=False)
    save_pipeline(pipeline, replacement, use_tuned_thresholds=True)
    artifact = load_pipeline(original)
    digest = artifact.manifest.pipeline_sha256
    model = _package(replacement, digest, artifact.manifest.fitted_engine)
    with pytest.raises(ValueError, match="identity differs"):
        load_local_package(tmp_path, model, digest)


@pytest.mark.parametrize("flag", [0, 1, "true", None, np.bool_(True)])
def test_payload_policy_requires_a_real_boolean(fitted, tmp_path, flag):
    """Checksum-valid payloads still require an unambiguous boolean decision policy."""
    pipeline, _ = fitted
    path = tmp_path / "artifact"
    save_pipeline(pipeline, path)
    _payload(path, {"pipeline": pipeline, "use_tuned_thresholds": flag}, version=2)
    with pytest.raises(ValueError, match="payload.*policy|policy.*boolean"):
        load_pipeline(path)


@pytest.mark.parametrize("flag", [0, 1, "true", None, np.bool_(True)])
def test_save_rejects_nonboolean_policy_before_creating_an_artifact(fitted, tmp_path, flag):
    """Invalid export options must not produce an artifact that fails only when loaded."""
    pipeline, _ = fitted
    path = tmp_path / "artifact"
    before = pickle.dumps(pipeline, protocol=pickle.HIGHEST_PROTOCOL)
    with pytest.raises(ValueError, match="use_tuned_thresholds"):
        save_pipeline(pipeline, path, use_tuned_thresholds=flag)
    assert not path.exists()
    assert pickle.dumps(pipeline, protocol=pickle.HIGHEST_PROTOCOL) == before


@pytest.mark.parametrize(
    "shape", ["bare", "missing_policy", "missing_pipeline", "extra", "wrong_pipeline"]
)
def test_version_two_requires_the_complete_payload_envelope(fitted, tmp_path, shape):
    """The new version cannot silently interpret bare or incomplete payloads as legacy data."""
    pipeline, _ = fitted
    variants = {
        "bare": pipeline,
        "missing_policy": {"pipeline": pipeline},
        "missing_pipeline": {"use_tuned_thresholds": False},
        "extra": {"pipeline": pipeline, "use_tuned_thresholds": False, "extra": 1},
        "wrong_pipeline": {"pipeline": {}, "use_tuned_thresholds": False},
    }
    path = tmp_path / "artifact"
    save_pipeline(pipeline, path)
    _payload(path, variants[shape], version=2)
    with pytest.raises(ValueError, match="payload"):
        load_pipeline(path)


def test_version_two_payload_cannot_downgrade_through_legacy_manifest(fitted, tmp_path):
    """Changing only the version must not bypass the policy-bearing payload contract."""
    pipeline, _ = fitted
    path = tmp_path / "artifact"
    save_pipeline(pipeline, path)
    _metadata(path, format_version=1)
    with pytest.raises(ValueError, match="payload|version 1"):
        load_pipeline(path)


@pytest.mark.parametrize("policy", [False, True])
def test_legacy_threshold_pipeline_requires_explicit_reexport(fitted, tmp_path, policy):
    """Legacy threshold-bearing bytes cannot prove either saved decision policy."""
    pipeline, _ = fitted
    path = tmp_path / "artifact"
    save_pipeline(pipeline, path, use_tuned_thresholds=policy)
    _payload(path, pipeline, version=1)
    with pytest.raises(ValueError, match="version 1.*re-export"):
        load_pipeline(path)


@pytest.mark.parametrize("engine", ["pandas", "polars"])
@pytest.mark.parametrize("regression", [False, True])
def test_legacy_pipeline_without_thresholds_keeps_identity_and_predictions(
    tmp_path, engine, regression
):
    """Unambiguous legacy packages stay readable without rewriting their pinned digest."""
    pipeline, query = _fit(engine, tune=False, regression=regression)
    path = tmp_path / "artifact"
    save_pipeline(pipeline, path)
    expected = predict_pipeline(query, load_pipeline(path))
    _payload(path, pipeline, version=1)
    document = json.loads((path / "manifest.json").read_text(encoding="utf-8"))
    document.pop("classification_probabilities")
    document.pop("use_tuned_thresholds")
    (path / "manifest.json").write_text(json.dumps(document), encoding="utf-8")
    artifact = load_pipeline(path)
    assert artifact.manifest.format_version == 1
    assert artifact.manifest.pipeline_sha256 == document["pipeline_sha256"]
    pd.testing.assert_frame_equal(predict_pipeline(query, artifact), expected)


@pytest.mark.parametrize("version", [0, 3, True, "2", 2.0, None])
def test_invalid_manifest_version_is_rejected_before_unpickling(tmp_path, monkeypatch, version):
    """Version parsing must not weaken the validation boundary before executable pickle loads."""
    from skyulf.inference import fitted_pipeline

    pipeline, _ = _fit("pandas", tune=False)
    path = tmp_path / "artifact"
    save_pipeline(pipeline, path)
    _metadata(path, format_version=version)

    def reject_unpickle(payload):
        """Fail the test if invalid metadata reaches pickle decoding."""
        pytest.fail("Invalid manifest version reached pickle.loads")

    monkeypatch.setattr(fitted_pipeline.pickle, "loads", reject_unpickle)
    with pytest.raises(ValueError, match="format_version"):
        load_pipeline(path)
