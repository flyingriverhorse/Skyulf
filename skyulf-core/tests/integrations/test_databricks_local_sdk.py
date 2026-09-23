"""Preflight for the first bounded pandas/Polars Databricks workflow."""

from __future__ import annotations

from dataclasses import replace

import numpy as np
import pandas as pd
import polars as pl
import pytest
from pydantic import ValidationError

from skyulf.data.dataset import SplitDataset
from skyulf.inference.bundle import build_bundle, save_bundle
from skyulf.inference.local_pipeline import load_local_pipeline, save_local_pipeline
from skyulf.integrations.databricks.local_sdk import (
    InputSource,
    LocalWorkflowConfig,
    ModelSelection,
    OutputSink,
    PreflightError,
    preflight_local,
    prepare_local_workflow,
)
from skyulf.pipeline import SkyulfPipeline


def _artifact(tmp_path, *, steps=None, model="linear_regression"):
    """Create a real fitted artifact so metadata and predictor cannot diverge."""
    frame = pd.DataFrame({"x": np.arange(8, dtype="float64"), "target": np.arange(8) * 2.0})
    pipeline = SkyulfPipeline({"preprocessing": steps or [], "modeling": {"type": model}})
    pipeline.fit(SplitDataset(train=frame, test=frame.head(0)), target_column="target")
    path = tmp_path / "local"
    save_local_pipeline(pipeline, path)
    return path, load_local_pipeline(path)


def _config(path, **changes):
    """Keep each rejection test focused on one workflow decision."""
    values = {
        "runtime": "databricks",
        "engine": "pandas",
        "source": InputSource(kind="caller_frame"),
        "model": ModelSelection(kind="local_pipeline", path=str(path)),
        "sink": OutputSink(kind="return_frame"),
    }
    values.update(changes)
    return LocalWorkflowConfig(**values)


def test_config_load_is_immutable_and_offline(tmp_path, monkeypatch) -> None:
    """Reading SDK config must never resolve aliases or mutate a workspace."""
    from skyulf.integrations.mlflow import registry

    monkeypatch.setattr(registry, "resolve_model", lambda *a, **k: pytest.fail("remote call"))
    config = _config(tmp_path / "local")
    loaded = LocalWorkflowConfig.model_validate_json(config.model_dump_json())
    assert loaded == config
    field_name = "engine"
    with pytest.raises(ValidationError):
        setattr(loaded, field_name, "polars")


def test_preflight_selects_local_predictor_and_metadata(tmp_path) -> None:
    """The selected engine and feature order must come from the fitted artifact."""
    path, artifact = _artifact(tmp_path)
    config = _config(path)
    result = preflight_local(config, artifact=artifact)
    prepared = prepare_local_workflow(config)
    actual = prepared.predict(pd.DataFrame({"x": [2.0, 4.0]}))
    assert result.ready and prepared.preflight.ready
    assert result.feature_order == ("x",)
    assert result.model_digest == artifact.manifest.pipeline_sha256
    assert result.output_columns == ("prediction",)
    np.testing.assert_allclose(actual["prediction"], [4.0, 8.0])


@pytest.mark.parametrize(
    ("changes", "code"),
    [
        ({"engine": "polars"}, "engine_mismatch"),
        ({"sink": OutputSink(kind="uc_delta", table="c.s.predictions")}, "sink_unavailable"),
        ({"source": InputSource(kind="uc_table", table="c.s.source")}, "source_unbounded"),
        ({"runtime": "spark"}, "runtime_unsupported"),
    ],
)
def test_incompatible_config_fails_before_loading(tmp_path, monkeypatch, changes, code) -> None:
    """Bad engine, source, sink or runtime must fail before artifact or job access."""
    path, artifact = _artifact(tmp_path)
    config = _config(path, **changes)
    result = preflight_local(config, artifact=artifact)
    assert code in {issue.code for issue in result.issues}
    with pytest.raises(PreflightError):
        prepare_local_workflow(config)


def test_changed_in_memory_pipeline_contract_is_reported(tmp_path) -> None:
    """A supplied artifact cannot claim a model or FE config different from its payload."""
    path, artifact = _artifact(tmp_path)
    pipeline = artifact.pipeline
    pipeline.preprocessing_steps = [{"name": "custom", "transformer": "SomeFutureNode"}]
    changed = replace(artifact, pipeline=pipeline)
    result = preflight_local(_config(path), artifact=changed)
    assert "node_contract_mismatch" in {issue.code for issue in result.issues}
    manifest = artifact.manifest.model_copy(update={"model_class": "other.FutureModel"})
    result = preflight_local(_config(path), artifact=replace(artifact, manifest=manifest))
    assert "model_contract_mismatch" in {issue.code for issue in result.issues}


def test_registry_requires_explicit_remote_preparation(tmp_path, monkeypatch) -> None:
    """Local preflight must describe missing registry evidence without contacting MLflow."""
    from skyulf.integrations.mlflow import registry

    monkeypatch.setattr(registry, "resolve_model", lambda *a, **k: pytest.fail("remote call"))
    config = _config(
        tmp_path / "unused",
        model=ModelSelection(kind="local_pipeline", name="catalog.schema.model", alias="champion"),
    )
    result = preflight_local(config)
    assert not result.ready
    assert result.remote_checked is False
    assert "model_unresolved" in {issue.code for issue in result.issues}


def test_portable_bundle_kind_must_match_artifact(tmp_path) -> None:
    """Preflight rejects sending a local pickle package down the portable route."""
    path, artifact = _artifact(tmp_path)
    config = _config(path, model=ModelSelection(kind="portable_bundle", path=str(path)))
    result = preflight_local(config, artifact=artifact)
    assert "artifact_kind_mismatch" in {issue.code for issue in result.issues}


def test_portable_bundle_uses_its_saved_feature_and_output_schema(tmp_path) -> None:
    """The portable route retains bundle order and schema without changing engines."""
    frame = pd.DataFrame({"x": np.arange(8, dtype="float64"), "target": np.arange(8) * 2.0})
    pipeline = SkyulfPipeline({"preprocessing": [], "modeling": {"type": "linear_regression"}})
    pipeline.fit(SplitDataset(train=frame, test=frame.head(0)), target_column="target")
    path = tmp_path / "bundle"
    save_bundle(build_bundle(pipeline, input_stage="raw", feature_order=("x",)), path)
    config = _config(path, model=ModelSelection(kind="portable_bundle", path=str(path)))
    prepared = prepare_local_workflow(config)
    assert prepared.preflight.feature_order == ("x",)
    assert prepared.preflight.output_schema[0].dtype == "float64"
    np.testing.assert_allclose(prepared.predict(pd.DataFrame({"x": [2.0]}))["prediction"], [4.0])


def test_caller_frame_size_is_checked_before_prediction(tmp_path) -> None:
    """A caller cannot bypass the explicit driver-memory budget at score time."""
    path, _ = _artifact(tmp_path)
    config = _config(path, source=InputSource(kind="caller_frame", max_rows=1, max_bytes=1024))
    prepared = prepare_local_workflow(config)
    with pytest.raises(ValueError, match="max_rows"):
        prepared.predict(pd.DataFrame({"x": [1.0, 2.0]}))
    small = _config(path, source=InputSource(kind="caller_frame", max_rows=10, max_bytes=1))
    with pytest.raises(ValueError, match="max_bytes"):
        prepare_local_workflow(small).predict(pd.DataFrame({"x": [1.0]}))


def test_optional_probe_reports_real_prediction_incompatibility(tmp_path) -> None:
    """A sample checks fitted FE/model behavior without a fabricated SDK allowlist."""
    path, artifact = _artifact(tmp_path)
    config = _config(path)
    bad = pd.DataFrame({"other": [1.0]})
    result = preflight_local(config, artifact=artifact, probe_frame=bad)
    assert "prediction_probe_failed" in {issue.code for issue in result.local_issues}
    with pytest.raises(PreflightError) as failure:
        prepare_local_workflow(config, probe_frame=bad)
    assert failure.value.result.issues[0].code == "prediction_probe_failed"


def test_registry_alias_is_resolved_once_and_passed_as_a_version(tmp_path, monkeypatch) -> None:
    """A moving alias cannot change the artifact selected inside one preparation."""
    from skyulf.integrations.mlflow import registry

    _, artifact = _artifact(tmp_path)
    calls = []
    pinned = registry.ResolvedModel(
        name="catalog.schema.model",
        version="7",
        model_uri="models:/catalog.schema.model/7",
        signature=None,
        digest=artifact.manifest.pipeline_sha256,
    )

    def resolve(*args, **kwargs):
        """Return the single concrete version selected for this test job."""
        calls.append(("resolve", args, kwargs))
        return pinned

    def load(resolved, **kwargs):
        """Ensure loading follows the prior resolution rather than the alias."""
        calls.append(("load", resolved, kwargs))
        return artifact

    monkeypatch.setattr(registry, "resolve_model", resolve)
    monkeypatch.setattr(registry, "load_registered_local_pipeline", load)
    config = _config(
        tmp_path / "unused",
        model=ModelSelection(kind="local_pipeline", name=pinned.name, alias="champion"),
    )
    prepared = prepare_local_workflow(config)
    assert [call[0] for call in calls] == ["resolve", "load"]
    assert calls[1][1] is pinned
    assert prepared.preflight.model_version == "7"


def test_remote_read_failure_returns_actionable_preflight_issue(tmp_path, monkeypatch) -> None:
    """A registry denial must be distinguishable from a local model mismatch."""
    from skyulf.integrations.mlflow import registry

    def denied(*args, **kwargs):
        """Simulate read-only MLflow access being rejected by the registry."""
        raise registry.RegistryAccessError("access denied")

    monkeypatch.setattr(registry, "resolve_model", denied)
    config = _config(
        tmp_path / "unused",
        model=ModelSelection(kind="local_pipeline", name="catalog.schema.model", version="7"),
    )
    with pytest.raises(PreflightError) as failure:
        prepare_local_workflow(config)
    result = failure.value.result
    assert result.remote_checked
    assert result.issues[0].code == "registry_access_denied"
    assert result.remote_issues == result.issues
    assert result.local_issues == ()
    assert result.issues[0].fix


@pytest.mark.parametrize(
    "uri",
    [
        "https://name:secret@example.com/mlflow",
        "https://example.com/mlflow?access_token=secret",
    ],
)
def test_store_uri_rejects_embedded_secret(uri) -> None:
    """Serialized workflow config must not carry a store password or token."""
    from pydantic import ValidationError

    with pytest.raises(ValidationError, match="credentials"):
        ModelSelection(
            kind="local_pipeline",
            name="model",
            version="1",
            tracking_uri=uri,
        )


@pytest.mark.parametrize("engine", ["pandas", "polars"])
def test_impute_and_scale_replay_is_eligible_for_local_sdk(engine, tmp_path) -> None:
    """Common numeric FE must replay from a saved package on either local engine."""
    train = pd.DataFrame(
        {"x": [1.0, np.nan, 3.0, 4.0, 5.0, 6.0], "target": [2.0, 4.0, 6.0, 8.0, 10.0, 12.0]}
    )
    steps = [
        {"name": "fill", "transformer": "SimpleImputer", "params": {"strategy": "mean"}},
        {"name": "scale", "transformer": "StandardScaler", "params": {"columns": ["x"]}},
    ]
    pipeline = SkyulfPipeline({"preprocessing": steps, "modeling": {"type": "linear_regression"}})
    native = pl.from_pandas(train) if engine == "polars" else train
    pipeline.fit(SplitDataset(train=native, test=native.head(0)), target_column="target")
    path = tmp_path / "pipeline"
    save_local_pipeline(pipeline, path)
    query = pd.DataFrame({"x": [2.0, np.nan]})
    expected = pipeline.predict(pl.from_pandas(query) if engine == "polars" else query)
    prepared = prepare_local_workflow(_config(path, engine=engine))
    assert prepared.preflight.ready
    np.testing.assert_allclose(prepared.predict(query)["prediction"], expected)


@pytest.mark.parametrize("engine", ["pandas", "polars"])
def test_existing_library_node_and_model_need_no_sdk_allowlist(engine, tmp_path) -> None:
    """A fitted library model and FE node remain usable without SDK registration."""
    train = pd.DataFrame({"x": np.arange(12, dtype="float64"), "target": np.arange(12) ** 2.0})
    steps = [{"name": "range", "transformer": "MinMaxScaler", "params": {"columns": ["x"]}}]
    pipeline = SkyulfPipeline(
        {"preprocessing": steps, "modeling": {"type": "random_forest_regressor"}}
    )
    native = pl.from_pandas(train) if engine == "polars" else train
    pipeline.fit(SplitDataset(train=native, test=native.head(0)), target_column="target")
    path = tmp_path / "pipeline"
    save_local_pipeline(pipeline, path)
    query = pd.DataFrame({"x": [2.0, 9.0]})
    expected = pipeline.predict(pl.from_pandas(query) if engine == "polars" else query)
    prepared = prepare_local_workflow(_config(path, engine=engine))
    assert prepared.preflight.ready
    np.testing.assert_allclose(prepared.predict(query)["prediction"], expected)
