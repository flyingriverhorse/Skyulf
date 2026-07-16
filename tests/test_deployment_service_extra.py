"""Focused unit tests for backend.ml_pipeline.deployment.service.DeploymentService.

These target the pure/static helper methods extracted during the T4 complexity
cleanup, exercising branches not covered by the higher-level flow tests in
test_deployment.py (deploy -> predict happy path).
"""

from datetime import UTC, datetime, timezone
from types import SimpleNamespace
from unittest.mock import AsyncMock, MagicMock, patch

import numpy as np
import pandas as pd
import pytest
import pytest_asyncio
from sqlalchemy.ext.asyncio import AsyncSession, create_async_engine
from sqlalchemy.orm import sessionmaker

from backend.database.models import Base, Deployment
from backend.ml_pipeline.deployment.service import DeploymentService, _maybe_decode_predictions

TEST_DATABASE_URL = "sqlite+aiosqlite:///:memory:"


@pytest_asyncio.fixture
async def async_session():
    """Provides an isolated in-memory async SQLite session per test."""
    engine = create_async_engine(TEST_DATABASE_URL, echo=False)
    async with engine.begin() as conn:
        await conn.run_sync(Base.metadata.create_all)

    async_session_maker = sessionmaker(engine, class_=AsyncSession, expire_on_commit=False)
    async with async_session_maker() as session:
        yield session

    await engine.dispose()


class _EncoderEngineer:
    """Minimal feature engineer exposing a target LabelEncoder like _IdentityEngineer."""

    def __init__(self, target_label_encoder):
        self.fitted_steps = [
            {
                "name": "label_encode_target",
                "type": "LabelEncoder",
                "applier": None,
                "artifact": {"encoders": {"__target__": target_label_encoder}, "columns": []},
            }
        ]


class _NoEncoderEngineer:
    """Feature engineer with no fitted_steps at all -> no target encoder found."""

    fitted_steps = None


# ---------------------------------------------------------------------------
# _maybe_decode_predictions
# ---------------------------------------------------------------------------


def test_maybe_decode_predictions_no_encoder_returns_as_is():
    """No target encoder found -> predictions returned unchanged (line 35)."""
    preds = [0, 1]
    result = _maybe_decode_predictions(preds, _NoEncoderEngineer())
    assert result == preds


def test_maybe_decode_predictions_non_int_dtype_success():
    """Non int/uint/bool dtype (e.g. float) still decodes successfully (line 47)."""
    from sklearn.preprocessing import LabelEncoder

    le = LabelEncoder()
    le.fit(["cat", "dog"])
    engineer = _EncoderEngineer(le)

    preds = np.array([0.0, 1.0])
    result = _maybe_decode_predictions(preds, engineer)
    assert list(result) == ["cat", "dog"]


def test_maybe_decode_predictions_decode_failure_returns_original():
    """Non-numeric strings that can't cast to int raise -> caught, original returned (lines 48-50)."""
    from sklearn.preprocessing import LabelEncoder

    le = LabelEncoder()
    le.fit(["cat", "dog"])
    engineer = _EncoderEngineer(le)

    preds = np.array(["not_a_number", "also_not"])
    result = _maybe_decode_predictions(preds, engineer)
    # Falls back to original predictions unchanged
    assert list(result) == ["not_a_number", "also_not"]


# ---------------------------------------------------------------------------
# _validate_job_for_deployment
# ---------------------------------------------------------------------------


def test_validate_job_for_deployment_missing_job_raises():
    with pytest.raises(ValueError, match="not found"):
        DeploymentService._validate_job_for_deployment(None, "job-x")


def test_validate_job_for_deployment_not_completed_raises():
    db_job = SimpleNamespace(status="failed")
    with pytest.raises(ValueError, match="not completed successfully"):
        DeploymentService._validate_job_for_deployment(db_job, "job-x")


def test_validate_job_for_deployment_completed_ok():
    db_job = SimpleNamespace(status="completed")
    # Should not raise
    DeploymentService._validate_job_for_deployment(db_job, "job-x")


# ---------------------------------------------------------------------------
# _resolve_final_deployment_uri
# ---------------------------------------------------------------------------


def test_resolve_final_deployment_uri_falsy_returns_as_is():
    assert DeploymentService._resolve_final_deployment_uri(None, "job1", "pipe1") is None
    assert DeploymentService._resolve_final_deployment_uri("", "job1", "pipe1") == ""


def test_resolve_final_deployment_uri_s3_prefix_appends_job_id():
    result = DeploymentService._resolve_final_deployment_uri("s3://bucket/models/", "job1", "pipe1")
    assert result == "s3://bucket/models/job1.joblib"


def test_resolve_final_deployment_uri_s3_full_path_unchanged():
    result = DeploymentService._resolve_final_deployment_uri(
        "s3://bucket/models/job1.joblib", "job1", "pipe1"
    )
    assert result == "s3://bucket/models/job1.joblib"


def test_resolve_final_deployment_uri_directory_appends_job_id(tmp_path):
    result = DeploymentService._resolve_final_deployment_uri(str(tmp_path), "job1", "pipe1")
    assert result == str(tmp_path / "job1.joblib")


def test_resolve_final_deployment_uri_bare_id_builds_abstract_uri():
    result = DeploymentService._resolve_final_deployment_uri("node123", "job1", "pipe1")
    assert result == "pipe1/job1"


def test_resolve_final_deployment_uri_file_path_returned_as_is(tmp_path):
    file_path = str(tmp_path / "some_model.joblib")
    result = DeploymentService._resolve_final_deployment_uri(file_path, "job1", "pipe1")
    assert result == file_path


# ---------------------------------------------------------------------------
# deploy_model: fallback artifact_uri + unreachable None branch
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_deploy_model_falls_back_to_node_id_when_no_artifact_uri(async_session):
    db_job = SimpleNamespace(
        artifact_uri=None,
        pipeline_id="pipe1",
        node_id="node1",
        model_type="dummy",
        status="completed",
    )
    with patch(
        "backend.ml_pipeline.deployment.service.JobService.get_job_by_id",
        new=AsyncMock(return_value=db_job),
    ):
        deployment = await DeploymentService.deploy_model(async_session, "job1")
    assert deployment.artifact_uri == "pipe1/job1"


@pytest.mark.asyncio
async def test_deploy_model_none_db_job_after_validate_noop_raises():
    """Covers the defensive `if db_job is None` branch (unreachable in normal flow)."""
    with (
        patch(
            "backend.ml_pipeline.deployment.service.JobService.get_job_by_id",
            new=AsyncMock(return_value=None),
        ),
        patch.object(DeploymentService, "_validate_job_for_deployment", return_value=None),
        pytest.raises(ValueError, match="not found"),
    ):
        await DeploymentService.deploy_model(MagicMock(), "job1")


# ---------------------------------------------------------------------------
# deactivate_current_deployment
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_deactivate_current_deployment(async_session):
    # Should run without error against an empty deployments table
    await DeploymentService.deactivate_current_deployment(async_session)
    active = await DeploymentService.get_active_deployment(async_session)
    assert active is None


# ---------------------------------------------------------------------------
# _resolve_predict_store_and_key_s3
# ---------------------------------------------------------------------------


def test_resolve_predict_store_and_key_s3_with_joblib_suffix():
    store_uri, key = DeploymentService._resolve_predict_store_and_key_s3(
        "s3://bucket/prefix/job1.joblib"
    )
    assert store_uri == "s3://bucket/prefix"
    assert key == "job1"


def test_resolve_predict_store_and_key_s3_without_suffix():
    store_uri, key = DeploymentService._resolve_predict_store_and_key_s3(
        "s3://bucket/pipeline1/node1"
    )
    assert store_uri == "s3://bucket"
    assert key == "pipeline1/node1"


# ---------------------------------------------------------------------------
# _resolve_predict_store_and_key_local / _resolve_pipeline_node_path
# ---------------------------------------------------------------------------


def test_resolve_pipeline_node_path():
    store_uri, key = DeploymentService._resolve_pipeline_node_path("pipe1", "node1")
    assert store_uri.endswith("exports/models/pipe1")
    assert key == "node1"


def test_resolve_predict_store_and_key_local_absolute(tmp_path):
    abs_path = tmp_path / "model.joblib"
    store_uri, key = DeploymentService._resolve_predict_store_and_key_local(str(abs_path))
    assert store_uri == str(tmp_path)
    assert key == "model.joblib"


def test_resolve_predict_store_and_key_local_two_parts_nonexistent():
    store_uri, key = DeploymentService._resolve_predict_store_and_key_local("pipeA/nodeB")
    assert store_uri.endswith("exports/models/pipeA")
    assert key == "nodeB"


def test_resolve_predict_store_and_key_local_existing_relative_path(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    rel_dir = tmp_path / "somedir"
    rel_dir.mkdir()
    (rel_dir / "model.joblib").touch()
    store_uri, key = DeploymentService._resolve_predict_store_and_key_local("somedir/model.joblib")
    assert store_uri == "somedir"
    assert key == "model.joblib"


def test_resolve_predict_store_and_key_local_nonexistent_three_parts(tmp_path, monkeypatch):
    """A separator-containing URI that doesn't exist and splits into != 2 parts falls
    back to (parent, name) instead of the pipeline_id/node_id shortcut."""
    monkeypatch.chdir(tmp_path)
    store_uri, key = DeploymentService._resolve_predict_store_and_key_local(
        "pipeA/subdir/nodeB.joblib"
    )
    assert store_uri == "pipeA/subdir"
    assert key == "nodeB.joblib"


def test_resolve_predict_store_and_key_local_bare_two_parts():
    store_uri, key = DeploymentService._resolve_predict_store_and_key_local("pipeA/nodeB")
    assert key == "nodeB"


def test_resolve_predict_store_and_key_local_single_part_raises():
    with pytest.raises(ValueError, match="Invalid artifact URI format"):
        DeploymentService._resolve_predict_store_and_key_local("just_a_name")


def test_resolve_predict_store_and_key_dispatches_s3():
    store_uri, key = DeploymentService._resolve_predict_store_and_key("s3://bucket/a/b.joblib")
    assert store_uri == "s3://bucket/a"
    assert key == "b"


# ---------------------------------------------------------------------------
# _load_predict_artifact
# ---------------------------------------------------------------------------


def test_load_predict_artifact_wraps_failure_in_value_error():
    deployment = SimpleNamespace(artifact_uri="just_a_name")
    with pytest.raises(ValueError, match="Could not load model artifact"):
        DeploymentService._load_predict_artifact(deployment)


def test_load_predict_artifact_unwraps_tuple(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    models_dir = tmp_path / "exports" / "models" / "pipe1"
    models_dir.mkdir(parents=True)
    from backend.ml_pipeline.artifacts.local import LocalArtifactStore

    store = LocalArtifactStore(str(models_dir))
    store.save("node1", ("actual_model", {"meta": True}))

    deployment = SimpleNamespace(artifact_uri="pipe1/node1")
    artifact = DeploymentService._load_predict_artifact(deployment)
    assert artifact == "actual_model"


# ---------------------------------------------------------------------------
# _drop_target_and_dropped_columns
# ---------------------------------------------------------------------------


def test_drop_target_and_dropped_columns_drops_target():
    df = pd.DataFrame({"a": [1], "target": [2]})
    result = DeploymentService._drop_target_and_dropped_columns(df, "target", None)
    assert "target" not in result.columns


def test_drop_target_and_dropped_columns_str_dropped_cols():
    df = pd.DataFrame({"a": [1], "b": [2]})
    result = DeploymentService._drop_target_and_dropped_columns(df, None, "b")
    assert "b" not in result.columns
    assert "a" in result.columns


def test_drop_target_and_dropped_columns_list_dropped_cols_partial_existing():
    df = pd.DataFrame({"a": [1], "b": [2]})
    result = DeploymentService._drop_target_and_dropped_columns(df, None, ["b", "nonexistent"])
    assert list(result.columns) == ["a"]


def test_drop_target_and_dropped_columns_no_dropped_cols_present():
    df = pd.DataFrame({"a": [1]})
    result = DeploymentService._drop_target_and_dropped_columns(df, None, ["missing"])
    assert list(result.columns) == ["a"]


# ---------------------------------------------------------------------------
# _unwrap_tuple_estimator
# ---------------------------------------------------------------------------


def test_unwrap_tuple_estimator_tuple():
    assert DeploymentService._unwrap_tuple_estimator(("model", "meta")) == "model"


def test_unwrap_tuple_estimator_non_tuple():
    obj = object()
    assert DeploymentService._unwrap_tuple_estimator(obj) is obj


# ---------------------------------------------------------------------------
# _transform_bundled_features
# ---------------------------------------------------------------------------


def test_transform_bundled_features_failure_wraps_value_error():
    class _Failing:
        def transform(self, df):
            raise RuntimeError("boom")

    with pytest.raises(ValueError, match="Feature engineering failed"):
        DeploymentService._transform_bundled_features(_Failing(), pd.DataFrame({"a": [1]}))


# ---------------------------------------------------------------------------
# _predict_and_decode
# ---------------------------------------------------------------------------


def test_predict_and_decode_failure_wraps_value_error():
    class _Failing:
        def predict(self, X):
            raise RuntimeError("prediction boom")

    with pytest.raises(ValueError, match="Prediction failed"):
        DeploymentService._predict_and_decode(_Failing(), pd.DataFrame({"a": [1]}), None, None)


def test_predict_and_decode_plain_list_without_tolist():
    """Predictions without a .tolist() attribute go through the `list(predictions)` branch."""

    class _PlainListPredictor:
        def predict(self, X):
            return (1, 2, 3)  # tuple has no .tolist()

    result = DeploymentService._predict_and_decode(
        _PlainListPredictor(), pd.DataFrame({"a": [1]}), _NoEncoderEngineer(), None
    )
    assert result == [1, 2, 3]


# ---------------------------------------------------------------------------
# _predict_with_legacy_artifact
# ---------------------------------------------------------------------------


def test_predict_with_legacy_artifact_fills_missing_columns_and_reorders():
    class _LegacyModel:
        feature_names_in_ = np.array(["a", "b"])

        def predict(self, df):
            # Ensure columns were reordered/filled as expected before predict is called
            assert list(df.columns) == ["a", "b"]
            return np.array([df["a"].iloc[0] + df["b"].iloc[0]])

    df = pd.DataFrame({"b": [5], "a": [3]})
    # 'a' and 'b' both present but out of order -> should be reordered to match model_cols
    result = DeploymentService._predict_with_legacy_artifact(_LegacyModel(), df)
    assert result == [8]


def test_predict_with_legacy_artifact_fills_zero_for_missing_column():
    class _LegacyModel:
        feature_names_in_ = np.array(["a", "b", "c"])

        def predict(self, df):
            assert "c" in df.columns
            assert (df["c"] == 0).all()
            return df["a"] + df["b"] + df["c"]

    df = pd.DataFrame({"a": [1], "b": [2]})
    result = DeploymentService._predict_with_legacy_artifact(_LegacyModel(), df)
    assert list(result) == [3]


# ---------------------------------------------------------------------------
# predict(): no active deployment / unrecognized artifact
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_predict_raises_when_no_active_deployment(async_session):
    with pytest.raises(ValueError, match="No active model deployed"):
        await DeploymentService.predict(async_session, [{"a": 1}])


@pytest.mark.asyncio
async def test_predict_raises_for_unrecognized_artifact_format(async_session):
    with (
        patch.object(
            DeploymentService,
            "get_active_deployment",
            new=AsyncMock(return_value=SimpleNamespace(artifact_uri="whatever")),
        ),
        patch.object(DeploymentService, "_load_predict_artifact", return_value=object()),
        pytest.raises(ValueError, match="not a valid predictor"),
    ):
        await DeploymentService.predict(async_session, [{"a": 1}])


# ---------------------------------------------------------------------------
# _load_artifact_from_s3_for_details
# ---------------------------------------------------------------------------


def test_load_artifact_from_s3_for_details_builds_store_and_loads():
    fake_store = MagicMock()
    fake_store.load.return_value = "the_artifact"
    with patch(
        "backend.ml_pipeline.deployment.service.S3ArtifactStore", return_value=fake_store
    ) as mock_store_cls:
        result = DeploymentService._load_artifact_from_s3_for_details("s3://mybucket/path/key")
    assert result == "the_artifact"
    fake_store.load.assert_called_once_with("path/key")
    mock_store_cls.assert_called_once()
    _, kwargs = mock_store_cls.call_args
    assert kwargs["bucket_name"] == "mybucket"


# ---------------------------------------------------------------------------
# _resolve_local_base_and_key_for_details
# ---------------------------------------------------------------------------


def test_resolve_local_base_and_key_for_details_absolute(tmp_path):
    abs_path = tmp_path / "model.joblib"
    base, key = DeploymentService._resolve_local_base_and_key_for_details(str(abs_path))
    assert base == str(tmp_path)
    assert key == "model.joblib"


def test_resolve_local_base_and_key_for_details_two_part_relative():
    base, key = DeploymentService._resolve_local_base_and_key_for_details("pipeA/nodeB")
    assert base.endswith("exports/models/pipeA")
    assert key == "nodeB"


def test_resolve_local_base_and_key_for_details_existing_relative_path(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    rel_dir = tmp_path / "somedir"
    rel_dir.mkdir()
    (rel_dir / "model.joblib").touch()
    base, key = DeploymentService._resolve_local_base_and_key_for_details("somedir/model.joblib")
    assert base == "somedir"
    assert key == "model.joblib"


def test_resolve_local_base_and_key_for_details_bare_fallback(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    base, key = DeploymentService._resolve_local_base_and_key_for_details("bare_name")
    assert base == str(tmp_path)
    assert key == "bare_name"


# ---------------------------------------------------------------------------
# _load_artifact_for_details
# ---------------------------------------------------------------------------


def test_load_artifact_for_details_s3(tmp_path):
    fake_store = MagicMock()
    fake_store.load.return_value = "s3_artifact"
    with patch("backend.ml_pipeline.deployment.service.S3ArtifactStore", return_value=fake_store):
        result = DeploymentService._load_artifact_for_details("s3://bucket/key.joblib")
    assert result == "s3_artifact"


def test_load_artifact_for_details_local_not_exists(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    result = DeploymentService._load_artifact_for_details("nonexistent/missing_node")
    assert result is None


def test_load_artifact_for_details_local_exists(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    from backend.ml_pipeline.artifacts.local import LocalArtifactStore

    models_dir = tmp_path / "exports" / "models" / "pipeX"
    models_dir.mkdir(parents=True)
    store = LocalArtifactStore(str(models_dir))
    store.save("nodeX", "my_artifact")

    result = DeploymentService._load_artifact_for_details("pipeX/nodeX")
    assert result == "my_artifact"


# ---------------------------------------------------------------------------
# _extract_features_from_engineer
# ---------------------------------------------------------------------------


def test_extract_features_from_engineer_direct_attr():
    fe = SimpleNamespace(feature_names_in_=np.array(["a", "b"]))
    result = DeploymentService._extract_features_from_engineer(fe)
    assert list(result) == ["a", "b"]


def test_extract_features_from_engineer_from_first_step():
    transformer = SimpleNamespace(feature_names_in_=np.array(["x", "y"]))
    fe = SimpleNamespace(steps=[("step1", transformer)])
    result = DeploymentService._extract_features_from_engineer(fe)
    assert list(result) == ["x", "y"]


def test_extract_features_from_engineer_no_features_found():
    fe = SimpleNamespace(steps=[])
    result = DeploymentService._extract_features_from_engineer(fe)
    assert result == []


def test_extract_features_from_engineer_first_step_not_tuple():
    fe = SimpleNamespace(steps=["not_a_tuple"])
    result = DeploymentService._extract_features_from_engineer(fe)
    assert result == []


# ---------------------------------------------------------------------------
# _extract_features_from_bundled_artifact
# ---------------------------------------------------------------------------


def test_extract_features_from_bundled_artifact_uses_engineer():
    fe = SimpleNamespace(feature_names_in_=np.array(["a"]))
    artifact = {"feature_engineer": fe, "model": SimpleNamespace()}
    result = DeploymentService._extract_features_from_bundled_artifact(artifact)
    assert list(result) == ["a"]


def test_extract_features_from_bundled_artifact_falls_back_to_model():
    fe = SimpleNamespace()  # no feature_names_in_, no steps
    model = SimpleNamespace(feature_names_in_=np.array(["m1", "m2"]))
    artifact = {"feature_engineer": fe, "model": model}
    result = DeploymentService._extract_features_from_bundled_artifact(artifact)
    assert list(result) == ["m1", "m2"]


def test_extract_features_from_bundled_artifact_model_is_tuple():
    fe = SimpleNamespace()
    model = SimpleNamespace(feature_names_in_=np.array(["m1"]))
    artifact = {"feature_engineer": fe, "model": (model, "meta")}
    result = DeploymentService._extract_features_from_bundled_artifact(artifact)
    assert list(result) == ["m1"]


# ---------------------------------------------------------------------------
# _extract_input_features
# ---------------------------------------------------------------------------


def test_extract_input_features_bundled_dict():
    fe = SimpleNamespace(feature_names_in_=np.array(["a", "b"]))
    artifact = {"feature_engineer": fe}
    result = DeploymentService._extract_input_features(artifact)
    assert result == ["a", "b"]


def test_extract_input_features_direct_model():
    artifact = SimpleNamespace(feature_names_in_=np.array(["c", "d"]))
    result = DeploymentService._extract_input_features(artifact)
    assert result == ["c", "d"]


def test_extract_input_features_none_found():
    artifact = SimpleNamespace()
    result = DeploymentService._extract_input_features(artifact)
    assert result == []


# ---------------------------------------------------------------------------
# _extract_target_column_from_graph
# ---------------------------------------------------------------------------


def test_extract_target_column_from_graph_dict_node():
    graph = {"nodes": [{"params": {"target_column": "y"}}]}
    result = DeploymentService._extract_target_column_from_graph(graph)
    assert result == "y"


def test_extract_target_column_from_graph_object_node():
    node = SimpleNamespace(params={"target_column": "z"})
    graph = {"nodes": [node]}
    result = DeploymentService._extract_target_column_from_graph(graph)
    assert result == "z"


def test_extract_target_column_from_graph_not_found():
    graph = {"nodes": [{"params": {}}]}
    result = DeploymentService._extract_target_column_from_graph(graph)
    assert result is None


def test_extract_target_column_from_graph_no_nodes_key():
    result = DeploymentService._extract_target_column_from_graph({})
    assert result is None


# ---------------------------------------------------------------------------
# _build_input_schema_from_artifact
# ---------------------------------------------------------------------------


def test_build_input_schema_from_artifact_none_artifact(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    result = DeploymentService._build_input_schema_from_artifact("nonexistent/missing_node")
    assert result is None


def test_build_input_schema_from_artifact_tuple_unwrapped_no_features(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    from backend.ml_pipeline.artifacts.local import LocalArtifactStore

    models_dir = tmp_path / "exports" / "models" / "pipeY"
    models_dir.mkdir(parents=True)
    store = LocalArtifactStore(str(models_dir))
    store.save("nodeY", (SimpleNamespace(), "meta"))

    result = DeploymentService._build_input_schema_from_artifact("pipeY/nodeY")
    assert result is None


def test_build_input_schema_from_artifact_with_features(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    from backend.ml_pipeline.artifacts.local import LocalArtifactStore

    models_dir = tmp_path / "exports" / "models" / "pipeZ"
    models_dir.mkdir(parents=True)
    model = SimpleNamespace(feature_names_in_=np.array(["f1", "f2"]))
    store = LocalArtifactStore(str(models_dir))
    store.save("nodeZ", model)

    result = DeploymentService._build_input_schema_from_artifact("pipeZ/nodeZ")
    assert result == [{"name": "f1", "type": "unknown"}, {"name": "f2", "type": "unknown"}]


# ---------------------------------------------------------------------------
# _lookup_target_column
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_lookup_target_column_found(async_session):
    job = SimpleNamespace(graph={"nodes": [{"params": {"target_column": "label"}}]})
    with patch(
        "backend.ml_pipeline._execution.jobs.JobManager.get_job",
        new=AsyncMock(return_value=job),
    ):
        result = await DeploymentService._lookup_target_column(async_session, "job1")
    assert result == "label"


@pytest.mark.asyncio
async def test_lookup_target_column_no_job(async_session):
    with patch(
        "backend.ml_pipeline._execution.jobs.JobManager.get_job",
        new=AsyncMock(return_value=None),
    ):
        result = await DeploymentService._lookup_target_column(async_session, "job1")
    assert result is None


@pytest.mark.asyncio
async def test_lookup_target_column_job_no_graph(async_session):
    job = SimpleNamespace(graph=None)
    with patch(
        "backend.ml_pipeline._execution.jobs.JobManager.get_job",
        new=AsyncMock(return_value=job),
    ):
        result = await DeploymentService._lookup_target_column(async_session, "job1")
    assert result is None


# ---------------------------------------------------------------------------
# get_deployment_details
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_get_deployment_details_success(async_session):
    deployment = Deployment(
        id=1,
        job_id="job1",
        model_type="dummy",
        artifact_uri="pipeQ/nodeQ",
        is_active=True,
        deployed_by=None,
        created_at=datetime.now(UTC),
        updated_at=datetime.now(UTC),
    )

    with (
        patch.object(
            DeploymentService,
            "_build_input_schema_from_artifact",
            return_value=[{"name": "f1", "type": "unknown"}],
        ),
        patch.object(
            DeploymentService,
            "_lookup_target_column",
            new=AsyncMock(return_value="target_col"),
        ),
    ):
        info = await DeploymentService.get_deployment_details(async_session, deployment)

    assert info["input_schema"] == [{"name": "f1", "type": "unknown"}]
    assert info["target_column"] == "target_col"


@pytest.mark.asyncio
async def test_get_deployment_details_handles_exception(async_session):
    deployment = Deployment(
        id=2,
        job_id="job2",
        model_type="dummy",
        artifact_uri="pipeQ/nodeQ",
        is_active=True,
        deployed_by=None,
        created_at=datetime.now(UTC),
        updated_at=datetime.now(UTC),
    )

    with patch.object(
        DeploymentService,
        "_build_input_schema_from_artifact",
        side_effect=RuntimeError("boom"),
    ):
        info = await DeploymentService.get_deployment_details(async_session, deployment)

    # Exception is swallowed; schema fields remain at their defaults
    assert info["input_schema"] is None
    assert info["output_schema"] is None
