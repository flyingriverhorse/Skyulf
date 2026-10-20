"""Standalone projects must replay the selected fitted model without the Kayra backend."""

import importlib.util
import io
import json
import subprocess
import sys
import zipfile
from types import SimpleNamespace

import httpx
import numpy as np
import pandas as pd
import pytest
from fastapi import FastAPI
from sklearn.linear_model import LinearRegression, LogisticRegression
from sqlalchemy.ext.asyncio import AsyncSession, create_async_engine

from backend.config import get_settings
from backend.database.models import Base, TrainingJob
from backend.ml_pipeline.artifacts.local import LocalArtifactStore
from backend.ml_pipeline.deployment.api import get_async_session, router
from backend.ml_pipeline.deployment.service import DeploymentService
from skyulf.preprocessing.pipeline import FeatureEngineer


def fitted_bundle(classification=False, engine="pandas"):
    """Fit real preprocessing so exported predictions cannot pass by returning constants."""
    import polars as pl

    frame = pd.DataFrame({"x": [0.0, 1.0, np.nan, 3.0, 4.0, 5.0], "z": [1.0] * 6})
    engineer = FeatureEngineer(
        [
            {"name": "fill", "transformer": "SimpleImputer", "params": {"columns": ["x"]}},
            {"name": "scale", "transformer": "StandardScaler", "params": {"columns": ["x"]}},
        ]
    )
    features, _ = engineer.fit_transform(frame if engine == "pandas" else pl.from_pandas(frame))
    if engine == "polars":
        features = features.to_pandas()
    model = (
        LogisticRegression().fit(features, ["no", "no", "no", "yes", "yes", "yes"])
        if classification
        else LinearRegression().fit(features, [1.0, 3.0, 5.0, 7.0, 9.0, 11.0])
    )
    return {
        "model": model,
        "feature_engineer": engineer,
        "feature_columns": ["x", "z"],
        "input_schema": [{"name": name, "dtype": "float64"} for name in frame.columns],
        "target_column": "target",
        "engine": engine,
        "training_data": "PRIVATE_SOURCE_SENTINEL",
        "credentials": "PRIVATE_CREDENTIAL_SENTINEL",
    }


def export_job(**changes):
    """Represent only persisted job values consumed by the pure archive builder."""
    values = {
        "id": "job-1",
        "version": 2,
        "model_type": "regression",
        "tuned_thresholds_enabled": False,
        "tuned_thresholds": None,
    }
    return SimpleNamespace(**(values | changes))


def unpack_project(tmp_path, bundle, job=None, mode="both"):
    """Extract the genuine ZIP and import its generated common scorer."""
    from backend.ml_pipeline.project_export.service import build_project

    content, filename = build_project(bundle, job or export_job(), mode)
    assert filename == "kayra_model_job-1.zip"
    with zipfile.ZipFile(io.BytesIO(content)) as archive:
        archive.extractall(tmp_path)
        members = set(archive.namelist())
    spec = importlib.util.spec_from_file_location("exported_scoring", tmp_path / "scoring.py")
    assert spec and spec.loader
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module.Predictor(tmp_path), members


@pytest.mark.parametrize("engine", ["pandas", "polars"])
@pytest.mark.parametrize("classification", [False, True])
def test_exported_model_matches_existing_serving(tmp_path, engine, classification):
    """Freshly loaded fitted preprocessing and saved thresholds preserve production predictions."""
    bundle = fitted_bundle(classification, engine)
    job = export_job(
        tuned_thresholds_enabled=classification,
        tuned_thresholds={"thresholds": {"no": 0.8, "yes": 0.2}} if classification else None,
    )
    thresholds = job.tuned_thresholds["thresholds"] if classification else None
    rows = [{"z": 1.0, "x": 0.2}, {"z": 1.0, "x": None}, {"z": 1.0, "x": 4.2}]
    expected, _ = DeploymentService._predict_with_bundled_artifact(
        bundle, pd.DataFrame(rows), thresholds
    )
    predictor, members = unpack_project(tmp_path, bundle, job)
    actual = predictor.predict(pd.DataFrame(rows))
    assert actual["predictions"] == expected
    assert actual["thresholds_applied"] == thresholds
    assert actual["model_version"] == "job-1"
    assert {"model.pkl", "app.py", "batch.py", "schema.json", "requirements.txt"} <= members
    payload = (tmp_path / "model.pkl").read_bytes()
    assert b"PRIVATE_SOURCE_SENTINEL" not in payload
    assert b"PRIVATE_CREDENTIAL_SENTINEL" not in payload
    assert "backend" not in (tmp_path / "scoring.py").read_text()


@pytest.fixture
async def export_client(tmp_path, monkeypatch):
    """Use real artifact storage and SQL transactions without creating a deployment."""
    root = tmp_path / "models"
    monkeypatch.setattr(get_settings(), "TRAINING_ARTIFACT_DIR", str(root))
    monkeypatch.setattr(get_settings(), "TESTING", False)
    store = LocalArtifactStore(str(root / "pipeline"))
    store.save("job-1", fitted_bundle())
    database = create_async_engine(f"sqlite+aiosqlite:///{tmp_path / 'export.sqlite'}")
    async with database.begin() as connection:
        await connection.run_sync(Base.metadata.create_all)
    async with AsyncSession(database, expire_on_commit=False) as session:
        session.add(
            TrainingJob(
                id="job-1",
                pipeline_id="pipeline",
                node_id="model",
                dataset_source_id="source",
                status="completed",
                run_mode="fixed",
                model_type="linear_regression",
                graph={},
                artifact_uri=store.get_artifact_uri("job-1"),
            )
        )
        await session.commit()

    async def sessions():
        """Reload job state for each HTTP request."""
        async with AsyncSession(database, expire_on_commit=False) as session:
            yield session

    app = FastAPI()
    app.include_router(router)
    app.dependency_overrides[get_async_session] = sessions
    try:
        async with httpx.AsyncClient(
            transport=httpx.ASGITransport(app=app), base_url="http://test"
        ) as client:
            yield client, database
    finally:
        await database.dispose()


async def test_download_does_not_deploy_or_retrain(export_client):
    """Downloading a chosen version cannot change the active serving model."""
    client, database = export_client
    response = await client.post("/deployment/export/job-1?mode=batch")
    assert response.status_code == 200, response.text
    assert response.headers["content-type"] == "application/zip"
    assert "kayra_model_job-1.zip" in response.headers["content-disposition"]
    with zipfile.ZipFile(io.BytesIO(response.content)) as archive:
        assert "batch.py" in archive.namelist()
        assert "app.py" not in archive.namelist()
    async with AsyncSession(database) as session:
        assert await DeploymentService.get_active_deployment(session) is None


@pytest.mark.parametrize(
    "job_id,mode,status", [("missing", "both", 404), ("job-1", "unknown", 422)]
)
async def test_download_errors_are_explicit(export_client, job_id, mode, status):
    """Missing jobs and unsupported output formats must never yield a misleading ZIP."""
    client, _ = export_client
    response = await client.post(f"/deployment/export/{job_id}?mode={mode}")
    assert response.status_code == status


@pytest.mark.parametrize("artifact_uri", [None, "pipeline/private-missing-model.pkl"])
async def test_download_rejects_unavailable_artifacts_without_leaking_paths(
    export_client, artifact_uri
):
    """Missing saved state must return an actionable error without exposing storage paths."""
    client, database = export_client
    async with AsyncSession(database) as session:
        job = await session.get(TrainingJob, "job-1")
        job.artifact_uri = artifact_uri
        await session.commit()

    response = await client.post("/deployment/export/job-1")

    assert response.status_code == 400
    detail = response.json()["detail"]
    assert "artifact" in detail
    assert "private-missing-model" not in detail
    async with AsyncSession(database) as session:
        assert await DeploymentService.get_active_deployment(session) is None


async def test_download_hides_unexpected_packaging_errors(export_client, monkeypatch):
    """Packaging failures must not leak local paths or return a partial downloadable project."""
    from backend.ml_pipeline.project_export import service

    def unreadable_installation():
        """Represent a filesystem failure while reading the installed Core distribution."""
        raise OSError("Permission denied: PRIVATE_INSTALLATION_PATH")

    monkeypatch.setattr(service, "bundled_core_wheel", unreadable_installation)
    client, database = export_client

    response = await client.post("/deployment/export/job-1")

    assert response.status_code == 500
    assert response.json() == {"detail": "Model project export failed."}
    assert "content-disposition" not in response.headers
    async with AsyncSession(database) as session:
        assert await DeploymentService.get_active_deployment(session) is None


@pytest.mark.parametrize("engine", ["pandas", "polars"])
def test_actual_training_records_raw_schema_before_encoding(tmp_path, monkeypatch, engine):
    """The downloaded project must accept raw categories, never generated dummy columns."""
    from backend.data.catalog import FileSystemCatalog
    from backend.ml_pipeline._execution.engine import PipelineEngine
    from backend.ml_pipeline._execution.schemas import NodeConfig, PipelineConfig

    monkeypatch.setattr(get_settings(), "SKYULF_ENGINE", engine)
    source = tmp_path / "source.csv"
    pd.DataFrame(
        {
            "city": ["London", "Paris"] * 30,
            "x": list(range(60)),
            "target": [0, 1] * 30,
        }
    ).to_csv(source, index=False)
    store = LocalArtifactStore(str(tmp_path / "artifacts"))
    pipeline = PipelineEngine(store, FileSystemCatalog(str(tmp_path)))
    result = pipeline.run(
        PipelineConfig(
            "export-test",
            [
                NodeConfig("source", "data_loader", {"path": str(source)}),
                NodeConfig("split", "TrainTestSplitter", {"target_column": "target"}, ["source"]),
                NodeConfig("encode", "DummyEncoder", {"columns": ["city"]}, ["split"]),
                NodeConfig(
                    "trainer",
                    "training",
                    {
                        "model_type": "logistic_regression",
                        "target_column": "target",
                        "cv_enabled": False,
                    },
                    ["encode"],
                ),
            ],
        ),
        job_id="job-1",
    )
    assert result.status == "success", result
    bundle = store.load("job-1")
    assert [col["name"] for col in bundle["input_schema"]] == ["city", "x"]
    assert "city" not in bundle["feature_columns"]
    predictor, _ = unpack_project(tmp_path / "project", bundle)
    rows = pd.DataFrame({"city": ["London", "Paris"], "x": [4, 55]})
    expected, _ = DeploymentService._predict_with_bundled_artifact(bundle, rows)
    assert predictor.predict(rows)["predictions"] == expected


@pytest.mark.parametrize("mode", ["api", "batch", "both"])
def test_generated_entrypoints_run_without_backend_imports(tmp_path, mode):
    """Both entry points replay the same serialized state in a fresh Python process."""
    predictor, members = unpack_project(tmp_path, fitted_bundle(), mode=mode)
    rows = pd.DataFrame({"x": [0.5, 4.5], "z": [1.0, 1.0]})
    expected = predictor.predict(rows)["predictions"]
    rows.to_parquet(tmp_path / "input.parquet", index=False)
    rows.to_csv(tmp_path / "input.csv", index=False)
    (tmp_path / "request.json").write_text(json.dumps({"data": rows.to_dict("records")}))
    script = """
import importlib.abc, json, runpy, sys
from pathlib import Path
sys.path.insert(0, str(Path.cwd()))
class NoBackend(importlib.abc.MetaPathFinder):
    def find_spec(self, fullname, path=None, target=None):
        if fullname.split(".")[0] == "backend":
            raise AssertionError("Export tried to import backend")
sys.meta_path.insert(0, NoBackend())
if Path("app.py").exists():
    from fastapi.testclient import TestClient
    from app import app
    with TestClient(app) as client:
        assert client.get("/health").json()["model_version"] == "job-1"
        assert client.get("/schema").status_code == 200
        request = json.loads(Path("request.json").read_text())
        response = client.post("/predict", json=request)
        assert response.status_code == 200, response.text
        Path("api-result.json").write_text(json.dumps(response.json()))
        assert client.post("/predict", json={"data": []}).json()["predictions"] == []
        assert client.post("/predict", json={"data": [{"x": 1}]}).status_code == 422
if Path("batch.py").exists():
    for suffix in ("csv", "parquet"):
        sys.argv = ["batch.py", "--input", "input."+suffix, "--output", "output."+suffix]
        runpy.run_path("batch.py", run_name="__main__")
sys.argv = ["smoke_test.py", "--request", "request.json"]
runpy.run_path("smoke_test.py", run_name="__main__")
"""
    result = subprocess.run(
        [sys.executable, "-I", "-c", script],
        cwd=tmp_path,
        text=True,
        capture_output=True,
        timeout=60,
    )
    assert result.returncode == 0, result.stdout + result.stderr
    if mode != "batch":
        assert json.loads((tmp_path / "api-result.json").read_text())["predictions"] == expected
    if mode != "api":
        np.testing.assert_allclose(pd.read_csv(tmp_path / "output.csv")["prediction"], expected)
        np.testing.assert_allclose(
            pd.read_parquet(tmp_path / "output.parquet")["prediction"], expected
        )
    assert ("app.py" in members) == (mode != "batch")
    assert ("batch.py" in members) == (mode != "api")


@pytest.mark.parametrize("damage", ["schema", "context", "custom", "unfitted"])
def test_unsupported_artifacts_fail_before_download(tmp_path, damage):
    """Unproven exports must be actionable errors instead of broken runnable-looking projects."""
    from backend.ml_pipeline.project_export.service import build_project

    bundle = fitted_bundle()
    if damage == "schema":
        bundle.pop("input_schema")
    elif damage == "context":
        engineer = FeatureEngineer(
            [
                {
                    "name": "history",
                    "transformer": "RollingAggregate",
                    "params": {"columns": ["x"], "window": 2},
                }
            ]
        )
        engineer.fit_transform(pd.DataFrame({"x": [1.0, 2.0, 3.0]}))
        bundle["feature_engineer"] = engineer
    elif damage == "custom":
        bundle["model"].callback = lambda value: value
    else:
        bundle["model"] = LinearRegression()
    with pytest.raises(ValueError):
        build_project(bundle, export_job(), "both")


def test_saved_disabled_thresholds_and_corruption(tmp_path):
    """Only enabled policies are frozen; corrupted models must fail before deserialization."""
    bundle = fitted_bundle(classification=True)
    job = export_job(tuned_thresholds={"thresholds": {"no": 0.8, "yes": 0.2}})
    predictor, _ = unpack_project(tmp_path, bundle, job)
    assert predictor.predict(pd.DataFrame({"x": [3.0], "z": [1.0]}))["thresholds_applied"] is None
    (tmp_path / "model.pkl").write_bytes(b"not a pickle")
    with pytest.raises(ValueError, match="checksum"):
        type(predictor)(tmp_path)


def test_target_labels_are_decoded(tmp_path):
    """Encoded training targets must be returned with their original string identities."""
    frame = pd.DataFrame({"x": range(8), "target": ["low"] * 4 + ["high"] * 4})
    engineer = FeatureEngineer(
        [
            {
                "name": "labels",
                "transformer": "LabelEncoder",
                "params": {"columns": ["target"]},
            }
        ]
    )
    transformed, _ = engineer.fit_transform(frame, target_column="target")
    bundle = fitted_bundle()
    bundle.update(
        feature_engineer=engineer,
        model=LogisticRegression().fit(transformed[["x"]], transformed["target"]),
        input_schema=[{"name": "x", "dtype": "int64"}],
        feature_columns=["x"],
    )
    predictor, _ = unpack_project(tmp_path, bundle)
    rows = pd.DataFrame({"x": [0, 7]})
    expected, _ = DeploymentService._predict_with_bundled_artifact(bundle, rows)
    assert predictor.predict(rows)["predictions"] == expected
    assert set(expected) == {"low", "high"}


def test_invalid_input_and_nonfinite_outputs_are_rejected(tmp_path):
    """Bad row/schema/output shapes cannot look like valid prediction responses."""
    predictor, _ = unpack_project(tmp_path, fitted_bundle())
    with pytest.raises(ValueError, match="Missing required"):
        predictor.predict(pd.DataFrame({"x": [1.0]}))
    with pytest.raises(ValueError, match="unique"):
        predictor.predict(pd.DataFrame([[1, 2]], columns=["x", "x"]))
    assert predictor.predict(pd.DataFrame(columns=["x", "z"]))["predictions"] == []
    predictor.state["model"].intercept_ = np.inf
    with pytest.raises(ValueError):
        predictor.predict(pd.DataFrame({"x": [1.0], "z": [1.0]}))


@pytest.mark.parametrize("package", ["xgboost", "lightgbm"])
def test_native_model_dependencies_are_pinned(tmp_path, package):
    """Optional fitted native estimators must not disappear from exported dependencies."""
    native = pytest.importorskip(package)
    from importlib.metadata import version

    bundle = fitted_bundle()
    features = bundle["feature_engineer"].transform(
        pd.DataFrame(
            {
                "x": list(range(10)),
                "z": [1.0] * 10,
            }
        )
    )
    estimator = native.XGBRegressor if package == "xgboost" else native.LGBMRegressor
    bundle["model"] = estimator(n_estimators=2, n_jobs=1).fit(features, np.arange(10))
    predictor, _ = unpack_project(tmp_path, bundle)
    assert f"{package}=={version(package)}" in (tmp_path / "requirements.txt").read_text()
    rows = pd.DataFrame({"x": [2.0, 7.0], "z": [1.0, 1.0]})
    expected, _ = DeploymentService._predict_with_bundled_artifact(bundle, rows)
    np.testing.assert_allclose(predictor.predict(rows)["predictions"], expected)


@pytest.mark.parametrize("status", ["running", "failed"])
async def test_incomplete_jobs_cannot_export(export_client, status):
    """A known but unsuccessful training run is an explicit 400, never a stale model download."""
    client, database = export_client
    async with AsyncSession(database) as session:
        job = await session.get(TrainingJob, "job-1")
        job.status = status
        await session.commit()
    response = await client.post("/deployment/export/job-1")
    assert response.status_code == 400
    assert "not completed" in response.json()["detail"]


def test_engine_schema_state_is_reset_between_runs(tmp_path):
    """An engine reused for another graph must not attach a previous dataset's schema."""
    from backend.data.catalog import FileSystemCatalog
    from backend.ml_pipeline._execution.engine import PipelineEngine
    from backend.ml_pipeline._execution.schemas import NodeConfig, PipelineConfig

    engine = PipelineEngine(LocalArtifactStore(str(tmp_path)), FileSystemCatalog(str(tmp_path)))
    engine._raw_input_schemas["source"] = [{"name": "stale", "dtype": "float64"}]
    engine._init_run_state(PipelineConfig("next", []), "next")
    assert engine._raw_input_schemas == {}
    assert (
        engine._export_input_schema(NodeConfig("model", "training", {}, ["unknown"]), "y") is None
    )


def test_lazy_h3_dependency_is_pinned(tmp_path):
    """A primitive H3 fitted state still needs its lazy optional runtime dependency."""
    pytest.importorskip("h3")
    from importlib.metadata import version

    raw = pd.DataFrame({"lat": [10.0, 20.0, 30.0], "lon": [2.0, 3.0, 4.0]})
    engineer = FeatureEngineer(
        [
            {
                "name": "cell",
                "transformer": "H3Index",
                "params": {"lat_col": "lat", "lon_col": "lon"},
            },
            {"name": "code", "transformer": "LabelEncoder", "params": {"columns": ["h3_index"]}},
        ]
    )
    features, _ = engineer.fit_transform(raw)
    bundle = fitted_bundle()
    bundle.update(
        feature_engineer=engineer,
        model=LinearRegression().fit(features, [1.0, 2.0, 3.0]),
        feature_columns=list(features.columns),
        input_schema=[{"name": name, "dtype": "float64"} for name in raw],
    )
    predictor, _ = unpack_project(tmp_path, bundle)
    assert f"h3=={version('h3')}" in (tmp_path / "requirements.txt").read_text()
    np.testing.assert_allclose(predictor.predict(raw)["predictions"], [1.0, 2.0, 3.0])


def test_embedded_sentence_snapshots_require_separate_packaging():
    """Do not accept opaque neural snapshots without proving their dependency closure."""
    from backend.ml_pipeline.project_export.service import build_project

    bundle = fitted_bundle()
    bundle["feature_engineer"].fitted_steps.append(
        {
            "name": "embedding",
            "type": "sentence_embedder",
            "artifact": {},
        }
    )
    with pytest.raises(ValueError, match="sentence_embedder"):
        build_project(bundle, export_job(), "both")


def test_regression_target_decoder_matches_existing_fallback(tmp_path):
    """An extrapolated regression result outside encoder classes retains serving behavior."""
    engineer = FeatureEngineer(
        [
            {
                "name": "encode",
                "transformer": "LabelEncoder",
                "params": {"columns": ["target"]},
            }
        ]
    )
    frame = pd.DataFrame({"x": [0.0, 1.0, 2.0], "target": [10, 20, 30]})
    features, _ = engineer.fit_transform(frame, target_column="target")
    bundle = fitted_bundle()
    bundle.update(
        feature_engineer=engineer,
        model=LinearRegression().fit(features[["x"]], features["target"]),
        feature_columns=["x"],
        input_schema=[{"name": "x", "dtype": "float64"}],
    )
    predictor, _ = unpack_project(tmp_path, bundle)
    rows = pd.DataFrame({"x": [10.0]})
    expected, _ = DeploymentService._predict_with_bundled_artifact(bundle, rows)
    assert predictor.predict(rows)["predictions"] == expected


def test_editable_install_is_explained_without_leaking_its_path(tmp_path, monkeypatch):
    """An unreleased source install must include its wheel without leaking local paths."""
    from importlib.metadata import version

    from backend.ml_pipeline.project_export import service

    metadata = {
        "url": "file:///PRIVATE_WORKSPACE_SENTINEL",
        "dir_info": {"editable": True},
    }
    monkeypatch.setattr(
        service,
        "distribution",
        lambda name: SimpleNamespace(
            version=version(name),
            read_text=lambda filename: json.dumps(metadata),
        ),
        raising=False,
    )
    _, _ = unpack_project(tmp_path, fitted_bundle())
    manifest = json.loads((tmp_path / "manifest.json").read_text())
    assert manifest["core_installation"]["source"] == "editable"
    assert manifest["core_installation"]["package_index_availability"] == "not_checked"
    assert manifest["core_installation"]["matching_wheel_recommended"] is False
    assert (tmp_path / manifest["core_installation"]["bundled_wheel"]).is_file()
    readme = (tmp_path / "README.md").read_text()
    assert "wheels/" in readme
    assert "PRIVATE_WORKSPACE_SENTINEL" not in json.dumps(manifest) + readme


@pytest.mark.parametrize("zone", ["Europe/Vilnius", "dateutil/Europe/London"])
def test_retained_timezone_dependencies_are_portable(tmp_path, zone):
    """Learned timezone-aware scalar state must carry the native timezone dependency."""
    timezone = (
        pytest.importorskip("dateutil.tz").gettz("Europe/London")
        if zone.startswith("dateutil/")
        else pytest.importorskip("pytz").timezone(zone)
    )
    bundle = fitted_bundle()
    bundle["model"].learned_time_ = pd.Timestamp("2024-01-01", tz=timezone)
    predictor, _ = unpack_project(tmp_path, bundle)
    assert predictor.state["model"].learned_time_ == bundle["model"].learned_time_
    requirements = (tmp_path / "requirements.txt").read_text()
    assert ("python-dateutil==" if zone.startswith("dateutil/") else "pytz==") in requirements


@pytest.mark.parametrize("engine", ["pandas", "polars"])
@pytest.mark.parametrize("layout", ["bare", "split_only", "grouped", "separate"])
def test_pipeline_graph_layouts_export_raw_input_parity(tmp_path, monkeypatch, engine, layout):
    """Node grouping must not lose raw schemas or duplicate column-generating transforms."""
    from backend.data.catalog import FileSystemCatalog
    from backend.ml_pipeline._execution.engine import PipelineEngine
    from backend.ml_pipeline._execution.schemas import NodeConfig, PipelineConfig

    monkeypatch.setattr(get_settings(), "SKYULF_ENGINE", engine)
    frame = pd.DataFrame(
        {
            "x": np.linspace(1, 60, 60),
            "z": np.tile([1.0, 2.0, 3.0], 20),
            "target": np.linspace(3, 121, 60),
        }
    )
    source = tmp_path / "source.csv"
    frame.to_csv(source, index=False)
    steps = [
        {
            "name": "split",
            "transformer": "TrainTestSplitter",
            "params": {"target_column": "target", "random_state": 42},
        },
        {"name": "scale", "transformer": "StandardScaler", "params": {"columns": ["x", "z"]}},
        {"name": "poly", "transformer": "PolynomialFeatures", "params": {"columns": ["x"]}},
    ]
    nodes = [NodeConfig("source", "data_loader", {"path": str(source)})]
    if layout == "grouped":
        nodes.append(NodeConfig("features", "feature_engineering", {"steps": steps}, ["source"]))
    else:
        selected = {"bare": [], "split_only": steps[:1], "separate": steps}[layout]
        for step in selected:
            nodes.append(
                NodeConfig(step["name"], step["transformer"], step["params"], [nodes[-1].node_id])
            )
    nodes.append(
        NodeConfig(
            "trainer",
            "training",
            {"model_type": "linear_regression", "target_column": "target", "cv_enabled": False},
            [nodes[-1].node_id],
        )
    )
    store = LocalArtifactStore(str(tmp_path / "artifacts"))
    result = PipelineEngine(store, FileSystemCatalog(str(tmp_path))).run(
        PipelineConfig("export-layout", nodes), job_id="job-1"
    )
    assert result.status == "success", result
    bundle = store.load("job-1")
    assert [column["name"] for column in bundle["input_schema"]] == ["x", "z"]
    predictor, _ = unpack_project(tmp_path / "project", bundle)
    rows = pd.DataFrame({"z": [2.0, 3.0], "x": [1.5, 62.5]})
    if layout == "bare":
        assert bundle["transformers"] == []
        expected = bundle["model"].predict(rows[bundle["feature_columns"]]).tolist()
    else:
        expected, _ = DeploymentService._predict_with_bundled_artifact(bundle, rows)
    actual = predictor.predict(rows)["predictions"]
    np.testing.assert_allclose(actual, expected)
    assert len(actual) == len(rows)


@pytest.mark.parametrize(
    "model_name",
    [
        "GradientBoostingClassifier",
        "GradientBoostingRegressor",
        "HistGradientBoostingClassifier",
        "HistGradientBoostingRegressor",
        "SGDClassifier",
    ],
)
def test_sklearn_compiled_loss_state_exports(tmp_path, model_name):
    """Built-in sklearn estimators must not be mistaken for application-specific classes."""
    from sklearn import ensemble, linear_model

    model_class = getattr(linear_model if model_name == "SGDClassifier" else ensemble, model_name)
    frame = pd.DataFrame({"x": [0.0, 1.0, 2.0, 3.0, 4.0, 5.0], "z": [1.0] * 6})
    target = [0, 0, 0, 1, 1, 1] if model_name.endswith("Classifier") else frame["x"] * 2
    bundle = fitted_bundle()
    features = bundle["feature_engineer"].transform(frame)
    bundle["model"] = model_class(random_state=42).fit(features, target)
    predictor, _ = unpack_project(tmp_path, bundle)
    expected, _ = DeploymentService._predict_with_bundled_artifact(bundle, frame)
    np.testing.assert_allclose(predictor.predict(frame)["predictions"], expected)


@pytest.mark.parametrize("engine", ["pandas", "polars"])
@pytest.mark.parametrize("transformer", ["Normalizer", "QuantileTransformer"])
def test_native_scalers_replay_under_pandas_output_config(tmp_path, engine, transformer):
    """Serving's sklearn pandas output setting must not break native scaler restoration."""
    import polars as pl

    frame = pd.DataFrame({"x": [0.0, 1.0, 2.0, 3.0, 4.0, 5.0], "z": [1.0] * 6})
    params = {"columns": ["x", "z"]}
    if transformer == "QuantileTransformer":
        params["n_quantiles"] = 6
    engineer = FeatureEngineer([{"name": "scale", "transformer": transformer, "params": params}])
    features, _ = engineer.fit_transform(frame if engine == "pandas" else pl.from_pandas(frame))
    features = features if isinstance(features, pd.DataFrame) else features.to_pandas()
    bundle = fitted_bundle()
    bundle.update(
        feature_engineer=engineer,
        model=LinearRegression().fit(features, frame["x"]),
        engine=engine,
    )
    predictor, _ = unpack_project(tmp_path, bundle)
    rows = pd.DataFrame({"x": [0.5, 4.5], "z": [1.0, 1.0]})
    expected, _ = DeploymentService._predict_with_bundled_artifact(bundle, rows)
    np.testing.assert_allclose(predictor.predict(rows)["predictions"], expected)
