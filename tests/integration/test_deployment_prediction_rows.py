"""Persisted deployment artifacts must not expose unaligned prediction batches over HTTP."""

from typing import Any

import httpx
import numpy as np
import pandas as pd
import polars as pl
import pytest
from fastapi import FastAPI
from scipy.sparse import csr_matrix
from sklearn.base import BaseEstimator
from sklearn.linear_model import LinearRegression
from sqlalchemy.ext.asyncio import AsyncSession, create_async_engine

from backend.config import get_settings
from backend.database.models import Base, TrainingJob
from backend.ml_pipeline.artifacts.local import LocalArtifactStore
from backend.ml_pipeline.deployment.api import get_async_session, router
from backend.ml_pipeline.deployment.service import DeploymentService
from skyulf.pipeline import SkyulfPipeline
from skyulf.preprocessing.pipeline import FeatureEngineer


class _FilteringEngineer:
    """Represent a supported external transformer with no Skyulf prediction-mode keyword."""

    def transform(self, frame):
        """Keep configured inliers, exposing legacy bundle cardinality loss."""
        return frame.loc[frame["x"] < 100]


class _SparseEngineer:
    """Represent an external transformer emitting a supported sparse feature matrix."""

    def transform(self, frame):
        """Retain every input row when converting to the estimator's sparse format."""
        return csr_matrix(frame[["x"]].to_numpy())


class _ShortPredictor(BaseEstimator):
    """Represent a fitted external predictor returning fewer results than requested."""

    def __init__(self):
        """Expose a real fitted sklearn child and fitted-state attributes for promotion."""
        self.model_ = LinearRegression().fit(pd.DataFrame({"x": [0, 1]}), [1, 3])

    def fit(self, X, y=None):
        """Retain this deliberately malformed predictor's fixed test state."""
        return self

    def predict(self, frame):
        """Violate the output-count contract after otherwise valid preprocessing."""
        return self.model_.predict(frame)[:-1]


def _artifact(kind: str, engine: str) -> Any:
    """Create actual fitted filtering/clipping pipelines or external compatibility controls."""
    frame = pd.DataFrame({"x": np.arange(100, dtype=float)})
    frame["target"] = 2 * frame["x"] + 1
    native = frame if engine == "pandas" else pl.from_pandas(frame)
    if kind == "short_predictor":
        return _ShortPredictor()
    if kind == "sparse_bundle":
        return {
            "feature_engineer": _SparseEngineer(),
            "model": LinearRegression().fit(csr_matrix(frame[["x"]].to_numpy()), frame["target"]),
            "feature_columns": ["x"],
        }
    if kind == "external_bundle":
        return {
            "feature_engineer": _FilteringEngineer(),
            "model": LinearRegression().fit(frame[["x"]], frame["target"]),
            "feature_columns": ["x"],
        }
    transformer = "Winsorize" if kind == "winsorize" else "IQR"
    params: dict[str, Any] = {"columns": ["x"]}
    if kind == "sorted_bundle":
        transformer = "RollingAggregate"
        params.update({"sort_by": "x", "window": 2, "min_periods": 1})
    pipeline = SkyulfPipeline(
        {
            "preprocessing": [{"name": "outliers", "transformer": transformer, "params": params}],
            "modeling": {"type": "linear_regression"},
        }
    )
    pipeline.fit(native, target_column="target")
    if kind == "legacy_pipeline":
        return pipeline
    estimator = pipeline.model_estimator
    assert estimator is not None
    features = native.drop("target") if engine == "polars" else native.drop(columns="target")
    return {
        "feature_engineer": pipeline.feature_engineer,
        "model": estimator.model,
        "target_column": "target",
        "feature_columns": list(pipeline.feature_engineer.transform(features).columns),
    }


@pytest.fixture
async def deployed_client(tmp_path, monkeypatch):
    """Promote a stored model into an isolated database and call the mounted prediction router."""
    root = tmp_path / "models"
    monkeypatch.setattr(get_settings(), "TRAINING_ARTIFACT_DIR", str(root))
    store = LocalArtifactStore(str(root / "pipeline"))
    database = create_async_engine(f"sqlite+aiosqlite:///{tmp_path / 'deployment.sqlite'}")
    async with database.begin() as connection:
        await connection.run_sync(Base.metadata.create_all)

    async def sessions():
        """Open fresh request transactions so HTTP requests reload persisted deployment state."""
        async with AsyncSession(database, expire_on_commit=False) as session:
            yield session

    async def promote(artifact):
        """Save the bundle with the real artifact store and promote its completed job."""
        store.save("job", artifact)
        async with AsyncSession(database, expire_on_commit=False) as session:
            session.add(
                TrainingJob(
                    id="job",
                    pipeline_id="pipeline",
                    node_id="model",
                    dataset_source_id="source",
                    status="completed",
                    run_mode="fixed",
                    model_type="linear_regression",
                    graph={},
                    artifact_uri=store.get_artifact_uri("job"),
                )
            )
            await session.commit()
            await DeploymentService.deploy_model(session, "job")

    app = FastAPI()
    app.include_router(router)
    app.dependency_overrides[get_async_session] = sessions
    try:
        async with httpx.AsyncClient(
            transport=httpx.ASGITransport(app=app), base_url="http://test"
        ) as client:
            yield client, promote
    finally:
        await database.dispose()


@pytest.mark.parametrize("engine", ["pandas", "polars"])
@pytest.mark.parametrize("kind", ["bundle", "legacy_pipeline"])
async def test_predict_http_rejects_filtered_batches_and_keeps_the_deployment_usable(
    deployed_client, engine, kind
):
    """An HTTP caller must receive an explicit error instead of ambiguous shortened predictions."""
    client, promote = deployed_client
    await promote(_artifact(kind, engine))
    rejected = await client.post(
        "/deployment/predict", json={"data": [{"x": 20}, {"x": 1000}, {"x": 40}]}
    )
    assert rejected.status_code == 400, rejected.text
    assert "row count" in rejected.json()["detail"]
    assert "3" in rejected.json()["detail"] and "2" in rejected.json()["detail"]
    accepted = await client.post("/deployment/predict", json={"data": [{"x": 20}, {"x": 40}]})
    assert accepted.status_code == 200, accepted.text
    np.testing.assert_allclose(accepted.json()["predictions"], [41, 81])


@pytest.mark.parametrize("kind", ["external_bundle", "short_predictor"])
async def test_legacy_external_artifacts_cannot_bypass_the_row_contract(deployed_client, kind):
    """External transformer/predictor interfaces must remain supported with the same safety check."""
    client, promote = deployed_client
    await promote(_artifact(kind, "pandas"))
    response = await client.post(
        "/deployment/predict", json={"data": [{"x": 20}, {"x": 1000}, {"x": 40}]}
    )
    assert response.status_code == 400, response.text
    assert "row count" in response.json()["detail"]


async def test_winsorize_http_retains_all_rows_and_applies_the_saved_bounds(deployed_client):
    """Clipping must still produce one ordered prediction for every submitted observation."""
    client, promote = deployed_client
    artifact = _artifact("winsorize", "pandas")
    await promote(artifact)
    data = pd.DataFrame({"x": [20, 1000, 40]})
    engineer = artifact["feature_engineer"]
    assert isinstance(engineer, FeatureEngineer)
    expected = artifact["model"].predict(engineer.transform(data))
    response = await client.post("/deployment/predict", json={"data": data.to_dict("records")})
    assert response.status_code == 200, response.text
    assert len(response.json()["predictions"]) == 3
    np.testing.assert_allclose(response.json()["predictions"], expected)


async def test_external_sparse_transform_keeps_valid_predictions(deployed_client):
    """Sparse matrix length is ambiguous, but its first shape dimension counts observations."""
    client, promote = deployed_client
    await promote(_artifact("sparse_bundle", "pandas"))
    response = await client.post("/deployment/predict", json={"data": [{"x": 20}, {"x": 40}]})
    assert response.status_code == 200, response.text
    np.testing.assert_allclose(response.json()["predictions"], [41, 81])


@pytest.mark.parametrize("engine", ["pandas", "polars"])
async def test_sorted_preprocessing_cannot_relabel_predictions_by_position(deployed_client, engine):
    """A reordered temporal result must not be attached to the caller's original row order."""
    client, promote = deployed_client
    artifact = _artifact("sorted_bundle", engine)
    await promote(artifact)
    reordered = await client.post("/deployment/predict", json={"data": [{"x": 40}, {"x": 20}]})
    assert reordered.status_code == 400, reordered.text
    assert "row order" in reordered.json()["detail"]
    ordered = pd.DataFrame({"x": [20, 40]})
    accepted = await client.post("/deployment/predict", json={"data": ordered.to_dict("records")})
    assert accepted.status_code == 200, accepted.text
    expected = artifact["model"].predict(artifact["feature_engineer"].transform(ordered))
    np.testing.assert_allclose(accepted.json()["predictions"], expected)
