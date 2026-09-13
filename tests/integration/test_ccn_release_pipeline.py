"""Run a real branched pipeline through the code simplified for release 0.8.22."""

import io
import json
from copy import deepcopy
from dataclasses import asdict
from pathlib import Path
from typing import Any
from unittest.mock import patch

import numpy as np
import pandas as pd
import pytest
import pytest_asyncio
from fastapi import UploadFile
from sqlalchemy.ext.asyncio import AsyncSession, create_async_engine
from starlette.requests import Request

from backend.config import get_settings
from backend.data.catalog import FileSystemCatalog
from backend.database.models import Base, TrainingJob
from backend.ml_pipeline._execution.diagram import build_pipeline_diagram
from backend.ml_pipeline._execution.engine import PipelineEngine
from backend.ml_pipeline._execution.schemas import NodeConfig, PipelineConfig
from backend.ml_pipeline._internal._routers.preview import (
    _extract_preview,
    _run_preview_sub_pipelines,
)
from backend.ml_pipeline.artifacts.local import LocalArtifactStore
from backend.ml_pipeline.deployment.service import DeploymentService
from backend.monitoring.router import calculate_drift
from skyulf.modeling.base import extract_xy
from skyulf.pipeline.seal import artifact_digest
from skyulf.preprocessing.scaling.standard import StandardScalerCalculator


def _customer_data() -> pd.DataFrame:
    """Make repeatable mixed data with a known signal and genuinely missing numeric cells."""
    rng = np.random.default_rng(822)
    churn = np.arange(300) % 2
    frame = pd.DataFrame(
        {
            "age": rng.integers(20, 70, 300).astype(str),
            "usage": np.where(churn == 1, 8.0, 80.0) + rng.normal(0, 2, 300),
            "monthly_spend": rng.uniform(20, 120, 300),
            "city": np.tile(["Ankara", "Izmir", "Istanbul"], 100),
            "note": np.tile(["mobile monthly", "web annual", "mobile annual"], 100),
            "churn": churn,
        }
    )
    frame.loc[::13, "monthly_spend"] = np.nan
    return frame


def _node(node_id: str, step_type: str, inputs: list[str], **params: Any) -> NodeConfig:
    """Keep each graph node explicit and reusable in the exported execution config."""
    return NodeConfig(node_id=node_id, step_type=step_type, inputs=inputs, params=params)


def _customer_graph(source: Path, strategy: str, branched: bool = False) -> PipelineConfig:
    """Build a linear chain or fork three feature branches after a shared train/test split."""
    config = PipelineConfig(
        pipeline_id=f"ccn-0822-customer-{strategy}",
        nodes=[
            _node("source", "data_loader", [], source="csv", path=str(source)),
            _node("cast", "Casting", ["source"], column_types={"age": "float"}),
            _node(
                "split",
                "TrainTestSplitter",
                ["cast"],
                target_column="churn",
                test_size=0.2,
                random_state=42,
                stratify=True,
            ),
            _node(
                "impute",
                "SimpleImputer",
                ["split"],
                columns=["monthly_spend"],
                strategy="median",
            ),
            _node(
                "log",
                "GeneralTransformation",
                ["impute"],
                transformations=[{"column": "monthly_spend", "method": "log"}],
            ),
            _node(
                "scale",
                "StandardScaler",
                ["log"],
                columns=["age", "usage", "monthly_spend"],
            ),
            _node("encode", "OneHotEncoder", ["split"], columns=["city"]),
            _node("tfidf", "tfidf_vectorizer", ["split"], columns=["note"], drop_original=True),
            _node(
                "numeric",
                "DropMissingColumns",
                ["scale"],
                columns=["city", "note"],
                missing_threshold=0,
            ),
            _node(
                "model",
                "training",
                ["encode", "tfidf", "numeric"],
                algorithm="random_forest_classifier",
                target_column="churn",
                run_mode="tuned",
                tuning_config={
                    "strategy": strategy,
                    "metric": "accuracy",
                    "cv_enabled": True,
                    "cv_folds": 3,
                    "random_state": 42,
                    "n_jobs": 1,
                    "search_space": {
                        "n_estimators": [16],
                        "n_jobs": [1],
                        "max_depth": [2, 4],
                        "min_samples_leaf": [1, 3],
                    },
                },
            ),
        ],
    )
    if not branched:
        config.nodes = [node for node in config.nodes if node.node_id != "numeric"]
        inputs = {"encode": ["scale"], "tfidf": ["encode"], "model": ["tfidf"]}
        for node in config.nodes:
            node.inputs = inputs.get(node.node_id, node.inputs)
    return config


def _pandas(frame: Any) -> pd.DataFrame:
    """Compare results by values regardless of the selected native frame engine."""
    if hasattr(frame, "to_native"):
        frame = frame.to_native()
    return frame.to_pandas() if hasattr(frame, "to_pandas") else frame


@pytest.fixture(params=["pandas", "polars"])
def frame_engine(request):
    """Exercise both native execution engines, including pandas serving of Polars artifacts."""
    return request.param


@pytest.fixture(params=["grid", "halving_grid"])
def strategy(request):
    """Cover manual fold replay and sklearn's successive-halving search pipeline."""
    return request.param


@pytest_asyncio.fixture
async def pipeline_session(tmp_path):
    """Persist real jobs, deployments and drift alerts in an isolated temporary database."""
    database = create_async_engine(f"sqlite+aiosqlite:///{tmp_path / 'pipeline.db'}")
    try:
        async with database.begin() as connection:
            await connection.run_sync(Base.metadata.create_all)
        async with AsyncSession(database, expire_on_commit=False) as session:
            yield session
    finally:
        await database.dispose()


@pytest.fixture
def trained_customer(tmp_path, monkeypatch, frame_engine, strategy, request):
    """Train from a real CSV and retain all fitted artifacts for downstream service checks."""
    monkeypatch.setenv("SKYULF_ENGINE", frame_engine)
    monkeypatch.setattr(get_settings(), "SKYULF_ENGINE", frame_engine)
    monkeypatch.setattr(get_settings(), "TUNING_N_JOBS", 1)
    scaler_fit_rows = []
    real_fit = StandardScalerCalculator.fit

    def record_fit(self, data, config):
        """Observe row identity while retaining actual scaler learning and transformations."""
        features = data[0] if isinstance(data, tuple) else data
        scaler_fit_rows.append(set(_pandas(features)["usage"]))
        return real_fit(self, data, config)

    monkeypatch.setattr(StandardScalerCalculator, "fit", record_fit)
    raw = _customer_data()
    source = tmp_path / "customers.csv"
    raw.to_csv(source, index=False)
    config = _customer_graph(source, strategy, branched=getattr(request, "param", False))
    graph = asdict(config)
    (tmp_path / "pipeline.json").write_text(json.dumps(graph, indent=2), encoding="utf-8")
    store = LocalArtifactStore(str(tmp_path / "artifacts"))
    logs: list[str] = []
    engine = PipelineEngine(
        store, catalog=FileSystemCatalog(str(tmp_path)), log_callback=logs.append
    )
    result = engine.run(
        deepcopy(config), job_id="customer-job", dataset_name="customers", inspect_all=True
    )
    (tmp_path / "execution.log").write_text("\n".join(logs), encoding="utf-8")
    assert result.status == "success", "\n".join(
        f"{key}: {node.error}" for key, node in result.node_results.items() if node.error
    )
    assert all(node.status == "success" for node in result.node_results.values())
    metrics = result.node_results["model"].metrics
    assert metrics["best_score"] >= 0.95
    assert "fold_refit_fallback" not in metrics
    assert metrics["fold_refit_audit"]["isolation_ok"] is True
    assert metrics["test_accuracy"] == 1.0
    training_features, _ = extract_xy(store.load("split").train, "churn")
    training_rows = set(_pandas(training_features)["usage"])
    assert len(training_rows) == 240
    assert scaler_fit_rows and all(rows <= training_rows for rows in scaler_fit_rows)
    assert any(len(rows) < 240 for rows in scaler_fit_rows)
    diagram = build_pipeline_diagram(result.node_results, model_type="random_forest_classifier")
    assert diagram and "Random Forest" in diagram
    (tmp_path / "pipeline.mmd").write_text(diagram, encoding="utf-8")
    (tmp_path / "training_summary.json").write_text(
        json.dumps(
            {
                "engine": frame_engine,
                "strategy": strategy,
                "best_score": metrics["best_score"],
                "test_accuracy": metrics["test_accuracy"],
                "best_params": metrics["best_params"],
                "fold_refit_audit": metrics["fold_refit_audit"],
                "observed_scaler_fit_rows": [len(rows) for rows in scaler_fit_rows],
            },
            indent=2,
        ),
        encoding="utf-8",
    )
    return store, config, raw


def _held_out(store):
    """Recover untouched held-out inputs so inference must replay every learned transform."""
    split = store.load("split")
    features, labels = extract_xy(split.test, "churn")
    return _pandas(features).reset_index(drop=True), np.asarray(labels).ravel()


def _check_preview(tmp_path, config):
    """Run actual preview partitioning and inspection serialization against the same source."""
    store = LocalArtifactStore(str(tmp_path / "preview"))
    inspections = []
    with patch(
        "backend.ml_pipeline._internal._routers.preview.create_catalog_from_options",
        return_value=FileSystemCatalog(str(tmp_path)),
    ):
        results = _run_preview_sub_pipelines(
            deepcopy(config),
            deepcopy(config.nodes),
            config.nodes,
            None,
            None,
            store,
            node_inspections=inspections,
            inspect_all=True,
        )
    assert results and all(result.status == "success" for _, _, result in results)
    assert all("model" not in result.node_results for _, _, result in results)
    _, source_totals, _ = _extract_preview(store, "source")
    _, split_totals, _ = _extract_preview(store, "split")
    assert source_totals == {"_total": 300}
    assert split_totals == {"train_X": 240, "train_y": 240, "test_X": 60, "test_y": 60}
    scaled = [item for item in inspections if item.node_id == "scale"]
    assert scaled and all(item.output.status == "available" for item in scaled)
    assert all(item.path_id and item.path_label for item in scaled)
    (tmp_path / "preview_inspections.json").write_text(
        json.dumps([item.model_dump() for item in inspections], indent=2),
        encoding="utf-8",
    )
    assert all(
        table.row_count == {"train": 240, "test": 60}[table.split]
        for item in scaled
        for table in item.output.tables
    )


async def _drift_report(session, store, current):
    """Resolve the saved job/source and persist its drift report using local artifacts."""
    with patch("backend.monitoring.router.ArtifactFactory.get_discovery") as discovery:
        discovery.return_value.get_store_for_job.return_value = store
        return await calculate_drift(
            request=Request({"type": "http", "path": "/monitoring/drift/calculate"}),
            job_id="customer-job",
            dataset_name="customers",
            file=UploadFile(
                file=io.BytesIO(current.to_csv(index=False).encode()), filename="current.csv"
            ),
            threshold_psi=None,
            threshold_ks=None,
            threshold_wasserstein=None,
            threshold_kl=None,
            db=session,
        )


@pytest.mark.parametrize("trained_customer", [False, True], indirect=True)
async def test_tuning_preview_deployment_reload_and_drift(
    trained_customer, pipeline_session, tmp_path
):
    """A real mixed-data job must preserve predictions, preview counts and drift semantics."""
    store, config, raw = trained_customer
    _check_preview(tmp_path, config)
    session = pipeline_session
    session.add(
        TrainingJob(
            id="customer-job",
            pipeline_id=config.pipeline_id,
            node_id="model",
            dataset_source_id="customers",
            status="completed",
            run_mode="tuned",
            model_type="random_forest_classifier",
            graph=asdict(config),
            artifact_uri=store.get_artifact_uri("customer-job"),
        )
    )
    await session.commit()
    deployment = await DeploymentService.deploy_model(session, "customer-job")
    assert deployment.is_active

    bundle = store.load("customer-job")
    assert bundle["feature_engineer"] is not None
    before = artifact_digest(bundle)
    inference, test_labels = _held_out(store)
    predictions, applied = await DeploymentService.predict(session, inference.to_dict("records"))
    np.testing.assert_array_equal(predictions, test_labels)
    assert applied is None
    assert artifact_digest(bundle) == before
    reopened = LocalArtifactStore(str(tmp_path / "artifacts")).load("customer-job")
    reloaded_predictions, _ = DeploymentService._predict_with_bundled_artifact(reopened, inference)
    assert reloaded_predictions == predictions
    assert artifact_digest(reopened) == before
    examples = inference.assign(expected=test_labels, predicted=predictions)
    examples.to_csv(tmp_path / "held_out_predictions.csv", index=False)
    threshold_predictions, applied = await DeploymentService.predict(
        session,
        inference.to_dict("records"),
        override_thresholds={"0": 0.5, "1": 0.5},
    )
    assert threshold_predictions == predictions and applied == {"0": 0.5, "1": 0.5}
    same = await _drift_report(session, store, raw)
    shifted = await _drift_report(session, store, raw.assign(usage=raw["usage"] + 200))
    assert same.reference_rows == same.current_rows == 300
    assert same.drifted_columns_count == 0
    assert shifted.column_drifts["usage"]["drift_detected"] is True
    assert shifted.drifted_columns_count == 1
    assert same.alert_id and shifted.alert_id and same.alert_id != shifted.alert_id
    (tmp_path / "drift_summary.json").write_text(
        json.dumps(
            {"same": same.model_dump(), "shifted": shifted.model_dump()}, indent=2, default=str
        ),
        encoding="utf-8",
    )
    assert artifact_digest(store.load("customer-job")) == before


@pytest.mark.parametrize("trained_customer", [True], indirect=True)
def test_branched_tuning_and_preview(trained_customer, tmp_path):
    """Merging three independently fitted branches must retain signal and fold isolation."""
    store, config, _ = trained_customer
    _check_preview(tmp_path, config)
    bundle = store.load("customer-job")
    assert "usage" in bundle["feature_columns"] and len(bundle["feature_columns"]) == 10


@pytest.mark.parametrize("trained_customer", [True], indirect=True)
def test_branched_inference_keeps_inputs_needed_by_encoders(trained_customer):
    """Raw inputs must reach branch encoders before the fitted column-drop step executes."""
    store, _, _ = trained_customer
    inference, labels = _held_out(store)
    predictions, _ = DeploymentService._predict_with_bundled_artifact(
        store.load("customer-job"), inference[inference.columns[::-1]]
    )
    np.testing.assert_array_equal(predictions, labels)
