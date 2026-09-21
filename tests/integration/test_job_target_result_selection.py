"""Persist trainer results when an unfinished canvas branch remains in the partition."""

import pandas as pd
import pytest
from sqlalchemy import create_engine
from sqlalchemy.orm import Session

from backend.config import get_settings
from backend.data.catalog import FileSystemCatalog
from backend.database.models import Base, TrainingJob
from backend.ml_pipeline._execution.engine import PipelineEngine
from backend.ml_pipeline._execution.graph_utils import partition_parallel_pipeline
from backend.ml_pipeline._execution.schemas import (
    NodeConfig,
    NodeExecutionResult,
    PipelineConfig,
    PipelineExecutionResult,
)
from backend.ml_pipeline._execution.strategies import (
    AdvancedTuningStrategy,
    BasicTrainingStrategy,
)
from backend.ml_pipeline._services import pipeline_execution_service
from backend.ml_pipeline.artifacts.local import LocalArtifactStore


@pytest.mark.parametrize("run_mode", ["fixed", "tuned"])
@pytest.mark.parametrize("trainer_last", [False, True])
def test_job_metrics_follow_target_in_real_partition(tmp_path, monkeypatch, run_mode, trainer_last):
    """Reordering a sibling scaler must not replace a trainer's metrics or thresholds."""
    monkeypatch.setattr(get_settings(), "TUNING_N_JOBS", 1)
    source = tmp_path / "data.csv"
    pd.DataFrame({"x": list(range(120)), "target": [0] * 60 + [1] * 60}).to_csv(source, index=False)
    trainer = NodeConfig(
        "trainer",
        "training",
        inputs=["split"],
        params={
            "algorithm": "logistic_regression",
            "target_column": "target",
            "run_mode": run_mode,
            "tune_threshold": True,
            "tuning_config": {
                "strategy": "grid",
                "metric": "accuracy",
                "cv_enabled": False,
                "n_jobs": 1,
                "tune_threshold": True,
                "search_space": {"C": [1.0]},
            },
        },
    )
    sibling = NodeConfig("unfinished_scale", "StandardScaler", {"columns": ["x"]}, ["split"])
    leaves = [sibling, trainer] if trainer_last else [trainer, sibling]
    config = PipelineConfig(
        "target-result",
        [
            NodeConfig("source", "data_loader", {"path": str(source)}),
            NodeConfig(
                "split",
                "TrainTestSplitter",
                {
                    "target_column": "target",
                    "test_size": 0.2,
                    "validation_size": 0.2,
                    "random_state": 42,
                    "stratify": True,
                },
                ["source"],
            ),
            *leaves,
        ],
    )
    partitions = partition_parallel_pipeline(config)
    partition_ids = [[node.node_id for node in part.nodes] for part in partitions]
    expected_ids = ["source", "split", *[node.node_id for node in leaves]]
    assert partition_ids == [expected_ids]
    (tmp_path / "partition_nodes.txt").write_text(repr(partition_ids), encoding="utf-8")
    engine = PipelineEngine(
        LocalArtifactStore(str(tmp_path / "artifacts")), FileSystemCatalog(str(tmp_path))
    )
    result = engine.run(partitions[0], job_id="target-result-job")
    assert result.status == "success", result.node_results
    assert list(result.node_results) == expected_ids
    target = result.node_results["trainer"]
    assert target.metrics["best_score"] > 0.9
    assert target.metrics["trials"]
    assert target.metrics["decision_thresholds"]
    job = TrainingJob(node_id="trainer", run_mode=run_mode)
    strategy = AdvancedTuningStrategy() if run_mode == "tuned" else BasicTrainingStrategy()

    strategy.handle_success(job, result)

    for key in ("best_score", "best_params", "trials", "decision_thresholds"):
        assert job.metrics[key] == target.metrics[key]
    assert job.metrics["summary"] == target.metadata["summary"]
    assert {row["node_id"] for row in job.metrics["node_timings"]} == set(expected_ids)
    assert job.metrics["leakage_gate"] == result.leakage_verdict
    if run_mode == "tuned":
        assert job.best_params == target.metrics["best_params"]
        assert job.best_score == target.metrics["best_score"]
        assert job.results == target.metrics["trials"]
        assert job.scoring == target.metrics["scoring_metric"]
    assert job.tuned_thresholds["thresholds"] == target.metrics["decision_thresholds"]
    assert job.tuned_thresholds_enabled is True


@pytest.mark.parametrize("strategy", [BasicTrainingStrategy(), AdvancedTuningStrategy()])
@pytest.mark.parametrize("target_status", [None, "failed", "skipped"])
def test_missing_or_unsuccessful_target_cannot_use_sibling(strategy, target_status):
    """A successful sibling cannot disguise missing or unsuccessful target execution."""
    job = TrainingJob(node_id="trainer", metrics={"original": True})
    nodes = {"sibling": NodeExecutionResult("sibling", "success", metrics={"accuracy": 1.0})}
    if target_status is not None:
        nodes["trainer"] = NodeExecutionResult("trainer", target_status)
    result = PipelineExecutionResult("target-result", "success", node_results=nodes)

    with pytest.raises(ValueError, match="trainer"):
        strategy.handle_success(job, result)

    assert job.metrics == {"original": True}


@pytest.mark.parametrize("run_mode", ["fixed", "tuned"])
def test_execution_service_persists_failure_when_target_is_missing(tmp_path, monkeypatch, run_mode):
    """The job boundary must roll back optimistic completion when only a sibling ran."""
    source = tmp_path / "data.csv"
    pd.DataFrame({"x": [1, 2, 3]}).to_csv(source, index=False)
    store = LocalArtifactStore(str(tmp_path / "artifacts"))
    events = []

    def local_store(session, job, job_id, pipeline_config_dict):
        """Keep service artifacts inside this test's temporary directory."""
        return store, str(tmp_path / "artifacts"), "test-data"

    monkeypatch.setattr(pipeline_execution_service, "_create_artifact_store", local_store)
    monkeypatch.setattr(pipeline_execution_service, "publish_job_event", events.append)
    database = create_engine("sqlite://")
    Base.metadata.create_all(database)
    try:
        with Session(database) as session:
            session.add(
                TrainingJob(
                    id="missing-target-job",
                    pipeline_id="missing-target-pipeline",
                    node_id="trainer",
                    dataset_source_id="data",
                    model_type="logistic_regression",
                    run_mode=run_mode,
                    graph={},
                )
            )
            session.commit()
            pipeline_execution_service.execute_pipeline(
                "missing-target-job",
                {
                    "pipeline_id": "missing-target-pipeline",
                    "nodes": [
                        {
                            "node_id": "source",
                            "step_type": "data_loader",
                            "params": {"path": str(source)},
                            "inputs": [],
                        },
                        {
                            "node_id": "sibling",
                            "step_type": "StandardScaler",
                            "params": {"columns": ["x"]},
                            "inputs": ["source"],
                        },
                    ],
                },
                session,
            )
        with Session(database) as session:
            saved = session.get(TrainingJob, "missing-target-job")
            assert saved is not None
            assert saved.status == "failed"
            assert saved.error_message == "No execution result for job target node 'trainer'"
            assert saved.metrics is None
            assert saved.best_params is None
            assert saved.results is None
            assert saved.tuned_thresholds is None
            assert saved.tuned_thresholds_enabled is False
    finally:
        database.dispose()
    assert store.exists("sibling")
    assert [event.status for event in events] == ["running", "failed"]
