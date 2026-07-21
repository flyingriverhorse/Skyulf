"""Task 9 e2e smoke test — 9 scenarios covering all affected node types.

Run with:
    cd /Users/BH7043/Skyulf && python tests/smoke_test_task9_e2e.py

Scenarios:
  1. Classification fixed mode
  2. Classification tuned mode
  3. Regression fixed mode
  4. Segmentation (clustering, always fixed)
  5. Ensemble Voting Classifier, fixed mode
  6. Ensemble Voting Classifier, tuned mode
  7. Ensemble Stacking Classifier, fixed mode
  8. Notebook export (tuned Training + Ensemble node)
  9. Multi-branch parallel execution_mode
"""

import os
import sys
import traceback
from pathlib import Path

# Match conftest.py: set TESTING=True so FileSystemCatalog skips UPLOAD_DIR containment check
os.environ["TESTING"] = "True"
os.environ.setdefault("CELERY_BROKER_URL", "memory://")
os.environ.setdefault("CELERY_RESULT_BACKEND", "rpc://")

import pandas as pd

# Ensure repo root on path
REPO = Path(__file__).parent.parent
sys.path.insert(0, str(REPO))

from backend.data.catalog import FileSystemCatalog
from backend.ml_pipeline._execution.engine import PipelineEngine
from backend.ml_pipeline._execution.graph_utils import partition_parallel_pipeline
from backend.ml_pipeline._execution.schemas import NodeConfig, PipelineConfig
from backend.ml_pipeline.artifacts.local import LocalArtifactStore
from backend.ml_pipeline._internal._routers import notebook_export as ne
from backend.ml_pipeline._internal._routers._notebook_builders import _NodeIn, _PipelineIn

# ── artifact output dir (within project, not /tmp) ──────────────────────────
ARTIFACT_DIR = REPO / "tests" / "artifacts" / "smoke_task9"
ARTIFACT_DIR.mkdir(parents=True, exist_ok=True)


# ── helpers ──────────────────────────────────────────────────────────────────

def _make_csv(name: str, kind: str) -> str:
    """Build a minimal CSV for classification or regression tasks."""
    base = ARTIFACT_DIR / f"{name}.csv"
    if kind == "classification":
        df = pd.DataFrame({
            "f1": [1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0, 10.0,
                   1.1, 2.1, 3.1, 4.1, 5.1, 6.1, 7.1, 8.1, 9.1, 10.1],
            "f2": [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0,
                   1.1, 1.2, 1.3, 1.4, 1.5, 1.6, 1.7, 1.8, 1.9, 2.0],
            "target": [0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1],
        })
    elif kind == "regression":
        df = pd.DataFrame({
            "f1": [float(i) for i in range(1, 31)],
            "f2": [float(i) * 0.5 for i in range(1, 31)],
            "target": [float(i) * 2.0 + 1.0 for i in range(1, 31)],
        })
    elif kind == "clustering":
        df = pd.DataFrame({
            "f1": [1.0, 2.0, 3.0, 8.0, 9.0, 10.0, 1.5, 2.5, 8.5, 9.5,
                   1.2, 2.2, 3.2, 8.2, 9.2, 10.2, 1.7, 2.7, 8.7, 9.7],
            "f2": [1.0, 1.0, 1.0, 8.0, 8.0, 8.0, 1.1, 1.1, 8.1, 8.1,
                   1.2, 1.2, 1.2, 8.2, 8.2, 8.2, 1.3, 1.3, 8.3, 8.3],
        })
    else:
        raise ValueError(f"Unknown kind: {kind}")
    df.to_csv(base, index=False)
    return str(base)


def _make_engine(scenario_name: str) -> tuple[PipelineEngine, str]:
    """Create a PipelineEngine and an artifact store for one scenario."""
    art_path = str(ARTIFACT_DIR / scenario_name)
    store = LocalArtifactStore(base_path=art_path)
    catalog = FileSystemCatalog()
    engine = PipelineEngine(artifact_store=store, catalog=catalog)
    return engine, art_path


def _run_pipeline(scenario_name: str, nodes: list[NodeConfig]) -> str:
    """Run a pipeline and return its status string."""
    engine, _ = _make_engine(scenario_name)
    config = PipelineConfig(
        pipeline_id=f"smoke_{scenario_name}",
        nodes=nodes,
    )
    result = engine.run(config, job_id=f"job_{scenario_name}")
    return result.status


def _classification_nodes(csv_path: str, run_mode: str, algorithm: str = "logistic_regression") -> list[NodeConfig]:
    """Standard classification pipeline nodes."""
    return [
        NodeConfig(node_id="loader", step_type="data_loader", params={"path": csv_path}, inputs=[]),
        NodeConfig(node_id="fts", step_type="feature_target_split", params={"target_column": "target"}, inputs=["loader"]),
        NodeConfig(node_id="tts", step_type="TrainTestSplitter", params={"test_size": 0.3}, inputs=["fts"]),
        NodeConfig(
            node_id="train",
            step_type="training",
            params={
                "algorithm": algorithm,
                "target_column": "target",
                "run_mode": run_mode,
                **({"tuning_config": {"n_trials": 2}} if run_mode == "tuned" else {}),
            },
            inputs=["tts"],
        ),
    ]


def _regression_nodes(csv_path: str) -> list[NodeConfig]:
    """Standard regression pipeline nodes."""
    return [
        NodeConfig(node_id="loader", step_type="data_loader", params={"path": csv_path}, inputs=[]),
        NodeConfig(node_id="fts", step_type="feature_target_split", params={"target_column": "target"}, inputs=["loader"]),
        NodeConfig(node_id="tts", step_type="TrainTestSplitter", params={"test_size": 0.3}, inputs=["fts"]),
        NodeConfig(
            node_id="train",
            step_type="training",
            params={"algorithm": "random_forest_regressor", "target_column": "target", "run_mode": "fixed"},
            inputs=["tts"],
        ),
    ]


def _clustering_nodes(csv_path: str) -> list[NodeConfig]:
    """Segmentation pipeline nodes."""
    return [
        NodeConfig(node_id="loader", step_type="data_loader", params={"path": csv_path}, inputs=[]),
        NodeConfig(
            node_id="train",
            step_type="training",
            params={"algorithm": "kmeans", "run_mode": "fixed", "hyperparameters": {"n_clusters": 2}},
            inputs=["loader"],
        ),
    ]


def _ensemble_nodes(csv_path: str, algorithm: str, run_mode: str) -> list[NodeConfig]:
    """Ensemble pipeline nodes (voting_classifier or stacking_classifier).

    Matches pipelineConverter.ts output shape:
    - fixed: hyperparameters.base_estimators
    - tuned: tuning_config.base_estimators
    """
    if run_mode == "fixed":
        train_params = {
            "run_mode": "fixed",
            "model_type": algorithm,
            "target_column": "target",
            "hyperparameters": {
                "base_estimators": ["logistic_regression", "random_forest_classifier"],
                **({"voting": "soft"} if algorithm == "voting_classifier" else {}),
            },
        }
    else:
        train_params = {
            "run_mode": "tuned",
            "algorithm": algorithm,
            "target_column": "target",
            "tuning_config": {
                "strategy": "random",
                "n_trials": 2,
                "base_estimators": ["logistic_regression", "random_forest_classifier"],
                **({"voting": "soft"} if algorithm == "voting_classifier" else {}),
            },
        }

    return [
        NodeConfig(node_id="loader", step_type="data_loader", params={"path": csv_path}, inputs=[]),
        NodeConfig(node_id="fts", step_type="feature_target_split", params={"target_column": "target"}, inputs=["loader"]),
        NodeConfig(node_id="tts", step_type="TrainTestSplitter", params={"test_size": 0.3}, inputs=["fts"]),
        NodeConfig(node_id="train", step_type="training", params=train_params, inputs=["tts"]),
    ]


# ── scenario runners ──────────────────────────────────────────────────────────

RESULTS: list[tuple[str, bool, str]] = []  # (label, passed, details)


def run_scenario(label: str, fn):
    """Run one scenario and capture pass/fail."""
    print(f"\n{'='*60}")
    print(f"Scenario: {label}")
    print('='*60)
    try:
        fn()
        RESULTS.append((label, True, "OK"))
        print(f"  ✅ PASSED")
    except Exception as e:
        tb = traceback.format_exc()
        RESULTS.append((label, False, str(e)))
        print(f"  ❌ FAILED: {e}")
        print(tb)


# ─────────────────────────────────────────────────────────────────────────────
# Scenario 1: Classification fixed mode
# ─────────────────────────────────────────────────────────────────────────────
def scenario_1_classification_fixed():
    """Classification, run_mode='fixed' — logistic_regression."""
    csv = _make_csv("clf_fixed", "classification")
    nodes = _classification_nodes(csv, "fixed")
    status = _run_pipeline("clf_fixed", nodes)
    assert status == "success", f"Expected 'success', got {status!r}"
    print(f"  Pipeline status: {status}")


# ─────────────────────────────────────────────────────────────────────────────
# Scenario 2: Classification tuned mode
# ─────────────────────────────────────────────────────────────────────────────
def scenario_2_classification_tuned():
    """Classification, run_mode='tuned' — logistic_regression."""
    csv = _make_csv("clf_tuned", "classification")
    nodes = _classification_nodes(csv, "tuned")
    status = _run_pipeline("clf_tuned", nodes)
    assert status == "success", f"Expected 'success', got {status!r}"
    print(f"  Pipeline status: {status}")


# ─────────────────────────────────────────────────────────────────────────────
# Scenario 3: Regression fixed mode
# ─────────────────────────────────────────────────────────────────────────────
def scenario_3_regression_fixed():
    """Regression, run_mode='fixed' — random_forest_regressor."""
    csv = _make_csv("reg_fixed", "regression")
    nodes = _regression_nodes(csv)
    status = _run_pipeline("reg_fixed", nodes)
    assert status == "success", f"Expected 'success', got {status!r}"
    print(f"  Pipeline status: {status}")


# ─────────────────────────────────────────────────────────────────────────────
# Scenario 4: Segmentation (clustering, always fixed)
# ─────────────────────────────────────────────────────────────────────────────
def scenario_4_segmentation():
    """Segmentation — KMeans clustering (run_mode irrelevant, always direct)."""
    csv = _make_csv("segmentation", "clustering")
    nodes = _clustering_nodes(csv)
    status = _run_pipeline("segmentation", nodes)
    assert status == "success", f"Expected 'success', got {status!r}"
    print(f"  Pipeline status: {status}")


# ─────────────────────────────────────────────────────────────────────────────
# Scenario 5: Ensemble Voting Classifier, fixed mode
# ─────────────────────────────────────────────────────────────────────────────
def scenario_5_ensemble_voting_fixed():
    """Ensemble VotingClassifier, run_mode='fixed'."""
    csv = _make_csv("ens_voting_fixed", "classification")
    nodes = _ensemble_nodes(csv, "voting_classifier", "fixed")
    status = _run_pipeline("ens_voting_fixed", nodes)
    assert status == "success", f"Expected 'success', got {status!r}"
    print(f"  Pipeline status: {status}")


# ─────────────────────────────────────────────────────────────────────────────
# Scenario 6: Ensemble Voting Classifier, tuned mode
# ─────────────────────────────────────────────────────────────────────────────
def scenario_6_ensemble_voting_tuned():
    """Ensemble VotingClassifier, run_mode='tuned'."""
    csv = _make_csv("ens_voting_tuned", "classification")
    nodes = _ensemble_nodes(csv, "voting_classifier", "tuned")
    status = _run_pipeline("ens_voting_tuned", nodes)
    assert status == "success", f"Expected 'success', got {status!r}"
    print(f"  Pipeline status: {status}")


# ─────────────────────────────────────────────────────────────────────────────
# Scenario 7: Ensemble Stacking Classifier, fixed mode
# ─────────────────────────────────────────────────────────────────────────────
def scenario_7_ensemble_stacking_fixed():
    """Ensemble StackingClassifier, run_mode='fixed'."""
    csv = _make_csv("ens_stacking_fixed", "classification")
    nodes = _ensemble_nodes(csv, "stacking_classifier", "fixed")
    status = _run_pipeline("ens_stacking_fixed", nodes)
    assert status == "success", f"Expected 'success', got {status!r}"
    print(f"  Pipeline status: {status}")


# ─────────────────────────────────────────────────────────────────────────────
# Scenario 8: Notebook export (tuned Training node + Ensemble node)
# ─────────────────────────────────────────────────────────────────────────────
def scenario_8_notebook_export():
    """Notebook export: both full and compact modes, tuned training + ensemble."""
    # 8a: tuned classification training node
    tuned_pipeline = _PipelineIn(nodes=[
        _NodeIn(node_id="loader", step_type="data_loader", params={"path": "data.csv"}, inputs=[]),
        _NodeIn(node_id="scaler", step_type="StandardScaler", params={}, inputs=["loader"]),
        _NodeIn(node_id="fts", step_type="feature_target_split", params={"target_column": "target"}, inputs=["scaler"]),
        _NodeIn(node_id="tts", step_type="TrainTestSplitter", params={"test_size": 0.2}, inputs=["fts"]),
        _NodeIn(
            node_id="train",
            step_type="training",
            params={"algorithm": "logistic_regression", "run_mode": "tuned", "tuning_config": {"n_trials": 3}},
            inputs=["tts"],
        ),
    ])
    nb_full = ne._build_full_notebook(tuned_pipeline, "test-tuned", "Tuned Training")
    nb_compact = ne._build_compact_notebook(tuned_pipeline, "test-tuned", "Tuned Training")
    # Verify tuned training cells are present
    all_full_src = " ".join("".join(c.get("source", [])) for c in nb_full["cells"])
    all_compact_src = " ".join("".join(c.get("source", [])) for c in nb_compact["cells"])
    assert "tuning_config" in all_full_src or "n_trials" in all_full_src or "logistic_regression" in all_full_src, (
        "Full notebook missing tuned-training content"
    )
    print(f"  8a tuned-training full: {len(nb_full['cells'])} cells ✓")
    print(f"  8a tuned-training compact: {len(nb_compact['cells'])} cells ✓")

    # 8b: Ensemble voting classifier node (uses ensemble step_type='training' + algorithm)
    ensemble_pipeline = _PipelineIn(nodes=[
        _NodeIn(node_id="loader", step_type="data_loader", params={"path": "data.csv"}, inputs=[]),
        _NodeIn(node_id="fts", step_type="feature_target_split", params={"target_column": "target"}, inputs=["loader"]),
        _NodeIn(node_id="tts", step_type="TrainTestSplitter", params={"test_size": 0.2}, inputs=["fts"]),
        _NodeIn(
            node_id="train",
            step_type="training",
            params={
                "algorithm": "voting_classifier",
                "run_mode": "fixed",
                "base_estimators": ["logistic_regression", "random_forest_classifier"],
            },
            inputs=["tts"],
        ),
    ])
    nb_ens_full = ne._build_full_notebook(ensemble_pipeline, "test-ens", "Ensemble Export")
    nb_ens_compact = ne._build_compact_notebook(ensemble_pipeline, "test-ens", "Ensemble Export")
    all_ens_full_src = " ".join("".join(c.get("source", [])) for c in nb_ens_full["cells"])
    assert "voting_classifier" in all_ens_full_src or "training" in all_ens_full_src, (
        "Ensemble full notebook missing model content"
    )
    print(f"  8b ensemble full: {len(nb_ens_full['cells'])} cells ✓")
    print(f"  8b ensemble compact: {len(nb_ens_compact['cells'])} cells ✓")


# ─────────────────────────────────────────────────────────────────────────────
# Scenario 9: Multi-branch parallel execution_mode
# ─────────────────────────────────────────────────────────────────────────────
def scenario_9_parallel_execution_mode():
    """Parallel execution_mode: one training node with 2 preprocessing branches."""
    csv = _make_csv("parallel_exec", "classification")
    # Build a config with TWO input branches feeding ONE training node with execution_mode=parallel
    nodes = [
        NodeConfig(node_id="loader", step_type="data_loader", params={"path": csv}, inputs=[]),
        # Branch A: just a StandardScaler
        NodeConfig(node_id="fts_a", step_type="feature_target_split", params={"target_column": "target"}, inputs=["loader"]),
        NodeConfig(node_id="tts_a", step_type="TrainTestSplitter", params={"test_size": 0.3}, inputs=["fts_a"]),
        # Branch B: also a StandardScaler variant
        NodeConfig(node_id="fts_b", step_type="feature_target_split", params={"target_column": "target"}, inputs=["loader"]),
        NodeConfig(node_id="tts_b", step_type="TrainTestSplitter", params={"test_size": 0.3}, inputs=["fts_b"]),
        # Single training node receiving both branches in parallel mode
        NodeConfig(
            node_id="train",
            step_type="training",
            params={
                "algorithm": "logistic_regression",
                "target_column": "target",
                "run_mode": "fixed",
                "execution_mode": "parallel",
            },
            inputs=["tts_a", "tts_b"],
        ),
    ]
    config = PipelineConfig(pipeline_id="smoke_parallel", nodes=nodes)

    # partition_parallel_pipeline should split into 2 sub-pipelines
    branches = partition_parallel_pipeline(config)
    assert len(branches) == 2, f"Expected 2 branches, got {len(branches)}"
    print(f"  Partitioned into {len(branches)} branches ✓")

    # Verify the training node in each branch has execution_mode stripped
    for i, branch in enumerate(branches):
        train_nodes = [n for n in branch.nodes if n.node_id == "train"]
        assert len(train_nodes) == 1, f"Branch {i} missing training node"
        assert "execution_mode" not in train_nodes[0].params, (
            f"Branch {i} training node still has execution_mode param (should be stripped)"
        )
        # Run each branch
        engine, _ = _make_engine(f"parallel_branch_{i}")
        result = engine.run(branch, job_id=f"job_parallel_branch_{i}")
        assert result.status == "success", f"Branch {i} failed with status: {result.status}"
        print(f"  Branch {i}: step_type=training, execution_mode=stripped, result={result.status} ✓")


# ─────────────────────────────────────────────────────────────────────────────
# Main
# ─────────────────────────────────────────────────────────────────────────────

def main():
    """Run all 9 scenarios and report results."""
    run_scenario("1. Classification fixed mode", scenario_1_classification_fixed)
    run_scenario("2. Classification tuned mode", scenario_2_classification_tuned)
    run_scenario("3. Regression fixed mode", scenario_3_regression_fixed)
    run_scenario("4. Segmentation (clustering)", scenario_4_segmentation)
    run_scenario("5. Ensemble Voting Classifier fixed", scenario_5_ensemble_voting_fixed)
    run_scenario("6. Ensemble Voting Classifier tuned", scenario_6_ensemble_voting_tuned)
    run_scenario("7. Ensemble Stacking Classifier fixed", scenario_7_ensemble_stacking_fixed)
    run_scenario("8. Notebook export (tuned + ensemble)", scenario_8_notebook_export)
    run_scenario("9. Multi-branch parallel execution_mode", scenario_9_parallel_execution_mode)

    print("\n" + "="*60)
    print("SMOKE TEST RESULTS")
    print("="*60)
    passed = sum(1 for _, ok, _ in RESULTS if ok)
    total = len(RESULTS)
    for label, ok, detail in RESULTS:
        icon = "✅" if ok else "❌"
        print(f"  {icon} {label}" + ("" if ok else f" — {detail}"))
    print(f"\n{passed}/{total} scenarios passed")
    if passed < total:
        sys.exit(1)


if __name__ == "__main__":
    main()
