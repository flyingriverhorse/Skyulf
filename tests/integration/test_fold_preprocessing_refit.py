"""F-15 backend wiring: always-on per-fold preprocessing refit at engine level.

The app threads a ``FeatureEngineerFoldAdapter`` + pre-transform train payload
into every tuning/CV run so preprocessing statistics never see held-out fold
rows — all tuning strategies included (halving/optuna get it via a Pipeline
wrapper around the searcher's internal CV). Graphs the hook cannot serve
(merged branches, validation-split holdout) fall back to pre-transformed
scoring with an explicit job-log warning — never a failed run, never a silent
leak.
"""

import numpy as np
import pandas as pd
import pytest

from backend.data.catalog import FileSystemCatalog
from backend.ml_pipeline._execution.engine import PipelineEngine
from backend.ml_pipeline._execution.schemas import NodeConfig, PipelineConfig
from backend.ml_pipeline.artifacts.local import LocalArtifactStore
from backend.ml_pipeline.constants import StepType

SPLITTER_STEP = {
    "name": "split",
    "transformer": "TrainTestSplitter",
    "params": {"target_column": "target", "test_size": 0.2},
}
WOE_STEP = {
    "name": "woe_city",
    "transformer": "WOEEncoder",
    "params": {"columns": ["city"], "regularization": 0.5},
}


@pytest.fixture
def noise_target_csv(tmp_path):
    """400 pure-noise rows / 200 categories: WOE can only 'predict' by memorising."""
    rng = np.random.default_rng(42)
    n, n_categories = 400, 200
    df = pd.DataFrame(
        {
            "city": [f"c{v}" for v in rng.integers(0, n_categories, size=n)],
            "target": rng.integers(0, 2, size=n),
        }
    )
    path = tmp_path / "noise.csv"
    df.to_csv(path, index=False)
    return str(path)


def _loader(node_id: str, path: str) -> NodeConfig:
    return NodeConfig(
        node_id=node_id, step_type=StepType.DATA_LOADER, params={"source": "csv", "path": path}
    )


def _training(inputs: list[str], node_id: str = "node_training", **extra) -> NodeConfig:
    params = {
        "target_column": "target",
        "algorithm": "logistic_regression",
        "hyperparameters": {"C": 1.0},
        "metric": "roc_auc",
        "cv_enabled": True,
        "cv_folds": 5,
        "evaluate": True,
    }
    params.update(extra)
    return NodeConfig(node_id=node_id, step_type=StepType.TRAINING, inputs=inputs, params=params)


def _run(tmp_path, nodes, job_id):
    logs: list[str] = []
    store = LocalArtifactStore(str(tmp_path / "artifacts"))
    engine = PipelineEngine(store, catalog=FileSystemCatalog(), log_callback=logs.append)
    result = engine.run(PipelineConfig(pipeline_id=job_id, nodes=nodes), job_id=job_id)
    return result, logs


def _disc(auc: float) -> float:
    return max(auc, 1.0 - auc)


def test_fe_node_woe_refit_keeps_cv_near_chance(noise_target_csv, tmp_path):
    """Combined FE node (splitter + WOE): refit is enabled and CV stays near chance.

    The pre-learning train payload is reconstructed by re-running the
    splitter-only step prefix; the memorising WOE step then refits per fold.
    """
    result, logs = _run(
        tmp_path,
        [
            _loader("node_data", noise_target_csv),
            NodeConfig(
                node_id="node_features",
                step_type=StepType.FEATURE_ENGINEERING,
                inputs=["node_data"],
                params={"steps": [SPLITTER_STEP, WOE_STEP]},
            ),
            _training(["node_features"]),
        ],
        job_id="f15-fe-node",
    )

    assert result.status == "success"
    assert any("Per-fold preprocessing refit enabled" in m for m in logs)
    metrics = result.node_results["node_training"].metrics
    assert "cv_roc_auc_mean" in metrics
    assert _disc(metrics["cv_roc_auc_mean"]) < 0.65, (
        f"per-fold refit CV should sit near chance, got {metrics['cv_roc_auc_mean']}"
    )


def test_transformer_chain_refit_keeps_cv_near_chance(noise_target_csv, tmp_path):
    """Separate splitter + WOE transformer nodes: payload loads from the splitter artifact."""
    result, logs = _run(
        tmp_path,
        [
            _loader("node_data", noise_target_csv),
            NodeConfig(
                node_id="node_split",
                step_type="TrainTestSplitter",
                inputs=["node_data"],
                params={"target_column": "target", "test_size": 0.2},
            ),
            NodeConfig(
                node_id="node_woe",
                step_type="WOEEncoder",
                inputs=["node_split"],
                params={"columns": ["city"], "regularization": 0.5},
            ),
            _training(["node_woe"]),
        ],
        job_id="f15-transformer-chain",
    )

    assert result.status == "success"
    assert any("Per-fold preprocessing refit enabled" in m for m in logs)
    metrics = result.node_results["node_training"].metrics
    assert _disc(metrics["cv_roc_auc_mean"]) < 0.65


def test_tuned_run_inner_and_outer_scores_both_near_chance(noise_target_csv, tmp_path):
    """Tuning (inner CV) and the tuned CV (outer) both refit per fold."""
    result, logs = _run(
        tmp_path,
        [
            _loader("node_data", noise_target_csv),
            NodeConfig(
                node_id="node_features",
                step_type=StepType.FEATURE_ENGINEERING,
                inputs=["node_data"],
                params={"steps": [SPLITTER_STEP, WOE_STEP]},
            ),
            _training(
                ["node_features"],
                run_mode="tuned",
                tuning_config={
                    "strategy": "grid",
                    "metric": "roc_auc",
                    "cv_folds": 2,
                    "cv_enabled": True,
                    "search_space": {"C": [0.1, 1.0]},
                },
            ),
        ],
        job_id="f15-tuned-inner-outer",
    )

    assert result.status == "success"
    assert any("Per-fold preprocessing refit enabled" in m for m in logs)
    metrics = result.node_results["node_training"].metrics
    assert _disc(metrics["best_score"]) < 0.65, (
        f"inner tuning score should sit near chance, got {metrics['best_score']}"
    )
    assert _disc(metrics["cv_roc_auc_mean"]) < 0.65, (
        f"outer CV score should sit near chance, got {metrics['cv_roc_auc_mean']}"
    )


def test_merged_branches_fall_back_with_warning(noise_target_csv, tmp_path):
    """A merge upstream of training disables the hook with an explicit warning."""
    result, logs = _run(
        tmp_path,
        [
            _loader("loader_a", noise_target_csv),
            _loader("loader_b", noise_target_csv),
            NodeConfig(
                node_id="node_features",
                step_type=StepType.FEATURE_ENGINEERING,
                inputs=["loader_a", "loader_b"],
                params={"steps": [SPLITTER_STEP, WOE_STEP]},
            ),
            _training(["node_features"]),
        ],
        job_id="f15-merge-fallback",
    )

    assert result.status == "success"
    assert any("Per-fold preprocessing refit skipped" in m and "linear chain" in m for m in logs)


def _splitter(node_id: str, inputs: list[str]) -> NodeConfig:
    return NodeConfig(
        node_id=node_id,
        step_type="TrainTestSplitter",
        inputs=inputs,
        params={"target_column": "target", "test_size": 0.2},
    )


def _woe_node(node_id: str, inputs: list[str], regularization: float) -> NodeConfig:
    return NodeConfig(
        node_id=node_id,
        step_type="WOEEncoder",
        inputs=inputs,
        params={"columns": ["city"], "regularization": regularization},
    )


def _fork_join_nodes(csv: str, training_params: dict | None = None) -> list[NodeConfig]:
    """Scenario-06 shape: loader -> splitter -> two WOE branches -> training."""
    nodes = [
        _loader("node_data", csv),
        _splitter("node_split", ["node_data"]),
        _woe_node("woe_a", ["node_split"], 0.5),
        _woe_node("woe_b", ["node_split"], 1.0),
        _training(["woe_a", "woe_b"], **(training_params or {})),
    ]
    return nodes


def test_fork_join_merged_branches_refit_per_fold(noise_target_csv, tmp_path):
    """Fork-join merge (last_wins): both WOE branches re-fit inside every fold,
    so the noise-target CV stays near chance instead of memorising."""
    result, logs = _run(tmp_path, _fork_join_nodes(noise_target_csv), job_id="f15-fork-join")

    assert result.status == "success"
    assert any(
        "Per-fold preprocessing refit enabled" in m and "merged branch(es)" in m for m in logs
    ), logs
    metrics = result.node_results["node_training"].metrics
    assert _disc(metrics["cv_roc_auc_mean"]) < 0.65, (
        f"fork-join refit CV should sit near chance, got {metrics['cv_roc_auc_mean']}"
    )


def test_fork_join_first_wins_refit_per_fold(noise_target_csv, tmp_path):
    """The first_wins merge strategy gets the same per-fold honesty guarantee."""
    nodes = _fork_join_nodes(noise_target_csv, training_params={"_merge_strategy": "first_wins"})
    result, logs = _run(tmp_path, nodes, job_id="f15-fork-join-first-wins")

    assert result.status == "success"
    assert any("merged branch(es)" in m for m in logs)
    metrics = result.node_results["node_training"].metrics
    assert _disc(metrics["cv_roc_auc_mean"]) < 0.65


def test_nested_merge_falls_back_with_warning(noise_target_csv, tmp_path):
    """A merge node nested inside a branch is outside fork-join scope: warn + skip."""
    result, logs = _run(
        tmp_path,
        [
            _loader("node_data", noise_target_csv),
            _splitter("node_split", ["node_data"]),
            _woe_node("woe_a", ["node_split"], 0.5),
            _woe_node("woe_b", ["node_split"], 1.0),
            NodeConfig(
                node_id="node_inner_merge",
                step_type=StepType.FEATURE_ENGINEERING,
                inputs=["woe_a", "woe_b"],
                params={"steps": []},
            ),
            _training(["node_inner_merge", "woe_b"]),
        ],
        job_id="f15-nested-merge-fallback",
    )

    assert result.status == "success"
    assert any(
        "Per-fold preprocessing refit skipped" in m and "linear transformer chain" in m
        for m in logs
    ), logs


def test_row_count_changing_branch_falls_back_with_warning(noise_target_csv, tmp_path):
    """A branch containing a row-count-changing step cannot run fold-wise:
    warn + skip (the branch itself keeps working in the full run)."""
    result, logs = _run(
        tmp_path,
        [
            _loader("node_data", noise_target_csv),
            _splitter("node_split", ["node_data"]),
            _woe_node("woe_a", ["node_split"], 0.5),
            NodeConfig(
                node_id="woe_b_drop",
                step_type=StepType.FEATURE_ENGINEERING,
                inputs=["node_split"],
                params={
                    "steps": [
                        {
                            "name": "woe_b",
                            "transformer": "WOEEncoder",
                            "params": {"columns": ["city"], "regularization": 1.0},
                        },
                        {
                            "name": "drop_rows",
                            "transformer": "DropMissingRows",
                            "params": {"columns": ["city"]},
                        },
                    ]
                },
            ),
            _training(["woe_a", "woe_b_drop"]),
        ],
        job_id="f15-row-count-fallback",
    )

    assert result.status == "success"
    assert any("Per-fold preprocessing refit skipped" in m and "row counts" in m for m in logs), (
        logs
    )


def test_learning_step_after_splitter_falls_back_with_warning(noise_target_csv, tmp_path):
    """A trunk node whose last step is not a splitter (splitter mid-chain) is
    outside fork-join scope: warn + skip."""
    result, logs = _run(
        tmp_path,
        [
            _loader("node_data", noise_target_csv),
            NodeConfig(
                node_id="node_trunk",
                step_type=StepType.FEATURE_ENGINEERING,
                inputs=["node_data"],
                params={"steps": [SPLITTER_STEP, WOE_STEP]},  # splitter NOT last
            ),
            _woe_node("woe_a", ["node_trunk"], 0.5),
            _woe_node("woe_b", ["node_trunk"], 1.0),
            _training(["woe_a", "woe_b"]),
        ],
        job_id="f15-mid-chain-splitter-fallback",
    )

    assert result.status == "success"
    assert any(
        "Per-fold preprocessing refit skipped" in m and "must end with" in m for m in logs
    ), logs


def test_leakage_dominated_contrast_end_to_end(noise_target_csv, tmp_path):
    """The leakage-dominated case, measured both ways on the same CSV.

    Leaky (pre-F-15 scoring, still what every fallback path reports): WOE is
    fitted once on the full training split, then CV scores the transformed
    data — every fold's validation rows shaped the encoding, so the noise
    target is memorised and AUC inflates far above chance (measured ~0.87).

    Leak-free (the app's always-on refit path): the identical pipeline
    through the engine re-fits WOE inside every fold and sits at chance.
    """
    from sklearn.model_selection import train_test_split

    from skyulf.modeling.classification import (
        LogisticRegressionApplier,
        LogisticRegressionCalculator,
    )
    from skyulf.modeling.cross_validation import perform_cross_validation
    from skyulf.preprocessing.encoding import WOEEncoderApplier, WOEEncoderCalculator

    # --- Leaky control: the pre-F-15 scoring discipline on the same CSV. ---
    frame = pd.read_csv(noise_target_csv)
    X = frame.drop(columns=["target"])
    y = frame["target"]
    X_train, _X_test, y_train, _y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    leaky_params = WOEEncoderCalculator().fit((X_train, y_train), WOE_STEP["params"])
    X_leaky, _ = WOEEncoderApplier().apply((X_train, y_train), dict(leaky_params))
    leaky_cv = perform_cross_validation(
        LogisticRegressionCalculator(),
        LogisticRegressionApplier(),
        X_leaky,
        y_train,
        config={},
        n_folds=5,
        cv_type="k_fold",
    )
    auc_leaky = leaky_cv["aggregated_metrics"]["roc_auc"]["mean"]
    assert _disc(auc_leaky) > 0.75, (
        f"legacy full-fit scoring should memorise the noise target, got {auc_leaky}"
    )

    # --- Leak-free: the identical pipeline through the app engine. ---
    result, logs = _run(
        tmp_path,
        [
            _loader("node_data", noise_target_csv),
            NodeConfig(
                node_id="node_features",
                step_type=StepType.FEATURE_ENGINEERING,
                inputs=["node_data"],
                params={"steps": [SPLITTER_STEP, WOE_STEP]},
            ),
            _training(["node_features"]),
        ],
        job_id="f15-leakage-contrast",
    )

    assert result.status == "success"
    assert any("Per-fold preprocessing refit enabled" in m for m in logs)
    auc_refit = result.node_results["node_training"].metrics["cv_roc_auc_mean"]
    assert _disc(auc_refit) < 0.65, f"per-fold refit CV should sit near chance, got {auc_refit}"
    assert _disc(auc_leaky) - _disc(auc_refit) > 0.10, (
        f"expected a large leak gap, got leaky={auc_leaky} vs refit={auc_refit}"
    )


def test_halving_strategy_refits_per_fold(noise_target_csv, tmp_path):
    """halving_grid runs CV inside the sklearn searcher; the Pipeline wrapper
    must refit WOE per fold there too, so the noise-target score stays honest."""
    result, logs = _run(
        tmp_path,
        [
            _loader("node_data", noise_target_csv),
            NodeConfig(
                node_id="node_features",
                step_type=StepType.FEATURE_ENGINEERING,
                inputs=["node_data"],
                params={"steps": [SPLITTER_STEP, WOE_STEP]},
            ),
            _training(
                ["node_features"],
                run_mode="tuned",
                tuning_config={
                    "strategy": "halving_grid",
                    "metric": "roc_auc",
                    "cv_folds": 2,
                    "cv_enabled": True,
                    "search_space": {"C": [0.1, 1.0]},
                },
            ),
        ],
        job_id="f15-halving-refit",
    )

    assert result.status == "success"
    assert any("Per-fold preprocessing refit enabled" in m for m in logs)
    metrics = result.node_results["node_training"].metrics
    assert _disc(metrics["best_score"]) < 0.65, (
        f"inner halving tuning score should sit near chance, got {metrics['best_score']}"
    )
    assert _disc(metrics["cv_roc_auc_mean"]) < 0.65, (
        f"outer CV score should sit near chance, got {metrics['cv_roc_auc_mean']}"
    )


def test_halving_with_nan_and_imputer_runs_refit(tmp_path):
    """NaN-bearing features + imputer step through the halving Pipeline wrapper:
    the NaN gate must let the pre-transform payload through and the run must
    complete with refit enabled."""
    rng = np.random.default_rng(3)
    n = 300
    num1 = rng.normal(0, 1, n)
    num2 = rng.normal(0, 1, n)
    num2[rng.random(n) < 0.15] = np.nan
    df = pd.DataFrame(
        {
            "num1": num1,
            "num2": num2,
            "city": [f"c{v}" for v in rng.integers(0, 8, size=n)],
            "target": rng.integers(0, 2, size=n),
        }
    )
    path = tmp_path / "nan_features.csv"
    df.to_csv(path, index=False)

    result, logs = _run(
        tmp_path,
        [
            _loader("node_data", str(path)),
            NodeConfig(
                node_id="node_features",
                step_type=StepType.FEATURE_ENGINEERING,
                inputs=["node_data"],
                params={
                    "steps": [
                        SPLITTER_STEP,
                        {
                            "name": "impute_num2",
                            "transformer": "SimpleImputer",
                            "params": {"strategy": "mean", "columns": ["num2"]},
                        },
                        WOE_STEP,
                    ]
                },
            ),
            _training(
                ["node_features"],
                run_mode="tuned",
                tuning_config={
                    "strategy": "halving_grid",
                    "metric": "roc_auc",
                    "cv_folds": 2,
                    "cv_enabled": True,
                    "search_space": {"C": [1.0]},
                },
            ),
        ],
        job_id="f15-halving-nan-imputer",
    )

    assert result.status == "success"
    assert any("Per-fold preprocessing refit enabled" in m for m in logs)
    metrics = result.node_results["node_training"].metrics
    assert np.isfinite(metrics["best_score"])


def test_two_independent_pipelines_both_refit(noise_target_csv, tmp_path):
    """One job, two disjoint loader→FE→training pipelines: each training node
    resolves its own chain and stays honest independently."""
    result, logs = _run(
        tmp_path,
        [
            _loader("loader_a", noise_target_csv),
            _loader("loader_b", noise_target_csv),
            NodeConfig(
                node_id="features_a",
                step_type=StepType.FEATURE_ENGINEERING,
                inputs=["loader_a"],
                params={"steps": [SPLITTER_STEP, WOE_STEP]},
            ),
            NodeConfig(
                node_id="features_b",
                step_type=StepType.FEATURE_ENGINEERING,
                inputs=["loader_b"],
                params={"steps": [SPLITTER_STEP, WOE_STEP]},
            ),
            _training(
                ["features_a"],
                node_id="training_a",
                run_mode="tuned",
                tuning_config={
                    "strategy": "grid",
                    "metric": "roc_auc",
                    "cv_folds": 2,
                    "cv_enabled": True,
                    "search_space": {"C": [1.0]},
                },
            ),
            _training(
                ["features_b"],
                node_id="training_b",
                run_mode="tuned",
                tuning_config={
                    "strategy": "grid",
                    "metric": "roc_auc",
                    "cv_folds": 2,
                    "cv_enabled": True,
                    "search_space": {"C": [1.0]},
                },
            ),
        ],
        job_id="f15-two-pipelines",
    )

    assert result.status == "success"
    enabled = [m for m in logs if "Per-fold preprocessing refit enabled" in m]
    assert len(enabled) >= 2, f"expected refit enabled for both branches, logs={logs}"
    for node_id in ("training_a", "training_b"):
        metrics = result.node_results[node_id].metrics
        assert _disc(metrics["best_score"]) < 0.65, (
            f"{node_id} tuning score should sit near chance, got {metrics['best_score']}"
        )
        assert _disc(metrics["cv_roc_auc_mean"]) < 0.65, (
            f"{node_id} CV score should sit near chance, got {metrics['cv_roc_auc_mean']}"
        )


def test_splitter_only_chain_skips_silently(tmp_path):
    """Only a splitter upstream: nothing data-dependent to refit, no warning needed."""
    rng = np.random.default_rng(0)
    df = pd.DataFrame(
        {
            "f1": rng.uniform(0, 10, size=120),
            "f2": rng.uniform(0, 10, size=120),
            "target": rng.integers(0, 2, size=120),
        }
    )
    path = tmp_path / "numeric.csv"
    df.to_csv(path, index=False)

    result, logs = _run(
        tmp_path,
        [
            _loader("node_data", str(path)),
            NodeConfig(
                node_id="node_split",
                step_type="TrainTestSplitter",
                inputs=["node_data"],
                params={"target_column": "target", "test_size": 0.2},
            ),
            _training(["node_split"]),
        ],
        job_id="f15-splitter-only",
    )

    assert result.status == "success"
    assert not any("Per-fold preprocessing refit" in m for m in logs)
