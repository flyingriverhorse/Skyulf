"""F-15 backend wiring: always-on per-fold preprocessing refit at engine level.

The app threads a ``FeatureEngineerFoldAdapter`` + pre-transform payloads
into every tuning/CV run so preprocessing statistics never see held-out fold
rows — all tuning strategies included (halving/optuna get it via a Pipeline
wrapper around the searcher's internal CV; holdout tuning with a validation
split refits on the train rows only and scores the untouched validation
split). Graphs the hook cannot serve (non-linear/unsupported upstream shapes,
failed payload reconstruction) fall back to pre-transformed scoring with an
explicit job-log warning — never a failed run, never a silent leak.
"""

import numpy as np
import pandas as pd
import pytest

from backend.data.catalog import FileSystemCatalog
from backend.ml_pipeline._execution.engine import PipelineEngine
from backend.ml_pipeline._execution.engine._feature_eng import FeatureEngMixin
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


def _numeric_csv(tmp_path) -> str:
    rng = np.random.default_rng(11)
    n = 400
    df = pd.DataFrame(
        {
            "id_col": [f"id{i}" for i in range(n)],
            "f1": rng.normal(0, 1, n),
            "f2": rng.normal(0, 1, n),
            "target": rng.integers(0, 2, size=n),
        }
    )
    path = tmp_path / "fork_join_numeric.csv"
    df.to_csv(path, index=False)
    return str(path)


def _scaler_node(node_id: str, inputs: list[str], columns: list[str]) -> NodeConfig:
    return NodeConfig(
        node_id=node_id,
        step_type="StandardScaler",
        inputs=inputs,
        params={"columns": columns},
    )


def test_learning_trunk_step_before_fork_splitter_falls_back(tmp_path):
    """A data-dependent trunk step before the fork splitter learned from the
    full trunk frame (held-out rows included) — fork-join falls back with a
    warning, exactly like the linear path does."""
    csv = _numeric_csv(tmp_path)
    result, logs = _run(
        tmp_path,
        [
            _loader("node_data", csv),
            NodeConfig(
                node_id="node_trunk",
                step_type=StepType.FEATURE_ENGINEERING,
                inputs=["node_data"],
                params={
                    "steps": [
                        {
                            "name": "drop_id",
                            "transformer": "DropMissingColumns",
                            "params": {"columns": ["id_col"]},
                        },
                        {
                            "name": "scale_f1",
                            "transformer": "StandardScaler",
                            "params": {"columns": ["f1"]},
                        },
                        SPLITTER_STEP,
                    ]
                },
            ),
            _scaler_node("scale_a", ["node_trunk"], ["f2"]),
            _scaler_node("scale_b", ["node_trunk"], ["f2"]),
            _training(["scale_a", "scale_b"]),
        ],
        job_id="f15-trunk-learner-fallback",
    )

    assert result.status == "success"
    assert any(
        "Per-fold preprocessing refit skipped" in m and "before the fork splitter" in m
        for m in logs
    ), logs


def test_stateless_trunk_step_before_fork_splitter_keeps_refit(tmp_path):
    """A stateless trunk step (explicit column drop) before the fork splitter
    is already applied by the fork artifact once — fork-join stays enabled."""
    csv = _numeric_csv(tmp_path)
    result, logs = _run(
        tmp_path,
        [
            _loader("node_data", csv),
            NodeConfig(
                node_id="node_trunk",
                step_type=StepType.FEATURE_ENGINEERING,
                inputs=["node_data"],
                params={
                    "steps": [
                        {
                            "name": "drop_id",
                            "transformer": "DropMissingColumns",
                            "params": {"columns": ["id_col"]},
                        },
                        SPLITTER_STEP,
                    ]
                },
            ),
            _scaler_node("scale_a", ["node_trunk"], ["f1"]),
            _scaler_node("scale_b", ["node_trunk"], ["f2"]),
            _training(["scale_a", "scale_b"]),
        ],
        job_id="f15-trunk-stateless-refit",
    )

    assert result.status == "success"
    assert any(
        "Per-fold preprocessing refit enabled" in m and "merged branch(es)" in m for m in logs
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


def test_learning_step_before_splitter_falls_back_with_warning(tmp_path):
    """A data-dependent step configured BEFORE the splitter cannot be re-fit
    per fold: payload reconstruction would fit it on the full frame (leaking
    held-out rows) and the per-fold adapter would apply it twice. The resolver
    must fall back to pre-transformed scoring with an explicit warning."""
    rng = np.random.default_rng(11)
    n = 240
    df = pd.DataFrame(
        {
            "f1": rng.normal(0, 1, n),
            "f2": rng.normal(0, 1, n),
            "target": rng.integers(0, 2, size=n),
        }
    )
    path = tmp_path / "numeric_pre_split.csv"
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
                        {"name": "scale", "transformer": "StandardScaler", "params": {}},
                        SPLITTER_STEP,
                    ]
                },
            ),
            _training(["node_features"]),
        ],
        job_id="f15-pre-split-learner-fallback",
    )

    assert result.status == "success"
    assert any(
        "Per-fold preprocessing refit skipped" in m and "before the last splitter" in m
        for m in logs
    ), f"expected the pre-splitter fallback warning, logs={logs}"
    assert not any("Per-fold preprocessing refit enabled" in m for m in logs)


def test_stateless_step_before_splitter_keeps_refit_enabled(tmp_path):
    """Param-aware exemption: an explicit-column DropMissingColumns before the
    splitter learns nothing from the rows, so per-fold refit stays enabled."""
    rng = np.random.default_rng(7)
    n, n_categories = 400, 200
    df = pd.DataFrame(
        {
            "id_col": [f"id{i}" for i in range(n)],
            "city": [f"c{v}" for v in rng.integers(0, n_categories, size=n)],
            "target": rng.integers(0, 2, size=n),
        }
    )
    path = tmp_path / "noise_with_id.csv"
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
                        {
                            "name": "drop_id",
                            "transformer": "DropMissingColumns",
                            "params": {"columns": ["id_col"]},
                        },
                        SPLITTER_STEP,
                        WOE_STEP,
                    ]
                },
            ),
            _training(["node_features"]),
        ],
        job_id="f15-stateless-pre-split",
    )

    assert result.status == "success"
    enabled = [m for m in logs if "Per-fold preprocessing refit enabled" in m]
    assert enabled
    # The stateless pre-split step was already applied during payload
    # reconstruction; only the post-split WOE step re-fits per fold.
    assert "1 step(s)" in enabled[0]
    metrics = result.node_results["node_training"].metrics
    assert _disc(metrics["cv_roc_auc_mean"]) < 0.65, (
        f"per-fold refit CV should sit near chance, got {metrics['cv_roc_auc_mean']}"
    )


# ---------------------------------------------------------------------------
# _step_learns_from_data — the resolver's per-step verdict
# ---------------------------------------------------------------------------


def test_step_learns_from_data_mirrors_the_leakage_gate():
    from backend.ml_pipeline._execution.engine._feature_eng import _step_learns_from_data

    # Param-aware stateless exemptions.
    assert not _step_learns_from_data(
        {"transformer": "SimpleImputer", "params": {"strategy": "constant"}}
    )
    assert not _step_learns_from_data(
        {"transformer": "MissingIndicator", "params": {"columns": ["x"]}}
    )
    assert not _step_learns_from_data({"transformer": "HashEncoder", "params": {"columns": ["x"]}})
    assert not _step_learns_from_data(
        {"transformer": "DropMissingColumns", "params": {"columns": ["id"]}}
    )
    # Data-dependent modes of the same nodes, and plain learners.
    assert _step_learns_from_data({"transformer": "SimpleImputer", "params": {"strategy": "mean"}})
    assert _step_learns_from_data(
        {"transformer": "DropMissingColumns", "params": {"missing_threshold": 50}}
    )
    assert _step_learns_from_data({"transformer": "StandardScaler", "params": {}})
    # Registered but stateless node.
    assert not _step_learns_from_data({"transformer": "DropMissingRows", "params": {}})
    # Unknown transformers fail closed.
    assert _step_learns_from_data({"transformer": "NoSuchNode", "params": {}})


# ---------------------------------------------------------------------------
# Remaining resolver branches: no-split payload, reconstruction failure,
# validation-split holdout tuning
# ---------------------------------------------------------------------------


def test_no_split_chain_refits_from_the_raw_loader_frame(tmp_path):
    """No splitter upstream: the raw loader frame is the pre-transform payload
    and the learning step refits per fold."""
    rng = np.random.default_rng(13)
    n = 240
    df = pd.DataFrame(
        {
            "f1": rng.normal(0, 1, n),
            "f2": rng.normal(0, 1, n),
            "target": rng.integers(0, 2, size=n),
        }
    )
    path = tmp_path / "no_split.csv"
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
                    "steps": [{"name": "scale", "transformer": "StandardScaler", "params": {}}]
                },
            ),
            _training(["node_features"]),
        ],
        job_id="f15-no-split-payload",
    )

    assert result.status == "success"
    assert any("Per-fold preprocessing refit enabled" in m for m in logs)


def test_payload_reconstruction_failure_never_fails_the_run(
    noise_target_csv, tmp_path, monkeypatch
):
    """If payload reconstruction raises, the resolver must swallow the error,
    warn explicitly, and let the job finish on pre-transformed scoring."""
    from backend.ml_pipeline._execution.engine._feature_eng import FeatureEngMixin

    def boom(self, output, target_col):
        raise RuntimeError("simulated artifact corruption")

    monkeypatch.setattr(FeatureEngMixin, "_split_train_payload", boom)

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
        job_id="f15-reconstruction-failure",
    )

    assert result.status == "success"
    assert any(
        "Per-fold preprocessing refit skipped" in m and "payload reconstruction failed" in m
        for m in logs
    ), f"expected the reconstruction-failure warning, logs={logs}"


def test_validation_split_tuning_refits_per_fold(noise_target_csv, tmp_path):
    """Holdout tuning with a validation split gets the per-fold discipline:
    WOE refits on the train rows only, candidates score against the untouched
    validation split, and the post-tuning CV refits too — a memorising WOE on
    a noise target must stay near chance in both scores."""
    result, logs = _run(
        tmp_path,
        [
            _loader("node_data", noise_target_csv),
            NodeConfig(
                node_id="node_split",
                step_type="TrainTestSplitter",
                inputs=["node_data"],
                params={
                    "target_column": "target",
                    "test_size": 0.2,
                    "validation_size": 0.2,
                },
            ),
            NodeConfig(
                node_id="node_features",
                step_type=StepType.FEATURE_ENGINEERING,
                inputs=["node_split"],
                params={"steps": [WOE_STEP]},
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
        job_id="f15-validation-split-refit",
    )

    assert result.status == "success"
    assert any("Per-fold preprocessing refit enabled" in m for m in logs), logs
    assert not any("Per-fold preprocessing refit skipped" in m for m in logs), logs
    metrics = result.node_results["node_training"].metrics
    assert _disc(metrics["best_score"]) < 0.65, (
        f"holdout tuning score should sit near chance, got {metrics['best_score']}"
    )
    assert _disc(metrics["cv_roc_auc_mean"]) < 0.65, (
        f"post-tuning CV score should sit near chance, got {metrics['cv_roc_auc_mean']}"
    )


# ---------------------------------------------------------------------------
# Holdout tuning + validation split + detailed preprocessing — dedicated
# end-to-end coverage of the v0.8.4 holdout refit at the app-engine level
# ---------------------------------------------------------------------------


IMPUTE_STEP = {
    "name": "impute_num2",
    "transformer": "SimpleImputer",
    "params": {"strategy": "mean", "columns": ["num2"]},
}
SCALE_ALL_STEP = {
    "name": "scale_num",
    "transformer": "StandardScaler",
    "params": {"columns": ["num1", "num2"]},
}
IMPUTE_F2_STEP = {
    "name": "impute_f2",
    "transformer": "SimpleImputer",
    "params": {"strategy": "mean", "columns": ["f2"]},
}
WOE_CAT_STEP = {
    "name": "woe_cat",
    "transformer": "WOEEncoder",
    "params": {"columns": ["cat"], "regularization": 0.5},
}
SCALE_SIGNAL_STEP = {
    "name": "scale_signal",
    "transformer": "StandardScaler",
    "params": {"columns": ["f1", "f2"]},
}


def _val_auc(metrics: dict) -> float:
    """The validation-split ROC-AUC, whatever the binary/multiclass key name."""
    for key, value in metrics.items():
        if key.startswith("val_roc_auc"):
            return float(value)
    raise AssertionError(f"no val_roc_auc* metric found in {sorted(metrics)}")


def _noise_csv_with_nan(tmp_path) -> str:
    """Noise target + NaN numeric column + memorising categorical column."""
    rng = np.random.default_rng(42)
    n, n_categories = 400, 200
    num1 = rng.normal(0, 1, n)
    num2 = rng.normal(0, 1, n)
    num2[rng.random(n) < 0.15] = np.nan
    df = pd.DataFrame(
        {
            "num1": num1,
            "num2": num2,
            "city": [f"c{v}" for v in rng.integers(0, n_categories, size=n)],
            "target": rng.integers(0, 2, size=n),
        }
    )
    path = tmp_path / "noise_nan.csv"
    df.to_csv(path, index=False)
    return str(path)


def _signal_csv(tmp_path, n: int = 400) -> str:
    """Informative numerics (f2 carries NaNs) + informative categorical +
    mild label noise."""
    rng = np.random.default_rng(5)
    y = rng.integers(0, 2, size=n)
    flip = rng.random(n) < 0.05
    y = np.where(flip, 1 - y, y)
    f2 = rng.normal(0, 1, n) - y * 1.0
    f2[rng.random(n) < 0.15] = np.nan
    df = pd.DataFrame(
        {
            "f1": rng.normal(0, 1, n) + y * 1.5,
            "f2": f2,
            "cat": [f"k{v}" for v in np.where(rng.random(n) < 0.8, y, 1 - y)],
            "target": y,
        }
    )
    path = tmp_path / "signal.csv"
    df.to_csv(path, index=False)
    return str(path)


def _splitter_with_validation(node_id: str, inputs: list[str]) -> NodeConfig:
    return NodeConfig(
        node_id=node_id,
        step_type="TrainTestSplitter",
        inputs=inputs,
        params={"target_column": "target", "test_size": 0.2, "validation_size": 0.2},
    )


def _holdout_tuning(**extra) -> dict:
    tuning = {
        "metric": "roc_auc",
        "cv_folds": 3,
        "cv_enabled": True,
        "search_space": {"C": [0.1, 1.0]},
    }
    tuning.update(extra)
    return tuning


def test_validation_split_multi_step_chain_refits_honest(tmp_path):
    """Imputer + memorising WOE + scaler + 3-way split, tuned holdout: the
    NaN gate passes the pre-transform payload, the chain refits on train rows
    only, and both scores stay near chance on the noise target."""
    csv = _noise_csv_with_nan(tmp_path)
    result, logs = _run(
        tmp_path,
        [
            _loader("node_data", csv),
            NodeConfig(
                node_id="node_features",
                step_type=StepType.FEATURE_ENGINEERING,
                inputs=["node_data"],
                params={
                    "steps": [
                        _splitter_with_validation_params(),
                        IMPUTE_STEP,
                        WOE_STEP,
                        SCALE_ALL_STEP,
                    ]
                },
            ),
            _training(
                ["node_features"],
                run_mode="tuned",
                tuning_config=_holdout_tuning(strategy="grid"),
            ),
        ],
        job_id="f15-val-multistep-noise",
    )

    assert result.status == "success", logs
    assert any("Per-fold preprocessing refit enabled" in m for m in logs), logs
    assert not any("Per-fold preprocessing refit skipped" in m for m in logs), logs
    metrics = result.node_results["node_training"].metrics
    assert _disc(metrics["best_score"]) < 0.65, (
        f"holdout tuning score should sit near chance, got {metrics['best_score']}"
    )
    assert _disc(metrics["cv_roc_auc_mean"]) < 0.65, (
        f"post-tuning CV score should sit near chance, got {metrics['cv_roc_auc_mean']}"
    )


def _splitter_with_validation_params() -> dict:
    return {
        "name": "split",
        "transformer": "TrainTestSplitter",
        "params": {"target_column": "target", "test_size": 0.2, "validation_size": 0.2},
    }


def test_validation_split_signal_run_keeps_honest_scores(tmp_path):
    """Optuna (wrapped path) + imputer/WOE/scaler chain + validation split on
    real signal: honest scores stay well above chance and the untouched
    validation split is evaluated alongside the test split."""
    csv = _signal_csv(tmp_path)
    result, logs = _run(
        tmp_path,
        [
            _loader("node_data", csv),
            _splitter_with_validation("node_split", ["node_data"]),
            NodeConfig(
                node_id="node_features",
                step_type=StepType.FEATURE_ENGINEERING,
                inputs=["node_split"],
                params={"steps": [IMPUTE_F2_STEP, WOE_CAT_STEP, SCALE_SIGNAL_STEP]},
            ),
            _training(
                ["node_features"],
                run_mode="tuned",
                tuning_config=_holdout_tuning(strategy="optuna", n_trials=3),
            ),
        ],
        job_id="f15-val-signal-optuna",
    )

    assert result.status == "success", logs
    assert any("Per-fold preprocessing refit enabled" in m for m in logs), logs
    assert not any("Per-fold preprocessing refit skipped" in m for m in logs), logs
    metrics = result.node_results["node_training"].metrics
    assert metrics["best_score"] > 0.8, (
        f"signal holdout tuning score should stay meaningful, got {metrics['best_score']}"
    )
    assert metrics["cv_roc_auc_mean"] > 0.8, (
        f"signal post-tuning CV should stay meaningful, got {metrics['cv_roc_auc_mean']}"
    )
    assert _val_auc(metrics) > 0.8, (
        f"untouched validation split should stay meaningful, got {_val_auc(metrics)}"
    )


def test_validation_split_fork_join_refits_honest(tmp_path):
    """Fork-join merged graph + validation split: both branches re-run on the
    fold-train rows and the memorising WOE column stays near chance.

    The merged-branch adapter merges with pure ``last_wins`` (ownership is
    inert for SplitDataset ancestors), so every branch must itself emit a
    fully-numeric frame — hence each branch imputes, WOE-encodes ``city`` and
    scales.
    """
    csv = _noise_csv_with_nan(tmp_path)

    def _woe_fe_branch(node_id: str, regularization: float) -> NodeConfig:
        return NodeConfig(
            node_id=node_id,
            step_type=StepType.FEATURE_ENGINEERING,
            inputs=["node_split"],
            params={
                "steps": [
                    IMPUTE_STEP,
                    {
                        "name": "woe_city",
                        "transformer": "WOEEncoder",
                        "params": {"columns": ["city"], "regularization": regularization},
                    },
                    SCALE_ALL_STEP,
                ]
            },
        )

    result, logs = _run(
        tmp_path,
        [
            _loader("node_data", csv),
            _splitter_with_validation("node_split", ["node_data"]),
            _woe_fe_branch("woe_branch_a", 0.5),
            _woe_fe_branch("woe_branch_b", 1.0),
            _training(
                ["woe_branch_a", "woe_branch_b"],
                run_mode="tuned",
                tuning_config=_holdout_tuning(strategy="grid"),
            ),
        ],
        job_id="f15-val-fork-join",
    )

    assert result.status == "success", logs
    assert any(
        "Per-fold preprocessing refit enabled" in m and "merged branch(es)" in m for m in logs
    ), logs
    metrics = result.node_results["node_training"].metrics
    assert _disc(metrics["best_score"]) < 0.65, (
        f"fork-join holdout score should sit near chance, got {metrics['best_score']}"
    )
    assert _disc(metrics["cv_roc_auc_mean"]) < 0.65, (
        f"fork-join post-tuning CV should sit near chance, got {metrics['cv_roc_auc_mean']}"
    )


def test_validation_split_fork_join_signal_keeps_scores(tmp_path):
    """Fork-join on signal data with a validation split: the merged-branch
    refit keeps every score meaningful (no silent degradation)."""
    csv = _signal_csv(tmp_path)
    result, logs = _run(
        tmp_path,
        [
            _loader("node_data", csv),
            _splitter_with_validation("node_split", ["node_data"]),
            NodeConfig(
                node_id="scale_branch",
                step_type=StepType.FEATURE_ENGINEERING,
                inputs=["node_split"],
                params={"steps": [IMPUTE_F2_STEP, SCALE_SIGNAL_STEP]},
            ),
            NodeConfig(
                node_id="encode_branch",
                step_type=StepType.FEATURE_ENGINEERING,
                inputs=["node_split"],
                params={"steps": [IMPUTE_F2_STEP, WOE_CAT_STEP, SCALE_SIGNAL_STEP]},
            ),
            _training(
                ["scale_branch", "encode_branch"],
                run_mode="tuned",
                tuning_config=_holdout_tuning(strategy="grid"),
            ),
        ],
        job_id="f15-val-fork-join-signal",
    )

    assert result.status == "success", logs
    assert any("merged branch(es)" in m for m in logs), logs
    metrics = result.node_results["node_training"].metrics
    assert metrics["best_score"] > 0.8, (
        f"fork-join holdout score should stay meaningful, got {metrics['best_score']}"
    )
    assert metrics["cv_roc_auc_mean"] > 0.8, (
        f"fork-join post-tuning CV should stay meaningful, got {metrics['cv_roc_auc_mean']}"
    )


def test_validation_split_refit_never_fits_on_validation_rows(tmp_path, monkeypatch):
    """Row-isolation proof through the real resolver: with a validation
    split, the chain fits only on train rows across holdout tuning and the
    post-tuning CV — a validation row never enters a fit."""
    from skyulf.preprocessing.fold_adapter import FeatureEngineerFoldAdapter

    rng = np.random.default_rng(17)
    n = 360
    y = rng.integers(0, 2, size=n)
    df = pd.DataFrame(
        {
            "f1": rng.normal(0, 1, n) + y * 1.5,
            "f2": rng.normal(0, 1, n) - y * 1.0,
            "target": y,
        }
    )
    csv = tmp_path / "isolation.csv"
    df.to_csv(csv, index=False)

    # ``train_test_split`` preserves scattered original indexes and the wrapped
    # halving/optuna path rebuilds them, so absolute row positions are not a
    # reliable identity. The robust isolation invariant is row *count*: in
    # holdout tuning the concat frame holds ``n_train`` train rows plus the
    # validation rows, so any preprocessing fit that saw validation rows would
    # receive more than ``n_train`` rows.
    fit_sizes: list[int] = []
    transform_sizes: list[int] = []
    original_fit_transform = FeatureEngineerFoldAdapter.fit_transform
    original_transform = FeatureEngineerFoldAdapter.transform

    def spy_fit_transform(self, X, y):
        fit_sizes.append(len(X))
        return original_fit_transform(self, X, y)

    def spy_transform(self, X, y):
        transform_sizes.append(len(X))
        return original_transform(self, X, y)

    monkeypatch.setattr(FeatureEngineerFoldAdapter, "fit_transform", spy_fit_transform)
    monkeypatch.setattr(FeatureEngineerFoldAdapter, "transform", spy_transform)

    result, logs = _run(
        tmp_path,
        [
            _loader("node_data", str(csv)),
            _splitter_with_validation("node_split", ["node_data"]),
            NodeConfig(
                node_id="node_features",
                step_type=StepType.FEATURE_ENGINEERING,
                inputs=["node_split"],
                params={
                    "steps": [{"name": "scale", "transformer": "StandardScaler", "params": {}}]
                },
            ),
            _training(
                ["node_features"],
                run_mode="tuned",
                tuning_config=_holdout_tuning(strategy="grid"),
            ),
        ],
        job_id="f15-val-row-isolation",
    )

    assert result.status == "success", logs
    assert any("Per-fold preprocessing refit enabled" in m for m in logs), logs
    metrics = result.node_results["node_training"].metrics
    n_train = int(metrics["n_rows"])
    assert fit_sizes, "expected the adapter to receive fit_transform calls"
    assert max(fit_sizes) == n_train, (
        f"expected preprocessing to fit on exactly the train split ({n_train} rows), "
        f"saw fit sizes {sorted(set(fit_sizes))} — a fit larger than the train split "
        "means validation rows leaked into a fit"
    )
    assert transform_sizes, "expected validation rows to be transformed (scored) untouched"


# ---------------------------------------------------------------------------
# _upstream_fe_chain — graph-shape edge branches
# ---------------------------------------------------------------------------


def _chain_mixin(configs: list[NodeConfig]) -> FeatureEngMixin:
    mixin = object.__new__(FeatureEngMixin)
    mixin._node_configs = {cfg.node_id: cfg for cfg in configs}
    return mixin


def test_upstream_chain_rejects_multi_input_training_node():
    mixin = _chain_mixin([_loader("node_data", "x.csv")])
    assert mixin._upstream_fe_chain(_training(["a", "b"])) is None


def test_upstream_chain_rejects_missing_node_reference():
    fe = NodeConfig(
        node_id="node_features",
        step_type=StepType.FEATURE_ENGINEERING,
        inputs=["ghost"],
        params={"steps": []},
    )
    mixin = _chain_mixin([fe])
    assert mixin._upstream_fe_chain(_training(["node_features"])) is None


def test_upstream_chain_rejects_unsupported_node_in_between():
    first = _training(["node_data"], node_id="node_training_a")
    second = _training(["node_training_a"], node_id="node_training_b")
    mixin = _chain_mixin([_loader("node_data", "x.csv"), first])
    assert mixin._upstream_fe_chain(second) is None


def test_upstream_chain_rejects_chain_without_loader():
    fe = NodeConfig(
        node_id="node_features",
        step_type=StepType.FEATURE_ENGINEERING,
        inputs=[],
        params={"steps": []},
    )
    mixin = _chain_mixin([fe])
    assert mixin._upstream_fe_chain(_training(["node_features"])) is None


# ---------------------------------------------------------------------------
# _branch_chain_up_to_loader / _try_fork_join_refit — fork-join bail branches
# ---------------------------------------------------------------------------


def test_branch_chain_rejects_missing_upstream_node():
    branch = NodeConfig(
        node_id="branch_a",
        step_type="WOEEncoder",
        inputs=["ghost"],
        params={"columns": ["city"]},
    )
    mixin = _chain_mixin([branch])
    assert mixin._branch_chain_up_to_loader("branch_a") is None


def test_branch_chain_rejects_training_node_mid_branch():
    branch = NodeConfig(
        node_id="branch_a",
        step_type="WOEEncoder",
        inputs=["node_training"],
        params={"columns": ["city"]},
    )
    mixin = _chain_mixin([branch, _training([], node_id="node_training")])
    assert mixin._branch_chain_up_to_loader("branch_a") is None


def test_branch_chain_rejects_branch_without_loader():
    branch = NodeConfig(
        node_id="branch_a",
        step_type="WOEEncoder",
        inputs=[],
        params={"columns": ["city"]},
    )
    mixin = _chain_mixin([branch])
    assert mixin._branch_chain_up_to_loader("branch_a") is None


def _refit_mixin(configs: list[NodeConfig], merge_order: list[str]) -> FeatureEngMixin:
    mixin = _chain_mixin(configs)
    mixin._merge_input_order = lambda _node: list(merge_order)
    return mixin


def test_fork_join_rejects_fewer_than_two_merged_inputs():
    mixin = _refit_mixin([], ["only_one"])
    resolved, reason = mixin._try_fork_join_refit(_training(["only_one"]), "target")
    assert resolved is None
    assert "fewer than two merged inputs" in reason


def test_fork_join_rejects_divergent_loaders():
    mixin = _refit_mixin(
        [
            _loader("loader_a", "a.csv"),
            _loader("loader_b", "b.csv"),
            _splitter("split_a", ["loader_a"]),
            _splitter("split_b", ["loader_b"]),
        ],
        ["split_a", "split_b"],
    )
    resolved, reason = mixin._try_fork_join_refit(_training(["split_a", "split_b"]), "target")
    assert resolved is None
    assert "do not share one data loader" in reason


def test_fork_join_rejects_branch_with_no_steps_after_fork():
    # Branch "node_split" IS the fork point (nothing after it); branch "woe_a"
    # continues past it. The shared-trunk prefix scan exhausts the shorter
    # branch without diverging, and that branch's post-fork step list is empty.
    mixin = _refit_mixin(
        [
            _loader("node_data", "x.csv"),
            _splitter("node_split", ["node_data"]),
            _woe_node("woe_a", ["node_split"], 0.5),
        ],
        ["node_split", "woe_a"],
    )
    resolved, reason = mixin._try_fork_join_refit(_training(["node_split", "woe_a"]), "target")
    assert resolved is None
    assert "no steps after the fork point" in reason


def test_fork_join_rejects_empty_fork_node():
    # An empty steps list on the shared-trunk end means there is no splitter
    # to fork on — bail with the fork-point reason, not an IndexError.
    mixin = _refit_mixin(
        [
            _loader("node_data", "x.csv"),
            NodeConfig(
                node_id="node_relay",
                step_type=StepType.FEATURE_ENGINEERING,
                inputs=["node_data"],
                params={"steps": []},
            ),
            _woe_node("woe_a", ["node_relay"], 0.5),
            _woe_node("woe_b", ["node_relay"], 1.0),
        ],
        ["woe_a", "woe_b"],
    )
    resolved, reason = mixin._try_fork_join_refit(_training(["woe_a", "woe_b"]), "target")
    assert resolved is None
    assert "no splitter fork point" in reason
