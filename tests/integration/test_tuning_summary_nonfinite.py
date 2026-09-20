"""Distinguish unavailable tuning scores from successful finite objectives."""

import pandas as pd
import pytest

from backend.data.catalog import FileSystemCatalog
from backend.ml_pipeline._execution.engine import PipelineEngine
from backend.ml_pipeline._execution.schemas import NodeConfig, PipelineConfig
from backend.ml_pipeline._execution.summary import build_summary
from backend.ml_pipeline.artifacts.local import LocalArtifactStore


@pytest.mark.parametrize("score", [float("nan"), float("inf"), -float("inf"), True, False, None])
def test_invalid_tuning_score_is_explicitly_unavailable(score):
    """An invalid objective must not become a numeric or fallback success headline."""
    summary = build_summary(
        step_type="training",
        output=None,
        metrics={
            "best_score": score,
            "scoring_metric": "neg_mean_squared_error",
            "trials": 3,
            "test_accuracy": 0.9,
        },
        params={"run_mode": "tuned"},
    )
    assert summary == "Tuning score unavailable · 3 trials"


@pytest.mark.parametrize("score,display", [(-12.34, "12.340"), (0.0, "0.000")])
def test_finite_negative_loss_still_displays_natural_magnitude(score, display):
    """Valid loss scores must retain sign conversion and zero without coercing bools."""
    summary = build_summary(
        step_type="training",
        output=None,
        metrics={"best_score": score, "scoring_metric": "neg_mean_squared_error", "trials": 2},
        params={"run_mode": "tuned"},
    )
    assert summary == f"mse {display} · 2 trials"


def test_all_failed_trials_leave_training_node_failed(tmp_path, monkeypatch):
    """Actual failed CV candidates must produce a failed node rather than a success summary."""
    from backend.config import get_settings

    monkeypatch.setattr(get_settings(), "TUNING_N_JOBS", 1)
    path = tmp_path / "data.csv"
    pd.DataFrame({"x": range(4), "target": range(4)}).to_csv(path, index=False)
    engine = PipelineEngine(LocalArtifactStore(str(tmp_path / "artifacts")), FileSystemCatalog())
    result = engine.run(
        PipelineConfig(
            pipeline_id="all-failed",
            nodes=[
                NodeConfig("data", "data_loader", params={"source": "csv", "path": str(path)}),
                NodeConfig(
                    "model",
                    "training",
                    inputs=["data"],
                    params={
                        "run_mode": "tuned",
                        "algorithm": "ridge_regression",
                        "target_column": "target",
                        "tuning_config": {
                            "strategy": "grid",
                            "metric": "r2",
                            "cv_folds": 4,
                            "search_space": {"alpha": [1.0]},
                        },
                    },
                ),
            ],
        )
    )
    assert result.status == "failed"
    assert result.node_results["model"].status == "failed"
    assert "All trials failed" in result.node_results["model"].error
    assert not result.node_results["model"].metadata.get("summary")
