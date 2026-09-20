"""Pin Canvas model resolution and explicit unsupported-model export failures."""

import pytest

from backend.ml_pipeline._internal._routers._notebook_builders import (
    _canvas_modeling_block,
    _NodeIn,
)
from skyulf.pipeline import SkyulfPipeline


@pytest.mark.parametrize(
    ("algorithm", "task", "expected"),
    [
        ("random_forest", "regression", "random_forest_regressor"),
        ("RandomForestClassifier", "classification", "random_forest_classifier"),
        ("ridge", "regression", "ridge_regression"),
    ],
)
def test_canvas_export_resolves_runtime_model_aliases(algorithm, task, expected):
    """Saved Core configurations must resolve the same model as live Canvas training."""
    node = _NodeIn(
        node_id="model",
        step_type="training",
        params={"model_type": algorithm, "task_type": task, "hyperparameters": {}},
    )
    config = _canvas_modeling_block(node)
    pipeline = SkyulfPipeline({"preprocessing": [], "modeling": config})
    assert config["base_model"]["type"] == expected
    assert pipeline.model_estimator is not None


def test_canvas_export_rejects_mismatched_task():
    """Exports must not silently reinterpret a classifier as a regression model."""
    node = _NodeIn(
        node_id="model",
        step_type="training",
        params={"model_type": "logistic_regression", "task_type": "regression"},
    )
    with pytest.raises(ValueError, match="incompatible with task_type"):
        _canvas_modeling_block(node)


@pytest.mark.parametrize("algorithm", ["voting_classifier", "stacking_regressor", "kmeans"])
@pytest.mark.parametrize("mode", ["fixed", "tuned"])
def test_canvas_export_rejects_unreconstructable_models(algorithm, mode):
    """Unsupported calculator state must fail clearly instead of producing a broken notebook."""
    node = _NodeIn(
        node_id="model",
        step_type="training",
        params={
            "model_type": algorithm,
            "run_mode": mode,
            "hyperparameters": {},
            "tuning_config": {},
        },
    )
    with pytest.raises(ValueError, match="Notebook export does not yet support"):
        _canvas_modeling_block(node)
