"""Tests for unified `training` step_type support in notebook export.

Mirrors the existing `basic_training`/`advanced_tuning` coverage but with the
canonical `step_type="training"` + `run_mode` param the frontend now emits,
to guard against the node being silently misclassified as a preprocessing
step (Finding 2a/2b) or exported without its tuning_config (Finding 2c).
"""

from backend.ml_pipeline._internal._routers import notebook_export as ne
from backend.ml_pipeline._internal._routers._notebook_branched import _SPLIT_OR_MODEL
from backend.ml_pipeline._internal._routers._notebook_builders import (
    _is_tuning_model,
    _NodeIn,
    modeling_cells,
)


def _base_nodes(train_node: _NodeIn) -> list[_NodeIn]:
    return [
        _NodeIn(node_id="loader", step_type="data_loader", params={"path": "data.csv"}, inputs=[]),
        _NodeIn(node_id="scaler", step_type="StandardScaler", params={}, inputs=["loader"]),
        _NodeIn(
            node_id="ft1",
            step_type="feature_target_split",
            params={"target_column": "y"},
            inputs=["scaler"],
        ),
        _NodeIn(
            node_id="tts1", step_type="TrainTestSplitter", params={"test_size": 0.2}, inputs=["ft1"]
        ),
        train_node,
    ]


class TestClassifyRecognizesTraining:
    """Finding 2a: `_MODELING_STEPS` must include `"training"`."""

    def test_classify_buckets_training_node_as_model(self):
        train_node = _NodeIn(
            node_id="train1",
            step_type="training",
            params={"algorithm": "xgboost_classifier", "run_mode": "fixed"},
            inputs=["tts1"],
        )
        nodes = _base_nodes(train_node)
        _loader, preprocess, _feat_target, _train_test, model = ne._classify(nodes)
        assert model is not None
        assert model.node_id == "train1"
        assert all(n.node_id != "train1" for n in preprocess)

    def test_terminal_models_includes_training_node(self):
        train_node = _NodeIn(
            node_id="train1",
            step_type="training",
            params={"algorithm": "xgboost_classifier", "run_mode": "fixed"},
            inputs=["tts1"],
        )
        nodes = _base_nodes(train_node)
        terminals = ne._terminal_models(nodes)
        assert [n.node_id for n in terminals] == ["train1"]


class TestSplitOrModelRecognizesTraining:
    """Finding 2b: `_SPLIT_OR_MODEL` in `_notebook_branched.py` must include `"training"`."""

    def test_training_in_split_or_model_set(self):
        assert "training" in _SPLIT_OR_MODEL


class TestIsTuningModel:
    """_is_tuning_model dispatches on run_mode param for `training` nodes."""

    def test_training_run_mode_tuned_is_tuning(self):
        model = _NodeIn(node_id="m", step_type="training", params={"run_mode": "tuned"}, inputs=[])
        assert _is_tuning_model(model) is True

    def test_training_run_mode_fixed_is_not_tuning(self):
        model = _NodeIn(node_id="m", step_type="training", params={"run_mode": "fixed"}, inputs=[])
        assert _is_tuning_model(model) is False

    def test_training_no_run_mode_is_not_tuning(self):
        model = _NodeIn(node_id="m", step_type="training", params={}, inputs=[])
        assert _is_tuning_model(model) is False


class TestModelingCellsForTrainingStepType:
    """End-to-end: `modeling_cells` renders the correct cell for `"training"`."""

    def test_fixed_run_mode_renders_basic_fit_cell(self):
        model = _NodeIn(
            node_id="train1",
            step_type="training",
            params={"algorithm": "xgboost_classifier", "run_mode": "fixed"},
            inputs=["tts1"],
        )
        cells = modeling_cells(model)
        code_cells = [c for c in cells if c["cell_type"] == "code"]
        assert len(code_cells) == 1
        src = "".join(code_cells[0]["source"])
        # Basic fit path: no tuning wrappers.
        assert "TuningCalculator" not in src
        assert "TuningApplier" not in src
        assert "estimator.fit_predict" in src

    def test_tuned_run_mode_renders_tuning_cell_with_config(self):
        model = _NodeIn(
            node_id="train1",
            step_type="training",
            params={
                "algorithm": "random_forest_classifier",
                "run_mode": "tuned",
                "tuning_config": {"n_trials": 25, "search": "optuna"},
            },
            inputs=["tts1"],
        )
        cells = modeling_cells(model)
        code_cells = [c for c in cells if c["cell_type"] == "code"]
        assert len(code_cells) == 1
        src = "".join(code_cells[0]["source"])
        # Tuning path must be used, and tuning_config must survive into the cell.
        assert "TuningCalculator" in src
        assert "TuningApplier" in src
        assert "n_trials" in src
        assert "25" in src
        assert "optuna" in src

    def test_full_notebook_includes_modeling_section_for_training_node(self):
        train_node = _NodeIn(
            node_id="train1",
            step_type="training",
            params={"algorithm": "xgboost_classifier", "run_mode": "fixed"},
            inputs=["tts1"],
        )
        cfg_nodes = _base_nodes(train_node)
        from backend.ml_pipeline._internal._routers._notebook_builders import _PipelineIn

        cfg = _PipelineIn(nodes=cfg_nodes)
        nb = ne._build_full_notebook(cfg, "test-id", "Training Node Pipeline")
        all_src = "".join(
            "".join(c.get("source", [])) for c in nb["cells"] if c["cell_type"] == "code"
        )
        assert "estimator.fit_predict" in all_src
