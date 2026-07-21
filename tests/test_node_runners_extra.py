"""Focused tests filling coverage gaps in ``_node_runners.py``'s extracted helpers.

Covers pure-logic branches (shape-metric recording, SplitDataset coercion, CV
metric aggregation/tuning-metric extraction, preview-summary building, model
component factory) via a lightweight harness, plus a few end-to-end
``PipelineEngine`` runs to exercise the CV/error/reference-data branches that
need real node execution plumbing (algorithm validation, cross-validation,
splitter reference-data saving, unknown-algorithm errors).
"""

from unittest.mock import MagicMock

import numpy as np
import pandas as pd
import pytest

from backend.data.catalog import FileSystemCatalog
from backend.ml_pipeline._execution.engine import PipelineEngine
from backend.ml_pipeline._execution.engine._node_runners import NodeRunnersMixin
from backend.ml_pipeline._execution.schemas import NodeConfig, PipelineConfig
from backend.ml_pipeline.artifacts.local import LocalArtifactStore
from backend.ml_pipeline.constants import StepType
from skyulf.data.dataset import SplitDataset


class _Harness(NodeRunnersMixin):
    """Minimal stand-in for :class:`PipelineEngine` for pure-logic helper tests."""

    def __init__(self):
        self.logs: list[str] = []
        self.artifact_store = MagicMock()
        self.catalog = MagicMock()
        self.executed_transformers = []

    def log(self, msg: str) -> None:
        self.logs.append(msg)


@pytest.fixture
def pipeline_data_csv(tmp_path):
    """Write a small, balanced classification CSV and return its path."""
    df = pd.DataFrame(
        {
            "f1": [1, 2, 3, 4, 5, 6] * 10,
            "f2": [10, 20, 30, 40, 50, 60] * 10,
            "target": [0, 0, 0, 1, 1, 1] * 10,
        }
    )
    path = tmp_path / "data.csv"
    df.to_csv(path, index=False)
    return str(path)


def _make_engine(tmp_path, name="artifacts"):
    artifact_store = LocalArtifactStore(str(tmp_path / name))
    catalog = FileSystemCatalog()
    return PipelineEngine(artifact_store, catalog=catalog)


# --- _record_split_dataset_shape_metrics / _record_tuple_shape_metrics ----


def test_record_split_dataset_shape_metrics_dataframe_train():
    """When train is a bare DataFrame (not a tuple), n_rows/n_features come from it."""
    harness = _Harness()
    train_df = pd.DataFrame({"a": [1, 2, 3], "b": [4, 5, 6], "target": [0, 1, 0]})
    data = SplitDataset(train=train_df, test=pd.DataFrame(), validation=None)
    metrics: dict = {}
    result = harness._record_split_dataset_shape_metrics(metrics, data, "target")
    assert result is True
    assert metrics["n_rows"] == 3
    # target column excluded from feature count
    assert metrics["n_features"] == 2


def test_record_split_dataset_shape_metrics_no_shape_returns_false():
    """Neither tuple-with-shape nor object-with-shape -> returns False, no metrics set."""
    harness = _Harness()
    data = SplitDataset(train=object(), test=pd.DataFrame(), validation=None)
    metrics: dict = {}
    result = harness._record_split_dataset_shape_metrics(metrics, data, "target")
    assert result is False
    assert metrics == {}


def test_record_tuple_shape_metrics_empty_tuple_noop():
    """An empty tuple leaves metrics untouched."""
    harness = _Harness()
    metrics: dict = {}
    harness._record_tuple_shape_metrics(metrics, ())
    assert metrics == {}


def test_record_tuple_shape_metrics_no_shape_attr_noop():
    """A tuple whose first element lacks .shape leaves metrics untouched."""
    harness = _Harness()
    metrics: dict = {}
    harness._record_tuple_shape_metrics(metrics, (object(), object()))
    assert metrics == {}


def test_record_tuple_shape_metrics_sets_values():
    """A tuple whose first element has .shape populates n_rows/n_features."""
    harness = _Harness()
    metrics: dict = {}
    X = np.zeros((5, 3))
    harness._record_tuple_shape_metrics(metrics, (X, np.zeros(5)))
    assert metrics["n_rows"] == 5
    assert metrics["n_features"] == 3


def test_record_data_shape_metrics_dispatches_to_tuple_branch():
    """A plain (X, y) tuple (not a SplitDataset/DataFrame) hits the tuple branch."""
    harness = _Harness()
    metrics: dict = {}
    X = np.zeros((4, 2))
    harness._record_data_shape_metrics(metrics, (X, np.zeros(4)), "target")
    assert metrics["n_rows"] == 4
    assert metrics["n_features"] == 2


def test_safe_record_data_shape_metrics_swallows_exceptions():
    """Exceptions inside _record_data_shape_metrics are caught and logged, not raised."""
    harness = _Harness()

    def boom(*args, **kwargs):
        raise RuntimeError("boom")

    harness._record_data_shape_metrics = boom
    metrics: dict = {}
    # Should not raise despite the patched method blowing up.
    harness._safe_record_data_shape_metrics(metrics, object(), "target", "node1")
    assert metrics == {}


# --- _get_training_input --------------------------------------------------


def test_get_training_input_rejects_model_artifact():
    """A model-like artifact (has .predict/.fit) upstream raises a clear ValueError."""
    harness = _Harness()
    node = NodeConfig(node_id="n1", step_type=StepType.TRAINING, params={})

    class FakeModel:
        def predict(self):
            pass

    harness._get_input = MagicMock(return_value=FakeModel())
    with pytest.raises(ValueError, match="received a Model object"):
        harness._get_training_input(node, "target")


def test_get_training_input_passes_through_dataframe():
    """A plain DataFrame upstream input passes through untouched."""
    harness = _Harness()
    node = NodeConfig(node_id="n1", step_type=StepType.TRAINING, params={})
    df = pd.DataFrame({"a": [1, 2]})
    harness._get_input = MagicMock(return_value=df)
    result = harness._get_training_input(node, "target")
    assert result is df


# --- _to_split_dataset -----------------------------------------------------


def test_to_split_dataset_dataframe_input():
    """A bare DataFrame becomes SplitDataset(train=df, test=empty)."""
    harness = _Harness()
    df = pd.DataFrame({"a": [1, 2], "target": [0, 1]})
    result = harness._to_split_dataset(df, "target")
    assert isinstance(result, SplitDataset)
    assert result.train is df
    assert result.test.empty


def test_to_split_dataset_train_test_tuple():
    """A (train_df, test_df) tuple with target_col present splits into SplitDataset."""
    harness = _Harness()
    train_df = pd.DataFrame({"a": [1, 2], "target": [0, 1]})
    test_df = pd.DataFrame({"a": [3], "target": [1]})
    result = harness._to_split_dataset((train_df, test_df), "target")
    assert isinstance(result, SplitDataset)
    assert result.train is train_df
    assert result.test is test_df


def test_to_split_dataset_xy_tuple_no_target_col():
    """An (X, y) tuple where X lacks target_col wraps as train=(X, y), test=empty."""
    harness = _Harness()
    X = pd.DataFrame({"a": [1, 2]})
    y = pd.Series([0, 1])
    result = harness._to_split_dataset((X, y), "target")
    assert isinstance(result, SplitDataset)
    assert result.train == (X, y)
    assert result.test.empty


def test_to_split_dataset_passthrough_for_other_types():
    """A value that's neither DataFrame nor tuple (e.g. already SplitDataset) passes through."""
    harness = _Harness()
    existing = SplitDataset(train=pd.DataFrame(), test=pd.DataFrame(), validation=None)
    assert harness._to_split_dataset(existing, "target") is existing


# --- _aggregate_cv_metrics / _extract_tuning_metrics -----------------------


def test_aggregate_cv_metrics_flattens_mean_std():
    harness = _Harness()
    cv_results = {
        "aggregated_metrics": {
            "accuracy": {"mean": 0.9, "std": 0.05},
            "non_numeric": "ignored",
        }
    }
    result = harness._aggregate_cv_metrics(cv_results)
    assert result == {"cv_accuracy_mean": 0.9, "cv_accuracy_std": 0.05}


def test_aggregate_cv_metrics_falls_back_to_raw_dict():
    """When there's no 'aggregated_metrics' key, iterate the dict itself."""
    harness = _Harness()
    cv_results = {"f1": {"mean": 0.8, "std": 0.1}}
    result = harness._aggregate_cv_metrics(cv_results)
    assert result == {"cv_f1_mean": 0.8, "cv_f1_std": 0.1}


def test_extract_tuning_metrics_no_tuning_result():
    """When estimator.model isn't a 2-tuple, no TuningResult is present -> empty metrics."""
    harness = _Harness()
    estimator = MagicMock()
    estimator.model = "just_a_model_no_tuple"
    tuning_result, metrics = harness._extract_tuning_metrics(estimator, {})
    assert tuning_result is None
    assert metrics == {}


def test_extract_tuning_metrics_with_tuning_result():
    harness = _Harness()
    estimator = MagicMock()
    fake_result = MagicMock()
    fake_result.best_score = 0.95
    fake_result.best_params = {"C": 1.0}
    fake_result.trials = [{"score": 0.9}]
    fake_result.scoring_metric = "accuracy"
    estimator.model = ("model_obj", fake_result)
    tuning_result, metrics = harness._extract_tuning_metrics(estimator, {})
    assert tuning_result is fake_result
    assert metrics["best_score"] == 0.95
    assert metrics["best_params"] == {"C": 1.0}
    assert metrics["scoring_metric"] == "accuracy"


# --- _get_model_components --------------------------------------------------


def test_get_model_components_unknown_algorithm_raises():
    harness = _Harness()
    with pytest.raises(ValueError, match="Unknown algorithm"):
        harness._get_model_components("totally_not_a_real_algorithm")


def test_get_model_components_known_alias():
    """Legacy alias names resolve via the alias map to a real registry id."""
    harness = _Harness()
    calculator, applier = harness._get_model_components("random_forest")
    assert calculator is not None
    assert applier is not None


# --- _preview_slot_info / _build_split_dataset_data_summary ----------------


def test_preview_slot_info_tuple_slot():
    harness = _Harness()
    X = pd.DataFrame({"a": [1, 2]})
    y = pd.Series([0, 1])
    info = harness._preview_slot_info((X, y), "Test")
    assert info["name"] == "Test (X)"
    assert info["shape"] == (2, 1)


def test_preview_slot_info_dataframe_slot():
    harness = _Harness()
    df = pd.DataFrame({"a": [1, 2]})
    info = harness._preview_slot_info(df, "Validation")
    assert info["name"] == "Validation"
    assert info["shape"] == (2, 1)


def test_preview_slot_info_empty_dataframe_returns_none():
    harness = _Harness()
    assert harness._preview_slot_info(pd.DataFrame(), "Validation") is None


def test_preview_slot_info_none_slot_returns_none():
    harness = _Harness()
    assert harness._preview_slot_info(None, "Validation") is None


def test_build_split_dataset_data_summary_full():
    """Exercises train-as-DataFrame, test tuple, and validation DataFrame branches together."""
    harness = _Harness()
    train_df = pd.DataFrame({"a": [1, 2, 3]})
    test_X = pd.DataFrame({"a": [4]})
    test_y = pd.Series([0])
    validation_df = pd.DataFrame({"a": [5, 6]})
    data = SplitDataset(train=train_df, test=(test_X, test_y), validation=validation_df)

    summary = harness._build_split_dataset_data_summary(data)

    assert summary["train"]["name"] == "Train"
    assert summary["test"]["name"] == "Test (X)"
    assert summary["validation"]["name"] == "Validation"


def test_build_split_dataset_data_summary_train_tuple():
    """Train as an (X, y) tuple takes the tuple branch instead of the DataFrame branch."""
    harness = _Harness()
    train_X = pd.DataFrame({"a": [1, 2]})
    train_y = pd.Series([0, 1])
    data = SplitDataset(train=(train_X, train_y), test=pd.DataFrame(), validation=None)

    summary = harness._build_split_dataset_data_summary(data)

    assert summary["train"]["name"] == "Train (X)"
    assert "test" not in summary
    assert "validation" not in summary


def test_build_split_dataset_data_summary_empty_test_slot_omitted():
    """An empty test DataFrame slot doesn't add a 'test' key to the summary."""
    harness = _Harness()
    data = SplitDataset(train=pd.DataFrame({"a": [1]}), test=pd.DataFrame(), validation=None)
    summary = harness._build_split_dataset_data_summary(data)
    assert "test" not in summary


# --- End-to-end PipelineEngine runs: CV branches, error paths, splitters ---


def test_basic_training_missing_algorithm_raises(pipeline_data_csv, tmp_path):
    """Basic training without 'algorithm'/'model_type' raises a clear ValueError."""
    engine = _make_engine(tmp_path, "artifacts_missing_algo")
    config = PipelineConfig(
        pipeline_id="p_missing_algo",
        nodes=[
            NodeConfig(
                node_id="node_data",
                step_type=StepType.DATA_LOADER,
                params={"source": "csv", "path": pipeline_data_csv},
            ),
            NodeConfig(
                node_id="node_training",
                step_type=StepType.TRAINING,
                inputs=["node_data"],
                params={"target_column": "target"},
            ),
        ],
    )
    result = engine.run(config)
    assert result.status == "failed"
    assert result.node_results["node_training"].status == "failed"
    assert "algorithm" in (result.node_results["node_training"].error or "").lower()


def test_advanced_tuning_missing_algorithm_raises(pipeline_data_csv, tmp_path):
    """Advanced tuning without 'algorithm'/'model_type' raises a clear ValueError."""
    engine = _make_engine(tmp_path, "artifacts_missing_algo_tuning")
    config = PipelineConfig(
        pipeline_id="p_missing_algo_tuning",
        nodes=[
            NodeConfig(
                node_id="node_data",
                step_type=StepType.DATA_LOADER,
                params={"source": "csv", "path": pipeline_data_csv},
            ),
            NodeConfig(
                node_id="node_tuning",
                step_type=StepType.TRAINING,
                inputs=["node_data"],
                params={
                    "run_mode": "tuned",
                    "target_column": "target",
                    "tuning_config": {"strategy": "grid", "metric": "accuracy", "cv_folds": 2},
                },
            ),
        ],
    )
    result = engine.run(config)
    assert result.status == "failed"
    assert "algorithm" in (result.node_results["node_tuning"].error or "").lower()


def test_basic_training_unknown_algorithm_raises(pipeline_data_csv, tmp_path):
    """An unrecognized algorithm name surfaces the registry factory's ValueError."""
    engine = _make_engine(tmp_path, "artifacts_unknown_algo")
    config = PipelineConfig(
        pipeline_id="p_unknown_algo",
        nodes=[
            NodeConfig(
                node_id="node_data",
                step_type=StepType.DATA_LOADER,
                params={"source": "csv", "path": pipeline_data_csv},
            ),
            NodeConfig(
                node_id="node_training",
                step_type=StepType.TRAINING,
                inputs=["node_data"],
                params={"target_column": "target", "algorithm": "not_a_real_algorithm"},
            ),
        ],
    )
    result = engine.run(config)
    assert result.status == "failed"
    assert "Unknown algorithm" in (result.node_results["node_training"].error or "")


def test_basic_training_with_cross_validation(pipeline_data_csv, tmp_path):
    """cv_enabled=True on a basic-training node runs CV and produces cv_* metrics."""
    engine = _make_engine(tmp_path, "artifacts_cv_basic")
    config = PipelineConfig(
        pipeline_id="p_cv_basic",
        nodes=[
            NodeConfig(
                node_id="node_data",
                step_type=StepType.DATA_LOADER,
                params={"source": "csv", "path": pipeline_data_csv},
            ),
            NodeConfig(
                node_id="node_training",
                step_type=StepType.TRAINING,
                inputs=["node_data"],
                params={
                    "target_column": "target",
                    "algorithm": "logistic_regression",
                    "hyperparameters": {"C": 1.0},
                    "evaluate": True,
                    "cv_enabled": True,
                    "cv_folds": 3,
                },
            ),
        ],
    )
    result = engine.run(config)
    assert result.status == "success"
    train_res = result.node_results["node_training"]
    assert train_res.status == "success"
    cv_keys = [k for k in train_res.metrics if k.startswith("cv_")]
    assert cv_keys, f"expected cv_ metrics, got: {train_res.metrics.keys()}"


def test_advanced_tuning_with_cross_validation_nested(pipeline_data_csv, tmp_path):
    """cv_enabled + cv_type='nested_cv' on tuning exercises the post-tuning CV downgrade path."""
    engine = _make_engine(tmp_path, "artifacts_cv_tuning")
    config = PipelineConfig(
        pipeline_id="p_cv_tuning",
        nodes=[
            NodeConfig(
                node_id="node_data",
                step_type=StepType.DATA_LOADER,
                params={"source": "csv", "path": pipeline_data_csv},
            ),
            NodeConfig(
                node_id="node_tuning",
                step_type=StepType.TRAINING,
                inputs=["node_data"],
                params={
                    "run_mode": "tuned",
                    "target_column": "target",
                    "algorithm": "logistic_regression",
                    "tuning_config": {
                        "strategy": "grid",
                        "metric": "accuracy",
                        "cv_folds": 2,
                        "search_space": {"C": [0.1, 1.0]},
                        "cv_enabled": True,
                        "cv_type": "nested_cv",
                    },
                },
            ),
        ],
    )
    result = engine.run(config)
    assert result.status == "success"
    tuning_res = result.node_results["node_tuning"]
    assert tuning_res.status == "success"
    cv_keys = [k for k in tuning_res.metrics if k.startswith("cv_")]
    assert cv_keys, f"expected cv_ metrics, got: {tuning_res.metrics.keys()}"


def test_splitter_transformer_saves_reference_data(pipeline_data_csv, tmp_path):
    """A Splitter-named single-transformer node ahead of a training node triggers ref-data save."""
    engine = _make_engine(tmp_path, "artifacts_splitter_ref")
    config = PipelineConfig(
        pipeline_id="p_splitter_ref",
        nodes=[
            NodeConfig(
                node_id="node_data",
                step_type=StepType.DATA_LOADER,
                params={"source": "csv", "path": pipeline_data_csv},
            ),
            NodeConfig(
                node_id="node_split",
                step_type="TrainTestSplitter",
                inputs=["node_data"],
                params={"target_column": "target"},
            ),
            NodeConfig(
                node_id="node_training",
                step_type=StepType.TRAINING,
                inputs=["node_split"],
                params={
                    "target_column": "target",
                    "algorithm": "logistic_regression",
                    "evaluate": False,
                },
            ),
        ],
    )
    result = engine.run(config, job_id="job123")
    assert result.status == "success"
    assert result.node_results["node_split"].status == "success"


def test_advanced_tuning_ensemble_auto_builds_search_space(pipeline_data_csv, tmp_path):
    """An ensemble algorithm with no explicit search_space auto-builds one from defaults."""
    engine = _make_engine(tmp_path, "artifacts_ensemble_tuning")
    config = PipelineConfig(
        pipeline_id="p_ensemble_tuning",
        nodes=[
            NodeConfig(
                node_id="node_data",
                step_type=StepType.DATA_LOADER,
                params={"source": "csv", "path": pipeline_data_csv},
            ),
            NodeConfig(
                node_id="node_tuning",
                step_type=StepType.TRAINING,
                inputs=["node_data"],
                params={
                    "run_mode": "tuned",
                    "target_column": "target",
                    "algorithm": "voting_classifier",
                    "tuning_config": {
                        "strategy": "random",
                        "metric": "accuracy",
                        "cv_folds": 2,
                        "n_trials": 2,
                    },
                },
            ),
        ],
    )
    result = engine.run(config)
    assert result.status == "success"
    assert result.node_results["node_tuning"].status == "success"


def test_advanced_tuning_evaluate_failure_is_logged_not_raised(
    pipeline_data_csv, tmp_path, monkeypatch
):
    """If post-tuning evaluation raises, the node still succeeds (error is logged, not re-raised)."""
    from skyulf.modeling.base import StatefulEstimator

    def boom_evaluate(self, *args, **kwargs):
        raise RuntimeError("evaluate boom")

    monkeypatch.setattr(StatefulEstimator, "evaluate", boom_evaluate)

    engine = _make_engine(tmp_path, "artifacts_tuning_eval_fail")
    config = PipelineConfig(
        pipeline_id="p_tuning_eval_fail",
        nodes=[
            NodeConfig(
                node_id="node_data",
                step_type=StepType.DATA_LOADER,
                params={"source": "csv", "path": pipeline_data_csv},
            ),
            NodeConfig(
                node_id="node_tuning",
                step_type=StepType.TRAINING,
                inputs=["node_data"],
                params={
                    "run_mode": "tuned",
                    "target_column": "target",
                    "algorithm": "logistic_regression",
                    "tuning_config": {
                        "strategy": "grid",
                        "metric": "accuracy",
                        "cv_folds": 2,
                        "search_space": {"C": [0.1, 1.0]},
                    },
                },
            ),
        ],
    )
    result = engine.run(config)
    assert result.status == "success"
    tuning_res = result.node_results["node_tuning"]
    assert tuning_res.status == "success"
    # Evaluation failed, so no train_/test_ prefixed metrics were added.
    assert not any(k.startswith("train_") or k.startswith("test_") for k in tuning_res.metrics)


def test_run_tuned_cv_exception_is_caught(pipeline_data_csv, tmp_path, monkeypatch):
    """If cross_validate raises during post-tuning CV, the exception is caught and cv metrics are empty."""
    from skyulf.modeling.base import StatefulEstimator

    def boom_cv(self, *args, **kwargs):
        raise RuntimeError("cv boom")

    monkeypatch.setattr(StatefulEstimator, "cross_validate", boom_cv)

    engine = _make_engine(tmp_path, "artifacts_tuning_cv_fail")
    config = PipelineConfig(
        pipeline_id="p_tuning_cv_fail",
        nodes=[
            NodeConfig(
                node_id="node_data",
                step_type=StepType.DATA_LOADER,
                params={"source": "csv", "path": pipeline_data_csv},
            ),
            NodeConfig(
                node_id="node_tuning",
                step_type=StepType.TRAINING,
                inputs=["node_data"],
                params={
                    "run_mode": "tuned",
                    "target_column": "target",
                    "algorithm": "logistic_regression",
                    "tuning_config": {
                        "strategy": "grid",
                        "metric": "accuracy",
                        "cv_folds": 2,
                        "search_space": {"C": [0.1, 1.0]},
                        "cv_enabled": True,
                    },
                },
            ),
        ],
    )
    result = engine.run(config)
    assert result.status == "success"
    tuning_res = result.node_results["node_tuning"]
    assert tuning_res.status == "success"
    assert not any(k.startswith("cv_") for k in tuning_res.metrics)


def test_data_loader_uses_dataset_id_param(pipeline_data_csv, tmp_path):
    """Data loader resolves 'dataset_id' directly, without falling back to 'path'."""
    engine = _make_engine(tmp_path, "artifacts_dataset_id")
    config = PipelineConfig(
        pipeline_id="p_dataset_id",
        nodes=[
            NodeConfig(
                node_id="node_data",
                step_type=StepType.DATA_LOADER,
                params={"source": "csv", "dataset_id": pipeline_data_csv},
            ),
        ],
    )
    result = engine.run(config)
    assert result.status == "success"
    assert result.node_results["node_data"].status == "success"


def test_data_loader_missing_path_raises_keyerror(tmp_path):
    """Data loader without 'dataset_id' or 'path' raises a KeyError describing the missing param."""
    engine = _make_engine(tmp_path, "artifacts_missing_path")
    config = PipelineConfig(
        pipeline_id="p_missing_path",
        nodes=[
            NodeConfig(
                node_id="node_data", step_type=StepType.DATA_LOADER, params={"source": "csv"}
            ),
        ],
    )
    result = engine.run(config)
    assert result.status == "failed"
    assert "dataset_id" in (result.node_results["node_data"].error or "")


def test_data_loader_sample_mode(pipeline_data_csv, tmp_path):
    """sample=True limits the loaded rows via the 'limit' param."""
    engine = _make_engine(tmp_path, "artifacts_sample")
    config = PipelineConfig(
        pipeline_id="p_sample",
        nodes=[
            NodeConfig(
                node_id="node_data",
                step_type=StepType.DATA_LOADER,
                params={"source": "csv", "path": pipeline_data_csv, "sample": True, "limit": 5},
            ),
        ],
    )
    result = engine.run(config)
    assert result.status == "success"
    df = engine.artifact_store.load("node_data")
    assert len(df) == 5


def test_data_loader_file_not_found(tmp_path):
    """A dataset path that doesn't exist raises a friendly FileNotFoundError."""
    engine = _make_engine(tmp_path, "artifacts_not_found")
    missing_path = str(tmp_path / "does_not_exist.csv")
    config = PipelineConfig(
        pipeline_id="p_not_found",
        nodes=[
            NodeConfig(
                node_id="node_data",
                step_type=StepType.DATA_LOADER,
                params={"source": "csv", "path": missing_path},
            ),
        ],
    )
    result = engine.run(config)
    assert result.status == "failed"
    assert "not found" in (result.node_results["node_data"].error or "").lower()


def test_data_preview_with_plain_dataframe(pipeline_data_csv, tmp_path):
    """A data_preview node fed a plain DataFrame (not SplitDataset) takes the 'full' branch."""
    engine = _make_engine(tmp_path, "artifacts_preview_df")
    config = PipelineConfig(
        pipeline_id="p_preview_df",
        nodes=[
            NodeConfig(
                node_id="node_data",
                step_type=StepType.DATA_LOADER,
                params={"source": "csv", "path": pipeline_data_csv},
            ),
            NodeConfig(
                node_id="node_preview",
                step_type="data_preview",
                inputs=["node_data"],
                params={},
            ),
        ],
    )
    result = engine.run(config)
    assert result.status == "success"
    preview_res = result.node_results["node_preview"]
    assert preview_res.metrics["operation_mode"] == "fit_transform"
    assert "full" in preview_res.metrics["data_summary"]


def test_data_preview_with_split_dataset(pipeline_data_csv, tmp_path):
    """A data_preview node fed a SplitDataset with test/validation exercises the preview summary."""
    engine = _make_engine(tmp_path, "artifacts_preview")
    config = PipelineConfig(
        pipeline_id="p_preview",
        nodes=[
            NodeConfig(
                node_id="node_data",
                step_type=StepType.DATA_LOADER,
                params={"source": "csv", "path": pipeline_data_csv},
            ),
            NodeConfig(
                node_id="node_split",
                step_type="TrainTestSplitter",
                inputs=["node_data"],
                params={"target_column": "target"},
            ),
            NodeConfig(
                node_id="node_preview",
                step_type="data_preview",
                inputs=["node_split"],
                params={},
            ),
        ],
    )
    result = engine.run(config)
    assert result.status == "success"
    preview_res = result.node_results["node_preview"]
    assert preview_res.status == "success"
    assert preview_res.metrics["operation_mode"].startswith("Train:")
    assert "train" in preview_res.metrics["data_summary"]


# --- Clustering (KMeans / Segmentation) ------------------------------------


def test_basic_training_kmeans_without_target_column_succeeds(pipeline_data_csv, tmp_path):
    """A KMeans basic-training node with no target_column should train, predict,
    and evaluate with clustering metrics — not crash on the missing target."""
    engine = _make_engine(tmp_path, "artifacts_kmeans")
    config = PipelineConfig(
        pipeline_id="p_kmeans",
        nodes=[
            NodeConfig(
                node_id="node_data",
                step_type=StepType.DATA_LOADER,
                params={"source": "csv", "path": pipeline_data_csv},
            ),
            NodeConfig(
                node_id="node_training",
                step_type=StepType.TRAINING,
                inputs=["node_data"],
                params={
                    "algorithm": "kmeans",
                    "hyperparameters": {"n_clusters": 2, "n_init": 5},
                    "evaluate": True,
                },
            ),
        ],
    )
    result = engine.run(config)
    assert result.status == "success"
    train_res = result.node_results["node_training"]
    assert train_res.status == "success"
    # No feature importances/SHAP for clustering — those keys should be absent.
    assert "feature_importances" not in train_res.metrics
    assert "shap_explanation" not in train_res.metrics
    # cv_ metrics should not appear even though cv_enabled defaults False on a
    # bare basic_training node (clustering skips CV entirely regardless).
    assert not [k for k in train_res.metrics if k.startswith("cv_")]


def test_advanced_tuning_clustering_algorithm_silently_runs_fixed_mode(pipeline_data_csv, tmp_path):
    """Phase 2b: clustering never had a supervised scorer to tune against, so a
    clustering algorithm reaching a tuning-mode node (e.g. Advanced Tuning /
    run_mode='tuned') now silently forces the plain direct-fit path instead of
    raising — clustering has no reachable "toggle mismatch" scenario via the
    UI (its model dropdown never offers a tuning mode), so this is a
    defensive-path behavior change, not a user-facing regression."""
    engine = _make_engine(tmp_path, "artifacts_kmeans_tuning_reject")
    config = PipelineConfig(
        pipeline_id="p_kmeans_tuning_reject",
        nodes=[
            NodeConfig(
                node_id="node_data",
                step_type=StepType.DATA_LOADER,
                params={"source": "csv", "path": pipeline_data_csv},
            ),
            NodeConfig(
                node_id="node_tuning",
                step_type=StepType.TRAINING,
                inputs=["node_data"],
                params={
                    "run_mode": "tuned",
                    "target_column": "target",
                    "algorithm": "kmeans",
                    "tuning_config": {"strategy": "grid", "metric": "accuracy", "cv_folds": 2},
                },
            ),
        ],
    )
    result = engine.run(config)
    assert result.status == "success"


def test_kmeans_drops_text_columns_and_bundles_feature_columns(tmp_path):
    """A text/id column left in by upstream nodes must not be fed to KMeans,
    and the bundled inference artifact must record the exact (numeric-only)
    feature columns the model was trained on — regression test for the
    "X has N features, but KMeans is expecting M features" deployment bug,
    which was ultimately caused by no numeric-only filtering existing before
    `.fit()`/`.predict()` and no authoritative feature list being persisted.
    """
    df = pd.DataFrame(
        {
            "customer_id": [f"cust_{i}" for i in range(20)],
            "amount": [10.0, 20.0, 30.0, 40.0] * 5,
            "frequency": [1, 2, 3, 4] * 5,
        }
    )
    csv_path = tmp_path / "segmentation_data.csv"
    df.to_csv(csv_path, index=False)

    engine = _make_engine(tmp_path, "artifacts_kmeans_text_cols")
    config = PipelineConfig(
        pipeline_id="p_kmeans_text_cols",
        nodes=[
            NodeConfig(
                node_id="node_data",
                step_type=StepType.DATA_LOADER,
                params={"source": "csv", "path": str(csv_path)},
            ),
            NodeConfig(
                node_id="node_training",
                step_type=StepType.TRAINING,
                inputs=["node_data"],
                params={
                    "algorithm": "kmeans",
                    "hyperparameters": {"n_clusters": 2, "n_init": 5},
                    "evaluate": True,
                },
            ),
        ],
    )
    result = engine.run(config, job_id="job_kmeans_text_cols")
    assert result.status == "success"

    bundled = engine.artifact_store.load("job_kmeans_text_cols")
    assert bundled["feature_columns"] == ["amount", "frequency"]
    assert "customer_id" not in bundled["feature_columns"]

    # Inference with the text column still present (as it would arrive from
    # a raw dataset row) must not crash — the applier drops it internally,
    # matching what happened at fit time.
    fresh_rows = pd.DataFrame(
        {
            "customer_id": ["cust_new"],
            "amount": [15.0],
            "frequency": [2],
        }
    )

    from skyulf.modeling.clustering import KMeansApplier

    preds = KMeansApplier().predict(fresh_rows, bundled["model"])
    assert len(preds) == 1


def test_kmeans_reference_column_excluded_and_crosstab_bundled(tmp_path):
    """A user-designated `reference_column` (e.g. a known label like species
    name, not used as a training feature) must be excluded from the model's
    features/`feature_columns`, but still be usable afterward to interpret
    which cluster corresponds to which real-world group via the
    `reference_crosstab` in the evaluation report.
    """
    df = pd.DataFrame(
        {
            "species": ["setosa"] * 10 + ["versicolor"] * 10,
            "petal_length": [1.0, 1.1, 1.2, 1.3, 1.4, 1.1, 1.2, 1.3, 1.4, 1.0]
            + [4.0, 4.1, 4.2, 4.3, 4.4, 4.1, 4.2, 4.3, 4.4, 4.0],
            "petal_width": [0.2, 0.2, 0.3, 0.2, 0.3, 0.2, 0.2, 0.3, 0.2, 0.3]
            + [1.3, 1.4, 1.3, 1.4, 1.3, 1.4, 1.3, 1.4, 1.3, 1.4],
        }
    )
    csv_path = tmp_path / "iris_like.csv"
    df.to_csv(csv_path, index=False)

    engine = _make_engine(tmp_path, "artifacts_kmeans_reference_column")
    config = PipelineConfig(
        pipeline_id="p_kmeans_reference_column",
        nodes=[
            NodeConfig(
                node_id="node_data",
                step_type=StepType.DATA_LOADER,
                params={"source": "csv", "path": str(csv_path)},
            ),
            NodeConfig(
                node_id="node_training",
                step_type=StepType.TRAINING,
                inputs=["node_data"],
                params={
                    "algorithm": "kmeans",
                    "hyperparameters": {"n_clusters": 2, "n_init": 5},
                    "reference_column": "species",
                    "evaluate": True,
                },
            ),
        ],
    )
    result = engine.run(config, job_id="job_kmeans_reference_column")
    assert result.status == "success"

    bundled = engine.artifact_store.load("job_kmeans_reference_column")
    assert "species" not in bundled["feature_columns"]
    assert set(bundled["feature_columns"]) == {"petal_length", "petal_width"}

    eval_data = engine.artifact_store.load("job_kmeans_reference_column_evaluation_data")
    train_clustering = eval_data["splits"]["train"]["clustering"]
    assert train_clustering["reference_column"] == "species"
    crosstab = train_clustering["reference_crosstab"]
    # Each cluster should be dominated by a single species, since the two
    # groups are numerically well-separated.
    assert len(crosstab) == 2
    for counts in crosstab.values():
        assert sum(counts.values()) == 10
        assert max(counts.values()) == 10  # perfectly pure cluster in this synthetic data

    # Auto-generated profile label present for every centroid, with no
    # reference-column leakage into the numeric profile.
    for centroid in train_clustering["centroids"]:
        assert centroid["profile"]
        assert "species" not in centroid["center"]


@pytest.mark.parametrize("algorithm", ["minibatch_kmeans", "gaussian_mixture", "birch"])
def test_additional_clustering_algorithms_train_and_bundle(tmp_path, algorithm):
    """Mini-Batch K-Means, Gaussian Mixture, and Birch must all be trainable
    through the same Basic Training pipeline as K-Means, and produce a
    bundled artifact with the expected feature_columns."""
    df = pd.DataFrame(
        {
            "a": [1.0, 2.0, 3.0, 4.0, 10.0, 11.0, 12.0, 13.0] * 3,
            "b": [1.0, 2.0, 1.0, 2.0, 10.0, 11.0, 10.0, 11.0] * 3,
        }
    )
    csv_path = tmp_path / f"cluster_data_{algorithm}.csv"
    df.to_csv(csv_path, index=False)

    engine = _make_engine(tmp_path, f"artifacts_{algorithm}")
    hyperparameters = {
        "minibatch_kmeans": {"n_clusters": 2},
        "gaussian_mixture": {"n_components": 2},
        "birch": {"n_clusters": 2},
    }[algorithm]
    config = PipelineConfig(
        pipeline_id=f"p_{algorithm}",
        nodes=[
            NodeConfig(
                node_id="node_data",
                step_type=StepType.DATA_LOADER,
                params={"source": "csv", "path": str(csv_path)},
            ),
            NodeConfig(
                node_id="node_training",
                step_type=StepType.TRAINING,
                inputs=["node_data"],
                params={
                    "algorithm": algorithm,
                    "hyperparameters": hyperparameters,
                    "evaluate": True,
                },
            ),
        ],
    )
    result = engine.run(config, job_id=f"job_{algorithm}")
    assert result.status == "success"

    bundled = engine.artifact_store.load(f"job_{algorithm}")
    assert bundled["feature_columns"] == ["a", "b"]
