"""Reviewed fitted trees must preserve predictions across independent worker batches."""

import numpy as np
import pandas as pd
import pytest

from skyulf.core.capabilities import UnsupportedExecutionError
from skyulf.data.dataset import SplitDataset
from skyulf.inference.local_pipeline import (
    load_local_pipeline,
    predict_local_pipeline,
    save_local_pipeline,
)
from skyulf.inference.partition_safety import require_partition_safe_pipeline
from skyulf.pipeline import SkyulfPipeline

TREE_NODES = [
    f"{family}_{task}"
    for family in ("decision_tree", "random_forest", "extra_trees")
    for task in ("regressor", "classifier")
]


def fitted_tree(tmp_path, node, *, tuned=False):
    """Use real Core calculators and persisted state rather than fabricated estimators."""
    classification = node.endswith("classifier")
    data = pd.DataFrame({"x": [-4.0, -3.0, -2.0, -1.0, 1.0, 2.0, 3.0, 4.0]})
    data["target"] = (
        ["low", "low", "low", "low", "high", "high", "high", "high"]
        if classification
        else data.x * 2 + 1
    )
    params = {"max_depth": 3, "random_state": 42}
    if not node.startswith("decision_tree"):
        params.update(n_estimators=3, n_jobs=1)
    modeling = {"type": node, "params": params}
    if tuned:
        modeling = {
            "type": "hyperparameter_tuner",
            "base_model": modeling,
            "strategy": "grid",
            "metric": "accuracy" if classification else "mse",
            "search_space": {},
            "cv_folds": 2,
            "n_trials": 1,
            "n_jobs": 1,
        }
    pipeline = SkyulfPipeline(
        {
            "preprocessing": [
                {"name": "fill", "transformer": "SimpleImputer", "params": {"columns": ["x"]}}
            ],
            "modeling": modeling,
        }
    )
    pipeline.fit(SplitDataset(train=data, test=data.iloc[:0]), target_column="target")
    if tuned and classification:
        assert pipeline.model_estimator is not None
        assert isinstance(pipeline.model_estimator.model, tuple)
        pipeline.model_estimator.model[1].decision_thresholds = {"high": 0.8, "low": 0.2}
    destination = tmp_path / node
    save_local_pipeline(pipeline, destination)
    return load_local_pipeline(destination)


@pytest.mark.parametrize("node", TREE_NODES)
@pytest.mark.parametrize("tuned", [False, True])
def test_exact_tree_models_replay_saved_state_in_any_batch(tmp_path, node, tuned):
    """Tree labels, probabilities and tuned wrappers must survive splits and reordering."""
    artifact = fitted_tree(tmp_path, node, tuned=tuned)
    before = require_partition_safe_pipeline(artifact)
    query = pd.DataFrame({"x": [np.nan, -2.0, 0.0, 9.0, np.nan]}, index=[90, 1, 15, 3, 4])
    whole = predict_local_pipeline(query, artifact)
    split = pd.concat(
        [predict_local_pipeline(query.iloc[part], artifact) for part in ([0], [1, 2], [3, 4])]
    )
    reordered = predict_local_pipeline(query.iloc[[4, 2, 0, 3, 1]], artifact)
    pd.testing.assert_frame_equal(whole, split)
    pd.testing.assert_frame_equal(whole.sort_index(), reordered.sort_index())
    empty_features = artifact.pipeline.feature_engineer.transform(
        query.iloc[:0], preserve_rows=True
    )
    assert empty_features.shape == (0, 1)
    assert require_partition_safe_pipeline(artifact) == before


@pytest.mark.parametrize(
    "change",
    [
        "subclass",
        "predict",
        "child_predict",
        "child_subclass",
        "tree",
        "tree_cycle",
        "tree_value",
        "empty_forest",
        "registry",
    ],
)
def test_tree_admission_rejects_substituted_or_unsafe_nested_state(tmp_path, monkeypatch, change):
    """An exact forest must not hide custom child callbacks or invalid native tree state."""
    artifact = fitted_tree(tmp_path, "random_forest_regressor")
    model = artifact.pipeline.model_estimator.model
    if change == "subclass":
        model.__class__ = type("CustomForest", (type(model),), {})
    elif change == "predict":
        model.predict = lambda values: np.zeros(len(values))
    elif change == "child_predict":
        model.estimators_[0].predict = lambda values, **kwargs: np.zeros(len(values))
    elif change == "child_subclass":
        child = model.estimators_[0]
        child.__class__ = type("CustomTree", (type(child),), {})
    elif change == "tree":
        model.estimators_[0].tree_ = object()
    elif change == "tree_cycle":
        model.estimators_[0].tree_.children_left[0] = 0
    elif change == "tree_value":
        model.estimators_[0].tree_.value[0] = np.nan
    elif change == "empty_forest":
        model.estimators_ = []
    else:
        from skyulf.registry import NodeRegistry

        monkeypatch.setitem(NodeRegistry._appliers, "random_forest_regressor", object)
    with pytest.raises(UnsupportedExecutionError):
        require_partition_safe_pipeline(artifact)


def test_changed_tree_values_change_worker_evidence(tmp_path):
    """Structurally valid learned changes must invalidate a previously issued certificate."""
    artifact = fitted_tree(tmp_path, "decision_tree_regressor")
    before = require_partition_safe_pipeline(artifact)
    artifact.pipeline.model_estimator.model.tree_.value[:] += 1.0
    assert require_partition_safe_pipeline(artifact).state_sha256 != before.state_sha256


@pytest.mark.parametrize("change", ["label_array", "class_count", "child_class_count"])
def test_tree_classification_rejects_custom_or_misaligned_class_axes(tmp_path, change):
    """Native probabilities and label lookup must use the exact same reviewed class axis."""
    artifact = fitted_tree(tmp_path, "random_forest_classifier")
    model = artifact.pipeline.model_estimator.model
    if change == "label_array":

        class CustomLabels(np.ndarray):
            """A class-axis subclass could implement batch-sensitive label lookup."""

            def take(self, *args, **kwargs):
                """Admission must inspect the class axis without executing it."""
                raise AssertionError("custom class lookup executed")

        model.classes_ = model.classes_.view(CustomLabels)
    elif change == "class_count":
        model.n_classes_ += 1
    else:
        model.estimators_[0].n_classes_ += 1
    with pytest.raises(UnsupportedExecutionError):
        require_partition_safe_pipeline(artifact)
