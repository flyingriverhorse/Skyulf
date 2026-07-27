"""Tests for SkyulfPipeline.optimize_thresholds() and predict(use_tuned_thresholds=...)."""

import numpy as np
import pytest
from sklearn.metrics import f1_score

from skyulf.pipeline import SkyulfPipeline


def _binary_config(test_size=0.25, random_state=42):
    return {
        "preprocessing": [
            {
                "name": "imputer",
                "transformer": "SimpleImputer",
                "params": {"strategy": "mean"},
            },
            {
                "name": "split",
                "transformer": "TrainTestSplitter",
                "params": {"test_size": test_size, "random_state": random_state},
            },
        ],
        "modeling": {"type": "logistic_regression"},
    }


def test_optimize_thresholds_returns_dict_covering_both_classes(sample_classification_data):
    data = sample_classification_data.drop(columns=["category"])
    pipeline = SkyulfPipeline(_binary_config())
    X_train, y_train, X_val, y_val = pipeline.get_fitted_split(data, target_column="target")
    pipeline.fit(data, target_column="target")

    def metric(y_true, y_pred):
        return f1_score(y_true, y_pred, average="macro")

    thresholds = pipeline.optimize_thresholds(X_val, y_val, metric=metric)
    assert set(thresholds.keys()) == set(np.unique(y_train))


def test_optimize_thresholds_stores_result_on_instance(sample_classification_data):
    data = sample_classification_data.drop(columns=["category"])
    pipeline = SkyulfPipeline(_binary_config())
    pipeline.fit(data, target_column="target")
    _, _, X_val, y_val = pipeline.get_fitted_split(data, target_column="target")

    assert pipeline._tuned_thresholds is None
    thresholds = pipeline.optimize_thresholds(
        X_val, y_val, metric=lambda a, b: f1_score(a, b, average="macro")
    )
    assert pipeline._tuned_thresholds == thresholds


def test_optimize_thresholds_raises_if_pipeline_not_fitted(sample_classification_data):
    pipeline = SkyulfPipeline(_binary_config())
    data = sample_classification_data.drop(columns=["category"])
    with pytest.raises(ValueError, match="fitted"):
        pipeline.optimize_thresholds(
            data.drop(columns=["target"]),
            data["target"],
            metric=lambda a, b: f1_score(a, b, average="macro"),
        )


def test_predict_use_tuned_thresholds_raises_before_tuning(sample_classification_data):
    data = sample_classification_data.drop(columns=["category"])
    pipeline = SkyulfPipeline(_binary_config())
    pipeline.fit(data, target_column="target")
    X_test = data.drop(columns=["target"])

    with pytest.raises(ValueError, match="optimize_thresholds"):
        pipeline.predict(X_test, use_tuned_thresholds=True)


def test_predict_use_tuned_thresholds_applies_stored_thresholds(sample_classification_data):
    data = sample_classification_data.drop(columns=["category"])
    pipeline = SkyulfPipeline(_binary_config())
    pipeline.fit(data, target_column="target")
    _, _, X_val, y_val = pipeline.get_fitted_split(data, target_column="target")
    pipeline.optimize_thresholds(X_val, y_val, metric=lambda a, b: f1_score(a, b, average="macro"))

    X_test = data.drop(columns=["target"])
    tuned_preds = pipeline.predict(X_test, use_tuned_thresholds=True)
    assert len(tuned_preds) == len(X_test)
    assert set(np.unique(tuned_preds)).issubset(set(np.unique(data["target"])))


def test_predict_default_behavior_unchanged_when_flag_is_false(sample_classification_data):
    """Regression check: use_tuned_thresholds=False (the default) must behave
    exactly like predict() did before this feature existed."""
    data = sample_classification_data.drop(columns=["category"])
    pipeline = SkyulfPipeline(_binary_config())
    pipeline.fit(data, target_column="target")
    X_test = data.drop(columns=["target"])

    default_preds = pipeline.predict(X_test)
    explicit_false_preds = pipeline.predict(X_test, use_tuned_thresholds=False)
    np.testing.assert_array_equal(np.asarray(default_preds), np.asarray(explicit_false_preds))
