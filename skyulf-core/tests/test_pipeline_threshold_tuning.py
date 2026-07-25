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
