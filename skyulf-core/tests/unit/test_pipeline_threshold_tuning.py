"""Tests for SkyulfPipeline.optimize_thresholds() and predict(use_tuned_thresholds=...)."""

from typing import Any

import numpy as np
import pytest
from sklearn.metrics import f1_score
from sklearn.model_selection import train_test_split

from skyulf.modeling._evaluation.thresholds import apply_thresholds
from skyulf.pipeline import SkyulfPipeline


def _binary_config(test_size=0.25, random_state=42):
    """Fit imputation only on training rows before testing threshold behavior."""
    return {
        "preprocessing": [
            {
                "name": "split",
                "transformer": "TrainTestSplitter",
                "params": {"test_size": test_size, "random_state": random_state},
            },
            {
                "name": "imputer",
                "transformer": "SimpleImputer",
                "params": {"strategy": "mean"},
            },
        ],
        "modeling": {"type": "logistic_regression"},
    }


def _scaling_config(test_size=0.25, random_state=42):
    """Binary config whose preprocessing is *not* idempotent.

    A second pass through ``StandardScaler`` shifts the features again, so a
    double transform shows up in the predicted probabilities. Mean imputation
    cannot serve here: imputing an already-imputed frame changes nothing.
    """
    return {
        "preprocessing": [
            {
                "name": "split",
                "transformer": "TrainTestSplitter",
                "params": {"test_size": test_size, "random_state": random_state},
            },
            {
                "name": "imputer",
                "transformer": "SimpleImputer",
                "params": {"strategy": "mean"},
            },
            {
                "name": "scaler",
                "transformer": "StandardScaler",
                "params": {"columns": ["feature1", "feature2"]},
            },
        ],
        "modeling": {"type": "logistic_regression"},
    }


def _raw_holdout(data, target_column="target", test_size=0.25, random_state=0):
    """Carve the raw validation holdout ``optimize_thresholds()`` expects.

    Returns ``(train_raw, X_val, y_val)`` with the target still on ``train_raw``
    and already off ``X_val``, so both halves are untransformed rows.
    """
    train_raw, val_raw = train_test_split(data, test_size=test_size, random_state=random_state)
    return train_raw, val_raw.drop(columns=[target_column]), val_raw[target_column]


def _macro_f1(y_true, y_pred):
    """Macro F1, the caller-supplied metric the parity test maximizes."""
    return f1_score(y_true, y_pred, average="macro")


def test_optimize_thresholds_returns_dict_covering_both_classes(sample_classification_data):
    """The tuned dict carries one cutoff per class present in the holdout."""
    data = sample_classification_data.drop(columns=["category"])
    pipeline = SkyulfPipeline(_binary_config())
    train_raw, X_val, y_val = _raw_holdout(data)
    pipeline.fit(train_raw, target_column="target")

    def metric(y_true, y_pred):
        return f1_score(y_true, y_pred, average="macro")

    thresholds = pipeline.optimize_thresholds(X_val, y_val, metric=metric)
    assert set(thresholds.keys()) == set(np.unique(y_val))


def test_optimize_thresholds_stores_result_on_instance(sample_classification_data):
    """The search result is kept on the instance for predict(use_tuned_thresholds=True)."""
    data = sample_classification_data.drop(columns=["category"])
    pipeline = SkyulfPipeline(_binary_config())
    train_raw, X_val, y_val = _raw_holdout(data)
    pipeline.fit(train_raw, target_column="target")

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
    train_raw, X_val, y_val = _raw_holdout(data)
    pipeline.fit(train_raw, target_column="target")
    pipeline.optimize_thresholds(X_val, y_val, metric=lambda a, b: f1_score(a, b, average="macro"))

    X_test = data.drop(columns=["target"])
    tuned_preds = pipeline.predict(X_test, use_tuned_thresholds=True)
    assert len(tuned_preds) == len(X_test)
    assert set(np.unique(tuned_preds)).issubset(set(np.unique(data["target"])))


def test_optimize_thresholds_sees_the_same_probabilities_as_predict(
    sample_classification_data, monkeypatch
):
    """Tuning and predict(use_tuned_thresholds=True) must feed the model identical probabilities.

    ``optimize_thresholds()`` runs the fitted preprocessing on ``X_val`` exactly
    once, the same single pass ``predict()`` makes. Handing it the
    already-preprocessed frames ``get_fitted_split()`` returns transforms the
    holdout a second time, so the cutoffs get fitted against a distribution
    inference never reproduces — silently, because the returned dict still looks
    plausible.
    """
    data = sample_classification_data.drop(columns=["category"])
    pipeline = SkyulfPipeline(_scaling_config())
    train_raw, X_val, y_val = _raw_holdout(data)
    pipeline.fit(train_raw, target_column="target")

    seen: list[np.ndarray] = []
    estimator = pipeline.model_estimator
    assert estimator is not None, "fit() must install the model estimator"
    applier = estimator.applier
    real_predict_proba = applier.predict_proba

    def spy(transformed, model):
        proba = real_predict_proba(transformed, model)
        seen.append(np.asarray(proba))
        return proba

    monkeypatch.setattr(applier, "predict_proba", spy)

    pipeline.optimize_thresholds(X_val, y_val, metric=_macro_f1)
    assert len(seen) == 1, "tuning must transform X_val exactly once"
    tuning_proba = seen[0]

    seen.clear()
    pipeline.predict(X_val, use_tuned_thresholds=True)
    assert len(seen) == 1, "inference must transform its input exactly once"
    inference_proba = seen[0]

    np.testing.assert_allclose(tuning_proba, inference_proba)

    # Teeth: the same rows handed over pre-transformed — what get_fitted_split()
    # returns — make tuning see a different distribution than inference does.
    seen.clear()
    pipeline.optimize_thresholds(
        pipeline.feature_engineer.transform(X_val), y_val, metric=_macro_f1
    )
    assert not np.allclose(seen[0], inference_proba)


def test_predict_default_behavior_unchanged_when_flag_is_false(sample_classification_data):
    """Regression check: use_tuned_thresholds=False (the default) must behave
    exactly like predict() did before this feature existed.
    """
    data = sample_classification_data.drop(columns=["category"])
    pipeline = SkyulfPipeline(_binary_config())
    pipeline.fit(data, target_column="target")
    X_test = data.drop(columns=["target"])

    default_preds = pipeline.predict(X_test)
    explicit_false_preds = pipeline.predict(X_test, use_tuned_thresholds=False)
    np.testing.assert_array_equal(np.asarray(default_preds), np.asarray(explicit_false_preds))


@pytest.mark.parametrize("labels", [[0, 1], ["no", "yes"], ["red", "green", "blue"]])
def test_tuned_classifier_supports_post_training_thresholds(sample_classification_data, labels):
    """Tuning artifacts must preserve class labels through threshold search and prediction."""
    data = sample_classification_data.drop(columns=["category"])
    data["target"] = np.asarray(labels)[np.arange(len(data)) % len(labels)]
    config: dict[str, Any] = _scaling_config()
    config["modeling"] = {
        "type": "hyperparameter_tuner",
        "base_model": {"type": "logistic_regression"},
        "strategy": "grid",
        "search_space": {"C": [0.1, 1.0]},
        "metric": "f1_macro",
        "cv_folds": 3,
    }
    pipeline = SkyulfPipeline(config)
    train_raw, X_val, y_val = _raw_holdout(data)
    pipeline.fit(train_raw, target_column="target")
    default_predictions = pipeline.predict(X_val)

    thresholds = pipeline.optimize_thresholds(X_val, y_val, metric=_macro_f1)
    assert set(thresholds) == set(labels)

    estimator = pipeline.model_estimator
    assert estimator is not None and isinstance(estimator.model, tuple)
    classifier, _ = estimator.model
    transformed = pipeline.feature_engineer.transform(X_val)
    probabilities = classifier.predict_proba(np.asarray(transformed))
    expected = apply_thresholds(probabilities, thresholds, classes=classifier.classes_)
    np.testing.assert_array_equal(pipeline.predict(X_val, use_tuned_thresholds=True), expected)
    np.testing.assert_array_equal(pipeline.predict(X_val), default_predictions)


def test_tuned_regressor_rejects_threshold_optimization(sample_regression_data):
    """Unwrapping a tuning artifact must not admit models without class probabilities."""
    data = sample_regression_data.drop(columns=["category"])
    config: dict[str, Any] = _binary_config()
    config["modeling"] = {
        "type": "hyperparameter_tuner",
        "base_model": {"type": "ridge_regression"},
        "strategy": "grid",
        "search_space": {"alpha": [1.0]},
        "metric": "r2",
        "cv_folds": 3,
    }
    pipeline = SkyulfPipeline(config)
    train_raw, X_val, y_val = _raw_holdout(data)
    pipeline.fit(train_raw, target_column="target")

    with pytest.raises(ValueError, match="threshold tuning requires a classifier"):
        pipeline.optimize_thresholds(X_val, y_val, metric=_macro_f1)
