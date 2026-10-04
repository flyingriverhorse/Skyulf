"""Binary tuning uses the same positive class as evaluation and threshold tuning."""

import numpy as np
import pandas as pd
import pytest
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import f1_score, make_scorer

from skyulf.modeling._evaluation.metrics import calculate_classification_metrics
from skyulf.modeling._tuning.engine import TuningCalculator
from skyulf.modeling._tuning.metrics import resolve_scorer
from skyulf.modeling._tuning.refit import resolve_threshold_metric
from skyulf.modeling._tuning.schemas import TuningConfig
from skyulf.modeling.classification import LogisticRegressionCalculator


def _binary_holdout(labels):
    """Keep unequal class-specific F1 scores so choosing the wrong class is observable."""
    X = pd.DataFrame({"x": np.arange(40, dtype=float)})
    y = pd.Series([labels[0]] * 30 + [labels[1]] * 10)
    X_val = pd.DataFrame({"x": [0.0, 10.0, 20.0, 25.0, 29.0, 30.0, 35.0, 39.0]})
    y_val = pd.Series([labels[index] for index in [0, 0, 0, 0, 0, 1, 0, 1]])
    return X, y, X_val, y_val


@pytest.mark.parametrize("labels", [[0, 1], [1, 2], [-1, 1], ["no", "yes"]])
@pytest.mark.parametrize("metric", ["f1", "precision", "recall", "average_precision"])
def test_named_binary_scorers_follow_evaluation_positive_class(labels, metric):
    """Numeric label 1 must not override the model's second probability class."""
    X, y, X_val, y_val = _binary_holdout(labels)
    model = LogisticRegression().fit(X, y)
    scorer = resolve_scorer(metric, y, "classification")
    evaluated = calculate_classification_metrics(model, X_val, y_val)
    metric_key = "pr_auc" if metric == "average_precision" else metric
    assert scorer(model, X_val, y_val) == pytest.approx(evaluated[metric_key])


@pytest.mark.parametrize("strategy", ["grid", "random", "halving_grid", "halving_random", "optuna"])
def test_f1_tuning_matches_evaluation_and_threshold_metric(strategy):
    """All search strategies must rank {1,2} models using class 2's F1."""
    if strategy == "optuna":
        pytest.importorskip("optuna")
    X, y, X_val, y_val = _binary_holdout([1, 2])
    config = TuningConfig(strategy=strategy, metric="f1", search_space={"C": [1.0]}, n_trials=1)
    model, result = TuningCalculator(LogisticRegressionCalculator()).fit(
        X, y, config=config, validation_data=(X_val, y_val)
    )
    threshold_metric, _ = resolve_threshold_metric("f1", None, pos_label=model.classes_[1])
    evaluated = calculate_classification_metrics(model, X_val, y_val)
    assert result.best_score == pytest.approx(0.8)
    assert result.best_score == pytest.approx(evaluated["f1"])
    assert result.best_score == pytest.approx(threshold_metric(y_val, model.predict(X_val)))


def test_explicit_positive_class_scorers_and_threshold_metrics_are_preserved():
    """Callers choosing class 1 explicitly must keep that choice after default reconciliation."""
    X, y, X_val, y_val = _binary_holdout([1, 2])
    model = LogisticRegression().fit(X, y)
    explicit = make_scorer(f1_score, pos_label=1)
    resolved = resolve_scorer(explicit, y, "classification")
    threshold_metric, _ = resolve_threshold_metric("f1", None, pos_label=1)
    assert resolved(model, X_val, y_val) == pytest.approx(10 / 11)
    assert threshold_metric(y_val, model.predict(X_val)) == pytest.approx(10 / 11)
