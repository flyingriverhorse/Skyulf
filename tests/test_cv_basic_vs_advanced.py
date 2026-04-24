"""
Compare cross-validation metrics between basic training and advanced tuning.

Verifies:
1. Both paths produce cv_* metrics when cv_enabled=True.
2. nested_cv with advanced tuning does NOT trigger _perform_nested_cv
   (inner loop already ran during tuning search).
3. Metrics from both paths are structurally identical.
"""

import numpy as np
import pandas as pd
import pytest

from skyulf.data.dataset import SplitDataset
from skyulf.modeling.base import StatefulEstimator
from skyulf.modeling.classification import (
    LogisticRegressionApplier,
    LogisticRegressionCalculator,
)
from skyulf.modeling.regression import (
    RidgeRegressionApplier,
    RidgeRegressionCalculator,
)
from skyulf.modeling.tuning import TuningCalculator, TuningConfig
from skyulf.modeling.tuning.engine import TuningApplier


@pytest.fixture
def classification_dataset() -> SplitDataset:
    rng = np.random.RandomState(42)
    n = 100
    X = rng.randn(n, 3)
    y = (X[:, 0] + X[:, 1] > 0).astype(int)
    df = pd.DataFrame({"f1": X[:, 0], "f2": X[:, 1], "f3": X[:, 2], "target": y})
    train = df.iloc[:80].reset_index(drop=True)
    test = df.iloc[80:].reset_index(drop=True)
    return SplitDataset(train=train, test=test)


@pytest.fixture
def regression_dataset() -> SplitDataset:
    rng = np.random.RandomState(42)
    n = 100
    X = rng.randn(n, 3)
    y = X[:, 0] * 2.0 + X[:, 1] * 0.5 + rng.randn(n) * 0.1
    df = pd.DataFrame({"f1": X[:, 0], "f2": X[:, 1], "f3": X[:, 2], "target": y})
    train = df.iloc[:80].reset_index(drop=True)
    test = df.iloc[80:].reset_index(drop=True)
    return SplitDataset(train=train, test=test)


def _make_basic_estimator() -> StatefulEstimator:
    return StatefulEstimator(
        LogisticRegressionCalculator(),
        LogisticRegressionApplier(),
        "basic_node",
    )


# -- Basic Training CV Tests --


@pytest.mark.parametrize("cv_type", ["k_fold", "stratified_k_fold", "nested_cv"])
def test_basic_training_cv(classification_dataset: SplitDataset, cv_type: str) -> None:
    """Basic training produces cv metrics for all CV types."""
    est = _make_basic_estimator()
    result = est.cross_validate(
        classification_dataset,
        "target",
        {},
        n_folds=5,
        cv_type=cv_type,
        shuffle=True,
        random_state=42,
    )

    agg = result["aggregated_metrics"]
    assert len(agg) > 0, f"No aggregated metrics for cv_type={cv_type}"

    for metric_name, stats in agg.items():
        assert "mean" in stats, f"Missing mean for {metric_name}"
        assert "std" in stats, f"Missing std for {metric_name}"
        assert 0.0 <= stats["mean"] <= 1.0, f"{metric_name} mean out of range"

    assert "cv_config" in result
    if cv_type == "nested_cv":
        assert result["cv_config"]["cv_type"] == "nested_cv"
        assert "inner_folds" in result["cv_config"]


# -- Advanced Tuning CV Tests --


def test_advanced_tuning_cv_kfold(classification_dataset: SplitDataset) -> None:
    """Advanced tuning with k_fold produces cv metrics via post-tuning CV."""
    calc = LogisticRegressionCalculator()
    applier = LogisticRegressionApplier()

    tuning_config = TuningConfig(
        strategy="grid",
        metric="accuracy",
        search_space={"C": [0.1, 1.0]},
        cv_enabled=True,
        cv_folds=3,
        cv_type="k_fold",
        cv_shuffle=True,
        cv_random_state=42,
    )

    tuner = TuningCalculator(calc)
    tuner_applier = TuningApplier(applier)

    # Run tuning (this uses inner CV to score candidates)
    tuning_est = StatefulEstimator(tuner, tuner_applier, "tuning_node")
    tuning_est.fit_predict(
        classification_dataset,
        "target",
        tuning_config.__dict__,
    )

    # Extract best params
    model_artifact = tuning_est.model
    assert isinstance(model_artifact, tuple), "Expected (model, tuning_result) tuple"
    _, tuning_result = model_artifact
    assert tuning_result is not None
    assert tuning_result.best_params is not None
    best_params = tuning_result.best_params

    # Post-tuning CV (same as engine.py does)
    cv_est = StatefulEstimator(calc, applier, "cv_node")
    result = cv_est.cross_validate(
        classification_dataset,
        "target",
        {"params": best_params},
        n_folds=3,
        cv_type="k_fold",
        shuffle=True,
        random_state=42,
    )

    agg = result["aggregated_metrics"]
    assert len(agg) > 0, "No CV metrics from post-tuning evaluation"
    assert result["cv_config"]["cv_type"] == "k_fold"


def test_advanced_tuning_nested_cv_uses_standard_fold(
    classification_dataset: SplitDataset,
) -> None:
    """
    When nested_cv is selected with advanced tuning, the post-tuning CV
    should use stratified_k_fold (not nested_cv) because the inner loop
    already ran during the tuning search.

    This simulates what engine.py does: override nested_cv -> stratified_k_fold.
    """
    calc = LogisticRegressionCalculator()
    applier = LogisticRegressionApplier()

    tuning_config = TuningConfig(
        strategy="grid",
        metric="accuracy",
        search_space={"C": [0.1, 1.0]},
        cv_enabled=True,
        cv_folds=5,
        cv_type="nested_cv",
        cv_shuffle=True,
        cv_random_state=42,
    )

    tuner = TuningCalculator(calc)
    tuner_applier = TuningApplier(applier)

    # Run tuning — inner CV (3 folds) scores each candidate during search
    tuning_est = StatefulEstimator(tuner, tuner_applier, "tuning_node")
    tuning_est.fit_predict(
        classification_dataset,
        "target",
        tuning_config.__dict__,
    )

    _, tuning_result = tuning_est.model  # type: ignore[misc]
    best_params = tuning_result.best_params

    # Simulate engine.py fix: override nested_cv -> stratified_k_fold
    post_cv_type = "nested_cv"
    if post_cv_type == "nested_cv":
        post_cv_type = "stratified_k_fold"

    cv_est = StatefulEstimator(calc, applier, "cv_node")
    result = cv_est.cross_validate(
        classification_dataset,
        "target",
        {"params": best_params},
        n_folds=5,
        cv_type=post_cv_type,
        shuffle=True,
        random_state=42,
    )

    agg = result["aggregated_metrics"]
    assert len(agg) > 0, "No CV metrics"

    # Confirm it ran as stratified_k_fold, NOT nested_cv
    assert result["cv_config"]["cv_type"] == "stratified_k_fold"
    assert (
        "inner_folds" not in result["cv_config"]
    ), "Should NOT have inner_folds — that means _perform_nested_cv ran"


def test_basic_vs_advanced_metric_keys_match(
    classification_dataset: SplitDataset,
) -> None:
    """Both basic and advanced produce the same set of cv metric names."""
    calc = LogisticRegressionCalculator()
    applier = LogisticRegressionApplier()

    # Basic CV
    basic_est = StatefulEstimator(calc, applier, "basic_node")
    basic_result = basic_est.cross_validate(
        classification_dataset,
        "target",
        {},
        n_folds=3,
        cv_type="k_fold",
        shuffle=True,
        random_state=42,
    )

    # Advanced tuning + post-tuning CV
    tuning_config = TuningConfig(
        strategy="grid",
        metric="accuracy",
        search_space={"C": [0.1, 1.0]},
        cv_enabled=True,
        cv_folds=3,
        cv_type="k_fold",
    )
    tuner = TuningCalculator(calc)
    tuner_applier = TuningApplier(applier)
    tuning_est = StatefulEstimator(tuner, tuner_applier, "tuning_node")
    tuning_est.fit_predict(
        classification_dataset,
        "target",
        tuning_config.__dict__,
    )
    _, tuning_result = tuning_est.model  # type: ignore[misc]
    best_params = tuning_result.best_params

    adv_est = StatefulEstimator(calc, applier, "cv_node")
    adv_result = adv_est.cross_validate(
        classification_dataset,
        "target",
        {"params": best_params},
        n_folds=3,
        cv_type="k_fold",
        shuffle=True,
        random_state=42,
    )

    basic_keys = set(basic_result["aggregated_metrics"].keys())
    adv_keys = set(adv_result["aggregated_metrics"].keys())

    assert basic_keys == adv_keys, f"Metric keys differ: basic={basic_keys}, advanced={adv_keys}"


# -- Regression: nested_cv -> k_fold downgrade --


def test_regression_nested_cv_uses_k_fold(
    regression_dataset: SplitDataset,
) -> None:
    """
    For regression, nested_cv downgrade should use k_fold (not stratified).
    Simulates engine.py logic: problem_type != classification -> k_fold.
    """
    calc = RidgeRegressionCalculator()
    applier = RidgeRegressionApplier()

    tuning_config = TuningConfig(
        strategy="grid",
        metric="r2",
        search_space={"alpha": [0.1, 1.0, 10.0]},
        cv_enabled=True,
        cv_folds=5,
        cv_type="nested_cv",
    )

    tuner = TuningCalculator(calc)
    tuner_applier = TuningApplier(applier)
    tuning_est = StatefulEstimator(tuner, tuner_applier, "tuning_node")
    tuning_est.fit_predict(
        regression_dataset,
        "target",
        tuning_config.__dict__,
    )

    _, tuning_result = tuning_est.model  # type: ignore[misc]
    best_params = tuning_result.best_params

    # Engine.py logic: regression -> k_fold
    is_classification = getattr(calc, "problem_type", "") == "classification"
    post_cv_type = "stratified_k_fold" if is_classification else "k_fold"

    assert post_cv_type == "k_fold", "Regression should downgrade to k_fold, not stratified"

    cv_est = StatefulEstimator(calc, applier, "cv_node")
    result = cv_est.cross_validate(
        regression_dataset,
        "target",
        {"params": best_params},
        n_folds=5,
        cv_type=post_cv_type,
        shuffle=True,
        random_state=42,
    )

    agg = result["aggregated_metrics"]
    assert len(agg) > 0, "No CV metrics for regression"
    assert result["cv_config"]["cv_type"] == "k_fold"
    assert "inner_folds" not in result["cv_config"]


def test_classification_nested_cv_uses_stratified(
    classification_dataset: SplitDataset,
) -> None:
    """
    For classification, nested_cv downgrade should use stratified_k_fold.
    Simulates engine.py logic: problem_type == classification -> stratified_k_fold.
    """
    calc = LogisticRegressionCalculator()
    applier = LogisticRegressionApplier()

    tuning_config = TuningConfig(
        strategy="grid",
        metric="accuracy",
        search_space={"C": [0.1, 1.0]},
        cv_enabled=True,
        cv_folds=5,
        cv_type="nested_cv",
    )

    tuner = TuningCalculator(calc)
    tuner_applier = TuningApplier(applier)
    tuning_est = StatefulEstimator(tuner, tuner_applier, "tuning_node")
    tuning_est.fit_predict(
        classification_dataset,
        "target",
        tuning_config.__dict__,
    )

    _, tuning_result = tuning_est.model  # type: ignore[misc]
    best_params = tuning_result.best_params

    # Engine.py logic: classification -> stratified_k_fold
    is_classification = getattr(calc, "problem_type", "") == "classification"
    post_cv_type = "stratified_k_fold" if is_classification else "k_fold"

    assert (
        post_cv_type == "stratified_k_fold"
    ), "Classification should downgrade to stratified_k_fold"

    cv_est = StatefulEstimator(calc, applier, "cv_node")
    result = cv_est.cross_validate(
        classification_dataset,
        "target",
        {"params": best_params},
        n_folds=5,
        cv_type=post_cv_type,
        shuffle=True,
        random_state=42,
    )

    agg = result["aggregated_metrics"]
    assert len(agg) > 0, "No CV metrics for classification"
    assert result["cv_config"]["cv_type"] == "stratified_k_fold"
    assert "inner_folds" not in result["cv_config"]
