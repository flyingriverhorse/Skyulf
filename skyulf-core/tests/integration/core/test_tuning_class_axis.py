"""Probability scorers retain the trained label axis across holdout subsets."""

import numpy as np
import pandas as pd
import pytest
from sklearn.linear_model import LogisticRegression

from skyulf.modeling._tuning.engine import TuningCalculator
from skyulf.modeling._tuning.metrics import resolve_scorer
from skyulf.modeling._tuning.schemas import TuningConfig
from skyulf.modeling.classification import LogisticRegressionCalculator


@pytest.mark.parametrize("strategy", ["grid", "random", "halving_grid", "halving_random", "optuna"])
@pytest.mark.parametrize("metric", ["pr_auc_weighted", "pr_auc"])
def test_weighted_pr_tuning_accepts_holdout_missing_a_trained_class(strategy, metric):
    """Every tuning path must score a three-column model against a two-class holdout."""
    if strategy == "optuna":
        pytest.importorskip("optuna")
    X = pd.DataFrame({"x": np.arange(60, dtype=float)})
    y = pd.Series(np.repeat(["a", "b", "c"], 20))
    config = TuningConfig(
        strategy=strategy,
        metric=metric,
        search_space={"C": [1.0]},
        n_trials=1,
        cv_folds=2,
    )

    model, result = TuningCalculator(LogisticRegressionCalculator()).fit(
        X, y, config=config, validation_data=(X.iloc[:40], y.iloc[:40])
    )

    assert model.classes_.tolist() == ["a", "b", "c"]
    assert result.best_score == pytest.approx(1.0)
    assert result.scoring_metric == "pr_auc_weighted"
    assert len(result.trials) == 1
    assert result.trials[0]["score"] == pytest.approx(1.0)


@pytest.mark.parametrize("held_out_classes", [["a", "b"], ["a", "c"], ["b", "c"], ["b"]])
def test_weighted_pr_scorer_aligns_missing_first_middle_last_classes(held_out_classes):
    """Absent classes keep their probability columns and zero support in weighted AP."""
    X = pd.DataFrame({"x": np.arange(60, dtype=float)})
    y = pd.Series(np.repeat(["a", "b", "c"], 20))
    model = LogisticRegression().fit(X, y)
    held_out = y.isin(held_out_classes)
    scorer = resolve_scorer("pr_auc_weighted", y[held_out], "classification")

    score = scorer(model, X[held_out], y[held_out])

    assert score == pytest.approx(1.0)


@pytest.mark.parametrize("labels", [[0, 1], [1, 2], ["no", "yes"]])
@pytest.mark.parametrize("held_out_class", [None, 0, 1])
def test_weighted_pr_binary_scorer_uses_trained_positive_class(labels, held_out_class):
    """Binary probability columns must keep their meaning when the positive class is absent."""
    X = pd.DataFrame({"x": np.arange(40, dtype=float)})
    y = pd.Series(np.repeat(labels, 20))
    model = LogisticRegression().fit(X, y)
    held_out = (
        np.ones(len(y), dtype=bool) if held_out_class is None else y == labels[held_out_class]
    )
    scorer = resolve_scorer("pr_auc_weighted", y[held_out], "classification")

    score = scorer(model, X[held_out], y[held_out])

    assert score == pytest.approx(0.0 if held_out_class == 0 else 1.0)
