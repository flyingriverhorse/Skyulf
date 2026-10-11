"""Regression coverage for the modeling findings from the 0.9.2 review."""

import numpy as np
import pandas as pd
import polars as pl
import pytest

from skyulf.data.dataset import SplitDataset
from skyulf.modeling._tuning.engine import TuningApplier, TuningCalculator
from skyulf.modeling._tuning.schemas import TuningConfig
from skyulf.modeling.base import StatefulEstimator
from skyulf.modeling.cross_validation import perform_cross_validation
from skyulf.pipeline import SkyulfPipeline
from skyulf.preprocessing.fold_adapter import FeatureEngineerFoldAdapter, MergedBranchFoldAdapter
from skyulf.registry import NodeRegistry


class FitOnlyEncoder:
    """Encode training labels while preserving evaluation labels, as custom adapters may do."""

    def fit_transform(self, X, y):
        """Use overlapping integer spaces to expose accidental double decoding."""
        return X, np.asarray(y) - 1

    def transform(self, X, y):
        """Leave the held-out target in its original space."""
        return X, y


@pytest.mark.parametrize("adapter_kind", ["fit_only", "builtin", "merged"])
def test_halving_preserves_overlapping_original_target_labels(adapter_kind):
    """Searcher fold scoring must retain class identity across every supported adapter."""
    X = pd.DataFrame({"x": np.tile(np.arange(20), 6)})
    y = pd.Series(np.where(X.x < 10, 1, 2), name="target")
    steps = [{"name": "labels", "transformer": "LabelEncoder", "params": {"columns": ["target"]}}]
    adapters = {
        "fit_only": FitOnlyEncoder(),
        "builtin": FeatureEngineerFoldAdapter(steps, "target"),
        "merged": MergedBranchFoldAdapter([steps, steps], "last_wins", "target"),
    }
    _, result = TuningCalculator(NodeRegistry.get_calculator("decision_tree_classifier")()).fit(
        X,
        y,
        TuningConfig(
            strategy="halving_grid",
            search_space={"max_depth": [1]},
            metric="accuracy",
            cv_folds=3,
            strategy_params={"min_resources": len(X)},
        ),
        preprocessing=adapters[adapter_kind],
    )
    assert result.best_score == pytest.approx(1.0)


@pytest.mark.parametrize(
    "labels,declared",
    [((1, 2), np.int64(1)), ((1.0, 2.0), np.float64(1)), (("no", "yes"), np.str_("no"))],
)
def test_numpy_positive_class_matches_its_python_scalar(labels, declared):
    """Python callers may supply NumPy labels without losing strict class type checking."""
    pipeline = SkyulfPipeline(
        {
            "preprocessing": [
                {"name": "labels", "transformer": "LabelEncoder", "params": {"columns": ["target"]}}
            ],
            "modeling": {"type": "logistic_regression"},
            "decision_threshold": {"positive_class": declared},
        }
    )
    pipeline.fit(
        pd.DataFrame({"x": range(24), "target": [labels[0]] * 12 + [labels[1]] * 12}), "target"
    )
    assert pipeline._decision_positive_class() == 0


@pytest.mark.parametrize("target_type", [list, tuple])
@pytest.mark.parametrize("cv_type", ["group_k_fold", "stratified_group_k_fold"])
def test_explicit_group_cv_accepts_python_target_sequences(target_type, cv_type):
    """Explicit split policies must accept the same positional targets as ordinary KFold."""
    X = pd.DataFrame({"x": np.tile(np.arange(6), 8), "group": np.repeat(np.arange(8), 6)})
    y = np.where(X.x < 3, 0, 1)
    calculator = NodeRegistry.get_calculator("decision_tree_classifier")()
    applier = NodeRegistry.get_applier("decision_tree_classifier")()
    report = perform_cross_validation(
        calculator,
        applier,
        X,
        target_type(y),
        {"max_depth": 1},
        cv_type=cv_type,
        group_column="group",
        n_folds=3,
    )
    assert report["aggregated_metrics"]["accuracy"]["mean"] == pytest.approx(1.0)
    assert len(report["folds"]) == 3


@pytest.mark.parametrize("engine", ["pandas", "polars"])
@pytest.mark.parametrize("separated", [False, True])
@pytest.mark.parametrize("preprocess", [False, True])
def test_tuned_fit_treats_empty_validation_as_absent(engine, separated, preprocess):
    """An empty split must use training CV instead of failing every holdout candidate."""
    train = pd.DataFrame({"x": np.tile(np.arange(10), 6), "target": np.tile(np.arange(10) >= 5, 6)})
    if engine == "polars":
        train = pl.from_pandas(train)
    validation = train[:0]
    if separated:
        validation = (validation[["x"]], validation["target"])
    calculator = NodeRegistry.get_calculator("decision_tree_classifier")()
    estimator = StatefulEstimator(
        TuningCalculator(calculator),
        TuningApplier(NodeRegistry.get_applier("decision_tree_classifier")()),
        "tuned",
    )
    features = train[["x"]]
    labels = train["target"]
    fold_args = (
        {
            "preprocessing": FeatureEngineerFoldAdapter([], "target"),
            "preprocessing_train": (features, labels),
            "preprocessing_validation": (features[:0], labels[:0]),
        }
        if preprocess
        else {}
    )
    predictions = estimator.fit_predict(
        SplitDataset(train=train, test=train[:0], validation=validation),
        "target",
        {"strategy": "grid", "search_space": {"max_depth": [1]}, "cv_folds": 3},
        **fold_args,
    )
    assert set(predictions) == {"train"}
    assert len(predictions["train"]) == len(train)
    assert estimator.model is not None
    assert estimator.model[1].best_score == pytest.approx(1.0)


def test_nested_scoring_metric_uses_sample_standard_deviation():
    """The selected metric must use the same fold dispersion convention as its neighbours."""
    rng = np.random.default_rng(23)
    X = pd.DataFrame({"x": rng.normal(size=72)})
    y = pd.Series((X.x + rng.normal(size=72) > 0).astype(int))
    _, result = TuningCalculator(NodeRegistry.get_calculator("decision_tree_classifier")()).fit(
        X,
        y,
        TuningConfig(
            strategy="grid",
            search_space={"max_depth": [1]},
            metric="accuracy",
            cv_type="nested_cv",
            cv_folds=3,
        ),
    )
    report = result.nested_cv
    assert report is not None
    scores = [fold["outer_score"] for fold in report["folds"]]
    assert np.std(scores, ddof=1) > 0
    assert report["aggregated_metrics"]["accuracy"]["std"] == pytest.approx(np.std(scores, ddof=1))
