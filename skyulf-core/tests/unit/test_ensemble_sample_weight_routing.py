"""Exercise real composite fitting across direct, cross-validation and tuning routes."""

from collections import Counter
from contextlib import ExitStack
from functools import wraps
from unittest.mock import patch

import numpy as np
import pandas as pd
import pytest
from sklearn.calibration import CalibratedClassifierCV, _SigmoidCalibration
from sklearn.ensemble import (
    StackingClassifier,
    StackingRegressor,
    VotingClassifier,
    VotingRegressor,
)
from sklearn.linear_model import LinearRegression, LogisticRegression, Ridge
from sklearn.model_selection import check_cv
from sklearn.naive_bayes import GaussianNB

from skyulf.modeling._tuning.engine import TuningCalculator
from skyulf.modeling._tuning.schemas import TuningConfig
from skyulf.modeling.cross_validation import perform_cross_validation
from skyulf.pipeline import SkyulfPipeline
from skyulf.registry import NodeRegistry

KINDS = [
    "voting_classifier",
    "stacking_classifier",
    "calibrated_classifier",
    "voting_classifier_calibrated",
    "stacking_classifier_calibrated",
    "voting_regressor",
    "stacking_regressor",
]
ROUTES = ["direct", "cv", "grid", "random", "halving_grid", "halving_random", "optuna"]
ROUTES += ["nested_" + strategy for strategy in ROUTES[2:]]
ROUTES += ["pipeline_grid", "pipeline_optuna", "threshold_nested_grid", "threshold_nested_optuna"]
COMPOSITES = (
    VotingClassifier,
    VotingRegressor,
    StackingClassifier,
    StackingRegressor,
    CalibratedClassifierCV,
)


def setup(kind):
    """Build two real base estimators and a distinct identifiable stacking final estimator."""
    node = kind.removesuffix("_calibrated")
    calculator = NodeRegistry.get_calculator(node)()
    classifier = calculator.problem_type == "classification"
    if kind == "calibrated_classifier":
        params = {"base_estimator": "logistic_regression", "cv": 2, "n_jobs": 1}
        space = {"cv": [2], "estimator__C": [0.7, 1.3]}
    else:
        params = {
            "base_estimators": ["logistic_regression", "gaussian_nb"]
            if classifier
            else ["ridge", "linear_regression"],
            "n_jobs": 1,
            "cv": 2,
        }
        if classifier:
            params["base_estimator_params"] = {"logistic_regression": {"class_weight": "balanced"}}
        if kind.startswith("stacking"):
            params["final_estimator"] = "logistic_regression" if classifier else "ridge"
            params["final_estimator_params"] = {"C": 0.1234} if classifier else {"alpha": 0.1234}
        if kind.endswith("_calibrated"):
            params.update(calibrate_base_models=True, calibration_cv=2)
        key = "logistic_regression__" if classifier else "ridge__"
        key += "estimator__" if kind.endswith("_calibrated") else ""
        key += "C" if classifier else "alpha"
        space = {key: [0.7, 1.3]}
    return calculator, params, space, classifier, node


def instrument(stack, counts):
    """Observe genuine leaf, stacking-final and calibration-fold fit weights."""
    meta_context = []
    calibration_context = []

    def wrap_meta(cls):
        """Observe parent fit rows while retaining the original sklearn signature."""
        original = cls.fit

        @wraps(original)
        def fit(model, X, y, sample_weight=None, **kwargs):
            """Assert aligned row weights before executing the original fit."""
            assert sample_weight is not None
            rows = np.asarray(X)[:, 0]
            np.testing.assert_array_equal(sample_weight, rows + 1)
            counts[cls.__name__] += 1
            is_stacking = isinstance(model, (StackingClassifier, StackingRegressor))
            is_calibration = isinstance(model, CalibratedClassifierCV)
            if is_stacking:
                meta_context.append((np.asarray(y), np.asarray(sample_weight)))
            if is_calibration:
                cv = check_cv(model.cv, y, classifier=True)
                calibration_context.append(
                    [np.asarray(sample_weight)[test] for _, test in cv.split(X, y)]
                )
            try:
                return original(model, X, y, sample_weight=sample_weight, **kwargs)
            finally:
                if is_stacking:
                    meta_context.pop()
                if is_calibration:
                    calibration_context.pop()

        stack.enter_context(patch.object(cls, "fit", fit))

    for cls in COMPOSITES:
        wrap_meta(cls)

    def wrap_leaf(cls):
        """Observe actual leaf weights and identify stacking final fits."""
        original = cls.fit

        @wraps(original)
        def fit(model, X, y, sample_weight=None, **kwargs):
            """Assert aligned row weights before executing the original fit."""
            assert sample_weight is not None
            final = (isinstance(model, LogisticRegression) and model.C == 0.1234) or (
                isinstance(model, Ridge) and model.alpha == 0.1234
            )
            if final:
                expected_y, expected = meta_context[-1]
                np.testing.assert_array_equal(y, expected_y)
                counts["stacking_final"] += 1
            else:
                expected = np.asarray(X)[:, 0] + 1
            np.testing.assert_array_equal(sample_weight, expected)
            counts[cls.__name__] += 1
            return original(model, X, y, sample_weight=sample_weight, **kwargs)

        stack.enter_context(patch.object(cls, "fit", fit))

    for cls in (LogisticRegression, GaussianNB, Ridge, LinearRegression):
        wrap_leaf(cls)
    original_sigmoid = _SigmoidCalibration.fit

    def sigmoid(model, X, y, sample_weight=None):
        """Assert calibration uses its held-out fold weights."""
        assert sample_weight is not None
        assert any(np.array_equal(sample_weight, expected) for expected in calibration_context[-1])
        counts["sigmoid_calibration"] += 1
        return original_sigmoid(model, X, y, sample_weight=sample_weight)

    stack.enter_context(patch.object(_SigmoidCalibration, "fit", sigmoid))


def exercise(kind, route):
    """Use real Skyulf entrypoints and assert weights in actual sklearn fits."""
    calculator, params, space, classifier, node = setup(kind)
    ids = np.random.default_rng(11).permutation(96)
    X = pd.DataFrame({"row": ids.astype(float), "feature": np.sin(ids)}, index=np.zeros(96))
    y = pd.Series(ids % 2 if classifier else ids / 10 + np.cos(ids), index=X.index)
    weights = ids + 1.0
    counts = Counter()
    with ExitStack() as stack:
        instrument(stack, counts)
        if route.startswith("pipeline_"):
            frame = X.copy()
            frame.loc[frame.row.astype(int) % 7 == 0, "feature"] = np.nan
            frame["target"] = y.to_numpy()
            pipeline = SkyulfPipeline(
                {
                    "preprocessing": [
                        {
                            "name": "impute",
                            "transformer": "KNNImputer",
                            "params": {"columns": ["row", "feature"], "n_neighbors": 3},
                        }
                    ],
                    "modeling": {
                        "type": "hyperparameter_tuner",
                        "base_model": {"type": node, "params": params},
                        "strategy": route.removeprefix("pipeline_"),
                        "search_space": space,
                        "metric": "accuracy" if classifier else "mse",
                        "cv_folds": 2,
                        "cv_type": "stratified_k_fold" if classifier else "k_fold",
                        "n_trials": 2,
                        "n_jobs": 1,
                    },
                }
            )
            pipeline.fit(frame, "target", sample_weight=weights)
            assert len(pipeline.predict(frame.drop(columns=["target"]))) == len(frame)
        elif route == "direct":
            model = calculator.fit(X, y, {"params": params}, sample_weight=weights)
            assert len(model.predict(X.to_numpy())) == len(y)
        elif route == "cv":
            result = perform_cross_validation(
                calculator,
                NodeRegistry.get_applier(node)(),
                X,
                y,
                {"params": params},
                n_folds=2,
                cv_type="stratified_k_fold" if classifier else "k_fold",
                sample_weight=weights,
            )
            assert result["aggregated_metrics"]
        else:
            calculator.prepare_tuning_params(params)
            tuning_route = route.removeprefix("threshold_")
            config = TuningConfig(
                strategy=tuning_route.removeprefix("nested_"),
                search_space=space,
                metric="accuracy" if classifier else "mse",
                cv_folds=2,
                cv_inner_folds=2,
                cv_type="nested_cv"
                if tuning_route.startswith("nested_")
                else "stratified_k_fold"
                if classifier
                else "k_fold",
                n_jobs=1,
                n_trials=2,
                strategy_params={"min_resources": 32},
                tune_threshold=route.startswith("threshold_"),
            )
            model, result = TuningCalculator(calculator).fit(X, y, config, sample_weight=weights)
            assert np.isfinite(result.best_score)
            assert len(model.predict(X.to_numpy())) == len(y)
            assert not tuning_route.startswith("nested_") or result.nested_cv is not None
            assert not route.startswith("threshold_") or result.decision_thresholds is not None
        assert sum(counts.values()) > 0
        if kind.startswith("stacking"):
            assert counts["stacking_final"] > 0
        if "calibrated" in kind:
            assert counts["sigmoid_calibration"] > 0
    return dict(counts)


@pytest.mark.parametrize(
    "kind,route",
    [
        (kind, route)
        for kind in KINDS
        for route in ROUTES
        if not (route.startswith("threshold_") and "regressor" in kind)
    ],
)
def test_composite_weight_routing(kind, route):
    """Every leaf, calibration fold and stacking final receives aligned user weights."""
    assert exercise(kind, route)


@pytest.mark.parametrize("kind", KINDS)
def test_composite_weighted_fit_without_instrumentation(kind):
    """Original sklearn signatures must work without spies changing introspection."""
    calculator, params, _, classifier, _ = setup(kind)
    X = pd.DataFrame({"x": np.arange(40, dtype=float), "z": np.sin(np.arange(40))})
    y = pd.Series(np.arange(40) % 2 if classifier else np.arange(40) / 3)
    model = calculator.fit(X, y, {"params": params}, sample_weight=np.arange(40) + 1.0)
    assert len(model.predict(X.to_numpy())) == 40


@pytest.mark.parametrize(
    "kind",
    ["calibrated_classifier", "voting_classifier_calibrated", "stacking_classifier_calibrated"],
)
@pytest.mark.parametrize("method", ["sigmoid", "isotonic"])
def test_multiclass_calibration_with_numeric_class_and_row_weights(kind, method):
    """Three-class calibration preserves combined explicit class and row weighting."""
    calculator, params, _, _, _ = setup(kind)
    X = pd.DataFrame({"x": np.arange(90, dtype=float), "z": np.sin(np.arange(90))})
    y = pd.Series(np.arange(90) % 3)
    params.update(class_weight={0: 1.0, 1: 2.0, 2: 0.5})
    params["method" if kind == "calibrated_classifier" else "calibration_method"] = method
    weights = np.arange(90) + 1.0
    model = calculator.fit(X, y, {"params": params}, sample_weight=weights)
    expected = calculator.fit(
        X,
        y,
        {"params": {**params, "class_weight": None}},
        sample_weight=weights * np.array([1.0, 2.0, 0.5])[y.to_numpy()],
    )
    np.testing.assert_allclose(
        model.predict_proba(X.to_numpy()), expected.predict_proba(X.to_numpy())
    )


@pytest.mark.parametrize("strategy", ["grid", "random", "halving_grid", "halving_random", "optuna"])
def test_tuning_rejects_unsupported_final_before_fitting(strategy):
    """Every tuning strategy rejects a configured unsupported stacking final learner."""
    from skyulf.modeling._sample_weights import SampleWeightError

    calculator, params, space, _, _ = setup("stacking_classifier")
    params["final_estimator"] = "knn"
    params.pop("final_estimator_params")
    calculator.prepare_tuning_params(params)
    X = pd.DataFrame({"x": np.arange(40, dtype=float)})
    y = pd.Series(np.arange(40) % 2)
    config = TuningConfig(
        strategy=strategy,
        search_space=space,
        metric="accuracy",
        cv_folds=2,
        n_jobs=1,
        n_trials=1,
        strategy_params={"min_resources": 20},
    )
    with pytest.raises(SampleWeightError, match="KNeighborsClassifier"):
        TuningCalculator(calculator).fit(X, y, config, sample_weight=np.ones(40))


@pytest.mark.parametrize("candidate", ["final_estimator", "logistic_regression"])
def test_tuning_rejects_unsupported_structural_candidate(candidate):
    """A search-space replacement cannot bypass recursive support of fixed defaults."""
    from sklearn.neighbors import KNeighborsClassifier

    from skyulf.modeling._sample_weights import SampleWeightError

    calculator, params, _, _, _ = setup("stacking_classifier")
    calculator.prepare_tuning_params(params)
    config = TuningConfig(
        strategy="grid",
        search_space={candidate: [KNeighborsClassifier()]},
        metric="accuracy",
        cv_folds=2,
        n_jobs=1,
    )
    X = pd.DataFrame({"x": np.arange(40, dtype=float)})
    with pytest.raises(SampleWeightError, match="KNeighborsClassifier"):
        TuningCalculator(calculator).fit(
            X, pd.Series(np.arange(40) % 2), config, sample_weight=np.ones(40)
        )
