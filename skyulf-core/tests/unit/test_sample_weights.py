"""Pin direct model weighting, validation, and native class-weight composition."""

from __future__ import annotations

from decimal import Decimal

import numpy as np
import pandas as pd
import pytest
from sklearn.base import BaseEstimator
from sklearn.calibration import CalibratedClassifierCV
from sklearn.ensemble import (
    RandomForestClassifier,
    StackingClassifier,
    StackingRegressor,
    VotingClassifier,
    VotingRegressor,
)
from sklearn.linear_model import LinearRegression, LogisticRegression
from sklearn.neighbors import KNeighborsClassifier, KNeighborsRegressor
from sklearn.tree import DecisionTreeClassifier

from skyulf.modeling._class_weights import sample_weight_for_fit
from skyulf.modeling.classification import (
    CalibratedClassifierCalculator,
    LogisticRegressionCalculator,
)
from skyulf.modeling.ensemble import (
    StackingClassifierCalculator,
    StackingRegressorCalculator,
    VotingClassifierCalculator,
    VotingRegressorCalculator,
)
from skyulf.modeling.regression import LinearRegressionCalculator
from skyulf.modeling.sklearn_wrapper import SklearnCalculator


@pytest.mark.parametrize("values", [[Decimal("0.5"), Decimal("1.5")]])
def test_decimal_sample_weights(values):
    """Databricks DecimalType columns must retain their numeric weight meaning."""
    from skyulf.modeling._sample_weights import validate_sample_weight

    result = validate_sample_weight(values, 2)
    assert result is not None
    np.testing.assert_array_equal(result, [0.5, 1.5])


@pytest.mark.parametrize("value", [Decimal("NaN"), Decimal("Infinity"), Decimal("-Infinity")])
def test_nonfinite_decimal_sample_weights(value):
    """Decimal special values must fail the same finite-value contract as floats."""
    from skyulf.modeling._sample_weights import SampleWeightError, validate_sample_weight

    with pytest.raises(SampleWeightError, match="sample_weight"):
        validate_sample_weight([value, Decimal("1")], 2)


@pytest.mark.parametrize("direct_wrapper", [False, True])
def test_weighted_clustering_rejected_before_fit(direct_wrapper):
    """Supervised-only V1 weights must not reach an unsupervised estimator fit."""
    from skyulf.modeling._sample_weights import SampleWeightError
    from skyulf.modeling.clustering import KMeansCalculator

    calculator = (
        SklearnCalculator(_KwargsOnlyEstimator, {}, "clustering")
        if direct_wrapper
        else KMeansCalculator()
    )
    with pytest.raises(SampleWeightError, match="sample_weight.*classification.*regression"):
        calculator.fit(pd.DataFrame({"x": [0.0, 1.0]}), None, {}, sample_weight=[1, 1])


@pytest.mark.parametrize(
    "values",
    [
        [True, 1],
        [np.bool_(False), 1.0],
        [None, 1],
        [pd.NA, 1],
        ["1", "2"],
        [1 + 2j, 1],
        [np.nan, 1],
        [np.inf, 1],
        [-1, 2],
        [0, 0],
        [1e308, 1e308],
        [[1], [2]],
        [1],
        1,
        [],
    ],
)
def test_invalid_sample_weights(values):
    """Invalid vectors must fail before an estimator can consume them."""
    from skyulf.modeling._sample_weights import validate_sample_weight

    with pytest.raises(ValueError, match="sample_weight"):
        validate_sample_weight(values, 2)


@pytest.mark.parametrize("values", [[0, 2, 4], np.array([0.0, 2.0, 4.0]), pd.Series([0, 2, 4])])
def test_validation_copies_without_normalizing(values):
    """Estimator mutation must not change the caller's unnormalized weights."""
    from skyulf.modeling._sample_weights import validate_sample_weight

    result = validate_sample_weight(values, 3)
    assert result is not None
    np.testing.assert_array_equal(result, [0.0, 2.0, 4.0])
    assert result.dtype == np.dtype(float)
    result[1] = 99
    np.testing.assert_array_equal(values, [0, 2, 4])


def test_none_preserves_unweighted_fit():
    """No vector must leave the existing unweighted fitting path intact."""
    from skyulf.modeling._sample_weights import validate_sample_weight

    assert validate_sample_weight(None, 3) is None
    assert sample_weight_for_fit(KNeighborsRegressor(), None, [0, 1, 2]) is None


def test_weighted_linear_regression_matches_sklearn():
    """A downweighted outlier must change the actual regression fit."""
    X = pd.DataFrame({"x": [0.0, 1.0, 2.0, 3.0]})
    y = pd.Series([0.0, 1.0, 2.0, 20.0])
    weights = np.array([1.0, 1.0, 1.0, 0.01])
    model = LinearRegressionCalculator().fit(X, y, {}, sample_weight=weights)
    reference = LinearRegression().fit(X.to_numpy(), y.to_numpy(), sample_weight=weights)
    np.testing.assert_allclose(model.predict(X.to_numpy()), reference.predict(X.to_numpy()))
    assert not np.allclose(model.coef_, LinearRegression().fit(X, y).coef_)
    np.testing.assert_array_equal(weights, [1.0, 1.0, 1.0, 0.01])


@pytest.mark.parametrize("class_weight", [None, "balanced", {0: 1.0, 1: 3.0}])
@pytest.mark.parametrize("weighted", [False, True])
def test_native_classification_matches_sklearn(class_weight, weighted):
    """Native class weights must combine with user weights only inside sklearn."""
    X = pd.DataFrame({"x": [-3.0, -2.0, -1.0, 0.0, 1.0, 2.0]})
    y = pd.Series([0, 0, 0, 0, 1, 1])
    weights = np.array([1.0, 2.0, 0.0, 3.0, 7.0, 1.0]) if weighted else None
    params = {"class_weight": class_weight, "random_state": 42, "max_iter": 1000}
    model = LogisticRegressionCalculator().fit(X, y, {"params": params}, sample_weight=weights)
    reference = LogisticRegression(**params).fit(X.to_numpy(), y.to_numpy(), sample_weight=weights)
    np.testing.assert_allclose(
        model.predict_proba(X.to_numpy()), reference.predict_proba(X.to_numpy())
    )


@pytest.mark.parametrize("model_class", [DecisionTreeClassifier, RandomForestClassifier])
@pytest.mark.parametrize(
    ("class_weight", "weights"),
    [
        ({0: 0.0, 1: 1.0}, [1.0, 1.0, 0.0, 0.0]),
        ({0: 1e308, 1: 1.0}, [2.0, 2.0, 1.0, 1.0]),
        ({0: -1.0, 1: 1.0}, [1.0, 1.0, 1.0, 1.0]),
    ],
)
def test_native_class_weight_product_is_validated(model_class, class_weight, weights):
    """Native weighting must reject unusable products before fitting NaN predictions."""
    from skyulf.modeling._sample_weights import SampleWeightError

    X = pd.DataFrame({"x": [0.0, 1.0, 2.0, 3.0]})
    calculator = SklearnCalculator(model_class, {}, "classification")
    with pytest.raises(SampleWeightError, match="sample_weight"):
        calculator.fit(
            X,
            pd.Series([0, 0, 1, 1]),
            {"params": {"class_weight": class_weight, "random_state": 42}},
            sample_weight=weights,
        )


@pytest.mark.parametrize(
    ("model_class", "class_weight"),
    [
        (DecisionTreeClassifier, "balanced"),
        (DecisionTreeClassifier, {0: 2.0, 1: 3.0}),
        (RandomForestClassifier, "balanced"),
        (RandomForestClassifier, "balanced_subsample"),
        (RandomForestClassifier, {0: 2.0, 1: 3.0}),
    ],
)
def test_native_tree_weights_preserve_sklearn_probabilities(model_class, class_weight):
    """Validation cannot double class weights or alter native bootstrap balancing."""
    X = pd.DataFrame({"x": np.arange(12, dtype=float)})
    y = pd.Series([0, 1, 0, 0, 1, 0, 0, 0, 1, 0, 1, 0])
    weights = np.array([1, 2, 0, 3, 7, 1, 2, 3, 4, 5, 6, 8], dtype=float)
    params = {"class_weight": class_weight, "random_state": 42, "max_depth": 2}
    calculator = SklearnCalculator(model_class, {}, "classification")
    model = calculator.fit(X, y, {"params": params}, sample_weight=weights)
    reference = model_class(**params).fit(X.to_numpy(), y, sample_weight=weights)
    np.testing.assert_allclose(
        model.predict_proba(X.to_numpy()), reference.predict_proba(X.to_numpy())
    )


@pytest.mark.parametrize("labels", [[0, 0, 1, 1], ["cat", "cat", "dog", "dog"]])
def test_voting_child_native_class_product_is_validated(labels):
    """Encoded voting targets must not hide a child's zero effective weight mass."""
    from skyulf.modeling._sample_weights import SampleWeightError

    params = {
        "estimators": [("tree", DecisionTreeClassifier(class_weight={0: 0.0, 1: 1.0}))],
        "voting": "soft",
    }
    calculator = SklearnCalculator(VotingClassifier, {}, "classification")
    with pytest.raises(SampleWeightError, match="positive finite total"):
        calculator.fit(
            pd.DataFrame({"x": [0.0, 1.0, 2.0, 3.0]}),
            pd.Series(labels),
            {"params": params},
            sample_weight=[1.0, 1.0, 0.0, 0.0],
        )


def test_voting_child_native_weights_preserve_encoded_label_semantics():
    """Validation must use voting's encoded classes while preserving native predictions."""
    X = pd.DataFrame({"x": np.arange(12, dtype=float)})
    y = pd.Series(["cat", "dog", "cat", "cat", "dog", "cat"] * 2)
    weights = np.arange(1, 13, dtype=float)
    params = {
        "estimators": [
            ("tree", DecisionTreeClassifier(class_weight={0: 2.0, 1: 3.0}, max_depth=2))
        ],
        "voting": "soft",
    }
    model = SklearnCalculator(VotingClassifier, {}, "classification").fit(
        X, y, {"params": params}, sample_weight=weights
    )
    reference = VotingClassifier(**params).fit(X.to_numpy(), y, sample_weight=weights)
    np.testing.assert_allclose(
        model.predict_proba(X.to_numpy()), reference.predict_proba(X.to_numpy())
    )


@pytest.mark.parametrize("class_weight", [None, [{0: 2.0, 1: 3.0}, {0: 1.0, 1: 4.0}]])
def test_weighted_multilabel_stacking_matches_sklearn(class_weight):
    """Weight validation must retain stacking's independent encoding of each target."""
    X = pd.DataFrame({"x": np.arange(30), "z": np.arange(30) % 5})
    y = np.column_stack([np.arange(30) % 2, np.arange(30) % 3 == 0]).astype(int)
    weights = np.arange(1, 31, dtype=float)
    params = {
        "estimators": [
            (
                "rf",
                RandomForestClassifier(n_estimators=3, random_state=2, class_weight=class_weight),
            )
        ],
        "final_estimator": RandomForestClassifier(
            n_estimators=3, random_state=2, class_weight=class_weight
        ),
        "cv": 2,
    }
    model = SklearnCalculator(StackingClassifier, {}, "classification").fit(
        X, y, {"params": params}, sample_weight=weights
    )
    reference = StackingClassifier(**params).fit(X.to_numpy(), y, sample_weight=weights)
    np.testing.assert_allclose(
        model.predict_proba(X.to_numpy()), reference.predict_proba(X.to_numpy())
    )


@pytest.mark.parametrize("child", ["base", "final"])
def test_multilabel_stacking_rejects_invalid_child_class_product(child):
    """Multioutput child maps must still reject zero-mass products before fitting."""
    from skyulf.modeling._sample_weights import SampleWeightError

    X = pd.DataFrame({"x": np.arange(30), "z": np.arange(30) % 5})
    y = np.column_stack([np.arange(30) % 2, np.arange(30) % 3 == 0]).astype(int)
    weights = np.asarray(y[:, 0] == 0, dtype=float)
    invalid = [{0: 0.0, 1: 1.0}, {0: 1.0, 1: 1.0}]
    params = {
        "estimators": [
            (
                "rf",
                RandomForestClassifier(class_weight=invalid if child == "base" else None),
            )
        ],
        "final_estimator": RandomForestClassifier(
            class_weight=invalid if child == "final" else None
        ),
        "cv": 2,
    }
    with pytest.raises(SampleWeightError, match="positive finite total"):
        SklearnCalculator(StackingClassifier, {}, "classification").fit(
            X, y, {"params": params}, sample_weight=weights
        )


@pytest.mark.parametrize(
    ("class_weight", "weights", "expected"),
    [
        (None, [2, 4, 6, 8], [2, 4, 6, 8]),
        ("balanced", None, [2 / 3, 2 / 3, 2 / 3, 2]),
        ("balanced", [3, 6, 9, 12], [2, 4, 6, 24]),
        ({0: 2, 1: 3}, [1, 2, 3, 4], [2, 4, 6, 12]),
    ],
)
def test_nonnative_weights_combine_once(class_weight, weights, expected):
    """The shared helper must multiply class and row weights exactly once."""
    result = sample_weight_for_fit(LinearRegression(), class_weight, [0, 0, 0, 1], weights)
    assert result is not None
    np.testing.assert_allclose(result, expected)


@pytest.mark.parametrize(
    ("class_weight", "weights"),
    [
        ({0: 0, 1: 1}, [1, 0]),
        ({0: 1e308, 1: 1}, [2, 0]),
        ({0: -1, 1: 1}, [1, 1]),
        ({0: 0, 1: 0}, None),
    ],
)
def test_combined_weights_are_validated(class_weight, weights):
    """A valid input vector cannot hide an invalid effective fit vector."""
    with pytest.raises(ValueError, match="sample_weight"):
        sample_weight_for_fit(LinearRegression(), class_weight, [0, 1], weights)


class _KwargsOnlyEstimator(BaseEstimator):
    """Expose the silent-drop failure mode of an arbitrary-kwargs fit."""

    def fit(self, X, y, **kwargs):
        """Fail if rejection did not precede estimator execution."""
        raise AssertionError("fit must not start")


@pytest.mark.parametrize("model_class", [KNeighborsRegressor, _KwargsOnlyEstimator])
def test_unsupported_estimator_rejected_before_fit(model_class):
    """Arbitrary fit kwargs are not evidence of weight support."""
    calculator = SklearnCalculator(model_class, {}, "regression")
    with pytest.raises(ValueError, match="sample_weight"):
        calculator.fit(pd.DataFrame({"x": [0, 1]}), pd.Series([0, 1]), {}, sample_weight=[1, 1])


@pytest.mark.parametrize(
    "calculator_class",
    [
        CalibratedClassifierCalculator,
        VotingClassifierCalculator,
        VotingRegressorCalculator,
        StackingClassifierCalculator,
        StackingRegressorCalculator,
    ],
)
def test_composite_calculators_reject_unsupported_children(calculator_class):
    """Unsupported children must fail before any composite fitting starts."""
    params = (
        {"estimator": KNeighborsClassifier()}
        if calculator_class is CalibratedClassifierCalculator
        else {"base_estimators": ["knn"]}
    )
    with pytest.raises(ValueError, match="sample_weight"):
        calculator_class().fit(
            pd.DataFrame({"x": [0, 1]}), pd.Series([0, 1]), {"params": params}, sample_weight=[1, 1]
        )


@pytest.mark.parametrize(
    "model",
    [
        CalibratedClassifierCV(),
        VotingClassifier([]),
        VotingRegressor([]),
        StackingClassifier([]),
        StackingRegressor([]),
    ],
)
def test_shared_helper_supports_composites(model):
    """Tuning callers share the direct-fit recursive routing policy."""
    result = sample_weight_for_fit(model, None, [0, 1], [1, 1])
    assert result is not None
    np.testing.assert_array_equal(result, [1, 1])


def test_existing_class_weight_only_composite_path_is_preserved():
    """Adding row weights must not block the established class-only helper path."""
    result = sample_weight_for_fit(CalibratedClassifierCV(), "balanced", [0, 0, 0, 1])
    assert result is not None
    np.testing.assert_allclose(result, [2 / 3, 2 / 3, 2 / 3, 2])


@pytest.mark.parametrize("problem_type", ["classification", "regression"])
def test_lightgbm_overrides_forward_weights(problem_type):
    """Warning-suppression overrides must not drop a requested fit vector."""
    pytest.importorskip("lightgbm")
    from skyulf.modeling.classification import LGBMClassifierCalculator
    from skyulf.modeling.regression import LGBMRegressorCalculator

    calculator = (
        LGBMClassifierCalculator()
        if problem_type == "classification"
        else LGBMRegressorCalculator()
    )
    X = pd.DataFrame({"x": np.arange(40, dtype=float)})
    y = pd.Series(([0] * 30 + [1] * 10) if problem_type == "classification" else np.arange(40))
    weights = np.array([1.0] * 30 + [10.0] * 10)
    params = {"n_estimators": 3, "min_child_samples": 2, "n_jobs": 1, "random_state": 42}
    model = calculator.fit(X, y, {"params": params}, sample_weight=weights)
    reference = calculator.model_class(**{**calculator.default_params, **params}).fit(
        X.to_numpy(), y.to_numpy(), sample_weight=weights
    )
    np.testing.assert_allclose(model.predict(X.to_numpy()), reference.predict(X.to_numpy()))


def test_custom_composite_cannot_inherit_kwargs_routing_approval():
    """Overriding a known ensemble fit must not silently discard explicit row weights."""
    from sklearn.ensemble import VotingClassifier
    from sklearn.linear_model import LogisticRegression

    from skyulf.modeling._sample_weights import SampleWeightError, ensure_sample_weight_support

    class CustomVoting(VotingClassifier):
        """Represent a user subclass whose kwargs contract is not sklearn's implementation."""

        def fit(self, X, y, **kwargs):
            """Deliberately omit kwargs to expose unsafe inherited capability approval."""
            return super().fit(X, y)

    model = CustomVoting([("logistic", LogisticRegression())])
    with pytest.raises(SampleWeightError, match="does not support sample_weight"):
        ensure_sample_weight_support(model)
