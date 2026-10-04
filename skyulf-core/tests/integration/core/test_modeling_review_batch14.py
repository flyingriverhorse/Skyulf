"""Public regressions for threshold semantics, ensemble flags and sequence targets."""

import os
from copy import deepcopy
from pathlib import Path
from unittest.mock import Mock

import numpy as np
import pandas as pd
import pytest
from sklearn.calibration import CalibratedClassifierCV
from sklearn.metrics import accuracy_score

import skyulf.modeling._evaluation.thresholds as threshold_module
import skyulf.modeling.cross_validation as cv_module
import skyulf.modeling.ensemble as ensemble_module
from skyulf.modeling._evaluation.thresholds import apply_thresholds, optimize_thresholds
from skyulf.modeling.classification import (
    DecisionTreeClassifierApplier,
    DecisionTreeClassifierCalculator,
)
from skyulf.modeling.cross_validation import perform_cross_validation
from skyulf.modeling.ensemble import StackingClassifierCalculator, VotingClassifierCalculator
from skyulf.preprocessing.fold_adapter import FeatureEngineerFoldAdapter


@pytest.fixture(scope="module", autouse=True)
def verify_source_imports():
    """Explicit snapshot runs must use the requested source; installed-wheel runs stay valid."""
    expected = os.environ.get("SKYULF_EXPECTED_SOURCE_ROOT")
    if expected:
        source_root = Path(expected).resolve()
        for module in (threshold_module, cv_module, ensemble_module):
            assert Path(module.__file__).resolve().is_relative_to(source_root)


@pytest.mark.parametrize("classes", [["negative", "positive"], [42, 11]])
@pytest.mark.parametrize("position", [0, 1])
@pytest.mark.parametrize("cutoff", [0.0, 0.6, 1.0])
def test_single_binary_threshold_uses_its_named_probability_column(classes, position, cutoff):
    """A named cutoff must select that label, including ties and noncanonical class order."""
    probabilities = np.array([[1.0, 0.0], [0.6, 0.4], [0.4, 0.6], [0.0, 1.0]])
    thresholds = {classes[position]: cutoff}
    expected = np.where(
        probabilities[:, position] >= cutoff, classes[position], classes[1 - position]
    )

    implicit = apply_thresholds(probabilities, thresholds, classes)
    explicit = apply_thresholds(
        probabilities, thresholds, classes, positive_class=classes[position]
    )

    np.testing.assert_array_equal(implicit, expected)
    np.testing.assert_array_equal(explicit, expected)
    assert thresholds == {classes[position]: cutoff}


@pytest.mark.parametrize(
    "key,positive", [("typo", None), ("negative", "positive"), ("typo", "positive")]
)
def test_single_binary_threshold_rejects_unknown_or_conflicting_label(key, positive):
    """Misspelled keys and an incompatible positive-class request must fail clearly."""
    with pytest.raises(ValueError, match="class|key|label"):
        apply_thresholds(
            [[0.7, 0.3]], {key: 0.6}, ["negative", "positive"], positive_class=positive
        )


@pytest.mark.parametrize("cutoff", [float("nan"), float("inf"), -0.1, 1.1])
def test_named_binary_cutoff_requires_valid_probability(cutoff):
    """Named dictionary cutoffs must use the existing finite probability bounds."""
    with pytest.raises(ValueError, match="finite|zero|one"):
        apply_thresholds([[0.7, 0.3]], {"negative": cutoff}, ["negative", "positive"])


def test_scalar_and_full_binary_pair_retain_second_class_ties():
    """Fixing named cutoffs must leave scalar and full-pair probability weighting unchanged."""
    probabilities = [[0.4, 0.6], [0.5, 0.5], [1.0, 0.0], [0.0, 1.0]]
    classes = ["last", "first"]
    scalar = apply_thresholds(probabilities, 0.6, classes)
    paired = apply_thresholds(probabilities, {"last": 0.4, "first": 0.6}, classes)
    named = apply_thresholds(
        probabilities, {"last": 0.4, "first": 0.6}, classes, positive_class="last"
    )

    assert scalar.tolist() == paired.tolist() == ["first", "last", "last", "first"]
    assert named.tolist() == ["last", "last", "last", "first"]


@pytest.mark.parametrize("class_count", [3, 4])
def test_multiclass_grid_is_rejected_before_scoring(class_count):
    """An explicitly binary strategy must not return an unusable partial multiclass map."""
    classes = [f"label_{index}" for index in range(class_count)]
    metric = Mock(return_value=1.0)
    with pytest.raises(ValueError, match="grid.*binary|binary.*grid"):
        optimize_thresholds(classes, np.eye(class_count), metric, classes, strategy="grid")
    metric.assert_not_called()


@pytest.mark.parametrize("class_count,strategy", [(2, "grid"), (3, None), (3, "nelder-mead")])
def test_supported_threshold_search_returns_full_applicable_class_map(class_count, strategy):
    """Default multiclass and supported explicit searches must retain every probability axis."""
    classes = [f"label_{index}" for index in range(class_count)]
    probabilities = np.eye(class_count) * 0.8 + 0.2 / class_count
    thresholds = optimize_thresholds(
        classes, probabilities, accuracy_score, classes, strategy=strategy
    )
    predicted = apply_thresholds(probabilities, thresholds, classes)

    assert set(thresholds) == set(classes)
    assert predicted.shape == (class_count,)
    assert predicted.tolist() == classes


@pytest.fixture
def ensemble_data():
    """Keep real ensemble fits small while supplying enough observations for inner CV."""
    rng = np.random.default_rng(614)
    frame = pd.DataFrame({"a": rng.normal(size=36), "b": rng.normal(size=36)})
    return frame, pd.Series(np.where(frame["a"] > 0, "positive", "negative"))


@pytest.mark.parametrize("flag", ["calibrate_base_models", "passthrough"])
@pytest.mark.parametrize("invalid", ["false", "true", 0, 1, None])
@pytest.mark.parametrize("nested", [False, True])
def test_ensemble_defaults_reject_nonboolean_flags(flag, invalid, nested):
    """Structural reconstruction must not silently enable options through Python truthiness."""
    params = {"base_estimators": ["gaussian_nb"], flag: invalid}
    config = {"params": params} if nested else params
    before = deepcopy(config)
    calculator = StackingClassifierCalculator()
    calculator.prepare_tuning_params(config)

    with pytest.raises(ValueError, match=flag + " must be boolean"):
        _ = calculator.default_params

    assert config == before


@pytest.mark.parametrize("flag", ["calibrate_base_models", "passthrough"])
@pytest.mark.parametrize("nested", [False, True])
def test_ensemble_fit_rejects_text_false(flag, nested, ensemble_data):
    """Direct fitting and tuning reconstruction must enforce the same boolean contract."""
    frame, labels = ensemble_data
    params = {"base_estimators": ["gaussian_nb"], "cv": 2, flag: "false"}
    config = {"params": params} if nested else params

    with pytest.raises(ValueError, match=flag + " must be boolean"):
        StackingClassifierCalculator().fit(frame, labels, config)


@pytest.mark.parametrize("enabled", [False, True])
def test_valid_ensemble_booleans_control_wrapping_and_passthrough(enabled, ensemble_data):
    """Actual booleans must still select calibration and the intended stacking feature width."""
    frame, labels = ensemble_data
    model = StackingClassifierCalculator().fit(
        frame,
        labels,
        {
            "base_estimators": ["gaussian_nb"],
            "cv": 2,
            "calibration_cv": 2,
            "calibrate_base_models": enabled,
            "passthrough": enabled,
        },
    )

    assert isinstance(model.estimators[0][1], CalibratedClassifierCV) is enabled
    assert model.passthrough is enabled
    assert model.transform(frame.to_numpy()).shape[1] == 1 + (frame.shape[1] if enabled else 0)


@pytest.mark.parametrize("enabled", [False, True, "false"])
def test_ensemble_search_space_calibration_matches_boolean_contract(enabled):
    """Search parameter paths must address the same calibration wrapper used for fitting."""
    calculator = VotingClassifierCalculator()
    config = {
        "tune_base_models": True,
        "base_estimators": ["decision_tree"],
        "calibrate_base_models": enabled,
    }
    if isinstance(enabled, str):
        with pytest.raises(ValueError, match="calibrate_base_models must be boolean"):
            calculator.build_tuning_search_space(config, "grid")
    else:
        space = calculator.build_tuning_search_space(config, "grid")
        path = "decision_tree__estimator__max_depth" if enabled else "decision_tree__max_depth"
        assert path in space


@pytest.mark.parametrize(
    "cv_type", ["k_fold", "stratified_k_fold", "shuffle_split", "time_series_split", "nested_cv"]
)
@pytest.mark.parametrize(
    "container", [list, tuple, np.asarray], ids=["list", "tuple", "array-control"]
)
@pytest.mark.parametrize("weighted", [False, True])
def test_legacy_cv_sequence_targets_preserve_fold_rows_and_weights(
    cv_type, container, weighted, monkeypatch
):
    """Every fold must retain positional string labels and training weights through preprocessing."""
    row_ids = np.random.default_rng(613).permutation(36)
    expected_labels = np.where(np.arange(36) % 2, "odd", "even")
    frame = pd.DataFrame(
        {"row_id": row_ids, "signal": np.where(row_ids % 2, 1.0, -1.0)}, index=[99] * 36
    )
    if cv_type == "time_series_split":
        frame["time"] = row_ids
    labels = container(expected_labels[row_ids].tolist())
    weights = pd.Series(row_ids + 1.0, index=np.arange(36)[::-1]) if weighted else None
    preprocessing = FeatureEngineerFoldAdapter(
        [
            {"name": "scale", "transformer": "StandardScaler", "params": {"columns": ["signal"]}},
        ],
        target_column="target",
    )
    calculator = DecisionTreeClassifierCalculator()
    original_fit = calculator.fit
    calls = []

    def aligned_fit(X, y, config, **kwargs):
        """Check real fold inputs before delegating to the actual model calculator."""
        native = X.to_native() if hasattr(X, "to_native") else X
        indices = np.asarray(native["row_id"], dtype=int)
        np.testing.assert_array_equal(np.asarray(y), expected_labels[indices])
        if weighted:
            np.testing.assert_array_equal(kwargs["sample_weight"], indices + 1.0)
        else:
            assert kwargs.get("sample_weight") is None
        calls.append(indices.tolist())
        return original_fit(X, y, config, **kwargs)

    monkeypatch.setattr(calculator, "fit", aligned_fit)
    result = perform_cross_validation(
        calculator,
        DecisionTreeClassifierApplier(),
        frame,
        labels,
        {"max_depth": 2},
        n_folds=3,
        cv_type=cv_type,
        shuffle=False,
        time_column="time" if cv_type == "time_series_split" else None,
        preprocessing=preprocessing,
        sample_weight=weights,
    )

    assert len(result["folds"]) == 3
    assert result["aggregated_metrics"]["accuracy"]["mean"] == 1.0
    assert len(calls) == (9 if cv_type == "nested_cv" else 3)
    assert list(labels) == expected_labels[row_ids].tolist()
    assert frame["row_id"].tolist() == row_ids.tolist()
