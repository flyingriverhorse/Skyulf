"""Fold prediction must preserve original class and observation identities."""

from typing import Any

import numpy as np
import pandas as pd
import polars as pl
import pytest
from sklearn.linear_model import LogisticRegression

from skyulf.modeling._tuning.fold_pipeline import FoldAwareModelStep
from skyulf.preprocessing.fold_adapter import AuditedFoldPreprocessor, FeatureEngineerFoldAdapter


def _steps(encode: bool, drop: bool = False) -> list[dict[str, Any]]:
    """Sort twice, impute the first lag, and optionally recode the target."""
    steps = [
        {
            "name": "lag",
            "transformer": "LagFeatures",
            "params": {"columns": ["x"], "sort_by": "t", "lags": [1], "drop_na": drop},
        },
        {
            "name": "impute",
            "transformer": "SimpleImputer",
            "params": {"columns": ["x_lag_1"], "strategy": "mean"},
        },
        {
            "name": "rolling",
            "transformer": "RollingAggregate",
            "params": {"columns": ["x"], "sort_by": "u", "window": 2, "aggregations": ["mean"]},
        },
    ]
    if encode:
        steps.append(
            {"name": "labels", "transformer": "LabelEncoder", "params": {"columns": ["target"]}}
        )
    return steps


def _data() -> tuple[pd.DataFrame, pd.Series]:
    """Duplicate index labels make accidental index-based restoration observable."""
    rng = np.random.default_rng(41)
    frame = pd.DataFrame(
        {"x": rng.normal(size=80), "t": rng.permutation(80), "u": rng.integers(0, 5, 80)},
        index=[7] * 80,
    )
    target = pd.Series(np.where(frame["x"] > 0, 10, 2), index=frame.index, name="target")
    return frame, target


@pytest.mark.parametrize("engine", ["pandas", "polars"])
@pytest.mark.parametrize("encode", [False, True])
@pytest.mark.parametrize("audited", [False, True])
def test_sorted_fold_predictions_restore_original_rows(engine, encode, audited):
    """Two temporal permutations must compose without corrupting target decoding."""
    frame, target = _data()
    first = np.argsort(frame["t"].to_numpy(), kind="stable")
    positions = first[np.argsort(frame["u"].to_numpy()[first], kind="stable")]
    X: Any = pl.from_pandas(frame) if engine == "polars" else frame
    y: Any = pl.from_pandas(target) if engine == "polars" else target
    adapter: Any = FeatureEngineerFoldAdapter(_steps(encode), "target")
    if audited:
        adapter = AuditedFoldPreprocessor(adapter)
    step = FoldAwareModelStep(LogisticRegression(max_iter=1000), adapter).fit(X, y)
    reference = FeatureEngineerFoldAdapter(_steps(encode), "target")
    transformed, encoded_y = reference.fit_transform(frame, target)
    native = LogisticRegression(max_iter=1000).fit(transformed, encoded_y)
    raw_predictions = native.predict(transformed)
    if encode:
        raw_predictions = np.where(raw_predictions == 0, 10, 2)
    expected_predictions = raw_predictions[np.argsort(positions)]
    expected_probabilities = native.predict_proba(transformed)[np.argsort(positions)]
    expected_margins = native.decision_function(transformed)[np.argsort(positions)]
    if encode:
        expected_probabilities = expected_probabilities[:, ::-1]
        expected_margins = -expected_margins

    np.testing.assert_array_equal(step.classes_, [2, 10])
    np.testing.assert_array_equal(step.predict(X), expected_predictions)
    np.testing.assert_allclose(step.predict_proba(X), expected_probabilities)
    np.testing.assert_allclose(step.decision_function(X), expected_margins)


def test_fold_prediction_rejects_removed_rows():
    """A response with fewer rows cannot safely be compared to untouched sklearn targets."""
    frame, target = _data()
    step = FoldAwareModelStep(
        LogisticRegression(max_iter=1000), FeatureEngineerFoldAdapter(_steps(True, True), "target")
    ).fit(frame, target)

    with pytest.raises(ValueError, match="row"):
        step.predict(frame)


@pytest.mark.parametrize("metric", ["f1", "average_precision", "pr_auc_weighted"])
def test_grid_target_encoding_preserves_original_positive_class(metric):
    """Recoding the target must not change which class a candidate search optimizes."""
    from sklearn.datasets import make_classification

    from skyulf.modeling._tuning.engine import TuningCalculator
    from skyulf.modeling.classification import LogisticRegressionCalculator

    values, labels = make_classification(
        n_samples=160, n_features=6, n_informative=4, weights=[0.75], random_state=7
    )
    frame = pd.DataFrame(values, columns=list("abcdef"))
    target = pd.Series(np.where(labels == 1, 10, 2), name="target")
    config = {
        "strategy": "grid",
        "metric": metric,
        "cv_folds": 3,
        "search_space": {"C": [0.1, 1.0]},
        "n_jobs": 1,
    }
    adapter = FeatureEngineerFoldAdapter(
        [{"name": "labels", "transformer": "LabelEncoder", "params": {"columns": ["target"]}}],
        "target",
    )
    tuner = TuningCalculator(LogisticRegressionCalculator())
    _, native = tuner.fit(frame, target, config)

    _, encoded = tuner.fit(frame, target, config, preprocessing=adapter)

    assert encoded.best_score == pytest.approx(native.best_score)
    assert encoded.best_params == native.best_params


@pytest.mark.parametrize("metric", ["f1", "precision", "recall", "roc_auc", "pr_auc"])
def test_polars_multiclass_target_uses_same_metric_as_pandas(metric):
    """Container choice must not leave multiclass tuning with a binary-only scorer."""
    from skyulf.modeling._tuning.metrics import resolve_metric
    from skyulf.modeling._tuning.schemas import TuningConfig

    values = [0, 1, 2, 0, 1, 2]
    config = TuningConfig(metric=metric)
    expected = resolve_metric(config, pd.Series(values), "classification")

    assert resolve_metric(config, pl.Series(values), "classification") == expected
