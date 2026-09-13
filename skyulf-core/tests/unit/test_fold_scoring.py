"""Fold scorer adapters preserve row identity, labels and fitted preprocessing state."""

import numpy as np
import pandas as pd
import pytest
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.svm import SVC

from skyulf.modeling._tuning.fold_pipeline import FoldAwareModelStep
from skyulf.modeling._tuning.fold_scoring import wrap_fold_scorer
from skyulf.modeling._tuning.metrics import resolve_scorer


class FilterScaleEncode:
    """Combine learned feature/target transforms with validation filtering and reordering."""

    def fit_transform(self, X, y):
        """Learn scale and label vocabulary only from training rows."""
        self.scaler = StandardScaler().fit(X)
        self.encoder = LabelEncoder().fit(y)
        self.transform_calls = 0
        return self.scaler.transform(X), self.encoder.transform(y)

    def transform(self, X, y):
        """Drop a middle outlier and reverse rows, making length-based slicing incorrect."""
        self.transform_calls += 1
        retained = np.flatnonzero(X["x"].to_numpy() < 100)[::-1]
        labels = self.encoder.transform(np.asarray(y)[retained]) if y is not None else None
        return self.scaler.transform(X.iloc[retained]), labels


def _fitted_case(pipeline, model):
    """Keep original labels and repeated indexes visible at the scorer boundary."""
    X = pd.DataFrame({"x": np.tile(np.arange(10, dtype=float), 6)})
    y = pd.Series(np.where(X["x"] >= 5, "zebra", "ant"), name="target")
    step = FoldAwareModelStep(estimator=model, preprocessor=FilterScaleEncode())
    estimator = Pipeline([("model", step)]) if pipeline else step
    estimator.fit(X, y)
    validation_X = pd.DataFrame({"x": [1.0, 9.0, 1000.0, 4.0, 8.0, 3.0]}, index=[9, 2, 2, 9, 1, 1])
    validation_y = pd.Series(
        ["ant", "zebra", "ant", "zebra", "ant", "ant"], index=validation_X.index
    )
    return estimator, step, y, validation_X, validation_y


@pytest.mark.parametrize("pipeline", [False, True])
@pytest.mark.parametrize("metric", ["f1", "average_precision", "roc_auc"])
def test_filtered_encoded_validation_uses_original_labels_and_correct_responses(pipeline, metric):
    """Label/probability/decision scorers must compare the same retained observations."""
    model = SVC(probability=False) if metric == "roc_auc" else LogisticRegression()
    estimator, step, train_y, X, y = _fitted_case(pipeline, model)
    scorer = resolve_scorer(metric, train_y, "classification")
    adapted = wrap_fold_scorer(estimator, scorer)
    worker = step.preprocessor_
    retained = np.flatnonzero(X["x"].to_numpy() < 100)[::-1]
    expected_X = worker.scaler.transform(X.iloc[retained])
    expected_y = worker.encoder.transform(y.iloc[retained])
    expected_scorer = resolve_scorer(metric, worker.encoder.transform(train_y), "classification")
    expected = expected_scorer(step.model_, expected_X, expected_y)
    before_X, before_y = X.copy(deep=True), y.copy(deep=True)
    assert adapted(estimator, X, y) == pytest.approx(expected)
    assert worker.transform_calls == 1
    assert step.preprocessor_ is worker
    assert list(step.classes_) == ["ant", "zebra"]
    pd.testing.assert_frame_equal(X, before_X)
    pd.testing.assert_series_equal(y, before_y)


@pytest.mark.parametrize("raises", [False, True])
def test_custom_multiresponse_scorer_transforms_once_and_preserves_live_pipeline(raises):
    """Repeated predictions or a failing metric must not disable the fitted preprocessor."""
    estimator, step, _, X, y = _fitted_case(True, LogisticRegression())
    worker = step.preprocessor_

    def scorer(view, features, target):
        """Exercise the full callable-scorer interface on one shared filtered payload."""
        assert isinstance(view, Pipeline)
        predictions = view.predict(features)
        assert view.predict_proba(features).shape == (5, 2)
        assert view.decision_function(features).shape == (5,)
        assert len(target) == len(predictions) == 5
        if raises:
            raise ValueError("metric unavailable")
        return accuracy_score(target, predictions)

    adapted = wrap_fold_scorer(estimator, scorer)
    if raises:
        with pytest.raises(ValueError, match="metric unavailable"):
            adapted(estimator, X, y)
    else:
        assert 0 <= adapted(estimator, X, y) <= 1
    assert worker.transform_calls == 1
    assert step.preprocessor_ is worker
    assert estimator.predict(X).shape == (5,)
    assert worker.transform_calls == 2
