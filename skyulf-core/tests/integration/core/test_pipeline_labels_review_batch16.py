"""Keep original target identity across prediction and threshold calibration."""

import copy
import json
import pickle

import numpy as np
import pandas as pd
import polars as pl
import pytest
from sklearn.metrics import accuracy_score

from skyulf.pipeline import SkyulfPipeline


def _pipeline(labels, encoder):
    """Fit a tiny public pipeline whose target encoding may reverse its class axis."""
    params = {"columns": ["target"]}
    if encoder == "OrdinalEncoder":
        params["categories_order"] = ",".join(map(str, reversed(labels)))
    pipeline = SkyulfPipeline(
        {
            "preprocessing": [{"name": "target", "transformer": encoder, "params": params}],
            "modeling": {"type": "logistic_regression", "params": {"C": 0.05}},
            "decision_threshold": {"positive_class": labels[0]},
        }
    )
    pipeline.fit(
        pd.DataFrame({"x": range(24), "target": [labels[0]] * 12 + [labels[1]] * 12}), "target"
    )
    return pipeline


@pytest.mark.parametrize("labels", [("no", "yes"), (10, 20)])
@pytest.mark.parametrize("encoder", ["LabelEncoder", "OrdinalEncoder"])
@pytest.mark.parametrize("engine", ["pandas", "polars"])
def test_prediction_decodes_original_target_once(labels, encoder, engine):
    """Public predictions and pickle replay must retain the training target's identity."""
    pipeline = _pipeline(labels, encoder)
    frame = pd.DataFrame({"x": [1, 22]}, index=[91, 34])
    source = pl.from_pandas(frame) if engine == "polars" else frame
    result = pipeline.predict(source)
    np.testing.assert_array_equal(np.asarray(result), labels)
    if isinstance(result, pd.Series) and engine == "pandas":
        assert result.index.tolist() == [91, 34]
    restored = pickle.loads(pickle.dumps(pipeline))
    np.testing.assert_array_equal(np.asarray(restored.predict(source)), labels)


@pytest.mark.parametrize("labels", [("no", "yes"), (0, 1)])
@pytest.mark.parametrize("encoder", ["LabelEncoder", "OrdinalEncoder"])
def test_threshold_optimization_keeps_original_and_model_positive_identity(labels, encoder):
    """Recalibration must retain the encoded positive class when label order changes."""
    pipeline = _pipeline(labels, encoder)
    model = pipeline.model_estimator.model
    mapping = pipeline.feature_engineer.fitted_steps[0]["artifact"].get(
        "target_label_map", dict(zip(model.classes_, labels, strict=True))
    )
    positive = next(key for key, value in mapping.items() if value == labels[0])
    pipeline._decision_threshold_evidence = {
        "positive_class": labels[0],
        "model_positive_class": positive,
        "source": "training",
    }
    old_evidence = copy.deepcopy(pipeline._decision_threshold_evidence)
    frame = pd.DataFrame({"x": [1, 8, 14, 22]})
    original = np.array([labels[0], labels[0], labels[1], labels[1]])
    pipeline.optimize_thresholds(frame, original, accuracy_score)
    assert pipeline._decision_positive_class() == positive
    assert pipeline._decision_threshold_evidence["positive_class"] == old_evidence["positive_class"]
    assert pipeline._decision_threshold_evidence["model_positive_class"] == positive
    actual = pipeline.predict(frame, use_tuned_thresholds=True)
    proba = model.predict_proba(frame.to_numpy())
    cutoff = pipeline._tuned_thresholds[positive]
    pos_index = list(model.classes_).index(positive)
    expected = np.where(proba[:, pos_index] >= cutoff, labels[0], labels[1])
    np.testing.assert_array_equal(actual, expected)


def test_configured_string_positive_is_resolved_without_external_evidence():
    """Core-only threshold optimization must accept the original configured class label."""
    pipeline = _pipeline(("no", "yes"), "LabelEncoder")
    pipeline.optimize_thresholds(pd.DataFrame({"x": [1, 22]}), ["no", "yes"], accuracy_score)
    assert pipeline._decision_positive_class() == 0


@pytest.mark.parametrize("labels", [("no", "yes"), (10, 20)])
def test_threshold_class_identity_uses_json_safe_original_scalars(labels):
    """Positive-class evidence must retain the original label through JSON persistence."""
    pipeline = _pipeline(labels, "OrdinalEncoder")
    pipeline.optimize_thresholds(pd.DataFrame({"x": [1, 22]}), list(labels), accuracy_score)
    evidence = json.loads(json.dumps(pipeline._decision_threshold_evidence, allow_nan=False))
    assert evidence == {"positive_class": labels[0], "model_positive_class": 1.0}
