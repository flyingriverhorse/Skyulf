"""Stored decision thresholds must participate in a pipeline's prediction identity."""

import copy
import pickle

import numpy as np
import pandas as pd
import pytest
from sklearn.metrics import accuracy_score

from skyulf.pipeline import SkyulfPipeline


@pytest.fixture(params=[(0, 1), ("no", "yes")])
def fitted_pipeline(request, monkeypatch):
    """Train a weak classifier whose holdout supports distinct decision cutoffs."""
    monkeypatch.setenv("SKYULF_ENGINE", "pandas")
    negative, positive = request.param
    data = pd.DataFrame({"x": list(range(20)), "target": [negative] * 10 + [positive] * 10})
    pipeline = SkyulfPipeline(
        {
            "preprocessing": [],
            "modeling": {"type": "logistic_regression", "params": {"C": 0.001}},
        }
    )
    pipeline.fit(data, target_column="target")
    return pipeline, data, negative, positive


def test_different_tuned_predictions_have_different_fingerprints(fitted_pipeline):
    """Equal trained weights cannot conceal different saved prediction rules."""
    first, _, negative, positive = fitted_pipeline
    second = copy.deepcopy(first)
    holdout = pd.DataFrame({"x": list(range(6, 14))})
    first_labels = [negative] * 2 + [positive] * 6
    second_labels = [negative] * 6 + [positive] * 2

    first_thresholds = first.optimize_thresholds(holdout, first_labels, accuracy_score)
    second_thresholds = second.optimize_thresholds(holdout, second_labels, accuracy_score)

    assert first_thresholds[positive] < second_thresholds[positive]
    np.testing.assert_array_equal(first.predict(holdout), second.predict(holdout))
    first_predictions = first.predict(holdout, use_tuned_thresholds=True)
    second_predictions = second.predict(holdout, use_tuned_thresholds=True)
    assert np.count_nonzero(first_predictions != second_predictions) == 4
    assert first.fingerprint() != second.fingerprint()


def test_refit_clears_threshold_identity_with_threshold_state(fitted_pipeline):
    """Retraining must discard the old cutoff and restore the same untuned identity."""
    pipeline, data, negative, positive = fitted_pipeline
    original_fingerprint = pipeline.fingerprint()
    pipeline.optimize_thresholds(
        pd.DataFrame({"x": list(range(6, 14))}),
        [negative] * 2 + [positive] * 6,
        accuracy_score,
    )
    tuned_fingerprint = pipeline.fingerprint()
    pipeline.fit(data, target_column="target")

    assert pipeline._tuned_thresholds is None
    assert pipeline.fingerprint() == original_fingerprint
    assert tuned_fingerprint != original_fingerprint


@pytest.mark.parametrize("protocol", [2, pickle.HIGHEST_PROTOCOL])
def test_threshold_fingerprint_survives_serialization_and_mapping_order(fitted_pipeline, protocol):
    """Saving or reordering equivalent threshold mappings cannot alter model identity."""
    pipeline, _, negative, positive = fitted_pipeline
    holdout = pd.DataFrame({"x": list(range(6, 14))})
    thresholds = pipeline.optimize_thresholds(
        holdout, [negative] * 2 + [positive] * 6, accuracy_score
    )
    original_fingerprint = pipeline.fingerprint()
    pipeline._tuned_thresholds = dict(reversed(list(thresholds.items())))
    restored = pickle.loads(pickle.dumps(pipeline, protocol=protocol))

    assert restored.fingerprint() == original_fingerprint
    assert restored.export_model_card()["fingerprint"] == original_fingerprint
    np.testing.assert_array_equal(
        restored.predict(holdout, use_tuned_thresholds=True),
        pipeline.predict(holdout, use_tuned_thresholds=True),
    )


def test_legacy_pipeline_without_threshold_attribute_keeps_untuned_identity(fitted_pipeline):
    """Older saved pipelines must remain fingerprintable without threshold metadata."""
    pipeline, _, _, _ = fitted_pipeline
    original_fingerprint = pipeline.fingerprint()
    del pipeline._tuned_thresholds

    assert pipeline.fingerprint() == original_fingerprint
