"""Unit tests for the tuning-runner's live ``trial`` event emission.

``_emit_trial_event`` is the single point where a completed tuning trial
becomes a WebSocket event for the live trial chart. It must never impact
training: skipped trials stay silent, the engine's "unknown" job sentinel
stays silent, and transport failures are swallowed.
"""

import numpy as np
import pytest

from backend.ml_pipeline._execution.engine import _node_runners
from backend.realtime.events import JobEvent


@pytest.fixture
def captured(monkeypatch):
    """Capture published JobEvents instead of hitting local bus / Redis."""
    published: list[JobEvent] = []
    monkeypatch.setattr(_node_runners, "publish_job_event", published.append)
    return published


def test_valid_trial_publishes_one_event(captured):
    _node_runners._emit_trial_event("job-1", 3, 10, 0.85, "f1")
    assert len(captured) == 1
    event = captured[0]
    assert event.event == "trial"
    assert event.job_id == "job-1"
    assert event.trial_number == 3
    assert event.trial_total == 10
    assert event.trial_score == 0.85
    assert event.trial_metric == "f1"


def test_numpy_score_is_cast_to_native_float(captured):
    _node_runners._emit_trial_event("job-1", 1, 5, np.float64(0.91), "accuracy")
    assert type(captured[0].trial_score) is float


def test_none_score_publishes_nothing(captured):
    """Failed/pruned Optuna trials report score=None — nothing to chart."""
    _node_runners._emit_trial_event("job-1", 2, 10, None, "f1")
    assert captured == []


def test_unknown_job_sentinel_publishes_nothing(captured):
    """Preview-path runs use job_id='unknown'; no job row to attach to."""
    _node_runners._emit_trial_event("unknown", 1, 5, 0.7, "accuracy")
    assert captured == []


def test_transport_failure_is_swallowed(monkeypatch):
    def explode(_event):
        raise RuntimeError("bus down")

    monkeypatch.setattr(_node_runners, "publish_job_event", explode)
    # Must not raise — a transport hiccup must never break training.
    _node_runners._emit_trial_event("job-1", 1, 5, 0.7, "accuracy")
