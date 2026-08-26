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


def test_non_finite_score_publishes_nothing(captured):
    """Degenerate trials can report -inf/NaN; the jobs API serializes with
    allow_nan=False, so these must be dropped, not charted."""
    from backend.realtime import trial_buffer

    try:
        _node_runners._emit_trial_event("job-finite", 1, 10, float("-inf"), "f1")
        _node_runners._emit_trial_event("job-finite", 2, 10, float("nan"), "f1")
        assert captured == []
        assert trial_buffer.get_trials("job-finite") == []
    finally:
        trial_buffer.clear_trials("job-finite")


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


# --- boosting iteration events ---------------------------------------------


def test_valid_iteration_publishes_one_event(captured):
    _node_runners._emit_iteration_event("job-1", 40, 200, 0.42, "logloss", "minimize")
    assert len(captured) == 1
    event = captured[0]
    assert event.event == "iteration"
    assert event.job_id == "job-1"
    assert event.iteration_number == 40
    assert event.iteration_total == 200
    assert event.iteration_score == 0.42
    assert event.iteration_metric == "logloss"
    assert event.iteration_direction == "minimize"


def test_iteration_numpy_score_is_cast_to_native_float(captured):
    _node_runners._emit_iteration_event("job-1", 1, 5, np.float64(0.31), "rmse", "minimize")
    assert type(captured[0].iteration_score) is float


def test_iteration_none_score_publishes_nothing(captured):
    _node_runners._emit_iteration_event("job-1", 2, 10, None, "logloss", "minimize")
    assert captured == []


def test_iteration_non_finite_score_publishes_nothing(captured):
    from backend.realtime import trial_buffer

    try:
        _node_runners._emit_iteration_event("job-ifin", 1, 10, float("-inf"), "logloss", "minimize")
        _node_runners._emit_iteration_event("job-ifin", 2, 10, float("nan"), "logloss", "minimize")
        assert captured == []
        assert trial_buffer.get_iterations("job-ifin") == []
    finally:
        trial_buffer.clear_iterations("job-ifin")


def test_iteration_unknown_job_sentinel_publishes_nothing(captured):
    _node_runners._emit_iteration_event("unknown", 1, 5, 0.7, "logloss", "minimize")
    assert captured == []


def test_iteration_transport_failure_is_swallowed(monkeypatch):
    def explode(_event):
        raise RuntimeError("bus down")

    monkeypatch.setattr(_node_runners, "publish_job_event", explode)
    _node_runners._emit_iteration_event("job-1", 1, 5, 0.7, "logloss", "minimize")


def test_iteration_is_recorded_for_backfill(captured):
    from backend.realtime import trial_buffer

    try:
        _node_runners._emit_iteration_event("job-bf", 1, 3, 0.5, "logloss", "minimize")
        assert trial_buffer.get_iterations("job-bf") == [
            {
                "iteration": 1,
                "total": 3,
                "score": 0.5,
                "metric": "logloss",
                "direction": "minimize",
            }
        ]
    finally:
        trial_buffer.clear_iterations("job-bf")
