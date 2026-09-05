"""Unit tests for the ``trial`` extension of ``backend.realtime.events``.

The live tuning-trial chart adds a structured ``trial`` event to the
existing invalidator-style ``/ws/jobs`` channel. These tests pin the
contract: the new event validates, legacy events stay byte-identical,
payloads carry aggregate scalars only (the channel is a global
unauthenticated broadcast), and numpy scores serialize cleanly.
"""

import numpy as np
import orjson
import pytest

from backend.realtime.events import JobEvent


def test_trial_event_validates():
    event = JobEvent(
        event="trial",
        job_id="job-1",
        trial_number=3,
        trial_total=10,
        trial_score=0.85,
        trial_metric="f1",
    )
    assert event.trial_number == 3
    assert event.trial_total == 10
    assert event.trial_score == 0.85
    assert event.trial_metric == "f1"


@pytest.mark.parametrize("legacy_event", ["status", "progress", "created", "deleted"])
def test_legacy_event_types_still_validate(legacy_event):
    assert JobEvent(event=legacy_event, job_id="job-1").event == legacy_event


def test_legacy_progress_payload_unchanged():
    """Extension, not replacement: existing consumers see the same bytes."""
    event = JobEvent(event="progress", job_id="job-1", progress=42)
    assert event.model_dump(exclude_none=True) == {
        "event": "progress",
        "job_id": "job-1",
        "progress": 42,
    }


def test_model_has_no_params_field():
    """/ws/jobs broadcasts to every client without auth — hyperparameter
    payloads must not ride along with trial events."""
    assert "params" not in JobEvent.model_fields


def test_numpy_score_coerces_to_native_float():
    event = JobEvent(
        event="trial",
        job_id="job-1",
        trial_number=1,
        trial_total=5,
        trial_score=np.float64(0.91),
        trial_metric="accuracy",
    )
    assert type(event.trial_score) is float
    # orjson rejects numpy scalars outright — this must serialize.
    payload = orjson.dumps(event.model_dump(exclude_none=True))
    assert orjson.loads(payload)["trial_score"] == 0.91


def test_iteration_event_validates():
    """Boosting fits stream one ``iteration`` event per boosting round."""
    event = JobEvent(
        event="iteration",
        job_id="job-1",
        iteration_number=40,
        iteration_total=200,
        iteration_score=0.42,
        iteration_metric="logloss",
        iteration_direction="minimize",
    )
    assert event.iteration_number == 40
    assert event.iteration_total == 200
    assert event.iteration_score == 0.42
    assert event.iteration_metric == "logloss"
    assert event.iteration_direction == "minimize"


def test_iteration_payload_dumps_only_set_fields():
    event = JobEvent(
        event="iteration",
        job_id="job-1",
        iteration_number=1,
        iteration_total=2,
        iteration_score=0.5,
        iteration_metric="auc",
        iteration_direction="maximize",
    )
    assert event.model_dump(exclude_none=True) == {
        "event": "iteration",
        "job_id": "job-1",
        "iteration_number": 1,
        "iteration_total": 2,
        "iteration_score": 0.5,
        "iteration_metric": "auc",
        "iteration_direction": "maximize",
    }


def test_iteration_numpy_score_coerces_to_native_float():
    event = JobEvent(
        event="iteration",
        job_id="job-1",
        iteration_number=1,
        iteration_total=5,
        iteration_score=np.float32(0.31),
    )
    assert type(event.iteration_score) is float
    payload = orjson.dumps(event.model_dump(exclude_none=True))
    assert orjson.loads(payload)["iteration_score"] == pytest.approx(0.31)
