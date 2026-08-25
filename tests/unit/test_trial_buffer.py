"""Unit tests for the running-job trial backfill buffer.

The buffer exists so a client that opens a tuning job mid-run can fetch the
trials it missed (the WebSocket only broadcasts to connected subscribers).
It must record independently of transport success, stay bounded, and never
leak into training.
"""

import numpy as np
import pytest

from backend.ml_pipeline._execution.engine import _node_runners
from backend.realtime import trial_buffer


@pytest.fixture
def buffer():
    job_ids: list[str] = []

    class _Handle:
        def add(self, job_id: str) -> str:
            job_ids.append(job_id)
            return job_id

    yield _Handle()
    for job_id in job_ids:
        trial_buffer.clear_trials(job_id)


def test_emit_records_trial_for_late_openers(buffer, monkeypatch):
    monkeypatch.setattr(_node_runners, "publish_job_event", lambda _e: None)
    job = buffer.add("job-buf-1")
    _node_runners._emit_trial_event(job, 1, 5, 0.6, "accuracy")
    _node_runners._emit_trial_event(job, 2, 5, np.float64(0.8), "accuracy")

    trials = trial_buffer.get_trials(job)
    assert [t["trial"] for t in trials] == [1, 2]
    assert [t["total"] for t in trials] == [5, 5]
    assert [t["score"] for t in trials] == [0.6, 0.8]
    assert type(trials[1]["score"]) is float
    assert trials[0]["metric"] == "accuracy"


def test_scoreless_and_unknown_trials_are_not_recorded(buffer):
    _node_runners._emit_trial_event("job-buf-2", 2, 10, None, "f1")
    _node_runners._emit_trial_event("unknown", 1, 5, 0.7, "accuracy")
    assert trial_buffer.get_trials("job-buf-2") == []
    assert trial_buffer.get_trials("unknown") == []


def test_recording_survives_transport_failure(buffer, monkeypatch):
    def explode(_event):
        raise RuntimeError("bus down")

    monkeypatch.setattr(_node_runners, "publish_job_event", explode)
    job = buffer.add("job-buf-3")
    _node_runners._emit_trial_event(job, 1, 5, 0.7, "accuracy")
    # Backfill is independent of the broadcast — late openers still see it.
    assert len(trial_buffer.get_trials(job)) == 1


def test_get_trials_returns_a_copy(buffer):
    job = buffer.add("job-buf-4")
    _node_runners._emit_trial_event(job, 1, 3, 0.5, None)
    snapshot = trial_buffer.get_trials(job)
    snapshot.append({"trial": 99, "total": 3, "score": 1.0, "metric": None})
    assert len(trial_buffer.get_trials(job)) == 1


def test_per_job_cap_trims_oldest_trials(buffer):
    job = buffer.add("job-buf-5")
    for i in range(1, trial_buffer._MAX_TRIALS_PER_JOB + 100):
        trial_buffer.record_trial(job, i, 9999, 0.5, None)
    trials = trial_buffer.get_trials(job)
    assert len(trials) == trial_buffer._MAX_TRIALS_PER_JOB
    assert trials[0]["trial"] == 100
    assert trials[-1]["trial"] == trial_buffer._MAX_TRIALS_PER_JOB + 99


def test_job_count_cap_evicts_least_recent_job(buffer):
    jobs = [buffer.add(f"job-lru-{i}") for i in range(trial_buffer._MAX_JOBS + 5)]
    for job in jobs:
        trial_buffer.record_trial(job, 1, 1, 0.5, None)
    # Oldest jobs evicted once the cap is crossed; newest kept.
    assert trial_buffer.get_trials(jobs[0]) == []
    assert len(trial_buffer.get_trials(jobs[-1])) == 1
    assert len(trial_buffer._buffers) == trial_buffer._MAX_JOBS


def test_clear_trials_drops_the_buffer(buffer):
    job = buffer.add("job-buf-6")
    trial_buffer.record_trial(job, 1, 1, 0.5, None)
    trial_buffer.clear_trials(job)
    assert trial_buffer.get_trials(job) == []


def test_trials_endpoint_serves_snapshot(buffer):
    from fastapi import FastAPI
    from fastapi.testclient import TestClient

    from backend.ml_pipeline._internal._routers.jobs import router as jobs_router

    app = FastAPI()
    app.include_router(jobs_router, prefix="/api/pipeline")
    client = TestClient(app)

    job = buffer.add("job-buf-7")
    trial_buffer.record_trial(job, 1, 9, 0.6, "accuracy")
    trial_buffer.record_trial(job, 2, 9, 0.8, "accuracy")

    response = client.get(f"/api/pipeline/jobs/{job}/trials")
    assert response.status_code == 200
    body = response.json()
    assert [t["trial"] for t in body["trials"]] == [1, 2]
    assert body["metric"] == "accuracy"

    assert client.get("/api/pipeline/jobs/job-never-seen/trials").json() == {
        "trials": [],
        "metric": None,
    }
