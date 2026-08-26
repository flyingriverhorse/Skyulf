"""In-memory per-job trial history for late-arriving chart subscribers.

The ``/ws/jobs`` socket is a live broadcast: a client that opens a running
tuning job only sees trials emitted *after* it subscribed. This buffer
records every published trial so the jobs API can hand a late opener the
trials it missed (1..now); the persisted ``metrics.trials`` list takes over
once the job is terminal.

Bounded on purpose — this is chart backfill, not persistence: a fixed
number of jobs, each with a fixed number of trials, evicted LRU.
"""

import threading
from collections import OrderedDict
from typing import Any

_MAX_JOBS = 128
_MAX_TRIALS_PER_JOB = 2000

_lock = threading.Lock()
_buffers: OrderedDict[str, list[dict[str, Any]]] = OrderedDict()


def record_trial(
    job_id: str,
    trial_number: int,
    trial_total: int,
    trial_score: float,
    trial_metric: str | None = None,
) -> None:
    """Append one completed trial to the job's backfill buffer."""
    with _lock:
        buf = _buffers.get(job_id)
        if buf is None:
            if len(_buffers) >= _MAX_JOBS:
                _buffers.popitem(last=False)
            buf = _buffers[job_id] = []
        buf.append(
            {
                "trial": trial_number,
                "total": trial_total,
                "score": trial_score,
                "metric": trial_metric,
            }
        )
        if len(buf) > _MAX_TRIALS_PER_JOB:
            del buf[: len(buf) - _MAX_TRIALS_PER_JOB]
        _buffers.move_to_end(job_id)


def get_trials(job_id: str) -> list[dict[str, Any]]:
    """Snapshot copy of the job's recorded trials (empty when unknown)."""
    with _lock:
        return [dict(entry) for entry in _buffers.get(job_id, ())]


def clear_trials(job_id: str) -> None:
    """Drop the job's buffer (called when it can no longer be backfilled)."""
    with _lock:
        _buffers.pop(job_id, None)


# Boosting iteration history (XGBoost/LightGBM) mirrors the trial buffer:
# one live chart series per job, same LRU bounds, independent of trials.
_iteration_buffers: OrderedDict[str, list[dict[str, Any]]] = OrderedDict()


def record_iteration(
    job_id: str,
    iteration_number: int,
    iteration_total: int,
    iteration_score: float,
    iteration_metric: str | None = None,
    iteration_direction: str | None = None,
) -> None:
    """Append one completed boosting iteration to the job's backfill buffer."""
    with _lock:
        buf = _iteration_buffers.get(job_id)
        if buf is None:
            if len(_iteration_buffers) >= _MAX_JOBS:
                _iteration_buffers.popitem(last=False)
            buf = _iteration_buffers[job_id] = []
        buf.append(
            {
                "iteration": iteration_number,
                "total": iteration_total,
                "score": iteration_score,
                "metric": iteration_metric,
                "direction": iteration_direction,
            }
        )
        if len(buf) > _MAX_TRIALS_PER_JOB:
            del buf[: len(buf) - _MAX_TRIALS_PER_JOB]
        _iteration_buffers.move_to_end(job_id)


def get_iterations(job_id: str) -> list[dict[str, Any]]:
    """Snapshot copy of the job's recorded iterations (empty when unknown)."""
    with _lock:
        return [dict(entry) for entry in _iteration_buffers.get(job_id, ())]


def clear_iterations(job_id: str) -> None:
    """Drop the job's iteration buffer."""
    with _lock:
        _iteration_buffers.pop(job_id, None)
