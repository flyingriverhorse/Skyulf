"""In-memory per-job trial history for late-arriving chart subscribers.

The ``/ws/jobs`` socket is a live broadcast: a client that opens a running
tuning job only sees trials emitted *after* it subscribed. This buffer
records every published trial so the jobs API can hand a late opener the
trials it missed (1..now); the persisted ``metrics.trials`` list takes over
once the job is terminal.

Bounded on purpose — this is chart backfill, not persistence: a fixed
number of jobs, each with a fixed number of trials, evicted LRU.
"""

from __future__ import annotations

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
