"""Unit tests for ``JobManager._job_summary_entry``.

The node-card summary builder behind ``GET /jobs/node-summaries``. It reads a
``summary`` string out of a job's ``metrics`` blob and derives a wall-clock
duration from the job's timestamps, so both the "no usable summary" exits and
the duration arithmetic are worth pinning.
"""

from datetime import UTC, datetime, timedelta

from backend.ml_pipeline._execution.jobs import JobManager


class _Job:
    """Minimal stand-in for a ``TrainingJob`` row."""

    def __init__(
        self,
        metrics=None,
        *,
        branch_index=0,
        pipeline_id="pipe-1",
        parent_pipeline_id=None,
        start_time=None,
        end_time=None,
    ):
        self.metrics = metrics
        self.branch_index = branch_index
        self.pipeline_id = pipeline_id
        self.parent_pipeline_id = parent_pipeline_id
        self.start_time = start_time
        self.end_time = end_time


def test_missing_metrics_yields_no_entry() -> None:
    assert JobManager._job_summary_entry(_Job(metrics=None)) is None


def test_empty_metrics_yields_no_entry() -> None:
    assert JobManager._job_summary_entry(_Job(metrics={})) is None


def test_metrics_without_a_summary_key_yields_no_entry() -> None:
    assert JobManager._job_summary_entry(_Job(metrics={"accuracy": 0.91})) is None


def test_non_string_summary_yields_no_entry() -> None:
    assert JobManager._job_summary_entry(_Job(metrics={"summary": 42})) is None


def test_whitespace_only_summary_yields_no_entry() -> None:
    assert JobManager._job_summary_entry(_Job(metrics={"summary": "   "})) is None


def test_summary_is_stripped_and_metadata_carried_through() -> None:
    end = datetime(2026, 9, 5, 12, 0, 0, tzinfo=UTC)
    entry = JobManager._job_summary_entry(
        _Job(
            metrics={"summary": "  Trained on 3 features  "},
            branch_index=2,
            pipeline_id="pipe-9",
            parent_pipeline_id="parent-9",
            end_time=end,
        )
    )

    assert entry == {
        "summary": "Trained on 3 features",
        "branch_index": 2,
        "pipeline_id": "pipe-9",
        "parent_pipeline_id": "parent-9",
        "finished_at": end.isoformat(),
    }


def test_duration_is_derived_from_both_timestamps() -> None:
    start = datetime(2026, 9, 5, 12, 0, 0, tzinfo=UTC)
    entry = JobManager._job_summary_entry(
        _Job(
            metrics={"summary": "ok"},
            start_time=start,
            end_time=start + timedelta(milliseconds=1250),
        )
    )

    assert entry["duration_ms"] == 1250


def test_duration_is_clamped_at_zero_for_clock_skew() -> None:
    start = datetime(2026, 9, 5, 12, 0, 0, tzinfo=UTC)
    entry = JobManager._job_summary_entry(
        _Job(metrics={"summary": "ok"}, start_time=start, end_time=start - timedelta(seconds=5))
    )

    assert entry["duration_ms"] == 0


def test_no_duration_when_a_timestamp_is_missing() -> None:
    end = datetime(2026, 9, 5, 12, 0, 0, tzinfo=UTC)
    entry = JobManager._job_summary_entry(_Job(metrics={"summary": "ok"}, end_time=end))

    assert "duration_ms" not in entry


def test_finished_at_is_null_without_an_end_time() -> None:
    entry = JobManager._job_summary_entry(_Job(metrics={"summary": "ok"}))

    assert entry["finished_at"] is None
