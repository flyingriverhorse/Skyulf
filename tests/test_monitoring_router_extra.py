"""Focused unit tests for helper functions in backend.monitoring.router.

These target helper functions that were extracted during a pure-refactor
pass (extract-method complexity cleanup) and lost direct test coverage
attribution. We test them directly (mocking the DB session / artifact
store where needed) to restore coverage without depending on external
infra (S3, Celery, real drift-analysis math, etc).
"""

from datetime import UTC, datetime, timedelta
from unittest.mock import AsyncMock, MagicMock, patch

import pytest
from fastapi import HTTPException

from backend.monitoring.router import (
    DriftJobOption,
    _accumulate_node_timing,
    _build_drift_column_summary,
    _build_drift_metric_summary,
    _build_drift_thresholds,
    _build_slow_node_aggregates,
    _build_zero_filled_hour_buckets,
    _clamp_slow_nodes_params,
    _enrich_drift_job,
    _extract_drift_target_column,
    _fetch_drift_job_rows,
    _fill_error_buckets,
    _find_reference_key,
    _load_current_dataframe,
    _load_feature_importances,
    _load_reference_dataframe,
    _percentile,
    _save_drift_check_result,
    _scan_slow_node_jobs,
    list_drift_jobs,
    list_slow_nodes,
)


def _make_job_row(**kwargs):
    """Build a lightweight mock standing in for a TrainingJob row."""
    row = MagicMock()
    row.id = kwargs.get("id", "job-1")
    row.model_type = kwargs.get("model_type", "RandomForest")
    row.graph = kwargs.get("graph", {})
    row.node_id = kwargs.get("node_id", "node-1")
    row.job_metadata = kwargs.get("job_metadata", {})
    row.metrics = kwargs.get("metrics", {})
    return row


def _make_db_execute_result(scalars_list):
    """Build a mock `db.execute(...)` result whose `.scalars().all()` returns scalars_list."""
    result = MagicMock()
    result.scalars.return_value.all.return_value = scalars_list
    return result


# ---------------------------------------------------------------------------
# _fetch_drift_job_rows
# ---------------------------------------------------------------------------


async def test_fetch_drift_job_rows_merges_both_tables():
    """Rows from both run_modes ("fixed" and "tuned") are merged by id."""
    db = AsyncMock()
    row1 = _make_job_row(id="job-1")
    row2 = _make_job_row(id="job-2")
    db.execute.return_value = _make_db_execute_result([row1, row2])

    db_jobs = await _fetch_drift_job_rows(db, ["job-1", "job-2"])
    assert set(db_jobs.keys()) == {"job-1", "job-2"}


async def test_fetch_drift_job_rows_swallows_db_errors():
    """A DB failure while enriching is logged and results in an empty dict, not raised."""
    db = AsyncMock()
    db.execute.side_effect = RuntimeError("db down")

    db_jobs = await _fetch_drift_job_rows(db, ["job-1"])
    assert db_jobs == {}


# ---------------------------------------------------------------------------
# _extract_drift_target_column
# ---------------------------------------------------------------------------


def test_extract_drift_target_column_success():
    """Target column is resolved via extract_job_details when graph/node_id are valid."""
    row = _make_job_row(
        graph={"nodes": [{"id": "node-1", "type": "target", "data": {"column": "y"}}]},
        node_id="node-1",
    )
    with patch(
        "backend.monitoring.router.extract_job_details", return_value=("pipeline", "y", "cls")
    ):
        assert _extract_drift_target_column(row) == "y"


def test_extract_drift_target_column_handles_errors():
    """Any failure while resolving the target column returns None instead of raising."""
    row = _make_job_row(graph=None, node_id=None)
    with patch("backend.monitoring.router.extract_job_details", side_effect=ValueError("bad")):
        assert _extract_drift_target_column(row) is None


# ---------------------------------------------------------------------------
# _build_drift_metric_summary
# ---------------------------------------------------------------------------


def test_build_drift_metric_summary_formats_known_metrics():
    """Known metric keys are formatted into a compact `label: value` summary string."""
    summary = _build_drift_metric_summary({"test_accuracy": 0.987654, "test_f1_weighted": 0.5})
    assert summary == "acc: 0.9877 | f1: 0.5000"


def test_build_drift_metric_summary_returns_none_when_empty():
    """No recognised metric keys present returns None."""
    assert _build_drift_metric_summary({"unrelated": 1}) is None


def test_build_drift_metric_summary_ignores_non_numeric_values():
    """Non-numeric metric values are skipped even if the key matches."""
    assert _build_drift_metric_summary({"test_accuracy": "not-a-number"}) is None


# ---------------------------------------------------------------------------
# _enrich_drift_job
# ---------------------------------------------------------------------------


def test_enrich_drift_job_populates_fields():
    """Model type, target column, description and metric summary are copied from the DB row."""
    job = DriftJobOption(job_id="job-1", dataset_name="ds", filename="ds.csv")
    row = _make_job_row(
        model_type="XGBoost",
        job_metadata={"description": "my job"},
        metrics={"n_rows": 100, "n_features": 5, "test_accuracy": 0.9},
    )
    with patch("backend.monitoring.router._extract_drift_target_column", return_value="y"):
        _enrich_drift_job(job, row)

    assert job.model_type == "XGBoost"
    assert job.target_column == "y"
    assert job.description == "my job"
    assert job.n_rows == 100
    assert job.n_features == 5
    assert job.best_metric == "acc: 0.9000"


# ---------------------------------------------------------------------------
# list_drift_jobs endpoint
# ---------------------------------------------------------------------------


async def test_list_drift_jobs_returns_empty_when_no_artifacts():
    """No discovered reference artifacts short-circuits to an empty list."""
    db = AsyncMock()
    discovery = MagicMock()
    discovery.list_reference_artifacts.return_value = []

    with patch("backend.monitoring.router.ArtifactFactory") as mock_factory:
        mock_factory.get_discovery.return_value = discovery
        result = await list_drift_jobs(db)

    assert result == []


async def test_list_drift_jobs_enriches_and_sorts():
    """Discovered jobs are enriched from the DB and sorted by created_at descending."""
    db = AsyncMock()
    ref_old = MagicMock(
        job_id="job-1", dataset_name="ds1", filename="ds1.csv", created_at="2024-01-01"
    )
    ref_new = MagicMock(
        job_id="job-2", dataset_name="ds2", filename="ds2.csv", created_at="2024-06-01"
    )
    discovery = MagicMock()
    discovery.list_reference_artifacts.return_value = [ref_old, ref_new]

    row = _make_job_row(id="job-1")
    db.execute.side_effect = [
        _make_db_execute_result([row]),
        _make_db_execute_result([]),
    ]

    with patch("backend.monitoring.router.ArtifactFactory") as mock_factory:
        mock_factory.get_discovery.return_value = discovery
        result = await list_drift_jobs(db)

    assert [j.job_id for j in result] == ["job-2", "job-1"]


# ---------------------------------------------------------------------------
# _find_reference_key
# ---------------------------------------------------------------------------


def test_find_reference_key_by_dataset_name():
    """An exact sanitized dataset_name match is preferred when it exists."""
    artifact_store = MagicMock()
    artifact_store.exists.return_value = True

    key = _find_reference_key(artifact_store, "My Dataset!", "job-1")
    assert key == "reference_data_My_Dataset__job-1"
    artifact_store.list_artifacts.assert_not_called()


def test_find_reference_key_falls_back_to_search():
    """When no exact dataset_name match exists, search all artifacts for a suffix match."""
    artifact_store = MagicMock()
    artifact_store.exists.return_value = False
    artifact_store.list_artifacts.return_value = [
        "unrelated_key",
        "reference_data_other_job-1",
    ]

    key = _find_reference_key(artifact_store, "My Dataset", "job-1")
    assert key == "reference_data_other_job-1"


def test_find_reference_key_no_dataset_name_searches_directly():
    """With no dataset_name, go straight to searching all artifacts."""
    artifact_store = MagicMock()
    artifact_store.list_artifacts.return_value = ["reference_data_x_job-9"]

    key = _find_reference_key(artifact_store, None, "job-9")
    assert key == "reference_data_x_job-9"


def test_find_reference_key_not_found():
    """No matching artifact returns None."""
    artifact_store = MagicMock()
    artifact_store.exists.return_value = False
    artifact_store.list_artifacts.return_value = ["reference_data_other_job-2"]

    key = _find_reference_key(artifact_store, "ds", "job-1")
    assert key is None


# ---------------------------------------------------------------------------
# _load_reference_dataframe
# ---------------------------------------------------------------------------


def test_load_reference_dataframe_converts_pandas():
    """A pandas artifact is converted to a Polars DataFrame."""
    import pandas as pd

    artifact_store = MagicMock()
    artifact_store.load.return_value = pd.DataFrame({"a": [1, 2]})

    import polars as pl

    df = _load_reference_dataframe(artifact_store, "ref-key", "job-1")
    assert isinstance(df, pl.DataFrame)


def test_load_reference_dataframe_raises_skyulf_exception_on_failure():
    """A load failure surfaces as a SkyulfException, not the raw exception."""
    from backend.exceptions.core import SkyulfException

    artifact_store = MagicMock()
    artifact_store.load.side_effect = RuntimeError("boom")

    with pytest.raises(SkyulfException):
        _load_reference_dataframe(artifact_store, "ref-key", "job-1")


# ---------------------------------------------------------------------------
# _load_current_dataframe
# ---------------------------------------------------------------------------


def _make_upload_file(filename: str, content: bytes, max_read: int | None = None):
    upload = MagicMock()
    upload.filename = filename

    async def _read(n=-1):
        return content

    upload.read = AsyncMock(side_effect=_read)
    return upload


async def test_load_current_dataframe_csv():
    """A .csv upload is parsed with pl.read_csv."""
    upload = _make_upload_file("data.csv", b"a,b\n1,2\n")
    df = await _load_current_dataframe(upload)
    assert df.columns == ["a", "b"]


async def test_load_current_dataframe_parquet():
    """A .parquet upload is parsed with pl.read_parquet."""
    import io

    import polars as pl

    buf = io.BytesIO()
    pl.DataFrame({"a": [1, 2]}).write_parquet(buf)
    upload = _make_upload_file("data.parquet", buf.getvalue())

    df = await _load_current_dataframe(upload)
    assert df.columns == ["a"]


async def test_load_current_dataframe_too_large_raises_400():
    """Content exceeding MAX_UPLOAD_SIZE is rejected.

    Note: the inner HTTPException(413) is raised inside the function's own
    try block, so it gets caught by the broad `except Exception` alongside
    genuine parse failures and re-raised as a 400 - this is the real,
    existing behavior of the endpoint (not something this test suite
    should silently "fix").
    """
    fake_settings = MagicMock()
    fake_settings.MAX_UPLOAD_SIZE = 5
    upload = _make_upload_file("data.csv", b"1234567890")

    with (
        patch("backend.config.get_settings", return_value=fake_settings),
        pytest.raises(HTTPException) as exc_info,
    ):
        await _load_current_dataframe(upload)

    assert exc_info.value.status_code == 400


async def test_load_current_dataframe_parse_failure_raises_400():
    """Malformed content that fails to parse raises an HTTP 400."""
    upload = _make_upload_file("data.parquet", b"not-a-parquet-file")

    with pytest.raises(HTTPException) as exc_info:
        await _load_current_dataframe(upload)

    assert exc_info.value.status_code == 400


# ---------------------------------------------------------------------------
# _build_drift_thresholds
# ---------------------------------------------------------------------------


def test_build_drift_thresholds_all_none():
    """All-None thresholds produce an empty dict."""
    assert _build_drift_thresholds(None, None, None, None) == {}


def test_build_drift_thresholds_partial():
    """Only the provided thresholds are included, under their canonical keys."""
    thresholds = _build_drift_thresholds(0.1, None, 0.3, 0.4)
    assert thresholds == {"psi": 0.1, "wasserstein": 0.3, "kl_divergence": 0.4}


# ---------------------------------------------------------------------------
# _build_drift_column_summary
# ---------------------------------------------------------------------------


def test_build_drift_column_summary():
    """Per-column metrics are collapsed into a compact drifted/psi/wasserstein/ks summary."""
    metric = MagicMock(metric="psi", value=0.5)
    col_drift = MagicMock(drift_detected=True, metrics=[metric])
    report = MagicMock(column_drifts={"col_a": col_drift})

    summary = _build_drift_column_summary(report)
    assert summary["col_a"]["drifted"] is True
    assert summary["col_a"]["psi"] == 0.5
    assert summary["col_a"]["wasserstein"] is None


# ---------------------------------------------------------------------------
# _save_drift_check_result
# ---------------------------------------------------------------------------


async def test_save_drift_check_result_persists():
    """A successful save adds a DriftCheckResult row and commits."""
    db = AsyncMock()
    db.add = MagicMock()
    report = MagicMock(
        reference_rows=10,
        current_rows=12,
        drifted_columns_count=1,
        column_drifts={},
    )
    report.model_dump.return_value = {}

    await _save_drift_check_result(db, report, "job-1", "ds")

    db.add.assert_called_once()
    db.commit.assert_awaited_once()


async def test_save_drift_check_result_swallows_errors():
    """A failure while persisting is logged, not raised (history is best-effort)."""
    db = AsyncMock()
    db.add = MagicMock(side_effect=RuntimeError("boom"))
    report = MagicMock(
        reference_rows=10, current_rows=12, drifted_columns_count=0, column_drifts={}
    )
    report.model_dump.return_value = {}

    # Should not raise.
    await _save_drift_check_result(db, report, "job-1", "ds")


# ---------------------------------------------------------------------------
# _load_feature_importances
# ---------------------------------------------------------------------------


async def test_load_feature_importances_found():
    """Feature importances are returned when present on the job's metrics."""
    db = AsyncMock()
    row = _make_job_row(metrics={"feature_importances": {"a": 0.5}})
    db.execute.return_value = _make_db_execute_result_scalar_one(row)

    result = await _load_feature_importances(db, "job-1")
    assert result == {"a": 0.5}


async def test_load_feature_importances_not_found():
    """No matching job row across either table returns None."""
    db = AsyncMock()
    db.execute.return_value = _make_db_execute_result_scalar_one(None)

    result = await _load_feature_importances(db, "job-1")
    assert result is None


async def test_load_feature_importances_swallows_errors():
    """A DB error is logged and treated as no feature importances available."""
    db = AsyncMock()
    db.execute.side_effect = RuntimeError("boom")

    result = await _load_feature_importances(db, "job-1")
    assert result is None


def _make_db_execute_result_scalar_one(row):
    result = MagicMock()
    result.scalar_one_or_none.return_value = row
    return result


# ---------------------------------------------------------------------------
# _build_zero_filled_hour_buckets / _fill_error_buckets
# ---------------------------------------------------------------------------


def test_build_zero_filled_hour_buckets():
    """Buckets are created hourly starting at cutoff, all initialised to zero."""
    cutoff = datetime(2024, 1, 1, 10, 30, tzinfo=UTC)
    buckets = _build_zero_filled_hour_buckets(cutoff, 3)
    assert list(buckets.keys()) == [
        "2024-01-01T10:00",
        "2024-01-01T11:00",
        "2024-01-01T12:00",
    ]
    assert all(v == 0 for v in buckets.values())


def test_fill_error_buckets_increments_matching_slots():
    """Timestamps falling within a known bucket increment its count; unknown ones are ignored."""
    buckets = {"2024-01-01T10:00": 0, "2024-01-01T11:00": 0}
    timestamps = [
        datetime(2024, 1, 1, 10, 15, tzinfo=UTC),
        datetime(2024, 1, 1, 10, 45),  # naive -> normalised to UTC
        datetime(2024, 1, 1, 23, 0, tzinfo=UTC),  # outside buckets, ignored
        None,  # ignored
    ]
    _fill_error_buckets(buckets, timestamps)
    assert buckets["2024-01-01T10:00"] == 2
    assert buckets["2024-01-01T11:00"] == 0


# ---------------------------------------------------------------------------
# _percentile
# ---------------------------------------------------------------------------


def test_percentile_empty_list():
    """An empty list of values returns 0.0."""
    assert _percentile([], 95) == 0.0


def test_percentile_single_value():
    """A single-element list returns that element regardless of percentile."""
    assert _percentile([42.0], 50) == 42.0


def test_percentile_multiple_values():
    """A percentile of a sorted list returns a plausible nearest-rank value."""
    values = [1.0, 2.0, 3.0, 4.0, 5.0]
    assert _percentile(values, 0) == 1.0
    assert _percentile(values, 100) == 5.0


# ---------------------------------------------------------------------------
# _clamp_slow_nodes_params
# ---------------------------------------------------------------------------


def test_clamp_slow_nodes_params_within_bounds():
    """Values within the configured caps are returned unchanged."""
    fake_settings = MagicMock()
    fake_settings.MONITORING_MAX_SLOWNODES_DAYS = 30
    fake_settings.MAX_PAGE_SIZE = 100

    with patch("backend.config.get_settings", return_value=fake_settings):
        days, limit = _clamp_slow_nodes_params(7, 10)
    assert (days, limit) == (7, 10)


def test_clamp_slow_nodes_params_clamped_to_caps():
    """Values exceeding the configured caps are clamped down, and non-positive values clamped up."""
    fake_settings = MagicMock()
    fake_settings.MONITORING_MAX_SLOWNODES_DAYS = 30
    fake_settings.MAX_PAGE_SIZE = 100

    with patch("backend.config.get_settings", return_value=fake_settings):
        days, limit = _clamp_slow_nodes_params(9999, -5)
    assert (days, limit) == (30, 1)


# ---------------------------------------------------------------------------
# _accumulate_node_timing
# ---------------------------------------------------------------------------


def test_accumulate_node_timing_valid_entry():
    """A valid dict entry with positive execution_time contributes to the aggregates."""
    by_step: dict = {}
    sample_node: dict = {}
    contributed = _accumulate_node_timing(
        {"step_type": "impute", "execution_time": 1.5, "node_id": "n1"}, by_step, sample_node
    )
    assert contributed is True
    assert by_step["impute"] == [1.5]
    assert sample_node["impute"] == "n1"


def test_accumulate_node_timing_non_dict_entry_ignored():
    """Non-dict entries are ignored."""
    by_step: dict = {}
    sample_node: dict = {}
    assert _accumulate_node_timing("not-a-dict", by_step, sample_node) is False
    assert by_step == {}


def test_accumulate_node_timing_zero_or_negative_time_ignored():
    """Entries with zero or negative execution_time do not contribute."""
    by_step: dict = {}
    sample_node: dict = {}
    assert (
        _accumulate_node_timing({"step_type": "x", "execution_time": 0}, by_step, sample_node)
        is False
    )
    assert by_step == {}


def test_accumulate_node_timing_invalid_time_type_ignored():
    """Entries with a non-numeric execution_time are ignored rather than raising."""
    by_step: dict = {}
    sample_node: dict = {}
    result = _accumulate_node_timing(
        {"step_type": "x", "execution_time": "bad"}, by_step, sample_node
    )
    assert result is False
    assert by_step == {}


def test_accumulate_node_timing_missing_step_type_defaults_unknown():
    """A missing step_type defaults to 'unknown' rather than raising a KeyError."""
    by_step: dict = {}
    sample_node: dict = {}
    _accumulate_node_timing({"execution_time": 2.0}, by_step, sample_node)
    assert "unknown" in by_step


# ---------------------------------------------------------------------------
# _scan_slow_node_jobs
# ---------------------------------------------------------------------------


async def test_scan_slow_node_jobs_aggregates_across_tables():
    """Completed jobs from the unified table are scanned and their node_timings aggregated."""
    db = AsyncMock()
    job1 = MagicMock(metrics={"node_timings": [{"step_type": "impute", "execution_time": 1.0}]})
    job2 = MagicMock(metrics={"node_timings": [{"step_type": "impute", "execution_time": 2.0}]})
    db.execute.return_value = _make_db_execute_result([job1, job2])

    cutoff = datetime.now(UTC) - timedelta(days=7)
    by_step, sample_node, jobs_scanned, runs_seen = await _scan_slow_node_jobs(db, cutoff)

    assert jobs_scanned == 2
    assert runs_seen == 2
    assert by_step["impute"] == [1.0, 2.0]


async def test_scan_slow_node_jobs_skips_jobs_without_node_timings():
    """Jobs whose metrics lack a `node_timings` list are counted as scanned but contribute no runs."""
    db = AsyncMock()
    job_no_metrics = MagicMock(metrics=None)
    job_bad_shape = MagicMock(metrics={"node_timings": "not-a-list"})
    db.execute.return_value = _make_db_execute_result([job_no_metrics, job_bad_shape])

    cutoff = datetime.now(UTC) - timedelta(days=7)
    by_step, sample_node, jobs_scanned, runs_seen = await _scan_slow_node_jobs(db, cutoff)

    assert jobs_scanned == 2
    assert runs_seen == 0
    assert by_step == {}


# ---------------------------------------------------------------------------
# _build_slow_node_aggregates
# ---------------------------------------------------------------------------


def test_build_slow_node_aggregates_sorted_by_total_desc():
    """Aggregates are computed per step and sorted by total_seconds descending."""
    by_step = {"impute": [1.0, 2.0], "scale": [10.0]}
    sample_node = {"impute": "n1", "scale": "n2"}

    aggregates = _build_slow_node_aggregates(by_step, sample_node)
    assert [a.step_type for a in aggregates] == ["scale", "impute"]
    assert aggregates[1].count == 2
    assert aggregates[1].avg_seconds == 1.5
    assert aggregates[0].sample_node_id == "n2"


# ---------------------------------------------------------------------------
# list_slow_nodes endpoint
# ---------------------------------------------------------------------------


async def test_list_slow_nodes_end_to_end():
    """The endpoint clamps params, scans jobs, aggregates, and truncates to `limit`."""
    db = AsyncMock()
    job = MagicMock(
        status="completed",
        metrics={
            "node_timings": [
                {"step_type": "impute", "execution_time": 1.0},
                {"step_type": "scale", "execution_time": 5.0},
            ]
        },
    )
    db.execute.side_effect = [
        _make_db_execute_result([job]),
        _make_db_execute_result([]),
    ]

    fake_settings = MagicMock()
    fake_settings.MONITORING_MAX_SLOWNODES_DAYS = 30
    fake_settings.MAX_PAGE_SIZE = 100

    with patch("backend.config.get_settings", return_value=fake_settings):
        response = await list_slow_nodes(days=7, limit=1, db=db)

    assert response.days == 7
    assert response.total_jobs_scanned == 1
    assert response.total_node_runs == 2
    # Truncated to `limit=1`, and the highest total_seconds entry ("scale") wins.
    assert len(response.aggregates) == 1
    assert response.aggregates[0].step_type == "scale"


# ---------------------------------------------------------------------------
# calculate_drift endpoint (full HTTP round-trip)
# ---------------------------------------------------------------------------


def test_calculate_drift_end_to_end(tmp_path):
    """Full round trip through the /monitoring/drift/calculate endpoint.

    Exercises the endpoint's decorator lines and its full body (reference
    lookup, current-data parsing, real DriftCalculator run, history save,
    and feature-importance lookup) using a real small dataset so no drift
    math needs mocking.
    """
    import polars as pl
    from fastapi.testclient import TestClient

    from backend.dependencies import get_db
    from backend.main import app

    ref_df = pl.DataFrame({"a": [1, 2, 3, 4, 5] * 5})

    artifact_store = MagicMock()
    artifact_store.exists.return_value = True
    artifact_store.load.return_value = ref_df

    discovery = MagicMock()
    discovery.get_store_for_job.return_value = artifact_store

    db = AsyncMock()
    db.add = MagicMock()
    db.execute.return_value = _make_db_execute_result_scalar_one(None)

    async def _override_get_db():
        yield db

    app.dependency_overrides[get_db] = _override_get_db
    try:
        with (
            patch("backend.monitoring.router.ArtifactFactory") as mock_factory,
            TestClient(app, base_url="http://localhost") as client,
        ):
            mock_factory.get_discovery.return_value = discovery
            csv_bytes = b"a\n" + b"\n".join(str(v).encode() for v in [1, 2, 3, 4, 5] * 5)
            response = client.post(
                "/api/monitoring/drift/calculate",
                data={"job_id": "job-1", "dataset_name": "ds"},
                files={"file": ("current.csv", csv_bytes, "text/csv")},
            )
    finally:
        app.dependency_overrides.pop(get_db, None)

    assert response.status_code == 200, response.text
    body = response.json()
    assert body["reference_rows"] == 25
    assert "a" in body["column_drifts"]
