"""Tests for the OPS-003 durable drift alert-to-remediation lifecycle.

Covers severity classification, threshold-version pinning (a threshold
change never rewrites a past alert's evaluated-against values), the
disposition state machine (new -> acknowledged -> reopened -> resolved,
with actor/timestamp and rejected invalid transitions), and the explicit
no-baseline/evaluation-failed persistence paths.
"""

from datetime import UTC, datetime
from unittest.mock import AsyncMock, MagicMock

import polars as pl
import pytest
from fastapi import HTTPException

from backend.monitoring.router import (
    DriftDispositionUpdate,
    _classify_drift_severity,
    _effective_drift_thresholds,
    _get_or_create_threshold_version,
    _save_drift_alert,
    get_drift_alert,
    get_drift_history,
    update_drift_alert_disposition,
)
from skyulf.profiling.drift import DriftCalculator


def _make_report(
    drifted_columns_count: int,
    total_columns: int,
    missing_columns: list[str] | None = None,
    new_columns: list[str] | None = None,
) -> MagicMock:
    """Build a lightweight mock standing in for a `DriftReport`."""
    report = MagicMock()
    report.reference_rows = 100
    report.current_rows = 100
    report.drifted_columns_count = drifted_columns_count
    report.missing_columns = missing_columns or []
    report.new_columns = new_columns or []
    column_drifts: dict[str, MagicMock] = {}
    for i in range(total_columns):
        col = MagicMock()
        col.drift_detected = i < drifted_columns_count
        metric = MagicMock()
        metric.metric = "psi"
        metric.value = 0.5 if col.drift_detected else 0.01
        col.metrics = [metric]
        col.model_dump.return_value = {"column": f"col_{i}", "drift_detected": col.drift_detected}
        column_drifts[f"col_{i}"] = col
    report.column_drifts = column_drifts
    report.model_dump.return_value = {"column_drifts": {}}
    return report


def _make_threshold_version(version: int, **kwargs) -> MagicMock:
    row = MagicMock()
    row.version = version
    row.psi = kwargs.get("psi", 0.2)
    row.ks = kwargs.get("ks", 0.1)
    row.wasserstein = kwargs.get("wasserstein", 0.1)
    row.kl_divergence = kwargs.get("kl_divergence", 0.1)
    return row


def _make_alert_row(**kwargs) -> MagicMock:
    """Build a lightweight mock standing in for a `DriftCheckResult` row."""
    row = MagicMock()
    row.id = kwargs.get("id", 1)
    row.job_id = kwargs.get("job_id", "job-1")
    row.dataset_name = kwargs.get("dataset_name", "sales")
    row.reference_rows = kwargs.get("reference_rows", 100)
    row.current_rows = kwargs.get("current_rows", 100)
    row.drifted_columns_count = kwargs.get("drifted_columns_count", 1)
    row.total_columns = kwargs.get("total_columns", 4)
    row.summary = kwargs.get("summary", {"col_0": {"drifted": True}})
    row.column_drifts = kwargs.get("column_drifts", {"col_0": {"drifted": True}})
    row.created_at = kwargs.get("created_at", datetime(2026, 8, 7, 10, 0, tzinfo=UTC))
    row.severity = kwargs.get("severity", "warning")
    row.status = kwargs.get("status", "new")
    row.owner = kwargs.get("owner")
    row.acknowledged_at = kwargs.get("acknowledged_at")
    row.resolved_at = kwargs.get("resolved_at")
    row.disposition_history = kwargs.get("disposition_history", [])
    row.threshold_version = kwargs.get("threshold_version", 1)
    row.threshold_psi = kwargs.get("threshold_psi", 0.2)
    row.threshold_ks = kwargs.get("threshold_ks", 0.05)
    row.threshold_wasserstein = kwargs.get("threshold_wasserstein", 0.1)
    row.threshold_kl = kwargs.get("threshold_kl", 0.1)
    row.deployment_id = kwargs.get("deployment_id")
    row.model_version = kwargs.get("model_version")
    row.evaluation_status = kwargs.get("evaluation_status", "completed")
    row.error_message = kwargs.get("error_message")
    return row


def _make_db(scalar_one_or_none: object = None, scalars_all: list | None = None) -> AsyncMock:
    """Build a mock `db` whose `execute(...)` result serves both query styles used here."""
    db = AsyncMock()
    db.add = MagicMock()  # SQLAlchemy's AsyncSession.add() is synchronous, not awaitable
    result = MagicMock()
    result.scalar_one_or_none.return_value = scalar_one_or_none
    if scalars_all is not None:
        result.scalars.return_value.all.return_value = scalars_all
    db.execute.return_value = result
    return db


class TestClassifyDriftSeverity:
    """Severity is derived once at evaluation time from the report's shape."""

    def test_no_drift_is_none(self) -> None:
        report = _make_report(drifted_columns_count=0, total_columns=4)
        assert _classify_drift_severity(report) == "none"

    def test_minor_drift_is_warning(self) -> None:
        report = _make_report(drifted_columns_count=1, total_columns=10)
        assert _classify_drift_severity(report) == "warning"

    def test_majority_drifted_is_critical(self) -> None:
        report = _make_report(drifted_columns_count=4, total_columns=10)
        assert _classify_drift_severity(report) == "critical"

    def test_schema_drift_is_always_critical(self) -> None:
        report = _make_report(
            drifted_columns_count=0, total_columns=4, missing_columns=["gone_col"]
        )
        assert _classify_drift_severity(report) == "critical"

    def test_new_columns_is_always_critical(self) -> None:
        report = _make_report(drifted_columns_count=0, total_columns=4, new_columns=["new_col"])
        assert _classify_drift_severity(report) == "critical"

    def test_schema_drift_count_and_severity_agree(self) -> None:
        """OC-45: a real schema-drift report cannot say critical and zero drift at once.

        The two tests above pin the classifier's early return with a mock whose
        count is 0 — a shape the calculator no longer produces, so neither can
        catch the contradiction. This one uses the real calculator: a dropped
        column has to land in `drifted_columns_count`, because the drift-status
        dashboard counts drifted jobs by that field and used to report no drift
        for a job this same report classified as critical.
        """
        shared = [1.0, 2.0, 3.0, 4.0, 5.0]
        report = DriftCalculator(
            pl.DataFrame({"stable": shared, "dropped": shared}),
            pl.DataFrame({"stable": shared}),
        ).calculate_drift()

        assert report.column_drifts["stable"].drift_detected is False
        assert report.drifted_columns_count == 1
        assert _classify_drift_severity(report) == "critical"


class TestEffectiveDriftThresholds:
    """Overrides layer onto the calculator's defaults for durable recording."""

    def test_empty_overrides_yield_defaults(self) -> None:
        effective = _effective_drift_thresholds({})
        assert effective == {
            "psi": 0.2,
            "ks_statistic": 0.1,
            "wasserstein": 0.1,
            "kl_divergence": 0.1,
        }

    def test_partial_override_keeps_other_defaults(self) -> None:
        effective = _effective_drift_thresholds({"psi": 0.4})
        assert effective["psi"] == 0.4
        assert effective["ks_statistic"] == 0.1


class TestThresholdVersioning:
    """A changed threshold set always gets a new version; unchanged ones reuse it."""

    @pytest.mark.asyncio
    async def test_first_check_creates_version_one(self) -> None:
        db = _make_db(scalar_one_or_none=None)
        version = await _get_or_create_threshold_version(
            db, {"psi": 0.2, "ks_statistic": 0.1, "wasserstein": 0.1, "kl_divergence": 0.1}
        )
        assert version.version == 1
        db.add.assert_called_once()

    @pytest.mark.asyncio
    async def test_unchanged_thresholds_reuse_latest_version(self) -> None:
        latest = _make_threshold_version(3)
        db = _make_db(scalar_one_or_none=latest)
        version = await _get_or_create_threshold_version(
            db, {"psi": 0.2, "ks_statistic": 0.1, "wasserstein": 0.1, "kl_divergence": 0.1}
        )
        assert version is latest
        db.add.assert_not_called()

    @pytest.mark.asyncio
    async def test_changed_threshold_creates_new_version_not_mutating_latest(self) -> None:
        latest = _make_threshold_version(3, psi=0.2)
        db = _make_db(scalar_one_or_none=latest)
        version = await _get_or_create_threshold_version(
            db, {"psi": 0.4, "ks_statistic": 0.1, "wasserstein": 0.1, "kl_divergence": 0.1}
        )
        # A new version row is created rather than overwriting `latest`.
        assert version is not latest
        assert version.version == 4
        assert latest.psi == 0.2  # past version's recorded value is untouched
        db.add.assert_called_once()


class TestSaveDriftAlertExplicitOutcomes:
    """No-alerts (none-severity), no-baseline, and evaluation-failed all persist a row."""

    @pytest.mark.asyncio
    async def test_completed_check_with_no_drift_persists_none_severity(self) -> None:
        report = _make_report(drifted_columns_count=0, total_columns=4)
        db = AsyncMock()
        db.add = MagicMock()
        alert = await _save_drift_alert(
            db,
            job_id="job-1",
            dataset_name="sales",
            evaluation_status="completed",
            report=report,
            effective_thresholds={
                "psi": 0.2,
                "ks_statistic": 0.1,
                "wasserstein": 0.1,
                "kl_divergence": 0.1,
            },
        )
        assert alert is not None
        assert alert.evaluation_status == "completed"
        assert alert.severity == "none"
        db.commit.assert_awaited()

    @pytest.mark.asyncio
    async def test_column_drifts_evidence_is_flat_per_column(self) -> None:
        """Evidence must mirror `EnrichedDriftReport.column_drifts`'s flat shape, not a nested dump."""
        report = _make_report(drifted_columns_count=1, total_columns=2)
        db = AsyncMock()
        db.add = MagicMock()
        alert = await _save_drift_alert(
            db,
            job_id="job-1",
            dataset_name="sales",
            evaluation_status="completed",
            report=report,
            effective_thresholds={
                "psi": 0.2,
                "ks_statistic": 0.1,
                "wasserstein": 0.1,
                "kl_divergence": 0.1,
            },
        )
        assert alert is not None
        assert set(alert.column_drifts.keys()) == {"col_0", "col_1"}
        assert "drift_detected" in alert.column_drifts["col_0"]

    @pytest.mark.asyncio
    async def test_no_baseline_persists_explicit_status(self) -> None:
        db = AsyncMock()
        db.add = MagicMock()
        alert = await _save_drift_alert(
            db,
            job_id="job-1",
            dataset_name="sales",
            evaluation_status="no_baseline",
            error_message="Reference data not found for job job-1",
        )
        assert alert is not None
        assert alert.evaluation_status == "no_baseline"
        assert alert.error_message == "Reference data not found for job job-1"
        assert alert.severity == "none"

    @pytest.mark.asyncio
    async def test_evaluation_failed_persists_error_message(self) -> None:
        db = AsyncMock()
        db.add = MagicMock()
        alert = await _save_drift_alert(
            db,
            job_id="job-1",
            dataset_name="sales",
            evaluation_status="failed",
            error_message="scipy is required for drift calculation",
        )
        assert alert is not None
        assert alert.evaluation_status == "failed"
        assert alert.error_message == "scipy is required for drift calculation"


class TestDriftHistoryFieldMapping:
    """History entries surface severity/status/owner/threshold-version/links."""

    @pytest.mark.asyncio
    async def test_history_entry_carries_full_lifecycle_fields(self) -> None:
        row = _make_alert_row(
            severity="critical",
            status="acknowledged",
            owner="alice",
            threshold_version=2,
            deployment_id=42,
            model_version="v3",
        )
        db = _make_db(scalars_all=[row])
        entries = await get_drift_history("job-1", db=db)
        assert len(entries) == 1
        entry = entries[0]
        assert entry.severity == "critical"
        assert entry.status == "acknowledged"
        assert entry.owner == "alice"
        assert entry.threshold_version == 2
        assert entry.deployment_id == 42
        assert entry.model_version == "v3"

    @pytest.mark.asyncio
    async def test_no_baseline_entry_is_distinguishable_from_no_drift(self) -> None:
        row = _make_alert_row(
            evaluation_status="no_baseline",
            drifted_columns_count=None,
            total_columns=None,
            summary=None,
            column_drifts=None,
            severity="none",
            error_message="Reference data not found for job job-1",
        )
        db = _make_db(scalars_all=[row])
        entries = await get_drift_history("job-1", db=db)
        assert entries[0].evaluation_status == "no_baseline"
        assert entries[0].drifted_columns_count is None
        assert entries[0].error_message == "Reference data not found for job job-1"


class TestGetDriftAlertDetail:
    """Full alert detail exposes evidence and the disposition audit trail."""

    @pytest.mark.asyncio
    async def test_detail_includes_evidence_and_disposition_history(self) -> None:
        row = _make_alert_row(
            disposition_history=[
                {
                    "status": "acknowledged",
                    "actor": "alice",
                    "note": None,
                    "at": "2026-08-07T11:00:00+00:00",
                }
            ],
        )
        db = _make_db(scalar_one_or_none=row)
        detail = await get_drift_alert(1, db=db)
        assert detail.column_drifts == {"col_0": {"drifted": True}}
        assert detail.disposition_history[0]["actor"] == "alice"

    @pytest.mark.asyncio
    async def test_missing_alert_raises_404(self) -> None:
        db = _make_db(scalar_one_or_none=None)
        with pytest.raises(HTTPException) as exc_info:
            await get_drift_alert(999, db=db)
        assert exc_info.value.status_code == 404


class TestDispositionStateMachine:
    """new -> acknowledged -> reopened -> resolved, with actor/timestamp recorded each step."""

    @pytest.mark.asyncio
    async def test_acknowledge_from_new_records_actor_and_timestamp(self) -> None:
        row = _make_alert_row(status="new")
        db = _make_db(scalar_one_or_none=row)
        result = await update_drift_alert_disposition(
            1,
            DriftDispositionUpdate(action="acknowledge", actor="alice", note="Looking into it"),
            db=db,
        )
        assert result.status == "acknowledged"
        assert result.owner == "alice"
        assert row.acknowledged_at is not None
        assert result.disposition_history[-1]["actor"] == "alice"
        assert result.disposition_history[-1]["note"] == "Looking into it"

    @pytest.mark.asyncio
    async def test_resolve_from_acknowledged_succeeds(self) -> None:
        row = _make_alert_row(status="acknowledged", owner="alice")
        db = _make_db(scalar_one_or_none=row)
        result = await update_drift_alert_disposition(
            1, DriftDispositionUpdate(action="resolve", actor="alice"), db=db
        )
        assert result.status == "resolved"
        assert row.resolved_at is not None

    @pytest.mark.asyncio
    async def test_reopen_from_resolved_clears_resolved_at(self) -> None:
        row = _make_alert_row(status="resolved", resolved_at=datetime(2026, 8, 7, tzinfo=UTC))
        db = _make_db(scalar_one_or_none=row)
        result = await update_drift_alert_disposition(
            1, DriftDispositionUpdate(action="reopen", actor="bob", note="Recurred"), db=db
        )
        assert result.status == "reopened"
        assert row.resolved_at is None
        assert result.owner == "bob"

    @pytest.mark.asyncio
    async def test_reacknowledge_after_reopen_completes_full_lifecycle(self) -> None:
        row = _make_alert_row(status="reopened")
        db = _make_db(scalar_one_or_none=row)
        result = await update_drift_alert_disposition(
            1, DriftDispositionUpdate(action="acknowledge", actor="carol"), db=db
        )
        assert result.status == "acknowledged"

    @pytest.mark.asyncio
    async def test_resolve_before_acknowledge_is_rejected(self) -> None:
        row = _make_alert_row(status="new")
        db = _make_db(scalar_one_or_none=row)
        with pytest.raises(HTTPException) as exc_info:
            await update_drift_alert_disposition(
                1, DriftDispositionUpdate(action="resolve", actor="alice"), db=db
            )
        assert exc_info.value.status_code == 409

    @pytest.mark.asyncio
    async def test_acknowledge_an_already_acknowledged_alert_is_rejected(self) -> None:
        row = _make_alert_row(status="acknowledged")
        db = _make_db(scalar_one_or_none=row)
        with pytest.raises(HTTPException) as exc_info:
            await update_drift_alert_disposition(
                1, DriftDispositionUpdate(action="acknowledge", actor="alice"), db=db
            )
        assert exc_info.value.status_code == 409

    @pytest.mark.asyncio
    async def test_disposition_on_missing_alert_raises_404(self) -> None:
        db = _make_db(scalar_one_or_none=None)
        with pytest.raises(HTTPException) as exc_info:
            await update_drift_alert_disposition(
                999, DriftDispositionUpdate(action="acknowledge", actor="alice"), db=db
            )
        assert exc_info.value.status_code == 404
