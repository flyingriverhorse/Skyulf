"""Round-5 patch-coverage tests for the monitoring router's ks_statistic rename (Codecov follow-up).

Exercises the changed `ks_statistic` threshold plumbing that the fifth
Codecov patch report still saw uncovered: the custom-threshold builder, the
threshold-version match/create paths, the per-column drift summary, and the
alert persistence threshold assignment.
"""

from types import SimpleNamespace
from unittest.mock import AsyncMock, MagicMock

from backend.monitoring import router as monitoring_router

_EFFECTIVE = {"psi": 0.2, "ks_statistic": 0.1, "wasserstein": 0.1, "kl_divergence": 0.1}


class TestBuildDriftThresholds:
    def test_empty_when_no_overrides(self):
        assert monitoring_router._build_drift_thresholds(None, None, None, None) == {}

    def test_ks_override_lands_under_ks_statistic_key(self):
        assert monitoring_router._build_drift_thresholds(0.3, 0.2, 0.15, 0.12) == {
            "psi": 0.3,
            "ks_statistic": 0.2,
            "wasserstein": 0.15,
            "kl_divergence": 0.12,
        }


class TestThresholdVersionMatchAndCreate:
    @staticmethod
    def _db_with_latest(latest):
        result = MagicMock()
        result.scalar_one_or_none.return_value = latest
        db = MagicMock()
        db.execute = AsyncMock(return_value=result)
        db.add = MagicMock()
        db.flush = AsyncMock()
        return db

    async def test_matching_latest_version_is_reused(self):
        latest = SimpleNamespace(version=3, psi=0.2, ks=0.1, wasserstein=0.1, kl_divergence=0.1)
        db = self._db_with_latest(latest)
        assert await monitoring_router._get_or_create_threshold_version(db, _EFFECTIVE) is latest

    async def test_new_version_created_from_ks_statistic(self):
        db = self._db_with_latest(None)
        version = await monitoring_router._get_or_create_threshold_version(db, _EFFECTIVE)
        assert version.version == 1
        assert version.ks == 0.1
        db.add.assert_called_once_with(version)


class TestDriftColumnSummaryKsStatistic:
    def test_summary_includes_ks_statistic(self):
        col = SimpleNamespace(
            drift_detected=True,
            metrics=[
                SimpleNamespace(metric="psi", value=0.1),
                SimpleNamespace(metric="wasserstein_distance", value=0.05),
                SimpleNamespace(metric="ks_statistic", value=0.42),
                SimpleNamespace(metric="ks_test_p_value", value=0.001),
            ],
        )
        report = SimpleNamespace(column_drifts={"f1": col})
        summary = monitoring_router._build_drift_column_summary(report)
        assert summary["f1"]["ks_statistic"] == 0.42
        assert summary["f1"]["ks_p_value"] == 0.001


class TestSaveDriftAlertThresholdAssignment:
    async def test_thresholds_recorded_on_persisted_alert(self):
        db = MagicMock()
        db.commit = AsyncMock()
        db.refresh = AsyncMock()
        check = await monitoring_router._save_drift_alert(
            db,
            job_id="job-1",
            dataset_name="iris",
            evaluation_status="failed",
            effective_thresholds=_EFFECTIVE,
            error_message="boom",
        )
        assert check is not None
        assert check.threshold_ks == 0.1
        assert check.threshold_psi == 0.2
        db.commit.assert_awaited_once()
