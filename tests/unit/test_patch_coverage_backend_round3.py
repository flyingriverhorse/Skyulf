"""Round-3 patch-coverage tests for backend defensive branches (Codecov follow-up).

Forces the previously-unexecuted except/fallback paths flagged by the third
Codecov patch report: monitoring optional-metadata lookups, profiler
per-column isolation, the ingestion task boundary, registry directory
creation, startup migrations and the DB health probe, DataSource creator
fallback, hash-query fallback, async error persistence, the strategy
summary fallback, and the dtype-breakdown renderer.
"""

import logging
from pathlib import Path
from types import SimpleNamespace
from typing import Any, cast
from unittest.mock import AsyncMock, MagicMock

from backend.data_ingestion import tasks as tasks_mod
from backend.data_ingestion.engine.profiler import DataProfiler
from backend.database import async_registry
from backend.database import engine as db_engine_mod
from backend.database.models import DataSource
from backend.database.repository import DataSourceRepository
from backend.exceptions import handlers as handlers_mod
from backend.ml_pipeline._execution import strategies as strategies_mod
from backend.ml_pipeline._execution.strategies import BasicTrainingStrategy
from backend.ml_pipeline._execution.summary import _dtype_breakdown
from backend.monitoring import router as monitoring_router


def _raiser(exc: Exception):
    def fn(*args, **kwargs):
        raise exc

    return fn


class TestMonitoringOptionalMetadataFallbacks:
    async def test_find_deployment_context_returns_none_pair_on_db_error(self):
        db = MagicMock()
        db.execute = AsyncMock(side_effect=RuntimeError("db down"))
        assert await monitoring_router._find_deployment_context(db, "job-1") == (None, None)

    async def test_load_feature_importances_returns_none_on_db_error(self):
        db = MagicMock()
        db.execute = AsyncMock(side_effect=RuntimeError("db down"))
        assert await monitoring_router._load_feature_importances(db, "job-1") is None

    async def test_save_drift_alert_returns_none_when_persistence_fails(self):
        db = MagicMock()
        db.add = MagicMock(side_effect=RuntimeError("db down"))
        db.rollback = AsyncMock()
        result = await monitoring_router._save_drift_alert(
            db, job_id="job-1", dataset_name="iris", evaluation_status="completed"
        )
        assert result is None
        db.rollback.assert_awaited_once()


class TestProfilerPerColumnIsolation:
    class _BrokenValueCountsSeries:
        dtype = "String"

        def null_count(self):
            return 0

        def n_unique(self):
            return 1

        def value_counts(self):
            raise RuntimeError("value_counts broken")

    class _FakeProfileFrame:
        columns = ["a"]

        def __len__(self):
            return 2

        def __getitem__(self, col):
            return TestProfilerPerColumnIsolation._BrokenValueCountsSeries()

        def is_duplicated(self):
            return SimpleNamespace(sum=lambda: 0)

    def test_failing_top_values_is_skipped_not_fatal(self, caplog):
        with caplog.at_level(logging.WARNING):
            profile = DataProfiler.profile(cast("Any", self._FakeProfileFrame()))
        assert "type" in profile["columns"]["a"]
        assert "top_values" not in profile["columns"]["a"]
        assert "Failed to compute top values" in caplog.text


class TestIngestionTaskBoundary:
    def test_task_failure_is_routed_to_failure_handler(self, monkeypatch):
        session = MagicMock()
        session.query.side_effect = RuntimeError("db down")
        monkeypatch.setattr(tasks_mod, "get_db_session", lambda: session)
        handled: list[tuple[int, Exception]] = []
        monkeypatch.setattr(
            tasks_mod, "_handle_ingestion_failure", lambda s, sid, e: handled.append((sid, e))
        )

        tasks_mod.ingest_data_task(7)  # must not raise

        assert handled and handled[0][0] == 7
        session.close.assert_called_once()


class TestRegistryDirectoryBestEffort:
    def test_mkdir_failure_does_not_block_path_resolution(self, monkeypatch):
        monkeypatch.setattr(Path, "mkdir", _raiser(OSError("no permission")))
        settings = SimpleNamespace(DB_PATH="nonexistent_subdir_xyz/registry.db")
        resolved = async_registry._resolve_registry_db_path(cast("Any", settings))
        assert resolved.name == "registry.db"


class TestDatabaseEngineDefenses:
    class _ExplodingConn:
        async def execute(self, *args, **kwargs):
            raise RuntimeError("duplicate column")

    class _FakeBeginCM:
        async def __aenter__(self):
            return TestDatabaseEngineDefenses._ExplodingConn()

        async def __aexit__(self, *exc):
            return False

    async def test_migrations_skip_failing_statements(self, monkeypatch):
        fake_engine = MagicMock()
        fake_engine.begin.return_value = self._FakeBeginCM()
        monkeypatch.setattr(db_engine_mod, "async_engine", fake_engine)
        await db_engine_mod._run_migrations()  # must not raise

    async def test_health_check_returns_false_on_probe_failure(self, monkeypatch):
        fake_engine = MagicMock()
        fake_engine.begin.return_value = self._FakeBeginCM()
        monkeypatch.setattr(db_engine_mod, "async_engine", fake_engine)
        assert await db_engine_mod.health_check() is False


class TestDataSourceCreatorFallback:
    def test_to_dict_reports_unknown_creator_when_access_fails(self, monkeypatch):
        def _boom(self):
            raise RuntimeError("creator relation unavailable")

        monkeypatch.setattr(DataSource, "creator", property(_boom))
        data = DataSource().to_dict()
        assert data["created_by"] == "Unknown"


class TestDataSourceHashQueryFallback:
    async def test_query_failure_returns_none(self):
        session = MagicMock()
        session.execute = AsyncMock(side_effect=RuntimeError("query broken"))
        repo = DataSourceRepository(session)
        assert await repo.get_by_file_hash("abc123") is None


class TestAsyncErrorPersistenceBestEffort:
    async def test_record_error_swallows_session_factory_failures(self, monkeypatch):
        monkeypatch.setattr(db_engine_mod, "async_session_factory", _raiser(RuntimeError("no db")))
        # Must not raise: error persistence is best-effort.
        await handlers_mod._record_error("route", "ErrorType", "msg", "tb", 500)


class TestStrategySummaryFallback:
    def test_build_summary_failure_renders_none(self, monkeypatch):
        monkeypatch.setattr(strategies_mod, "build_summary", _raiser(RuntimeError("broken")))
        strategy = object.__new__(BasicTrainingStrategy)
        job = SimpleNamespace(step_type="training")
        last_result = SimpleNamespace(metadata=None)
        assert strategy._resolve_job_summary(cast("Any", job), last_result, {}) is None


class TestSummaryDtypeBreakdownFallback:
    def test_select_dtypes_failure_returns_none(self):
        class _BrokenSelectDtypes:
            def select_dtypes(self, include=None):
                raise RuntimeError("select_dtypes broken")

        assert _dtype_breakdown(cast("Any", _BrokenSelectDtypes())) is None
