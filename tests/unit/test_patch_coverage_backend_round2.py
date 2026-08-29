"""Round-2 patch-coverage tests for backend defensive branches (Codecov follow-up).

Forces the previously-unexecuted except/fallback paths in the S3 connector
probes, pipeline-error persistence, artifact extraction, warning capture,
evaluation label decoding, drift-job enrichment, and the realtime WS handler.
"""

import importlib
import logging
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import AsyncMock, MagicMock

import polars as pl

from backend.data_ingestion.connectors import s3 as s3_mod
from backend.data_ingestion.connectors.s3 import S3Connector
from backend.exceptions.handlers import record_pipeline_error
from backend.ml_pipeline._execution.engine._artifacts import ArtifactsMixin
from backend.ml_pipeline._execution.engine._warning_capture import WarningCaptureHandler
from backend.ml_pipeline._services.evaluation_service import EvaluationService
from backend.monitoring import router as monitoring_router

# backend.realtime.__init__ shadows the `router` submodule with the APIRouter
# object, so pull the real module out of sys.modules via importlib.
realtime_router = importlib.import_module("backend.realtime.router")


def _raiser(exc: Exception):
    def fn(*args, **kwargs):
        raise exc

    return fn


class _BrokenStr:
    def __str__(self):
        raise RuntimeError("cannot stringify")


class TestS3ConnectorProbeFallbacks:
    def test_try_csv_schema_returns_none_on_scan_failure(self, monkeypatch):
        monkeypatch.setattr(s3_mod.pl, "scan_csv", _raiser(RuntimeError("not csv")))
        conn = S3Connector("s3://bucket/data.csv")
        assert conn._try_csv_schema({}) is None

    async def test_get_schema_falls_back_to_csv_when_parquet_probe_fails(self, monkeypatch):
        monkeypatch.setattr(s3_mod.pl, "scan_parquet", _raiser(RuntimeError("not parquet")))
        lf = MagicMock()
        lf.collect_schema.return_value = {"a": "Int64"}
        monkeypatch.setattr(s3_mod.pl, "scan_csv", lambda path, storage_options=None: lf)

        conn = S3Connector("s3://bucket/data.bin")
        assert await conn.get_schema() == {"a": "Int64"}

    async def test_fetch_data_format_probe_falls_back_to_csv(self, monkeypatch):
        # scan_parquet returns a lazy frame whose schema probe then fails —
        # the fallback to CSV must kick in.
        broken_lf = MagicMock()
        broken_lf.collect_schema.side_effect = RuntimeError("not parquet")
        monkeypatch.setattr(s3_mod.pl, "scan_parquet", lambda path, storage_options=None: broken_lf)
        lf = MagicMock()
        lf.collect.return_value = pl.DataFrame({"a": [1, 2]})
        monkeypatch.setattr(s3_mod.pl, "scan_csv", lambda path, storage_options=None: lf)

        conn = S3Connector("s3://bucket/data.unknown")
        df = await conn.fetch_data()
        assert df.height == 2


class TestRecordPipelineErrorBestEffort:
    def test_engine_failure_is_swallowed(self, monkeypatch):
        monkeypatch.setattr("sqlalchemy.create_engine", _raiser(RuntimeError("no db")))
        # Must not raise: error persistence is best-effort.
        record_pipeline_error("job-1", "pipeline blew up", "traceback text")


class TestArtifactsMixinFailureBranches:
    def test_extract_feature_importances_returns_none_on_failure(self):
        mixin = object.__new__(ArtifactsMixin)
        mixin._feature_names_for_importance = _raiser(RuntimeError("names broken"))
        assert mixin._extract_feature_importances(MagicMock(), object(), "target") is None

    def test_save_reference_data_swallows_persistence_errors(self):
        mixin = object.__new__(ArtifactsMixin)
        mixin._normalize_train_frame = _raiser(RuntimeError("normalize broken"))
        # Must not raise: reference-data save is best-effort.
        mixin._save_reference_data(object(), "job-1", "target")

    def test_save_reference_data_skips_unknown_job(self):
        mixin = object.__new__(ArtifactsMixin)
        mixin._normalize_train_frame = _raiser(RuntimeError("must not be called"))
        mixin._save_reference_data(object(), "unknown", "target")


class TestWarningCaptureHandlerDefenses:
    def test_emit_swallows_unformattable_records(self, capsys):
        handler = WarningCaptureHandler()
        record = logging.LogRecord(
            "skyulf.test", logging.WARNING, str(Path(__file__)), 1, _BrokenStr(), None, None
        )
        handler.emit(record)  # handleError path: logs the error, never raises
        assert handler.drain() == []
        capsys.readouterr()  # discard handleError's stderr traceback

    def test_detach_swallows_remove_handler_failures(self):
        handler = WarningCaptureHandler()
        handler.attach()
        assert handler._attached
        handler._attached[0].removeHandler = _raiser(RuntimeError("remove broken"))
        handler.detach()  # must not raise
        assert handler._attached == []


class TestEvaluationDecodeBestEffort:
    def test_decode_target_labels_swallows_loader_errors(self, monkeypatch):
        monkeypatch.setattr(
            EvaluationService,
            "_load_target_label_encoder",
            staticmethod(_raiser(RuntimeError("bundle broken"))),
        )
        data = {"problem_type": "classification", "splits": {"test": {}}}
        EvaluationService._decode_target_labels(data, MagicMock(), "job-1")
        assert data["problem_type"] == "classification"

    def test_decode_reference_column_swallows_loader_errors(self, monkeypatch):
        monkeypatch.setattr(
            EvaluationService,
            "_load_feature_engineer_bundle",
            staticmethod(_raiser(RuntimeError("bundle broken"))),
        )
        data = {
            "problem_type": "clustering",
            "splits": {"train": {"clustering": {"reference_column": "species"}}},
        }
        EvaluationService._decode_reference_column(data, MagicMock(), "job-1")
        assert data["problem_type"] == "clustering"


class TestMonitoringEnrichmentFallbacks:
    async def test_fetch_drift_job_rows_returns_empty_on_db_error(self):
        db = MagicMock()
        db.execute = AsyncMock(side_effect=RuntimeError("db down"))
        assert await monitoring_router._fetch_drift_job_rows(db, ["job-1"]) == {}

    def test_extract_drift_target_column_returns_none_on_bad_graph(self, monkeypatch):
        monkeypatch.setattr(
            monitoring_router, "extract_job_details", _raiser(RuntimeError("bad graph"))
        )
        row = SimpleNamespace(graph={"nodes": []}, node_id="n1")
        assert monitoring_router._extract_drift_target_column(row) is None


class TestRunPipelineDispatchBookkeeping:
    async def test_dispatch_swallows_attach_celery_task_id_failures(self, monkeypatch):
        from backend.ml_pipeline._internal._routers import run_pipeline as rp_mod

        monkeypatch.setattr(
            rp_mod.JobManager,
            "attach_celery_task_id",
            AsyncMock(side_effect=RuntimeError("db down")),
        )
        celery_task = MagicMock()
        celery_task.id = "celery-123"
        monkeypatch.setattr(rp_mod.run_pipeline_batch_task, "delay", lambda payloads: celery_task)

        settings = SimpleNamespace(USE_CELERY=True)
        # Must not raise: task-id bookkeeping failure must not block dispatch.
        await rp_mod._dispatch_branch_tasks(
            [("job-1", {}), ("job-2", {})], settings, MagicMock(), MagicMock()
        )


class TestRealtimeWsJobsFailurePaths:
    async def test_failed_accept_exits_without_disconnect(self, monkeypatch):
        connect = AsyncMock(side_effect=RuntimeError("accept failed"))
        disconnect = AsyncMock()
        monkeypatch.setattr(realtime_router.connection_manager, "connect", connect)
        monkeypatch.setattr(realtime_router.connection_manager, "disconnect", disconnect)

        await realtime_router.ws_jobs(MagicMock())
        connect.assert_awaited_once()
        disconnect.assert_not_awaited()

    async def test_receive_error_ends_loop_and_disconnects(self, monkeypatch):
        monkeypatch.setattr(realtime_router.connection_manager, "connect", AsyncMock())
        disconnect = AsyncMock()
        monkeypatch.setattr(realtime_router.connection_manager, "disconnect", disconnect)
        ws = MagicMock()
        ws.receive_text = AsyncMock(side_effect=RuntimeError("receive broke"))

        await realtime_router.ws_jobs(ws)
        disconnect.assert_awaited_once()
