"""Round-4 patch-coverage tests for backend defensive branches (Codecov follow-up).

Forces the previously-unexecuted except/fallback paths flagged by the fourth
Codecov patch report: the job-retry celery bookkeeping fallback, the cached
dataset-profile fallback, the throttled log-callback DB failure branch, the
execute_pipeline exception boundary, label-decode failure, S3 list fallback,
and the non-HTTP dataset-resolution fallback.
"""

from datetime import datetime, timedelta
from types import SimpleNamespace
from typing import Any, cast
from unittest.mock import AsyncMock, MagicMock

from backend.ml_pipeline._internal._routers import meta as meta_mod
from backend.ml_pipeline._internal._routers import run_pipeline as rp_mod
from backend.ml_pipeline._services import pipeline_execution_service as pes_mod
from backend.ml_pipeline._services.prediction_utils import decode_int_like
from backend.ml_pipeline.artifacts.s3 import S3ArtifactStore
from backend.ml_pipeline.constants import StepType
from backend.ml_pipeline.resolution import resolve_pipeline_nodes


def _raiser(exc: Exception):
    def fn(*args, **kwargs):
        raise exc

    return fn


class TestResubmitJobFromGraph:
    async def test_attach_celery_task_id_failure_does_not_block_resubmit(self, monkeypatch):
        monkeypatch.setattr(
            rp_mod, "_submit_or_dedupe_branch_job", AsyncMock(return_value=("job-9", False))
        )
        monkeypatch.setattr(rp_mod, "publish_job_event", lambda event: None)
        monkeypatch.setattr(rp_mod, "get_settings", lambda: SimpleNamespace(USE_CELERY=True))
        monkeypatch.setattr(
            rp_mod.run_pipeline_batch_task, "delay", lambda payloads: SimpleNamespace(id="celery-9")
        )
        monkeypatch.setattr(
            rp_mod.JobManager,
            "attach_celery_task_id",
            AsyncMock(side_effect=RuntimeError("db down")),
        )

        job = SimpleNamespace(
            graph={
                "pipeline_id": "p1",
                "nodes": [{"node_id": "n1", "step_type": "training", "params": {}, "inputs": []}],
                "metadata": {},
            },
            pipeline_id="p1",
            dataset_id="d1",
            branch_index=0,
            node_id="n1",
            job_type="training",
            model_type="RandomForest",
        )
        new_job_id = await rp_mod.resubmit_job_from_graph(
            MagicMock(), cast("Any", job), MagicMock()
        )
        assert new_job_id == "job-9"

    async def test_resubmit_without_graph_raises(self):
        job = SimpleNamespace(graph=None)
        try:
            await rp_mod.resubmit_job_from_graph(MagicMock(), cast("Any", job), MagicMock())
            raise AssertionError("expected ValueError")
        except ValueError as exc:
            assert "no stored pipeline graph" in str(exc)


class TestDatasetSchemaCachedProfileFallback:
    async def test_bad_cached_profile_falls_back_to_sample(self, monkeypatch):
        service = MagicMock()
        service.get_source = AsyncMock(
            return_value=SimpleNamespace(source_metadata={"profile": "garbage"})
        )
        monkeypatch.setattr(meta_mod, "DataIngestionService", lambda session: service)
        monkeypatch.setattr(
            meta_mod, "_profile_from_cached_metadata", _raiser(ValueError("bad profile"))
        )
        monkeypatch.setattr(
            meta_mod, "_profile_from_sample", lambda ds, dataset_id: {"fallback": True}
        )

        result = await meta_mod.get_dataset_schema(7, MagicMock())
        assert result == {"fallback": True}


class _FastClock:
    """Stand-in datetime whose now() advances 5s per call to trip the 2s throttle."""

    _t: datetime | None = None

    @classmethod
    def now(cls, tz=None):
        if cls._t is None:
            cls._t = datetime(2026, 1, 1)
        else:
            cls._t += timedelta(seconds=5)
        return cls._t


class TestLogCallbackDbFailure:
    def test_db_write_failure_swallowed_by_log_callback(self, monkeypatch):
        _FastClock._t = None
        monkeypatch.setattr(pes_mod, "datetime", _FastClock)
        job_logs: list[str] = []
        cb = pes_mod._make_log_callback(MagicMock(), None, "job-1", job_logs)

        cb("hello")  # job is None -> internal RuntimeError caught by the except branch

        assert len(job_logs) == 1
        assert "hello" in job_logs[0]


class TestExecutePipelineBoundary:
    def test_exception_is_routed_to_failure_handler(self, monkeypatch):
        monkeypatch.setattr(pes_mod.JobStrategyFactory, "find_job", _raiser(RuntimeError("boom")))
        captured: list[tuple[str, Exception]] = []
        monkeypatch.setattr(
            pes_mod,
            "_handle_execution_exception",
            lambda session, job_id, exc: captured.append((job_id, exc)),
        )

        pes_mod.execute_pipeline("job-x", {"nodes": []}, MagicMock())

        assert len(captured) == 1
        job_id, exc = captured[0]
        assert job_id == "job-x"
        assert str(exc) == "boom"


class TestDecodeIntLikeFallback:
    def test_decode_failure_returns_original_values(self):
        encoder = MagicMock()
        encoder.inverse_transform = _raiser(RuntimeError("bad indices"))
        assert decode_int_like([0, 1], encoder) == [0, 1]


class TestS3ListArtifactsFallback:
    def test_fs_errors_degrade_to_empty_list(self):
        store = object.__new__(S3ArtifactStore)
        store.bucket_name = "bucket"
        store.prefix = ""
        store.fs = MagicMock()
        store.fs.exists = _raiser(RuntimeError("s3 down"))
        assert store.list_artifacts() == []


class TestResolutionNonHttpFallback:
    async def test_non_http_errors_are_logged_and_skipped(self, monkeypatch):
        from backend.ml_pipeline import resolution as resolution_mod

        monkeypatch.setattr(
            resolution_mod,
            "_resolve_and_apply_dataset_path",
            AsyncMock(side_effect=RuntimeError("db down")),
        )
        nodes = [{"params": {"dataset_id": "d1"}, "step_type": StepType.DATA_LOADER, "inputs": []}]
        assert await resolve_pipeline_nodes(nodes, MagicMock()) == {}
