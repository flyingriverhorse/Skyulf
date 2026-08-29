"""Coverage tests for the defensive except/fallback branches touched by the
BLE001 noqa triage (Codecov patch-coverage follow-up).

Each test forces one previously-unexecuted branch: serialization capability
probes, file-cleanup failure isolation, catalog resolution fallbacks, startup
teardown tolerance, lazy-scan probes, and health/readiness degradation.
"""

from pathlib import Path
from types import SimpleNamespace
from typing import Any
from unittest.mock import MagicMock

import numpy as np
import pandas as pd
import polars as pl
import pytest

from backend.data.catalog import FileSystemCatalog, SmartCatalog
from backend.data_ingestion.serialization import DataTypeConverter, JSONSafeSerializer
from backend.services.data_service import DataService
from backend.utils import file_utils
from backend.utils.file_utils import (
    _remove_excess_files,
    _remove_files_older_than,
    cleanup_empty_directories,
    cleanup_old_files,
    safe_delete_path,
)


def _raiser(exc: Exception):
    def fn(*args, **kwargs):
        raise exc

    return fn


class _BrokenStr:
    def __str__(self):
        raise RuntimeError("cannot stringify")


class _BrokenIsna:
    dtype = "object"

    def isna(self):
        raise RuntimeError("isna broken")


class _BrokenToDict:
    def to_dict(self, *args, **kwargs):
        raise RuntimeError("to_dict broken")


class _BrokenIsoformat:
    def isoformat(self):
        raise RuntimeError("isoformat broken")


class _BrokenFloat:
    def __float__(self):
        raise RuntimeError("float broken")


class TestSerializationProbeBranches:
    def test_sync_fallback_returns_none_when_str_raises(self):
        assert JSONSafeSerializer._handle_fallback(_BrokenStr()) is None

    async def test_async_fallback_returns_none_when_str_raises(self):
        from backend.data_ingestion.serialization import AsyncJSONSafeSerializer

        assert await AsyncJSONSafeSerializer._handle_fallback(_BrokenStr()) is None

    def test_isna_probe_failure_falls_through(self):
        result = JSONSafeSerializer._handle_pandas_object(_BrokenIsna())
        assert result is JSONSafeSerializer._NOT_HANDLED

    def test_numpy_inf_nan_guard_failure_returns_value(self, monkeypatch):
        monkeypatch.setattr(np, "isinf", _raiser(RuntimeError("numpy guard broken")))
        monkeypatch.setattr(np, "isnan", _raiser(RuntimeError("numpy guard broken")))
        assert JSONSafeSerializer._handle_float_edge_cases(1.5) == 1.5

    def test_to_dict_probe_failure_is_not_handled(self):
        assert (
            JSONSafeSerializer._handle_dataframe_like(_BrokenToDict())
            is JSONSafeSerializer._NOT_HANDLED
        )

    def test_isoformat_probe_failure_is_not_handled(self):
        assert (
            JSONSafeSerializer._handle_datetime(_BrokenIsoformat())
            is JSONSafeSerializer._NOT_HANDLED
        )

    def test_float_conversion_probe_failure_is_not_handled(self):
        assert (
            JSONSafeSerializer._handle_decimal_like(_BrokenFloat())
            is JSONSafeSerializer._NOT_HANDLED
        )

    async def test_convert_dataframe_types_isolates_per_column_failure(self):
        df = pd.DataFrame({"a": [1, 2, 3], "b": [4.0, 5.0, 6.0]})
        with pytest.MonkeyPatch.context() as mp:
            mp.setattr(
                DataTypeConverter,
                "_convert_column",
                staticmethod(_raiser(RuntimeError("convert broken"))),
            )
            result = await DataTypeConverter.convert_dataframe_types(df, {"a": "integer"})
        assert result["a"].tolist() == [1, 2, 3]


class TestFileUtilsFailureIsolation:
    def test_delete_immediately_exception_returns_false(self, tmp_path, monkeypatch):
        file_path = tmp_path / "f.txt"
        file_path.write_text("x")
        monkeypatch.setattr(Path, "unlink", _raiser(RuntimeError("locked")))
        assert file_utils._delete_immediately(file_path) is False

    def test_safe_delete_path_exception_returns_false(self, tmp_path, monkeypatch):
        file_path = tmp_path / "f.txt"
        file_path.write_text("x")
        monkeypatch.setattr(file_utils, "_delete_immediately", _raiser(RuntimeError("boom")))
        assert safe_delete_path(file_path) is False

    def test_cleanup_empty_directories_walk_failure(self, tmp_path, monkeypatch):
        monkeypatch.setattr(Path, "walk", _raiser(RuntimeError("boom")))
        assert cleanup_empty_directories(tmp_path) == 0

    def test_cleanup_empty_directories_rmdir_failure(self, tmp_path, monkeypatch):
        (tmp_path / "empty").mkdir()
        monkeypatch.setattr(Path, "rmdir", _raiser(OSError("not empty")))
        assert cleanup_empty_directories(tmp_path) == 0
        assert (tmp_path / "empty").exists()

    def test_cleanup_old_files_error_status(self, tmp_path, monkeypatch):
        monkeypatch.setattr(file_utils, "_do_cleanup_old_files", _raiser(RuntimeError("boom")))
        result = cleanup_old_files(tmp_path)
        assert result["status"] == "error"
        assert result["files_removed"] == 0

    def test_remove_files_older_than_skips_unlink_failure(self, tmp_path, monkeypatch):
        import os
        from datetime import UTC, datetime, timedelta

        old = tmp_path / "old.txt"
        old.write_text("x")
        old_ts = (datetime.now(UTC) - timedelta(days=30)).timestamp()
        os.utime(old, (old_ts, old_ts))
        monkeypatch.setattr(Path, "unlink", _raiser(PermissionError("denied")))
        count, removed = _remove_files_older_than([old], datetime.now(UTC) - timedelta(days=7))
        assert count == 0
        assert removed == []
        assert old.exists()

    def test_remove_excess_files_skips_unlink_failure(self, tmp_path, monkeypatch):
        files = []
        for i in range(3):
            p = tmp_path / f"f{i}.txt"
            p.write_text(str(i))
            files.append(p)
        monkeypatch.setattr(Path, "unlink", _raiser(PermissionError("denied")))
        count, removed = _remove_excess_files(files, max_files=1)
        assert count == 0
        assert removed == []
        assert all(p.exists() for p in files)


class TestCatalogFallbackBranches:
    def test_filesystem_catalog_unsupported_format_raises_value_error(self, tmp_path):
        catalog = FileSystemCatalog(base_path=str(tmp_path))
        # File exists but has no recognized extension and is not valid parquet,
        # so the fallback parquet probe fails and the clean ValueError follows.
        (tmp_path / "dataset.weird").write_text("not parquet bytes")
        with pytest.raises(ValueError, match="Unsupported format"):
            catalog.load("dataset.weird")

    def test_smart_catalog_resolve_id_db_error_falls_back(self):
        session = MagicMock()
        session.query.side_effect = RuntimeError("db down")
        catalog = SmartCatalog(session=session)
        assert catalog._resolve_id("123") == ("123", {})

    def test_smart_catalog_get_dataset_name_db_error_returns_none(self):
        session = MagicMock()
        session.query.side_effect = RuntimeError("db down")
        catalog = SmartCatalog(session=session)
        assert catalog.get_dataset_name("123") is None

    def test_s3_catalog_cache_validation_failure_returns_none(self, tmp_path):
        from backend.data.catalog import S3Catalog

        catalog = object.__new__(S3Catalog)
        catalog.fs = MagicMock(info=_raiser(RuntimeError("s3 down")))
        cache_path = tmp_path / "cached.csv"
        cache_path.write_text("a\n1\n")

        result = catalog._load_from_cache_if_fresh("s3://b/data.csv", str(cache_path), None, {})
        assert result is None

    def test_s3_catalog_write_to_cache_failure_is_logged_only(self, tmp_path):
        from backend.data.catalog import S3Catalog

        df = MagicMock(spec=["to_csv"])
        df.to_csv.side_effect = RuntimeError("disk full")
        S3Catalog._write_to_cache(df, str(tmp_path / "c.csv"), "s3://b/data.csv")  # must not raise

    def test_s3_catalog_save_cache_update_failure_is_logged_only(self, monkeypatch):
        from backend.data.catalog import S3Catalog

        catalog = object.__new__(S3Catalog)
        catalog.bucket_name = "b"
        catalog.storage_options = {}
        monkeypatch.setattr(S3Catalog, "_prepare_s3fs_options", staticmethod(lambda opts: {}))
        monkeypatch.setattr(
            S3Catalog, "_get_cache_path", _raiser(RuntimeError("cache dir missing"))
        )
        data = MagicMock()
        catalog.save("data.parquet", data)  # must not raise
        data.to_parquet.assert_called_once()


class TestMainStartupTolerance:
    def test_reset_stale_jobs_swallows_engine_failure(self, monkeypatch):
        import sqlalchemy

        import backend.main as main_mod

        monkeypatch.setattr(sqlalchemy, "create_engine", _raiser(RuntimeError("db unreachable")))
        main_mod._reset_stale_jobs()  # must not raise

    async def test_lifespan_survives_realtime_failure(self, monkeypatch):
        from fastapi import FastAPI

        import backend.main as main_mod

        async def noop():
            return None

        monkeypatch.setattr(main_mod, "init_db", noop)
        monkeypatch.setattr(main_mod, "create_tables", noop)
        monkeypatch.setattr(main_mod, "_reset_stale_jobs", lambda: None)
        monkeypatch.setattr(
            main_mod.connection_manager, "start", _raiser(RuntimeError("realtime down"))
        )
        monkeypatch.setattr(
            main_mod.connection_manager, "stop", _raiser(RuntimeError("realtime down"))
        )

        async with main_mod.lifespan(FastAPI()):
            pass  # startup failure must not block the app

    def test_setup_templates_and_static_swallows_mount_failure(self, monkeypatch):
        import backend.main as main_mod

        call_count = {"n": 0}

        class FlakyStaticFiles:
            """Raises on first mount (covered branch); later mounts succeed so
            the frontend-assets mount outside the try still works."""

            def __init__(self, directory=None):
                call_count["n"] += 1
                if call_count["n"] == 1:
                    raise RuntimeError("static dir missing")

        monkeypatch.setattr(main_mod, "StaticFiles", FlakyStaticFiles)
        app = MagicMock()
        main_mod._setup_templates_and_static(app)  # must not raise


class TestLocalFileConnectorLazyProbeFailures:
    @staticmethod
    def _connector(path: str):
        from backend.data_ingestion.connectors.file import LocalFileConnector

        connector = object.__new__(LocalFileConnector)
        connector.file_path = path
        connector.kwargs = {}
        connector._df = None
        connector._schema = None
        return connector

    def test_scan_probe_failure_returns_none(self, monkeypatch):
        import backend.data_ingestion.connectors.file as file_mod

        monkeypatch.setattr(file_mod.pl, "scan_csv", _raiser(RuntimeError("scan broken")))
        connector = self._connector("any.csv")
        assert connector._scan() is None

    def test_lazy_schema_probe_failure_returns_none(self, monkeypatch):
        import backend.data_ingestion.connectors.file as file_mod

        lf = MagicMock()
        lf.collect_schema.side_effect = RuntimeError("schema broken")
        monkeypatch.setattr(file_mod.pl, "scan_csv", lambda *a, **k: lf)
        connector = self._connector("any.csv")
        assert connector._try_lazy_schema() is None

    def test_lazy_head_probe_failure_returns_none(self, monkeypatch):
        import backend.data_ingestion.connectors.file as file_mod

        lf = MagicMock()
        lf.head.return_value.collect.side_effect = RuntimeError("collect broken")
        monkeypatch.setattr(file_mod.pl, "scan_csv", lambda *a, **k: lf)
        connector = self._connector("any.csv")
        assert connector._try_lazy_head(query=None, limit=5) is None


class TestHealthDegradedBranches:
    async def test_detailed_health_db_and_cache_failures_mark_unhealthy(self, monkeypatch):
        from backend.config import get_settings
        from backend.database import engine as db_engine_mod
        from backend.health.routes import detailed_health_check

        async def boom():
            raise RuntimeError("db probe failed")

        monkeypatch.setattr(db_engine_mod, "health_check", boom)
        settings = get_settings()
        monkeypatch.setattr(settings, "USE_CELERY", True, raising=False)
        monkeypatch.setattr(
            settings, "CELERY_BROKER_URL", "redis://127.0.0.1:59999/0", raising=False
        )
        monkeypatch.setattr(settings, "REDIS_HEALTHCHECK_TIMEOUT_SECONDS", 1, raising=False)

        response = await detailed_health_check(settings=settings)
        assert response.dependencies_healthy is False
        assert response.status == "degraded"

    async def test_readiness_probe_returns_503_when_fit_fails(self, monkeypatch):
        from fastapi.responses import JSONResponse
        from sklearn.preprocessing import StandardScaler

        from backend.health.routes import readiness_check

        def boom(self, X, y=None):
            raise RuntimeError("sklearn broken")

        monkeypatch.setattr(StandardScaler, "fit_transform", boom)
        response = await readiness_check()
        assert isinstance(response, JSONResponse)
        assert response.status_code == 503


class TestPipelineEngineMetadataBranches:
    def test_build_node_metadata_survives_all_load_failures(self, monkeypatch):
        from typing import cast

        from backend.ml_pipeline._execution.engine import PipelineEngine
        from backend.ml_pipeline._execution.schemas import NodeConfig

        class ExplodingStore:
            def load(self, key):
                raise RuntimeError("artifact unreadable")

        engine = object.__new__(PipelineEngine)
        engine.artifact_store = cast("Any", ExplodingStore())
        node = cast(
            "NodeConfig",
            SimpleNamespace(node_id="n1", inputs=["u1"], step_type="transformer", params={}),
        )

        monkeypatch.setattr(
            "backend.ml_pipeline._execution.engine.build_summary",
            _raiser(RuntimeError("summary broken")),
        )
        assert engine._build_node_metadata(node, {}) == {}


class TestFeatureEngMixinFailureBranches:
    @staticmethod
    def _mixin(store):
        from backend.ml_pipeline._execution.engine._feature_eng import FeatureEngMixin

        mixin = object.__new__(FeatureEngMixin)
        mixin.artifact_store = store
        mixin.executed_transformers = []
        return mixin

    class _ExplodingStore:
        def load(self, key):
            raise RuntimeError("artifact unreadable")

    def test_merge_fitted_steps_skips_unloadable_artifacts(self):
        mixin = self._mixin(self._ExplodingStore())
        assert mixin._merge_fitted_steps(["k1", "k2"]) == []

    def test_resolve_bundle_feature_engineer_load_failure_keeps_override(self):
        mixin = self._mixin(self._ExplodingStore())
        override = SimpleNamespace(transform=lambda df: df)
        assert mixin._resolve_bundle_feature_engineer(override, "artifact_key") is override
        assert mixin._resolve_bundle_feature_engineer(None, "artifact_key") is None

    def test_build_legacy_transformer_bundle_load_failure(self):
        mixin = self._mixin(self._ExplodingStore())
        mixin.executed_transformers = [
            {
                "artifact_key": "t1",
                "node_id": "n1",
                "transformer_name": "scaler",
                "column_name": "a",
                "transformer_type": "StandardScaler",
            }
        ]
        bundle = mixin._build_legacy_transformer_bundle(
            model_artifact="model", job_id="job", target_column="y", dropped_columns=None
        )
        assert bundle["transformers"] == []
        assert bundle["transformer_plan"] == []


class TestDataServiceFallbackBranches:
    def test_load_polars_with_fallback_falls_back_to_pandas(self, tmp_path, monkeypatch):
        path = tmp_path / "data.csv"
        path.write_text("a,b\n1,2\n3,4\n")
        service = DataService()
        monkeypatch.setattr(service, "_load_polars", _raiser(RuntimeError("polars unavailable")))

        result = service._load_polars_with_fallback(str(path), None)

        assert isinstance(result, pl.DataFrame)
        assert result.to_dicts() == [{"a": 1, "b": 2}, {"a": 3, "b": 4}]

    def test_try_polars_lazy_sample_scan_failure_returns_none(self, tmp_path):
        service = DataService()
        assert service._try_polars_lazy_sample(str(tmp_path / "missing.parquet"), 5) is None

    def test_save_via_polars_conversion_falls_back_to_pandas(self, tmp_path):
        service = DataService()
        data = MagicMock(spec=["to_arrow", "to_pandas"])
        data.to_arrow.side_effect = RuntimeError("arrow broken")
        data.to_pandas.return_value = pd.DataFrame({"a": [1, 2]})
        target = tmp_path / "out.parquet"

        service._save_via_polars_conversion(data, str(target))

        assert target.exists()
        assert pl.read_parquet(target).to_dicts() == [{"a": 1}, {"a": 2}]
