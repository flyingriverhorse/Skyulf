from datetime import UTC
from unittest.mock import MagicMock, patch

import pandas as pd
import pytest

from backend.data.catalog import FileSystemCatalog, S3Catalog, SmartCatalog

# Mock data
MOCK_DF = pd.DataFrame({"a": [1, 2, 3], "b": [4, 5, 6]})


def _has_pyarrow() -> bool:
    try:
        import pyarrow  # noqa: F401

        return True
    except ImportError:
        return False


class TestFileSystemCatalog:
    @pytest.fixture(autouse=True)
    def _pandas_engine(self, monkeypatch):
        # These round-trip tests pin the legacy pandas path; the polars path
        # is covered by tests/integration/test_catalog_polars_ingestion.py.
        from backend.config import get_settings

        monkeypatch.setattr(get_settings(), "SKYULF_ENGINE", "pandas", raising=False)

    def test_load_save_csv(self, tmp_path):
        catalog = FileSystemCatalog(base_path=str(tmp_path))
        dataset_id = "test_data.csv"

        # Save
        catalog.save(dataset_id, MOCK_DF)
        assert (tmp_path / "test_data.csv").exists()

        # Load
        loaded_df = catalog.load(dataset_id)
        pd.testing.assert_frame_equal(MOCK_DF, loaded_df)

    @pytest.mark.skipif(not _has_pyarrow(), reason="pyarrow not installed")
    def test_load_save_parquet(self, tmp_path):
        catalog = FileSystemCatalog(base_path=str(tmp_path))
        dataset_id = "test_data.parquet"

        # Save
        catalog.save(dataset_id, MOCK_DF)
        assert (tmp_path / "test_data.parquet").exists()

        # Load
        loaded_df = catalog.load(dataset_id)
        pd.testing.assert_frame_equal(MOCK_DF, loaded_df)

    def test_security_check(self, tmp_path):
        catalog = FileSystemCatalog(base_path=str(tmp_path))
        # Attempt directory traversal
        dataset_id = "../secret.txt"

        # Should resolve to base_path/secret.txt (basename check), NOT outside
        path = catalog._get_path(dataset_id)
        assert path == str(tmp_path / "secret.txt")

    @pytest.mark.skipif(not _has_pyarrow(), reason="pyarrow not installed")
    def test_unknown_extension_falls_back_to_parquet_with_pandas(self, tmp_path):
        """Under the pandas engine, an unrecognized extension is read as
        parquet via pandas (the else side of the engine check)."""
        catalog = FileSystemCatalog(base_path=str(tmp_path))
        MOCK_DF.to_parquet(tmp_path / "data.bin")

        loaded = catalog.load("data.bin")

        assert isinstance(loaded, pd.DataFrame)
        pd.testing.assert_frame_equal(MOCK_DF, loaded)


class TestSmartCatalog:
    def test_resolve_numeric_id(self):
        # Mock Session and DataSource
        session = MagicMock()
        mock_ds = MagicMock()
        mock_ds.to_dict.return_value = {"config": {"file_path": "uploads/data/resolved.csv"}}
        session.query.return_value.filter.return_value.first.return_value = mock_ds

        # Mock Catalogs
        fs_catalog = MagicMock()
        fs_catalog.load.return_value = MOCK_DF

        # Create SmartCatalog
        catalog = SmartCatalog(session=session, fs_catalog=fs_catalog)

        # Patch extract_file_path_from_source where it is defined
        with patch("backend.utils.file_utils.extract_file_path_from_source") as mock_extract:
            mock_extract.return_value = "uploads/data/resolved.csv"

            # Load with numeric ID
            catalog.load("28")

            # Verify fs_catalog called with RESOLVED path
            fs_catalog.load.assert_called_with("uploads/data/resolved.csv")

    def test_resolve_numeric_id_with_credentials(self):
        # Mock Session and DataSource with credentials
        session = MagicMock()
        mock_ds = MagicMock()
        mock_ds.to_dict.return_value = {
            "config": {
                "file_path": "s3://bucket/data.csv",
                "storage_options": {"key": "AKIA", "secret": "SECRET"},
            }
        }
        # Also need to mock config access on the object itself for _resolve_id
        mock_ds.config = {
            "file_path": "s3://bucket/data.csv",
            "storage_options": {"key": "AKIA", "secret": "SECRET"},
        }

        session.query.return_value.filter.return_value.first.return_value = mock_ds

        # Mock Catalogs
        s3_catalog = MagicMock()
        s3_catalog.load.return_value = MOCK_DF

        # Create SmartCatalog
        catalog = SmartCatalog(session=session, s3_catalog=s3_catalog)

        with patch("backend.utils.file_utils.extract_file_path_from_source") as mock_extract:
            mock_extract.return_value = "s3://bucket/data.csv"

            # Load with numeric ID
            catalog.load("99")

            # Verify s3_catalog called with RESOLVED path AND credentials
            s3_catalog.load.assert_called_with(
                "s3://bucket/data.csv", storage_options={"key": "AKIA", "secret": "SECRET"}
            )

    def test_pass_through_string_id(self):
        session = MagicMock()
        fs_catalog = MagicMock()
        catalog = SmartCatalog(session=session, fs_catalog=fs_catalog)

        catalog.load("some_file.csv")
        fs_catalog.load.assert_called_with("some_file.csv")

    def test_dispatch_to_s3(self):
        session = MagicMock()
        fs_catalog = MagicMock()
        s3_catalog = MagicMock()
        s3_catalog.load.return_value = MOCK_DF

        catalog = SmartCatalog(session=session, fs_catalog=fs_catalog, s3_catalog=s3_catalog)

        # Load S3 path
        catalog.load("s3://my-bucket/data.csv")

        # Verify s3_catalog called
        s3_catalog.load.assert_called_with("s3://my-bucket/data.csv")
        # Verify fs_catalog NOT called
        fs_catalog.load.assert_not_called()


class TestS3Catalog:
    def test_s3_paths(self):
        # We mock s3fs import check
        with patch.dict("sys.modules", {"s3fs": MagicMock()}):
            catalog = S3Catalog(bucket_name="my-bucket")

            assert catalog._get_s3_path("data.csv") == "s3://my-bucket/data.csv"
            assert (
                catalog._get_s3_path("s3://other-bucket/data.csv") == "s3://other-bucket/data.csv"
            )

    @patch("pandas.read_csv")
    def test_load_s3(self, mock_read_csv):
        with (
            patch.dict("sys.modules", {"s3fs": MagicMock()}),
            patch("backend.data.catalog.get_settings") as mock_settings,
        ):
            mock_settings.return_value = MagicMock(SKYULF_ENGINE="pandas", AWS_ENDPOINT_URL=None)
            catalog = S3Catalog(bucket_name="my-bucket")
            catalog.load("data.csv")

            mock_read_csv.assert_called_with(
                "s3://my-bucket/data.csv", nrows=None, storage_options={}
            )

    @patch("backend.data.catalog.pl.from_pandas")
    @patch("pandas.read_csv")
    def test_load_s3_converts_to_polars_when_engine_polars(self, mock_read_csv, mock_from_pandas):
        """With SKYULF_ENGINE=polars, the pandas source read is converted before return."""
        with (
            patch.dict("sys.modules", {"s3fs": MagicMock()}),
            patch("backend.data.catalog.get_settings") as mock_settings,
        ):
            mock_settings.return_value = MagicMock(SKYULF_ENGINE="polars", AWS_ENDPOINT_URL=None)
            catalog = S3Catalog(bucket_name="my-bucket")
            catalog.load("data.csv")

            mock_read_csv.assert_called_once()
            mock_from_pandas.assert_called_once_with(mock_read_csv.return_value)

    def test_caller_supplied_endpoint_url_is_dropped(self):
        """SSRF fix: a caller-supplied endpoint_url must never reach s3fs unless the
        server operator has configured AWS_ENDPOINT_URL themselves."""
        with (
            patch.dict("sys.modules", {"s3fs": MagicMock()}),
            patch("backend.data.catalog.get_settings") as mock_get_settings,
        ):
            mock_get_settings.return_value = MagicMock(AWS_ENDPOINT_URL=None)
            catalog = S3Catalog(
                bucket_name="my-bucket",
                storage_options={"endpoint_url": "http://169.254.169.254/latest/meta-data/"},
            )

            prepared = catalog._prepare_s3fs_options(catalog.storage_options)

            assert "client_kwargs" not in prepared or "endpoint_url" not in prepared.get(
                "client_kwargs", {}
            )

    def test_server_configured_endpoint_url_is_used_instead(self):
        """When AWS_ENDPOINT_URL is configured server-side, it wins over whatever the
        caller supplied — the caller's value is discarded, not merged."""
        with (
            patch.dict("sys.modules", {"s3fs": MagicMock()}),
            patch("backend.data.catalog.get_settings") as mock_get_settings,
        ):
            mock_get_settings.return_value = MagicMock(
                AWS_ENDPOINT_URL="https://trusted-minio.internal:9000"
            )
            catalog = S3Catalog(
                bucket_name="my-bucket",
                storage_options={"endpoint_url": "http://169.254.169.254/latest/meta-data/"},
            )

            prepared = catalog._prepare_s3fs_options(catalog.storage_options)

            assert (
                prepared["client_kwargs"]["endpoint_url"] == "https://trusted-minio.internal:9000"
            )


class TestS3CatalogPolarsPaths:
    """S3Catalog reads/writes must honor SKYULF_ENGINE=polars end to end."""

    def _catalog(self) -> "S3Catalog":
        with patch.dict("sys.modules", {"s3fs": MagicMock()}):
            return S3Catalog(bucket_name="my-bucket")

    def test_fresh_csv_cache_is_read_with_polars(self, tmp_path, monkeypatch):
        """A fresh local CSV cache is returned as a Polars frame under the
        polars engine, not a pandas frame."""
        pl = pytest.importorskip("polars")
        from datetime import datetime, timedelta

        from backend.config import get_settings

        monkeypatch.setattr(get_settings(), "SKYULF_ENGINE", "polars", raising=False)
        catalog = self._catalog()
        cache_path = tmp_path / "data.csv"
        pd.DataFrame({"a": [1, 2]}).to_csv(cache_path, index=False)
        catalog.fs.info.return_value = {"LastModified": datetime.now(UTC) - timedelta(days=1)}

        out = catalog._load_from_cache_if_fresh(
            "s3://my-bucket/data.csv", str(cache_path), None, {}
        )

        assert isinstance(out, pl.DataFrame)
        assert out["a"].to_list() == [1, 2]

    @pytest.mark.skipif(not _has_pyarrow(), reason="pyarrow not installed")
    def test_fresh_parquet_cache_is_read_with_polars(self, tmp_path, monkeypatch):
        """Same for parquet caches, with the row limit applied after the read."""
        pl = pytest.importorskip("polars")
        from datetime import datetime, timedelta

        from backend.config import get_settings

        monkeypatch.setattr(get_settings(), "SKYULF_ENGINE", "polars", raising=False)
        catalog = self._catalog()
        cache_path = tmp_path / "data.parquet"
        pd.DataFrame({"a": [1, 2, 3]}).to_parquet(cache_path)
        catalog.fs.info.return_value = {"LastModified": datetime.now(UTC) - timedelta(days=1)}

        out = catalog._load_from_cache_if_fresh(
            "s3://my-bucket/data.parquet", str(cache_path), 2, {}
        )

        assert isinstance(out, pl.DataFrame)
        assert out["a"].to_list() == [1, 2]

    @patch("pandas.read_parquet")
    def test_read_from_source_dispatches_parquet(self, mock_read_parquet):
        """The S3 source-read dispatcher must route `.parquet` paths to the
        parquet reader (not the fallback), regardless of engine."""
        with patch("backend.data.catalog.get_settings") as mock_settings:
            mock_settings.return_value = MagicMock(SKYULF_ENGINE="pandas")
            mock_read_parquet.return_value = pd.DataFrame({"a": [1]})

            out = S3Catalog._read_from_source("s3://my-bucket/data.parquet", None, {})

        mock_read_parquet.assert_called_once_with("s3://my-bucket/data.parquet", storage_options={})
        assert isinstance(out, pd.DataFrame)

    @patch("pandas.read_json")
    def test_read_from_source_dispatches_json(self, mock_read_json):
        """`.json` paths route to the JSON reader — the else sides of the
        csv/parquet checks."""
        with patch("backend.data.catalog.get_settings") as mock_settings:
            mock_settings.return_value = MagicMock(SKYULF_ENGINE="pandas")
            mock_read_json.return_value = pd.DataFrame({"a": [1]})

            out = S3Catalog._read_from_source("s3://my-bucket/data.json", None, {})

        mock_read_json.assert_called_once_with("s3://my-bucket/data.json", storage_options={})
        assert isinstance(out, pd.DataFrame)

    def test_write_to_cache_accepts_polars_frames(self, tmp_path):
        """A Polars frame must be cached via its native writers — the pandas
        `to_csv`/`to_parquet` API does not exist on pl.DataFrame."""
        pl = pytest.importorskip("polars")
        df = pl.DataFrame({"a": [1, 2]})

        csv_path = tmp_path / "cache.csv"
        S3Catalog._write_to_cache(df, str(csv_path), "s3://my-bucket/data.csv")
        assert pl.read_csv(csv_path)["a"].to_list() == [1, 2]

        parquet_path = tmp_path / "cache.parquet"
        S3Catalog._write_to_cache(df, str(parquet_path), "s3://my-bucket/data.parquet")
        assert pl.read_parquet(parquet_path)["a"].to_list() == [1, 2]

    def test_fresh_csv_cache_is_read_with_pandas(self, tmp_path, monkeypatch):
        """The else side of the cache engine check: under the pandas engine a
        fresh CSV cache stays a pandas frame."""
        from datetime import datetime, timedelta

        from backend.config import get_settings

        monkeypatch.setattr(get_settings(), "SKYULF_ENGINE", "pandas", raising=False)
        catalog = self._catalog()
        cache_path = tmp_path / "data.csv"
        pd.DataFrame({"a": [1, 2]}).to_csv(cache_path, index=False)
        catalog.fs.info.return_value = {"LastModified": datetime.now(UTC) - timedelta(days=1)}

        out = catalog._load_from_cache_if_fresh(
            "s3://my-bucket/data.csv", str(cache_path), None, {}
        )

        assert isinstance(out, pd.DataFrame)
        assert out["a"].tolist() == [1, 2]

    @pytest.mark.skipif(not _has_pyarrow(), reason="pyarrow not installed")
    def test_fresh_parquet_cache_is_read_with_pandas(self, tmp_path, monkeypatch):
        """Same else side for parquet caches."""
        from datetime import datetime, timedelta

        from backend.config import get_settings

        monkeypatch.setattr(get_settings(), "SKYULF_ENGINE", "pandas", raising=False)
        catalog = self._catalog()
        cache_path = tmp_path / "data.parquet"
        pd.DataFrame({"a": [1, 2, 3]}).to_parquet(cache_path)
        catalog.fs.info.return_value = {"LastModified": datetime.now(UTC) - timedelta(days=1)}

        out = catalog._load_from_cache_if_fresh(
            "s3://my-bucket/data.parquet", str(cache_path), None, {}
        )

        assert isinstance(out, pd.DataFrame)
        assert out["a"].tolist() == [1, 2, 3]

    def test_write_to_cache_accepts_pandas_frames(self, tmp_path):
        """The non-Polars branch of the cache writer keeps using the pandas API."""
        df = pd.DataFrame({"a": [1, 2]})

        csv_path = tmp_path / "cache.csv"
        S3Catalog._write_to_cache(df, str(csv_path), "s3://my-bucket/data.csv")
        assert pd.read_csv(csv_path)["a"].tolist() == [1, 2]

        parquet_path = tmp_path / "cache.parquet"
        S3Catalog._write_to_cache(df, str(parquet_path), "s3://my-bucket/data.parquet")
        assert pd.read_parquet(parquet_path)["a"].tolist() == [1, 2]
