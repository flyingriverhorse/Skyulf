"""Distinct S3 object identities must never reuse each other's cached rows."""

from datetime import UTC, datetime
from pathlib import Path
from unittest.mock import MagicMock, patch

import pandas as pd
import polars as pl
import pytest

from backend.config import get_settings
from backend.data.catalog import S3Catalog


@pytest.mark.parametrize("engine", ["pandas", "polars"])
@pytest.mark.parametrize("extension", ["csv", "parquet"])
@pytest.mark.parametrize("keys", [("data/train", "data_train"), ("a/b_c", "a_b/c")])
def test_distinct_keys_keep_distinct_cached_rows(tmp_path, monkeypatch, engine, extension, keys):
    """Fresh cache hits must retain each remote object's own data on both engines."""
    monkeypatch.setattr(get_settings(), "SKYULF_ENGINE", engine)
    with patch.dict("sys.modules", {"s3fs": MagicMock()}):
        catalog = S3Catalog("bucket", cache_dir=str(tmp_path))
    catalog.fs.info.return_value = {"LastModified": datetime(2000, 1, 1, tzinfo=UTC)}
    paths = [f"s3://bucket/{key}.{extension}" for key in keys]
    frames = {path: pd.DataFrame({"value": [index]}) for index, path in enumerate(paths)}
    if engine == "polars":
        frames = {path: pl.from_pandas(frame) for path, frame in frames.items()}
    source = MagicMock(side_effect=lambda path, limit, options: frames[path])
    monkeypatch.setattr(catalog, "_read_from_source", source)

    for _ in range(2):
        for index, path in enumerate(paths):
            result = catalog.load(path)
            assert result["value"].to_list() == [index]
    assert source.call_count == 2
    assert catalog._get_cache_path(paths[0]) != catalog._get_cache_path(paths[1])


@pytest.mark.parametrize("engine", ["pandas", "polars"])
def test_ambiguous_legacy_cache_is_not_reused(tmp_path, monkeypatch, engine):
    """An old flattened cache cannot establish which remote object its rows belong to."""
    monkeypatch.setattr(get_settings(), "SKYULF_ENGINE", engine)
    pd.DataFrame({"value": [-1]}).to_csv(tmp_path / "bucket_data_train.csv", index=False)
    with patch.dict("sys.modules", {"s3fs": MagicMock()}):
        catalog = S3Catalog("bucket", cache_dir=str(tmp_path))
    catalog.fs.info.return_value = {"LastModified": datetime(2000, 1, 1, tzinfo=UTC)}
    frame = pd.DataFrame({"value": [42]})
    source = MagicMock(return_value=pl.from_pandas(frame) if engine == "polars" else frame)
    monkeypatch.setattr(catalog, "_read_from_source", source)

    result = catalog.load("data/train.csv")

    source.assert_called_once()
    assert result["value"].to_list() == [42]


def test_cache_identity_preserves_bucket_case_and_key_characters(tmp_path):
    """Case-insensitive local filesystems must still distinguish case-sensitive S3 keys."""
    with patch.dict("sys.modules", {"s3fs": MagicMock()}):
        catalog = S3Catalog("bucket", cache_dir=str(tmp_path))
    uris = [
        "s3://bucket/train.csv",
        "s3://another-bucket/train.csv",
        "s3://bucket/Train.csv",
        "s3://bucket/data/train.csv",
        "s3://bucket/data_train.csv",
        "s3://bucket/data\\train.csv",
        "s3://bucket/veri/ölçüm.csv",
    ]
    paths = [Path(catalog._get_cache_path(uri)) for uri in uris]
    assert all(path.parent == tmp_path.resolve() for path in paths)
    assert catalog._get_cache_path(uris[0]) == str(paths[0])
    assert len({path.name.casefold() for path in paths}) == len(uris)
