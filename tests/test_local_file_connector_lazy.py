"""Tests for `LocalFileConnector` lazy schema / sample fast paths (A6)."""

from __future__ import annotations

from pathlib import Path

import pandas as pd
import polars as pl
import pytest

from backend.data_ingestion.connectors.file import LocalFileConnector


def _write_csv(tmp_path: Path, rows: int = 100) -> Path:
    path = tmp_path / "data.csv"
    df = pd.DataFrame({"a": range(rows), "b": [f"row-{i}" for i in range(rows)]})
    df.to_csv(path, index=False)
    return path


def _write_parquet(tmp_path: Path, rows: int = 100) -> Path:
    path = tmp_path / "data.parquet"
    df = pd.DataFrame({"x": range(rows), "y": [float(i) for i in range(rows)]})
    pl.from_pandas(df).write_parquet(path)
    return path


@pytest.mark.asyncio
async def test_get_schema_csv_lazy_does_not_materialise_df(tmp_path: Path) -> None:
    path = _write_csv(tmp_path, rows=1000)
    conn = LocalFileConnector(str(path))
    await conn.connect()

    schema = await conn.get_schema()

    assert set(schema.keys()) == {"a", "b"}
    # Lazy path must avoid eagerly caching the full frame.
    assert conn._df is None
    # Calling again should hit the cached schema (no exception, same result).
    assert await conn.get_schema() == schema


@pytest.mark.asyncio
async def test_get_schema_parquet_lazy_does_not_materialise_df(tmp_path: Path) -> None:
    path = _write_parquet(tmp_path, rows=500)
    conn = LocalFileConnector(str(path))
    await conn.connect()

    schema = await conn.get_schema()

    assert set(schema.keys()) == {"x", "y"}
    assert conn._df is None


@pytest.mark.asyncio
async def test_fetch_data_with_limit_csv_lazy_does_not_materialise(tmp_path: Path) -> None:
    path = _write_csv(tmp_path, rows=10_000)
    conn = LocalFileConnector(str(path))
    await conn.connect()

    head = await conn.fetch_data(limit=5)

    assert isinstance(head, pl.DataFrame)
    assert head.height == 5
    assert list(head.columns) == ["a", "b"]
    # Lazy fast path must not cache the full dataframe.
    assert conn._df is None


@pytest.mark.asyncio
async def test_fetch_data_no_limit_falls_back_to_eager(tmp_path: Path) -> None:
    path = _write_csv(tmp_path, rows=50)
    conn = LocalFileConnector(str(path))
    await conn.connect()

    df = await conn.fetch_data()

    assert df.height == 50
    # Eager path caches.
    assert conn._df is not None


@pytest.mark.asyncio
async def test_get_schema_after_eager_fetch_uses_cached_df(tmp_path: Path) -> None:
    path = _write_csv(tmp_path, rows=20)
    conn = LocalFileConnector(str(path))
    await conn.connect()

    await conn.fetch_data()  # eager; populates self._df
    schema = await conn.get_schema()

    assert set(schema.keys()) == {"a", "b"}
