"""Phase 2 (backend Polars migration): engine-selected ingestion in the catalogs.

``catalog.load`` must honor ``Settings.SKYULF_ENGINE``:

- ``polars`` (the platform default) returns Polars frames end-to-end.
- ``pandas`` keeps the historical behavior exactly.

The ingestion parity cases required by the migration plan are pinned here:
a literal ``NaN`` token, empty fields, and mixed int/null columns. Note the
deliberate engine difference for the ``NaN`` token: ``pl.read_csv`` stores it
as float ``NaN`` (distinct from null in Polars) where pandas stores missing —
skyulf-core's NaN-aware operators (F-04/F-06/F-13) handle both forms.
"""

from io import StringIO

import pandas as pd
import polars as pl
import pytest

from backend.config import get_settings
from backend.data.catalog import FileSystemCatalog

NAN_CSV = "a,b,c\n1,x,NaN\n,y,2\n3,z,\n"


@pytest.fixture
def polars_engine(monkeypatch):
    settings = get_settings()
    monkeypatch.setattr(settings, "SKYULF_ENGINE", "polars", raising=False)
    return settings


@pytest.fixture
def pandas_engine(monkeypatch):
    settings = get_settings()
    monkeypatch.setattr(settings, "SKYULF_ENGINE", "pandas", raising=False)
    return settings


def _write(tmp_path, name, text):
    path = tmp_path / name
    path.write_text(text, encoding="utf-8")
    return str(path)


# ── Default engine is polars (the value itself is pinned in test_settings_engine) ──


def test_load_csv_returns_polars_by_default(tmp_path, polars_engine):
    catalog = FileSystemCatalog(base_path=str(tmp_path))
    _write(tmp_path, "d.csv", "a,b\n1,x\n2,y\n3,z\n")
    df = catalog.load("d.csv")
    assert isinstance(df, pl.DataFrame)
    assert df.shape == (3, 2)


def test_load_csv_limit_maps_to_n_rows(tmp_path, polars_engine):
    catalog = FileSystemCatalog(base_path=str(tmp_path))
    _write(tmp_path, "d.csv", "a\n1\n2\n3\n4\n5\n")
    df = catalog.load("d.csv", limit=2)
    assert isinstance(df, pl.DataFrame)
    assert df["a"].to_list() == [1, 2]


def test_load_parquet_returns_polars(tmp_path, polars_engine):
    pytest.importorskip("pyarrow")
    catalog = FileSystemCatalog(base_path=str(tmp_path))
    pd.DataFrame({"a": [1, 2, 3, 4]}).to_parquet(tmp_path / "d.parquet")
    df = catalog.load("d.parquet", limit=3)
    assert isinstance(df, pl.DataFrame)
    assert df["a"].to_list() == [1, 2, 3]


def test_load_json_returns_polars(tmp_path, polars_engine):
    catalog = FileSystemCatalog(base_path=str(tmp_path))
    (tmp_path / "d.json").write_text('[{"a": 1, "b": "x"}, {"a": 2, "b": "y"}]', encoding="utf-8")
    df = catalog.load("d.json")
    assert isinstance(df, pl.DataFrame)
    assert df.shape == (2, 2)


def test_load_excel_returns_polars(tmp_path, polars_engine):
    pytest.importorskip("openpyxl")
    catalog = FileSystemCatalog(base_path=str(tmp_path))
    pd.DataFrame({"a": [1, 2]}).to_excel(tmp_path / "d.xlsx", index=False)
    df = catalog.load("d.xlsx")
    assert isinstance(df, pl.DataFrame)
    assert df["a"].to_list() == [1, 2]


def test_load_extensionless_fallback_parquet_returns_polars(tmp_path, polars_engine):
    pytest.importorskip("pyarrow")
    catalog = FileSystemCatalog(base_path=str(tmp_path))
    pd.DataFrame({"a": [1, 2]}).to_parquet(tmp_path / "dataset_no_ext")
    df = catalog.load("dataset_no_ext")
    assert isinstance(df, pl.DataFrame)
    assert df.shape == (2, 1)


# ── pandas stays pandas ───────────────────────────────────────────────────


def test_load_csv_returns_pandas_when_engine_pandas(tmp_path, pandas_engine):
    catalog = FileSystemCatalog(base_path=str(tmp_path))
    _write(tmp_path, "d.csv", "a,b\n1,x\n2,y\n")
    df = catalog.load("d.csv")
    assert isinstance(df, pd.DataFrame)


def test_load_parquet_returns_pandas_when_engine_pandas(tmp_path, pandas_engine):
    pytest.importorskip("pyarrow")
    catalog = FileSystemCatalog(base_path=str(tmp_path))
    pd.DataFrame({"a": [1, 2, 3]}).to_parquet(tmp_path / "d.parquet")
    df = catalog.load("d.parquet", limit=2)
    assert isinstance(df, pd.DataFrame)
    assert df["a"].tolist() == [1, 2]


# ── Ingestion parity: NaN token, empty fields, mixed int/null ─────────────


def _missing_mask_polars(series: pl.Series) -> list[bool]:
    # In Polars, missingness is null OR float NaN — pandas isna() covers both.
    mask = series.is_null()
    if series.dtype.is_float():
        mask = mask | series.is_nan()
    return mask.to_list()


def test_csv_nan_token_and_empty_field_parity(tmp_path, polars_engine):
    catalog = FileSystemCatalog(base_path=str(tmp_path))
    _write(tmp_path, "d.csv", NAN_CSV)

    pdf = pd.read_csv(tmp_path / "d.csv")
    pol = catalog.load("d.csv")
    assert isinstance(pol, pl.DataFrame)

    # The F-13 trigger: polars keeps the NaN token as float NaN, not null.
    assert pd.isna(pol["c"][0])

    # Missingness masks must agree across engines on every column.
    for col in ("a", "b", "c"):
        assert _missing_mask_polars(pol[col]) == pd.isna(pdf[col]).tolist(), col

    # Non-missing values agree too.
    assert pol["a"].drop_nulls().to_list() == [1, 3]
    assert pol["b"].to_list() == ["x", "y", "z"]
    assert pol["c"].drop_nulls().drop_nans().to_list() == [2.0]


def test_csv_mixed_int_null_column_parity(tmp_path, polars_engine):
    catalog = FileSystemCatalog(base_path=str(tmp_path))
    _write(tmp_path, "d.csv", "n\n1\n\n3\n")

    pol = catalog.load("d.csv")
    # Polars keeps nullable integers as integers (pandas upcasts to float64).
    assert pol["n"].dtype == pl.Int64
    assert pol["n"].is_null().to_list() == [False, True, False]
    assert pol["n"].drop_nulls().to_list() == [1, 3]


# ── DataProfiler tolerates Polars frames (preview path, meta.py) ──────────


def test_data_profiler_matches_across_engines():
    from backend.ml_pipeline._internal._advisor import DataProfiler

    pdf = pd.read_csv(StringIO(NAN_CSV))
    pol = pl.read_csv(StringIO(NAN_CSV))

    profile_pd = DataProfiler.generate_profile(pdf)
    profile_pl = DataProfiler.generate_profile(pol)

    assert profile_pl.row_count == profile_pd.row_count == 3
    for col in ("a", "b", "c"):
        assert (
            profile_pl.columns[col]["missing_count"] == profile_pd.columns[col]["missing_count"]
        ), col
        assert profile_pl.columns[col]["unique_count"] == profile_pd.columns[col]["unique_count"]
