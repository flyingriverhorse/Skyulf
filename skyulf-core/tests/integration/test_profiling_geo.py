"""Tests for skyulf.profiling._analyzer.geo.GeoMixin."""

import numpy as np
import polars as pl
import pytest
from tests.utils.dataset_loader import load_sample_dataset
from tests.utils.test_case_loader import TestCaseLoader

from skyulf.profiling.analyzer import EDAAnalyzer

_geo_candidate_cases = TestCaseLoader("profiling/geo_candidate_inference").load()


def _geo_df(n: int = 50) -> pl.DataFrame:
    """Small dataset with 'lat'/'lon'-style columns and a target for labeling."""
    rng = np.random.default_rng(7)
    lat = rng.uniform(30.0, 45.0, n)
    lon = rng.uniform(-10.0, 10.0, n)
    target = rng.choice(["A", "B"], size=n)
    return pl.DataFrame({"latitude": lat, "longitude": lon, "target": target})


def test_analyze_geospatial_explicit_columns() -> None:
    """Explicit lat_col/lon_col should be honored, bbox + sample points populated."""
    df = _geo_df()
    analyzer = EDAAnalyzer(df)
    result = analyzer._analyze_geospatial(
        numeric_cols=["latitude", "longitude"],
        target_col="target",
        lat_col="latitude",
        lon_col="longitude",
    )
    assert result is not None
    assert result.lat_col == "latitude"
    assert result.lon_col == "longitude"
    assert result.min_lat <= result.max_lat
    assert result.min_lon <= result.max_lon
    assert len(result.sample_points) > 0
    assert all(p.label in ("A", "B") for p in result.sample_points)


class TestGeoCandidateInference:
    """Auto-detection of lat/lon-like column names — scenarios loaded from
    ``tests/test_cases/profiling/geo_candidate_inference.json``.
    """

    @pytest.mark.parametrize(*_geo_candidate_cases)
    def test_analyze_geospatial_infers_candidate(
        self,
        columns: dict[str, list[float]],
        numeric_cols: list[str],
        expected_lat_col: str | None,
        expected_lon_col: str | None,
    ) -> None:
        df = pl.DataFrame(columns)
        analyzer = EDAAnalyzer(df)
        result = analyzer._analyze_geospatial(numeric_cols=numeric_cols)
        assert result is not None
        if expected_lat_col is not None:
            assert result.lat_col == expected_lat_col
        if expected_lon_col is not None:
            assert result.lon_col == expected_lon_col


def test_analyze_geospatial_missing_columns_returns_none() -> None:
    """No lat/lon-like columns at all should short-circuit to None."""
    df = pl.DataFrame({"a": [1, 2, 3], "b": [4, 5, 6]})
    analyzer = EDAAnalyzer(df)
    assert analyzer._analyze_geospatial(numeric_cols=["a", "b"]) is None


def test_analyze_geospatial_non_numeric_casts_return_none() -> None:
    """All-string lat/lon columns that fail Float64 casting are not geospatial (line 66)."""
    df = pl.DataFrame(
        {
            "lat": ["north", "south", "east"],
            "lon": ["west", "central", "far"],
        }
    )
    analyzer = EDAAnalyzer(df)
    result = analyzer._analyze_geospatial(numeric_cols=["lat", "lon"], lat_col="lat", lon_col="lon")
    assert result is None


def test_analyze_geospatial_skips_null_sample_rows() -> None:
    """Null lat/lon values in the sampled rows should be skipped (line 84)."""
    n = 20
    lat = [float(i) if i % 4 != 0 else None for i in range(n)]
    lon = [float(i) * 2 if i % 4 != 0 else None for i in range(n)]
    df = pl.DataFrame({"lat": lat, "lon": lon})
    analyzer = EDAAnalyzer(df)
    result = analyzer._analyze_geospatial(numeric_cols=["lat", "lon"], lat_col="lat", lon_col="lon")
    assert result is not None
    # None of the null rows should have produced a sample point.
    assert all(p.lat is not None and p.lon is not None for p in result.sample_points)


def test_analyze_geospatial_exception_returns_none(monkeypatch) -> None:
    """An unexpected exception during analysis should be caught and return None (lines 101-103)."""
    df = _geo_df()
    analyzer = EDAAnalyzer(df)

    def _boom(*args, **kwargs):
        raise RuntimeError("boom")

    monkeypatch.setattr(analyzer.lazy_df, "select", _boom)
    result = analyzer._analyze_geospatial(
        numeric_cols=["latitude", "longitude"], lat_col="latitude", lon_col="longitude"
    )
    assert result is None


class TestRealShapedDataset:
    """Integration-style check against the checked-in ``customers.csv`` sample,
    which has missing ``lat``/``lon`` rows (city "Houston"... rest have coords,
    row 6 has both null) — closer to production data than the small synthetic
    ``_geo_df()`` frame used elsewhere in this file.
    """

    def test_analyze_geospatial_on_customers_skips_missing_coordinates(self) -> None:
        df = pl.from_pandas(load_sample_dataset("customers"))
        analyzer = EDAAnalyzer(df)
        result = analyzer._analyze_geospatial(
            numeric_cols=["lat", "lon"],
            target_col="plan_type",
            lat_col="lat",
            lon_col="lon",
        )

        assert result is not None
        assert result.lat_col == "lat"
        assert result.lon_col == "lon"
        assert result.min_lat <= result.max_lat
        assert result.min_lon <= result.max_lon
        # Row 6 (customer_id=6) has null lat/lon and must not produce a sample point.
        assert all(p.lat is not None and p.lon is not None for p in result.sample_points)
        assert all(p.label in ("premium", "basic", "enterprise") for p in result.sample_points)


class TestCoordinateRangeValidation:
    """Regression guard for the out-of-range lat/lon false-positive fix."""

    def test_analyze_geospatial_rejects_out_of_range_latitude(self) -> None:
        """A column named 'lat' holding values outside [-90, 90] (e.g. an ID
        or percentage column) must not be reported as valid geospatial data."""
        df = pl.DataFrame(
            {
                "lat": [100.0, 200.0, 300.0, 400.0],
                "lon": [10.0, 20.0, 30.0, 40.0],
            }
        )
        analyzer = EDAAnalyzer(df)
        result = analyzer._analyze_geospatial(
            numeric_cols=["lat", "lon"], lat_col="lat", lon_col="lon"
        )
        assert result is None

    def test_analyze_geospatial_rejects_out_of_range_longitude(self) -> None:
        """A column named 'lon' holding values outside [-180, 180] must not be
        reported as valid geospatial data."""
        df = pl.DataFrame(
            {
                "lat": [10.0, 20.0, 30.0, 40.0],
                "lon": [500.0, 600.0, 700.0, 800.0],
            }
        )
        analyzer = EDAAnalyzer(df)
        result = analyzer._analyze_geospatial(
            numeric_cols=["lat", "lon"], lat_col="lat", lon_col="lon"
        )
        assert result is None

    def test_analyze_geospatial_accepts_valid_boundary_coordinates(self) -> None:
        """Values exactly at the valid boundary (+/-90, +/-180) must still be accepted."""
        df = pl.DataFrame(
            {
                "lat": [-90.0, 0.0, 90.0, 45.0],
                "lon": [-180.0, 0.0, 180.0, 90.0],
            }
        )
        analyzer = EDAAnalyzer(df)
        result = analyzer._analyze_geospatial(
            numeric_cols=["lat", "lon"], lat_col="lat", lon_col="lon"
        )
        assert result is not None
