"""Tests for skyulf.profiling._analyzer.geo.GeoMixin."""

import numpy as np
import polars as pl

from skyulf.profiling.analyzer import EDAAnalyzer


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


def test_analyze_geospatial_infers_latitude_candidate() -> None:
    """Auto-detection should recognize a column literally named 'latitude' (line 28)."""
    df = pl.DataFrame({"latitude": [1.0, 2.0, 3.0], "lng": [10.0, 11.0, 12.0]})
    analyzer = EDAAnalyzer(df)
    result = analyzer._analyze_geospatial(numeric_cols=["latitude", "lng"])
    assert result is not None
    assert result.lat_col == "latitude"


def test_analyze_geospatial_infers_longitude_candidate() -> None:
    """lon_col=None with a 'longitude' column present should be auto-detected (line 30)."""
    df = pl.DataFrame({"lat": [1.0, 2.0, 3.0], "longitude": [10.0, 11.0, 12.0]})
    analyzer = EDAAnalyzer(df)
    result = analyzer._analyze_geospatial(numeric_cols=["lat", "longitude"])
    assert result is not None
    assert result.lat_col == "lat"
    assert result.lon_col == "longitude"


def test_analyze_geospatial_infers_lng_candidate() -> None:
    """Auto-detection should recognize a column literally named 'lng' (line 35)."""
    df = pl.DataFrame({"lat": [1.0, 2.0, 3.0], "lng": [10.0, 11.0, 12.0]})
    analyzer = EDAAnalyzer(df)
    result = analyzer._analyze_geospatial(numeric_cols=["lat", "lng"])
    assert result is not None
    assert result.lon_col == "lng"


def test_analyze_geospatial_infers_lon_candidate() -> None:
    """Auto-detection should recognize a column literally named 'lon' (line 37)."""
    df = pl.DataFrame({"lat": [1.0, 2.0, 3.0], "lon": [10.0, 11.0, 12.0]})
    analyzer = EDAAnalyzer(df)
    result = analyzer._analyze_geospatial(numeric_cols=["lat", "lon"])
    assert result is not None
    assert result.lon_col == "lon"


def test_analyze_geospatial_infers_long_candidate() -> None:
    """Auto-detection should recognize a column literally named 'long' (line 39)."""
    df = pl.DataFrame({"lat": [1.0, 2.0, 3.0], "long": [10.0, 11.0, 12.0]})
    analyzer = EDAAnalyzer(df)
    result = analyzer._analyze_geospatial(numeric_cols=["lat", "long"])
    assert result is not None
    assert result.lon_col == "long"


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
