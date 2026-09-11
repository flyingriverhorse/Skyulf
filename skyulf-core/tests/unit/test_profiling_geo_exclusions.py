"""Geographic profiling must honor the analyzer's persistent column selection."""

import polars as pl
import pytest

from skyulf.profiling.analyzer import EDAAnalyzer


@pytest.mark.parametrize(
    ("excluded_columns", "lat_col", "lon_col"),
    [
        (["lat"], "lat", "lon"),
        (["lon"], "lat", "lon"),
        (["lat", "lon"], "lat", "lon"),
        (["lat"], "lat", None),
        (["lon"], None, "lon"),
        (["lat"], None, None),
        (["lon"], None, None),
    ],
)
def test_excluded_coordinates_do_not_appear_in_public_profile(
    excluded_columns: list[str], lat_col: str | None, lon_col: str | None
) -> None:
    """Excluding either coordinate must suppress points, bounds, and centroids."""
    frame = pl.DataFrame(
        {"lat": [1.25, 2.5, 3.75], "lon": [10.0, 20.0, 30.0], "x": [101, 102, 103]}
    )

    profile = EDAAnalyzer(frame).analyze(
        exclude_cols=excluded_columns, lat_col=lat_col, lon_col=lon_col
    )

    assert profile.excluded_columns == excluded_columns
    assert set(profile.columns).isdisjoint(excluded_columns)
    assert profile.sample_data is not None
    assert all(set(row).isdisjoint(excluded_columns) for row in profile.sample_data)
    assert profile.geospatial is None
    assert profile.model_dump()["geospatial"] is None


@pytest.mark.parametrize(
    ("lat_name", "lon_name", "explicit"),
    [("north", "east", True), ("Latitude", "Longitude", False)],
)
@pytest.mark.parametrize("as_strings", [False, True])
def test_allowed_coordinates_remain_available(
    lat_name: str, lon_name: str, explicit: bool, as_strings: bool
) -> None:
    """Selection checks must preserve explicit custom names and inferred string coordinates."""
    frame = pl.DataFrame(
        {lat_name: [1.25, 2.5, 3.75], lon_name: [10.0, 20.0, 30.0], "private": [7, 8, 9]}
    )
    if as_strings:
        frame = frame.with_columns(pl.col(lat_name, lon_name).cast(pl.String))

    profile = EDAAnalyzer(frame).analyze(
        exclude_cols=["private"],
        lat_col=lat_name if explicit else None,
        lon_col=lon_name if explicit else None,
    )

    geo = profile.geospatial
    assert geo is not None
    assert (geo.lat_col, geo.lon_col) == (lat_name, lon_name)
    assert (geo.min_lat, geo.max_lat, geo.centroid_lat) == (1.25, 3.75, 2.5)
    assert (geo.min_lon, geo.max_lon, geo.centroid_lon) == (10.0, 30.0, 20.0)
    assert sorted((point.lat, point.lon) for point in geo.sample_points) == [
        (1.25, 10.0),
        (2.5, 20.0),
        (3.75, 30.0),
    ]


def test_explicit_coordinates_cannot_restore_persistently_excluded_data() -> None:
    """Reusing an analyzer must not expose coordinates removed by an earlier analysis."""
    analyzer = EDAAnalyzer(
        pl.DataFrame({"lat": [1.25, 2.5, 3.75], "lon": [10.0, 20.0, 30.0], "x": [101, 102, 103]})
    )
    first = analyzer.analyze(exclude_cols=["lat", "lon"])

    repeated = analyzer.analyze(lat_col="lat", lon_col="lon")

    assert repeated.excluded_columns == first.excluded_columns == ["lat", "lon"]
    assert list(repeated.columns) == list(first.columns) == ["x"]
    assert repeated.sample_data == first.sample_data == [{"x": 101}, {"x": 102}, {"x": 103}]
    assert first.geospatial is None
    assert repeated.geospatial is None
