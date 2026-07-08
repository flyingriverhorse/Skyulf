"""Tests for the geo preprocessing nodes: GeoDistance and H3Index.

Covers haversine distance correctness against a known reference value,
euclidean-approximation sanity, H3 index parity against calling ``h3``
directly, column validation errors, and pandas/polars engine parity.
"""

import numpy as np
import pandas as pd
import polars as pl
import pytest
from tests.utils.dataset_loader import load_sample_dataset
from tests.utils.test_case_loader import TestCaseLoader

from skyulf.preprocessing.geo import (
    GeoDistanceApplier,
    GeoDistanceCalculator,
    H3IndexApplier,
    H3IndexCalculator,
)

_geo_distance_validation_cases = TestCaseLoader("geo/geo_distance_validation").load()

# New York City (JFK) and Los Angeles (LAX) airport coordinates.
# Known great-circle distance is ~3936 km (commonly cited reference value).
_NYC_LAT, _NYC_LON = 40.6413, -73.7781
_LAX_LAT, _LAX_LON = 33.9416, -118.4085
_NYC_LAX_KM = 3936.0


def _cities_df() -> pd.DataFrame:
    """Two-row pandas DataFrame pairing NYC with LAX, and NYC with itself."""
    return pd.DataFrame(
        {
            "lat1": [_NYC_LAT, _NYC_LAT],
            "lon1": [_NYC_LON, _NYC_LON],
            "lat2": [_LAX_LAT, _NYC_LAT],
            "lon2": [_LAX_LON, _NYC_LON],
        }
    )


class TestHaversineDistance:
    """Haversine distance correctness against known reference values."""

    def test_nyc_to_lax_matches_known_distance(self) -> None:
        df = _cities_df()
        calc = GeoDistanceCalculator()
        applier = GeoDistanceApplier()
        params = calc.fit(
            df, {"lat1_col": "lat1", "lon1_col": "lon1", "lat2_col": "lat2", "lon2_col": "lon2"}
        )
        result = applier.apply(df, params)

        assert result["geo_distance_km"].iloc[0] == pytest.approx(_NYC_LAX_KM, rel=0.01)

    def test_same_point_distance_is_zero(self) -> None:
        df = _cities_df()
        calc = GeoDistanceCalculator()
        applier = GeoDistanceApplier()
        params = calc.fit(
            df, {"lat1_col": "lat1", "lon1_col": "lon1", "lat2_col": "lat2", "lon2_col": "lon2"}
        )
        result = applier.apply(df, params)

        assert result["geo_distance_km"].iloc[1] == pytest.approx(0.0, abs=1e-6)

    def test_miles_unit_conversion(self) -> None:
        df = _cities_df()
        calc = GeoDistanceCalculator()
        applier = GeoDistanceApplier()
        params_km = calc.fit(
            df, {"lat1_col": "lat1", "lon1_col": "lon1", "lat2_col": "lat2", "lon2_col": "lon2"}
        )
        params_mi = calc.fit(
            df,
            {
                "lat1_col": "lat1",
                "lon1_col": "lon1",
                "lat2_col": "lat2",
                "lon2_col": "lon2",
                "unit": "mi",
                "output_column": "geo_distance_mi",
            },
        )
        result_km = applier.apply(df, params_km)
        result_mi = applier.apply(df, params_mi)

        np.testing.assert_allclose(
            result_mi["geo_distance_mi"].iloc[0],
            result_km["geo_distance_km"].iloc[0] * 0.6213711922,
            rtol=1e-6,
        )

    def test_custom_output_column_name(self) -> None:
        df = _cities_df()
        calc = GeoDistanceCalculator()
        applier = GeoDistanceApplier()
        params = calc.fit(
            df,
            {
                "lat1_col": "lat1",
                "lon1_col": "lon1",
                "lat2_col": "lat2",
                "lon2_col": "lon2",
                "output_column": "dist",
            },
        )
        result = applier.apply(df, params)

        assert "dist" in result.columns
        assert "geo_distance_km" not in result.columns


class TestEuclideanDistance:
    """Flat-plane approximation sanity checks."""

    def test_euclidean_close_to_haversine_for_short_distance(self) -> None:
        # Small offsets: the flat-plane approximation should be very close
        # to the great-circle distance over short ranges.
        df = pd.DataFrame({"lat1": [10.0], "lon1": [10.0], "lat2": [10.01], "lon2": [10.01]})
        calc = GeoDistanceCalculator()
        applier = GeoDistanceApplier()
        params_hav = calc.fit(
            df, {"lat1_col": "lat1", "lon1_col": "lon1", "lat2_col": "lat2", "lon2_col": "lon2"}
        )
        params_euc = calc.fit(
            df,
            {
                "lat1_col": "lat1",
                "lon1_col": "lon1",
                "lat2_col": "lat2",
                "lon2_col": "lon2",
                "method": "euclidean",
                "output_column": "euclidean_km",
            },
        )
        result_hav = applier.apply(df, params_hav)
        result_euc = applier.apply(df, params_euc)

        np.testing.assert_allclose(
            result_euc["euclidean_km"].iloc[0],
            result_hav["geo_distance_km"].iloc[0],
            rtol=1e-2,
        )


class TestGeoDistanceValidation:
    """Config validation errors — scenarios loaded from
    ``tests/test_cases/geo/geo_distance_validation.json``.
    """

    @pytest.mark.parametrize(*_geo_distance_validation_cases)
    def test_invalid_config_raises(
        self,
        lat1_col: str,
        lon1_col: str,
        lat2_col: str,
        lon2_col: str,
        extra_config: dict,
        error_match: str,
    ) -> None:
        df = _cities_df()
        calc = GeoDistanceCalculator()
        with pytest.raises(ValueError, match=error_match):
            calc.fit(
                df,
                {
                    "lat1_col": lat1_col,
                    "lon1_col": lon1_col,
                    "lat2_col": lat2_col,
                    "lon2_col": lon2_col,
                    **extra_config,
                },
            )

    def test_non_numeric_column_raises(self) -> None:
        df = _cities_df()
        df["lat1"] = df["lat1"].astype(str)
        calc = GeoDistanceCalculator()
        with pytest.raises(ValueError):
            calc.fit(
                df, {"lat1_col": "lat1", "lon1_col": "lon1", "lat2_col": "lat2", "lon2_col": "lon2"}
            )


class TestGeoDistanceEngineParity:
    """Pandas and polars engines must produce numerically identical output."""

    def test_haversine_parity(self) -> None:
        df_pd = _cities_df()
        df_pl = pl.from_pandas(df_pd)

        calc = GeoDistanceCalculator()
        applier = GeoDistanceApplier()
        params = calc.fit(
            df_pd,
            {"lat1_col": "lat1", "lon1_col": "lon1", "lat2_col": "lat2", "lon2_col": "lon2"},
        )

        result_pd = applier.apply(df_pd, params)
        result_pl = applier.apply(df_pl, params)

        np.testing.assert_allclose(
            result_pd["geo_distance_km"].to_numpy(),
            result_pl["geo_distance_km"].to_numpy(),
            rtol=1e-9,
        )

    def test_euclidean_parity(self) -> None:
        df_pd = _cities_df()
        df_pl = pl.from_pandas(df_pd)

        calc = GeoDistanceCalculator()
        applier = GeoDistanceApplier()
        params = calc.fit(
            df_pd,
            {
                "lat1_col": "lat1",
                "lon1_col": "lon1",
                "lat2_col": "lat2",
                "lon2_col": "lon2",
                "method": "euclidean",
            },
        )

        result_pd = applier.apply(df_pd, params)
        result_pl = applier.apply(df_pl, params)

        np.testing.assert_allclose(
            result_pd["geo_distance_km"].to_numpy(),
            result_pl["geo_distance_km"].to_numpy(),
            rtol=1e-9,
        )


class TestRealShapedDataset:
    """Integration-style check against the checked-in ``customers.csv`` sample,
    which has missing lat/lon rows — closer to production data than the
    small synthetic ``_cities_df()`` frame used elsewhere in this file.
    """

    def test_distance_from_nyc_handles_missing_coordinates(self) -> None:
        df = load_sample_dataset("customers")
        # Match the exact New York coordinates used in customers.csv (Manhattan),
        # not the airport-based _NYC_LAT/_NYC_LON reference used elsewhere in this file.
        df["ref_lat"] = 40.7128
        df["ref_lon"] = -74.0060

        calc = GeoDistanceCalculator()
        applier = GeoDistanceApplier()
        params = calc.fit(
            df, {"lat1_col": "lat", "lon1_col": "lon", "lat2_col": "ref_lat", "lon2_col": "ref_lon"}
        )
        result = applier.apply(df, params)

        missing_mask = df["lat"].isna() | df["lon"].isna()
        assert result.loc[missing_mask, "geo_distance_km"].isna().all()
        assert result.loc[~missing_mask, "geo_distance_km"].notna().all()
        # The New York customers should be ~0 km from the NYC reference point.
        ny_rows = df["city"] == "New York"
        np.testing.assert_allclose(result.loc[ny_rows, "geo_distance_km"], 0.0, atol=1.0)


h3 = pytest.importorskip("h3", reason="h3 is an optional dependency (pip install skyulf-core[geo])")


def _points_df() -> pd.DataFrame:
    """Small pandas DataFrame of lat/lon points for H3 index tests."""
    return pd.DataFrame(
        {
            "lat": [_NYC_LAT, _LAX_LAT, 0.0],
            "lon": [_NYC_LON, _LAX_LON, 0.0],
        }
    )


class TestH3Index:
    """H3 index correctness, verified directly against the ``h3`` package."""

    def test_matches_h3_latlng_to_cell(self) -> None:
        df = _points_df()
        calc = H3IndexCalculator()
        applier = H3IndexApplier()
        params = calc.fit(df, {"lat_col": "lat", "lon_col": "lon", "resolution": 9})
        result = applier.apply(df, params)

        expected = [h3.latlng_to_cell(lat, lon, 9) for lat, lon in zip(df["lat"], df["lon"], strict=True)]
        assert result["h3_index"].tolist() == expected

    def test_different_resolution_changes_cell(self) -> None:
        df = _points_df()
        calc = H3IndexCalculator()
        applier = H3IndexApplier()
        params_low = calc.fit(df, {"lat_col": "lat", "lon_col": "lon", "resolution": 3})
        params_high = calc.fit(df, {"lat_col": "lat", "lon_col": "lon", "resolution": 9})

        result_low = applier.apply(df, params_low)
        result_high = applier.apply(df, params_high)

        assert result_low["h3_index"].iloc[0] != result_high["h3_index"].iloc[0]

    def test_custom_output_column_name(self) -> None:
        df = _points_df()
        calc = H3IndexCalculator()
        applier = H3IndexApplier()
        params = calc.fit(
            df, {"lat_col": "lat", "lon_col": "lon", "resolution": 9, "output_column": "cell"}
        )
        result = applier.apply(df, params)

        assert "cell" in result.columns
        assert "h3_index" not in result.columns

    def test_polars_engine_parity(self) -> None:
        df_pd = _points_df()
        df_pl = pl.from_pandas(df_pd)

        calc = H3IndexCalculator()
        applier = H3IndexApplier()
        params = calc.fit(df_pd, {"lat_col": "lat", "lon_col": "lon", "resolution": 9})

        result_pd = applier.apply(df_pd, params)
        result_pl = applier.apply(df_pl, params)

        assert result_pd["h3_index"].tolist() == result_pl["h3_index"].to_list()


class TestH3IndexValidation:
    """Config validation errors."""

    def test_missing_column_raises(self) -> None:
        df = _points_df()
        calc = H3IndexCalculator()
        with pytest.raises(ValueError):
            calc.fit(df, {"lat_col": "lat", "lon_col": "missing", "resolution": 9})

    def test_non_numeric_column_raises(self) -> None:
        df = _points_df()
        df["lat"] = df["lat"].astype(str)
        calc = H3IndexCalculator()
        with pytest.raises(ValueError):
            calc.fit(df, {"lat_col": "lat", "lon_col": "lon", "resolution": 9})

    def test_out_of_range_resolution_raises(self) -> None:
        df = _points_df()
        calc = H3IndexCalculator()
        with pytest.raises(ValueError):
            calc.fit(df, {"lat_col": "lat", "lon_col": "lon", "resolution": 20})
