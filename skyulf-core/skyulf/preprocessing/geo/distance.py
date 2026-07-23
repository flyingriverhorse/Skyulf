"""GeoDistance node — great-circle (haversine) or flat-plane distance between
two lat/lon coordinate pairs.

Pure math (no optional geospatial dependency): both the pandas and polars
engines compute the distance directly with trigonometric expressions, so this
node works without ``geopandas``/``shapely``/``h3`` installed.
"""

from typing import Any, cast

import numpy as np
import pandas as pd

from ..._validation import raise_invalid_choice
from ...core.artifacts import GeoDistanceArtifact
from ...core.meta.decorators import node_meta
from ...registry import NodeRegistry
from .._helpers import to_pandas
from ..base import BaseApplier, BaseCalculator, apply_method, fit_method
from ..dispatcher import apply_dual_engine

_SUPPORTED_METHODS = ("haversine", "euclidean")
_SUPPORTED_UNITS = ("km", "mi")

# Mean Earth radius, matching the reference value used by most GIS libraries.
_EARTH_RADIUS_KM = 6371.0088
_KM_TO_MI = 0.6213711922


def _radius_for_unit(unit: str) -> float:
    """Return the Earth radius in the requested unit ("km" or "mi")."""
    return _EARTH_RADIUS_KM if unit == "km" else _EARTH_RADIUS_KM * _KM_TO_MI


def _haversine_pandas(
    lat1: pd.Series, lon1: pd.Series, lat2: pd.Series, lon2: pd.Series, radius: float
) -> pd.Series:
    """Great-circle distance between two lat/lon series, in ``radius``'s unit."""
    lat1_r, lon1_r, lat2_r, lon2_r = (
        np.radians(lat1.astype(float)),
        np.radians(lon1.astype(float)),
        np.radians(lat2.astype(float)),
        np.radians(lon2.astype(float)),
    )
    dlat = lat2_r - lat1_r
    dlon = lon2_r - lon1_r
    a = np.sin(dlat / 2.0) ** 2 + np.cos(lat1_r) * np.cos(lat2_r) * np.sin(dlon / 2.0) ** 2
    c = 2.0 * np.arctan2(np.sqrt(a), np.sqrt(1.0 - a))
    return radius * c


def _euclidean_pandas(
    lat1: pd.Series, lon1: pd.Series, lat2: pd.Series, lon2: pd.Series, radius: float
) -> pd.Series:
    """Cheap flat-plane (equirectangular) approximation, in ``radius``'s unit."""
    lat1_r, lon1_r, lat2_r, lon2_r = (
        np.radians(lat1.astype(float)),
        np.radians(lon1.astype(float)),
        np.radians(lat2.astype(float)),
        np.radians(lon2.astype(float)),
    )
    mean_lat = (lat1_r + lat2_r) / 2.0
    x = (lon2_r - lon1_r) * np.cos(mean_lat)
    y = lat2_r - lat1_r
    return radius * np.sqrt(x**2 + y**2)


def _geo_distance_apply_pandas(X: Any, _y: Any, params: dict[str, Any]) -> tuple[Any, Any]:
    """Compute the configured distance and append it as a new pandas column."""
    lat1_col, lon1_col = params["lat1_col"], params["lon1_col"]
    lat2_col, lon2_col = params["lat2_col"], params["lon2_col"]
    if not all(c in X.columns for c in (lat1_col, lon1_col, lat2_col, lon2_col)):
        return X, _y

    radius = _radius_for_unit(params.get("unit", "km"))
    fn = (
        _haversine_pandas if params.get("method", "haversine") == "haversine" else _euclidean_pandas
    )
    distance = fn(X[lat1_col], X[lon1_col], X[lat2_col], X[lon2_col], radius)

    out = X.copy()
    out[params.get("output_column", "geo_distance_km")] = distance
    return out, _y


def _geo_distance_apply_polars(X: Any, _y: Any, params: dict[str, Any]) -> tuple[Any, Any]:
    """Compute the configured distance and append it as a new polars column."""
    import polars as pl

    lat1_col, lon1_col = params["lat1_col"], params["lon1_col"]
    lat2_col, lon2_col = params["lat2_col"], params["lon2_col"]
    if not all(c in X.columns for c in (lat1_col, lon1_col, lat2_col, lon2_col)):
        return X, _y

    radius = _radius_for_unit(params.get("unit", "km"))
    lat1_r = pl.col(lat1_col).cast(pl.Float64).radians()
    lon1_r = pl.col(lon1_col).cast(pl.Float64).radians()
    lat2_r = pl.col(lat2_col).cast(pl.Float64).radians()
    lon2_r = pl.col(lon2_col).cast(pl.Float64).radians()

    if params.get("method", "haversine") == "haversine":
        dlat = lat2_r - lat1_r
        dlon = lon2_r - lon1_r
        a = (dlat / 2.0).sin() ** 2 + lat1_r.cos() * lat2_r.cos() * (dlon / 2.0).sin() ** 2
        c = 2.0 * pl.arctan2(a.sqrt(), (1.0 - a).sqrt())
        expr = radius * c
    else:
        mean_lat = (lat1_r + lat2_r) / 2.0
        x = (lon2_r - lon1_r) * mean_lat.cos()
        y = lat2_r - lat1_r
        expr = radius * (x**2 + y**2).sqrt()

    output_column = params.get("output_column", "geo_distance_km")
    return X.with_columns(expr.alias(output_column)), _y


def _validate_geo_distance_columns(X_pd: pd.DataFrame, cols: list[str]) -> None:
    """Raise ValueError if any of `cols` is missing or non-numeric in X_pd."""
    missing = [c for c in cols if not c or c not in X_pd.columns]
    if missing:
        raise ValueError(f"GeoDistance: columns not found in data: {missing}")

    non_numeric = [c for c in cols if not pd.api.types.is_numeric_dtype(X_pd[c])]
    if non_numeric:
        raise ValueError(f"GeoDistance requires numeric columns; non-numeric: {non_numeric}")


def _validate_geo_distance_method_unit(method: str, unit: str) -> None:
    """Raise ValueError if `method` or `unit` aren't in the supported sets."""
    if method not in _SUPPORTED_METHODS:
        raise_invalid_choice(method, _SUPPORTED_METHODS, "GeoDistance method")

    if unit not in _SUPPORTED_UNITS:
        raise_invalid_choice(unit, _SUPPORTED_UNITS, "GeoDistance unit")


class GeoDistanceApplier(BaseApplier):
    @apply_method
    def apply(self, X: Any, _y: Any, params: dict[str, Any]) -> Any:  # pylint: disable=arguments-differ
        return apply_dual_engine(X, params, _geo_distance_apply_polars, _geo_distance_apply_pandas)


@NodeRegistry.register("GeoDistance", GeoDistanceApplier)
@node_meta(
    id="GeoDistance",
    name="Geo Distance",
    category="Feature Engineering",
    description=(
        "Compute the great-circle (haversine) or flat-plane (euclidean) distance "
        "between two lat/lon coordinate pairs, appended as a new numeric column."
    ),
    params={
        "lat1_col": "",
        "lon1_col": "",
        "lat2_col": "",
        "lon2_col": "",
        "method": "haversine",
        "unit": "km",
        "output_column": "geo_distance_km",
    },
)
class GeoDistanceCalculator(BaseCalculator):
    @fit_method
    def fit(self, X: Any, _y: Any, config: dict[str, Any]) -> GeoDistanceArtifact:  # pylint: disable=arguments-differ
        X_pd = to_pandas(X)

        lat1_col = config.get("lat1_col", "")
        lon1_col = config.get("lon1_col", "")
        lat2_col = config.get("lat2_col", "")
        lon2_col = config.get("lon2_col", "")
        cols = [lat1_col, lon1_col, lat2_col, lon2_col]
        _validate_geo_distance_columns(X_pd, cols)

        method = config.get("method", "haversine")
        unit = config.get("unit", "km")
        _validate_geo_distance_method_unit(method, unit)

        output_column = config.get("output_column", "geo_distance_km")

        return cast(
            GeoDistanceArtifact,
            {
                "type": "geo_distance",
                "lat1_col": lat1_col,
                "lon1_col": lon1_col,
                "lat2_col": lat2_col,
                "lon2_col": lon2_col,
                "method": method,
                "unit": unit,
                "output_column": output_column,
            },
        )
