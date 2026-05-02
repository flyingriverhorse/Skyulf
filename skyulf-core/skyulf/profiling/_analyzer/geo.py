"""Geospatial analysis (lat/lon detection + sample point extraction)."""

from typing import List, Optional

import polars as pl

from ..schemas import GeoPoint, GeospatialStats
from ._utils import _AnalyzerState, _collect


class GeoMixin(_AnalyzerState):
    """Geospatial helpers for :class:`EDAAnalyzer`."""

    def _analyze_geospatial(
        self,
        numeric_cols: List[str],
        target_col: Optional[str] = None,
        lat_col: Optional[str] = None,
        lon_col: Optional[str] = None,
    ) -> Optional[GeospatialStats]:
        """Detect lat/lon columns by name, then return bbox + sample points."""
        try:
            if not lat_col or not lon_col:
                # Check ALL columns (may have been inferred as strings).
                candidates = {c.lower(): c for c in self.columns}  # type: ignore[attr-defined]
                if not lat_col:
                    if "latitude" in candidates:
                        lat_col = candidates["latitude"]
                    elif "lat" in candidates:
                        lat_col = candidates["lat"]
                if not lon_col:
                    if "longitude" in candidates:
                        lon_col = candidates["longitude"]
                    elif "lng" in candidates:
                        lon_col = candidates["lng"]
                    elif "lon" in candidates:
                        lon_col = candidates["lon"]
                    elif "long" in candidates:
                        lon_col = candidates["long"]

            if not lat_col or not lon_col:
                return None

            geo_df = self.lazy_df.select(  # type: ignore[attr-defined]
                [
                    pl.col(lat_col).cast(pl.Float64, strict=False).alias("lat"),
                    pl.col(lon_col).cast(pl.Float64, strict=False).alias("lon"),
                ]
            )

            stats = _collect(
                geo_df.select(
                    [
                        pl.col("lat").min().alias("min_lat"),
                        pl.col("lat").max().alias("max_lat"),
                        pl.col("lat").mean().alias("mean_lat"),
                        pl.col("lon").min().alias("min_lon"),
                        pl.col("lon").max().alias("max_lon"),
                        pl.col("lon").mean().alias("mean_lon"),
                    ]
                )
            ).row(0)

            # All casts failed → not actually geospatial.
            if stats[0] is None or stats[3] is None:
                return None

            sample_size = min(5000, self.row_count)  # type: ignore[attr-defined]

            cols_to_fetch = [
                pl.col(lat_col).cast(pl.Float64, strict=False).alias("lat"),
                pl.col(lon_col).cast(pl.Float64, strict=False).alias("lon"),
            ]
            if target_col and target_col in self.columns:  # type: ignore[attr-defined]
                cols_to_fetch.append(pl.col(target_col).alias("target"))

            sample_df = _collect(self.lazy_df.select(cols_to_fetch)).sample(  # type: ignore[attr-defined]
                n=sample_size, with_replacement=False, seed=42
            )

            points = []
            for row in sample_df.to_dicts():
                if row["lat"] is None or row["lon"] is None:
                    continue
                label = (
                    str(row["target"])
                    if "target" in row and row["target"] is not None
                    else None
                )
                points.append(GeoPoint(lat=row["lat"], lon=row["lon"], label=label))

            return GeospatialStats(
                lat_col=lat_col,
                lon_col=lon_col,
                min_lat=stats[0],
                max_lat=stats[1],
                centroid_lat=stats[2],
                min_lon=stats[3],
                max_lon=stats[4],
                centroid_lon=stats[5],
                sample_points=points,
            )
        except Exception as e:
            print(f"Error in geospatial analysis: {e}")
            return None
