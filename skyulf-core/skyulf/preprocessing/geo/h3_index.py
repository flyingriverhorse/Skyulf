"""H3Index node — Uber H3 hierarchical hexagonal grid cell index per row.

Optional dependency: requires the ``h3`` package (install via
``pip install skyulf-core[geo]`` or ``pip install h3``). The import is lazy
so the rest of skyulf-core works without it installed.

``h3`` has no vectorized numpy/polars-native API, so cell indices are computed
row-wise: the pandas engine via ``.apply()``, the polars engine over the two
coordinate columns as numpy arrays (the frame itself is never converted).
"""

from collections.abc import Hashable
from typing import Any, cast

import pandas as pd
import polars as pl

from ...core.artifacts import H3IndexArtifact
from ...core.capabilities import ExecutionCapability
from ...core.meta.decorators import node_meta
from ...registry import NodeRegistry
from .._fitted_validation import local_state_fields
from .._helpers import select_then_to_pandas
from ..base import BaseApplier, BaseCalculator, apply_method, fit_method
from ..dispatcher import apply_dual_engine

_MIN_RESOLUTION = 0
_MAX_RESOLUTION = 15

_INSTALL_HINT = (
    "H3Index requires the 'h3' package. "
    "Install it with: pip install skyulf-core[geo]  (or  pip install h3)"
)


def _import_h3() -> Any:
    """Lazily import the optional ``h3`` package."""
    try:
        import h3  # ty: ignore[unresolved-import]  # noqa: PLC0415 - optional geo extra
    except ImportError as exc:  # pragma: no cover - depends on optional extra
        raise ImportError(_INSTALL_HINT) from exc
    return h3


def _h3_cell_or_none(lat: Any, lon: Any, h3: Any, resolution: int) -> Any:
    """Compute an H3 cell index for one row, returning ``None`` on missing/invalid input."""
    if pd.isna(lat) or pd.isna(lon):
        return None
    try:
        latitude, longitude = float(lat), float(lon)
        if not (-90 <= latitude <= 90 and -180 <= longitude <= 180):
            return None
        return h3.latlng_to_cell(latitude, longitude, resolution)
    except (ValueError, TypeError):
        return None


def _h3_index_apply_pandas(X: Any, _y: Any, params: dict[str, Any]) -> tuple[Any, Any]:
    """Compute the H3 cell index per row and append it as a new pandas column."""
    lat_col, lon_col = params["lat_col"], params["lon_col"]
    if lat_col not in X.columns or lon_col not in X.columns:
        return X, _y

    h3 = _import_h3()
    resolution = params.get("resolution", 9)

    out = X.copy()
    out[params.get("output_column", "h3_index")] = out.apply(
        lambda row: _h3_cell_or_none(row[lat_col], row[lon_col], h3, resolution),
        axis=1,
    )
    return out, _y


def _h3_index_apply_polars(X: Any, _y: Any, params: dict[str, Any]) -> tuple[Any, Any]:
    """Compute the H3 cell index from the lat/lon columns and attach it natively.

    Only the two coordinate columns cross into numpy; the rest of the frame is
    never converted, so unrelated column dtypes are preserved.
    """
    lat_col, lon_col = params["lat_col"], params["lon_col"]
    if lat_col not in X.columns or lon_col not in X.columns:
        return X, _y

    h3 = _import_h3()
    resolution = params.get("resolution", 9)

    lats = X.get_column(lat_col).to_numpy()
    lons = X.get_column(lon_col).to_numpy()
    cells = [
        _h3_cell_or_none(lat, lon, h3, resolution) for lat, lon in zip(lats, lons, strict=True)
    ]
    output_column = params.get("output_column", "h3_index")
    return X.with_columns(pl.Series(output_column, cells, dtype=pl.String)), _y


def _validate_h3_columns(X_pd: pd.DataFrame, lat_col: str, lon_col: str) -> None:
    """Raise ValueError if lat/lon columns are missing or non-numeric in X_pd."""
    cols = [lat_col, lon_col]
    missing = [c for c in cols if not c or c not in X_pd.columns]
    if missing:
        raise ValueError(f"H3Index: columns not found in data: {missing}")

    non_numeric = [c for c in cols if not pd.api.types.is_numeric_dtype(X_pd[c])]
    if non_numeric:
        raise ValueError(f"H3Index requires numeric columns; non-numeric: {non_numeric}")


def _validate_h3_resolution(resolution: Any) -> None:
    """Raise ValueError if resolution isn't an int within the allowed H3 range."""
    if not isinstance(resolution, int) or not (_MIN_RESOLUTION <= resolution <= _MAX_RESOLUTION):
        raise ValueError(
            f"H3Index resolution must be an int in [{_MIN_RESOLUTION}, {_MAX_RESOLUTION}], "
            f"got {resolution!r}"
        )


class H3IndexApplier(BaseApplier):
    """Append each row's H3 hexagonal cell index as a new string column."""

    @staticmethod
    def validate_inference_state(raw: dict) -> dict:
        """Inspect saved coordinate choices without importing H3 or computing cells."""
        local_state_fields(
            raw, "h3_index", {"type", "lat_col", "lon_col", "resolution", "output_column"}
        )
        if any(not isinstance(raw[key], str) or not raw[key] for key in ("lat_col", "lon_col")):
            raise ValueError("H3Index fitted coordinate names must be nonempty strings.")
        if not isinstance(raw["output_column"], Hashable):
            raise ValueError("H3Index fitted output name must be hashable.")
        _validate_h3_resolution(raw["resolution"])
        return raw

    @staticmethod
    def inference_capability(state: dict, *, engine: str) -> ExecutionCapability | None:
        """Describe saved cell indexing while retaining the optional runtime dependency."""
        if engine not in ("pandas", "polars"):
            return None
        H3IndexApplier.validate_inference_state(state)
        return ExecutionCapability(engine, "apply", "local", "preserve", "row")

    @apply_method
    def apply(self, X: Any, _y: Any, params: dict[str, Any]) -> Any:  # pylint: disable=arguments-differ
        """Compute the per-row cell indices on the active engine; ``y`` passes through."""
        return apply_dual_engine(
            X, params, {"polars": _h3_index_apply_polars, "pandas": _h3_index_apply_pandas}
        )


@NodeRegistry.register("H3Index", H3IndexApplier)
@node_meta(
    id="H3Index",
    name="H3 Index",
    category="Feature Engineering",
    description=(
        "Compute the Uber H3 hierarchical hexagonal grid cell index for each "
        "lat/lon pair, appended as a new string column. Requires the optional "
        "'h3' package (pip install skyulf-core[geo])."
    ),
    params={"lat_col": "", "lon_col": "", "resolution": 9, "output_column": "h3_index"},
    learns_from_data=False,
)
class H3IndexCalculator(BaseCalculator):
    """Validate the lat/lon columns and resolution for H3 cell indexing."""

    @fit_method
    def fit(self, X: Any, _y: Any, config: dict[str, Any]) -> H3IndexArtifact:  # pylint: disable=arguments-differ
        """Import ``h3`` eagerly to fail fast, then validate the columns and resolution."""
        # Fail fast with a clear message if the optional dependency is missing,
        # rather than only failing later inside Applier.apply().
        _import_h3()

        lat_col = config.get("lat_col", "")
        lon_col = config.get("lon_col", "")
        X_pd = select_then_to_pandas(X, [lat_col, lon_col])
        _validate_h3_columns(X_pd, lat_col, lon_col)

        resolution = config.get("resolution", 9)
        _validate_h3_resolution(resolution)

        output_column = config.get("output_column", "h3_index")

        return cast(
            H3IndexArtifact,
            {
                "type": "h3_index",
                "lat_col": lat_col,
                "lon_col": lon_col,
                "resolution": resolution,
                "output_column": output_column,
            },
        )
