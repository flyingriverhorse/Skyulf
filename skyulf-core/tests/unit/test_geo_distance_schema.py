"""Distance schema preview must expose usable numeric columns before execution."""

import pandas as pd
import polars as pl
import pytest

from skyulf.core.schema import SkyulfSchema
from skyulf.preprocessing.geo import GeoDistanceApplier, GeoDistanceCalculator


@pytest.mark.parametrize("engine", ["pandas", "polars"])
@pytest.mark.parametrize("method", ["haversine", "euclidean"])
@pytest.mark.parametrize(
    "unit, output", [("km", ""), ("mi", ""), ("km", "distance"), ("mi", "lat1")]
)
def test_distance_prediction_matches_runtime_schema(engine, method, unit, output):
    """Downstream selectors must see the right name, position and float dtype, including overwrites."""
    frame = pd.DataFrame({"lat1": [0, 1], "lon1": [0, 0], "lat2": [0, 1], "lon2": [1, 1]})
    if engine == "polars":
        frame = pl.from_pandas(frame)
    schema = SkyulfSchema.from_dataframe(frame)
    config = {
        "lat1_col": "lat1",
        "lon1_col": "lon1",
        "lat2_col": "lat2",
        "lon2_col": "lon2",
        "method": method,
        "unit": unit,
        "output_column": output,
    }
    calculator = GeoDistanceCalculator()
    predicted = calculator.infer_output_schema(schema, config)
    fitted = calculator.fit(frame, config)
    transformed = GeoDistanceApplier().apply(frame, fitted)
    actual = SkyulfSchema.from_dataframe(transformed)

    assert predicted is not None
    assert predicted.columns == actual.columns
    assert {key: value.lower() for key, value in predicted.dtypes.items()} == {
        key: value.lower() for key, value in actual.dtypes.items()
    }
    assert predicted.dtypes[output or f"geo_distance_{unit}"] == "float64"
    assert schema == SkyulfSchema.from_dataframe(frame)


@pytest.mark.parametrize(
    "config", [{}, {"lat1_col": "missing", "lon1_col": "lon", "lat2_col": "lat", "lon2_col": "lon"}]
)
def test_distance_schema_is_unknown_until_coordinates_are_resolved(config):
    """An incomplete form must not promise a generated column before its inputs are available."""
    schema = SkyulfSchema.from_columns(["lat", "lon"], {"lat": "float64", "lon": "float64"})
    assert GeoDistanceCalculator().infer_output_schema(schema, config) is None
