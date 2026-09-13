"""Exercise Canvas bounds and distance parameters through backend graph execution."""

import math
from copy import deepcopy
from pathlib import Path
from unittest.mock import patch

import numpy as np
import pandas as pd
import polars as pl
import pytest

from backend.config import get_settings
from backend.data.catalog import FileSystemCatalog
from backend.ml_pipeline._execution._schema_graph import predict_schemas
from backend.ml_pipeline._execution._schema_validator import find_broken_references
from backend.ml_pipeline._execution.engine import PipelineEngine
from backend.ml_pipeline._execution.schemas import NodeConfig, PipelineConfig
from backend.ml_pipeline._internal._routers.preview import (
    _extract_preview,
    _run_preview_sub_pipelines,
)
from backend.ml_pipeline.artifacts.local import LocalArtifactStore
from skyulf.modeling.base import extract_xy
from skyulf.pipeline.seal import artifact_digest
from skyulf.preprocessing import SkyulfSchema
from skyulf.preprocessing.geo.distance import GeoDistanceCalculator
from skyulf.preprocessing.outliers.manual_bounds import ManualBoundsCalculator


@pytest.fixture(params=["pandas", "polars"])
def frame_engine(request, monkeypatch):
    """Each graph must use the requested engine at both ingestion and Core execution."""
    monkeypatch.setenv("SKYULF_ENGINE", request.param)
    monkeypatch.setattr(get_settings(), "SKYULF_ENGINE", request.param)
    return request.param


def _rows():
    """Give every row an identifiable target and independently checkable coordinates."""
    return pd.DataFrame(
        {
            "row_id": range(9),
            "age": [-1.0, 0.0, 17.0, 18.0, 40.0, 65.0, 66.0, None, np.nan],
            "lat1": [0.0, 30.0, 0.0, 0.0, 60.0, 0.0, 80.0, 0.0, 0.0],
            "lon1": [0.0] * 9,
            "lat2": [0.0, 30.0, 0.0, 0.0, 60.0, 0.0, 80.0, 0.0, 0.0],
            "lon2": [1.0, 1.0, 1.0, 1.0, 1.0, 0.0, 1.0, 2.0, 1.0],
            "target": [row_id * 10 + 7 for row_id in range(9)],
        }
    )


def _geo_params(method="haversine", unit="km", output_column=""):
    """Use the exact keys and blank automatic output name serialized by Canvas."""
    return {
        "lat1_col": "lat1",
        "lon1_col": "lon1",
        "lat2_col": "lat2",
        "lon2_col": "lon2",
        "method": method,
        "unit": unit,
        "output_column": output_column,
    }


def _native(value):
    """Unwrap public engine adapters before checking concrete frame types."""
    return value.to_native() if hasattr(value, "to_native") else value


def _pandas(value):
    """Compare values without depending on engine-specific row indexing."""
    value = _native(value)
    return value.to_pandas() if isinstance(value, pl.DataFrame) else value


def _run_graph(tmp_path, nodes):
    """Load an isolated real CSV and execute the requested backend graph."""
    source = tmp_path / "coordinates.csv"
    _rows().to_csv(source, index=False)
    config = PipelineConfig(
        pipeline_id="canvas-bounds-geo",
        nodes=[
            NodeConfig("source", "data_loader", params={"source": "csv", "path": str(source)}),
            *nodes,
        ],
    )
    store = LocalArtifactStore(str(tmp_path / "artifacts"))
    engine = PipelineEngine(store, catalog=FileSystemCatalog(str(tmp_path)))
    result = engine.run(deepcopy(config), inspect_all=True)
    assert result.status == "success", result.node_results
    assert all(node.status == "success" for node in result.node_results.values())
    return store, config


@pytest.mark.parametrize("field", ["bounds", "lat1_col", "lon1_col", "lat2_col", "lon2_col"])
def test_canvas_bounds_geo_reports_missing_upstream_columns(field):
    """Removed upstream columns must produce an actionable Canvas issue before execution."""
    params = (
        {"bounds": {"removed_column": {"lower": 0}}}
        if field == "bounds"
        else {**_geo_params(), field: "removed_column"}
    )
    config = PipelineConfig(
        pipeline_id="canvas-stale-column",
        nodes=[
            NodeConfig("source", "data_loader"),
            NodeConfig(
                "candidate",
                "ManualBounds" if field == "bounds" else "GeoDistance",
                params=params,
                inputs=["source"],
            ),
        ],
    )
    predicted = predict_schemas(config, {"source": SkyulfSchema.from_dataframe(_rows())})
    assert find_broken_references(config, predicted) == [
        {
            "node_id": "candidate",
            "field": field,
            "column": "removed_column",
            "upstream_node_id": "source",
        }
    ]


@pytest.mark.parametrize(
    ("bounds", "expected_ids"),
    [
        ({"lower": 18, "upper": 65}, [3, 4, 5, 7, 8]),
        ({"lower": 0}, [1, 2, 3, 4, 5, 6, 7, 8]),
        ({"upper": 0}, [0, 1, 7, 8]),
        ({"lower": 0, "upper": 0}, [1, 7, 8]),
    ],
    ids=["inclusive", "lower-zero", "upper-zero", "equal-zero"],
)
def test_canvas_manual_bounds_preserves_target_alignment(
    tmp_path, frame_engine, bounds, expected_ids
):
    """Inclusive and one-sided zero limits must filter X and y together and retain missing ages."""
    store, _config = _run_graph(
        tmp_path,
        [
            NodeConfig(
                "target",
                "feature_target_split",
                params={"target_column": "target"},
                inputs=["source"],
            ),
            NodeConfig(
                "bounds", "ManualBounds", params={"bounds": {"age": bounds}}, inputs=["target"]
            ),
        ],
    )
    features, labels = extract_xy(store.load("bounds"), "target")
    assert isinstance(_native(features), pd.DataFrame if frame_engine == "pandas" else pl.DataFrame)
    actual = _pandas(features).reset_index(drop=True)
    expected = _rows().iloc[expected_ids].drop(columns="target").reset_index(drop=True)
    pd.testing.assert_frame_equal(actual, expected, check_dtype=False)
    np.testing.assert_array_equal(np.asarray(labels).ravel(), [i * 10 + 7 for i in expected_ids])
    assert store.load("exec_bounds_pipeline").fitted_steps[0]["artifact"]["bounds"] == {
        "age": bounds
    }


@pytest.mark.parametrize("method", ["haversine", "euclidean"])
@pytest.mark.parametrize(
    ("unit", "output_column"), [("km", ""), ("mi", ""), ("mi", "route_distance")]
)
def test_canvas_geo_distance_units_names_and_input_columns(
    tmp_path, frame_engine, method, unit, output_column
):
    """Distances must use geographic units while preserving coordinate columns and targets."""
    store, _config = _run_graph(
        tmp_path,
        [
            NodeConfig(
                "distance",
                "GeoDistance",
                params=_geo_params(method, unit, output_column),
                inputs=["source"],
            )
        ],
    )
    output = store.load("distance")
    assert isinstance(_native(output), pd.DataFrame if frame_engine == "pandas" else pl.DataFrame)
    actual = _pandas(output).reset_index(drop=True)
    original = _pandas(store.load("source")).reset_index(drop=True)
    name = output_column or f"geo_distance_{unit}"
    unit_factor = 1.0 if unit == "km" else 0.6213711922
    equator_degree = math.pi * 6371.0088 / 180
    # At latitude 60, the equirectangular approximation halves east-west distances;
    # the spherical law of cosines provides an independent great-circle reference.
    latitude_60 = (
        6371.0088 * math.acos(0.75 + 0.25 * math.cos(math.pi / 180))
        if method == "haversine"
        else equator_degree / 2
    )
    np.testing.assert_allclose(
        actual.loc[[3, 4, 5, 7], name],
        np.array([equator_degree, latitude_60, 0.0, 2 * equator_degree]) * unit_factor,
        rtol=1e-10,
        atol=1e-10,
    )
    pd.testing.assert_frame_equal(actual[list(original.columns)], original)
    assert list(actual.columns) == [*original.columns, name]


def test_canvas_bounds_geo_joint_schema_preview_and_runtime(tmp_path, frame_engine):
    """Preview receipts and predicted columns must agree with filtered train/test artifacts."""
    store, config = _run_graph(
        tmp_path,
        [
            NodeConfig(
                "split",
                "TrainTestSplitter",
                params={"target_column": "target", "test_size": 0.33, "random_state": 42},
                inputs=["source"],
            ),
            NodeConfig(
                "bounds",
                "ManualBounds",
                params={"bounds": {"age": {"lower": 18, "upper": 65}}},
                inputs=["split"],
            ),
            NodeConfig("distance", "GeoDistance", params=_geo_params(), inputs=["bounds"]),
        ],
    )
    predicted = predict_schemas(
        config, {"source": SkyulfSchema.from_dataframe(store.load("source"))}
    )
    assert predicted["bounds"] is not None
    assert predicted["distance"] is not None
    all_ids = []
    expected_totals = {}
    for split in ("train", "test"):
        features, labels = extract_xy(getattr(store.load("distance"), split), "target")
        frame = _pandas(features)
        ids = frame["row_id"].tolist()
        all_ids.extend(ids)
        np.testing.assert_array_equal(np.asarray(labels).ravel(), [i * 10 + 7 for i in ids])
        # Canvas retains the target in its schema even when the runtime carries y separately.
        predicted_features = [col for col in predicted["distance"].column_list() if col != "target"]
        assert predicted_features == list(frame.columns)
        assert predicted["distance"].dtypes["geo_distance_km"] == "float64"
        expected_totals.update({f"{split}_X": len(ids), f"{split}_y": len(ids)})
    assert sorted(all_ids) == [3, 4, 5, 7, 8]

    preview_store = LocalArtifactStore(str(tmp_path / "preview"))
    inspections = []
    with patch(
        "backend.ml_pipeline._internal._routers.preview.create_catalog_from_options",
        return_value=FileSystemCatalog(str(tmp_path)),
    ):
        results = _run_preview_sub_pipelines(
            deepcopy(config),
            deepcopy(config.nodes),
            config.nodes,
            None,
            None,
            preview_store,
            node_inspections=inspections,
            inspect_all=True,
        )
    assert results and all(result.status == "success" for _, _, result in results)
    _, totals, _frame = _extract_preview(preview_store, "distance")
    assert totals == expected_totals
    inspected = [item for item in inspections if item.node_id == "distance"]
    assert inspected and all(item.output.status == "available" for item in inspected)
    assert all(item.path_id and item.path_label for item in inspected)
    assert all(
        table.row_count == expected_totals[f"{table.split}_X"]
        for item in inspected
        for table in item.output.tables
    )


def test_saved_canvas_nodes_replay_without_refitting(tmp_path, frame_engine):
    """Reloaded artifacts must retain fixed bounds, unit, and name on fresh input rows."""
    store, _config = _run_graph(
        tmp_path,
        [
            NodeConfig(
                "bounds",
                "ManualBounds",
                params={"bounds": {"age": {"lower": 18, "upper": 65}}},
                inputs=["source"],
            ),
            NodeConfig(
                "distance",
                "GeoDistance",
                params=_geo_params(unit="mi", output_column="trip_miles"),
                inputs=["bounds"],
            ),
        ],
    )
    reloaded = LocalArtifactStore(str(Path(store.base_path)))
    bounds = reloaded.load("exec_bounds_pipeline")
    distance = reloaded.load("exec_distance_pipeline")
    before = artifact_digest([bounds, distance])
    fresh_rows = {
        "row_id": [20, 21, 22, 23, 24],
        "age": [17.0, 18.0, 65.0, None, np.nan],
        "lat1": [0.0] * 5,
        "lon1": [0.0] * 5,
        "lat2": [0.0] * 5,
        "lon2": [1.0, 2.0, 3.0, 4.0, 5.0],
    }
    fresh = pd.DataFrame(fresh_rows)
    labels = pd.Series([207, 217, 227, 237, 247], name="target")
    if frame_engine == "polars":
        fresh = pl.DataFrame(fresh_rows)
        labels = pl.from_pandas(labels)
    with (
        patch.object(ManualBoundsCalculator, "fit", side_effect=AssertionError("unexpected refit")),
        patch.object(GeoDistanceCalculator, "fit", side_effect=AssertionError("unexpected refit")),
    ):
        features, target = distance.transform(bounds.transform((fresh, labels)))
    actual = _pandas(features)
    assert actual["row_id"].tolist() == [21, 22, 23, 24]
    np.testing.assert_array_equal(np.asarray(target).ravel(), [217, 227, 237, 247])
    np.testing.assert_allclose(
        actual["trip_miles"], np.array([2, 3, 4, 5]) * math.pi * 6371.0088 / 180 * 0.6213711922
    )
    if frame_engine == "polars":
        assert _native(features)["age"].null_count() == 1
        assert _native(features)["age"].is_nan().sum() == 1
    assert artifact_digest([bounds, distance]) == before
