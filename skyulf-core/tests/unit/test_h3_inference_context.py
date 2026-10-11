"""Inspect fitted H3 choices while retaining the native optional-package boundary."""

import sys
from typing import Any

import numpy as np
import pandas as pd
import polars as pl
import pytest

h3 = pytest.importorskip("h3")

from skyulf.core.capabilities import ExecutionCapability
from skyulf.inference.preprocessing_probe import _probe_step
from skyulf.pipeline.seal import artifact_digest
from skyulf.preprocessing.geo import h3_index
from skyulf.preprocessing.inference_context import get_inference_capability
from skyulf.registry import NodeRegistry


def _frame(engine):
    """Include missing and invalid coordinates without losing stable input dtypes."""
    frame = pd.DataFrame({"lat": [40.6, 51.5, None, 91.0], "lon": [-73.7, -0.1, 0.0, 0.0]})
    return pl.from_pandas(frame) if engine == "polars" else frame


def _record(engine, **options):
    """Fit the public node so state inspection receives actual saved settings."""
    config = {"lat_col": "lat", "lon_col": "lon", **options}
    return {
        "name": "cells",
        "type": "H3Index",
        "params": config,
        "artifact": NodeRegistry.get_calculator("H3Index")().fit(_frame(engine), config),
        "applier": NodeRegistry.get_applier("H3Index")(),
    }


@pytest.mark.parametrize("engine", ["pandas", "polars"])
def test_h3_metadata_inspection_needs_no_execution_or_import(engine, monkeypatch):
    """Reading context must not refit, transform, or import the optional runtime package."""
    record = _record(engine)
    state = record["artifact"]
    before = artifact_digest(state)
    owner: Any = type(record["applier"])

    def forbidden(*args, **kwargs):
        """Only actual scoring may load the H3 runtime."""
        raise AssertionError("Unexpected execution")

    monkeypatch.setattr(h3_index, "_import_h3", forbidden)
    monkeypatch.setattr(owner, "apply", forbidden)
    monkeypatch.setattr(NodeRegistry.get_calculator("H3Index"), "fit", forbidden)
    assert get_inference_capability("H3Index", record["params"], state, engine=engine) == (
        ExecutionCapability(engine, "apply", "local", "preserve", "row")
    )
    assert owner.validate_inference_state(state) is state
    assert artifact_digest(state) == before


@pytest.mark.parametrize("engine", ["pandas", "polars"])
@pytest.mark.parametrize(
    "resolution, output_name", [(0, "cell"), (15, ""), (True, np.str_("cell"))]
)
def test_h3_saved_choices_match_cells_through_empty_requests(
    engine, resolution, output_name, monkeypatch
):
    """Saved resolution and names must be reused for null, invalid and singleton requests."""
    record = _record(engine, resolution=resolution, output_column=output_name)
    frame = _frame(engine)
    before = artifact_digest(record)

    def forbidden(*args, **kwargs):
        """Prediction must not call the calculator again."""
        raise AssertionError("Unexpected fit")

    monkeypatch.setattr(NodeRegistry.get_calculator("H3Index"), "fit", forbidden)
    detail, output = _probe_step(
        record,
        {"name": "cells", "transformer": "H3Index", "params": record["params"]},
        frame,
        engine,
        (1, 3),
        (64, 100000),
        active=True,
        project_sha=None,
    )
    assert output[output_name].to_list() == [
        h3.latlng_to_cell(40.6, -73.7, resolution),
        h3.latlng_to_cell(51.5, -0.1, resolution),
        None,
        None,
    ]
    assert detail["context"] == "row" and detail["state_validation"] == "node_owned"
    checks = {check["name"]: check for check in detail["checks"]}
    assert all(
        checks[name]["status"] == "passed"
        for name in ("full", "repeat", "chunks:1", "chunks:3", "reverse")
    )
    assert detail["status"] == "passed", detail
    assert artifact_digest(record) == before


@pytest.mark.parametrize(
    "field,value",
    [
        ("type", "other"),
        ("lat_col", ""),
        ("lon_col", None),
        ("resolution", -1),
        ("resolution", 16),
        ("resolution", 1.5),
        ("output_column", []),
        ("unexpected", 1),
    ],
)
def test_h3_malformed_saved_state_fails_inspection(field, value):
    """Changed artifact shape or invalid coordinate choices must not receive a declaration."""
    state = _record("pandas")["artifact"]
    state[field] = value
    with pytest.raises(ValueError):
        get_inference_capability("H3Index", {}, state, engine="pandas")


@pytest.mark.parametrize("engine", ["pandas", "polars"])
def test_h3_missing_runtime_is_reported_without_false_success(engine, monkeypatch):
    """A saved H3 configuration still requires its optional package during scoring."""
    record = _record(engine)
    monkeypatch.setitem(sys.modules, "h3", None)
    detail, output = _probe_step(
        record,
        {"name": "cells", "transformer": "H3Index", "params": record["params"]},
        _frame(engine),
        engine,
        (1,),
        (64, 100000),
        active=True,
        project_sha=None,
    )
    assert detail["checks"] == [
        {"name": "full", "status": "failed", "reason": "apply_error", "error_type": "ImportError"}
    ]
    assert detail["status"] == "failed" and output.equals(_frame(engine))


@pytest.mark.parametrize("engine", ["pandas", "polars"])
def test_h3_missing_coordinate_is_existing_noop(engine, monkeypatch):
    """Absent coordinates retain the existing passthrough and do not import the package."""
    record = _record(engine)
    frame = _frame(engine)
    frame = frame.drop(columns=["lat"]) if engine == "pandas" else frame.drop("lat")
    monkeypatch.setitem(sys.modules, "h3", None)
    detail, output = _probe_step(
        record,
        {"name": "cells", "transformer": "H3Index", "params": record["params"]},
        frame,
        engine,
        (1,),
        (64, 100000),
        active=True,
        project_sha=None,
    )
    assert detail["status"] == "passed" and output.equals(frame)


@pytest.mark.parametrize(
    "engine,name,expected",
    [
        ("pandas", None, None),
        ("polars", None, ""),
        ("pandas", 1, 1),
        ("pandas", ("cell", "id"), ("cell", "id")),
    ],
)
def test_h3_native_output_labels_survive_inspection(engine, name, expected):
    """Inspection must not reject column labels already accepted by real fit and apply."""
    record = _record(engine, output_column=name)
    state = record["artifact"]
    before = artifact_digest(state)
    output = record["applier"].apply(_frame(engine), state)
    empty = _frame(engine).head(0)
    empty_output = record["applier"].apply(empty, state)
    if engine == "pandas":
        pd.testing.assert_frame_equal(empty_output, output.head(0))
        pd.testing.assert_frame_equal(empty, _frame(engine).head(0))
    else:
        assert empty_output.schema == output.schema
        assert empty_output.equals(output.head(0))
    assert expected in output.columns
    assert output[expected].to_list()[0] == h3.latlng_to_cell(40.6, -73.7, 9)
    owner: Any = type(record["applier"])
    assert owner.validate_inference_state(state) is state
    capability = get_inference_capability("H3Index", record["params"], state, engine=engine)
    assert capability is not None and capability.context == "row"
    assert artifact_digest(state) == before
