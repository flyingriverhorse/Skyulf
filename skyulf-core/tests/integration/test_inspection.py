"""Integration tests for the Inspection nodes (DatasetProfile, DataSnapshot).

Both nodes are read-only: the Calculators collect summary statistics and the
Appliers are pure passthroughs. These tests drive the full
Calculator/Applier API on both engines (the input frame type selects the
engine), verifying artifact shapes, numeric-stats gating, snapshot row
counts, and that apply never mutates the frame.
"""

from typing import Any, cast

import numpy as np
import pandas as pd
import polars as pl
import pytest
from tests.utils.test_case_loader import TestCaseLoader

from skyulf.preprocessing.inspection import (
    DatasetProfileApplier,
    DatasetProfileCalculator,
    DataSnapshotApplier,
    DataSnapshotCalculator,
)
from skyulf.registry import NodeRegistry

_engine, _engine_scenarios, _engine_ids = TestCaseLoader(
    "preprocessing/preprocessing_inspection"
).load_with_ids()


def _make_profile_frame(engine: str) -> pd.DataFrame | pl.DataFrame:
    """Build a 5-row frame with a float, an int, and a string column.

    ``a`` has one NaN/null, ``c`` has one null, so both engines report the
    same missing counts.
    """
    data = {
        "a": [1.0, 2.0, np.nan, 4.0, 5.0],
        "b": [3, 4, 5, 6, 7],
        "c": ["x", "y", None, "z", "x"],
    }
    if engine == "polars":
        return pl.DataFrame(data)
    return pd.DataFrame(data)


def _make_text_only_frame(engine: str) -> pd.DataFrame | pl.DataFrame:
    data = {"c1": ["p", "q", None], "c2": ["r", "s", "t"]}
    if engine == "polars":
        return pl.DataFrame(data)
    return pd.DataFrame(data)


def _fit_apply(
    calculator: DatasetProfileCalculator | DataSnapshotCalculator,
    applier: DatasetProfileApplier | DataSnapshotApplier,
    frame: pd.DataFrame | pl.DataFrame,
    config: dict[str, Any] | None = None,
) -> tuple[dict[str, Any], pd.DataFrame | pl.DataFrame]:
    # The artifact TypedDicts satisfy the base ``apply`` contract but its
    # ``dict[str, Any]`` annotation does not admit a union of TypedDicts.
    params = cast(dict[str, Any], calculator.fit(frame, config or {}))
    return params, applier.apply(frame, params)


@pytest.mark.parametrize(_engine, _engine_scenarios, ids=_engine_ids)
class TestDatasetProfile:
    # ==================== Artifact shape ====================

    def test_artifact_shape_and_row_counts(self, engine: str) -> None:
        frame = _make_profile_frame(engine)
        params, _ = _fit_apply(DatasetProfileCalculator(), DatasetProfileApplier(), frame)

        assert params["type"] == "dataset_profile"
        profile = params["profile"]
        assert profile["rows"] == 5
        assert profile["columns"] == 3
        assert set(profile["dtypes"]) == {"a", "b", "c"}
        # Both engines report nulls in ``a`` (1) and ``c`` (1).
        assert profile["missing"]["a"] == 1
        assert profile["missing"]["b"] == 0
        assert profile["missing"]["c"] == 1

    def test_numeric_stats_present_and_correct(self, engine: str) -> None:
        frame = _make_profile_frame(engine)
        params, _ = _fit_apply(DatasetProfileCalculator(), DatasetProfileApplier(), frame)
        profile = params["profile"]

        stats = profile["numeric_stats"]
        # Only the numeric columns are profiled; the string column is not.
        assert set(stats) == {"a", "b"}
        # Mean of ``a`` over its 4 non-null values is 3.0 in both engines.
        assert stats["a"]["mean"] == pytest.approx(3.0)
        assert stats["b"]["mean"] == pytest.approx(5.0)

    def test_text_only_frame_omits_numeric_stats(self, engine: str) -> None:
        frame = _make_text_only_frame(engine)
        params, _ = _fit_apply(DatasetProfileCalculator(), DatasetProfileApplier(), frame)

        profile = params["profile"]
        assert "numeric_stats" not in profile
        assert profile["rows"] == 3
        assert profile["missing"]["c1"] == 1
        assert profile["missing"]["c2"] == 0

    # ==================== Passthrough apply ====================

    def test_apply_is_passthrough(self, engine: str) -> None:
        frame = _make_profile_frame(engine)
        params, result = _fit_apply(DatasetProfileCalculator(), DatasetProfileApplier(), frame)

        # The applier is a pure passthrough: same object, no new columns.
        assert result is frame
        assert list(frame.columns) == ["a", "b", "c"]


@pytest.mark.parametrize(_engine, _engine_scenarios, ids=_engine_ids)
class TestDataSnapshot:
    # ==================== Artifact shape ====================

    def test_default_n_rows_is_five(self, engine: str) -> None:
        data = {"a": [1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0], "b": [8, 7, 6, 5, 4, 3, 2, 1]}
        frame = pl.DataFrame(data) if engine == "polars" else pd.DataFrame(data)
        params, _ = _fit_apply(DataSnapshotCalculator(), DataSnapshotApplier(), frame)

        assert params["type"] == "data_snapshot"
        snapshot = params["snapshot"]
        assert len(snapshot) == 5
        assert set(snapshot[0]) == {"a", "b"}
        assert snapshot[0]["a"] == 1.0
        assert snapshot[4]["b"] == 4

    def test_custom_n_rows(self, engine: str) -> None:
        data = {"a": [1.0, 2.0, 3.0], "b": ["p", "q", "r"]}
        frame = pl.DataFrame(data) if engine == "polars" else pd.DataFrame(data)
        params, _ = _fit_apply(
            DataSnapshotCalculator(), DataSnapshotApplier(), frame, {"n_rows": 2}
        )

        snapshot = params["snapshot"]
        assert len(snapshot) == 2
        assert snapshot[0]["b"] == "p"
        assert snapshot[1]["b"] == "q"

    def test_n_rows_larger_than_frame_returns_all_rows(self, engine: str) -> None:
        frame = _make_profile_frame(engine)
        params, _ = _fit_apply(
            DataSnapshotCalculator(), DataSnapshotApplier(), frame, {"n_rows": 10}
        )

        assert len(params["snapshot"]) == 5

    # ==================== Passthrough apply ====================

    def test_apply_is_passthrough(self, engine: str) -> None:
        frame = _make_profile_frame(engine)
        params, result = _fit_apply(DataSnapshotCalculator(), DataSnapshotApplier(), frame)

        assert result is frame
        assert list(frame.columns) == ["a", "b", "c"]


class TestRegistryNames:
    def test_inspection_nodes_registered(self) -> None:
        assert NodeRegistry.get_calculator("DatasetProfile") is DatasetProfileCalculator
        assert NodeRegistry.get_applier("DatasetProfile") is DatasetProfileApplier
        assert NodeRegistry.get_calculator("DataSnapshot") is DataSnapshotCalculator
        assert NodeRegistry.get_applier("DataSnapshot") is DataSnapshotApplier

    def test_inspection_nodes_categorized(self) -> None:
        metadata = NodeRegistry.get_all_metadata()
        assert metadata["DatasetProfile"]["category"] == "Inspection"
        assert metadata["DataSnapshot"]["category"] == "Inspection"
