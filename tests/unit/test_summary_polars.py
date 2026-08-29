"""Node-summary coverage for the polars engine.

`build_summary` was pandas-only: polars-engine jobs produced ``None`` for
every frame-output node, so canvas node cards and the pipeline diagram's
detail lines were silently empty while trainer summaries (metrics-based)
still rendered. These tests pin the engine-agnostic behavior.
"""

import pandas as pd
import polars as pl

from backend.ml_pipeline._execution.summary import (
    _dtype_breakdown,
    _shape_of,
    build_summary,
)
from skyulf.data.dataset import SplitDataset


def _mixed_pl() -> pl.DataFrame:
    return pl.DataFrame(
        {
            "age": [1, 2, 3],
            "fare": [1.0, 2.0, 3.0],
            "city": ["a", "b", "c"],
            "flag": [True, False, True],
        }
    )


class TestShapeOf:
    def test_polars_frame(self):
        assert _shape_of(_mixed_pl()) == (3, 4)

    def test_pandas_frame_still_works(self):
        assert _shape_of(pd.DataFrame({"a": [1, 2]})) == (2, 1)

    def test_non_frame_is_none(self):
        assert _shape_of("nope") is None

    def test_tuple_with_polars_frame_counts_y_column(self):
        # A (X, y) tuple whose X is a polars frame must count y as one extra column.
        assert _shape_of((pl.DataFrame({"a": [1, 2]}), [0, 1])) == (2, 2)


class TestDtypeBreakdown:
    def test_mixed_polars_frame(self):
        assert _dtype_breakdown(_mixed_pl()) == "2 num · 1 cat · 1 bool"

    def test_uniform_polars_frame_returns_none(self):
        assert _dtype_breakdown(pl.DataFrame({"a": [1], "b": [2.0]})) is None

    def test_datetime_polars_frame(self):
        df = pl.DataFrame(
            {"a": [1, 2], "when": [pd.Timestamp("2026-01-01"), pd.Timestamp("2026-01-02")]}
        )
        assert _dtype_breakdown(df) == "1 num · 1 dt"

    def test_categorical_and_enum_count_as_cat(self):
        df = pl.DataFrame(
            {
                "n": [1, 2],
                "c": pl.Series(["a", "b"], dtype=pl.Categorical),
                "e": pl.Series(["x", "y"], dtype=pl.Enum(["x", "y"])),
            }
        )
        assert _dtype_breakdown(df) == "1 num · 2 cat"


class TestBuildSummaryPolars:
    def test_same_shape_scaler_summary(self):
        text = build_summary(
            step_type="standard_scaler",
            output=_mixed_pl(),
            metrics={},
            input_shape=(3, 4),
            params={},
        )
        assert text == "standard · 4 cols"

    def test_row_drop_delta_summary(self):
        text = build_summary(
            step_type="DropMissingRows",
            output=pl.DataFrame({"a": [1, 2]}),
            metrics={},
            input_shape=(5, 1),
            params={},
        )
        assert text is not None
        assert "−3 rows (60.0%)" in text

    def test_loader_summary_includes_dtype_mix(self):
        text = build_summary(step_type="data_loader", output=_mixed_pl(), metrics={})
        assert text == "3 rows × 4 cols (2 num · 1 cat · 1 bool)"

    def test_split_summary_with_polars_frames(self):
        ds = SplitDataset(
            train=pl.DataFrame({"a": list(range(7))}),
            test=pl.DataFrame({"a": list(range(3))}),
        )
        text = build_summary(step_type="train_test_splitter", output=ds, metrics={})
        assert text is not None
        assert "7 / 3" in text
        assert "× 1 cols" in text
