"""Repeated profiling must keep excluded data outside samples and frame statistics."""

import polars as pl
import pytest
from polars.testing import assert_frame_equal

from skyulf.profiling.analyzer import EDAAnalyzer


@pytest.mark.parametrize("later_exclusions", [["private"], None, ["unknown"]])
def test_repeated_analysis_preserves_active_column_exclusions(later_exclusions) -> None:
    """Previously excluded values must never reappear when an analyzer is reused."""
    frame = pl.DataFrame({"x": [1, 1, 2, 2], "private": [None, None, "hidden-b", "hidden-c"]})
    analyzer = EDAAnalyzer(frame)
    first = analyzer.analyze(exclude_cols=["private"])

    repeated = analyzer.analyze(exclude_cols=later_exclusions)

    assert repeated.sample_data == first.sample_data == [{"x": value} for value in [1, 1, 2, 2]]
    assert repeated.missing_cells_percentage == first.missing_cells_percentage == 0
    assert repeated.duplicate_rows == first.duplicate_rows == 4
    assert repeated.excluded_columns == first.excluded_columns == ["private"]
    assert list(repeated.columns) == ["x"]
    assert repeated.column_count == 1
    assert repeated.row_count == 4
    assert_frame_equal(analyzer.df, frame)
