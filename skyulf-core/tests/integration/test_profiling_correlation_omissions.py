"""Saved correlation profiles must explain columns excluded by the analysis cap."""

import polars as pl

from skyulf.profiling.analyzer import EDAAnalyzer
from skyulf.profiling.correlations import calculate_correlations
from skyulf.profiling.schemas import CorrelationMatrix, DatasetProfile


def test_profile_round_trip_preserves_correlation_cap_metadata():
    """A capped matrix must still reveal omitted numeric columns after report serialization."""
    frame = pl.DataFrame(
        {f"column_{index}": [float(value + index) for value in range(6)] for index in range(25)}
    )

    profile = EDAAnalyzer(frame).analyze()
    restored = DatasetProfile.model_validate_json(profile.model_dump_json())

    assert restored.correlations is not None
    assert restored.correlations.columns == [f"column_{index}" for index in range(20)]
    assert restored.correlations.total_columns == 25
    assert restored.correlations.omitted_columns == [f"column_{index}" for index in range(20, 25)]


def test_legacy_correlation_matrix_needs_no_omission_metadata():
    """Previously saved reports must validate without adding required wire fields."""
    matrix = CorrelationMatrix.model_validate(
        {"columns": ["a", "b"], "values": [[1, 0.5], [0.5, 1]]}
    )

    assert matrix.total_columns is None
    assert matrix.omitted_columns == []


def test_correlation_cap_metadata_keeps_constant_exclusions_distinct():
    """The original count must not shrink when constant columns leave the computed matrix."""
    frame = pl.DataFrame(
        {
            f"column_{index}": [0.0] * 4 if index == 0 else [1.0, 3.0, 2.0, 5.0]
            for index in range(25)
        }
    )

    matrix = calculate_correlations(frame.lazy(), frame.columns)

    assert matrix is not None and len(matrix.columns) == 19
    assert matrix.total_columns == 25
    assert matrix.omitted_columns == [f"column_{index}" for index in range(20, 25)]
