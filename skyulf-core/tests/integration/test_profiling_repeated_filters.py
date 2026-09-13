"""Profiles from a reused analyzer must describe every filter affecting its frame."""

import polars as pl
import pytest

from skyulf.profiling.analyzer import EDAAnalyzer


@pytest.mark.parametrize(
    "later_filters", [None, [], [{"column": "unknown", "operator": "==", "value": 1}]]
)
def test_repeated_analysis_preserves_active_filters(later_filters) -> None:
    """Omitted, empty, or skipped filters must not erase the provenance of retained rows."""
    analyzer = EDAAnalyzer(pl.DataFrame({"x": range(10)}))
    filters = [{"column": "x", "operator": ">=", "value": 5}]
    first = analyzer.analyze(filters=filters)

    repeated = analyzer.analyze(filters=later_filters)

    assert repeated.row_count == first.row_count == 5
    assert repeated.sample_data == [{"x": x} for x in range(5, 10)]
    assert [item.model_dump() for item in repeated.active_filters or []] == filters


def test_repeated_analysis_accumulates_filters_without_changing_prior_profile() -> None:
    """New filters must describe cumulative state while previous profiles remain snapshots."""
    analyzer = EDAAnalyzer(pl.DataFrame({"x": range(10)}))
    lower = {"column": "x", "operator": ">=", "value": 5}
    upper = {"column": "x", "operator": "<", "value": 8}
    first = analyzer.analyze(filters=[lower])

    narrowed = analyzer.analyze(filters=[upper])

    assert narrowed.row_count == 3
    assert [item.model_dump() for item in narrowed.active_filters or []] == [lower, upper]
    assert [item.model_dump() for item in first.active_filters or []] == [lower]


def test_repeated_empty_analysis_keeps_active_filters() -> None:
    """An empty result must retain the applied filter so its cause remains visible on reuse."""
    analyzer = EDAAnalyzer(pl.DataFrame({"x": range(10)}))
    filters = [{"column": "x", "operator": ">", "value": 20}]
    first = analyzer.analyze(filters=filters)

    repeated = analyzer.analyze()

    assert repeated.row_count == first.row_count == 0
    assert [item.model_dump() for item in repeated.active_filters or []] == filters


def test_active_filter_snapshots_do_not_share_mutable_values() -> None:
    """Changing an input filter or returned profile must not rewrite analyzer provenance."""
    analyzer = EDAAnalyzer(pl.DataFrame({"x": range(10)}))
    selected = [2, 4, 6]
    first = analyzer.analyze(filters=[{"column": "x", "operator": "in", "value": selected}])
    selected.append(8)
    assert first.active_filters is not None
    first.active_filters[0].value.append(9)

    repeated = analyzer.analyze()

    assert repeated.sample_data == [{"x": value} for value in [2, 4, 6]]
    assert [item.model_dump() for item in repeated.active_filters or []] == [
        {"column": "x", "operator": "in", "value": [2, 4, 6]}
    ]
