"""Public regressions for bounded targets, unsupported values and exact drill-down."""

import json
import operator
from functools import partial
from pathlib import Path

import numpy as np
import polars as pl
import pytest

import skyulf.profiling._analyzer.decomposition as decomposition_module
import skyulf.profiling.analyzer as analyzer_module
import skyulf.profiling.correlations as correlations_module
import skyulf.profiling.visualizer as visualizer_module
from skyulf.profiling.analyzer import EDAAnalyzer
from skyulf.profiling.schemas import DatasetProfile


@pytest.fixture(scope="module", autouse=True)
def verify_source_imports():
    """An isolated test run must exercise the package beside these tests."""
    source_root = Path(__file__).resolve().parents[3] / "skyulf"
    for module in (analyzer_module, correlations_module, decomposition_module, visualizer_module):
        assert Path(module.__file__).resolve().is_relative_to(source_root)


@pytest.mark.parametrize("feature_count", [19, 20, 23])
def test_bounded_target_matrix_reserves_target_and_reports_actual_omissions(feature_count):
    """A wide target matrix must retain its target without changing the feature-only cap."""
    rng = np.random.default_rng(713)
    features = {f"feature_{index}": rng.normal(size=40) for index in range(feature_count)}
    frame = pl.DataFrame(features | {"target": 2 * features["feature_0"] + rng.normal(size=40)})

    profile = EDAAnalyzer(frame).analyze(target_col="target", task_type="Regression")
    restored = DatasetProfile.model_validate_json(profile.model_dump_json())

    matrix = restored.correlations_with_target
    assert matrix is not None
    assert matrix.columns == list(features)[:19] + ["target"]
    assert matrix.total_columns == feature_count + 1
    assert matrix.omitted_columns == list(features)[19:]
    assert matrix.values[0][-1] == pytest.approx(
        np.corrcoef(frame["feature_0"], frame["target"])[0, 1]
    )
    assert restored.correlations is not None
    assert restored.correlations.columns == list(features)[:20]
    assert restored.target_correlations is not None
    assert set(restored.target_correlations) == set(features)


@pytest.mark.parametrize("object_only", [False, True], ids=["mixed", "object-only"])
def test_object_columns_do_not_suppress_supported_profiling(object_only):
    """Unsupported Python objects must retain missing counts without invented cardinality."""
    payload = pl.Series("payload", [{"key": 1}, None, {"key": 1}, {"key": 2}], dtype=pl.Object)
    frame = pl.DataFrame({"payload": payload})
    if not object_only:
        frame = frame.with_columns(pl.Series("signal", [1.0, 2.0, 4.0, 9.0]))

    profile = EDAAnalyzer(frame).analyze()
    restored = DatasetProfile.model_validate_json(profile.model_dump_json())

    assert restored.row_count == 4
    assert restored.column_count == frame.width
    assert restored.columns["payload"].dtype == "Unknown"
    assert restored.columns["payload"].missing_count == 1
    assert restored.columns["payload"].categorical_stats is None
    assert restored.columns["payload"].is_constant is False
    assert restored.columns["payload"].is_unique is False
    assert any(
        alert.type == "Unsupported Type" and alert.column == "payload" for alert in restored.alerts
    )
    if object_only:
        assert restored.duplicate_rows is None
        assert any(alert.type == "Duplicate Count Unavailable" for alert in restored.alerts)
    if not object_only:
        assert restored.columns["signal"].numeric_stats is not None
        assert restored.columns["signal"].numeric_stats.mean == 4.0
    assert restored.sample_data == frame.to_dicts()


def test_hashable_object_columns_preserve_native_duplicate_count():
    """An Object dtype alone must not invalidate a successful native equality calculation."""
    values = ["".join(["repeat", "ed"]) for _ in range(2)] + ["other"]
    frame = pl.DataFrame({"payload": pl.Series(values, dtype=pl.Object)})

    profile = EDAAnalyzer(frame).analyze()

    assert profile.duplicate_rows == 2
    assert not any(alert.type == "Duplicate Count Unavailable" for alert in profile.alerts)


def test_supported_columns_keep_duplicate_and_cardinality_statistics():
    """Unsupported-type guards must leave ordinary row equality and categorical counts intact."""
    frame = pl.DataFrame({"signal": [1.0, 1.0, 3.0, 3.0], "label": ["a", "a", "b", "b"]})

    profile = EDAAnalyzer(frame).analyze()

    assert profile.duplicate_rows == 4
    assert profile.columns["label"].categorical_stats is not None
    assert profile.columns["label"].categorical_stats.unique_count == 2


@pytest.mark.parametrize("filtered", [False, True])
def test_excluding_all_columns_preserves_filtered_row_metadata(filtered):
    """An empty column selection must not erase row counts or reveal excluded samples."""
    frame = pl.DataFrame({"signal": [1.0, 2.0, 3.0], "private": ["a", "b", "c"]})
    analyzer = EDAAnalyzer(frame)
    filters = [{"column": "signal", "operator": ">", "value": 1}] if filtered else None

    profile = analyzer.analyze(exclude_cols=frame.columns, filters=filters, target_col="signal")
    repeated = analyzer.analyze()

    assert profile.row_count == repeated.row_count == (2 if filtered else 3)
    assert profile.column_count == repeated.column_count == 0
    assert profile.columns == repeated.columns == {}
    assert profile.sample_data == repeated.sample_data == []
    assert profile.excluded_columns == repeated.excluded_columns == frame.columns
    assert profile.missing_cells_percentage == 0.0
    assert profile.duplicate_rows is None
    assert profile.correlations is None and profile.correlations_with_target is None
    assert any(alert.type == "No Columns" for alert in profile.alerts)
    assert not any(alert.type == "Empty Data" for alert in profile.alerts)
    assert len(profile.active_filters or []) == int(filtered)


def test_empty_selection_summary_reports_unavailable_duplicate_count(capsys, monkeypatch):
    """A terminal summary must not present an unknown duplicate count as zero."""
    console_module = pytest.importorskip("rich.console")
    monkeypatch.setattr(
        console_module,
        "Console",
        partial(console_module.Console, force_jupyter=False, color_system=None),
    )
    profile = EDAAnalyzer(pl.DataFrame({"private": [1, 2]})).analyze(exclude_cols=["private"])

    visualizer_module.EDAVisualizer(profile).summary()

    output = capsys.readouterr().out
    assert "Duplicate Rows" in output
    assert "Unavailable" in output


def test_excluding_some_columns_retains_selected_data_control():
    """An empty-selection guard must not change ordinary exclusion reporting."""
    frame = pl.DataFrame({"signal": [1.0, 2.0, 3.0], "private": ["a", "b", "c"]})

    profile = EDAAnalyzer(frame).analyze(exclude_cols=["private"])

    assert profile.row_count == 3 and profile.column_count == 1
    assert profile.sample_data == [{"signal": value} for value in [1.0, 2.0, 3.0]]
    assert not any(alert.type == "No Columns" for alert in profile.alerts)


@pytest.mark.parametrize(
    "dtype,selected",
    [
        pytest.param(pl.Int64, 2**53 + 1, id="positive"),
        pytest.param(pl.Int64, -(2**53 + 1), id="negative"),
        pytest.param(pl.Int64, 2**63 - 1, id="int64-limit"),
        pytest.param(pl.UInt64, 2**64 - 1, id="uint64-limit"),
    ],
)
@pytest.mark.parametrize("suffix", ["", ".0", "e0"], ids=["bucket", "decimal", "scientific"])
def test_decomposition_large_integer_filters_preserve_exact_bucket(dtype, selected, suffix):
    """JSON drill-down must not round a large integer into a neighboring group."""
    frame = pl.DataFrame(
        {"key": pl.Series([selected - 2, selected - 1, selected], dtype=dtype), "weight": [3, 5, 7]}
    )
    analyzer = EDAAnalyzer(frame)
    buckets = json.loads(json.dumps(analyzer.get_decomposition_split("weight", "sum", "key", [])))
    selected_bucket = next(bucket for bucket in buckets if bucket["filter_value"] == str(selected))

    result = analyzer.get_decomposition_split(
        "weight",
        "sum",
        None,
        [{"column": "key", "operator": "==", "value": selected_bucket["filter_value"] + suffix}],
    )

    assert result == [{"name": "Total", "value": 7, "ratio": 1.0}]


@pytest.mark.parametrize(
    "symbol,compare",
    [
        ("==", operator.eq),
        ("!=", operator.ne),
        (">", operator.gt),
        ("<", operator.lt),
        (">=", operator.ge),
        ("<=", operator.le),
    ],
)
def test_large_integer_comparison_boundaries_match_python(symbol, compare):
    """Exact coercion must preserve every numeric comparison boundary, not just equality."""
    selected = 2**53 + 1
    values = [selected - 1, selected, selected + 1]
    weights = [3, 5, 7]
    analyzer = EDAAnalyzer(pl.DataFrame({"key": values, "weight": weights}))

    result = analyzer.get_decomposition_split(
        "weight",
        "sum",
        None,
        [{"column": "key", "operator": symbol, "value": str(selected)}],
    )

    expected = sum(
        weight for key, weight in zip(values, weights, strict=True) if compare(key, selected)
    )
    assert result == [{"name": "Total", "value": expected, "ratio": 1.0}]


@pytest.mark.parametrize(
    "value,expected", [("1", 3), ("1.0", 3), ("1.9", 3), ("1e0", 3), ("invalid", 0)]
)
def test_integer_filter_legacy_coercion_controls(value, expected):
    """Exact large-label handling must retain documented decimal forms and legacy fallback."""
    analyzer = EDAAnalyzer(pl.DataFrame({"key": [1, 2], "weight": [3, 5]}))

    result = analyzer.get_decomposition_split(
        "weight",
        "sum",
        None,
        [{"column": "key", "operator": "==", "value": value}],
    )

    assert result == [{"name": "Total", "value": expected, "ratio": 1.0}]
