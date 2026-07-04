"""Tests for skyulf.profiling._analyzer.column.ColumnMixin.

Focuses on branches of ``_get_semantic_type`` and ``_analyze_column`` that are
hard to reach from a full ``analyze()`` pipeline run: low-cardinality integer
dtype, real ``pl.Categorical`` dtype, the normality-test exception path, the
"Possible ID" / "High Cardinality" categorical alert branches, and the
DateTime/Text histogram exception paths.
"""

from typing import Any

import numpy as np
import polars as pl
import pytest

from skyulf.profiling.analyzer import EDAAnalyzer
from skyulf.profiling.schemas import CategoricalStats, NumericStats


def _basic_analyzer(df: pl.DataFrame) -> EDAAnalyzer:
    """Build a bare EDAAnalyzer instance around ``df`` without running analyze()."""
    return EDAAnalyzer(df)


def test_get_semantic_type_categorical_dtype_returns_categorical() -> None:
    """A real pl.Categorical column should map to the 'Categorical' bucket (line 50-51)."""
    df = pl.DataFrame({"c": pl.Series(["a", "b", "a", "c"]).cast(pl.Categorical)})
    analyzer = _basic_analyzer(df)
    assert analyzer._get_semantic_type(df["c"]) == "Categorical"


def test_get_semantic_type_low_cardinality_int_stays_numeric() -> None:
    """Low-cardinality ints with ratio>=0.05 fall through to 'Numeric' (line 38)."""
    # 10 unique values across 100 rows -> ratio 0.10, which is >= 0.05, so the
    # "low ratio + low n_unique" categorical shortcut should NOT trigger.
    values = list(range(10)) * 10
    df = pl.DataFrame({"i": pl.Series(values, dtype=pl.Int64)})
    analyzer = _basic_analyzer(df)
    assert analyzer._get_semantic_type(df["i"]) == "Numeric"


def test_get_semantic_type_string_low_ratio_is_categorical() -> None:
    """A Utf8 column with unique-ratio < 0.05 should be classified Categorical (lines 50-53)."""
    values = ["only-one-value"] * 100
    df = pl.DataFrame({"s": values})
    analyzer = _basic_analyzer(df)
    assert analyzer._get_semantic_type(df["s"]) == "Categorical"


def test_get_semantic_type_unhandled_dtype_falls_back_to_text() -> None:
    """An Object dtype falls through every branch to the final 'Text' fallback (line 53)."""
    df = pl.DataFrame({"o": pl.Series([object(), object(), object()], dtype=pl.Object)})
    analyzer = _basic_analyzer(df)
    assert analyzer._get_semantic_type(df["o"]) == "Text"


def test_analyze_column_normality_test_uses_kstest_for_large_samples(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """>=5000 samples should use the Kolmogorov-Smirnov path instead of Shapiro (lines 114-116).

    The installed scipy version has an incompatibility where ``kstest(..., args=(mean, std))``
    raises internally (``ndtr() takes from 1 to 2 positional arguments but 3 were given``),
    which is unrelated to the code under test here. We stub ``kstest`` with a lightweight
    fake to exercise the mean/std/test_name assignment lines deterministically.
    """
    n = 5000
    rng = np.random.default_rng(0)
    values = rng.normal(0, 1, n)
    df = pl.DataFrame({"num": values})
    analyzer = _basic_analyzer(df)

    def _fake_kstest(data: Any, dist: str, args: Any = ()) -> Any:
        return (0.01, 0.42)

    monkeypatch.setattr("scipy.stats.kstest", _fake_kstest)

    basic_stats = {"num__null": 0}
    advanced_stats = {
        "num__mean": float(np.mean(values)),
        "num__median": float(np.median(values)),
        "num__std": float(np.std(values, ddof=1)),
        "num__var": float(np.var(values, ddof=1)),
        "num__min": float(np.min(values)),
        "num__max": float(np.max(values)),
        "num__q25": float(np.quantile(values, 0.25)),
        "num__q75": float(np.quantile(values, 0.75)),
        "num__skew": 0.0,
        "num__kurt": 0.0,
        "num__zeros": 0,
        "num__negatives": int((values < 0).sum()),
    }
    semantic_types = {"num": "Numeric"}

    profile, _alerts = analyzer._analyze_column("num", basic_stats, advanced_stats, semantic_types)

    assert profile.normality_test is not None
    assert profile.normality_test.test_name == "Kolmogorov-Smirnov"
    assert profile.normality_test.is_normal is True


def test_analyze_column_normality_test_exception_is_caught(monkeypatch: pytest.MonkeyPatch) -> None:
    """A scipy.stats.shapiro failure should be swallowed (lines 124-125)."""
    n = 100
    values = np.concatenate([np.zeros(1), np.ones(n - 1)])
    df = pl.DataFrame({"num": values})
    analyzer = _basic_analyzer(df)

    def _boom(*args: Any, **kwargs: Any) -> Any:
        raise RuntimeError("shapiro exploded")

    monkeypatch.setattr("scipy.stats.shapiro", _boom)

    basic_stats = {"num__null": 0}
    advanced_stats = {
        "num__mean": float(np.mean(values)),
        "num__median": float(np.median(values)),
        "num__std": float(np.std(values, ddof=1)),
        "num__var": float(np.var(values, ddof=1)),
        "num__min": float(np.min(values)),
        "num__max": float(np.max(values)),
        "num__q25": float(np.quantile(values, 0.25)),
        "num__q75": float(np.quantile(values, 0.75)),
        "num__skew": 0.0,
        "num__kurt": 0.0,
        "num__zeros": 1,
        "num__negatives": 0,
    }
    semantic_types = {"num": "Numeric"}

    profile, _alerts = analyzer._analyze_column("num", basic_stats, advanced_stats, semantic_types)

    # The exception path prints a warning and leaves normality_test unset.
    assert profile.normality_test is None
    assert profile.numeric_stats is not None


def test_analyze_column_categorical_possible_id() -> None:
    """unique_count == row_count for a Categorical column triggers 'Possible ID' (lines 172-181)."""
    df = pl.DataFrame({"cid": [f"id-{i}" for i in range(60)]})
    analyzer = _basic_analyzer(df)
    # Force row_count to match a hand-crafted categorical_stats.unique_count.
    analyzer.row_count = 60  # type: ignore[attr-defined]

    basic_stats = {"cid__null": 0}
    advanced_stats: dict = {}
    semantic_types = {"cid": "Categorical"}

    # Patch _analyze_categorical so we control unique_count precisely without
    # needing 1000+ real distinct rows.
    def _fake_categorical(col: str, adv: dict, basic: dict) -> CategoricalStats:
        return CategoricalStats(unique_count=60, top_k=[{"value": "id-0", "count": 1}])

    analyzer._analyze_categorical = _fake_categorical  # ty: ignore[invalid-assignment]

    profile, alerts = analyzer._analyze_column("cid", basic_stats, advanced_stats, semantic_types)

    assert profile.is_unique is True
    assert any(a.type == "Possible ID" for a in alerts)


def test_analyze_column_categorical_high_cardinality() -> None:
    """unique_count > 1000 (but not all rows) triggers 'High Cardinality' (lines 182-193)."""
    df = pl.DataFrame({"cat": [f"v{i}" for i in range(2000)]})
    analyzer = _basic_analyzer(df)
    analyzer.row_count = 5000  # type: ignore[attr-defined]  # more rows than uniques

    basic_stats = {"cat__null": 0}
    advanced_stats: dict = {}
    semantic_types = {"cat": "Categorical"}

    def _fake_categorical(col: str, adv: dict, basic: dict) -> CategoricalStats:
        return CategoricalStats(unique_count=1500, top_k=[{"value": "v0", "count": 1}])

    analyzer._analyze_categorical = _fake_categorical  # ty: ignore[invalid-assignment]

    profile, alerts = analyzer._analyze_column("cat", basic_stats, advanced_stats, semantic_types)

    assert profile.is_unique is False
    assert any(a.type == "High Cardinality" for a in alerts)


def test_analyze_column_datetime_histogram_exception_is_caught(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """An exception while building the DateTime histogram should be swallowed (lines 222-223)."""
    df = pl.DataFrame(
        {
            "d": pl.datetime_range(
                start=pl.datetime(2022, 1, 1),
                end=pl.datetime(2022, 1, 10),
                interval="1d",
                eager=True,
            )
        }
    )
    analyzer = _basic_analyzer(df)

    from polars.series.datetime import DateTimeNameSpace

    def _boom_timestamp(self: Any, *args: Any, **kwargs: Any) -> Any:
        raise RuntimeError("timestamp exploded")

    monkeypatch.setattr(DateTimeNameSpace, "timestamp", _boom_timestamp)

    import datetime as dt

    basic_stats = {"d__null": 0}
    advanced_stats = {"d__min": dt.datetime(2022, 1, 1), "d__max": dt.datetime(2022, 1, 10)}
    semantic_types = {"d": "DateTime"}

    profile, _alerts = analyzer._analyze_column("d", basic_stats, advanced_stats, semantic_types)

    assert profile.histogram is None


def test_analyze_column_text_histogram_exception_is_caught(monkeypatch: pytest.MonkeyPatch) -> None:
    """An exception while building the Text histogram should be swallowed (lines 246-247)."""
    df = pl.DataFrame({"t": [f"some free text number {i} with more words" for i in range(50)]})
    analyzer = _basic_analyzer(df)

    from polars.series.string import StringNameSpace

    def _boom_len_bytes(self: Any, *args: Any, **kwargs: Any) -> Any:
        raise RuntimeError("len_bytes exploded")

    monkeypatch.setattr(StringNameSpace, "len_bytes", _boom_len_bytes)

    basic_stats = {"t__null": 0}
    advanced_stats = {"t__avg_len": 30.0, "t__min_len": 20, "t__max_len": 40}
    semantic_types = {"t": "Text"}

    profile, _alerts = analyzer._analyze_column("t", basic_stats, advanced_stats, semantic_types)

    assert profile.histogram is None
    assert profile.text_stats is not None


def test_analyze_column_high_null_and_constant_alerts() -> None:
    """Sanity check that basic High Null / Constant alerts still fire alongside new branches."""
    df = pl.DataFrame({"n": [1.0] * 20})
    analyzer = _basic_analyzer(df)

    basic_stats = {"n__null": 2}
    advanced_stats = {
        "n__mean": 1.0,
        "n__median": 1.0,
        "n__std": 0.0,
        "n__var": 0.0,
        "n__min": 1.0,
        "n__max": 1.0,
        "n__q25": 1.0,
        "n__q75": 1.0,
        "n__skew": 0.0,
        "n__kurt": 0.0,
        "n__zeros": 0,
        "n__negatives": 0,
    }
    semantic_types = {"n": "Numeric"}

    profile, alerts = analyzer._analyze_column("n", basic_stats, advanced_stats, semantic_types)

    assert profile.is_constant is True
    assert any(a.type == "High Null" for a in alerts)
    assert any(a.type == "Constant" for a in alerts)
