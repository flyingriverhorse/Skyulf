"""Tests for skyulf.profiling._analyzer.text.TextMixin.

Covers ``_analyze_text``'s common-words exception guard and the branches of
``_analyze_sentiment`` that are hard to reach through the full pipeline:
VADER unavailability, empty/non-string text lists, the ``total == 0``
short-circuit, and the outer exception fallback.
"""

from typing import Any

import polars as pl
import pytest
from tests.utils.dataset_loader import load_sample_dataset

import skyulf.profiling._analyzer.text as text_module
from skyulf.profiling.analyzer import EDAAnalyzer


class _FakeTextSeries:
    """Minimal duck-typed stand-in for a polars Series used by _analyze_sentiment.

    _analyze_sentiment only calls .len(), .sample(), .drop_nulls() and
    .to_list() on its ``text_series`` argument, so a small stand-in lets us
    exercise mixed-type / empty-list branches that a real (homogeneous)
    polars Utf8 Series cannot represent.
    """

    def __init__(self, items: list[Any]) -> None:
        self._items = items

    def len(self) -> int:
        return len(self._items)

    def sample(self, n: int, seed: int = 42) -> "_FakeTextSeries":
        return self

    def drop_nulls(self) -> "_FakeTextSeries":
        return _FakeTextSeries([i for i in self._items if i is not None])

    def to_list(self) -> list[Any]:
        return self._items


def _text_analyzer(texts: list[str]) -> EDAAnalyzer:
    df = pl.DataFrame({"t": texts})
    return EDAAnalyzer(df)


def test_analyze_text_common_words_exception_is_caught(monkeypatch: pytest.MonkeyPatch) -> None:
    """A failure while tokenizing common words should be swallowed (lines 38-39)."""
    from polars.expr.string import ExprStringNameSpace

    def _boom(self: Any, *args: Any, **kwargs: Any) -> Any:
        raise RuntimeError("to_lowercase exploded")

    monkeypatch.setattr(ExprStringNameSpace, "to_lowercase", _boom)

    analyzer = _text_analyzer([f"hello world number {i}" for i in range(20)])
    stats = analyzer._analyze_text("t", {"t__avg_len": 20.0, "t__min_len": 15, "t__max_len": 25})

    assert stats.common_words == []
    assert stats.avg_length == 20.0


def test_analyze_sentiment_returns_none_when_vader_unavailable(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """When VADER_AVAILABLE is False, sentiment analysis is skipped entirely (line 51)."""
    monkeypatch.setattr(text_module, "VADER_AVAILABLE", False)
    analyzer = _text_analyzer(["good", "bad"])

    result = analyzer._analyze_sentiment(analyzer.df["t"])

    assert result is None


def test_analyze_sentiment_returns_none_for_empty_texts() -> None:
    """An all-null / empty text series returns None after drop_nulls() (line 63)."""
    analyzer = _text_analyzer(["placeholder"])
    empty_series = _FakeTextSeries([])

    result = analyzer._analyze_sentiment(empty_series)  # ty: ignore[invalid-argument-type]

    assert result is None


def test_analyze_sentiment_skips_non_string_items() -> None:
    """Non-string items in the texts list are skipped but strings are still scored (line 72)."""
    pytest.importorskip("vaderSentiment")
    analyzer = _text_analyzer(["placeholder"])
    mixed_series = _FakeTextSeries(["I love this product", 12345, "This is terrible"])

    result = analyzer._analyze_sentiment(mixed_series)  # ty: ignore[invalid-argument-type]

    assert result is not None
    assert abs(sum(result.values()) - 1.0) < 1e-9


def test_analyze_sentiment_returns_none_when_total_stays_zero() -> None:
    """If every item is non-string, `total` never increments and None is returned (line 85)."""
    analyzer = _text_analyzer(["placeholder"])
    all_non_string = _FakeTextSeries([1, 2, 3.5, True])

    result = analyzer._analyze_sentiment(all_non_string)  # ty: ignore[invalid-argument-type]

    assert result is None


def test_analyze_sentiment_outer_exception_returns_none(monkeypatch: pytest.MonkeyPatch) -> None:
    """A failure inside VADER's polarity_scores should be caught and return None (lines 92-93)."""
    pytest.importorskip("vaderSentiment")
    import vaderSentiment.vaderSentiment as vader_mod

    class _BoomAnalyzer:
        def polarity_scores(self, text: str) -> dict:
            raise RuntimeError("vader exploded")

    monkeypatch.setattr(vader_mod, "SentimentIntensityAnalyzer", _BoomAnalyzer)

    analyzer = _text_analyzer(["good text", "bad text"])
    result = analyzer._analyze_sentiment(analyzer.df["t"])

    assert result is None


class TestRealShapedDataset:
    """Integration-style check against the checked-in ``customers.csv`` sample,
    which has a repeated, missing-value ``city`` column — closer to
    production data than the small synthetic text lists used elsewhere in
    this file.
    """

    def test_analyze_text_common_words_on_city_column(self) -> None:
        df = load_sample_dataset("customers", engine="polars")
        analyzer = EDAAnalyzer(df)
        stats = analyzer._analyze_text(
            "city", {"city__avg_len": 8.0, "city__min_len": 7, "city__max_len": 11}
        )

        word_counts = {w["word"]: w["count"] for w in stats.common_words}
        # "Chicago"/"New York" each appear 4 times, "Houston"/"Los Angeles" 3 times;
        # one row has a missing city that must not surface as a word.
        assert word_counts["chicago"] == 4
        assert word_counts["york"] == 4
        assert word_counts["houston"] == 3
        assert stats.avg_length == 8.0
