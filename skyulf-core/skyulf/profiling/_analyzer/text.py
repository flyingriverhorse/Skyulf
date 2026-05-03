"""Text column profiling: stats, common words, sentiment, PII heuristics."""

import re
from typing import Dict, Optional

import polars as pl

from ..schemas import TextStats
from ._utils import VADER_AVAILABLE, _AnalyzerState


class TextMixin(_AnalyzerState):
    """Text helpers for :class:`EDAAnalyzer`."""

    def _analyze_text(self, col: str, advanced_stats: dict) -> TextStats:
        common_words = []
        try:
            # Cap at 1000 rows — common-word stats are illustrative, not statistical.
            sample_text = self.df.select(col).head(1000)  # type: ignore[attr-defined]
            words = sample_text.select(
                pl.col(col)
                .str.to_lowercase()
                .str.replace_all(r"[^\w\s]", "")
                .str.split(" ")
                .explode()
                .alias("word")
            ).filter(pl.col("word") != "")

            word_counts = (
                words.group_by("word")
                .agg(pl.count().alias("count"))
                .sort("count", descending=True)
                .head(10)
            )

            for row in word_counts.iter_rows(named=True):
                common_words.append({"word": row["word"], "count": row["count"]})
        except Exception as e:
            print(f"Error calculating common words for {col}: {e}")

        return TextStats(
            avg_length=advanced_stats.get(f"{col}__avg_len") or 0.0,
            min_length=int(advanced_stats.get(f"{col}__min_len") or 0),
            max_length=int(advanced_stats.get(f"{col}__max_len") or 0),
            common_words=common_words,
        )

    def _analyze_sentiment(self, text_series: pl.Series) -> Optional[Dict[str, float]]:
        """Return VADER sentiment distribution ratios, or ``None`` if unavailable."""
        if not VADER_AVAILABLE:
            return None

        try:
            from vaderSentiment.vaderSentiment import (  # ty: ignore[unresolved-import]
                SentimentIntensityAnalyzer,
            )

            # Cap sample for runtime budget.
            sample = text_series.sample(1000, seed=42) if text_series.len() > 1000 else text_series

            texts = sample.drop_nulls().to_list()
            if not texts:
                return None

            analyzer = SentimentIntensityAnalyzer()

            counts = {"positive": 0, "neutral": 0, "negative": 0}
            total = 0

            for text in texts:
                if not isinstance(text, str):
                    continue

                compound = analyzer.polarity_scores(text)["compound"]

                if compound >= 0.05:
                    counts["positive"] += 1
                elif compound <= -0.05:
                    counts["negative"] += 1
                else:
                    counts["neutral"] += 1
                total += 1

            if total == 0:
                return None

            return {
                "positive": counts["positive"] / total,
                "neutral": counts["neutral"] / total,
                "negative": counts["negative"] / total,
            }
        except Exception:
            return None

    def _check_pii(self, col: str) -> bool:
        # Simple heuristic on a small sample. Email-only for now.
        sample = self.df[col].drop_nulls().head(20).to_list()  # type: ignore[attr-defined]
        email_pattern = r"[^@]+@[^@]+\.[^@]+"

        for val in sample:
            if re.match(email_pattern, str(val)):
                return True
        return False
