"""Text column profiling: stats, common words, sentiment, PII heuristics."""

import logging
import re

import polars as pl

from ..schemas import TextStats
from ._utils import VADER_AVAILABLE, _AnalyzerState

logger = logging.getLogger(__name__)

# Canonical VADER thresholds: compound >= +0.05 is positive, <= -0.05 negative.
VADER_COMPOUND_CUTOFF = 0.05


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
                .agg(pl.len().alias("count"))
                .sort("count", descending=True)
                .head(10)
            )

            common_words.extend(
                {"word": row["word"], "count": row["count"]}
                for row in word_counts.iter_rows(named=True)
            )
        except Exception as e:  # noqa: BLE001 - text stats are best-effort; logged
            logger.warning(f"Error calculating common words for {col}: {e}")

        return TextStats(
            avg_length=advanced_stats.get(f"{col}__avg_len") or 0.0,
            min_length=int(advanced_stats.get(f"{col}__min_len") or 0),
            max_length=int(advanced_stats.get(f"{col}__max_len") or 0),
            common_words=common_words,
        )

    @staticmethod
    def _classify_sentiment_counts(analyzer, texts: list) -> tuple[dict[str, int], int]:
        """Classify each text's VADER compound score into positive/neutral/negative buckets."""
        counts = {"positive": 0, "neutral": 0, "negative": 0}
        total = 0

        for text in texts:
            if not isinstance(text, str):
                continue

            compound = analyzer.polarity_scores(text)["compound"]

            if compound >= VADER_COMPOUND_CUTOFF:
                counts["positive"] += 1
            elif compound <= -VADER_COMPOUND_CUTOFF:
                counts["negative"] += 1
            else:
                counts["neutral"] += 1
            total += 1

        return counts, total

    def _analyze_sentiment(self, text_series: pl.Series) -> dict[str, float] | None:
        """Return VADER sentiment distribution ratios, or ``None`` if unavailable."""
        if not VADER_AVAILABLE:
            return None

        try:
            from vaderSentiment.vaderSentiment import (  # ty: ignore[unresolved-import]  # noqa: PLC0415 - optional text extra
                SentimentIntensityAnalyzer,
            )

            # Cap sample for runtime budget.
            sample = text_series.sample(1000, seed=42) if text_series.len() > 1000 else text_series

            texts = sample.drop_nulls().to_list()
            if not texts:
                return None

            analyzer = SentimentIntensityAnalyzer()

            counts, total = self._classify_sentiment_counts(analyzer, texts)

            if total == 0:
                return None

            return {
                "positive": counts["positive"] / total,
                "neutral": counts["neutral"] / total,
                "negative": counts["negative"] / total,
            }
        except Exception:  # noqa: BLE001 - sentiment analysis is optional (vader); None means unavailable
            return None

    def _check_pii(self, col: str) -> bool:
        # Simple heuristic on a small sample. Alerts claim "Email/Phone", so
        # both patterns must actually be checked here (previously only email
        # was implemented, making the alert message misleading for
        # phone-number-only columns).
        sample = self.df[col].drop_nulls().head(20).to_list()  # type: ignore[attr-defined]
        if not sample:
            return False

        email_pattern = r"[^@\s]+@[^@\s]+\.[^@\s]+"
        # Require positive phone evidence: a plus prefix or visual separators,
        # plus the 10-11 digits used by the common formats we support. This
        # avoids treating plain IDs and ZIP+4 values as phone numbers.
        phone_pattern = r"^\+?[\d\s().-]{7,20}$"

        def _looks_like_phone(value: str) -> bool:
            if not re.fullmatch(phone_pattern, value):
                return False
            digits = sum(c.isdigit() for c in value)
            has_phone_marker = value.startswith("+") or any(char in value for char in " ()-")
            return has_phone_marker and 10 <= digits <= 11

        if any(re.fullmatch(email_pattern, str(val)) for val in sample):
            return True

        phone_matches = sum(_looks_like_phone(str(val)) for val in sample)
        required_matches = max(1, min(2, len(sample)))
        return phone_matches >= required_matches
