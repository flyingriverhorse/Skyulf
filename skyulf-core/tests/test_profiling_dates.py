"""Tests for skyulf.profiling._analyzer.dates.DatesMixin."""

import polars as pl
from tests.utils.dataset_loader import load_sample_dataset

from skyulf.profiling.analyzer import EDAAnalyzer


def test_cast_date_columns_skips_all_null_sample() -> None:
    """A date-keyword column whose non-null sample is empty should be left untouched."""
    df = pl.DataFrame({"event_date": pl.Series([None, None, None], dtype=pl.Utf8), "x": [1, 2, 3]})
    analyzer = EDAAnalyzer(df)
    analyzer._cast_date_columns()
    # Column dtype should remain Utf8 since there was nothing to parse.
    assert analyzer.df["event_date"].dtype == pl.Utf8


def test_cast_date_columns_prefers_date_generic_when_datetime_generic_fails(monkeypatch) -> None:
    """When generic datetime parsing errors out but generic date parsing succeeds.

    exercises the "date_generic" success branch and its corresponding cast-application
    branch (``method == "date_generic"``).
    """
    import polars.series.string as pl_series_string

    real_to_datetime = pl_series_string.StringNameSpace.to_datetime

    def failing_to_datetime(self, *args, **kwargs):
        raise ValueError("mocked datetime-generic parse failure")

    monkeypatch.setattr(pl_series_string.StringNameSpace, "to_datetime", failing_to_datetime)

    df = pl.DataFrame(
        {
            "event_date": ["2021-06-15", "2021-07-20", "2021-08-25", "2021-09-10"],
            "x": [1, 2, 3, 4],
        }
    )
    analyzer = EDAAnalyzer(df)
    analyzer._cast_date_columns()

    # Restore immediately so later assertions/casts inside the mixin aren't affected.
    monkeypatch.setattr(pl_series_string.StringNameSpace, "to_datetime", real_to_datetime)

    assert analyzer.df["event_date"].dtype == pl.Date
    assert analyzer.df["event_date"].drop_nulls().len() == 4


def test_cast_date_columns_falls_back_to_explicit_format(monkeypatch) -> None:
    """When both generic parsers fail, an explicit common format should still be tried.

    Exercises the format-loop success branch and the ``method == "datetime_format"``
    cast-application branch.
    """
    df = pl.DataFrame(
        {
            # Day values > 12 disambiguate month-first vs day-first parsing.
            "event_date": ["06/15/2021", "07/20/2021", "08/25/2021"],
            "x": [1, 2, 3],
        }
    )
    analyzer = EDAAnalyzer(df)
    analyzer._cast_date_columns()

    assert analyzer.df["event_date"].dtype in (pl.Datetime, pl.Date)
    parsed = analyzer.df["event_date"]
    assert parsed.drop_nulls().len() == 3


def test_cast_date_columns_prints_when_final_cast_raises(monkeypatch, capsys) -> None:
    """A parse-success-at-sample-level but failure-at-full-column-apply should be logged.

    Exercises the outer ``except Exception as e: print(...)`` branch in the final
    cast-application step, by making the full-column ``Expr``-level ``to_datetime``
    raise while leaving the ``Series``-level sample check untouched.
    """
    import polars.expr.string as pl_expr_string

    real_expr_to_datetime = pl_expr_string.ExprStringNameSpace.to_datetime

    def failing_expr_to_datetime(self, *args, **kwargs):
        raise ValueError("mocked full-column datetime cast failure")

    monkeypatch.setattr(pl_expr_string.ExprStringNameSpace, "to_datetime", failing_expr_to_datetime)

    df = pl.DataFrame(
        {
            "event_date": [
                "2021-06-15T00:00:00",
                "2021-07-20T00:00:00",
                "2021-08-25T00:00:00",
                "2021-09-10T00:00:00",
            ],
            "x": [1, 2, 3, 4],
        }
    )
    analyzer = EDAAnalyzer(df)
    analyzer._cast_date_columns()

    monkeypatch.setattr(pl_expr_string.ExprStringNameSpace, "to_datetime", real_expr_to_datetime)

    captured = capsys.readouterr()
    assert "Failed to cast column event_date" in captured.out
    # Column should remain untouched (still Utf8) since the cast raised.
    assert analyzer.df["event_date"].dtype == pl.Utf8


def test_analyze_date_computes_duration_in_days() -> None:
    """``_analyze_date`` should compute a duration in days from min/max datetimes."""
    import datetime

    analyzer = EDAAnalyzer(pl.DataFrame({"d": [1]}))
    row = {
        "d__min": datetime.datetime(2021, 1, 1),
        "d__max": datetime.datetime(2021, 1, 11),
    }
    stats = analyzer._analyze_date("d", row)
    assert stats.duration_days == 10
    assert stats.min_date == str(row["d__min"])
    assert stats.max_date == str(row["d__max"])


class TestRealShapedDataset:
    """Integration-style check against the checked-in ``customers.csv`` sample,
    whose ``signup_date`` column is a plain ISO date string — closer to
    production data than the hand-built date fixtures used elsewhere in this
    file.
    """

    def test_cast_date_columns_parses_signup_date(self) -> None:
        df = load_sample_dataset("customers", engine="polars")
        analyzer = EDAAnalyzer(df)
        analyzer._cast_date_columns()

        assert analyzer.df["signup_date"].dtype in (pl.Date, pl.Datetime)
        # All 15 rows have a signup_date, so parsing should not drop any values.
        assert analyzer.df["signup_date"].drop_nulls().len() == df.height
