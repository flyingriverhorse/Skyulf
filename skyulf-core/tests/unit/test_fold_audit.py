"""Unit tests for ``AuditedFoldPreprocessor`` / ``frame_rows``.

Covers the per-fold refit audit telemetry (findings 2026-08-26 §3/B): the
wrapper records every per-fold fit/transform input row count and exposes the
isolation invariant ``max(fit_rows) <= train_rows`` for post-hoc audit.
"""

import numpy as np
import pandas as pd
import polars as pl

from skyulf.preprocessing import AuditedFoldPreprocessor, frame_rows


class _RecordingPreprocessor:
    def __init__(self, changes_row_count: bool = False):
        self.changes_row_count = changes_row_count
        self.calls: list[tuple[str, int]] = []

    def fit_transform(self, X, y):
        self.calls.append(("fit", len(X)))
        return X, y

    def transform(self, X, y):
        self.calls.append(("transform", len(X)))
        return X, y


def test_frame_rows_handles_pandas_polars_numpy_and_scalars():
    assert frame_rows(pd.DataFrame({"a": range(5)})) == 5
    assert frame_rows(pl.DataFrame({"a": range(7)})) == 7
    assert frame_rows(np.zeros((3, 2))) == 3
    assert frame_rows(42) == -1


def test_wrapper_records_row_counts_and_passes_results_through():
    inner = _RecordingPreprocessor()
    audit = AuditedFoldPreprocessor(inner)
    X = pd.DataFrame({"a": range(10)})
    y = pd.Series(range(10))

    out_x, out_y = audit.fit_transform(X, y)
    assert out_x is X
    assert out_y is y
    small_x, _ = audit.transform(X.iloc[:4], y)
    assert len(small_x) == 4

    assert audit.fit_rows == [10]
    assert audit.transform_rows == [4]
    assert audit.inner is inner
    assert inner.calls == [("fit", 10), ("transform", 4)]


def test_summary_reports_isolation_ok_when_fits_stay_within_train_rows():
    audit = AuditedFoldPreprocessor(_RecordingPreprocessor())
    X = pd.DataFrame({"a": range(8)})
    audit.fit_transform(X, None)
    audit.fit_transform(X.iloc[:6], None)
    assert audit.summary(train_rows=8) == {
        "fit_calls": 2,
        "max_fit_rows": 8,
        "transform_calls": 0,
        "train_rows": 8,
        "isolation_ok": True,
    }


def test_summary_flags_a_fit_that_saw_more_rows_than_the_train_split():
    audit = AuditedFoldPreprocessor(_RecordingPreprocessor())
    audit.fit_transform(pd.DataFrame({"a": range(12)}), None)
    summary = audit.summary(train_rows=8)
    assert summary["isolation_ok"] is False
    assert summary["max_fit_rows"] == 12


def test_summary_without_train_rows_omits_the_invariant():
    audit = AuditedFoldPreprocessor(_RecordingPreprocessor())
    assert audit.summary() == {"fit_calls": 0, "max_fit_rows": 0, "transform_calls": 0}


def test_changes_row_count_is_delegated_from_the_wrapped_adapter():
    audit = AuditedFoldPreprocessor(_RecordingPreprocessor(changes_row_count=True))
    assert audit.changes_row_count is True


def test_wrapper_accepts_adapters_without_changes_row_count_attr():
    class _Bare:
        def fit_transform(self, X, y):
            return X, y

        def transform(self, X, y):
            return X, y

    audit = AuditedFoldPreprocessor(_Bare())
    assert audit.changes_row_count is False
    audit.fit_transform(pd.DataFrame({"a": [1]}), None)
    assert audit.fit_rows == [1]
