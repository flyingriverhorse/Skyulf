"""Failure-branch coverage for the elliptic-envelope outlier helpers.

Each test forces one defensive branch: non-coercible columns, failing
``predict`` calls, missing columns, and all-null columns must fail open
(keep every row) instead of raising.
"""

import logging

import numpy as np
import pandas as pd
import polars as pl

from skyulf.preprocessing.outliers.elliptic import (
    _coerce_column_to_float,
    _elliptic_filter_pandas,
    _elliptic_mask_numpy,
    _predict_inliers,
)


class _RaisingPredict:
    def predict(self, X):
        raise RuntimeError("predict broken")


class _StaticPredict:
    """Predicts a constant label for every row (1 = inlier, -1 = outlier)."""

    def __init__(self, value: int = 1) -> None:
        self.value = value

    def predict(self, X):
        return np.full(X.shape[0], self.value)


class TestCoerceColumnToFloat:
    def test_missing_column_returns_none(self):
        df = pl.DataFrame({"a": [1.0, 2.0]})
        assert _coerce_column_to_float(df, "missing") is None


class TestPredictInliers:
    def test_predict_failure_returns_none_and_warns(self, caplog):
        with caplog.at_level(logging.WARNING):
            result = _predict_inliers(_RaisingPredict(), np.array([1.0, 2.0]), "a")
        assert result is None
        assert "predict failed" in caplog.text

    def test_success_returns_predictions(self):
        result = _predict_inliers(_StaticPredict(1), np.array([1.0, 2.0]), "a")
        assert result.tolist() == [1, 1]


class TestEllipticMaskNumpyFailOpen:
    def _df(self) -> pl.DataFrame:
        return pl.DataFrame({"a": [1.0, 2.0]})

    def test_missing_model_column_keeps_all_rows(self):
        mask = _elliptic_mask_numpy(self._df(), {"missing": _StaticPredict()})
        assert mask.tolist() == [True, True]

    def test_non_coercible_column_keeps_all_rows(self):
        df = pl.DataFrame({"a": [[1, 2], [3]]})  # List column cannot cast to Float64
        mask = _elliptic_mask_numpy(df, {"a": _StaticPredict()})
        assert mask.tolist() == [True, True]

    def test_all_null_column_keeps_all_rows(self):
        df = pl.DataFrame({"a": [None, None]}, schema={"a": pl.Float64})
        mask = _elliptic_mask_numpy(df, {"a": _StaticPredict()})
        assert mask.tolist() == [True, True]

    def test_failing_predict_keeps_all_rows_and_warns(self, caplog):
        with caplog.at_level(logging.WARNING):
            mask = _elliptic_mask_numpy(self._df(), {"a": _RaisingPredict()})
        assert mask.tolist() == [True, True]
        assert "predict failed" in caplog.text

    def test_outlier_predictions_drop_rows(self):
        mask = _elliptic_mask_numpy(self._df(), {"a": _StaticPredict(value=-1)})
        assert mask.tolist() == [False, False]


class TestEllipticFilterPandasFailOpen:
    def test_predict_failure_keeps_all_rows_and_warns(self, caplog):
        X_pd = pd.DataFrame({"a": [1.0, 2.0]})
        with caplog.at_level(logging.WARNING):
            mask = _elliptic_filter_pandas(X_pd, {"a": _RaisingPredict()})
        assert mask.tolist() == [True, True]
        assert "predict failed" in caplog.text

    def test_missing_column_keeps_all_rows(self):
        X_pd = pd.DataFrame({"a": [1.0, 2.0]})
        mask = _elliptic_filter_pandas(X_pd, {"missing": _StaticPredict()})
        assert mask.tolist() == [True, True]

    def test_all_nan_column_keeps_all_rows(self):
        X_pd = pd.DataFrame({"a": [None, None]})
        mask = _elliptic_filter_pandas(X_pd, {"a": _StaticPredict()})
        assert mask.tolist() == [True, True]


class TestCalculatorFitFailureBranch:
    def test_per_column_fit_failure_is_recorded_as_warning(self, monkeypatch):
        from skyulf.preprocessing.outliers import elliptic as elliptic_mod

        class _BrokenEnvelope:
            def __init__(self, contamination: float = 0.01, random_state: int = 42) -> None:
                pass

            def fit(self, X):
                raise ValueError("singular covariance")

        monkeypatch.setattr(elliptic_mod, "EllipticEnvelope", _BrokenEnvelope)
        calc = elliptic_mod.EllipticEnvelopeCalculator()
        artifact = calc.fit(pd.DataFrame({"a": np.arange(10.0)}), {"columns": ["a"]})
        assert artifact["models"] == {}
        assert any("singular covariance" in w for w in artifact["warnings"])
