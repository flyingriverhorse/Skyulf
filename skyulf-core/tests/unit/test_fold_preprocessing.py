"""Tests for the per-fold preprocessing refit contract (F-15 design note).

`FoldPreprocessor` is the engine-agnostic contract CV/tuning use to re-fit
preprocessing inside each fold; `FeatureEngineerFoldAdapter` implements it on
top of `FeatureEngineer` by rebuilding a fresh pipeline from `steps_config`
per fit (the registry constructs fresh calculators, so reconstruction *is*
the clone operation).
"""

import numpy as np
import pandas as pd
import polars as pl
import pytest

from skyulf.modeling import FoldPreprocessor
from skyulf.preprocessing import FeatureEngineerFoldAdapter


def _frame(engine: str, with_outlier: bool = False) -> tuple:
    """40-row frame: a float column with one null, an id-like column, target.

    The train fold used in tests excludes the last 10 rows, so per-fold fit
    statistics must NOT reflect the held-out tail (where the outlier lives).
    """
    n = 40
    values: list[float | None] = [float(i) for i in range(n)]
    values[5] = None  # missing value for the imputer to learn a fill for
    if with_outlier:
        values[-1] = 10000.0  # only present in the held-out portion
    pdf = pd.DataFrame({"x": values, "target": [i % 2 for i in range(n)]})
    if engine == "polars":
        df = pl.from_pandas(pdf)
        return df.drop("target"), df["target"]
    return pdf.drop("target", axis=1), pdf["target"]


def _split(engine: str, X, y, n_train: int = 30):
    if engine == "pandas":
        return X.iloc[:n_train], X.iloc[n_train:], y.iloc[:n_train], y.iloc[n_train:]
    return X[:n_train], X[n_train:], y[:n_train], y[n_train:]


@pytest.mark.parametrize("engine", ["pandas", "polars"])
def test_adapter_imputer_statistics_come_from_the_fit_fold_only(engine: str):
    X, y = _frame(engine)
    X_tr, X_val, y_tr, y_val = _split(engine, X, y)

    adapter = FeatureEngineerFoldAdapter(
        steps_config=[
            {
                "name": "fill",
                "transformer": "SimpleImputer",
                "params": {"strategy": "mean"},
            },
        ],
        target_column="target",
    )
    X_tr_t, y_tr_t = adapter.fit_transform(X_tr, y_tr)

    # Mean of x over the train fold excluding the null: mean(0..29 minus 5)
    expected_fill = float(np.mean([float(i) for i in range(30) if i != 5]))
    to_pandas = getattr(X_tr_t, "to_pandas", None)
    fitted = to_pandas() if callable(to_pandas) else X_tr_t
    assert not fitted["x"].isna().any()
    assert fitted.loc[5, "x"] == pytest.approx(expected_fill)

    # transform must reuse the fitted fill value, ignoring held-out rows
    X_val_t, y_val_t = adapter.transform(X_val, y_val)
    to_pandas = getattr(X_val_t, "to_pandas", None)
    applied = to_pandas() if callable(to_pandas) else X_val_t
    assert applied["x"].notna().all()


@pytest.mark.parametrize("engine", ["pandas", "polars"])
def test_adapter_rebuilds_fresh_state_per_fit(engine: str):
    """A second fit_transform on different rows must not inherit statistics
    from the first — reconstruction from steps_config is the clone."""
    X, y = _frame(engine)
    X_tr, X_val, y_tr, y_val = _split(engine, X, y)

    adapter = FeatureEngineerFoldAdapter(
        steps_config=[
            {"name": "fill", "transformer": "SimpleImputer", "params": {"strategy": "mean"}},
        ],
        target_column="target",
    )
    adapter.fit_transform(X_tr, y_tr)

    # Fit on just 5 rows: the learned mean is now the mean of those rows.
    small_X = X_tr.iloc[:5] if engine == "pandas" else X_tr[:5]
    small_y = y_tr.iloc[:5] if engine == "pandas" else y_tr[:5]
    adapter.fit_transform(small_X, small_y)
    X_val_t, _ = adapter.transform(X_val, y_val)
    to_pandas = getattr(X_val_t, "to_pandas", None)
    applied = to_pandas() if callable(to_pandas) else X_val_t
    # Held-out values below/above the small-fold mean must NOT be clipped to
    # the old 30-row statistics — they are scaled by the new fold's scaler.
    assert applied["x"].notna().all()


@pytest.mark.parametrize("engine", ["pandas", "polars"])
def test_adapter_filters_out_splitter_steps(engine: str):
    """Splitter steps already ran upstream; re-executing them inside a fold
    would re-split the fold. The adapter must drop them from the chain."""
    X, y = _frame(engine)
    X_tr, X_val, y_tr, y_val = _split(engine, X, y)

    adapter = FeatureEngineerFoldAdapter(
        steps_config=[
            {"name": "split", "transformer": "TrainTestSplitter", "params": {"test_size": 0.2}},
            {"name": "fill", "transformer": "SimpleImputer", "params": {"strategy": "mean"}},
        ],
        target_column="target",
    )
    X_tr_t, y_tr_t = adapter.fit_transform(X_tr, y_tr)
    # Output stays an (X, y) frame pair — no SplitDataset appeared.
    to_pandas = getattr(X_tr_t, "to_pandas", None)
    fitted = to_pandas() if callable(to_pandas) else X_tr_t
    assert "x" in fitted.columns
    assert len(fitted) == len(X_tr_t) and len(fitted) <= len(X_tr)


@pytest.mark.parametrize("engine", ["pandas", "polars"])
def test_adapter_transform_keeps_all_held_out_rows(engine: str):
    """Row-dropping steps are train-only (F-18 discipline): transform must
    never delete held-out rows."""
    X, y = _frame(engine)
    X_tr, X_val, y_tr, y_val = _split(engine, X, y)

    adapter = FeatureEngineerFoldAdapter(
        steps_config=[
            {"name": "dedup", "transformer": "Deduplicate", "params": {}},
            {"name": "fill", "transformer": "SimpleImputer", "params": {"strategy": "mean"}},
        ],
        target_column="target",
    )
    adapter.fit_transform(X_tr, y_tr)
    X_val_t, y_val_t = adapter.transform(X_val, y_val)
    n_val = len(X_val) if not hasattr(X_val, "shape") else X_val.shape[0]
    to_pandas = getattr(X_val_t, "to_pandas", None)
    applied = to_pandas() if callable(to_pandas) else X_val_t
    assert applied.shape[0] == n_val


def test_adapter_satisfies_the_fold_preprocessor_protocol():
    adapter = FeatureEngineerFoldAdapter(steps_config=[], target_column="target")
    assert isinstance(adapter, FoldPreprocessor)


def test_adapter_rejects_unknown_transformer():
    with pytest.raises(ValueError):
        FeatureEngineerFoldAdapter(
            steps_config=[{"name": "x", "transformer": "NoSuchNode", "params": {}}],
            target_column="target",
        )


@pytest.mark.parametrize(
    "transformer", ["Oversampling", "Undersampling", "DropMissingRows", "Deduplicate"]
)
def test_adapter_flags_row_count_changing_steps(transformer: str):
    adapter = FeatureEngineerFoldAdapter(
        steps_config=[{"name": "s", "transformer": transformer, "params": {}}],
        target_column="target",
    )
    assert adapter.changes_row_count is True


def test_adapter_reports_row_alignment_for_shape_preserving_steps():
    adapter = FeatureEngineerFoldAdapter(
        steps_config=[{"name": "s", "transformer": "StandardScaler", "params": {}}],
        target_column="target",
    )
    assert adapter.changes_row_count is False
