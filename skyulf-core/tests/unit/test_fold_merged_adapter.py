"""Unit tests for ``MergedBranchFoldAdapter`` (task #11: fork-join per-fold refit).

The adapter must reproduce the engine's pure-strategy column-wise merge
(``_merge_frames_columnwise`` with no ownership baseline — the fork-join case,
where the nearest-common-ancestor artifact is a SplitDataset and ownership
analysis is inert) so fold scores match what the full run produces, with
preprocessing re-fit on fold-train rows only.
"""

import copy

import numpy as np
import pandas as pd
import pytest

from skyulf.modeling.base import extract_xy
from skyulf.preprocessing.fold_adapter import MergedBranchFoldAdapter
from skyulf.preprocessing.pipeline import FeatureEngineer

BRANCH_NUM = [{"name": "scale", "transformer": "StandardScaler", "params": {"columns": ["num1"]}}]
BRANCH_WOE = [
    {
        "name": "woe",
        "transformer": "WOEEncoder",
        "params": {"columns": ["city"], "regularization": 0.5},
    }
]
BRANCH_MISSING = [
    {"name": "missing", "transformer": "MissingIndicator", "params": {"columns": ["num2"]}}
]


def _payload(n: int = 200, seed: int = 0):
    rng = np.random.default_rng(seed)
    num2 = rng.normal(size=n)
    num2[rng.random(n) < 0.1] = np.nan
    df = pd.DataFrame(
        {
            "num1": rng.normal(size=n),
            "num2": num2,
            "city": [f"c{v}" for v in rng.integers(0, 20, size=n)],
            "target": rng.integers(0, 2, size=n),
        }
    )
    return extract_xy(df, "target")


def _branch_frame(steps, X, y) -> pd.DataFrame:
    out, _metrics = FeatureEngineer(steps).fit_transform((X, y))
    if isinstance(out, tuple) and len(out) == 2:
        return out[0]
    return out


def _pure_merge(frames: list[pd.DataFrame], strategy: str) -> pd.DataFrame:
    """Reference copy of the engine's pure-strategy column-wise merge path."""
    indexed = list(enumerate(frames))
    ordered = indexed if strategy == "last_wins" else list(reversed(indexed))
    result_cols: dict[str, pd.Series] = {}
    for _idx, df in ordered:
        df_aligned = df.reset_index(drop=True)
        for col in df_aligned.columns:
            result_cols[col] = df_aligned[col]
    return pd.DataFrame(result_cols)


def test_fit_transform_merges_branch_outputs_and_returns_target():
    X, y = _payload()
    adapter = MergedBranchFoldAdapter(
        [BRANCH_NUM, BRANCH_WOE], merge_strategy="last_wins", target_column="target"
    )
    X_out, y_out = adapter.fit_transform(X, y)
    assert isinstance(X_out, pd.DataFrame)
    assert len(X_out) == len(X)
    assert np.array_equal(np.asarray(y_out), np.asarray(y))
    expected_cols = set(_branch_frame(BRANCH_NUM, X, y).columns) | set(
        _branch_frame(BRANCH_WOE, X, y).columns
    )
    assert set(X_out.columns) == expected_cols


@pytest.mark.parametrize("strategy", ["last_wins", "first_wins"])
def test_merge_matches_engine_pure_strategy_path(strategy):
    X, y = _payload()
    adapter = MergedBranchFoldAdapter(
        [BRANCH_NUM, BRANCH_WOE], merge_strategy=strategy, target_column="target"
    )
    X_out, _ = adapter.fit_transform(X, y)
    frames = [_branch_frame(BRANCH_NUM, X, y), _branch_frame(BRANCH_WOE, X, y)]
    expected = _pure_merge(frames, strategy)
    pd.testing.assert_frame_equal(X_out.reset_index(drop=True), expected)


@pytest.mark.parametrize("strategy", ["last_wins", "first_wins"])
def test_merge_with_added_column_matches_engine_path(strategy):
    X, y = _payload()
    adapter = MergedBranchFoldAdapter(
        [BRANCH_NUM, BRANCH_MISSING], merge_strategy=strategy, target_column="target"
    )
    X_out, _ = adapter.fit_transform(X, y)
    frames = [_branch_frame(BRANCH_NUM, X, y), _branch_frame(BRANCH_MISSING, X, y)]
    expected = _pure_merge(frames, strategy)
    pd.testing.assert_frame_equal(X_out.reset_index(drop=True), expected)


def test_transform_keeps_all_held_out_rows():
    X, y = _payload()
    adapter = MergedBranchFoldAdapter(
        [BRANCH_NUM, BRANCH_WOE], merge_strategy="last_wins", target_column="target"
    )
    adapter.fit_transform(X.iloc[:150], y.iloc[:150])
    X_test, y_test = X.iloc[150:], y.iloc[150:]
    X_out, y_out = adapter.transform(X_test, y_test)
    assert len(X_out) == 50, "transform must keep every held-out row (F-18)"
    assert len(np.asarray(y_out)) == 50


def test_transform_before_fit_raises():
    adapter = MergedBranchFoldAdapter(
        [BRANCH_NUM], merge_strategy="last_wins", target_column="target"
    )
    X, y = _payload(n=10)
    with pytest.raises(RuntimeError, match="before fit_transform"):
        adapter.transform(X, y)


def test_drop_columns_are_stripped_after_merge():
    X, y = _payload()
    adapter = MergedBranchFoldAdapter(
        [BRANCH_NUM, BRANCH_WOE],
        merge_strategy="last_wins",
        target_column="target",
        drop_columns=["city"],
    )
    X_out, _ = adapter.fit_transform(X, y)
    assert "city" not in X_out.columns


@pytest.mark.parametrize(
    "bad_step",
    [
        {"name": "split", "transformer": "TrainTestSplitter", "params": {"test_size": 0.2}},
        {"name": "drop_rows", "transformer": "DropMissingRows", "params": {"columns": ["num2"]}},
        {"name": "iqr", "transformer": "IQR", "params": {"columns": ["num1"]}},
        {"name": "smote", "transformer": "Oversampling", "params": {}},
    ],
)
def test_construction_rejects_splitter_and_row_count_changing_steps(bad_step):
    with pytest.raises(ValueError, match="row counts|fold"):
        MergedBranchFoldAdapter([[bad_step]], merge_strategy="last_wins", target_column="target")


def test_construction_rejects_unknown_strategy():
    with pytest.raises(ValueError, match="merge strategy"):
        MergedBranchFoldAdapter([BRANCH_NUM], merge_strategy="bogus", target_column="target")


def test_rejects_payload_still_carrying_target():
    X, y = _payload(n=10)
    adapter = MergedBranchFoldAdapter(
        [BRANCH_NUM], merge_strategy="last_wins", target_column="target"
    )
    with pytest.raises(ValueError, match="target"):
        adapter.fit_transform(X.assign(target=y), y)


def test_deepcopy_is_safe_and_fold_refits_are_independent():
    adapter = MergedBranchFoldAdapter(
        [BRANCH_NUM], merge_strategy="last_wins", target_column="target"
    )
    # The tuning engine deep-copies preprocessors per worker; that must be safe.
    copy.deepcopy(adapter)

    X, y = _payload(n=120, seed=1)
    X_a, y_a = X.iloc[:60], y.iloc[:60]
    X_b = X.iloc[60:].copy()
    X_b["num1"] = X_b["num1"] + 100.0  # very different scale -> different per-fold statistics

    X_out_a, _ = adapter.fit_transform(X_a, y_a)
    X_out_b, _ = adapter.fit_transform(X_b, y.iloc[60:])
    assert abs(X_out_a["num1"].mean()) < 0.2, "fold A output must be standardized by fold A stats"
    assert abs(X_out_b["num1"].mean()) < 0.2, "fold B must refit, not reuse fold A statistics"
