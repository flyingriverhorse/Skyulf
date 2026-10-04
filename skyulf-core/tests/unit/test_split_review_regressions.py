"""Split decisions preserve observed class ratios and positional target alignment."""

import numpy as np
import pandas as pd
import polars as pl
import pytest

from skyulf.engines import SkyulfPolarsWrapper
from skyulf.preprocessing._helpers import is_polars
from skyulf.preprocessing.split import DataSplitter


@pytest.mark.parametrize(
    "value, expected",
    [
        ([0, 1], False),
        (np.array([0, 1]), False),
        (pd.Series([0, 1]), False),
        (None, False),
        (pl.Series([0, 1]), True),
        (pl.DataFrame({"x": [0, 1]}), True),
        (SkyulfPolarsWrapper(pl.DataFrame({"x": [0, 1]})), True),
    ],
)
def test_polars_predicate_does_not_guess_from_default_engine(value, expected):
    """Unregistered array-like values cannot be sent through Polars-only methods."""
    assert is_polars(value) is expected


@pytest.mark.parametrize("seed", [0, 1, 2, 3])
@pytest.mark.parametrize("weighted", [False, True])
def test_unused_categories_do_not_disable_stratification(seed, weighted):
    """A declared but absent category must not alter the observed minority holdout size."""
    features = pd.DataFrame({"id": range(100)})
    target = pd.Series(pd.Categorical(["a"] * 90 + ["b"] * 10, categories=["a", "b", "unused"]))
    split = DataSplitter(test_size=0.2, random_state=seed, stratify_col="target").split_xy(
        features, target, sample_weight=np.ones(100) if weighted else None
    )
    assert isinstance(split.train, tuple) and isinstance(split.test, tuple)
    assert isinstance(split.train[1], pd.Series) and isinstance(split.test[1], pd.Series)
    assert split.test[1].value_counts()["b"] == 2
    assert split.train[1].value_counts()["b"] == 8


@pytest.mark.parametrize("engine", ["pandas", "polars"])
@pytest.mark.parametrize("kind", ["numpy", "list"])
@pytest.mark.parametrize("stratified", [False, True])
@pytest.mark.parametrize("weighted", [False, True])
def test_array_targets_keep_rows_aligned_in_all_partitions(engine, kind, stratified, weighted):
    """Array target containers must work in every split path without changing row membership."""
    frame = pd.DataFrame({"id": range(100)})
    if engine == "polars":
        frame = pl.from_pandas(frame)
    target = np.arange(100) % 2
    if kind == "list":
        target = target.tolist()
    split = DataSplitter(
        test_size=0.2,
        validation_size=0.2,
        random_state=42,
        stratify_col="target" if stratified else None,
    ).split_xy(frame, target, sample_weight=np.ones(100) if weighted else None)

    seen = []
    for partition, expected_rows in [(split.train, 60), (split.test, 20), (split.validation, 20)]:
        assert isinstance(partition, tuple)
        features, labels = partition
        assert isinstance(features, (pd.DataFrame, pl.DataFrame))
        assert isinstance(labels, (list, np.ndarray))
        ids = features["id"].to_numpy()
        seen.extend(ids)
        assert len(ids) == len(labels) == expected_rows
        np.testing.assert_array_equal(labels, ids % 2)
        if stratified:
            assert np.count_nonzero(labels) == expected_rows // 2
    assert sorted(seen) == list(range(100))


@pytest.mark.parametrize("engine", ["pandas", "polars"])
@pytest.mark.parametrize("target_size", [9, 11])
@pytest.mark.parametrize("weighted", [False, True])
def test_split_rejects_mismatched_target_lengths(engine, target_size, weighted):
    """A longer target cannot be silently truncated to the feature positions."""
    frame = pd.DataFrame({"x": range(10)})
    target = pd.Series(range(target_size))
    if engine == "polars":
        frame, target = pl.from_pandas(frame), pl.from_pandas(target)
    with pytest.raises(ValueError, match="inconsistent numbers of samples"):
        DataSplitter().split_xy(frame, target, sample_weight=np.ones(10) if weighted else None)
