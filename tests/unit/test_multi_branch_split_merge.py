"""Unit tests for MergeMixin's Polars paths.

The integration suite (``tests/integration/test_merge_everywhere.py``)
exercises merging end-to-end through the engine; these unit tests pin the
engine-boundary behaviour added by the backend Polars migration that is
easy to lose silently:

- ``(X, y)`` tuple coercion and merging keep working when the frames are
  Polars, and the merged result is converted back to Polars so downstream
  nodes keep receiving the configured engine's frame type.
- An empty merged test split defaults to an empty frame of the *configured*
  engine (Polars), not a hard-coded pandas frame.
- Column stripping is a no-op on frames that don't carry the columns, and
  drops in-place on Polars frames.
"""

from types import SimpleNamespace

import pandas as pd
import polars as pl
import pytest

from backend.config import get_settings
from backend.ml_pipeline._execution.engine._merge import MergeMixin
from skyulf.data.dataset import SplitDataset


class _Merger(MergeMixin):
    """Minimal harness: no graph, default merge strategy, captured logs."""

    def __init__(self) -> None:
        self._node_configs: dict = {}
        self.logs: list[str] = []

    def log(self, msg: str) -> None:
        self.logs.append(msg)


def test_coerce_tuple_to_frame_polars_reattaches_target() -> None:
    """A Polars ``(X, y)`` payload must coerce to a frame with the target
    reattached — returning None here silently drops training data."""
    X = pl.DataFrame({"f": [1, 2]})
    out = _Merger()._coerce_tuple_to_frame((X, [0, 1]), target_col="target")
    assert isinstance(out, pl.DataFrame)
    assert out["f"].to_list() == [1, 2]
    assert out["target"].to_list() == [0, 1]


def test_merge_xy_tuples_polars_returns_polars_frame(monkeypatch) -> None:
    """Merging Polars X parts must yield a Polars merged X (engine round-trip),
    keeping y from the first edge."""
    monkeypatch.setattr(get_settings(), "SKYULF_ENGINE", "polars", raising=False)
    y = pl.Series([0, 1])
    artifacts = [
        (pl.DataFrame({"a": [1, 2]}), y),
        (pl.DataFrame({"b": [3.0, 4.0]}), y),
    ]
    node = SimpleNamespace(node_id="m1")

    merged_x, merged_y = _Merger()._merge_xy_tuples(node, artifacts)

    assert isinstance(merged_x, pl.DataFrame)
    assert sorted(merged_x.columns) == ["a", "b"]
    assert merged_x.height == 2
    assert merged_y is y


def test_merge_split_datasets_defaults_empty_test_to_polars_frame(monkeypatch) -> None:
    """When every branch's test split is empty, the merged test must default
    to an empty frame of the configured engine — hard-coding pandas here
    would hand downstream nodes a foreign frame type under SKYULF_ENGINE=polars."""
    monkeypatch.setattr(get_settings(), "SKYULF_ENGINE", "polars", raising=False)
    sd1 = SplitDataset(train=pl.DataFrame({"a": [1, 2]}), test=pl.DataFrame(), validation=None)
    sd2 = SplitDataset(train=pl.DataFrame({"b": [3, 4]}), test=pl.DataFrame(), validation=None)
    node = SimpleNamespace(node_id="m1")

    out = _Merger()._merge_split_datasets(node, [sd1, sd2], target_col="")

    assert isinstance(out.train, pl.DataFrame)
    assert sorted(out.train.columns) == ["a", "b"]
    assert isinstance(out.test, pl.DataFrame)
    assert out.test.is_empty()


def test_merge_split_datasets_defaults_empty_test_to_pandas_frame(monkeypatch) -> None:
    """The same default stays pandas when the configured engine is pandas."""
    monkeypatch.setattr(get_settings(), "SKYULF_ENGINE", "pandas", raising=False)
    sd1 = SplitDataset(train=pd.DataFrame({"a": [1, 2]}), test=pd.DataFrame(), validation=None)
    sd2 = SplitDataset(train=pd.DataFrame({"b": [3, 4]}), test=pd.DataFrame(), validation=None)
    node = SimpleNamespace(node_id="m1")

    out = _Merger()._merge_split_datasets(node, [sd1, sd2], target_col="")

    assert isinstance(out.test, pd.DataFrame)
    assert out.test.empty


def test_strip_columns_polars() -> None:
    """Column stripping must no-op when nothing matches and drop in-place on
    Polars frames (``df.drop(columns=...)`` is pandas-only API)."""
    merger = _Merger()
    df = pl.DataFrame({"a": [1], "b": [2]})

    assert merger._strip_columns(df, ["nonexistent"]) is df

    stripped = merger._strip_columns(df, ["a"])
    assert isinstance(stripped, pl.DataFrame)
    assert stripped.columns == ["b"]


def test_strip_columns_tuple_payload(monkeypatch) -> None:
    """An ``(X, y)`` tuple payload keeps its shape; only X loses the columns."""
    monkeypatch.setattr(get_settings(), "SKYULF_ENGINE", "polars", raising=False)
    merger = _Merger()
    X = pl.DataFrame({"a": [1], "b": [2]})
    y = pl.Series([0])

    (stripped_x, stripped_y) = merger._strip_columns((X, y), ["a"])

    assert stripped_x.columns == ["b"]
    assert stripped_y is y


def test_merge_split_dataset_xy_part_pandas_with_empty_branch(monkeypatch) -> None:
    """Pandas ``(X, y)`` parts merge column-wise, and a branch whose X is
    empty is skipped rather than failing the whole merge."""
    monkeypatch.setattr(get_settings(), "SKYULF_ENGINE", "pandas", raising=False)
    y = pd.Series([0, 1])
    non_empty = [
        (pd.DataFrame({"a": [1, 2]}), y),
        (pd.DataFrame(), y),  # empty X branch must be dropped, not fatal
        (pd.DataFrame({"b": [3, 4]}), y),
    ]
    node = SimpleNamespace(node_id="m1")

    merged_x, merged_y = _Merger()._merge_split_dataset_xy_part(node, "train", non_empty)

    assert isinstance(merged_x, pd.DataFrame)
    assert sorted(merged_x.columns) == ["a", "b"]
    assert merged_y is y


def test_merge_fallback_frames_raises_on_empty_input(monkeypatch) -> None:
    """An empty flattened input must fail loudly with actionable guidance,
    not merge into a silently degraded frame."""
    monkeypatch.setattr(get_settings(), "SKYULF_ENGINE", "polars", raising=False)
    node = SimpleNamespace(node_id="m1")

    with pytest.raises(ValueError, match="empty DataFrame"):
        _Merger()._merge_fallback_frames(node, [pl.DataFrame()], target_col="")
