"""Unit tests for skyulf.data.dataset.SplitDataset.

Tests cover construction, the copy() method across all split payload types,
and the optional validation slot.
"""

import typing

import numpy as np
import pandas as pd

from skyulf.data.dataset import SplitDataset, SplitPayload

# ---------------------------------------------------------------------------
# Construction
# ---------------------------------------------------------------------------


def test_split_dataset_basic_construction() -> None:
    """SplitDataset must store train and test payloads as given."""
    train = pd.DataFrame({"a": [1, 2]})
    test = pd.DataFrame({"a": [3]})
    ds = SplitDataset(train=train, test=test)
    assert ds.train is train
    assert ds.test is test
    assert ds.validation is None


def test_split_dataset_with_validation() -> None:
    """Validation slot must be stored when provided."""
    train = pd.DataFrame({"a": [1, 2]})
    test = pd.DataFrame({"a": [3]})
    val = pd.DataFrame({"a": [4, 5]})
    ds = SplitDataset(train=train, test=test, validation=val)
    assert ds.validation is val


def test_split_dataset_tuple_payload() -> None:
    """SplitDataset must accept (X, y) tuples as split payloads."""
    X_train = pd.DataFrame({"f": [1.0, 2.0]})
    y_train = pd.Series([0, 1])
    X_test = pd.DataFrame({"f": [3.0]})
    y_test = pd.Series([0])
    ds = SplitDataset(train=(X_train, y_train), test=(X_test, y_test))
    assert isinstance(ds.train, tuple)
    assert isinstance(ds.test, tuple)


# ---------------------------------------------------------------------------
# copy()
# ---------------------------------------------------------------------------


def test_copy_produces_independent_dataframes() -> None:
    """copy() must return new DataFrames so mutations don't affect the original."""
    train = pd.DataFrame({"a": [1, 2]})
    test = pd.DataFrame({"a": [3]})
    ds = SplitDataset(train=train, test=test)
    ds_copy = ds.copy()
    # Mutate the copy — original must be unchanged.
    assert not isinstance(ds_copy.train, tuple)
    assert not isinstance(ds.train, tuple)
    ds_copy.train.loc[0, "a"] = 999
    assert ds.train.loc[0, "a"] == 1


def test_copy_with_validation_copies_all_three_splits() -> None:
    """copy() must copy the validation split when present."""
    train = pd.DataFrame({"a": [1]})
    test = pd.DataFrame({"a": [2]})
    val = pd.DataFrame({"a": [3]})
    ds = SplitDataset(train=train, test=test, validation=val)
    ds_copy = ds.copy()
    # All three must be new objects.
    assert ds_copy.train is not train
    assert ds_copy.test is not test
    assert ds_copy.validation is not val


def test_copy_without_validation_keeps_none() -> None:
    """copy() must preserve validation=None when no validation split exists."""
    ds = SplitDataset(
        train=pd.DataFrame({"a": [1]}),
        test=pd.DataFrame({"a": [2]}),
    )
    ds_copy = ds.copy()
    assert ds_copy.validation is None


def test_copy_tuple_payloads_are_independent() -> None:
    """copy() must deep-copy (X, y) tuple payloads."""
    X_train = pd.DataFrame({"f": [1.0, 2.0]})
    y_train = pd.Series([0, 1], name="target")
    X_test = pd.DataFrame({"f": [3.0]})
    y_test = pd.Series([1], name="target")
    ds = SplitDataset(train=(X_train, y_train), test=(X_test, y_test))
    ds_copy = ds.copy()

    X_copy, y_copy = typing.cast(tuple[pd.DataFrame, pd.Series], ds_copy.train)
    X_copy.loc[0, "f"] = 999.0
    # Original X_train must still have 1.0.
    assert X_train.loc[0, "f"] == 1.0


def test_copy_preserves_data_equality() -> None:
    """Copy data must equal the original even though it is a different object."""
    train = pd.DataFrame({"x": [1.0, 2.0, 3.0]})
    test = pd.DataFrame({"x": [4.0]})
    ds = SplitDataset(train=train, test=test)
    ds_copy = ds.copy()
    pd.testing.assert_frame_equal(typing.cast(pd.DataFrame, ds_copy.train), train)
    pd.testing.assert_frame_equal(typing.cast(pd.DataFrame, ds_copy.test), test)


def test_copy_with_numpy_array_payload() -> None:
    """copy() must handle payloads that have neither copy() nor clone() without raising."""
    # Plain numpy arrays have .copy() so they will be copied.
    arr = np.array([1, 2, 3])
    ds = SplitDataset(train=typing.cast(SplitPayload, arr), test=typing.cast(SplitPayload, arr))
    ds_copy = ds.copy()
    # Should return without raising; data must be equivalent.
    np.testing.assert_array_equal(typing.cast(np.ndarray, ds_copy.train), arr)


class _CloneOnly:
    """Stub payload exposing only `clone()` (e.g. a lazy-frame-like engine object)."""

    def __init__(self, value: int) -> None:
        self.value = value

    def clone(self) -> "_CloneOnly":
        """Return a new instance with the same value."""
        return _CloneOnly(self.value)


class _NoCopyNoClone:
    """Stub payload exposing neither `copy()` nor `clone()`."""

    def __init__(self, value: int) -> None:
        self.value = value


def test_copy_uses_clone_when_copy_is_unavailable() -> None:
    """copy() must fall back to .clone() for payloads without a .copy() method."""
    train = _CloneOnly(1)
    test = _CloneOnly(2)
    ds = SplitDataset(train=typing.cast(SplitPayload, train), test=typing.cast(SplitPayload, test))
    ds_copy = ds.copy()
    assert ds_copy.train is not train
    assert typing.cast(_CloneOnly, ds_copy.train).value == 1


def test_copy_returns_same_object_when_neither_copy_nor_clone_available() -> None:
    """copy() must return the payload unchanged when it has no .copy()/.clone() method."""
    train = _NoCopyNoClone(1)
    test = _NoCopyNoClone(2)
    ds = SplitDataset(train=typing.cast(SplitPayload, train), test=typing.cast(SplitPayload, test))
    ds_copy = ds.copy()
    assert ds_copy.train is train
    assert ds_copy.test is test
