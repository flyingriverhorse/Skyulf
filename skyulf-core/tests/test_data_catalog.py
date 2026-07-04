"""Tests for the DataCatalog abstract interface."""

from typing import cast

import pandas as pd
import pytest

from skyulf.data.catalog import DataCatalog
from skyulf.engines import SkyulfDataFrame


class _InMemoryCatalog(DataCatalog):
    """Minimal concrete DataCatalog used to exercise the abstract base."""

    def __init__(self) -> None:
        self._store: dict[str, "pd.DataFrame | SkyulfDataFrame"] = {}

    def load(self, dataset_id: str, **kwargs) -> "pd.DataFrame | SkyulfDataFrame":
        return self._store[dataset_id]

    def save(self, dataset_id: str, data: "pd.DataFrame | SkyulfDataFrame", **kwargs) -> None:
        self._store[dataset_id] = data

    def exists(self, dataset_id: str) -> bool:
        return dataset_id in self._store


def test_data_catalog_cannot_be_instantiated_directly() -> None:
    """DataCatalog is an ABC: it must not be instantiable without overriding abstract methods."""
    with pytest.raises(TypeError):
        DataCatalog()  # type: ignore[abstract]


def test_concrete_catalog_save_load_roundtrip() -> None:
    """A concrete subclass should support the basic save/load/exists contract."""
    catalog = _InMemoryCatalog()
    df = pd.DataFrame({"a": [1, 2, 3]})

    assert catalog.exists("ds1") is False

    catalog.save("ds1", df)

    assert catalog.exists("ds1") is True
    pd.testing.assert_frame_equal(cast(pd.DataFrame, catalog.load("ds1")), df)


def test_get_dataset_name_default_returns_none() -> None:
    """The base class's default get_dataset_name implementation returns None unless overridden."""
    catalog = _InMemoryCatalog()

    assert catalog.get_dataset_name("any_id") is None
