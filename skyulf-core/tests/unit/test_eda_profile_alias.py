"""Compatibility checks for the public EDA profile name."""

import importlib
import pickle
from pathlib import Path
from typing import Literal

import pytest

from skyulf.profiling.schemas import DatasetProfile


@pytest.fixture
def legacy_profile() -> DatasetProfile:
    """Build the existing schema so old artifacts remain valid after the alias."""
    return DatasetProfile(
        row_count=3,
        column_count=1,
        duplicate_rows=1,
        missing_cells_percentage=0.0,
        memory_usage_mb=0.1,
        columns={},
        sample_data=[{"value": 1}, {"value": 2}, {"value": 2}],
    )


@pytest.mark.parametrize("module_name", ["skyulf", "skyulf.profiling"])
def test_eda_profile_is_existing_public_schema(module_name: str) -> None:
    """The clearer name must preserve existing imports and runtime type identity."""
    module = importlib.import_module(module_name)

    assert module.EDAProfile is module.DatasetProfile is DatasetProfile
    assert "EDAProfile" in module.__all__
    assert module.EDAProfile.__name__ == "DatasetProfile"
    assert module.EDAProfile.__module__ == "skyulf.profiling.schemas"


@pytest.mark.parametrize("mode", ["validation", "serialization"])
def test_eda_profile_preserves_json_schema(
    mode: Literal["validation", "serialization"],
) -> None:
    """A public import alias must not rename schema titles or change wire fields."""
    from skyulf import EDAProfile

    schema = EDAProfile.model_json_schema(mode=mode)

    assert schema == DatasetProfile.model_json_schema(mode=mode)
    assert schema["title"] == "DatasetProfile"


def test_eda_profile_loads_existing_pickle(legacy_profile: DatasetProfile) -> None:
    """Artifacts referencing the original schema must load as the same class."""
    payload = pickle.dumps(legacy_profile)
    from skyulf import EDAProfile

    restored = pickle.loads(payload)

    assert type(restored) is EDAProfile
    assert restored == legacy_profile


def test_eda_profile_json_roundtrip(legacy_profile: DatasetProfile) -> None:
    """Either public name must read JSON written through the other name."""
    from skyulf import EDAProfile

    restored = EDAProfile.model_validate_json(legacy_profile.model_dump_json())

    assert type(restored) is DatasetProfile
    assert DatasetProfile.model_validate_json(restored.model_dump_json()) == legacy_profile


def test_eda_profile_joblib_roundtrip(legacy_profile: DatasetProfile, tmp_path: Path) -> None:
    """Explicit serializer callers must retain the EDA schema's runtime identity."""
    from skyulf import EDAProfile
    from skyulf.core import JoblibModelSerializer

    serializer = JoblibModelSerializer()
    path = tmp_path / "profile.joblib"
    serializer.dump(legacy_profile, path)
    restored = serializer.load(path)

    assert type(restored) is EDAProfile
    assert restored == legacy_profile


def test_dataset_profile_inspection_node_keeps_existing_id() -> None:
    """Saved pipeline configurations must still resolve the inspection node."""
    from skyulf import EDAProfile, NodeRegistry
    from skyulf.preprocessing.inspection import (
        DatasetProfileApplier,
        DatasetProfileCalculator,
    )

    assert NodeRegistry.get_calculator("DatasetProfile") is DatasetProfileCalculator
    assert NodeRegistry.get_applier("DatasetProfile") is DatasetProfileApplier
    assert EDAProfile is not DatasetProfileCalculator
