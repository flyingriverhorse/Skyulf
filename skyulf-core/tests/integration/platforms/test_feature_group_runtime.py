"""Feature job repairs preserve snapshot identity and explicit source ownership."""

from pathlib import Path

import pytest

from skyulf.integrations.databricks.features.config import FeatureGroup, FeaturePlan
from skyulf.integrations.databricks.features.runtime import selected_groups, transform_source


def _plan():
    """A selected group has one project-contained transform and output table."""
    group = FeatureGroup(
        "company", "raw", "features", "src/feature_groups/company.py:compute", ("size",)
    )
    return FeaturePlan("base", "merged", ("id",), "at", (group,))


def test_selective_run_has_explicit_subset():
    """Typos must not silently skip a required feature group."""
    assert selected_groups(_plan(), "*") == {"company"}
    assert selected_groups(_plan(), "company") == {"company"}
    assert selected_groups(_plan(), "") == set()
    with pytest.raises(ValueError, match="Unknown"):
        selected_groups(_plan(), "comapny")


def test_transform_source_is_read_without_execution(tmp_path):
    """Initialize must pin project code without triggering a transformation early."""
    path = tmp_path / "src/feature_groups/company.py"
    path.parent.mkdir(parents=True)
    path.write_text("raise AssertionError('not now')\n")
    source, digest = transform_source(tmp_path, _plan().groups[0])
    assert source == path.read_bytes().decode("utf-8")
    assert len(digest) == 64


def test_transform_path_cannot_escape_project(tmp_path):
    """Even manually constructed plans must not load source outside the project root."""
    group = FeatureGroup("bad", "raw", "out", "../../outside.py:compute", ("size",))
    with pytest.raises(ValueError, match="inside"):
        transform_source(Path(tmp_path), group)
