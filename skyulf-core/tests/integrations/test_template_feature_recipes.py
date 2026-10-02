"""Shipped feature recipes select exact independent Core and custom step lists."""

import json
from pathlib import Path

import pytest

from skyulf.inference.project_code import load_project_module
from skyulf.integrations.databricks._project_files import project_source

FEATURES = (
    Path(__file__).resolve().parents[2]
    / "templates/databricks/template/{{.project_name}}/src/features"
)


@pytest.mark.parametrize(
    "recipe, expected",
    [
        ("default", []),
        ("none", []),
        ("example_frequency", ["FittedFunction"]),
        ("example_imputer", ["SimpleImputer"]),
        ("example_imputer_frequency", ["SimpleImputer", "FittedFunction"]),
        ("example_all", ["SimpleImputer", "ColumnFunction", "FittedFunction", "FittedFunction"]),
    ],
)
def test_shipped_preprocessing_recipes_select_exact_steps(recipe, expected):
    """A frequency-only branch must never silently inherit an imputer from another branch."""
    module = load_project_module(project_source(FEATURES))
    steps = module.build_preprocessing(recipe=recipe)
    assert len(steps) == len(expected)
    assert all(
        step["transformer"].endswith(kind) for step, kind in zip(steps, expected, strict=True)
    )


@pytest.mark.parametrize(
    "recipe, count",
    [("default", 0), ("none", 0), ("example_complete_inputs", 1), ("example_all", 2)],
)
def test_shipped_pre_split_recipes_are_independently_selectable(recipe, count):
    """Filtering may be shared or omitted independently of the learned transformations."""
    module = load_project_module(project_source(FEATURES))
    steps = module.build_pre_split_steps(recipe=recipe)
    assert len(steps) == count
    if steps:
        assert steps[0]["pre_split"]["learns_from_data"] is False


@pytest.mark.parametrize("factory", ["build_preprocessing", "build_pre_split_steps"])
def test_shipped_recipes_reject_unknown_names(factory):
    """A misspelled recipe must fail instead of silently training without intended steps."""
    module = load_project_module(project_source(FEATURES))
    with pytest.raises(ValueError, match="Unknown .* recipe"):
        getattr(module, factory)(recipe="typo")


def test_asset_instructions_are_inline_and_inactive():
    """Users can discover the full asset workflow without enabling sample business data."""
    manifest = json.loads((FEATURES / "assets.json").read_text(encoding="utf-8"))
    assert manifest["files"] == []
    instructions = "\n".join(manifest["_help"])
    assert "read_project_asset" in instructions
    assert "requirements.txt" in instructions
    assert "preprocessing" in instructions and "pre_split" in instructions
