"""Keep training YAML references aligned with the loader's accepted field sets."""

import re
from pathlib import Path

import pytest

from skyulf.integrations.databricks.projects.yaml_config import (
    MODEL_FIELDS,
    TRAINING_FIELDS,
    TUNING_FIELDS,
    read_training_config,
)

ROOT = Path(__file__).resolve().parents[4]
REFERENCE = ROOT / "docs/user_guide/databricks_training_config.md"
PROJECT_GUIDE = (
    ROOT / "skyulf-core/templates/databricks/template/{{.project_name}}/TRAINING.md.tmpl"
)


def _reference(path: Path) -> str:
    """Read a required guide without silently accepting a missing source page."""
    assert path.is_file(), f"Missing training reference: {path}"
    return path.read_text(encoding="utf-8")


@pytest.mark.parametrize(
    "section,accepted",
    [
        ("training-fields", TRAINING_FIELDS),
        ("model-fields", MODEL_FIELDS),
        ("tuning-fields", TUNING_FIELDS),
    ],
)
def test_reference_covers_accepted_fields(section, accepted):
    """Adding or removing a YAML field must update the authoritative reference."""
    document = _reference(REFERENCE)
    table = document.split(f"<!-- {section}:start -->", 1)[1].split(f"<!-- {section}:end -->", 1)[0]
    fields = re.findall(r"^\| `([a-z_]+)` \|", table, re.MULTILINE)
    assert len(fields) == len(set(fields)), "Each accepted field needs one reference row."
    assert set(fields) == accepted


def test_reference_examples_use_all_layouts_and_load(tmp_path):
    """Complete examples must remain accepted training documents for all three layouts."""
    pytest.importorskip("yaml")
    examples = re.findall(
        r"<!-- example:([a-z_]+) -->\s*```yaml\n(.*?)```", _reference(REFERENCE), re.DOTALL
    )
    assert {layout for layout, _ in examples} == {
        "single_model",
        "model_competition",
        "multi_target",
    }
    assert len(examples) == 3
    for layout, source in examples:
        directory = tmp_path / layout
        directory.mkdir()
        (directory / "training.yml").write_text(source, encoding="utf-8")
        document = read_training_config(directory)
        assert document is not None
        assert document["defaults"]["training_layout"] == layout
        assert len(document["models"]) == (1 if layout == "single_model" else 2)


def test_project_guide_links_to_the_single_field_reference():
    """Project instructions must route users to one maintained field contract."""
    guide = _reference(PROJECT_GUIDE)
    reference_url = "https://www.skyulf.com/manual/user_guide/databricks_training_config.html"
    assert reference_url in guide
    assert "<!-- training-fields:start -->" not in guide
    assert "<!-- model-fields:start -->" not in guide
    assert "<!-- tuning-fields:start -->" not in guide
