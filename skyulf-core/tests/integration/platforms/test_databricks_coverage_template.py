"""Editable coverage gates reach generated training and rollout policies."""

import json
import re

import pytest
import yaml
from test_databricks_bundle_generation import (
    CLI,
    OFFLINE_CLI,
    PROFILE,
    TEMPLATE_ROOT,
    _generate_project,
)


def test_coverage_answers_are_hidden_nullable_bounded_fields():
    """Answer files can configure every independent policy without extending the wizard."""
    properties = json.loads((TEMPLATE_ROOT / "databricks_template_schema.json").read_text())[
        "properties"
    ]
    for prefix in ("", *(f"branch_{slot}_" for slot in range(1, 9))):
        for field, default in (("min_scored_fraction", "0.90"), ("max_coverage_drop", "0.05")):
            spec = properties[prefix + field]
            assert spec["default"] == default
            assert spec["skip_prompt_if"] == {}
            for value in ("null", "0", "1", "0.85", "0.05"):
                assert re.fullmatch(spec["pattern"], value)
            for value in ("true", "-0.1", "1.1", "NaN", "Infinity"):
                assert not re.fullmatch(spec["pattern"], value)


@pytest.mark.skipif(
    not CLI or not (PROFILE or OFFLINE_CLI), reason="Requires opt-in CLI rendering."
)
@pytest.mark.parametrize("layout", ["single_model", "model_competition", "multi_target"])
@pytest.mark.parametrize("minimum,drop", [("0.90", "0.05"), ("0.85", "0.10"), ("null", "null")])
def test_coverage_answers_reach_each_training_layout(tmp_path, layout, minimum, drop):
    """Configured values and explicit disablement must survive generation as typed YAML."""
    values = {"training_layout": layout}
    if layout == "multi_target":
        values.update(branch_1_min_scored_fraction=minimum, branch_1_max_coverage_drop=drop)
    else:
        values.update(
            min_scored_fraction=minimum, max_coverage_drop=drop, include_daily_rollout="true"
        )
    project = _generate_project(tmp_path, **values)
    training = yaml.safe_load((project / "config/training.yml").read_text())
    expected = {
        "min_scored_fraction": yaml.safe_load(minimum),
        "max_coverage_drop": yaml.safe_load(drop),
    }
    if layout == "multi_target":
        first, second, *_ = training["models"].values()
        assert {name: first[name] for name in expected} == expected
        assert second["min_scored_fraction"] == 0.90
        assert second["max_coverage_drop"] == 0.05
    else:
        assert {name: training["defaults"][name] for name in expected} == expected
        serving = yaml.safe_load((project / "config/serving.yml").read_text())
        assert {name: serving["rollout"]["approval"][name] for name in expected} == expected
