"""Keep three actionable diagnostics tied to real validators and working fixes."""

from pathlib import Path

import pytest

from skyulf.integrations.databricks.feature_store.lifecycle_config import bind_lookup_tables
from skyulf.integrations.databricks.projects.workflow_config import validate_workflow_config
from skyulf.integrations.databricks.scoring.shared.scoring_pre_split import (
    resolve_pre_split_scoring,
)

GUIDE = Path(__file__).resolve().parents[4] / "docs/user_guide/databricks_troubleshooting.md"


def _diagnostic(section: str) -> str:
    """Require the emitted message, its cause and an editable correction together."""
    content = GUIDE.read_text(encoding="utf-8")
    entry = content.split(f"<!-- diagnostic:{section} -->", 1)[1].split("<!-- /diagnostic -->", 1)[
        0
    ]
    assert "**Cause.**" in entry
    assert "**Fix.**" in entry
    assert "**Check.**" in entry
    return entry


def test_target_pre_split_error_and_skip_fix_are_documented():
    """A valid training target filter needs an explicit scoring-time decision."""
    workflow = {
        "target_column": "churn",
        "input_columns": ["income"],
        "pre_split_steps": [
            {
                "name": "dropmissingrows",
                "transformer": "DropMissingRows",
                "params": {"subset": ["churn"], "how": "any"},
            }
        ],
    }
    request = {"reuse_pre_split": True, "skip_target_steps": False}
    message = (
        "Pre-split step 'dropmissingrows' reads target 'churn'. "
        "Set SKIP_TARGET_PRE_SPLIT_STEPS=True or SCORING_MODE='custom'."
    )
    with pytest.raises(ValueError) as error:
        resolve_pre_split_scoring(request, workflow)
    assert str(error.value) == message
    fixed = resolve_pre_split_scoring({**request, "skip_target_steps": True}, workflow)
    assert fixed is not None
    assert fixed["pre_split"]["steps"] == []
    assert fixed["pre_split"]["skipped_target_steps"] == ["dropmissingrows"]
    entry = _diagnostic("target-pre-split")
    assert message in entry
    assert "src/features/scoring.py" in entry
    assert 'SCORING_MODE = "pre_split"\nSKIP_TARGET_PRE_SPLIT_STEPS = True' in entry


def test_lookup_table_error_and_qualified_name_fix_are_documented():
    """An incomplete lookup must identify the exact zero-based YAML entry to repair."""
    settings = {
        "lookups": [
            {"table_name": "{catalog}.{input_schema}.history"},
            {"table_name": ""},
        ]
    }
    bindings = {"catalog": "workspace", "input_schema": "features"}
    message = (
        "feature_lookup.lookups[1].table_name: "
        "Table name must contain one to three simple identifiers."
    )
    with pytest.raises(ValueError) as error:
        bind_lookup_tables(settings, bindings)
    assert str(error.value) == message
    assert settings["lookups"][1]["table_name"] == ""
    settings["lookups"][1]["table_name"] = "{catalog}.{input_schema}.balances"
    fixed = bind_lookup_tables(settings, bindings)
    assert fixed["lookups"][1]["table_name"] == "workspace.features.balances"
    entry = _diagnostic("lookup-table")
    assert message in entry
    assert "config/training.yml" in entry
    assert 'table_name: "{catalog}.{input_schema}.balances"' in entry


def test_automatic_threshold_error_and_numeric_fix_are_documented(workflow_config):
    """Automatic activation must fail before remote work when its quality gate is absent."""
    settings = {**workflow_config, "promotion_policy": "automatic", "quality_threshold": None}
    message = "Automatic promotion requires quality_threshold."
    with pytest.raises(ValueError) as error:
        validate_workflow_config(settings, action="train")
    assert str(error.value) == message
    fixed = {**settings, "quality_threshold": 5.0}
    assert validate_workflow_config(fixed, action="train") == fixed
    entry = _diagnostic("automatic-threshold")
    assert message in entry
    assert "config/training.yml" in entry
    assert "metric: heldout_rmse\n  quality_threshold: 5.0" in entry
