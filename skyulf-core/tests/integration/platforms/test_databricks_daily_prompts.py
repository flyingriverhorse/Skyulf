"""Elapsed-day selection must collect explicit event and split settings in every layout."""

import json
from pathlib import Path

import pytest
from jsonschema import Draft7Validator

from skyulf.integrations.databricks.projects.workflow_config import preview_workflow_config

ROOT = Path(__file__).resolve().parents[3] / "templates/databricks"


@pytest.mark.parametrize("branch", [False, True])
@pytest.mark.parametrize("strategy", ["random", "temporal"])
@pytest.mark.parametrize("temporal_cv", [False, True])
def test_daily_selection_prompts_for_active_controls(branch, strategy, temporal_cv):
    """Root and branch daily windows must expose the actual date column and active duration."""
    fields = json.loads((ROOT / "databricks_template_schema.json").read_text())["properties"]
    values = {key: field["default"] for key, field in fields.items()}
    prefix = "branch_1_" if branch else ""
    values.update(training_layout="multi_target" if branch else "single_model")
    values.update(
        {
            prefix + "training_window_mode": "rolling_days",
            prefix + "split_strategy": strategy,
            prefix + "cv_enabled": "true" if temporal_cv else "false",
            prefix + "cv_type": "time_series_split" if temporal_cv else "k_fold",
        }
    )
    visible = {
        name
        for name, field in fields.items()
        if not Draft7Validator(field.get("skip_prompt_if", False)).is_valid(values)
    }
    assert {
        prefix + name for name in ("training_window_mode", "lookback_days", "event_column")
    } <= visible
    assert (prefix + "holdout_days" in visible) == (strategy == "temporal" or temporal_cv)
    assert {
        prefix + name
        for name in (
            "monthly_lookback_months",
            "holdout_months",
            "window_timezone",
            "start",
            "cutoff",
            "holdout_start",
        )
    }.isdisjoint(visible)


@pytest.mark.parametrize("strategy", ["random", "temporal"])
def test_daily_preview_accepts_fields_and_explains_runtime_boundaries(workflow_config, strategy):
    """Offline validation and previews must preserve daily controls before any source read."""
    workflow_config.update(
        training_window_mode="rolling_days",
        split_strategy=strategy,
        event_column="observed_on",
        lookback_days=90,
        holdout_days=14 if strategy == "temporal" else None,
        monthly_lookback_months=None,
        holdout_months=None,
        window_timezone=None,
    )
    report = preview_workflow_config(workflow_config, action="train")
    assert "last 90 elapsed UTC days" in report
    assert "Event column: observed_on" in report
    assert ("minus 14 elapsed days" in report) == (strategy == "temporal")
