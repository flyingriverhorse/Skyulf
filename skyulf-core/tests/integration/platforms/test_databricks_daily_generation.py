"""Real offline CLI generation must carry daily controls into each training layout."""

import pytest
from test_databricks_bundle_generation import (
    CLI,
    OFFLINE_CLI,
    PROFILE,
    _generate_project,
    _read_validated_config,
)

pytestmark = pytest.mark.skipif(
    not (PROFILE or OFFLINE_CLI) or not CLI,
    reason="Set SKYULF_BUNDLE_CLI_TEST_PROFILE to opt into installed CLI generation.",
)


@pytest.mark.parametrize("strategy", ["random", "temporal"])
@pytest.mark.parametrize("layout", ["single_model", "model_competition", "multi_target"])
def test_daily_windows_render_in_all_training_layouts(tmp_path, strategy, layout):
    """Generated root and branch configs must retain actual event names and active day counts."""
    settings = {
        "split_strategy": strategy,
        "training_window_mode": "rolling_days",
        "event_column": "observed_on",
        "lookback_days": "90",
        "holdout_days": "14",
    }
    if layout == "multi_target":
        settings = {
            f"branch_{slot}_{key}": value for slot in (1, 2) for key, value in settings.items()
        }
    project = _generate_project(tmp_path, training_layout=layout, **settings)
    config = _read_validated_config(project)
    configs = [config]
    if layout == "multi_target":
        from skyulf.integrations.databricks.projects.yaml_config import read_training_config
        from skyulf.integrations.databricks.projects.yaml_models import training_branches

        declarations = read_training_config(project / "config")
        assert declarations is not None
        branches, _ = training_branches(declarations)
        configs = [model["workflow"] for model in branches.values()]
    for workflow in configs:
        assert workflow["training_window_mode"] == "rolling_days"
        assert workflow["lookback_days"] == 90
        assert workflow.get("holdout_days") == (14 if strategy == "temporal" else None)
        assert workflow["event_column"] == "observed_on"
        assert workflow.get("monthly_lookback_months") is None
        assert workflow.get("holdout_months") is None
        assert workflow.get("window_timezone") is None
