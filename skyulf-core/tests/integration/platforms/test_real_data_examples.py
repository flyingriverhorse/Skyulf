"""Exercise the prepared real-data projects that demonstrate custom Spark admission."""

import json
from pathlib import Path

import pytest
import yaml

ROOT = Path(__file__).resolve().parents[3] / "templates/databricks/examples/real_data"


@pytest.mark.parametrize("name", ["diagnostic_single", "bike_demand_set"])
def test_real_data_example_runs_custom_code_on_spark(name):
    """Both examples must exercise custom admission rather than silently scoring locally."""
    answers = json.loads((ROOT / f"{name}.json").read_text(encoding="utf-8"))
    project = ROOT / name
    training = yaml.safe_load((project / "config/training.yml").read_text(encoding="utf-8"))
    inference = yaml.safe_load((project / "config/inference.yml").read_text(encoding="utf-8"))
    recipes = yaml.safe_load((project / "config/preprocessing.yml").read_text(encoding="utf-8"))
    assert answers["inference_mode"] == inference["inference_mode"] == "spark"
    assert answers["shap_enabled"] == answers["evaluation_charts_enabled"] == "true"
    assert answers["monitoring_schema"] == answers["schema"] == name
    for entry in training["models"].values():
        config = training["defaults"] | entry
        assert config["target_column"] not in config["input_columns"]
        assert any("custom" in step for step in recipes["recipes"][config["preprocessing_recipe"]])
        assert config["explainability"]["method"] == "shap"


def test_bike_targets_and_total_are_not_training_inputs():
    """Neither demand target nor their observed total may leak into either branch's inputs."""
    training = yaml.safe_load(
        (ROOT / "bike_demand_set/config/training.yml").read_text(encoding="utf-8")
    )
    assert set(training["models"]) == {"casual", "registered"}
    for entry in training["models"].values():
        config = training["defaults"] | entry
        assert config["split_strategy"] == "temporal"
        assert not {"casual", "registered", "cnt"}.intersection(config["input_columns"])
