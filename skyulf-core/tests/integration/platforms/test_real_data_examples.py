"""Exercise the prepared real-data projects that demonstrate custom Spark admission."""

import json
from pathlib import Path

import pytest
import yaml

ROOT = Path(__file__).resolve().parents[3] / "templates/databricks/examples/real_data"


@pytest.mark.parametrize("name", ["diagnostic_single", "bike_demand_set", "air_quality_no2"])
def test_real_data_example_runs_custom_code_on_spark(name):
    """Both examples must exercise custom admission rather than silently scoring locally."""
    answers = json.loads((ROOT / f"{name}.json").read_text(encoding="utf-8"))
    project = ROOT / name
    training = yaml.safe_load((project / "config/training.yml").read_text(encoding="utf-8"))
    inference = yaml.safe_load((project / "config/inference.yml").read_text(encoding="utf-8"))
    recipes = yaml.safe_load((project / "config/preprocessing.yml").read_text(encoding="utf-8"))
    assert answers["batch_compute"] == inference["batch_compute"] == "spark"
    assert answers["shap_enabled"] == answers["evaluation_charts_enabled"] == "true"
    assert answers["schema"] == name
    expected_store = "skyulf_monitoring" if name == "air_quality_no2" else name
    assert answers["monitoring_schema"] == expected_store
    for entry in training["models"].values():
        config = training["defaults"] | entry
        assert config["target_column"] not in config["input_columns"]
        recipe = recipes["recipes"][config.get("preprocessing_recipe", "default")]
        assert any("custom" in step for step in recipe)
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


def test_air_quality_inputs_exclude_reference_analyzers_that_leak():
    """NOx contains the NO2 target and C6H6 is derived from a sensor, so neither may be an input."""
    training = yaml.safe_load(
        (ROOT / "air_quality_no2/config/training.yml").read_text(encoding="utf-8")
    )
    config = training["defaults"] | training["models"]["main"]
    assert config["split_strategy"] == "temporal"
    assert config["target_column"] == "no2_gt"
    assert not {"nox_gt", "c6h6_gt", "nmhc_gt", "no2_gt"}.intersection(config["input_columns"])


def _loader_functions():
    """Load notebook functions without starting Spark or executing notebook widgets."""
    import ast

    import pandas as pd

    path = ROOT / "load_data.py"
    parsed = ast.parse(path.read_text(encoding="utf-8"))
    definitions = ast.Module(
        body=[node for node in parsed.body if isinstance(node, ast.FunctionDef)], type_ignores=[]
    )
    namespace = {"pd": pd}
    exec(compile(definitions, str(path), "exec"), namespace)
    return namespace


def test_air_quality_single_table_selects_seed_or_one_winter_day():
    """Loading a date must neither replay summer rows nor include the next day's boundary."""
    import pandas as pd

    functions = _loader_functions()
    assert "air_quality_batch" in functions, (
        "The loader needs an explicit single-table day selector."
    )
    frame = pd.DataFrame(
        {
            "record_id": [1, 2, 3, 4],
            "observed_at": pd.to_datetime(
                ["2004-10-31T23:00Z", "2004-11-01T00:00Z", "2004-11-01T23:00Z", "2004-11-02T00:00Z"]
            ),
            "no2_gt": [50.0, 60.0, -200.0, 70.0],
        }
    )
    select = functions["air_quality_batch"]
    assert select(frame, "")["record_id"].tolist() == [1]
    assert select(frame, "2004-11-01")["record_id"].tolist() == [2, 3]
    assert select(frame, "2004-11-02")["record_id"].tolist() == [4]
    assert frame["no2_gt"].tolist() == [50.0, 60.0, -200.0, 70.0]


@pytest.mark.parametrize(
    "day", ["2004-10-31", "2004-11-1", "2004-11-01T12:00Z", "invalid", "2006-01-01"]
)
def test_air_quality_single_table_rejects_invalid_or_absent_days(day):
    """A typo or summer date must fail before any append rather than create an empty batch."""
    import pandas as pd

    functions = _loader_functions()
    assert "air_quality_batch" in functions, (
        "The loader needs an explicit single-table day selector."
    )
    frame = pd.DataFrame({"observed_at": pd.to_datetime(["2004-11-01T00:00Z"])})
    with pytest.raises(ValueError, match="day"):
        functions["air_quality_batch"](frame, day)


def _air_quality_rows(count):
    """Plausible station hours with the -200 marker in the often offline CO analyzer."""
    import pandas as pd

    hours = [float(i % 24) for i in range(count)]
    return pd.DataFrame(
        {
            "co_gt": [-200.0 if i % 9 == 0 else 1.0 + (i % 7) / 2 for i in range(count)],
            "pt08_s1_co": [900.0 + (i % 40) * 10 for i in range(count)],
            "pt08_s2_nmhc": [800.0 + (i % 30) * 12 for i in range(count)],
            "pt08_s3_nox": [700.0 + (i % 20) * 15 for i in range(count)],
            "pt08_s4_no2": [1300.0 + (i % 25) * 20 for i in range(count)],
            "pt08_s5_o3": [900.0 + (i % 35) * 18 for i in range(count)],
            "temperature": [12.0 + (i % 16) for i in range(count)],
            "rel_humidity": [30.0 + (i % 40) for i in range(count)],
            "abs_humidity": [0.6 + (i % 12) / 10 for i in range(count)],
            "hour": hours,
            "weekday": [float(i // 24 % 7) for i in range(count)],
            "no2_gt": [60.0 + (i % 20) * 4 for i in range(count)],
        }
    )


def test_air_quality_recipe_seals_admission_and_explains_every_excluded_row(tmp_path):
    """Custom and built-in steps must pass Spark admission and keep every request row visible."""
    from skyulf.data.dataset import SplitDataset
    from skyulf.inference.fitted_pipeline import load_pipeline
    from skyulf.inference.partition_safety import require_partition_safe_pipeline
    from skyulf.inference.pipeline_scoring import score_pipeline
    from skyulf.integrations.databricks.lifecycle.workflow import training_spec
    from skyulf.integrations.databricks.projects.project import load_project_workflow
    from skyulf.integrations.databricks.projects.yaml_config import read_workflow_config
    from skyulf.integrations.databricks.scoring.batch.frame_batch import fit_workflow
    from skyulf.integrations.databricks.training.fitting.candidate import candidate_config
    from skyulf.integrations.databricks.training.shared.preprocessing_checks import (
        seal_partition_probe,
    )
    from skyulf.integrations.databricks.training.tuning.cv import CVSpec

    project = ROOT / "air_quality_no2"
    config = load_project_workflow(
        read_workflow_config(project / "config/training.yml"), project / "src/features"
    )
    spec = training_spec(config | {"training_table": "workspace.test.aq", "training_version": 0})
    assert spec.admit_probed_custom_steps is True  # Spark mode admits automatically.
    pipeline = candidate_config(
        spec,
        config["pipeline"],
        engine="pandas",
        cv=CVSpec.from_workflow(config),
        metric="heldout_rmse",
        min_improvement=0.0,
        champion_version=None,
        quality_threshold=60.0,
        risk_category=None,
    )
    rows = _air_quality_rows(240)
    path = tmp_path / "artifact"
    fitted = fit_workflow(
        pipeline,
        SplitDataset(train=rows.iloc[:180], test=rows.iloc[180:]),
        target_column="no2_gt",
        artifact_path=path,
        max_rows=500,
        max_bytes=4 * 1024 * 1024,
    )
    seal_partition_probe(fitted, rows.iloc[180:], path, enabled=True)
    artifact = load_pipeline(path)
    require_partition_safe_pipeline(artifact)
    requests = rows.iloc[:5].drop(columns="no2_gt").reset_index(drop=True)
    requests.loc[1, "pt08_s4_no2"] = -200.0  # Offline NO2 sensor: reused pre-split filter.
    requests.loc[2, "pt08_s3_nox"] = 9000.0  # Far outside the fitted IQR bounds.
    requests.loc[3, "rel_humidity"] = 5.0  # Custom eligibility rule.
    requests.loc[4, "co_gt"] = -200.0  # Offline CO analyzer: marked missing, then imputed.
    scored = score_pipeline(requests, artifact)
    assert scored["scoring_status"].tolist() == [
        "predicted",
        "excluded",
        "excluded",
        "excluded",
        "predicted",
    ]
    reasons = scored["exclusion_reason"].tolist()
    assert "sensors_online" in reasons[1]
    assert reasons[2] == "outlier:nox_sensor_outliers"
    assert reasons[3] == "humidity_out_of_range"
    assert scored["prediction"].isna().tolist() == [False, True, True, True, False]
    assert scored.loc[[0, 4], "no2_band"].notna().all()
