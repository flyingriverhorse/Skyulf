"""Saved scoring rules run on Spark and endpoints only with sealed lint and probe evidence."""

import re
import sys
from copy import deepcopy
from dataclasses import asdict
from types import ModuleType, SimpleNamespace
from typing import Any, cast

import pandas as pd
import pytest

from skyulf.core.capabilities import UnsupportedExecutionError
from skyulf.data.dataset import SplitDataset
from skyulf.inference.partition_safety import require_partition_safe_pipeline
from skyulf.inference.pipeline_scoring import score_pipeline
from skyulf.integrations.databricks.projects.project import load_project_workflow
from skyulf.integrations.databricks.scoring.batch.frame_batch import fit_workflow
from skyulf.integrations.databricks.training.shared.preprocessing_checks import (
    seal_partition_probe,
)

SOURCE = '''
import numpy as np
import pandas as pd
from skyulf.preprocessing import column_step


def doubled(df):
    """Row-local feature."""
    return df["x"] * 2.0


def too_large(frame, params):
    """Exclude rows above a saved limit."""
__ELIGIBILITY__


def band(frame, predictions, params):
    """Label each prediction against a saved cut."""
__OUTPUT__


def build_preprocessing(recipe="default"):
    """One custom row step."""
    return [column_step("double_x", doubled, output="z")]


def build_scoring():
    """Saved eligibility and output rules."""
    return {
        "eligibility": [
            {"name": "too_large", "version": "1", "function": "steps.too_large",
             "params": {"limit": 120.0}}
        ],
        "outputs": [
            {"name": "band", "version": "1", "function": "steps.band",
             "params": {"cut": 0.0}, "columns": [{"name": "band", "dtype": "string"}]}
        ],
    }
'''
ELIGIBILITY = {
    "row": 'return pd.Series(np.where(frame["x"] > params["limit"], "too_large", None), '
    "index=frame.index)",
    "lint": 'return pd.Series(np.where(frame["x"] > frame["x"].mean(), "too_large", None), '
    "index=frame.index)",
    "hidden": 'limit = getattr(frame["x"], "mean")()\n    return pd.Series(np.where('
    'frame["x"] > limit, "too_large", None), index=frame.index)',
}
OUTPUT = {
    "row": 'return pd.DataFrame({"band": np.where(predictions["prediction"] > params["cut"], '
    '"high", "low")}, index=frame.index)',
    "empty_fail": 'if frame.empty:\n        raise ValueError("no rows")\n    return '
    'pd.DataFrame({"band": np.where(predictions["prediction"] > params["cut"], "high", '
    '"low")}, index=frame.index)',
}
HOLDOUT = pd.DataFrame(
    {"x": [100.0 + ((7 * i) % 20) * 1.25 for i in range(20)], "target": [0.0] * 20}
)


def _sealed(tmp_path, eligibility="row", output="row", *, enabled=True):
    """Fit a project with saved rules and run the training admission probe."""
    root = tmp_path / "features"
    root.mkdir()
    (root / "__init__.py").write_text(
        "from .steps import build_preprocessing, build_scoring\n", encoding="utf-8"
    )
    source = SOURCE.replace("__ELIGIBILITY__", "    " + ELIGIBILITY[eligibility])
    source = source.replace("__OUTPUT__", "    " + OUTPUT[output])
    (root / "steps.py").write_text(source + f"\n# Project: {tmp_path.name}\n", encoding="utf-8")
    config = load_project_workflow(
        {"pipeline": {"preprocessing": [], "modeling": {"type": "linear_regression"}}}, root
    )
    pipeline = config["pipeline"]
    pipeline["admit_probed_custom_steps"] = enabled
    rows = pd.DataFrame({"x": [1.0, 2.0, 3.0, 4.0], "target": [3.0, 5.0, 7.0, 9.0]})
    path = tmp_path / "artifact"
    artifact = fit_workflow(
        pipeline,
        SplitDataset(train=rows, test=rows[:0]),
        target_column="target",
        artifact_path=path,
        max_rows=10,
        max_bytes=10000,
    )
    return seal_partition_probe(artifact, HOLDOUT, path, enabled=enabled)


def test_row_local_rules_are_admitted_with_rule_evidence(tmp_path):
    """Rules that pass lint and every probe check are certified next to the steps."""
    artifact = _sealed(tmp_path)
    rules = artifact.pipeline.config["partition_probe"]["rules"]
    assert [(rule["name"], rule["kind"]) for rule in rules] == [
        ("too_large", "eligibility"),
        ("band", "outputs"),
    ]
    assert {check["name"] for check in rules[0]["checks"]} >= {"lint", "perturb", "empty"}
    evidence = asdict(require_partition_safe_pipeline(artifact))
    assert [(s["name"], s["node_type"], s["action"]) for s in evidence["steps"]] == [
        ("double_x", "ColumnFunction", "apply_probed"),
        ("too_large", "scoring_rule:eligibility", "rules_probed"),
        ("band", "scoring_rule:outputs", "rules_probed"),
    ]
    assert [name for name, _ in evidence["output_schema"]] == [
        "prediction",
        "band",
        "scoring_status",
        "exclusion_reason",
    ]
    scored = score_pipeline(HOLDOUT.loc[:, ["x"]], artifact)
    assert scored["scoring_status"].eq("excluded").sum() == 3


def _fake_spark_types(monkeypatch):
    """Expose MLflow's real struct converter without a Spark JVM."""
    types = cast(Any, ModuleType("pyspark.sql.types"))
    for name in ("IntegerType", "LongType", "FloatType", "DoubleType", "BooleanType"):
        setattr(types, name, type(name, (), {}))
    types.StringType = type("StringType", (), {})
    types.ArrayType = type("ArrayType", (), {})
    types.MapType = type("MapType", (), {})

    class StructType:
        """Field lookup protocol consumed by MLflow."""

        def __init__(self, fields):
            self.fields = fields

        def fieldNames(self):
            """Return ordered field names."""
            return [field.name for field in self.fields]

        def __getitem__(self, name):
            """Return one named field."""
            return next(field for field in self.fields if field.name == name)

    types.StructType = StructType
    sql = cast(Any, ModuleType("pyspark.sql"))
    sql.types = types
    monkeypatch.setitem(sys.modules, "pyspark", ModuleType("pyspark"))
    monkeypatch.setitem(sys.modules, "pyspark.sql", sql)
    monkeypatch.setitem(sys.modules, "pyspark.sql.types", types)
    return types


def test_spark_output_and_endpoint_schema_carry_rule_columns(tmp_path, monkeypatch):
    """Worker output keeps nulls for excluded rows through MLflow's Spark struct conversion."""
    mlflow = pytest.importorskip("mlflow")
    from skyulf.integrations.databricks.serving.endpoints import _artifact_schema
    from skyulf.integrations.mlflow.models.pipeline_model import SkyulfPipelinePythonModel
    from skyulf.integrations.mlflow.spark import spark_model
    from skyulf.integrations.mlflow.spark._spark_output import SPARK_NULL_TEXT

    artifact = _sealed(tmp_path)
    _, outputs = _artifact_schema(artifact)
    assert [(c.name, c.dtype) for c in outputs] == [
        ("prediction", "float64"),
        ("band", "string"),
        ("scoring_status", "string"),
        ("exclusion_reason", "string"),
    ]
    model = SkyulfPipelinePythonModel(
        None,
        spark_model.partition_safety_certificate(artifact),
        spark_model.runtime_source_digest(),
    )
    model.load_context(SimpleNamespace(artifacts={"local_pipeline": str(tmp_path / "artifact")}))
    query = HOLDOUT.loc[:, ["x"]]
    local = model.predict(None, query)
    worker = model.predict(None, query, params={"spark_output": True})
    excluded = local["scoring_status"].eq("excluded")
    assert excluded.sum() == 3 and local.loc[excluded, "prediction"].isna().all()
    assert worker.loc[~excluded, "exclusion_reason"].eq(SPARK_NULL_TEXT).all()
    assert worker.loc[excluded, "prediction"].isna().all()
    types = _fake_spark_types(monkeypatch)
    schema = types.StructType(
        [
            SimpleNamespace(
                name=c.name,
                dataType=types.StringType() if c.dtype == "string" else types.DoubleType(),
            )
            for c in outputs
        ]
    )
    converted = mlflow.pyfunc._convert_struct_values(worker, schema)
    assert converted.loc[~excluded, "exclusion_reason"].eq(SPARK_NULL_TEXT).all()
    assert converted.loc[excluded, "exclusion_reason"].eq("too_large").all()
    assert converted.loc[excluded, "band"].eq(SPARK_NULL_TEXT).all()
    assert converted.loc[excluded, "prediction"].isna().all()


def test_rule_reading_frame_statistic_fails_lint(tmp_path):
    """A rule comparing against the batch mean is rejected with its file, line and call."""
    with pytest.raises(ValueError) as error:
        _sealed(tmp_path, eligibility="lint")
    message = str(error.value).replace('"', "'")
    assert re.search(
        r"Scoring rule 'too_large' reads a batch statistic \(steps\.py line \d+: "
        r"frame\['x'\]\.mean\(\)\)",
        message,
    ), message


def test_rule_hiding_a_frame_statistic_fails_the_probe(tmp_path):
    """The dynamic probe catches what the lint cannot see."""
    with pytest.raises(
        ValueError, match=r"Scoring rule 'too_large' changed results when scored in chunks of 1"
    ):
        _sealed(tmp_path, eligibility="hidden")


def test_output_rule_failing_on_empty_batch_fails_training(tmp_path):
    """Spark can send an empty batch to an output rule, so it must handle empty input."""
    with pytest.raises(
        ValueError, match=r"Scoring rule 'band' failed on an empty batch \(empty: not_supported\)"
    ):
        _sealed(tmp_path, output="empty_fail")


def test_changed_rule_params_invalidate_evidence(tmp_path):
    """Evidence binds each rule's params, so a changed limit is no longer admitted."""
    artifact = _sealed(tmp_path)
    artifact.pipeline.config = deepcopy(artifact.pipeline.config)
    artifact.pipeline.config["project_scoring"]["eligibility"][0]["params"]["limit"] = 1.0
    with pytest.raises(
        UnsupportedExecutionError,
        match="Scoring rule 'too_large': Training probe evidence does not match this saved rule",
    ):
        require_partition_safe_pipeline(artifact)


def test_rules_without_evidence_are_rejected(tmp_path):
    """A local project's rules were never probed, so Spark and endpoints reject them."""
    artifact = _sealed(tmp_path, enabled=False)
    with pytest.raises(UnsupportedExecutionError, match="admit_probed_custom_steps: true"):
        require_partition_safe_pipeline(artifact)


def test_reused_pre_split_without_evidence_is_rejected(tmp_path):
    """A pre-split recipe added after training has no sealed evidence and is not admitted."""
    artifact = _sealed(tmp_path)
    artifact.pipeline.config = deepcopy(artifact.pipeline.config)
    artifact.pipeline.config["project_scoring"]["pre_split"] = {"steps": []}
    with pytest.raises(
        UnsupportedExecutionError, match="Reused pre-split eligibility: .*has no entry"
    ):
        require_partition_safe_pipeline(artifact)


@pytest.mark.parametrize(
    "field,value",
    [("version", 1), ("float_tolerance", None), ("float_tolerance", {"rtol": 1e-6, "atol": 1e-12})],
)
def test_old_or_changed_comparison_policy_is_rejected(tmp_path, field, value):
    """Every admission route must require the trusted versioned tolerance policy."""
    artifact = _sealed(tmp_path)
    assert artifact.pipeline.config["partition_probe"]["version"] == 2
    assert artifact.pipeline.config["partition_probe"]["float_tolerance"] == {
        "rtol": 1e-9,
        "atol": 1e-12,
    }
    artifact.pipeline.config["partition_probe"][field] = value
    with pytest.raises(UnsupportedExecutionError, match="comparison policy"):
        require_partition_safe_pipeline(artifact)
