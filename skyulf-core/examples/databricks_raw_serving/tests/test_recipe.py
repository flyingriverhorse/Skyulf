"""The example must serve raw features using exactly the training transformations."""

import json
import sys
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import Mock

import numpy as np
import pandas as pd
import pytest
import yaml

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from raw_serving_demo import scoring
from raw_serving_demo.training import fit_customer_pipeline

from skyulf.inference.local_scoring import score_local_pipeline
from skyulf.integrations.mlflow.spark.spark_model import partition_safety_certificate


@pytest.fixture
def fitted(tmp_path):
    """Exercise the actual example YAML rather than a second test-only recipe."""
    config = yaml.safe_load((Path(__file__).resolve().parents[1] / "config.yml").read_text())
    values = np.arange(96)
    frame = pd.DataFrame(
        {
            "age": 20.0 + values % 50,
            "income": 20000.0 + values * 500,
            "tenure": 1.0 + values % 30,
            "segment": np.where(values % 2, "Mass", "Business"),
            "churn": values % 2,
        }
    )
    frame.loc[[2, 9], "income"] = np.nan
    artifact, holdout = fit_customer_pipeline(frame, config, tmp_path / "artifact")
    return artifact, holdout, config


def test_complete_example_accepts_raw_nulls_and_unknown_categories(fitted):
    """No caller-generated feature or merged table may be required for this model."""
    artifact, holdout, config = fitted
    raw = holdout[config["raw_columns"]].iloc[:3].reset_index(drop=True)
    raw.loc[0, "segment"] = "NewSegment"
    raw.loc[1, ["income", "age", "tenure"]] = np.nan
    before = partition_safety_certificate(artifact)
    expected = score_local_pipeline(raw, artifact)
    separate = pd.concat([score_local_pipeline(raw.iloc[[i]], artifact) for i in range(3)])
    pd.testing.assert_frame_equal(expected, separate)
    np.testing.assert_allclose(expected[["probability_0", "probability_1"]].sum(axis=1), 1)
    assert "income_x_tenure" in artifact.manifest.feature_columns
    assert list(artifact.manifest.input_columns) == config["raw_columns"]
    assert partition_safety_certificate(artifact) == before


def test_rest_uses_sdk_response_and_preserves_exact_keys(monkeypatch):
    """The demo must consume the real SDK response shape and keep IDs outside JSON features."""
    raw = pd.DataFrame({"customer_id": [2**53 + 1, 2**53 + 2], "age": [None, 35.0]})
    monkeypatch.setattr(scoring, "read_cohort", Mock(return_value=object()))
    monkeypatch.setattr(scoring, "bounded_pandas", Mock(return_value=raw))
    query = Mock(return_value=SimpleNamespace(predictions=[{"prediction": 1}, {"prediction": 0}]))
    monkeypatch.setattr(scoring, "query_named_records", query)
    writer = Mock()
    monkeypatch.setattr(scoring, "write_predictions", writer)
    plan = SimpleNamespace(
        input_columns=("age",),
        output_schema=(("prediction", "int64"),),
        spec=SimpleNamespace(model_uri="models:/workspace.demo.model/1"),
    )
    result = scoring.rest_predictions(
        Mock(), Mock(), plan, "workspace.demo", 0, {"prediction_rows": 2}
    )
    assert query.call_args.args[2] == [{"age": None}, {"age": 35.0}]
    assert writer.call_args.args[1] == [2**53 + 1, 2**53 + 2]
    assert result["rows"] == 2
    assert json.loads(json.dumps(query.call_args.args[2]))[0]["age"] is None


def test_reviewed_sql_definition_requires_readiness_before_any_sql(monkeypatch):
    """An external SQL compilation cannot bypass the pinned endpoint readiness check."""
    ready = Mock(side_effect=ValueError("endpoint changed"))
    monkeypatch.setattr(scoring, "require_pinned_endpoint_ready", ready)
    monkeypatch.setattr(scoring, "build_serving_sql_function", Mock())
    plan = SimpleNamespace(spec=SimpleNamespace(model_version="1"))
    spark = Mock()
    with pytest.raises(ValueError, match="endpoint changed"):
        scoring.sql_predictions(
            spark, Mock(), plan, "workspace.demo", 0, function_ddl="CREATE FUNCTION reviewed"
        )
    spark.sql.assert_not_called()


def test_reviewed_sql_definition_is_executed_without_regenerating_it(monkeypatch):
    """A fixed SQL compiler can serve an unchanged model runtime without altering its code."""
    function = Mock(function_name="workspace.demo.predict_customer_v1")
    function.call_sql.return_value = "workspace.demo.predict_customer_v1(age)"
    monkeypatch.setattr(scoring, "build_serving_sql_function", Mock(return_value=function))
    monkeypatch.setattr(scoring, "require_pinned_endpoint_ready", Mock())
    create = Mock()
    monkeypatch.setattr(scoring, "create_serving_sql_function", create)
    plan = SimpleNamespace(spec=SimpleNamespace(model_version="1", model_uri="models:/m/1"))
    spark = Mock()
    spark.table.return_value.count.return_value = 128
    result = scoring.sql_predictions(
        spark, Mock(), plan, "workspace.demo", 0, function_ddl="CREATE FUNCTION reviewed"
    )
    assert spark.sql.call_args_list[0].args == ("CREATE FUNCTION reviewed",)
    create.assert_not_called()
    assert result["rows"] == 128 and result["table"] == "workspace.demo.predictions_ai_query"
