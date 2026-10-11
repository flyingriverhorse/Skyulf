"""Demo acceptance evidence must remain valid under optimized Python execution."""

import importlib
from importlib import util
from pathlib import Path
from types import SimpleNamespace

import pandas as pd
import pytest

from skyulf.integrations.databricks.serving import PinnedEndpointSpec


@pytest.fixture
def optimized_demo(monkeypatch):
    """Execute the actual demo module with the same assertion removal as python -O."""
    root = Path(__file__).resolve().parents[3] / "examples/databricks_raw_serving/src"
    monkeypatch.syspath_prepend(str(root))
    module = importlib.import_module("raw_serving_demo.scoring")
    assert module.__file__ is not None
    spec = util.spec_from_file_location(module.__name__, module.__file__)
    assert spec is not None
    optimized = util.module_from_spec(spec)
    source = Path(module.__file__).read_text(encoding="utf-8")
    exec(compile(source, module.__file__, "exec", optimize=1), optimized.__dict__)
    return optimized


@pytest.mark.parametrize(
    "defect, message",
    [
        (None, None),
        ("key", "Exact key mismatch"),
        ("model", "Model version mismatch"),
        ("version", "Delta version mismatch"),
        ("null", "Null-input case missing"),
        ("category", "Unseen-category case missing"),
        ("columns", "Endpoint must accept only raw columns"),
        ("batch", "Unexpected batch publication"),
    ],
)
def test_optimized_demo_rejects_inconsistent_acceptance_evidence(
    monkeypatch, optimized_demo, defect, message
):
    """A mismatched identity or missing edge case must never produce a PASS receipt."""
    source = pd.DataFrame(
        {"customer_id": [9007199254740993], "age": [None], "segment": ["NewSegment"]}
    )
    spec = PinnedEndpointSpec("demo", "workspace.demo.customer", "1", "workspace", "demo", "logs")
    plan = SimpleNamespace(spec=spec, input_columns=("age", "segment"))
    actual = pd.DataFrame(
        {
            "customer_id": source.customer_id,
            "model_uri": [spec.model_uri],
            "source_version": [0],
            "prediction": [0.5],
        }
    )
    config = {
        "prediction_rows": 1,
        "raw_columns": ["age", "segment"],
        "write_batch_predictions": False,
    }
    if defect in {"key", "model", "version"}:
        field, value = {
            "key": ("customer_id", 3),
            "model": ("model_uri", "models:/other/1"),
            "version": ("source_version", 1),
        }[defect]
        actual[field] = value
    elif defect == "null":
        source["age"] = 40.0
    elif defect == "category":
        source["segment"] = "Mass"
    elif defect == "columns":
        plan.input_columns = ("segment", "age")
    spark = SimpleNamespace(
        table=lambda name: SimpleNamespace(orderBy=lambda column: actual),
        catalog=SimpleNamespace(tableExists=lambda name: defect == "batch"),
    )
    monkeypatch.setattr(optimized_demo, "require_pinned_endpoint_ready", lambda *args: None)
    monkeypatch.setattr(optimized_demo, "read_cohort", lambda *args: source)
    monkeypatch.setattr(optimized_demo, "bounded_pandas", lambda frame, limit: frame)
    monkeypatch.setattr(optimized_demo, "resolve_model", lambda *args, **kwargs: None)
    monkeypatch.setattr(
        optimized_demo,
        "load_registered_pipeline",
        lambda *args, **kwargs: SimpleNamespace(
            manifest=SimpleNamespace(feature_columns=("age", "segment"))
        ),
    )
    monkeypatch.setattr(
        optimized_demo, "score_pipeline", lambda *args: pd.DataFrame({"prediction": [0.5]})
    )
    if defect is None:
        receipt = optimized_demo.verify(spark, None, plan, "workspace.demo", 0, config)
        assert receipt["status"] == "PASS"
        assert receipt["rows_per_method"] == 1
        assert receipt["maximum_absolute_error"] == {"rest": 0.0, "sql": 0.0}
    else:
        with pytest.raises(ValueError, match=message):
            optimized_demo.verify(spark, None, plan, "workspace.demo", 0, config)


@pytest.mark.parametrize("version", [True, -1, "0; DROP TABLE other", 0.5])
def test_sql_demo_rejects_invalid_versions_before_any_external_call(optimized_demo, version):
    """Direct callers cannot interpolate an unchecked snapshot selector into SQL."""
    with pytest.raises(ValueError, match="Delta version"):
        optimized_demo.sql_predictions(None, None, None, "workspace.demo", version)
