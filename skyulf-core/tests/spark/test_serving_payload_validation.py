"""Serving validation keeps error precedence without a scan for every feature."""

import time

import pytest
from tests.spark.test_serving_payloads import NOW, PAYLOAD_SCHEMA, _parse, _row

from skyulf.integrations.databricks.observability.monitoring.serving import serving_payloads

ENTITY_SCHEMA = (
    "served_entity_id string, endpoint_name string, entity_name string, entity_version string"
)
ENTITY = ("entity-1", "endpoint", "cat.sch.model", "4")


@pytest.mark.parametrize("width", [1, 40])
def test_validation_actions_do_not_grow_with_feature_count(
    spark, monkeypatch, record_property, width
):
    """Wide model schemas must not trigger one distributed scan per typed feature."""
    from datetime import timedelta

    features = {f"feature_{index}": index + 0.5 for index in range(width)}
    payloads = spark.createDataFrame(
        [_row("wide", [features], [{"out_prediction": 2}])], PAYLOAD_SCHEMA
    )
    entities = spark.createDataFrame([ENTITY], ENTITY_SCHEMA)
    actions = []
    depth = 0

    def track(method):
        """Count outermost real Spark actions without replacing their execution."""

        def execute(frame, *args, **kwargs):
            """Avoid double-counting count implementations which delegate to collect."""
            nonlocal depth
            if depth == 0:
                actions.append(method.__name__)
            depth += 1
            try:
                return method(frame, *args, **kwargs)
            finally:
                depth -= 1

        return execute

    started = time.monotonic()
    with monkeypatch.context() as patch:
        frame_type = type(payloads)
        for name in ("collect", "count"):
            patch.setattr(frame_type, name, track(getattr(frame_type, name)))
        current, predictions, summary, observed = serving_payloads.parse_serving_payloads(
            payloads,
            entities,
            endpoint_name="endpoint",
            model_name="cat.sch.model",
            model_version="4",
            input_columns=tuple((name, "double") for name in features),
            output_columns=(("out_prediction", "double"),),
            output_prefix="out_",
            start=NOW - timedelta(hours=1),
            end=NOW + timedelta(hours=1),
        )
    record_property("parser_seconds", time.monotonic() - started)
    record_property("spark_actions", len(actions))
    assert current.first().asDict() == {
        "databricks_request_id": "wide",
        "request_row_index": 0,
        **features,
    }
    assert predictions.first().prediction == 2.0
    assert summary["request_count"] == summary["prediction_row_count"] == 1
    assert observed == NOW
    assert 0 < len(actions) <= 8, actions


@pytest.mark.parametrize(
    "ids,entities,message",
    [
        (
            [None, "missing", "entity-1"],
            [ENTITY, ("entity-1", None, None, None)],
            "served entity identity is missing",
        ),
        (
            ["missing", "entity-1"],
            [ENTITY, ("entity-1", None, None, None)],
            "served entity mapping is incomplete",
        ),
        (["missing", "entity-1"], [ENTITY, (*ENTITY[:3], "5")], "served entity mapping is missing"),
        (["entity-1"], [ENTITY, (*ENTITY[:3], "5")], "conflicting served entity mapping"),
    ],
)
def test_entity_errors_keep_priority_across_different_rows(spark, ids, entities, message):
    """A later mapping defect must not hide an earlier identity failure elsewhere."""
    payloads = spark.createDataFrame([(value,) for value in ids], "served_entity_id string")
    dimension = spark.createDataFrame(entities, ENTITY_SCHEMA)
    with pytest.raises(ValueError) as raised:
        serving_payloads._identity(payloads, dimension, "endpoint", "cat.sch.model", "4")
    assert str(raised.value) == message


@pytest.mark.parametrize(
    "overrides,message",
    [
        (
            {
                "databricks_request_id": None,
                "sampling_fraction": 0.5,
                "request_time": None,
                "status_code": None,
            },
            "request or served entity identity is missing",
        ),
        (
            {"sampling_fraction": 0.5, "request_time": None, "status_code": None},
            "sampling prevents complete serving observation",
        ),
        ({"request_time": None, "status_code": None}, "serving request time is missing"),
        ({"status_code": None}, "serving request metadata is incomplete"),
    ],
)
def test_capture_errors_keep_priority(spark, overrides, message):
    """Batching metadata predicates must keep the original first failure."""
    row = _row("bad", [{"x": 1}], [{"out_prediction": 2}], **overrides)
    payloads = spark.createDataFrame([row], PAYLOAD_SCHEMA)
    with pytest.raises(ValueError) as raised:
        serving_payloads._check_capture(payloads)
    assert str(raised.value) == message


@pytest.mark.parametrize(
    "inputs,outputs,message",
    [
        (None, None, "request must contain nonempty dataframe_records"),
        ([None], None, "response predictions must be record-oriented"),
        ([None, {"x": []}], [{"out_prediction": []}], "request and prediction row count mismatch"),
        ([None], [{"out_prediction": []}], "request and predictions must contain records"),
        (
            [{"x": []}],
            [{"out_prediction": []}],
            "request dataframe_records must contain finite scalar records",
        ),
        (
            [{"x": 1}],
            [{"out_prediction": []}],
            "response predictions must contain finite scalar records",
        ),
    ],
)
def test_envelope_errors_keep_priority(spark, inputs, outputs, message):
    """Scalar UDF checks must not mask earlier envelope or row-count errors."""
    frame = spark.createDataFrame([_row("bad", inputs, outputs)], PAYLOAD_SCHEMA)
    with pytest.raises(ValueError) as raised:
        serving_payloads._parse_envelopes(frame)
    assert str(raised.value) == message


@pytest.mark.parametrize("first", ["bad", "1"])
@pytest.mark.parametrize(
    "later,message",
    [
        (("wrong", "double"), "output column 'wrong' lacks prefix 'out_'"),
        (("out_", "double"), "invalid output column 'out_'"),
        (("out_y", "array<int>"), "unsupported scalar Spark type 'array<int>'"),
    ],
)
def test_bad_data_precedes_later_column_configuration(spark, first, later, message):
    """A schema error in a later column must not supersede earlier invalid payload data."""
    frame = spark.createDataFrame(
        [("r", 0, {"out_x": first, "out_y": "2"})],
        "databricks_request_id string, request_row_index int, _record map<string,string>",
    )
    with pytest.raises(ValueError) as raised:
        serving_payloads._typed_records(frame, (("out_x", "double"), later), "out_")
    assert str(raised.value) == (
        "invalid or missing payload column 'out_x'" if first == "bad" else message
    )


def test_typed_validation_preserves_column_order_and_nulls(spark):
    """Different failing rows retain schema priority while explicit nulls stay valid."""
    schema = "databricks_request_id string, request_row_index int, _record map<string,string>"
    frame = spark.createDataFrame(
        [("a", 0, {"x": "bad", "flag": "true"}), ("b", 0, {"x": "1", "flag": "yes"})], schema
    )
    with pytest.raises(ValueError) as raised:
        serving_payloads._typed_records(frame, (("x", "double"), ("flag", "boolean")))
    assert str(raised.value) == "invalid or missing payload column 'x'"
    nullable = spark.createDataFrame([("a", 0, {"x": None, "flag": None})], schema)
    result = serving_payloads._typed_records(nullable, (("x", "double"), ("flag", "boolean")))
    assert result.first().asDict() == {
        "databricks_request_id": "a",
        "request_row_index": 0,
        "x": None,
        "flag": None,
    }


def test_empty_window_keeps_empty_populations_and_unmeasured_summary(spark):
    """Null aggregates from an empty window must not manufacture validation failures."""
    current, predictions, summary, observed = _parse(spark, [])
    assert current.count() == predictions.count() == summary["request_count"] == 0
    assert current.columns == ["databricks_request_id", "request_row_index", "x", "name"]
    assert summary["latency_p99_ms"] is None and observed is None
