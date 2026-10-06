"""Inference-table payloads retain the actual served version and row identity."""

import json
from datetime import UTC, datetime, timedelta

import pytest

from skyulf.integrations.databricks.observability.monitoring.serving import serving_payloads
from skyulf.integrations.databricks.observability.monitoring.serving.serving_payloads import (
    parse_serving_payloads,
)

NOW = datetime(2026, 10, 6, tzinfo=UTC)
PAYLOAD_SCHEMA = (
    "databricks_request_id string, request_time timestamp, status_code int, "
    "sampling_fraction double, execution_duration_ms long, request string, "
    "response string, served_entity_id string, logging_error_codes array<string>"
)


def _row(request_id, inputs, outputs, *, entity="entity-1", status=200, **overrides):
    """Build a documented custom-model inference-table row."""
    row = (
        request_id,
        NOW,
        status,
        1.0,
        12,
        json.dumps({"dataframe_records": inputs}) if inputs is not None else None,
        json.dumps({"predictions": outputs}) if outputs is not None else None,
        entity,
        [],
    )
    names = [
        "databricks_request_id",
        "request_time",
        "status_code",
        "sampling_fraction",
        "execution_duration_ms",
        "request",
        "response",
        "served_entity_id",
        "logging_error_codes",
    ]
    values = dict(zip(names, row, strict=True)) | overrides
    return tuple(values[name] for name in names)


def _parse(spark, rows, entity_rows=None, *, output_columns=(("out_prediction", "double"),)):
    """Use one endpoint and concrete artifact schema for each test."""
    payloads = spark.createDataFrame(rows, PAYLOAD_SCHEMA)
    entities = spark.createDataFrame(
        entity_rows or [("entity-1", "endpoint", "cat.sch.model", "4")],
        "served_entity_id string, endpoint_name string, entity_name string, entity_version string",
    )
    return parse_serving_payloads(
        payloads,
        entities,
        endpoint_name="endpoint",
        model_name="cat.sch.model",
        model_version="4",
        input_columns=(("x", "double"), ("name", "string")),
        output_columns=output_columns,
        output_prefix="out_",
        start=NOW - timedelta(hours=1),
        end=NOW + timedelta(hours=1),
    )


def test_multirecord_request_keeps_ordinal_nulls_and_deduplicates(spark):
    """Retries cannot multiply predictions or lose null features and row order."""
    row = _row(
        "request-1",
        [{"x": "1.25", "name": None}, {"x": 2, "name": "b"}],
        [{"out_prediction": 4.0}, {"out_prediction": 5.0}],
    )
    current, predictions, summary, observed_at = _parse(spark, [row, row])
    actual = current.orderBy("request_row_index").collect()
    scored = predictions.orderBy("request_row_index").collect()
    assert [(r.databricks_request_id, r.request_row_index, r.x, r.name) for r in actual] == [
        ("request-1", 0, 1.25, None),
        ("request-1", 1, 2.0, "b"),
    ]
    assert [r.prediction for r in scored] == [4.0, 5.0]
    assert summary["request_count"] == 1
    assert observed_at == NOW


def test_version_isolation_and_failed_http_counts(spark):
    """A failed request and another version cannot contaminate model metrics."""
    good = _row("ok", [{"x": 1, "name": "a"}], [{"out_prediction": 2}])
    failed = _row("bad", None, None, status=500)
    other = _row("other", [{"x": 9, "name": "z"}], [{"out_prediction": 9}], entity="entity-2")
    _, predictions, summary, _ = _parse(
        spark,
        [good, failed, other],
        [
            ("entity-1", "endpoint", "cat.sch.model", "4"),
            ("entity-2", "endpoint", "cat.sch.model", "5"),
        ],
    )
    assert [r.databricks_request_id for r in predictions.collect()] == ["ok"]
    assert summary["request_count"] == 2
    assert summary["error_count"] == 1


@pytest.mark.parametrize(
    "bad_row,match",
    [
        (_row("bad", [{"x": 1, "name": "a"}], [3]), "predictions"),
        (_row("bad", [{"x": 1, "name": "a"}], []), "row count"),
        (_row("bad", None, [{"out_prediction": 3}]), "request"),
        (
            _row("bad", [{"x": 1, "name": "a"}], [{"out_prediction": 3}], sampling_fraction=0.5),
            "sampling",
        ),
        (
            _row(
                "bad",
                [{"x": 1, "name": "a"}],
                [{"out_prediction": 3}],
                logging_error_codes=["MAX_RESPONSE_SIZE_EXCEEDED"],
            ),
            "logging",
        ),
    ],
)
def test_partial_or_unsupported_capture_fails_closed(spark, bad_row, match):
    """Incomplete logs must be visible rather than treated as healthy traffic."""
    with pytest.raises(ValueError, match=match):
        _parse(spark, [bad_row])


def test_conflicting_request_copies_and_entity_mapping_fail(spark):
    """Conflicting retry content or dimension identity cannot be resolved arbitrarily."""
    first = _row("same", [{"x": 1, "name": "a"}], [{"out_prediction": 2}])
    changed = _row("same", [{"x": 1, "name": "a"}], [{"out_prediction": 3}])
    with pytest.raises(ValueError, match="conflicting"):
        _parse(spark, [first, changed])
    with pytest.raises(ValueError, match="conflicting"):
        _parse(
            spark,
            [first],
            [
                ("entity-1", "endpoint", "cat.sch.model", "4"),
                ("entity-1", "endpoint", "cat.sch.model", "5"),
            ],
        )


def test_missing_entity_mapping_fails_even_for_failed_request(spark):
    """Unmapped failures cannot silently disappear from operational counts."""
    failed = _row("unknown", None, None, status=500, entity="missing")
    with pytest.raises(ValueError, match="mapping is missing"):
        _parse(spark, [failed])


def test_excluded_model_row_preserves_status_and_null_prediction(spark):
    """Model-set exclusions keep their saved status for downstream quality counts."""
    row = _row(
        "excluded",
        [{"x": None, "name": "a"}],
        [{"branch__prediction": None, "branch__scoring_status": "excluded"}],
    )
    payloads = spark.createDataFrame([row], PAYLOAD_SCHEMA)
    entities = spark.createDataFrame(
        [("entity-1", "endpoint", "cat.sch.model", "4")],
        "served_entity_id string, endpoint_name string, entity_name string, entity_version string",
    )
    current, predictions, _, _ = parse_serving_payloads(
        payloads,
        entities,
        endpoint_name="endpoint",
        model_name="cat.sch.model",
        model_version="4",
        input_columns=(("x", "double"), ("name", "string")),
        output_columns=(("branch__prediction", "double"), ("branch__scoring_status", "string")),
        output_prefix="branch__",
        start=NOW - timedelta(hours=1),
        end=NOW + timedelta(hours=1),
    )
    assert current.first().x is None
    assert predictions.select("prediction", "scoring_status").first() == (None, "excluded")


def test_transport_keeps_large_integer_boolean_and_reordered_fields(spark):
    """JSON transport cannot round large IDs or coerce boolean strings loosely."""
    row = _row(
        "typed",
        [{"flag": "true", "large": "9007199254740993", "name": None, "x": "1.25"}],
        [{"out_prediction": 3}],
    )
    payloads = spark.createDataFrame([row], PAYLOAD_SCHEMA)
    entities = spark.createDataFrame(
        [("entity-1", "endpoint", "cat.sch.model", "4")],
        "served_entity_id string, endpoint_name string, entity_name string, entity_version string",
    )
    current, _, _, _ = parse_serving_payloads(
        payloads,
        entities,
        endpoint_name="endpoint",
        model_name="cat.sch.model",
        model_version="4",
        input_columns=(("large", "long"), ("flag", "boolean"), ("x", "double"), ("name", "string")),
        output_columns=(("out_prediction", "double"),),
        output_prefix="out_",
        start=NOW - timedelta(hours=1),
        end=NOW + timedelta(hours=1),
    )
    actual = current.first()
    assert (actual.large, actual.flag, actual.x, actual.name) == (
        9007199254740993,
        True,
        1.25,
        None,
    )


@pytest.mark.parametrize(
    "inputs",
    [
        [{"x": 1}],
        [{"x": 1, "name": "a", "flag": "yes"}],
    ],
)
def test_missing_field_and_invalid_boolean_fail(spark, inputs):
    """Missing names and loose boolean coercion must not change the input population."""
    row = _row("invalid", inputs, [{"out_prediction": 3}])
    if "flag" in inputs[0]:
        names = (("x", "double"), ("name", "string"), ("flag", "boolean"))
    else:
        names = (("x", "double"), ("name", "string"))
    payloads = spark.createDataFrame([row], PAYLOAD_SCHEMA)
    entities = spark.createDataFrame(
        [("entity-1", "endpoint", "cat.sch.model", "4")],
        "served_entity_id string, endpoint_name string, entity_name string, entity_version string",
    )
    with pytest.raises(ValueError, match="invalid or missing"):
        parse_serving_payloads(
            payloads,
            entities,
            endpoint_name="endpoint",
            model_name="cat.sch.model",
            model_version="4",
            input_columns=names,
            output_columns=(("out_prediction", "double"),),
            output_prefix="out_",
            start=NOW - timedelta(hours=1),
            end=NOW + timedelta(hours=1),
        )


@pytest.mark.parametrize("value", [{"nested": 1}, [1, 2]])
def test_nested_input_scalar_fails_before_string_coercion(spark, value):
    """Complex JSON cannot become an apparently valid string feature."""
    row = _row("nested", [{"x": 1, "name": value}], [{"out_prediction": 3}])
    with pytest.raises(ValueError, match="scalar"):
        _parse(spark, [row])


@pytest.mark.parametrize("value", [{"nested": 1}, [1, 2]])
def test_nested_output_scalar_fails_before_string_coercion(spark, value):
    """Complex predictions cannot masquerade as saved string class labels."""
    row = _row("nested", [{"x": 1, "name": "a"}], [{"out_prediction": value}])
    with pytest.raises(ValueError, match="scalar"):
        _parse(spark, [row], output_columns=(("out_prediction", "string"),))


def test_missing_request_time_cannot_disappear_from_window(spark):
    """Missing timestamps make attribution unavailable instead of hiding captured traffic."""
    row = _row("missing-time", [{"x": 1, "name": "a"}], [{"out_prediction": 3}], request_time=None)
    with pytest.raises(ValueError, match="request time"):
        _parse(spark, [row])


def test_missing_time_on_another_version_does_not_invalidate_selected_model(spark):
    """Known unrelated versions remain isolated even when their capture is incomplete."""
    good = _row("good", [{"x": 1, "name": "a"}], [{"out_prediction": 3}])
    other = _row("other", None, None, entity="entity-2", request_time=None)
    _, predictions, summary, _ = _parse(
        spark,
        [good, other],
        [
            ("entity-1", "endpoint", "cat.sch.model", "4"),
            ("entity-2", "endpoint", "cat.sch.model", "5"),
        ],
    )
    assert predictions.count() == summary["request_count"] == 1


@pytest.mark.parametrize("field", [1, 2, 3])
def test_incomplete_relevant_entity_mapping_fails(spark, field):
    """A present entity ID cannot conceal missing endpoint, model or version identity."""
    entity = ["entity-1", "endpoint", "cat.sch.model", "4"]
    entity[field] = None
    row = _row("mapped", [{"x": 1, "name": "a"}], [{"out_prediction": 3}])
    with pytest.raises(ValueError, match="mapping is incomplete"):
        _parse(spark, [row], [tuple(entity)])


def test_unrelated_entity_and_old_payload_do_not_fail_current_window(spark):
    """Malformed unrelated history must not invalidate a fully identified current population."""
    row = _row("current", [{"x": 1, "name": "a"}], [{"out_prediction": 3}])
    old = _row("old", None, None, entity="unmapped", request_time=NOW - timedelta(days=2))
    _, predictions, summary, _ = _parse(
        spark,
        [row, old],
        [("entity-1", "endpoint", "cat.sch.model", "4"), ("unused", None, None, None)],
    )
    assert predictions.count() == summary["request_count"] == 1


@pytest.mark.parametrize("value", [{"nested": 1}, [1, 2], float("nan"), float("inf")])
def test_scalar_envelope_validator_rejects_complex_or_nonfinite_values(value):
    """The executor validator must reject values that Spark string coercion would hide."""
    raw = json.dumps({"dataframe_records": [{"x": value}]})
    assert not serving_payloads._scalar_envelope(raw, "dataframe_records")


@pytest.mark.parametrize("value", [None, True, 9007199254740993, 1.25, "{literal}", "[literal]"])
def test_scalar_envelope_validator_preserves_json_primitives(value):
    """Strict shape checking must retain nulls, exact large integers and ordinary strings."""
    raw = json.dumps({"dataframe_records": [{"x": value}]})
    assert serving_payloads._scalar_envelope(raw, "dataframe_records")
