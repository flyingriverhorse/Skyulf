"""Inference-table payloads retain the actual served version and row identity."""

import json
from datetime import UTC, datetime, timedelta
from pathlib import Path

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


@pytest.mark.parametrize(
    "durations,percentile,maximum,expected",
    [
        ([10] * 999 + [1000], 10.0, 1000.0, "PASS"),
        ([10] * 980 + [200] * 20, 200.0, 200.0, "FAIL"),
        ([0] * 1000, 0.0, 0.0, "PASS"),
        ([None] * 1000, None, None, "HOLD"),
    ],
    ids=["isolated-outlier", "slow-tail", "measured-zero", "unmeasured"],
)
def test_native_latency_percentile_drives_rollout_health(
    spark, durations, percentile, maximum, expected
):
    """Native Spark aggregates must distinguish an isolated outlier from a slow tail."""
    from skyulf.integrations.databricks.serving.rollout_evidence import (
        RolloutHealthPolicy,
        _health_verdict,
    )

    payloads = spark.createDataFrame(
        [(200, duration, NOW, []) for duration in durations],
        "status_code int, execution_duration_ms long, request_time timestamp, "
        "logging_error_codes array<string>",
    )
    summary, observed = serving_payloads._summary(payloads, spark.range(0))
    assert summary["request_count"] == summary["latency_sample_count"] + durations.count(None)
    assert summary["latency_p99_ms"] == percentile
    assert summary["latency_max_ms"] == maximum
    assert observed == NOW
    health = RolloutHealthPolicy(
        minimum_requests=100,
        maximum_error_rate=0.01,
        maximum_latency_ms=100.0,
        minimum_labeled_rows=2,
        minimum_label_coverage=0.5,
    )
    assert _health_verdict(summary, health)[0] == expected


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


def test_operational_summary_keeps_status_classes_and_latency_bucket_boundaries(spark):
    """Every captured error and every timed response must belong to exactly one bucket."""
    durations = [0, 5, 10, 25, 50, 100, 250, 500, 1000, None]
    statuses = [200, 201, 400, 404, 500, 503, 302, 150, 650, 200]
    rows = [
        (status, duration, NOW, []) for status, duration in zip(statuses, durations, strict=True)
    ]
    payloads = spark.createDataFrame(
        rows,
        "status_code int, execution_duration_ms long, request_time timestamp, "
        "logging_error_codes array<string>",
    )
    summary, observed = serving_payloads._summary(payloads, spark.range(7))
    assert summary["request_count"] == 10 and summary["success_count"] == 3
    assert summary["logging_error_count"] == 0
    assert summary["status_4xx_count"] == summary["status_5xx_count"] == 2
    assert summary["status_other_count"] == 3
    classes = []
    for key in ("status_4xx_count", "status_5xx_count", "status_other_count"):
        count = summary[key]
        assert isinstance(count, int)
        classes.append(count)
    assert sum(classes) == summary["error_count"]
    assert summary["latency_sample_count"] == 9
    assert summary["latency_bucket_edges_ms"] == [5, 10, 25, 50, 100, 250, 500, 1000]
    assert summary["latency_bucket_counts"] == [1] * 9
    assert summary["latency_p50_ms"] == 50
    assert summary["latency_p95_ms"] == summary["latency_p99_ms"] == 1000
    assert summary["latency_mean_ms"] == pytest.approx(1940 / 9)
    assert summary["latency_max_ms"] == 1000
    assert summary["prediction_row_count"] == 7 and observed == NOW


@pytest.mark.parametrize("durations", [[], [None, None], [12]])
def test_operational_summary_empty_null_and_single_latency(spark, durations):
    """Unmeasured percentiles remain null and a single timed response has one occupied bucket."""
    rows = [(200, duration, NOW, None) for duration in durations]
    payloads = spark.createDataFrame(
        rows,
        "status_code int, execution_duration_ms long, request_time timestamp, "
        "logging_error_codes array<string>",
    )
    summary, observed = serving_payloads._summary(payloads, spark.range(0))
    expected = 12.0 if durations == [12] else None
    assert (
        summary["latency_p50_ms"]
        == summary["latency_p95_ms"]
        == summary["latency_p99_ms"]
        == expected
    )
    assert summary["latency_sample_count"] == (1 if expected else 0)
    buckets = summary["latency_bucket_counts"]
    assert isinstance(buckets, list)
    assert sum(buckets) == summary["latency_sample_count"]
    assert buckets[2] == (1 if expected else 0)
    assert (
        summary["error_count"]
        == summary["status_4xx_count"]
        == summary["status_5xx_count"]
        == summary["status_other_count"]
        == 0
    )
    assert observed == (NOW if durations else None)


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
    ],
)
def test_partial_or_unsupported_capture_fails_closed(spark, bad_row, match):
    """Incomplete logs must be visible rather than treated as healthy traffic."""
    with pytest.raises(ValueError, match=match):
        _parse(spark, [bad_row])


def test_logging_errors_exclude_payloads_but_keep_unique_request_counts(spark):
    """One truncated log must neither erase good predictions nor hide HTTP failures."""
    good = _row("good", [{"x": 1, "name": "a"}], [{"out_prediction": 2}])
    null_codes = _row(
        "null-codes", [{"x": 3, "name": "b"}], [{"out_prediction": 4}], logging_error_codes=None
    )
    truncated = _row(
        "truncated", None, None, logging_error_codes=["MAX_RESPONSE_SIZE_EXCEEDED", "OTHER"]
    )
    failed = _row("failed", None, None, status=503, logging_error_codes=["LOGGING_FAILURE"])
    other = _row("other", None, None, entity="entity-2", logging_error_codes=["OTHER"])
    current, predictions, summary, _ = _parse(
        spark,
        [good, null_codes, truncated, truncated, failed, other],
        [
            ("entity-1", "endpoint", "cat.sch.model", "4"),
            ("entity-2", "endpoint", "cat.sch.model", "5"),
        ],
    )
    assert sorted(r.databricks_request_id for r in current.collect()) == ["good", "null-codes"]
    assert sorted(r.prediction for r in predictions.collect()) == [2.0, 4.0]
    assert summary["request_count"] == 4
    assert summary["success_count"] == 3
    assert summary["error_count"] == summary["status_5xx_count"] == 1
    assert summary["logging_error_count"] == 2
    assert summary["prediction_row_count"] == 2


@pytest.mark.parametrize("valid_payload", [False, True])
def test_all_logging_errors_keep_empty_typed_populations_and_operational_counts(
    spark, valid_payload
):
    """An entirely unusable payload window has no invented feature or prediction rows."""
    row = _row(
        "truncated",
        [{"x": 1, "name": "a"}] if valid_payload else None,
        [{"out_prediction": 2}] if valid_payload else None,
        logging_error_codes=["MAX_REQUEST_SIZE_EXCEEDED"],
    )
    current, predictions, summary, observed = _parse(spark, [row])
    assert current.count() == predictions.count() == 0
    assert current.columns == ["databricks_request_id", "request_row_index", "x", "name"]
    assert "prediction" in predictions.columns
    assert summary["request_count"] == summary["success_count"] == 1
    assert summary["logging_error_count"] == 1
    assert summary["error_count"] == summary["prediction_row_count"] == 0
    assert observed == NOW


@pytest.mark.parametrize(
    "overrides,match",
    [
        ({"sampling_fraction": 0.5}, "sampling"),
        ({"databricks_request_id": None}, "identity"),
        ({"status_code": None}, "metadata"),
        ({"request_time": None}, "request time"),
    ],
)
def test_logging_errors_do_not_bypass_capture_identity_guards(spark, overrides, match):
    """Payload exclusion cannot hide missing identity, time, status or sampled traffic."""
    row = _row("bad", None, None, logging_error_codes=["LOGGING_FAILURE"], **overrides)
    with pytest.raises(ValueError, match=match):
        _parse(spark, [row])


@pytest.mark.parametrize(
    "all_flagged,reference_shift,expected_status,expected_decision",
    [
        (False, 0, "degraded", "incomplete_drift"),
        (True, 0, "no_data", "no_drift"),
        (False, -100, "drift", "ready"),
    ],
)
def test_partial_capture_metrics_preserve_drift_and_performance_eligibility(
    spark, monkeypatch, all_flagged, reference_shift, expected_status, expected_decision
):
    """Real subset metrics retain training signals; missing payloads never become zero error."""
    from types import SimpleNamespace

    from skyulf.integrations.databricks.jobs.lifecycle.retraining_task import observation_decision
    from skyulf.integrations.databricks.observability.monitoring import monitoring_performance
    from skyulf.integrations.databricks.observability.monitoring.monitoring_config import (
        MonitorConfig,
    )
    from skyulf.integrations.databricks.observability.monitoring.monitoring_output import (
        render_drift_output,
    )
    from skyulf.integrations.databricks.observability.monitoring.monitoring_store import result_row
    from skyulf.integrations.databricks.observability.monitoring.serving.serving_observation import (
        add_serving_report_details,
    )
    from skyulf.integrations.databricks.observability.monitoring.spark import (
        spark_monitoring_sources,
    )
    from skyulf.integrations.databricks.observability.monitoring.spark.spark_monitoring_metrics import (
        build_spark_monitoring_report,
    )

    values = range(1, 33)
    good = _row(
        "good",
        [{"x": x, "name": "a" if x % 2 else "b"} for x in values],
        [{"out_prediction": 2 * x + 5} for x in values],
    )
    bad = _row("bad", None, None, logging_error_codes=["MAX_RESPONSE_SIZE_EXCEEDED"])
    current, predictions, summary, observed = _parse(spark, [bad] if all_flagged else [good, bad])
    reference = spark.createDataFrame(
        [(float(x + reference_shift), "a" if x % 2 else "b") for x in values],
        "x double, name string",
    )
    labels = spark.createDataFrame(
        [("good", x - 1, float(2 * x), NOW) for x in values],
        "databricks_request_id string, request_row_index int, target double, available_at timestamp",
    )
    report = build_spark_monitoring_report(
        reference,
        current,
        predictions,
        labels,
        feature_columns=("x", "name"),
        record_key_columns=("databricks_request_id", "request_row_index"),
        target_column="target",
        result_available_at_column="available_at",
        as_of=NOW,
        task="regression",
    )
    add_serving_report_details(report, summary)
    config = MonitorConfig(
        environment="test",
        project="capture",
        model_name="cat.sch.model",
        model_version="4",
        source_table="cat.sch.payloads",
        prediction_table="cat.sch.payloads",
        serving_endpoint="endpoint",
        execution_engine="spark",
        reference_namespace="cat.monitoring",
        label_table="cat.sch.labels",
        result_available_at_column="available_at",
        performance_policy={
            "mode": "retrain",
            "metric": "mae",
            "direction": "lower",
            "baseline": {"kind": "training_holdout", "model_version": "4"},
            "tolerance": 1.0,
            "tolerance_mode": "absolute",
            "window_hours": 24,
            "label_delay_hours": 24,
            "minimum_labeled_rows": 10,
            "minimum_label_coverage": 0.8,
            "consecutive_windows": 1,
        },
    )
    row = result_row(
        config, "4", NOW, NOW - timedelta(hours=1), NOW, report, {}, observed_at=observed
    )
    assert row["status"] == expected_status
    assert (row["drifted_columns"] > 0) == (expected_status == "drift")
    assert (
        row["current_rows"]
        == row["scored_rows"]
        == row["labeled_rows"]
        == (0 if all_flagged else len(values))
    )
    saved = json.loads(row["report_json"])
    assert saved["serving"]["logging_error_count"] == 1
    mae = next(m for m in saved["metrics"] if m["metric_name"] == "mae")
    assert mae["value"] == (None if all_flagged else 5.0)
    assert "1 captured request" in render_drift_output(row)
    assert observation_decision(row, config, NOW) == expected_decision

    # Replace only external snapshot/history reads; Spark measurement and policy are real.
    evidence = {
        "source_table_id": "payloads",
        "prediction_table_id": "payloads",
        "serving": summary,
    }
    monkeypatch.setattr(
        monitoring_performance,
        "read_serving_observation",
        lambda *a, **kw: (current, predictions, evidence, observed),
    )
    monkeypatch.setattr(
        spark_monitoring_sources,
        "read_spark_labels",
        lambda *a: (labels, {"label_table_id": "labels"}),
    )
    monkeypatch.setattr(monitoring_performance, "load_performance_history", lambda *a: [])
    artifact = SimpleNamespace(
        manifest=SimpleNamespace(
            input_columns=("x", "name"),
            task="regression",
            classes=(),
            classification_probabilities=False,
            pipeline_sha256="a" * 64,
        ),
        pipeline=SimpleNamespace(config={}),
    )
    spec = SimpleNamespace(record_key_columns=("id",), target_column="target")
    baseline = {
        "model_version": "4",
        "metric": "mae",
        "contract_digest": monitoring_performance.performance_contract(artifact, spec, config),
        "value": 0.25,
        "labeled_rows": 32,
        "label_coverage": 1.0,
    }
    performance = monitoring_performance.observe_performance_safely(
        spark,
        "cat.monitoring",
        config,
        artifact,
        spec,
        {"model_version": "4", "performance_baseline": baseline},
        NOW + timedelta(days=2),
    )
    assert performance["measurement"]["evidence"]["serving"]["logging_error_count"] == 1
    assert performance["status"] == ("unavailable" if all_flagged else "degraded")
    assert performance["action"] == ("none" if all_flagged else "request_eligible")


def test_conflicting_request_copies_and_entity_mapping_fail(spark):
    """Conflicting retry content or dimension identity cannot be resolved arbitrarily."""
    first = _row("same", [{"x": 1, "name": "a"}], [{"out_prediction": 2}])
    changed = _row("same", [{"x": 1, "name": "a"}], [{"out_prediction": 3}])
    with pytest.raises(ValueError, match="conflicting"):
        _parse(spark, [first, changed])
    flagged = _row("same", None, None, logging_error_codes=["MAX_RESPONSE_SIZE_EXCEEDED"])
    with pytest.raises(ValueError, match="conflicting"):
        _parse(spark, [first, flagged])
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


def test_latency_distribution_query_keeps_only_valid_scoped_samples(spark):
    """Old or malformed arrays cannot become fake heatmap cells or cross model scope."""
    dashboard = json.loads(
        (
            Path(__file__).resolve().parents[2]
            / "templates/databricks/template/{{.project_name}}/src/monitoring/monitoring.lvdash.json"
        ).read_text(encoding="utf-8")
    )
    sql = "".join(
        next(d for d in dashboard["datasets"] if d["name"] == "serving_latency_distribution")[
            "queryLines"
        ]
    )
    valid = {
        "request_count": 3,
        "error_count": 1,
        "latency_sample_count": 2,
        "latency_bucket_edges_ms": [5, 10, 25, 50, 100, 250, 500, 1000],
        "latency_bucket_counts": [1, 0, 1, 0, 0, 0, 0, 0, 0],
    }
    cases = [
        ("valid", "healthy", valid),
        ("failed", "failed", valid),
        ("legacy", "healthy", {"request_count": 3}),
        ("null", "healthy", valid | {"latency_bucket_counts": [None] * 9}),
        ("negative", "healthy", valid | {"latency_bucket_counts": [-1, 0, 3, 0, 0, 0, 0, 0, 0]}),
        ("bad_sum", "healthy", valid | {"latency_sample_count": 3}),
        ("short", "healthy", valid | {"latency_bucket_counts": [2]}),
        (
            "bad_edges",
            "healthy",
            valid | {"latency_bucket_edges_ms": [6, 10, 25, 50, 100, 250, 500, 1000]},
        ),
        ("zero", "healthy", valid | {"latency_sample_count": 0, "latency_bucket_counts": [0] * 9}),
        ("too_many", "healthy", valid | {"request_count": 1}),
    ]
    reports = ",".join(
        f"('m','1','{name}',TIMESTAMP '2026-10-12 12:00:00',TIMESTAMP '2026-10-12 13:00:00',TIMESTAMP '2026-10-12 14:00:00','{status}','{json.dumps({'serving': summary})}','config')"
        for name, status, summary in cases
    )
    source = (
        "WITH model_inventory(monitor_id, model_name, enabled, updated_at, config_digest, environment, project, config_json) AS ("
        "SELECT * FROM VALUES ('m','model',true,1,'config','demo','serving','{\"model_version\":\"1\",\"serving_endpoint\":\"endpoint\"}')), "
        "monitoring_results(monitor_id, model_version, report_id, window_start, window_end, measured_at, status, report_json, config_digest) AS ("
        f"SELECT * FROM VALUES {reports}), " + sql.removeprefix("WITH ")
    )
    params = {"serving_model": "", "serving_version": "", "serving_monitor": ""}
    rows = spark.sql(source, args=params).collect()
    assert len(rows) == 9 and {r.report_id for r in rows} == {"valid"}
    assert sum(r.latency_fraction for r in rows) == 1.0
    assert [r.bucket_count for r in rows] == valid["latency_bucket_counts"]
    for field, value in (
        ("serving_model", "other"),
        ("serving_version", "2"),
        ("serving_monitor", "other context"),
    ):
        assert spark.sql(source, args=params | {field: value}).count() == 0
