"""Real gate decisions around injected native read boundaries for staged rollout evidence."""

from copy import deepcopy
from dataclasses import replace
from datetime import UTC, datetime, timedelta
from types import SimpleNamespace
from typing import Any

import pytest

pytest.importorskip("mlflow")

from tests.integration.platforms.test_serving_rollout_endpoints import (  # noqa: F401
    plans,
    ready_endpoint,
)

from skyulf.integrations.databricks.serving.rollout_endpoints import build_rollout_endpoint
from skyulf.integrations.databricks.serving.rollout_policy import RolloutState
from skyulf.integrations.mlflow.lifecycle.validation import ModelComparisonReport, comparison_digest

NOW = datetime(2026, 10, 11, tzinfo=UTC)
START = NOW - timedelta(hours=24)


def _state(percentage=10):
    """Pin a full observed stage without using aliases."""
    return RolloutState(
        "rollout",
        "rollout-test",
        "main.ml.model",
        "1",
        "main.ml.model",
        "2",
        percentage,
        START.isoformat(),
        START.isoformat(),
    )


def _report(**changes):
    """Use a concrete comparison with actual finite metrics and absolute gates."""
    values: dict[str, Any] = {
        "dataset_id": "data",
        "row_count": 20,
        "code_version": "1",
        "model_name": "main.ml.model",
        "candidate_version": "2",
        "candidate_digest": "a" * 64,
        "champion_version": "1",
        "champion_digest": "b" * 64,
        "metric": "heldout_rmse",
        "metric_direction": "minimize",
        "min_improvement": 0.0,
        "quality_threshold": 1.0,
        "candidate_metrics": {"heldout_rmse": 0.5},
        "champion_metrics": {"heldout_rmse": 0.8},
        "improvement": 0.3,
        "eligible": True,
        "reason": "improved",
    }
    return ModelComparisonReport(**(values | changes))


def _api():
    """Fail clearly until the trusted evidence producer exists."""
    from skyulf.integrations.databricks.serving import rollout_evidence

    return rollout_evidence


@pytest.fixture
def bootstrap(plans, monkeypatch):
    """Record native targeted invocations and replay a downloaded saved comparison."""
    api = _api()
    plan = build_rollout_endpoint(*plans)
    report = _report()
    calls = []
    endpoint = ready_endpoint(plan)

    def transport(**kwargs):
        """Return independently named native responses for each targeted route."""
        calls.append(kwargs)
        if kwargs["method"] == "GET":
            return deepcopy(endpoint)
        name = kwargs["path"].split("/")[-2]
        return {"served-model-name": name, "predictions": [{"prediction": 2.0}]}

    monkeypatch.setattr(
        api, "load_candidate_evidence", lambda *args, **kwargs: (report, None, None, None)
    )
    client = SimpleNamespace(
        api_client=SimpleNamespace(do=transport), config=SimpleNamespace(workspace_id=None)
    )
    return api, plan, report, client, calls, endpoint


def test_bootstrap_uses_saved_gates_and_both_targeted_responses(bootstrap):
    """Zero-percent candidate smoke must bypass traffic routing and attest both entity names."""
    api, plan, report, client, calls, _ = bootstrap
    result = api.observe_bootstrap_rollout(
        client,
        object(),
        plan,
        _state(0),
        comparison_sha256=comparison_digest(report),
        registry_uri="databricks-uc",
        records=[{"x": 1.0}],
        now=NOW,
    )
    assert result.evidence.verdict == "PASS"
    assert result.evidence.kind == "BOOTSTRAP"
    assert {call["path"].split("/")[-2] for call in calls if call["method"] == "POST"} == {
        "champion",
        "challenger",
    }
    assert result.details["comparison_sha256"] == comparison_digest(report)


@pytest.mark.parametrize(
    "change,expected",
    [
        ({"candidate_version": "3"}, "HOLD"),
        ({"candidate_metrics": {"heldout_rmse": 2.0}}, "FAIL"),
        ({"candidate_metrics": {}}, "HOLD"),
        ({"eligible": False}, "FAIL"),
    ],
)
def test_bootstrap_rechecks_actual_report_fields(bootstrap, monkeypatch, change, expected):
    """Saved eligible booleans cannot override unavailable or failing quality metrics."""
    api, plan, _, client, _, _ = bootstrap
    report = _report(**change)
    monkeypatch.setattr(
        api, "load_candidate_evidence", lambda *args, **kwargs: (report, None, None, None)
    )
    result = api.observe_bootstrap_rollout(
        client,
        object(),
        plan,
        _state(0),
        comparison_sha256=comparison_digest(report),
        registry_uri="databricks-uc",
        records=[{"x": 1.0}],
        now=NOW,
    )
    assert result.evidence.verdict == expected


@pytest.mark.parametrize(
    "response",
    [
        {"served-model-name": "champion", "predictions": [{"prediction": 2.0}]},
        {"predictions": [{"prediction": 2.0}]},
        {"served-model-name": "challenger", "predictions": [{"other": 2.0}]},
        {"served-model-name": "challenger", "predictions": [{"prediction": None}]},
    ],
)
def test_bootstrap_rejects_unattributed_or_invalid_smoke(bootstrap, response):
    """A returned response must attest the targeted entity and its exact finite output contract."""
    api, plan, report, client, _, endpoint = bootstrap

    def transport(**kwargs):
        """Expose a bad challenger while retaining a valid champion response."""
        if kwargs["method"] == "GET":
            return deepcopy(endpoint)
        if "/champion/" in kwargs["path"]:
            return {"served-model-name": "champion", "predictions": [{"prediction": 2.0}]}
        return response

    client.api_client.do = transport
    result = api.observe_bootstrap_rollout(
        client,
        object(),
        plan,
        _state(0),
        comparison_sha256=comparison_digest(report),
        registry_uri="databricks-uc",
        records=[{"x": 1.0}],
        now=NOW,
    )
    assert result.evidence.verdict != "PASS"


@pytest.mark.parametrize(
    "changes",
    [
        {"minimum_requests": True},
        {"minimum_requests": 0},
        {"maximum_error_rate": -1},
        {"maximum_error_rate": 2},
        {"maximum_latency_ms": float("nan")},
        {"minimum_labeled_rows": 1},
        {"minimum_label_coverage": False},
    ],
)
def test_health_policy_rejects_invalid_bounds(changes):
    """Booleans and invalid numbers must not weaken rollout admission thresholds."""
    api = _api()
    with pytest.raises(ValueError):
        api.RolloutHealthPolicy(
            **(
                {
                    "minimum_requests": 10,
                    "maximum_error_rate": 0.1,
                    "maximum_latency_ms": 100.0,
                    "minimum_labeled_rows": 2,
                    "minimum_label_coverage": 0.5,
                }
                | changes
            )
        )


@pytest.fixture
def live(plans, monkeypatch, tmp_path):
    """Use genuine fitted metadata and Core performance metrics behind native read boundaries."""
    import pandas as pd

    from skyulf.data.dataset import SplitDataset
    from skyulf.inference.fitted_pipeline import load_pipeline, save_pipeline
    from skyulf.integrations.databricks.observability.monitoring.monitoring_config import (
        MonitorConfig,
    )
    from skyulf.integrations.databricks.observability.monitoring.monitoring_metrics import (
        build_performance_report,
    )
    from skyulf.pipeline import SkyulfPipeline

    api = _api()
    frame = pd.DataFrame({"x": [1.0, 2.0, 3.0, 4.0], "target": [2.0, 4.0, 6.0, 8.0]})
    pipeline = SkyulfPipeline({"preprocessing": [], "modeling": {"type": "linear_regression"}})
    pipeline.fit(SplitDataset(train=frame, test=frame.head(0)), target_column="target")
    save_pipeline(pipeline, tmp_path / "model")
    artifact = load_pipeline(tmp_path / "model")
    plan = build_rollout_endpoint(*plans)
    config = MonitorConfig(
        environment="test",
        project="rollout",
        model_name="main.ml.model",
        model_version="2",
        source_table=plan.challenger.spec.inference_table,
        prediction_table=plan.challenger.spec.inference_table,
        label_table="main.obs.labels",
        result_available_at_column="available_at",
        execution_engine="spark",
        reference_namespace="main.obs",
        serving_endpoint="rollout-test",
    )
    predictions = pd.DataFrame(
        {"databricks_request_id": ["a", "b"], "request_row_index": [0, 0], "prediction": [2.0, 4.0]}
    )
    labels = pd.DataFrame(
        {
            "databricks_request_id": ["a", "b"],
            "request_row_index": [0, 0],
            "target": [2.0, 4.0],
            "available_at": [NOW, NOW],
        }
    )
    summary = {
        "request_count": 100,
        "success_count": 100,
        "error_count": 0,
        "latency_mean_ms": 10.0,
        "latency_max_ms": 20.0,
        "prediction_row_count": 2,
    }
    source = {
        "source_table": config.source_table,
        "prediction_table": config.source_table,
        "source_table_id": "physical-id",
        "prediction_table_id": "physical-id",
        "prediction_version": 7,
    }
    context: dict[str, Any] = {
        "summary": summary,
        "labels": labels,
        "observed": NOW - timedelta(minutes=1),
        "source": source,
        "calls": [],
    }
    monkeypatch.setattr(
        api, "read_serving_payload_snapshot", lambda *args: (object(), deepcopy(context["source"]))
    )
    monkeypatch.setattr(api, "verify_serving_payload_snapshot", lambda *args: None)

    def parse(*args, **kwargs):
        """Return service-shaped aggregates while recording exact requested window and schema."""
        context["calls"].append(kwargs)
        return None, predictions, deepcopy(context["summary"]), context["observed"]

    monkeypatch.setattr(api, "parse_serving_payloads", parse)
    monkeypatch.setattr(
        api,
        "read_spark_labels",
        lambda *args: (
            context["labels"],
            {"label_table": config.label_table, "label_table_id": "labels-id", "label_version": 9},
        ),
    )
    monkeypatch.setattr(api, "build_spark_performance_report", build_performance_report)
    health = api.RolloutHealthPolicy(
        minimum_requests=10,
        maximum_error_rate=0.1,
        maximum_latency_ms=100.0,
        minimum_labeled_rows=2,
        minimum_label_coverage=0.5,
    )
    spark = SimpleNamespace(table=lambda name: object())
    spec = SimpleNamespace(record_key_columns=("x",), target_column="target")
    return api, spark, plan, config, artifact, spec, health, context


def _observe(live, **changes):
    """Call the public producer with actual finite quality bounds."""
    api, spark, plan, config, artifact, spec, health, _ = live
    return api.observe_live_rollout(
        spark,
        plan,
        _state(),
        config,
        artifact,
        spec,
        health,
        metric="heldout_rmse",
        quality_threshold=1.0,
        now=NOW,
        **changes,
    )


def test_live_exact_stage_metrics_and_physical_evidence(live):
    """Quality derives from saved predictions and arrived labels, with original snapshot IDs retained."""
    result = _observe(live)
    assert result.evidence.verdict == "PASS"
    assert result.evidence.window_started_at == START.isoformat()
    assert result.evidence.window_ended_at == NOW.isoformat()
    assert result.details["source"]["source_table_id"] == "physical-id"
    assert result.details["labels"]["label_table_id"] == "labels-id"
    assert result.details["performance"]["values"]["rmse"] == 0.0
    assert len(result.details_digest) == 64
    assert live[-1]["calls"][0]["start"] == START
    assert live[-1]["calls"][0]["end"] == NOW


@pytest.mark.parametrize(
    "change,expected",
    [
        ({"request_count": 5, "success_count": 5}, "HOLD"),
        ({"error_count": 11, "success_count": 89}, "FAIL"),
        ({"latency_max_ms": 101.0}, "FAIL"),
        ({"latency_max_ms": None}, "HOLD"),
        ({"error_count": True}, "HOLD"),
    ],
)
def test_live_health_thresholds(live, change, expected):
    """Confirmed operational regression fails; absent or insufficient measurement holds."""
    live[-1]["summary"].update(change)
    assert _observe(live).evidence.verdict == expected


@pytest.mark.parametrize("age", [timedelta(hours=25), timedelta(minutes=-1)])
def test_stale_or_future_telemetry_holds(live, age):
    """Reissued stale telemetry and future requests cannot certify the current stage."""
    live[-1]["observed"] = NOW - age
    assert _observe(live).evidence.verdict == "HOLD"


def test_missing_labels_holds(live):
    """Delayed labels are unavailable evidence, never a confirmed model regression."""
    live[-1]["labels"] = None
    assert _observe(live).evidence.verdict == "HOLD"


def test_confirmed_quality_regression_fails(live):
    """Actual measured error beyond configured gates must stop further exposure."""
    live[-1]["labels"]["target"] = [20.0, 40.0]
    assert _observe(live).evidence.verdict == "FAIL"


def test_wrong_version_rejects_before_native_reads(live):
    """A valid monitor of another release cannot be relabeled as rollout evidence."""
    api, spark, plan, config, artifact, spec, health, context = live
    with pytest.raises(ValueError, match="identity"):
        api.observe_live_rollout(
            spark,
            plan,
            _state(),
            replace(config, model_version="3"),
            artifact,
            spec,
            health,
            metric="heldout_rmse",
            quality_threshold=1.0,
            now=NOW,
        )
    assert context["calls"] == []


def test_key_only_native_requests_use_admitted_source_schema(live):
    """Online request parsing must not demand enriched raw feature columns in payload logs."""
    api, spark, plan, config, artifact, spec, health, context = live
    plans = [
        replace(
            pinned, input_columns=("id",), input_schema=(("id", "int64"),), online_contract="a" * 64
        )
        for pinned in (plan.champion, plan.challenger)
    ]
    plan = build_rollout_endpoint(*plans)
    spec.record_key_columns = ("id",)
    result = api.observe_live_rollout(
        spark,
        plan,
        _state(),
        config,
        artifact,
        spec,
        health,
        metric="heldout_rmse",
        quality_threshold=1.0,
        now=NOW,
    )
    assert result.evidence.verdict == "PASS"
    assert context["calls"][0]["input_columns"] == (("id", "long"),)


def test_sampled_or_truncated_capture_holds(live, monkeypatch):
    """Existing strict telemetry parser errors cannot be converted into healthy metrics."""
    api = live[0]

    def reject(*args, **kwargs):
        """Represent the native parser's concrete sampled-capture rejection."""
        raise ValueError("sampling prevents complete serving observation")

    monkeypatch.setattr(api, "parse_serving_payloads", reject)
    result = _observe(live)
    assert result.evidence.verdict == "HOLD"
    assert "sampling" in result.evidence.reason


@pytest.mark.parametrize("offset", [None, timedelta(hours=1)])
def test_non_utc_clock_rejected_before_native_reads(bootstrap, offset):
    """Producer clocks cannot infer local time or use non-UTC offsets."""
    from datetime import timezone

    api, plan, report, client, calls, _ = bootstrap
    now = NOW.replace(tzinfo=None) if offset is None else NOW.astimezone(timezone(offset))
    with pytest.raises(ValueError, match="UTC"):
        api.observe_bootstrap_rollout(
            client,
            object(),
            plan,
            _state(0),
            comparison_sha256=comparison_digest(report),
            registry_uri="databricks-uc",
            records=[{"x": 1.0}],
            now=now,
        )
    assert calls == []


def test_changed_endpoint_after_smoke_holds(bootstrap):
    """Both successful responses cannot hide concurrent endpoint reconfiguration."""
    api, plan, report, client, _, endpoint = bootstrap
    reads = []

    def transport(**kwargs):
        """Change the concrete challenger only on final native readback."""
        if kwargs["method"] == "GET":
            result = deepcopy(endpoint)
            reads.append(result)
            if len(reads) == 2:
                result["config"]["served_entities"][1]["entity_version"] = "9"
            return result
        role = kwargs["path"].split("/")[-2]
        return {"served-model-name": role, "predictions": [{"prediction": 2.0}]}

    client.api_client.do = transport
    result = api.observe_bootstrap_rollout(
        client,
        object(),
        plan,
        _state(0),
        comparison_sha256=comparison_digest(report),
        registry_uri="databricks-uc",
        records=[{"x": 1.0}],
        now=NOW,
    )
    assert result.evidence.verdict == "HOLD"
    assert len(reads) == 2


def test_replaced_physical_source_holds(live, monkeypatch):
    """A table identity change during the stage observation invalidates otherwise healthy measurements."""

    def reject(*args, **kwargs):
        """Represent the shared pinned snapshot helper's actual replacement rejection."""
        raise ValueError("Serving inference table was replaced during observation.")

    monkeypatch.setattr(live[0], "verify_serving_payload_snapshot", reject)
    assert _observe(live).evidence.verdict == "HOLD"


def test_labels_arriving_after_cutoff_hold(live):
    """The shared metric engine must exclude labels unavailable at the observation cutoff."""
    live[-1]["labels"]["available_at"] = NOW + timedelta(hours=1)
    assert _observe(live).evidence.verdict == "HOLD"


def test_missing_source_keys_reject_before_native_reads(live):
    """Missing source key transport cannot be hidden by request-index monitoring keys."""
    api, spark, plan, config, artifact, spec, health, context = live
    spec.record_key_columns = ("missing_key",)
    with pytest.raises(ValueError, match="record keys"):
        api.observe_live_rollout(
            spark,
            plan,
            _state(),
            config,
            artifact,
            spec,
            health,
            metric="heldout_rmse",
            quality_threshold=1.0,
            now=NOW,
        )
    assert context["calls"] == []


def test_output_schema_mismatch_reject_before_native_reads(live):
    """A different fitted output contract cannot generate evidence for an admitted package."""
    api, spark, plan, config, artifact, spec, health, context = live
    selected = [
        replace(pinned, output_schema=(("different", "float64"),))
        for pinned in (plan.champion, plan.challenger)
    ]
    with pytest.raises(ValueError, match="output schema"):
        api.observe_live_rollout(
            spark,
            build_rollout_endpoint(*selected),
            _state(),
            config,
            artifact,
            spec,
            health,
            metric="heldout_rmse",
            quality_threshold=1.0,
            now=NOW,
        )
    assert context["calls"] == []


def test_boolean_saved_metric_cannot_certify_bootstrap(bootstrap, monkeypatch):
    """Malformed saved scalar metrics must not masquerade as measured zero error."""
    api, plan, _, client, _, _ = bootstrap
    report = _report(candidate_metrics={"heldout_rmse": False})
    monkeypatch.setattr(
        api, "load_candidate_evidence", lambda *args, **kwargs: (report, None, None, None)
    )
    result = api.observe_bootstrap_rollout(
        client,
        object(),
        plan,
        _state(0),
        comparison_sha256=comparison_digest(report),
        registry_uri="databricks-uc",
        records=[{"x": 1.0}],
        now=NOW,
    )
    assert result.evidence.verdict == "HOLD"
