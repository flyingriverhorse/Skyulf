"""Durable traffic changes survive process loss without repeating endpoint writes."""

import json
from contextlib import contextmanager
from copy import deepcopy
from dataclasses import asdict
from datetime import UTC, datetime, timedelta
from importlib import import_module
from importlib.util import find_spec
from pathlib import Path
from threading import Lock
from types import SimpleNamespace

import pytest

mlflow = pytest.importorskip("mlflow")

from skyulf.integrations.databricks.data.admission import (  # noqa: E402
    BatchConflictError,
    SingleWriterAdmission,
)
from skyulf.integrations.databricks.serving.contracts import (  # noqa: E402
    PinnedEndpointPlan,
    PinnedEndpointSpec,
)
from skyulf.integrations.databricks.serving.rollout_endpoints import (  # noqa: E402
    build_rollout_endpoint,
)
from skyulf.integrations.databricks.serving.rollout_policy import (  # noqa: E402
    DailyRolloutPolicy,
    RolloutEvidence,
)


class Tracking:
    """Emulate only the explicit MLflow boundary, including uncertain pointer writes."""

    def __init__(self):
        """Keep durable fake content across independent controller/store objects."""
        self.tags = {}
        self.artifacts = {}
        self.fail_pointer = False
        self.fail_artifact = False

    def get_run(self, run_id):
        """Refresh the backend tags on each read."""
        return SimpleNamespace(data=SimpleNamespace(tags=deepcopy(self.tags)))

    def list_artifacts(self, run_id, path):
        """Expose interrupted initialization instead of silently adopting it."""
        return [key for key in self.artifacts if key.startswith(path)]

    def log_dict(self, run_id, value, path):
        """Write actual JSON so serialization failures cannot hide behind a mock."""
        if self.fail_artifact:
            raise OSError("artifact unavailable")
        assert path not in self.artifacts
        self.artifacts[path] = json.dumps(value)

    def download_artifacts(self, run_id, path, dst_path):
        """Materialize artifact bytes through the same filesystem API as MLflow."""
        target = Path(dst_path) / "receipt.json"
        target.write_text(self.artifacts[path], encoding="utf-8")
        return str(target)

    def set_tag(self, run_id, key, value):
        """Model an uncommitted current pointer."""
        if self.fail_pointer:
            raise OSError("pointer unavailable")
        self.tags[key] = value


class Endpoint:
    """Model settled, pending, lost-response and unapplied native config updates."""

    def __init__(self, plan):
        """Pin one isolated endpoint and expose WorkspaceClient-shaped transport."""
        self.config = SimpleNamespace(host="https://workspace.example")
        self.api_client = SimpleNamespace(do=self.request)
        self.value = deepcopy(plan.config)
        self.value["id"] = "endpoint-id"
        self.value["state"] = {"ready": "READY", "config_update": "NOT_UPDATING"}
        self.value["config"]["config_version"] = 3
        if plan.champion.spec.logging_mode == "telemetry":
            self.value["telemetry_config"]["table_names"] = {
                "logs_table": plan.champion.spec.telemetry_logs_table
            }
            self.value["telemetry_config"]["inference_table_config"]["name"] = (
                plan.champion.spec.inference_table
            )
        self.puts = []
        self.mode = "ready"
        self.after_put = None

    def request(self, *, method, path, body=None):
        """Apply native mutation before a possible timeout or readiness delay."""
        if method == "GET":
            return deepcopy(self.value)
        assert method == "PUT" and path.endswith("/config")
        assert body is not None
        self.puts.append(deepcopy(body))
        if self.mode != "old":
            revision = self.value["config"]["config_version"] + 1
            self.value["config"] = deepcopy(body) | {"config_version": revision}
        if self.mode == "pending":
            self.value["state"]["config_update"] = "IN_PROGRESS"
        if self.after_put:
            self.after_put()
        if self.mode in {"timeout", "old"}:
            raise TimeoutError("response lost")
        return deepcopy(self.value)


@pytest.fixture
def api():
    """An absent controller is a direct failing behavior assertion."""
    name = "skyulf.integrations.databricks.serving.rollout"
    assert find_spec(name) is not None, "Durable rollout controller is not implemented"
    return import_module(name)


@pytest.fixture(params=["telemetry", "ai_gateway"])
def plan(request):
    """Exercise both admitted logging modes at the native boundary."""
    plans = []
    for version in ("1", "2"):
        spec = PinnedEndpointSpec(
            "daily", "main.ml.model", version, "main", "obs", "daily", request.param
        )
        config = {
            "name": "daily",
            "config": {
                "served_entities": [
                    {
                        "entity_name": spec.model_name,
                        "entity_version": version,
                        "workload_type": "CPU",
                        "workload_size": "Small",
                        "scale_to_zero_enabled": True,
                    }
                ]
            },
        }
        if request.param == "telemetry":
            config["telemetry_config"] = {
                "table_names": {
                    "logs_table": spec.telemetry_logs_table,
                    "traces_table": spec.telemetry_traces_table,
                    "metrics_table": spec.telemetry_metrics_table,
                },
                "inference_table_config": {"sampling_fraction": 1.0},
                "enabled_telemetry_features": ["TELEMETRY_FEATURE_INFERENCE_TABLE"],
            }
        else:
            config["ai_gateway"] = {
                "inference_table_config": {
                    "catalog_name": "main",
                    "schema_name": "obs",
                    "table_name_prefix": "daily",
                    "enabled": True,
                },
                "usage_tracking_config": {"enabled": True},
            }
        plans.append(
            PinnedEndpointPlan(
                spec, config, ("x",), (("x", "float64"),), (("prediction", "float64"),)
            )
        )
    return build_rollout_endpoint(*plans)


@pytest.fixture
def setup(api, plan):
    """Initialize an independently reloadable zero-exposure rollout."""
    tracking = Tracking()
    store = api.MLflowRolloutStore(tracking, "explicit-run")
    client = Endpoint(plan)
    clock = [datetime(2026, 10, 10, tzinfo=UTC)]
    admission = SingleWriterAdmission()
    result = api.initialize_rollout(
        client,
        plan,
        store=store,
        admission=admission,
        policy=DailyRolloutPolicy(),
        rollout_id="release-1",
        clock=lambda: clock[0],
    )
    return SimpleNamespace(
        api=api,
        plan=plan,
        tracking=tracking,
        store=store,
        client=client,
        clock=clock,
        admission=admission,
        result=result,
    )


def evidence(state, now, verdict="PASS"):
    """Bind trusted-producer input to the entire observed current stage."""
    identity = asdict(state)
    identity.pop("rollout_started_at")
    identity.pop("phase")
    return RolloutEvidence(
        **identity,
        window_started_at=state.stage_started_at,
        window_ended_at=now.isoformat(),
        observed_at=now.isoformat(),
        verdict=verdict,
        kind="BOOTSTRAP" if state.challenger_percentage == 0 else "LIVE",
        reason="verified quality",
    )


def advance(s, verdict="PASS"):
    """Execute one observed daily interval through the public API."""
    return s.api.advance_rollout(
        s.client,
        store=s.store,
        admission=s.admission,
        evidence=evidence(s.result.state, s.clock[0], verdict),
        clock=lambda: s.clock[0],
    )


def pending_native_response(s, *, extra=None):
    """Keep the old active config while native deployment exposes the requested pending one."""
    previous = deepcopy(s.client.value["config"])

    def after_put():
        """Expose actual native millisecond metadata without committing readiness early."""
        pending = deepcopy(s.client.value["config"])
        pending["start_time"] = 1791662862000
        pending.update(extra or {})
        s.client.value["pending_config"] = pending
        s.client.value["config"] = previous
        s.client.value["state"]["config_update"] = "IN_PROGRESS"

    s.client.after_put = after_put


def test_native_pending_start_time_stays_prepared_then_commits_once(setup):
    """An ordinary asynchronous native response must not become an unknown write outcome."""
    s = setup
    pending_native_response(s)
    s.clock[0] += timedelta(days=1)
    pending = advance(s)
    assert pending.status == "PREPARED"
    assert pending.state.challenger_percentage == 0
    assert len(s.client.puts) == 1
    s.client.value["config"] = s.client.value.pop("pending_config")
    s.client.value["config"].pop("start_time")
    s.client.value["state"]["config_update"] = "NOT_UPDATING"
    settled = s.api.reconcile_rollout(
        s.client, store=s.store, admission=s.admission, clock=lambda: s.clock[0]
    )
    assert settled.status == "COMMITTED"
    assert settled.state.challenger_percentage == 10
    assert len(s.client.puts) == 1


def test_native_pending_metadata_does_not_allow_unknown_configuration(setup):
    """An unadmitted pending option still blocks recovery after the single request."""
    s = setup
    pending_native_response(s, extra={"unknown_runtime_option": True})
    s.clock[0] += timedelta(days=1)
    with pytest.raises(s.api.RolloutOutcomeUnknownError):
        advance(s)
    with pytest.raises(ValueError, match="configuration"):
        s.api.reconcile_rollout(s.client, store=s.store, admission=s.admission)
    assert s.store.load()["status"] == "PREPARED"
    assert len(s.client.puts) == 1


def test_initialize_persists_complete_identity_and_rejects_overwrite(setup):
    """An existing rollout cannot silently adopt different models or reset its clock."""
    s = setup
    record = s.store.load()
    assert record["endpoint_id"] == "endpoint-id" and record["config_version"] == 3
    assert record["state"] == asdict(s.result.state)
    assert record["policy"] == asdict(DailyRolloutPolicy())
    assert record["plan"]["config"] == s.plan.config
    with pytest.raises(ValueError, match="already|existing"):
        s.api.initialize_rollout(
            s.client,
            s.plan,
            store=s.store,
            admission=s.admission,
            policy=DailyRolloutPolicy(),
            rollout_id="release-2",
            clock=lambda: s.clock[0],
        )
    assert s.client.puts == []


def test_daily_advance_uses_verified_readiness_clock_and_preserves_both_models(setup):
    """A deployment delay never counts toward the next observation stage."""
    s = setup
    s.clock[0] += timedelta(days=1)
    s.client.after_put = lambda: s.clock.__setitem__(0, s.clock[0] + timedelta(minutes=9))
    result = advance(s)
    assert result.state.challenger_percentage == 10
    assert result.state.stage_started_at == s.clock[0].isoformat()
    assert len(s.client.puts) == 1 and len(s.client.puts[0]["served_entities"]) == 2
    assert s.store.load()["config_version"] == 4


def test_hold_preserves_clock_and_does_not_write(setup):
    """Scheduler retries cannot reset observation time or manufacture progress."""
    s = setup
    result = advance(s)
    assert result.state == s.result.state and result.status == "HOLD"
    assert result.receipt_id == s.result.receipt_id and s.client.puts == []


@pytest.mark.parametrize("mode", ["timeout", "pending", "old"])
def test_unknown_outcome_reconciles_without_reissuing_mutation(setup, mode):
    """Only verified target readback can settle a durable prepared intent."""
    s = setup
    s.clock[0] += timedelta(days=1)
    s.client.mode = mode
    if mode == "pending":
        result = advance(s)
        assert result.status == "PREPARED"
    else:
        with pytest.raises(s.api.RolloutOutcomeUnknownError):
            advance(s)
    assert s.store.load()["status"] == "PREPARED"
    s.clock[0] += timedelta(hours=2)
    if mode == "pending":
        pending = s.api.reconcile_rollout(
            s.client, store=s.store, admission=s.admission, clock=lambda: s.clock[0]
        )
        assert pending.status == "PREPARED"
        s.client.value["state"]["config_update"] = "NOT_UPDATING"
    result = s.api.reconcile_rollout(
        s.client, store=s.store, admission=s.admission, clock=lambda: s.clock[0]
    )
    assert result.status == ("PREPARED" if mode == "old" else "COMMITTED")
    assert result.state.challenger_percentage == (0 if mode == "old" else 10)
    replay = s.api.advance_rollout(
        s.client, store=s.store, admission=s.admission, evidence=None, clock=lambda: s.clock[0]
    )
    assert replay.state == result.state and len(s.client.puts) == 1


@pytest.mark.parametrize("failure", ["fail_pointer", "fail_artifact"])
def test_store_failure_before_put_never_mutates_endpoint(setup, failure):
    """Traffic changes require a durable verified intent first."""
    s = setup
    s.clock[0] += timedelta(days=1)
    setattr(s.tracking, failure, True)
    with pytest.raises((OSError, RuntimeError)):
        advance(s)
    assert s.client.puts == []


def test_final_receipt_failure_is_unknown_and_recoverable(setup):
    """A native success cannot be reported before durable state commit."""
    s = setup
    s.clock[0] += timedelta(days=1)
    s.client.after_put = lambda: setattr(s.tracking, "fail_pointer", True)
    with pytest.raises(s.api.RolloutOutcomeUnknownError):
        advance(s)
    s.tracking.fail_pointer = False
    result = s.api.reconcile_rollout(
        s.client, store=s.store, admission=s.admission, clock=lambda: s.clock[0]
    )
    assert result.state.challenger_percentage == 10 and len(s.client.puts) == 1


@pytest.mark.parametrize("drift", ["id", "revision", "routes", "model", "settings"])
def test_foreign_endpoint_changes_stop_automation(setup, drift):
    """Outside writes are conflicts, never a new baseline to overwrite."""
    s = setup
    s.clock[0] += timedelta(days=1)
    endpoint = s.client.value
    if drift == "id":
        endpoint["id"] = "replacement"
    elif drift == "revision":
        endpoint["config"]["config_version"] += 2
    elif drift == "routes":
        endpoint["config"]["traffic_config"]["routes"][0]["traffic_percentage"] = 90
    elif drift == "model":
        endpoint["config"]["served_entities"][0]["entity_version"] = "7"
    else:
        endpoint["config"]["served_entities"][0]["environment_vars"] = {"X": "foreign"}
    with pytest.raises(ValueError):
        advance(s)
    assert s.client.puts == []


def test_corrupt_receipt_and_pointer_fail_closed(setup):
    """Restarted workers reject unverified durable JSON before endpoint writes."""
    s = setup
    path = next(iter(s.tracking.artifacts))
    s.tracking.artifacts[path] = s.tracking.artifacts[path].replace('"release-1"', '"foreign"')
    with pytest.raises(ValueError, match="digest|receipt"):
        s.api.reconcile_rollout(
            s.client, store=s.store, admission=s.admission, clock=lambda: s.clock[0]
        )
    assert s.client.puts == []


def test_terminal_complete_and_rollback_are_replay_safe(setup):
    """Completion waits through 100 percent and rollback never restarts itself."""
    s = setup
    for share in range(10, 101, 10):
        s.clock[0] += timedelta(days=1)
        s.result = advance(s)
        assert s.result.state.challenger_percentage == share
        assert not s.result.promotion_pending
    s.clock[0] += timedelta(days=1)
    s.result = advance(s)
    assert s.result.state.phase == "COMPLETE" and s.result.promotion_pending
    replay = advance(s)
    assert replay.receipt_id == s.result.receipt_id and len(s.client.puts) == 10


def test_failure_rolls_back_once_and_never_restarts(setup):
    """Confirmed live failure returns to champion before recording terminal state."""
    s = setup
    s.clock[0] += timedelta(days=1)
    s.result = advance(s)
    s.clock[0] += timedelta(minutes=1)
    s.result = advance(s, "FAIL")
    assert s.result.state.phase == "ROLLED_BACK" and s.result.state.challenger_percentage == 0
    s.clock[0] += timedelta(days=1)
    replay = advance(s)
    assert replay.state == s.result.state and len(s.client.puts) == 2


def test_real_mlflow_sqlite_reload(api, plan, tmp_path):
    """The actual MLflow client can persist and reload receipts across objects."""
    client = mlflow.MlflowClient(tracking_uri=f"sqlite:///{(tmp_path / 'tracking.db').as_posix()}")
    experiment = client.create_experiment(
        "rollout", artifact_location=(tmp_path / "artifacts").as_uri()
    )
    run = client.create_run(experiment)
    store = api.MLflowRolloutStore(client, run.info.run_id)
    endpoint = Endpoint(plan)
    now = datetime(2026, 10, 10, tzinfo=UTC)
    result = api.initialize_rollout(
        endpoint,
        plan,
        store=store,
        admission=SingleWriterAdmission(),
        policy=DailyRolloutPolicy(),
        rollout_id="real",
        clock=lambda: now,
    )
    fresh = api.MLflowRolloutStore(
        mlflow.MlflowClient(tracking_uri=client.tracking_uri), run.info.run_id
    )
    replay = api.reconcile_rollout(
        endpoint, store=fresh, admission=SingleWriterAdmission(), clock=lambda: now
    )
    assert replay.state == result.state and fresh.load()["run_id"] == run.info.run_id


def test_admission_rejects_two_contenders_and_host_local_provider(setup):
    """All rollout runs compete for the workspace endpoint rather than their run ID."""
    s = setup

    class Admission:
        local_only = False

        def __init__(self):
            """Share one actual nonblocking authority across controller contenders."""
            self.lock = Lock()
            self.resources = []

        @contextmanager
        def hold(self, resource):
            """Reject a second controller while the first retains ownership."""
            self.resources.append(resource)
            if not self.lock.acquire(blocking=False):
                raise BatchConflictError(resource)
            try:
                yield
            finally:
                self.lock.release()

    admission = Admission()
    s.admission = admission

    def contend():
        """Attempt a real nested controller invocation while mutation owns admission."""
        with pytest.raises(BatchConflictError):
            s.api.reconcile_rollout(
                s.client, store=s.store, admission=admission, clock=lambda: s.clock[0]
            )

    s.clock[0] += timedelta(days=1)
    s.client.after_put = contend
    result = advance(s)
    assert result.state.challenger_percentage == 10
    assert len(admission.resources) == 2 and len(set(admission.resources)) == 1
    local = SimpleNamespace(local_only=True, hold=s.admission.hold)
    with pytest.raises(ValueError, match="shared|distributed|local"):
        s.api.reconcile_rollout(s.client, store=s.store, admission=local, clock=lambda: s.clock[0])
    assert len(s.client.puts) == 1


def test_promotion_pin_is_detached_and_completion_guard_requires_verified_terminal(api, plan):
    """Aliases may consume only a durable fully observed endpoint under the same lock."""
    tracking = Tracking()
    store = api.MLflowRolloutStore(tracking, "promotion-run")
    client = Endpoint(plan)
    now = datetime(2026, 10, 10, tzinfo=UTC)
    pin = {"auto_promote": True, "config_sha256": "a" * 64}
    api.initialize_rollout(
        client,
        plan,
        store=store,
        admission=SingleWriterAdmission(),
        policy=DailyRolloutPolicy(increment_percentage=100),
        rollout_id="promote",
        clock=lambda: now,
        promotion=pin,
    )
    pin["auto_promote"] = False
    assert store.load()["promotion"]["auto_promote"] is True
    with (
        pytest.raises(ValueError, match="COMPLETE|complete"),
        api.hold_completed_rollout(client, store=store, admission=SingleWriterAdmission()),
    ):
        pytest.fail("active rollout must not admit alias promotion")
    for _ in range(2):
        now += timedelta(days=1)
        current = api.reconcile_rollout(
            client,
            store=store,
            admission=SingleWriterAdmission(),
            clock=lambda current_time=now: current_time,
        )
        result = api.advance_rollout(
            client,
            store=store,
            admission=SingleWriterAdmission(),
            evidence=evidence(current.state, now),
            clock=lambda current_time=now: current_time,
        )
    with api.hold_completed_rollout(
        client, store=store, admission=SingleWriterAdmission()
    ) as record:
        assert record["state"]["phase"] == "COMPLETE"
        assert record["receipt_id"] == result.receipt_id
    client.value["state"]["config_update"] = "IN_PROGRESS"
    with (
        pytest.raises(ValueError, match="ready|settled"),
        api.hold_completed_rollout(client, store=store, admission=SingleWriterAdmission()),
    ):
        pytest.fail("pending native deployment cannot admit alias promotion")
    assert len(client.puts) == 1


@pytest.mark.parametrize("change", ["missing", "malformed", "foreign_run", "nonfinite"])
def test_bad_durable_pointer_or_receipt_stops_before_mutation(setup, change):
    """Missing or corrupt durable identities cannot create a fresh rollout implicitly."""
    s = setup
    if change == "missing":
        s.tracking.tags.clear()
    elif change == "malformed":
        s.tracking.tags[next(iter(s.tracking.tags))] = '{"receipt_id":"../outside","sha256":"x"}'
    else:
        path = next(iter(s.tracking.artifacts))
        value = json.loads(s.tracking.artifacts[path])
        value["run_id" if change == "foreign_run" else "policy"] = (
            "different" if change == "foreign_run" else float("nan")
        )
        s.tracking.artifacts[path] = json.dumps(value)
    with pytest.raises(ValueError):
        s.api.reconcile_rollout(
            s.client, store=s.store, admission=s.admission, clock=lambda: s.clock[0]
        )
    assert s.client.puts == []


def test_initialization_rejects_missing_pointer_with_existing_artifacts(setup):
    """Lost tags cannot overwrite a prior rollout's durable receipts."""
    s = setup
    s.tracking.tags.clear()
    with pytest.raises(ValueError, match="existing|already"):
        s.api.initialize_rollout(
            s.client,
            s.plan,
            store=s.store,
            admission=s.admission,
            policy=DailyRolloutPolicy(),
            rollout_id="retry",
            clock=lambda: s.clock[0],
        )
    assert s.client.puts == []


def test_plan_mutation_is_rejected_before_initialization(api, plan):
    """Frozen dataclasses do not make their nested request dictionaries immutable."""
    plan.config["config"]["served_entities"][1]["entity_version"] = "99"
    client = Endpoint(plan)
    tracking = Tracking()
    with pytest.raises(ValueError, match="composition"):
        api.initialize_rollout(
            client,
            plan,
            store=api.MLflowRolloutStore(tracking, "mutated"),
            admission=SingleWriterAdmission(),
            policy=DailyRolloutPolicy(),
            rollout_id="mutated",
        )
    assert not tracking.artifacts and not client.puts


def test_readiness_clock_rollback_leaves_prepared_intent(setup):
    """Clock regression after native mutation cannot advance the persisted stage."""
    s = setup
    s.clock[0] += timedelta(days=1)
    s.client.after_put = lambda: s.clock.__setitem__(0, s.clock[0] - timedelta(minutes=1))
    with pytest.raises(s.api.RolloutOutcomeUnknownError, match="unknown"):
        advance(s)
    assert s.store.load()["status"] == "PREPARED"
    assert s.store.load()["state"]["challenger_percentage"] == 0


def test_foreign_pending_configuration_is_never_treated_as_ready(setup):
    """Native READY does not override an outstanding unowned pending configuration."""
    s = setup
    s.clock[0] += timedelta(days=1)
    s.client.value["pending_config"] = deepcopy(s.client.value["config"])
    with pytest.raises(ValueError, match="pending"):
        advance(s)
    assert s.client.puts == []


def test_initialization_rejects_unsettled_pending_config(api, plan):
    """A contradictory READY flag cannot admit a pending native configuration."""
    client = Endpoint(plan)
    client.value["pending_config"] = deepcopy(client.value["config"])
    tracking = Tracking()
    with pytest.raises(ValueError, match="pending|settled"):
        api.initialize_rollout(
            client,
            plan,
            store=api.MLflowRolloutStore(tracking, "pending"),
            admission=SingleWriterAdmission(),
            policy=DailyRolloutPolicy(),
            rollout_id="pending",
        )
    assert not tracking.artifacts and not client.puts


def test_same_workspace_spellings_share_one_admission_key(api):
    """Default HTTPS port and harmless URL spelling cannot split writer authority."""
    assert api.rollout_resource_id(
        "https://WORKSPACE.example:443/", "daily"
    ) == api.rollout_resource_id("https://workspace.example", "daily")
