"""Resolve real Bundle targets against local identity fixtures without cloud operations."""

import json
import subprocess
import threading
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from typing import Any

import pytest
import yaml
from test_databricks_bundle_generation import CLI, PROFILE, _generate_project

pytestmark = pytest.mark.skipif(not CLI or not PROFILE, reason="Requires opt-in installed CLI.")


@pytest.fixture
def offline_workspace(tmp_path, monkeypatch):
    """Permit only identity and workspace-path reads on a loopback HTTP server."""
    state: dict[str, Any] = {"id": "100001", "userName": "alice@example.invalid", "requests": []}

    class Handler(BaseHTTPRequestHandler):
        def log_message(self, format, *args):
            """Keep HTTP diagnostics in the fixture request list."""

        def do_GET(self):
            """Respond to CLI identity discovery and record unexpected endpoint use."""
            state["requests"].append(self.path)
            status = 200
            if self.path.startswith("/api/2.0/preview/scim/v2/Me"):
                result = {"id": state["id"], "userName": state["userName"], "active": True}
            elif self.path.startswith("/api/2.0/workspace/get-status"):
                result = {"object_type": "DIRECTORY", "object_id": 1}
            elif self.path.startswith("/api/2.0/policies/clusters/list"):
                result = {"policies": [{"policy_id": "local-policy-id", "name": "local-policy"}]}
            elif self.path == "/.well-known/databricks-config":
                result = {"oidc_endpoint": f"{host}/oidc"}
            else:
                status = 500
                result = {"message": "Unexpected request in offline validation"}
            self.send_response(status)
            self.send_header("Content-Type", "application/json")
            self.end_headers()
            self.wfile.write(json.dumps(result).encode())

    server = ThreadingHTTPServer(("127.0.0.1", 0), Handler)
    thread = threading.Thread(target=server.serve_forever, daemon=True)
    thread.start()
    host = f"http://127.0.0.1:{server.server_port}"
    config = tmp_path / "offline.cfg"
    config.write_text(f"[offline]\nhost={host}\ntoken=local-test-token\n", encoding="utf-8")
    for key in ("DATABRICKS_HOST", "DATABRICKS_TOKEN", "DATABRICKS_CONFIG_PROFILE"):
        monkeypatch.delenv(key, raising=False)
    monkeypatch.setenv("DATABRICKS_CONFIG_FILE", str(config))
    monkeypatch.setenv("DATABRICKS_AUTH_TYPE", "pat")
    monkeypatch.setattr("test_databricks_bundle_generation.PROFILE", "offline")
    state["host"] = host
    state["config"] = config
    try:
        yield state
    finally:
        server.shutdown()
        server.server_close()
        thread.join(timeout=5)


def _resolve(project, target, state):
    """Ask the real CLI to merge includes, targets, variables and personal identity."""
    state["config"].write_text(
        f"[offline]\nhost={state['host']}\ntoken=local-test-{state['id']}\n", encoding="utf-8"
    )
    dist = project / "dist"
    dist.mkdir(exist_ok=True)
    (dist / "skyulf_core-0.9.1-py3-none-any.whl").write_text(
        "Path resolution fixture only; no wheel is installed or executed.", encoding="utf-8"
    )
    targets_path = project / "deployment/targets.yml"
    targets = yaml.safe_load(targets_path.read_text())
    for settings in targets["targets"].values():
        settings.setdefault("workspace", {})["host"] = state["host"]
    targets_path.write_text(yaml.safe_dump(targets, sort_keys=False), encoding="utf-8")
    command = [
        str(CLI),
        "bundle",
        "validate",
        "--strict",
        "-t",
        target,
        "-p",
        "offline",
        "-o",
        "json",
    ]
    result = subprocess.run(command, cwd=project, capture_output=True, text=True, timeout=60)
    assert result.returncode == 0, result.stderr
    assert "Warning:" not in result.stderr, result.stderr
    return json.loads(result.stdout)


@pytest.mark.parametrize("target", ["test_development", "syst_development", "prod_development"])
@pytest.mark.parametrize("layout", ["single_model", "model_competition", "multi_target"])
def test_resolved_personal_targets_isolate_users_and_pause_clocks(
    tmp_path, target, layout, offline_workspace
):
    """Two users with the same short name must never resolve to the same model/output namespace."""
    project = _generate_project(
        tmp_path,
        training_layout=layout,
        personal_development_targets="true",
        deployment_identity="separate_service_principals",
        manage_job_permissions="true",
        retraining_mode="scheduled",
        scoring_mode="scheduled",
        retraining_pause_status="UNPAUSED",
        scoring_pause_status="UNPAUSED",
        model_set_output_mode="separate_views",
        combined_view_name="combined_results",
    )
    first = _resolve(project, target, offline_workspace)
    offline_workspace.update(id="200002", userName="alice@other.invalid")
    second = _resolve(project, target, offline_workspace)
    for bundle in (first, second):
        jobs = bundle["resources"]["jobs"]
        assert set(jobs) == {"train", "score"}
        for key, job in jobs.items():
            assert job["name"] == f"dev_alice_{key}"
            assert job["schedule"]["pause_status"] == "PAUSED"
            assert not job.get("run_as", {}).get("service_principal_name")
            assert not job.get("permissions")
            assert job["max_concurrent_runs"] == 1
        assert jobs["score"]["tasks"][0]["notebook_task"]["base_parameters"]["resource_suffix"]
    assert first["workspace"]["root_path"] != second["workspace"]["root_path"]
    assert (
        first["variables"]["resource_suffix"]["value"]
        != second["variables"]["resource_suffix"]["value"]
    )
    from skyulf.integrations.databricks.local_workflow import resolve_target_config

    config = json.loads((project / "config/workflow.json").read_text())
    names = []
    model_set_names = []
    for bundle in (first, second):
        bindings = {
            key: bundle["variables"][key]["value"]
            for key in (
                "catalog",
                "input_schema",
                "output_schema",
                "metadata_schema",
                "resource_suffix",
            )
        }
        names.append(resolve_target_config(config, bindings))
        if layout == "multi_target":
            model_set_names.append(_model_set_destinations(project, config, bindings))
    assert names[0]["training_table"] == names[1]["training_table"]
    assert names[0]["prediction_table"] != names[1]["prediction_table"]
    assert names[0]["model_name"] != names[1]["model_name"]
    if layout == "multi_target":
        assert model_set_names[0].isdisjoint(model_set_names[1])


def _model_set_destinations(project, config, bindings):
    """Resolve actual branch models and consumer views through the notebook loaders."""
    from skyulf.integrations.databricks.branch_notebook import load_training_branch_configs
    from skyulf.integrations.databricks.model_set_project import load_project_model_set

    values = {
        "config_path": str(project / "config/workflow.json"),
        "workflow_contract": "3",
        "deployed_score_handoff": config["score_handoff"],
        **bindings,
    }
    settings = load_project_model_set(values, config)
    assert settings is not None
    branches = load_training_branch_configs(values)
    assert branches
    publication = settings["publication"]
    assert publication["mode"] == "separate_views"
    destinations = {
        settings["model_name"],
        settings["prediction_table"],
        publication["combined_view"],
        *(branch["model_name"] for branch in branches.values()),
        *(publication["model_view_template"].format(branch=name) for name in branches),
    }
    assert all(bindings["resource_suffix"] in name for name in destinations)
    return destinations


@pytest.mark.parametrize(
    "identity", ["deployer", "shared_service_principal", "separate_service_principals"]
)
@pytest.mark.parametrize("target", ["test", "syst", "prod"])
def test_resolved_shared_targets_preserve_identity_and_acl_choices(
    tmp_path, identity, target, offline_workspace
):
    """CI-provided identities and per-job operator ACLs must reach the deployed job graph."""
    project = _generate_project(
        tmp_path, deployment_identity=identity, manage_job_permissions="true"
    )
    path = project / "deployment/variables.yml"
    contents = yaml.safe_load(path.read_text())
    variables = contents["variables"]
    for name in ("shared", "train", "score"):
        key = f"{name}_service_principal"
        if key in variables:
            variables[key]["default"] = f"{name}-application-id"
    for job in ("train", "score"):
        variables[f"{job}_permissions"]["default"] = [
            {"level": "CAN_MANAGE_RUN", "group_name": f"{job}-operators"},
            {"level": "CAN_MANAGE", "user_name": "alice@example.invalid"},
        ]
    path.write_text(yaml.safe_dump(contents, sort_keys=False), encoding="utf-8")
    bundle = _resolve(project, target, offline_workspace)
    for job in ("train", "score"):
        resource = bundle["resources"]["jobs"][job]
        assert resource["name"] == f"{project.name}_{job}"
        assert {"level": "CAN_MANAGE_RUN", "group_name": f"{job}-operators"} in resource[
            "permissions"
        ]
        if identity != "deployer":
            role = "shared" if identity == "shared_service_principal" else job
            assert resource["run_as"]["service_principal_name"] == f"{role}-application-id"
    assert bundle["variables"]["resource_suffix"]["value"] == ""


def test_external_identity_and_acl_management_remain_optional(tmp_path, offline_workspace):
    """An existing service account deployment must not require new identity or ACL variables."""
    project = _generate_project(tmp_path)
    result = _resolve(project, "prod", offline_workspace)
    assert "train_service_principal" not in result["variables"]
    assert "train_permissions" not in result["variables"]
    assert not result["resources"]["jobs"]["score"].get("permissions")


def test_personal_policy_compute_resolves_without_changing_job_identity(
    tmp_path, offline_workspace
):
    """Cluster-policy lookup must remain valid after splitting deployment settings."""
    project = _generate_project(
        tmp_path,
        personal_development_targets="true",
        compute_mode="policy_cluster",
        cluster_policy_name="local-policy",
    )
    result = _resolve(project, "test_development", offline_workspace)
    for job in result["resources"]["jobs"].values():
        assert job["job_clusters"][0]["new_cluster"]["policy_id"] == "local-policy-id"


@pytest.mark.parametrize("layout", ["single_model", "model_competition", "multi_target"])
@pytest.mark.parametrize("compute", ["serverless", "policy_cluster"])
@pytest.mark.parametrize("recovery", ["false", "true"])
def test_operational_settings_resolve_without_retrying_lifecycle(
    tmp_path, offline_workspace, layout, compute, recovery
):
    """Timeouts and alerts must reach every graph while only scoring can opt into retries."""
    project = _generate_project(
        tmp_path,
        training_layout=layout,
        compute_mode=compute,
        cluster_policy_name="local-policy",
        auto_rebuild_on_cdf_expiry=recovery,
    )
    path = project / "deployment/variables.yml"
    contents = yaml.safe_load(path.read_text())
    variables = contents["variables"]
    for role in ("train", "score"):
        assert variables[f"{role}_timeout_seconds"]["default"] == 0
        assert variables[f"{role}_task_timeout_seconds"]["default"] == 0
        assert variables[f"{role}_email_notifications"]["default"] == {}
        assert variables[f"{role}_webhook_notifications"]["default"] == {}
        assert variables[f"{role}_health_rules"]["default"] == []
        variables[f"{role}_timeout_seconds"]["default"] = 7200
        variables[f"{role}_task_timeout_seconds"]["default"] = 1800
        variables[f"{role}_health_rules"]["default"] = [
            {"metric": "RUN_DURATION_SECONDS", "op": "GREATER_THAN", "value": 3600}
        ]
        variables[f"{role}_email_notifications"]["default"] = {
            "on_failure": ["operator@example.invalid"],
            "on_duration_warning_threshold_exceeded": ["operator@example.invalid"],
        }
        variables[f"{role}_webhook_notifications"]["default"] = {
            "on_failure": [{"id": "00000000-0000-0000-0000-000000000001"}]
        }
    assert variables["score_max_retries"]["default"] == 0
    variables["score_max_retries"]["default"] = 2
    variables["score_min_retry_interval_millis"]["default"] = 60000
    path.write_text(yaml.safe_dump(contents, sort_keys=False), encoding="utf-8")
    bundle = _resolve(project, "test", offline_workspace)
    for role, job in bundle["resources"]["jobs"].items():
        assert job["timeout_seconds"] == 7200
        assert job["health"]["rules"][0]["value"] == 3600
        assert job["email_notifications"]["on_failure"] == ["operator@example.invalid"]
        assert job["webhook_notifications"]["on_failure"][0]["id"].endswith("000001")
        for task in job["tasks"]:
            expected_retries = 2 if role == "score" and "notebook_task" in task else 0
            assert task.get("max_retries", 0) == expected_retries
            assert not task.get("retry_on_timeout", False)
            if "notebook_task" in task:
                assert task["timeout_seconds"] == 1800
                if compute == "serverless":
                    assert task["disable_auto_optimization"] is True
    assert bundle["resources"]["jobs"]["score"]["tasks"][0]["min_retry_interval_millis"] == 60000
    score_tasks = {task["task_key"]: task for task in bundle["resources"]["jobs"]["score"]["tasks"]}
    assert ("recover_predictions" in score_tasks) is (recovery == "true")
    if recovery == "true":
        assert score_tasks["scoring_report"]["run_if"] == "NONE_FAILED"
