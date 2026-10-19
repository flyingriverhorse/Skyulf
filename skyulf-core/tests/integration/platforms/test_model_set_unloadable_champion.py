"""Cross-release sets retain their champion until a gated explicit replacement."""

import html
import json
from copy import deepcopy
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import Mock

import pytest


@pytest.fixture
def release_case(tmp_path, workflow_config, monkeypatch):
    """Save real fitted models and a validated set before changing the runtime pin."""
    mlflow = pytest.importorskip("mlflow")
    from test_training_branches import _configs, _data

    from skyulf.inference import _manifest
    from skyulf.integrations.databricks.model_sets import model_set_project as project
    from skyulf.integrations.databricks.model_sets import model_set_release as release
    from skyulf.integrations.databricks.training import branches as training
    from skyulf.integrations.databricks.training.fitting import candidate
    from skyulf.integrations.mlflow.spark import _spark_environment

    monkeypatch.chdir(tmp_path)
    uri = f"sqlite:///{(tmp_path / 'registry.db').as_posix()}"
    endpoints = {"tracking_uri": uri, "registry_uri": uri}
    client = mlflow.MlflowClient(**endpoints)
    client.create_experiment("recovery", artifact_location=(tmp_path / "runs").as_uri())
    data = _data()
    monkeypatch.setattr(
        candidate,
        "read_training_snapshot",
        lambda spark, spec: data.loc[:, list(spec.source_columns)].copy(),
    )
    monkeypatch.setattr(project, "approval_frame", lambda *args: data[["id", "x"]].copy())
    spark = Mock()
    spark.read.option.return_value.table.return_value = SimpleNamespace(dtypes=[("id", "bigint")])
    configs = _configs(workflow_config, store=uri)
    del configs["ensemble"]
    for name, config in configs.items():
        config.update(
            cv_enabled=False,
            min_improvement=0.0,
            quality_gates=None,
            quality_threshold=0.0 if name == "category" else 100.0,
        )
    settings = {
        "model_name": "workspace.test.recovery_set",
        "promotion_policy": "automatic",
        "composition_config": {"outputs": []},
    }
    base = {**configs["amount"], "max_rows": 100, "max_input_mb": 4}

    def train(number, *, recipes=None):
        """Use the real branch fit/save/register/package flow and one pinned baseline."""
        recipes = deepcopy(configs if recipes is None else recipes)
        pinned, versions = release.pin_model_set_baseline(settings, recipes, endpoints)
        branches = training.prepare_training_branches(None, recipes, champion_versions=versions)
        outcome = training.train_branches(
            None,
            branches,
            **endpoints,
            experiment_name="recovery",
            artifact_path=tmp_path / f"fit{number}",
        )
        saved = project.package_training_model_set(
            spark, branches, outcome, pinned, composition_source="", **endpoints
        )
        return saved, pinned

    first, pinned = train(1)
    initial = release.automatic_model_set_release(spark, first, pinned, base)
    assert initial["alias_change"]["kind"] == "initial"
    original_version = _manifest.version
    monkeypatch.setattr(
        _manifest,
        "version",
        lambda name: "0.9.5+recovery" if name == "skyulf-core" else original_version(name),
    )
    distribution = _spark_environment.distribution("skyulf-core")

    def runtime_distribution(name):
        """Keep the packaged wheel metadata consistent with the simulated release identity."""
        assert name == "skyulf-core"
        runtime = _manifest.version(name)
        return SimpleNamespace(
            version=runtime,
            read_text=lambda key: (distribution.read_text(key) or "").replace(
                f"Version: {distribution.version}\n", f"Version: {runtime}\n"
            ),
        )

    monkeypatch.setattr(_spark_environment, "distribution", runtime_distribution)
    return SimpleNamespace(
        client=client,
        endpoints=endpoints,
        first=first,
        settings=settings,
        configs=configs,
        base=base,
        spark=spark,
        train=train,
        release=release,
        original_version=original_version,
        directory=tmp_path,
    )


def test_real_saved_old_set_can_pin_the_training_baseline(release_case):
    """The runtime guard must prevent old pickle loads without blocking new training."""
    from skyulf.inference.fitted_pipeline import ArtifactRuntimeMismatchError
    from skyulf.integrations.mlflow.models.model_set import load_registered_model_set

    case = release_case
    with pytest.raises(ArtifactRuntimeMismatchError, match="skyulf-core"):
        load_registered_model_set(case.first, **case.endpoints)
    settings, versions = case.release.pin_model_set_baseline(
        case.settings, case.configs, case.endpoints
    )
    assert settings["expected_champion_version"] == "1"
    assert versions == {"amount": "1", "category": "1"}


def test_cross_release_retraining_requires_gated_explicit_approval(release_case, monkeypatch):
    """A release upgrade must preserve aliases until concrete audited consent succeeds."""
    from skyulf.inference import fitted_pipeline
    from skyulf.integrations.databricks.model_sets.model_set_project import (
        render_model_set_result,
        run_model_set_operator,
        set_operator_actions,
    )
    from skyulf.integrations.mlflow.lifecycle.promotion import AliasConflictError

    case = release_case
    real_load = fitted_pipeline._load_pipeline_payload

    def current_payload(payload, manifest):
        """Reject any attempted execution of an old pickle independently of runtime guards."""
        assert dict(manifest.requirements)["skyulf-core"] == "0.9.5+recovery"
        return real_load(payload, manifest)

    monkeypatch.setattr(fitted_pipeline, "_load_pipeline_payload", current_payload)
    candidate, pinned = case.train(2)
    pending = case.release.automatic_model_set_release(case.spark, candidate, pinned, case.base)
    assert pending["alias_change"] is None
    assert pending["quality"]["reason"] == "champion_runtime_unavailable"
    assert not pending["quality"]["passed"]
    assert str(case.client.get_model_version_by_alias(candidate.name, "champion").version) == "1"
    for result in pending["quality"]["components"].values():
        comparison = result["comparison"]
        assert comparison["champion_version"] == "1"
        assert comparison["champion_metrics"] is None
        assert (
            f"requires {case.original_version('skyulf-core')}, found"
            in comparison["champion_unavailable"]
        )
    payload = {
        **pending,
        "model_set_candidate": {"name": candidate.name, "version": candidate.version},
    }
    actions = set_operator_actions(payload)
    assert actions["approve"]["allow_unloadable_champion"] == "true"
    assert (
        "databricks bundle run train -t <target> --params lifecycle_action=approve,"
        "candidate_version=2,expected_champion_version=1,allow_unloadable_champion=true"
    ) in html.unescape(render_model_set_result(payload))
    with pytest.raises(case.release.ModelSetQualityError):
        case.release.approve_project_model_set(
            case.spark,
            candidate,
            case.base,
            expected_champion_version="1",
            policy="manual_approval",
        )
    with pytest.raises(AliasConflictError):
        case.release.approve_project_model_set(
            case.spark,
            candidate,
            case.base,
            expected_champion_version="2",
            policy="manual_approval",
            allow_unloadable_champion=True,
        )
    case.client.delete_registered_model_alias(candidate.name, "champion")
    with pytest.raises(AliasConflictError, match="receipt exists without its alias"):
        case.release.pin_model_set_baseline(case.settings, case.configs, case.endpoints)
    case.client.set_registered_model_alias(candidate.name, "champion", "1")
    proof_key = "model_set_validation_sha256"
    proof = case.client.get_model_version(candidate.name, "1").tags[proof_key]
    case.client.delete_model_version_tag(candidate.name, "1", proof_key)
    with pytest.raises(ValueError, match="no durable validation evidence"):
        case.release.approve_project_model_set(
            case.spark,
            candidate,
            case.base,
            expected_champion_version="1",
            policy="manual_approval",
            allow_unloadable_champion=True,
        )
    assert str(case.client.get_model_version_by_alias(candidate.name, "champion").version) == "1"
    case.client.set_model_version_tag(candidate.name, "1", proof_key, proof)
    output = run_model_set_operator(
        case.spark,
        {"lifecycle_action": "approve"},
        case.settings,
        {
            "candidate_version": "2",
            "expected_champion_version": "1",
            "allow_unloadable_champion": True,
        },
        config=case.base,
    )
    receipt = output["receipt"]
    assert (receipt["kind"], receipt["prior_version"], receipt["new_version"]) == (
        "promotion",
        "1",
        "2",
    )
    assert str(case.client.get_model_version_by_alias(candidate.name, "champion").version) == "2"
    version = case.client.get_model_version(candidate.name, "2")
    evidence_uri = version.tags["model_set_validation_artifact"]
    run_id, path = evidence_uri.removeprefix("runs:/").split("/", 1)
    evidence = json.loads(Path(case.client.download_artifacts(run_id, path)).read_text())
    assert evidence["quality"]["allow_unloadable_champion"] is True
    assert evidence["quality"]["expected_champion_version"] == "1"
    assert all(result["passed"] for result in evidence["quality"]["components"].values())
    _assert_approved_handoffs(case, output, monkeypatch)
    from skyulf.inference.fitted_pipeline import ArtifactRuntimeMismatchError
    from skyulf.integrations.mlflow.lifecycle.model_set_lifecycle import rollback_model_set
    from skyulf.integrations.mlflow.lifecycle.promotion import (
        AliasChangeReceipt,
        ExclusiveAliasWriterAdmission,
    )

    with pytest.raises(ArtifactRuntimeMismatchError):
        rollback_model_set(
            AliasChangeReceipt(**receipt),
            expected_current_version="2",
            admission=ExclusiveAliasWriterAdmission(),
            **case.endpoints,
        )
    assert str(case.client.get_model_version_by_alias(candidate.name, "champion").version) == "2"


def _assert_approved_handoffs(case, output, monkeypatch):
    """The real recovery receipt must reach score and concrete-version serving dispatch."""
    from skyulf.integrations.databricks.jobs import serving_endpoint_job
    from skyulf.integrations.databricks.jobs.training import training_node_notebook as notebook

    serving_config = case.directory / "serving.yml"
    serving_config.write_text(
        "serving_endpoint:\n  enabled: true\n  follow_champion: true\n", encoding="utf-8"
    )
    values = {
        "decision_result_state": "success",
        "deployed_serving_endpoint": "true",
        "bundle_mode": "production",
        "serving_config_path": str(serving_config),
    }
    store = SimpleNamespace(
        request={
            "action": "approve",
            "settings": case.settings,
            "config": case.base,
            "score_handoff": "after_alias_change",
        },
        receipt=lambda phase: {"output": output},
    )
    monkeypatch.setattr(notebook, "_saved", lambda _: (store, {}))
    monkeypatch.setattr(
        serving_endpoint_job,
        "read_serving_endpoint_settings",
        lambda _: {
            "enabled": True,
            "handoff": "after_alias_change",
            "model_name": case.first.name,
        },
    )
    dbutils = Mock()
    dbutils.widgets.getAll.return_value = values
    notebook.run_models_report_notebook(dbutils)
    emitted = {
        call.kwargs["key"]: call.kwargs["value"]
        for call in dbutils.jobs.taskValues.set.call_args_list
    }
    assert emitted == {
        "score_requested": True,
        "serving_requested": True,
        "serving_model_version": "2",
    }


@pytest.mark.parametrize("threshold", [None, 0.000001])
def test_recovery_cannot_override_missing_or_failed_candidate_gate(release_case, threshold):
    """Consent to an unavailable comparison must never waive the candidate's own quality."""
    case = release_case
    case.settings["promotion_policy"] = "manual_approval"
    recipes = deepcopy(case.configs)
    recipes["amount"]["quality_threshold"] = threshold
    recipes["amount"]["pipeline"]["modeling"] = {
        "type": "ridge_regression",
        "params": {"alpha": 1e6},
    }
    candidate, pinned = case.train(2, recipes=recipes)
    case.release.automatic_model_set_release(case.spark, candidate, pinned, case.base)
    with pytest.raises(case.release.ModelSetQualityError):
        case.release.approve_project_model_set(
            case.spark,
            candidate,
            case.base,
            expected_champion_version="1",
            policy="manual_approval",
            allow_unloadable_champion=True,
        )
    assert str(case.client.get_model_version_by_alias(candidate.name, "champion").version) == "1"


@pytest.mark.parametrize("relative", ["manifest.json", "components/amount/pipeline.pkl"])
def test_old_set_metadata_rejects_corrupt_package(release_case, relative):
    """A release mismatch must never turn modified package bytes into trusted references."""
    case = release_case
    source = case.client.get_model_version(case.first.name, "1").source
    assert source.startswith("runs:/")
    run_id, artifact_path = source.removeprefix("runs:/").split("/", 1)
    directory = case.directory / "runs" / run_id / "artifacts" / artifact_path
    files = list(directory.rglob("components/amount/pipeline.pkl"))
    assert len(files) == 1
    path = files[0].parents[2] / relative
    path.write_bytes(path.read_bytes() + b"broken")
    with pytest.raises(ValueError):
        case.release.pin_model_set_baseline(case.settings, case.configs, case.endpoints)
    assert str(case.client.get_model_version_by_alias(case.first.name, "champion").version) == "1"


def test_loadable_set_cannot_use_the_recovery_override(release_case, monkeypatch):
    """The explicit flag cannot waive a normal same-release comparison."""
    from skyulf.inference import _manifest
    from skyulf.integrations.mlflow.lifecycle import model_set_lifecycle

    case = release_case
    monkeypatch.setattr(_manifest, "version", case.original_version)
    metadata = Mock(
        side_effect=AssertionError("Same-release training must not use fallback metadata.")
    )
    monkeypatch.setattr(case.release, "read_registered_model_set_manifest", metadata)
    monkeypatch.setattr(model_set_lifecycle, "read_registered_model_set_manifest", metadata)
    candidate, pinned = case.train(2)
    pending = case.release.automatic_model_set_release(case.spark, candidate, pinned, case.base)
    assert pending["quality"]["reason"] == "no_component_improved"
    with pytest.raises(ValueError, match="only when the champion cannot be loaded"):
        case.release.approve_project_model_set(
            case.spark,
            candidate,
            case.base,
            expected_champion_version="1",
            policy="manual_approval",
            allow_unloadable_champion=True,
        )
    assert str(case.client.get_model_version_by_alias(candidate.name, "champion").version) == "1"
    metadata.assert_not_called()


@pytest.mark.parametrize("improvement,passed", [(-0.1, False), (0.0, True)])
def test_recovery_retains_comparable_component_non_regression(improvement, passed):
    """An unavailable peer must not conceal a measurable regression in another component."""
    from test_model_set_quality import _report

    from skyulf.integrations.databricks.model_sets.model_set_quality import (
        _set_decision,
        component_quality,
    )

    unavailable = _report(
        champion_version="1",
        champion_digest="b" * 64,
        reason="champion_runtime_unavailable",
        champion_unavailable="Core release mismatch",
    )
    comparable = _report(
        champion_version="1",
        champion_digest="c" * 64,
        improvement=improvement,
        reason="insufficient_improvement",
    )
    result = _set_decision(
        {
            "unavailable": component_quality(unavailable, allow_unloadable_champion=True),
            "comparable": component_quality(comparable, allow_unloadable_champion=True),
        },
        "1",
        True,
    )
    assert result["passed"] is passed
