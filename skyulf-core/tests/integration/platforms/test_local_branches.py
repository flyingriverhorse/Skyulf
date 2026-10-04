"""Pinned multi-target training preserves independent candidates and failure evidence."""

import json
from copy import deepcopy
from dataclasses import replace
from datetime import UTC, datetime
from pathlib import Path
from unittest.mock import MagicMock

import pandas as pd
import pytest

from skyulf.integrations.databricks import local_branches as branches
from skyulf.integrations.databricks import local_retraining as training


def _configs(workflow_config, *, engine="pandas", store=None):
    """Build independent target recipes over one shared immutable source."""
    result = {}
    for name in ("amount", "category", "ensemble"):
        config = deepcopy(workflow_config)
        config.update(
            engine=engine,
            score_handoff="disabled",
            training_version=7,
            training_window_mode="full_snapshot",
            split_strategy="random",
            filter_unavailable_results=False,
            record_key_columns=["id"],
            target_column=name,
            model_name=f"workspace.test.{name}",
            max_rows=100,
            quality_threshold=None,
        )
        for field in (
            "window_timezone",
            "monthly_lookback_months",
            "start",
            "cutoff",
            "holdout_start",
            "event_column",
            "result_cutoff",
            "result_available_at_column",
        ):
            config[field] = None
        if store:
            config.update(tracking_uri=store, registry_uri=store)
        if name == "category":
            config.update(task="classification", metric="heldout_accuracy", stratify=True)
            config["pipeline"]["modeling"] = {"type": "logistic_regression"}
        elif name == "ensemble":
            config["pipeline"]["modeling"] = {
                "type": "voting_regressor",
                "params": {"base_estimators": ["linear_regression", "ridge"]},
            }
        else:
            config.update(cv_enabled=True, cv_folds=2)
        result[name] = config
    return result


def test_prepare_validates_all_configs_before_source_or_registry(workflow_config, monkeypatch):
    """A later invalid branch must prevent even an earlier branch's external work."""
    configs = _configs(workflow_config)
    configs["ensemble"]["metric"] = "invalid"
    spark = MagicMock()
    champion = MagicMock()
    monkeypatch.setattr(branches, "controlled_champion_version", champion)
    with pytest.raises(ValueError):
        branches.prepare_training_branches(spark, configs)
    spark.sql.assert_not_called()
    champion.assert_not_called()


def test_latest_resolved_once_and_plan_roundtrips(workflow_config, monkeypatch):
    """A saved plan retains one source version without consulting latest on replay."""
    configs = _configs(workflow_config)
    for config in configs.values():
        config["training_version"] = None
    original = deepcopy(configs)
    spark = MagicMock()
    spark.sql.return_value.select.return_value.orderBy.return_value.first.return_value = {
        "version": 11
    }
    monkeypatch.setattr(branches, "controlled_champion_version", lambda *args, **kwargs: None)
    prepared = branches.prepare_training_branches(
        spark, configs, now=datetime(2026, 3, 1, tzinfo=UTC)
    )
    restored = branches.restore_training_branches(
        json.loads(json.dumps(branches.branch_training_payload(prepared)))
    )
    assert [item.name for item in prepared] == ["amount", "category", "ensemble"]
    assert all(item.spec.version == 11 and item.spec.drop_missing_labels for item in prepared)
    assert restored == prepared
    assert configs == original
    spark.sql.assert_called_once()


@pytest.mark.parametrize(
    "field,value",
    [
        ("target_column", "amount"),
        ("model_name", "workspace.test.amount"),
        ("training_table", "workspace.test.other"),
        ("training_version", 8),
        ("record_key_columns", ["other_id"]),
        ("input_columns", ["amount"]),
        ("promotion_policy", "automatic"),
        ("score_handoff", "after_alias_change"),
    ],
)
def test_prepare_rejects_incompatible_branches(workflow_config, monkeypatch, field, value):
    """Branches cannot compete on labels, leak sibling targets or diverge in source identity."""
    configs = _configs(workflow_config)
    configs["ensemble"][field] = value
    spark = MagicMock()
    champion = MagicMock()
    monkeypatch.setattr(branches, "controlled_champion_version", champion)
    with pytest.raises(ValueError):
        branches.prepare_training_branches(spark, configs)
    spark.sql.assert_not_called()
    champion.assert_not_called()


def test_explicit_version_pins_unspecified_siblings(workflow_config, monkeypatch):
    """An explicit common snapshot wins without an unnecessary latest lookup."""
    configs = _configs(workflow_config)
    configs["amount"]["training_version"] = None
    monkeypatch.setattr(branches, "controlled_champion_version", lambda *args, **kwargs: None)
    spark = MagicMock()
    prepared = branches.prepare_training_branches(spark, configs)
    spark.sql.assert_not_called()
    assert {branch.spec.version for branch in prepared} == {7}


@pytest.fixture
def tracked(tmp_path):
    """Use actual SQLite tracking and a local artifact store for end-to-end candidates."""
    mlflow = pytest.importorskip("mlflow")
    store = f"sqlite:///{(tmp_path / 'registry.db').as_posix()}"
    client = mlflow.MlflowClient(tracking_uri=store, registry_uri=store)
    client.create_experiment("branches", artifact_location=(tmp_path / "mlruns").as_uri())
    return store, client


def _data():
    """Each target has its own missing-label population over sixty shared records."""
    x = list(range(60))
    frame = pd.DataFrame(
        {
            "id": x,
            "x": [float(value) for value in x],
            "amount": [2.0 * value for value in x],
            "category": [float(value % 2) for value in x],
            "ensemble": [3.0 * value for value in x],
        }
    )
    frame.loc[:2, "amount"] = None
    frame.loc[3:7, "category"] = None
    frame.loc[8:15, "ensemble"] = None
    return frame


def _artifact(client, run_id, filename):
    """Read recorded JSON evidence through the same MLflow artifact interface as callers."""
    return json.loads(Path(client.download_artifacts(run_id, filename)).read_text(encoding="utf-8"))


@pytest.mark.parametrize("engine", ["pandas", "polars"])
def test_real_candidates_keep_independent_labels_and_replay_plan(
    workflow_config, monkeypatch, tmp_path, tracked, engine
):
    """Real models, CV and replay must retain independent labels and reproducible holdouts."""
    from skyulf.inference.local_pipeline import load_local_pipeline, predict_local_pipeline

    store, client = tracked
    configs = _configs(workflow_config, engine=engine, store=store)
    reads = []

    def read(spark, spec):
        """Replace only Spark materialization with the exact projected source snapshot."""
        reads.append((spec.version, spec.target_column))
        return _data().loc[:, list(spec.source_columns)].copy()

    monkeypatch.setattr(training, "read_training_snapshot", read)
    prepared = branches.prepare_training_branches(None, configs)
    plan = branches.branch_training_payload(prepared)
    results = []
    for attempt in range(2):
        root = tmp_path / f"fit_{attempt}"
        restored = branches.restore_training_branches(json.loads(json.dumps(plan)))
        result = branches.train_local_branches(
            None,
            restored,
            tracking_uri=store,
            registry_uri=store,
            experiment_name="branches",
            artifact_path=root,
        )
        results.append(result)
        assert client.get_run(result.parent_run_id).info.status == "FINISHED"
        assert _artifact(client, result.parent_run_id, "branch_training_plan.json") == plan
        progress = _artifact(client, result.parent_run_id, "branch_training_progress.json")
        assert progress["status"] == "complete"
        assert set(progress["completed"]) == set(configs)
        for name, candidate in result.components.items():
            child = client.get_run(candidate.run_id)
            assert child.info.status == "FINISHED"
            assert child.data.tags["mlflow.parentRunId"] == result.parent_run_id
            assert child.data.tags["skyulf.training.branch"] == name
            assert child.data.tags["skyulf.training.plan_sha256"] == result.plan_sha256
            assert (
                candidate.training_rows + candidate.holdout_rows
                == {"amount": 57, "category": 55, "ensemble": 52}[name]
            )
            assert candidate.unavailable_labels == {"amount": 3, "category": 5, "ensemble": 8}[name]
            assert client.get_registered_model(candidate.model_name).aliases == {}
            model = load_local_pipeline(root / name)
            assert model.manifest.fitted_engine == engine
            predictions = predict_local_pipeline(pd.DataFrame({"x": [1.0, 2.0]}), model)
            assert len(predictions) == 2
            if name == "ensemble":
                assert model.pipeline.config["modeling"]["params"]["base_estimators"] == [
                    "linear_regression",
                    "ridge",
                ]
                estimator = model.pipeline.model_estimator
                assert estimator is not None
                assert set(estimator._unwrap_tuned_model().named_estimators_) == {
                    "linear_regression",
                    "ridge",
                }
        cv = _artifact(client, result.components["amount"].run_id, "cross_validation.json")
        assert cv["training_rows"] == result.components["amount"].training_rows
    assert reads == [(7, name) for name in sorted(configs)] * 2
    assert results[0].parent_run_id != results[1].parent_run_id
    assert results[0].plan_sha256 == results[1].plan_sha256
    assert {name: item.holdout_key_sha256 for name, item in results[0].components.items()} == {
        name: item.holdout_key_sha256 for name, item in results[1].components.items()
    }


def test_partial_failure_preserves_candidates_and_stops_later_branches(
    workflow_config, monkeypatch, tmp_path, tracked
):
    """A later fit failure leaves a failed parent and usable completed immutable candidates."""
    store, client = tracked
    prepared = branches.prepare_training_branches(None, _configs(workflow_config, store=store))
    reads = []

    def read(spark, spec):
        """Fail the second source read after the first model has completed registration."""
        reads.append(spec.target_column)
        if spec.target_column == "category":
            raise RuntimeError("branch source failed")
        return _data().loc[:, list(spec.source_columns)].copy()

    monkeypatch.setattr(training, "read_training_snapshot", read)
    with pytest.raises(RuntimeError, match="branch source failed"):
        branches.train_local_branches(
            None,
            prepared,
            tracking_uri=store,
            registry_uri=store,
            experiment_name="branches",
            artifact_path=tmp_path / "fit",
        )
    runs = client.search_runs([client.get_experiment_by_name("branches").experiment_id])
    parent = next(run for run in runs if "mlflow.parentRunId" not in run.data.tags)
    assert parent.info.status == "FAILED"
    progress = _artifact(client, parent.info.run_id, "branch_training_progress.json")
    assert progress["status"] == "failed"
    assert progress["failed_branch"] == "category"
    assert set(progress["completed"]) == {"amount"}
    assert reads == ["amount", "category"]
    assert "branch_training_result.json" not in {
        item.path for item in client.list_artifacts(parent.info.run_id)
    }
    assert client.get_model_version("workspace.test.amount", "1").status == "READY"
    assert client.get_registered_model("workspace.test.amount").aliases == {}


@pytest.mark.parametrize("field", ["tracking_uri", "registry_uri"])
def test_endpoints_must_match_before_preparation_and_replay(
    workflow_config, monkeypatch, tmp_path, field
):
    """A concrete champion version must never silently move to a different registry."""
    configs = _configs(workflow_config)
    configs["ensemble"][field] = "sqlite:///different.db"
    champion = MagicMock(return_value=None)
    monkeypatch.setattr(branches, "controlled_champion_version", champion)
    with pytest.raises(ValueError, match=field):
        branches.prepare_training_branches(None, configs)
    champion.assert_not_called()
    prepared = branches.prepare_training_branches(None, _configs(workflow_config))
    restored = branches.restore_training_branches(branches.branch_training_payload(prepared))
    run = MagicMock()
    monkeypatch.setattr(branches, "track_run", run)
    params = {"tracking_uri": "databricks", "registry_uri": "databricks-uc"}
    params[field] = "sqlite:///different.db"
    with pytest.raises(ValueError, match=field):
        branches.train_local_branches(
            None, restored, **params, experiment_name="branches", artifact_path=tmp_path
        )
    run.assert_not_called()


def test_train_validates_later_branch_before_parent_or_reads(
    workflow_config, monkeypatch, tmp_path
):
    """Direct dataclass callers receive the same complete preflight as workflow callers."""
    monkeypatch.setattr(branches, "controlled_champion_version", lambda *args, **kwargs: None)
    prepared = branches.prepare_training_branches(None, _configs(workflow_config))
    invalid = (*prepared[:-1], replace(prepared[-1], metric="heldout_accuracy"))
    run = MagicMock()
    read = MagicMock()
    monkeypatch.setattr(branches, "track_run", run)
    monkeypatch.setattr(training, "read_training_snapshot", read)
    with pytest.raises(ValueError):
        branches.train_local_branches(
            None,
            invalid,
            tracking_uri="databricks",
            registry_uri="databricks-uc",
            experiment_name="branches",
            artifact_path=tmp_path,
        )
    run.assert_not_called()
    read.assert_not_called()


@pytest.mark.parametrize("name", ["../escape", "a/b", "a.b", "", "CON", "lpt1"])
def test_names_are_portable_artifact_identifiers(workflow_config, monkeypatch, name):
    """Names cannot escape artifact roots or create reserved Windows device paths."""
    config = _configs(workflow_config)["amount"]
    champion = MagicMock()
    monkeypatch.setattr(branches, "controlled_champion_version", champion)
    with pytest.raises(ValueError, match="Branch names"):
        branches.prepare_training_branches(None, {name: config})
    champion.assert_not_called()


def test_cross_target_filter_dependencies_are_rejected(workflow_config, monkeypatch):
    """A sibling label must not determine this target's eligible training population."""
    configs = _configs(workflow_config)
    configs["amount"]["pre_split_steps"] = [
        {"name": "leak", "transformer": "DropMissingRows", "params": {"subset": ["category"]}}
    ]
    champion = MagicMock()
    monkeypatch.setattr(branches, "controlled_champion_version", champion)
    with pytest.raises(ValueError, match="other targets"):
        branches.prepare_training_branches(None, configs)
    champion.assert_not_called()


@pytest.mark.parametrize(
    "mutation", ["unknown", "missing_spec", "unknown_cv", "source", "boolean_version"]
)
def test_saved_plan_rejects_ambiguous_or_changed_contracts(workflow_config, monkeypatch, mutation):
    """Replay cannot silently invent defaults or disregard unsupported saved decisions."""
    monkeypatch.setattr(branches, "controlled_champion_version", lambda *args, **kwargs: None)
    prepared = branches.prepare_training_branches(None, _configs(workflow_config))
    payload = branches.branch_training_payload(prepared)
    if mutation == "unknown":
        payload["branches"][0]["promotion_policy"] = "automatic"
    elif mutation == "missing_spec":
        del payload["branches"][0]["spec"]["drop_missing_labels"]
    elif mutation == "unknown_cv":
        payload["branches"][0]["cv"]["unknown"] = True
    elif mutation == "source":
        payload["source_version"] = 8
    else:
        payload["format_version"] = True
    with pytest.raises(ValueError):
        branches.restore_training_branches(payload)


def test_saved_custom_steps_register_before_spec_validation(workflow_config, monkeypatch):
    """Fresh-process replay restores project identities while retaining saved filter params."""
    from skyulf.inference.project_code import load_project_module
    from skyulf.registry import NodeRegistry

    source = """from skyulf.inference.project_code import custom_step
class Filter:
    def fit(self, df, config):
        return df, config
class Apply:
    def apply(self, df, params):
        return df
def build_preprocessing():
    return []
def build_pre_split_steps():
    return [custom_step("custom_filter", Filter, Apply, {"cutoff": 0}, pre_split={
        "effect": "filter", "required_columns": ["x"], "learns_from_data": False})]
"""
    module = load_project_module(source)
    configs = _configs(workflow_config)
    configs["amount"]["pipeline"]["project_python_source"] = source
    configs["amount"]["pre_split_steps"] = module.build_pre_split_steps()
    configs["amount"]["pre_split_steps"][0]["params"]["cutoff"] = 2
    monkeypatch.setattr(branches, "controlled_champion_version", lambda *args, **kwargs: None)
    prepared = branches.prepare_training_branches(None, configs)
    payload = branches.branch_training_payload(prepared)
    identity = prepared[0].spec.pre_split_steps[0]["transformer"]
    monkeypatch.delitem(NodeRegistry._calculators, identity)
    monkeypatch.delitem(NodeRegistry._appliers, identity)
    restored = branches.restore_training_branches(payload)
    assert restored == prepared
    assert restored[0].spec.pre_split_steps[0]["params"] == {"cutoff": 2}
    assert NodeRegistry.get_calculator(identity) is module.Filter


def test_post_fit_comparison_failure_marks_parent_failed(
    workflow_config, monkeypatch, tmp_path, tracked
):
    """Failure after registration remains visible even though the candidate fit run finished."""
    store, client = tracked
    prepared = branches.prepare_training_branches(None, _configs(workflow_config, store=store))
    monkeypatch.setattr(
        training,
        "read_training_snapshot",
        lambda spark, spec: _data().loc[:, list(spec.source_columns)].copy(),
    )
    original = training.compare_candidate

    def compare(*args, **kwargs):
        """Inject an error after the second branch has produced an immutable model version."""
        if kwargs["model_name"].endswith("category"):
            raise RuntimeError("comparison unavailable")
        return original(*args, **kwargs)

    monkeypatch.setattr(training, "compare_candidate", compare)
    with pytest.raises(RuntimeError, match="comparison unavailable"):
        branches.train_local_branches(
            None,
            prepared,
            tracking_uri=store,
            registry_uri=store,
            experiment_name="branches",
            artifact_path=tmp_path / "fit",
        )
    runs = client.search_runs([client.get_experiment_by_name("branches").experiment_id])
    parent = next(run for run in runs if "mlflow.parentRunId" not in run.data.tags)
    assert parent.info.status == "FAILED"
    assert len(runs) == 3
    assert client.get_model_version("workspace.test.category", "1").status == "READY"
    progress = _artifact(client, parent.info.run_id, "branch_training_progress.json")
    assert progress["failed_branch"] == "category"
    assert set(progress["completed"]) == {"amount"}
    assert "branch_training_result.json" not in {
        item.path for item in client.list_artifacts(parent.info.run_id)
    }


@pytest.mark.parametrize("value", [False, 0, "", "   "])
def test_malformed_endpoints_fail_before_source_or_registry(workflow_config, monkeypatch, value):
    """Falsy endpoint values must not accidentally select the default Databricks service."""
    configs = _configs(workflow_config)
    configs["amount"]["tracking_uri"] = value
    champion = MagicMock()
    monkeypatch.setattr(branches, "controlled_champion_version", champion)
    with pytest.raises((ValueError, TypeError), match="tracking_uri"):
        branches.prepare_training_branches(None, configs)
    champion.assert_not_called()


def test_champions_are_resolved_independently_and_expected_pins_checked(
    workflow_config, monkeypatch
):
    """Each target compares against its own immutable champion, including explicit expectations."""
    versions = {"amount": "2", "category": "4", "ensemble": None}
    champion = MagicMock(
        side_effect=lambda model_name, **kwargs: versions[model_name.rsplit(".", 1)[1]]
    )
    monkeypatch.setattr(branches, "controlled_champion_version", champion)
    configs = _configs(workflow_config)
    configs["category"]["champion_version"] = "4"
    prepared = branches.prepare_training_branches(None, configs)
    assert {branch.name: branch.champion_version for branch in prepared} == versions
    assert champion.call_count == 3
    configs["category"]["champion_version"] = "3"
    with pytest.raises(ValueError, match="category champion_version"):
        branches.prepare_training_branches(None, configs)


def test_failure_logging_cannot_replace_original_error(
    workflow_config, monkeypatch, tmp_path, tracked
):
    """A tracking outage during cleanup must preserve the actionable training failure."""
    store, client = tracked
    prepared = branches.prepare_training_branches(None, _configs(workflow_config, store=store))
    original = branches.log_progress

    def progress(*args, **kwargs):
        """Emulate only the failed-progress logging outage."""
        if kwargs["status"] == "failed":
            raise OSError("tracking unavailable")
        return original(*args, **kwargs)

    def train(*args, **kwargs):
        """Emulate a training error whose message must survive tracking cleanup."""
        raise RuntimeError("original training failure")

    monkeypatch.setattr(branches, "log_progress", progress)
    monkeypatch.setattr(branches, "train_branch", train)
    with pytest.raises(RuntimeError, match="original training failure"):
        branches.train_local_branches(
            None,
            prepared,
            tracking_uri=store,
            registry_uri=store,
            experiment_name="branches",
            artifact_path=tmp_path / "fit",
        )
    runs = client.search_runs([client.get_experiment_by_name("branches").experiment_id])
    assert len(runs) == 1
    assert runs[0].info.status == "FAILED"
