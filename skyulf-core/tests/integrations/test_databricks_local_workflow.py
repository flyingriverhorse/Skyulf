"""Reusable Databricks workflow services preserve selection and publication behavior."""

from datetime import UTC, datetime, timedelta, timezone
from types import SimpleNamespace
from unittest.mock import Mock

import pytest


def _workflow():
    """Use the public library that generated notebooks delegate to."""
    from skyulf.integrations.databricks import local_workflow

    return local_workflow


def _output():
    """Inspect output publication independently of notebook widgets."""
    from skyulf.integrations.databricks import prediction_output

    return prediction_output


def _config():
    """Keep a concrete, bounded source and model selection in each test."""
    return {
        "engine": "polars",
        "training_table": "workspace.test.training",
        "training_version": 0,
        "score_source_table": "workspace.test.source",
        "prediction_table": "workspace.test.predictions",
        "model_name": "workspace.test.model",
        "model_version": "1",
        "model_change_mode": "incremental_append",
        "champion_version": "1",
        "row_keys": ["entity_id"],
        "input_columns": ["x"],
        "target_column": "target",
        "event_column": "event_time",
        "label_time_column": "label_at",
        "start": "2026-01-01T00:00:00+00:00",
        "holdout_start": "2026-02-01T00:00:00+00:00",
        "cutoff": "2026-03-01T00:00:00+00:00",
        "max_rows": 100,
        "max_bytes": 100_000,
        "metric": "heldout_rmse",
        "min_improvement": 0.0,
        "quality_threshold": None,
        "pipeline": {"preprocessing": [], "modeling": {"type": "linear_regression"}},
    }


def _one_column_schema(kind):
    """Model the Spark field names/types used by view compatibility checks."""
    return SimpleNamespace(
        fields=[SimpleNamespace(name="prediction", dataType=SimpleNamespace(typeName=lambda: kind))]
    )


def test_train_preserves_selected_polars_engine_and_never_promotes(monkeypatch, tmp_path):
    """A training job must only produce a candidate with the chosen fit engine."""
    workflow = _workflow()
    train = Mock(return_value=SimpleNamespace(model_version="2"))
    monkeypatch.setattr(workflow, "train_local_candidate", train)
    monkeypatch.setattr(workflow, "_automatic_promotion", Mock(return_value=None))
    monkeypatch.setattr(workflow, "resolve_model", Mock(return_value=SimpleNamespace(version="1")))
    result = workflow.run_action(
        object(),
        _config(),
        "train",
        experiment_name="/Users/test/experiment",
        artifact_path=tmp_path / "artifact",
    )
    assert result.model_version == "2"
    assert train.call_args.kwargs["engine"] == "polars"
    assert train.call_args.args[1].version == 0
    assert callable(train.call_args.kwargs["on_registered"])


def test_monthly_window_pins_current_delta_version_across_year_boundary():
    """A delayed or timezone-shifted run must pin one reproducible monthly window."""
    workflow = _workflow()
    history = Mock()
    history.select.return_value.orderBy.return_value.first.return_value = {"version": 7}
    spark = Mock()
    spark.sql.return_value = history
    config = _config()
    config["monthly_lookback_months"] = 4
    spec = workflow._monthly_training_spec(
        spark, config, datetime(2027, 1, 3, 5, tzinfo=timezone(timedelta(hours=2)))
    )
    assert (spec.version, spec.start, spec.holdout_start, spec.cutoff) == (
        7,
        datetime(2026, 9, 1, tzinfo=UTC),
        datetime(2026, 12, 1, tzinfo=UTC),
        datetime(2027, 1, 1, tzinfo=UTC),
    )
    spark.sql.assert_called_once_with("DESCRIBE HISTORY workspace.test.training")


def test_monthly_training_rejects_missing_version_and_invalid_lookback():
    """A scheduled run must fail before fitting an unpinned or empty window."""
    workflow = _workflow()
    spark = Mock()
    spark.sql.return_value.select.return_value.orderBy.return_value.first.return_value = None
    config = _config()
    config["monthly_lookback_months"] = 1
    with pytest.raises(ValueError, match="monthly_lookback_months"):
        workflow._monthly_training_spec(spark, config, datetime(2027, 1, 3, tzinfo=UTC))
    config["monthly_lookback_months"] = 3
    with pytest.raises(ValueError, match="version"):
        workflow._monthly_training_spec(spark, config, datetime(2027, 1, 3, tzinfo=UTC))


def test_monthly_train_compares_pinned_champion_without_activation(monkeypatch, tmp_path):
    """Monthly nomination must not activate a model or change the scorer's pin."""
    workflow = _workflow()
    spec = workflow._training_spec(_config())
    train = Mock(return_value=SimpleNamespace(model_version="3"))
    champion = Mock(return_value=SimpleNamespace(version="2"))
    monkeypatch.setattr(workflow, "_monthly_training_spec", lambda *args: spec)
    monkeypatch.setattr(workflow, "resolve_model", champion)
    monkeypatch.setattr(workflow, "train_local_candidate", train)
    monkeypatch.setattr(workflow, "_automatic_promotion", Mock(return_value=None))
    config = _config()
    config["champion_version"] = None
    original_version = config["model_version"]
    workflow.run_action(
        object(),
        config,
        "train_monthly",
        experiment_name="/Users/test/experiment",
        artifact_path=tmp_path / "artifact",
        now=datetime(2027, 1, 3, tzinfo=UTC),
    )
    assert train.call_args.kwargs["champion_version"] == "2"
    assert config["model_version"] == original_version
    champion.assert_called_once()


def test_monthly_train_allows_first_model_but_propagates_registry_errors(monkeypatch):
    """Only a missing champion alias is a valid first-training condition."""
    workflow = _workflow()
    from skyulf.integrations.mlflow.registry import RegistryModelNotFoundError

    def missing(*args, **kwargs):
        """Represent a registry without a champion alias."""
        raise RegistryModelNotFoundError("alias missing")

    monkeypatch.setattr(workflow, "resolve_model", missing)
    assert workflow._monthly_champion_version(_config()) is None
    monkeypatch.setattr(
        workflow, "resolve_model", Mock(side_effect=RuntimeError("permission denied"))
    )
    with pytest.raises(RuntimeError, match="permission denied"):
        workflow._monthly_champion_version(_config())


def test_auto_champion_train_promotes_only_an_eligible_candidate(monkeypatch, tmp_path):
    """A passing candidate must use the pinned holdout and controlled alias APIs."""
    workflow = _workflow()
    config = _config()
    config.update(model_selection_mode="auto_champion", quality_threshold=1.0, engine="pandas")
    report = SimpleNamespace(champion_version="1", candidate_version="2", eligible=True)
    candidate = SimpleNamespace(comparison=report)
    frame = object()
    heldout = object()
    monkeypatch.setattr(workflow, "train_local_candidate", Mock(return_value=candidate))
    monkeypatch.setattr(workflow, "controlled_champion_version", Mock(return_value="1"))
    monkeypatch.setattr(workflow, "read_training_snapshot", Mock(return_value=frame), raising=False)
    monkeypatch.setattr(
        workflow,
        "split_labeled_snapshot",
        Mock(return_value=(object(), heldout, 0)),
        raising=False,
    )
    stage = Mock(return_value=object())
    promote = Mock(return_value=SimpleNamespace(new_version="2"))
    monkeypatch.setattr(workflow, "stage_challenger", stage, raising=False)
    monkeypatch.setattr(workflow, "promote_candidate", promote, raising=False)
    result = workflow.run_action(
        object(),
        config,
        "train",
        experiment_name="/Users/test/experiment",
        artifact_path=tmp_path / "artifact",
    )
    assert result.alias_change.new_version == "2"
    assert stage.call_args.args[1] is heldout
    assert promote.call_args.kwargs["expected_champion_version"] == "1"


def test_auto_champion_train_retains_challenger_when_candidate_fails_gate(monkeypatch, tmp_path):
    """An unsuccessful challenger stays visible without being promoted."""
    workflow = _workflow()
    config = _config()
    config.update(model_selection_mode="auto_champion", quality_threshold=1.0, engine="pandas")
    monkeypatch.setattr(
        workflow,
        "train_local_candidate",
        Mock(
            return_value=SimpleNamespace(
                comparison=SimpleNamespace(
                    champion_version="1", candidate_version="2", eligible=False
                )
            )
        ),
    )
    monkeypatch.setattr(workflow, "controlled_champion_version", Mock(return_value="1"))
    stage = Mock()
    monkeypatch.setattr(workflow, "stage_challenger", stage, raising=False)
    monkeypatch.setattr(workflow, "read_training_snapshot", Mock(return_value=object()))
    monkeypatch.setattr(workflow, "split_labeled_snapshot", Mock(return_value=(None, object(), 0)))
    promote = Mock()
    monkeypatch.setattr(workflow, "promote_candidate", promote)
    result = workflow.run_action(
        object(),
        config,
        "train",
        experiment_name="/Users/test/experiment",
        artifact_path=tmp_path / "artifact",
    )
    assert result.alias_change is None
    stage.assert_called_once()
    assert stage.call_args.kwargs["expected_challenger_version"] == "2"
    promote.assert_not_called()


def test_auto_champion_bootstraps_first_model_or_rejects_missing_threshold(monkeypatch, tmp_path):
    """First-model selection requires a real absolute gate before fitting."""
    workflow = _workflow()
    config = _config()
    config.update(model_selection_mode="auto_champion", quality_threshold=None, engine="pandas")
    train = Mock(
        return_value=SimpleNamespace(
            comparison=SimpleNamespace(
                champion_version=None,
                candidate_version="1",
                candidate_metrics={"heldout_rmse": 0.1},
                metric="heldout_rmse",
                metric_direction="minimize",
                quality_threshold=1.0,
            )
        )
    )
    monkeypatch.setattr(workflow, "train_local_candidate", train)
    with pytest.raises(ValueError, match="quality_threshold"):
        workflow.run_action(
            object(),
            config,
            "train",
            experiment_name="/Users/test/experiment",
            artifact_path=tmp_path / "artifact",
        )
    train.assert_not_called()
    config["quality_threshold"] = 1.0
    monkeypatch.setattr(workflow, "controlled_champion_version", Mock(return_value=None))
    monkeypatch.setattr(workflow, "read_training_snapshot", Mock(return_value=object()))
    monkeypatch.setattr(
        workflow,
        "split_labeled_snapshot",
        Mock(return_value=(object(), object(), 0)),
    )
    initialize = Mock(return_value=SimpleNamespace(new_version="1"))
    monkeypatch.setattr(workflow, "initialize_champion", initialize)
    monkeypatch.setattr(workflow, "stage_challenger", Mock())
    result = workflow.run_action(
        object(),
        config,
        "train",
        experiment_name="/Users/test/experiment",
        artifact_path=tmp_path / "artifact",
    )
    assert result.alias_change.new_version == "1"
    initialize.assert_called_once()


def test_bundle_records_comparison_error_without_promoting(monkeypatch, tmp_path):
    """A registered contender must keep an inspectable error when evaluation fails."""
    workflow = _workflow()
    monkeypatch.setattr(workflow, "resolve_model", Mock(return_value=SimpleNamespace(version="1")))
    lifecycle = Mock()
    monkeypatch.setattr(
        workflow, "ChallengerLifecycle", Mock(return_value=lifecycle), raising=False
    )

    def fail_after_registration(*args, **kwargs):
        """Expose a version before reproducing the comparison failure."""
        kwargs["on_registered"](object())
        raise RuntimeError("comparison failed")

    monkeypatch.setattr(workflow, "train_local_candidate", fail_after_registration)
    with pytest.raises(RuntimeError, match="comparison failed"):
        workflow.run_action(
            object(),
            _config(),
            "train",
            experiment_name="test",
            artifact_path=tmp_path / "artifact",
        )
    lifecycle.registered.assert_called_once()
    lifecycle.failed.assert_called_once()


def test_bundle_records_error_during_final_comparison(monkeypatch, tmp_path):
    """The second holdout read and comparison must not leave a contender pending forever."""
    workflow = _workflow()
    lifecycle = Mock()
    monkeypatch.setattr(workflow, "ChallengerLifecycle", Mock(return_value=lifecycle))
    monkeypatch.setattr(workflow, "resolve_model", Mock(return_value=SimpleNamespace(version="1")))
    monkeypatch.setattr(workflow, "train_local_candidate", Mock(return_value=object()))
    monkeypatch.setattr(
        workflow, "_automatic_promotion", Mock(side_effect=RuntimeError("recheck failed"))
    )
    with pytest.raises(RuntimeError, match="recheck failed"):
        workflow.run_action(
            object(),
            _config(),
            "train",
            experiment_name="test",
            artifact_path=tmp_path / "artifact",
        )
    lifecycle.failed.assert_called_once()


def test_pinned_training_rejects_stale_comparison_version_before_fit(monkeypatch, tmp_path):
    """A historical baseline must not produce a candidate before a known alias conflict."""
    workflow = _workflow()
    monkeypatch.setattr(workflow, "resolve_model", Mock(return_value=SimpleNamespace(version="2")))
    train = Mock()
    monkeypatch.setattr(workflow, "train_local_candidate", train)
    with pytest.raises(ValueError, match="champion_version"):
        workflow.run_action(
            object(),
            _config(),
            "train",
            experiment_name="test",
            artifact_path=tmp_path / "artifact",
        )
    train.assert_not_called()


def test_auto_champion_score_pins_resolved_version_before_target_selection(monkeypatch):
    """A moving alias must become one concrete model and full-rebuild table per run."""
    workflow = _workflow()
    config = _config()
    config.update(model_selection_mode="auto_champion", model_change_mode="full_rebuild")
    controlled = Mock(return_value="2")
    prepare = Mock(return_value=object())
    provision = Mock()
    monkeypatch.setattr(workflow, "controlled_champion_version", controlled)
    monkeypatch.setattr(workflow, "prepare_local_workflow", prepare)
    monkeypatch.setattr(workflow, "provision_prediction_table", provision)
    monkeypatch.setattr(workflow, "run_incremental_local_batch", Mock(return_value=object()))
    monkeypatch.setattr(workflow, "_activate_prediction_view", Mock())
    spark = Mock()
    spark.catalog.tableExists.return_value = False
    workflow.run_action(spark, config, "score")
    assert controlled.call_args.args == (config["model_name"],)
    assert prepare.call_args.args[0].model.version == "2"
    assert provision.call_args.args[1]["prediction_table"] == "workspace.test.predictions_v2"
    assert config["model_version"] == "1"


def test_score_uses_incremental_service_without_period_or_source_version(monkeypatch):
    """Recurring scoring must derive new inserts from committed Delta receipts."""
    workflow = _workflow()
    prepare = Mock(return_value=object())
    score = Mock(return_value=SimpleNamespace(noop=False, input_count=2))
    monkeypatch.setattr(workflow, "prepare_local_workflow", prepare)
    monkeypatch.setattr(workflow, "run_incremental_local_batch", score)
    monkeypatch.setattr(workflow, "provision_prediction_table", Mock())
    result = workflow.run_action(object(), _config(), "score")
    assert result.input_count == 2
    assert score.call_args.kwargs["row_keys"] == ("entity_id",)
    assert "period_start" not in score.call_args.kwargs
    assert "source_version" not in score.call_args.kwargs
    assert prepare.call_args.args[0].engine == "polars"
    assert isinstance(score.call_args.kwargs["admission"], workflow.SingleWriterAdmission)


def test_full_rebuild_scores_new_generation_before_view_activation(monkeypatch):
    """A v2 rebuild must fill a new table before changing the logical output view."""
    workflow = _workflow()
    config = _config()
    config.update(model_version="2", model_change_mode="full_rebuild")
    events = []
    prepared = Mock()
    provision = Mock(side_effect=lambda *args: events.append("provision"))
    monkeypatch.setattr(
        workflow,
        "prepare_local_workflow",
        Mock(side_effect=lambda cfg: events.append("prepare") or prepared),
    )
    monkeypatch.setattr(
        workflow,
        "provision_prediction_table",
        provision,
    )
    monkeypatch.setattr(
        workflow,
        "run_incremental_local_batch",
        Mock(side_effect=lambda *args, **kwargs: events.append("score") or object()),
    )
    activate = Mock(side_effect=lambda *args: events.append("activate"))
    monkeypatch.setattr(workflow, "_activate_prediction_view", activate)
    spark = Mock()
    spark.catalog.tableExists.return_value = False
    workflow.run_action(spark, config, "score")
    assert events == ["prepare", "provision", "score", "activate"]
    assert provision.call_args.args[1]["prediction_table"] == "workspace.test.predictions_v2"
    assert activate.call_args.args[1:] == (
        "workspace.test.predictions",
        "workspace.test.predictions_v2",
    )


def test_full_rebuild_does_not_activate_a_failed_generation(monkeypatch):
    """A failed full score must leave the active prediction view untouched."""
    workflow = _workflow()
    config = _config()
    config["model_change_mode"] = "full_rebuild"
    monkeypatch.setattr(workflow, "prepare_local_workflow", Mock(return_value=object()))
    monkeypatch.setattr(workflow, "provision_prediction_table", Mock())
    monkeypatch.setattr(
        workflow, "run_incremental_local_batch", Mock(side_effect=RuntimeError("score failed"))
    )
    activate = Mock()
    monkeypatch.setattr(workflow, "_activate_prediction_view", activate)
    spark = Mock()
    spark.catalog.tableExists.return_value = False
    with pytest.raises(RuntimeError, match="score failed"):
        workflow.run_action(spark, config, "score")
    activate.assert_not_called()


def test_full_rebuild_rejects_existing_table_before_creating_generation(monkeypatch):
    """A mode switch must fail before creating an orphan physical generation."""
    workflow = _workflow()
    config = _config()
    config["model_change_mode"] = "full_rebuild"
    spark = Mock()
    spark.catalog.tableExists.return_value = True
    spark.catalog.getTable.return_value.tableType = "MANAGED"
    prepare = Mock()
    provision = Mock()
    monkeypatch.setattr(workflow, "prepare_local_workflow", prepare)
    monkeypatch.setattr(workflow, "provision_prediction_table", provision)
    with pytest.raises(ValueError, match="view name"):
        workflow.run_action(spark, config, "score")
    prepare.assert_not_called()
    provision.assert_not_called()


@pytest.mark.parametrize(
    "mode,version", [("unknown", "2"), ("full_rebuild", "@champion"), ("full_rebuild", "02")]
)
def test_scoring_rejects_unsafe_model_change_selection(mode, version):
    """An invalid mode or moving version must fail before a score target is chosen."""
    workflow = _workflow()
    config = _config()
    config.update(model_change_mode=mode, model_version=version)
    with pytest.raises(ValueError):
        workflow._scoring_target(config)


def test_full_rebuild_refuses_existing_table_at_logical_view_name():
    """Switching modes must never replace an existing prediction Delta table."""
    workflow = _workflow()
    spark = Mock()
    spark.catalog.tableExists.return_value = True
    spark.catalog.getTable.return_value.tableType = "MANAGED"
    spark.table.return_value.limit.return_value.count.return_value = 2
    with pytest.raises(ValueError, match="view name"):
        workflow._activate_prediction_view(
            spark, "workspace.test.predictions", "workspace.test.predictions_v2"
        )
    spark.sql.assert_not_called()


def test_full_rebuild_refuses_foreign_view_and_incompatible_schema():
    """Activation must neither hijack a foreign view nor break its output schema."""
    workflow = _workflow()
    spark = Mock()
    spark.catalog.tableExists.return_value = True
    spark.catalog.getTable.return_value.tableType = "VIEW"
    spark.table.return_value.limit.return_value.count.return_value = 2
    spark.sql.return_value.first.return_value = {"value": "other"}
    with pytest.raises(ValueError, match="Skyulf"):
        workflow._activate_prediction_view(
            spark, "workspace.test.predictions", "workspace.test.predictions_v2"
        )
    spark.sql.reset_mock()
    spark.sql.return_value.first.return_value = {"value": "full_rebuild"}
    spark.table.side_effect = [
        SimpleNamespace(
            limit=Mock(return_value=SimpleNamespace(count=lambda: 2)),
            schema=_one_column_schema("double"),
        ),
        SimpleNamespace(schema=_one_column_schema("string")),
    ]
    with pytest.raises(ValueError, match="schema"):
        workflow._activate_prediction_view(
            spark, "workspace.test.predictions", "workspace.test.predictions_v2"
        )
    assert all(not call.args[0].startswith("ALTER VIEW") for call in spark.sql.call_args_list)


def test_full_rebuild_creates_owned_view_then_preserves_grants_on_switch():
    """A validated generation becomes active through create then ALTER VIEW."""
    workflow = _workflow()
    spark = Mock()
    spark.table.return_value.limit.return_value.count.return_value = 2
    spark.table.return_value.schema = _one_column_schema("double")
    spark.catalog.tableExists.side_effect = [False, True]
    spark.catalog.getTable.return_value.tableType = "VIEW"
    spark.sql.return_value.first.side_effect = [
        {"value": "full_rebuild"},
        {
            "createtab_stmt": "CREATE VIEW workspace.test.predictions AS SELECT * FROM workspace.test.predictions_v1"
        },
    ]
    workflow._activate_prediction_view(
        spark, "workspace.test.predictions", "workspace.test.predictions_v1"
    )
    workflow._activate_prediction_view(
        spark, "workspace.test.predictions", "workspace.test.predictions_v2"
    )
    statements = [call.args[0] for call in spark.sql.call_args_list]
    assert any(statement.startswith("CREATE VIEW") for statement in statements)
    assert any(statement.startswith("ALTER VIEW") for statement in statements)


def test_lifecycle_actions_are_not_bundle_entry_points():
    """The minimal Bundle must not accidentally mutate registry aliases."""
    workflow = _workflow()
    for action in ("compare", "stage", "promote", "setup"):
        with pytest.raises(ValueError, match="Unknown workflow action"):
            workflow.run_action(object(), _config(), action)


def test_target_binding_separates_test_and_prod_outputs():
    """One generated workflow must never reuse a test prediction table in prod."""
    workflow = _workflow()
    config = _config()
    config.update(
        training_table="{catalog}.{input_schema}.labeled_events",
        score_source_table="shared.raw.events",
        prediction_table="{catalog}.{output_schema}.predictions{resource_suffix}",
        model_name="{catalog}.{metadata_schema}.model{resource_suffix}",
    )
    shared = {
        "input_schema": "refined",
        "output_schema": "mlresult",
        "metadata_schema": "metadata",
    }
    test_config = workflow.resolve_target_config(
        config, {**shared, "catalog": "test_catalog", "resource_suffix": "_murat"}
    )
    syst_config = workflow.resolve_target_config(
        config, {**shared, "catalog": "syst_catalog", "resource_suffix": ""}
    )
    prod_config = workflow.resolve_target_config(
        config, {**shared, "catalog": "prod_catalog", "resource_suffix": ""}
    )
    assert test_config["prediction_table"] == "test_catalog.mlresult.predictions_murat"
    assert syst_config["prediction_table"] == "syst_catalog.mlresult.predictions"
    assert prod_config["prediction_table"] == "prod_catalog.mlresult.predictions"
    assert prod_config["score_source_table"] == "shared.raw.events"
    assert test_config["model_name"] != prod_config["model_name"]
    assert (
        len(
            {
                test_config["prediction_table"],
                syst_config["prediction_table"],
                prod_config["prediction_table"],
            }
        )
        == 3
    )
    assert config["prediction_table"] == "{catalog}.{output_schema}.predictions{resource_suffix}"


def test_target_binding_rejects_unsafe_catalog_before_any_job():
    """A malformed target parameter must fail before it reaches a UC operation."""
    workflow = _workflow()
    with pytest.raises(ValueError, match="catalog"):
        workflow.resolve_target_config(
            _config(),
            {
                "catalog": "prod;DROP TABLE x",
                "input_schema": "refined",
                "output_schema": "mlresult",
                "metadata_schema": "metadata",
                "resource_suffix": "",
            },
        )


def test_target_binding_rejects_cross_environment_output():
    """A prod job cannot write a model or prediction into a test catalog."""
    workflow = _workflow()
    config = _config()
    config.update(
        model_name="prod_catalog.metadata.model",
    )
    config["prediction_table"] = "test_catalog.mlresult.predictions"
    with pytest.raises(ValueError, match="prediction_table"):
        workflow.resolve_target_config(
            config,
            {
                "catalog": "prod_catalog",
                "input_schema": "refined",
                "output_schema": "mlresult",
                "metadata_schema": "metadata",
                "resource_suffix": "",
            },
        )


def test_target_binding_rejects_cross_environment_model():
    """A prod train job cannot register a model in a test catalog."""
    workflow = _workflow()
    config = _config()
    config.update(
        prediction_table="prod_catalog.mlresult.predictions",
    )
    config["model_name"] = "test_catalog.metadata.model"
    with pytest.raises(ValueError, match="model_name"):
        workflow.resolve_target_config(
            config,
            {
                "catalog": "prod_catalog",
                "input_schema": "refined",
                "output_schema": "mlresult",
                "metadata_schema": "metadata",
                "resource_suffix": "",
            },
        )


def test_score_provision_rejects_missing_source_without_creating_tables():
    """A wrong source reference must not create partial prediction resources."""
    workflow = _output()
    spark = Mock()
    spark.catalog.tableExists.return_value = False
    with pytest.raises(ValueError, match="source"):
        workflow.provision_prediction_table(spark, _config(), object())
    spark.sql.assert_not_called()


def test_score_does_not_require_labeled_training_table(monkeypatch):
    """A retired training snapshot must not prevent scoring a live source."""
    workflow = _output()
    config = _config()
    config["training_table"] = "workspace.test.retired_training"
    spark = Mock()
    spark.catalog.tableExists.side_effect = lambda name: name != config["training_table"]
    spark.sql.return_value.first.return_value = {
        "properties": {"delta.enableChangeDataFeed": "true"}
    }
    monkeypatch.setattr(workflow, "_prediction_columns", Mock(return_value=()))
    monkeypatch.setattr(workflow, "_check_existing_table", Mock())
    prepared = SimpleNamespace(preflight=SimpleNamespace(ready=True))
    assert workflow.provision_prediction_table(spark, config, prepared) is False


def test_score_replay_validates_existing_output_without_recounting_source(monkeypatch):
    """A later score must accept a growing source without recreating its output."""
    workflow = _output()
    config = _config()
    source = Mock()
    source.columns = ["entity_id", "x"]
    source.schema = {
        "entity_id": SimpleNamespace(dataType=SimpleNamespace(typeName=lambda: "string"))
    }
    spark = Mock()
    spark.catalog.tableExists.return_value = True
    spark.table.return_value = source
    spark.sql.return_value.first.return_value = {
        "properties": {"delta.enableChangeDataFeed": "true"}
    }
    check_target = Mock()
    monkeypatch.setattr(workflow, "_check_existing_table", check_target)
    prepared = SimpleNamespace(
        artifact=SimpleNamespace(manifest=SimpleNamespace(input_columns=("x",))),
        preflight=SimpleNamespace(
            ready=True,
            output_schema=(SimpleNamespace(name="prediction", dtype="float64"),),
        ),
    )
    result = workflow.provision_prediction_table(spark, config, prepared)
    check_target.assert_called_once()
    source.select.assert_not_called()
    assert result is False
    assert all(not call.args[0].startswith("CREATE TABLE") for call in spark.sql.call_args_list)


def test_first_score_provisions_prediction_before_incremental_write(monkeypatch):
    """The scheduled score job must handle first-run setup without another job."""
    workflow = _workflow()
    events = []
    prepared = object()
    monkeypatch.setattr(workflow, "prepare_local_workflow", Mock(return_value=prepared))
    monkeypatch.setattr(
        workflow,
        "provision_prediction_table",
        Mock(side_effect=lambda spark, config, model: events.append("provision")),
        raising=False,
    )
    monkeypatch.setattr(
        workflow,
        "run_incremental_local_batch",
        Mock(side_effect=lambda *args, **kwargs: events.append("score")),
    )
    workflow.run_action(object(), _config(), "score")
    assert events == ["provision", "score"]


def test_prediction_provision_creates_no_control_table(monkeypatch):
    """First scoring must create only the output table, with no lock table."""
    workflow = _output()
    config = _config()
    source = Mock()
    source.columns = ["entity_id", "x"]
    source.schema = {
        "entity_id": SimpleNamespace(dataType=SimpleNamespace(typeName=lambda: "string"))
    }
    source.select.return_value.limit.return_value.count.return_value = 2
    spark = Mock()
    spark.catalog.tableExists.side_effect = lambda name: (
        name
        in {
            config["training_table"],
            config["score_source_table"],
        }
    )
    spark.table.return_value = source
    spark.sql.return_value.first.return_value = {
        "properties": {"delta.enableChangeDataFeed": "true"}
    }
    prepared = SimpleNamespace(
        artifact=SimpleNamespace(manifest=SimpleNamespace(input_columns=("x",))),
        preflight=SimpleNamespace(
            ready=True,
            output_schema=(SimpleNamespace(name="prediction", dtype="float64"),),
        ),
    )
    created = workflow.provision_prediction_table(spark, config, prepared)
    statements = [call.args[0] for call in spark.sql.call_args_list]
    assert created is True
    assert len([statement for statement in statements if statement.startswith("CREATE TABLE")]) == 1
    assert all("admission" not in statement for statement in statements)


def test_full_rebuild_generation_records_model_identity(monkeypatch):
    """A new generation must carry enough provenance to reject unrelated tables."""
    workflow = _output()
    config = _config()
    config.update(
        model_change_mode="full_rebuild", prediction_table="workspace.test.predictions_v2"
    )
    spark = Mock()
    spark.catalog.tableExists.side_effect = lambda name: name == config["score_source_table"]
    spark.sql.return_value.first.return_value = {
        "properties": {"delta.enableChangeDataFeed": "true"}
    }
    spark.table.return_value.select.return_value.limit.return_value.count.return_value = 2
    monkeypatch.setattr(
        workflow, "_prediction_columns", Mock(return_value=(("prediction", "double", "DOUBLE"),))
    )
    prepared = SimpleNamespace(preflight=SimpleNamespace(ready=True, model_digest="a" * 64))
    assert workflow.provision_prediction_table(spark, config, prepared) is True
    created = next(
        call.args[0] for call in spark.sql.call_args_list if call.args[0].startswith("CREATE TABLE")
    )
    assert "'skyulf.mode' = 'full_rebuild'" in created
    assert "'skyulf.model_name' = 'workspace.test.model'" in created
    assert "'skyulf.model_version' = '1'" in created
    assert f"'skyulf.model_digest' = '{'a' * 64}'" in created


@pytest.mark.parametrize("changed", ["mode", "name", "digest"])
def test_full_rebuild_rejects_unrelated_or_changed_generation(monkeypatch, changed):
    """A same-named table must not be scored or activated under another model."""
    workflow = _output()
    config = _config()
    config.update(
        model_change_mode="full_rebuild", prediction_table="workspace.test.predictions_v2"
    )
    expected = {
        "skyulf.mode": "full_rebuild",
        "skyulf.model_name": config["model_name"],
        "skyulf.model_version": config["model_version"],
        "skyulf.model_digest": "a" * 64,
    }
    key = {"mode": "skyulf.mode", "name": "skyulf.model_name", "digest": "skyulf.model_digest"}[
        changed
    ]
    expected[key] = "other"
    spark = Mock()
    spark.catalog.tableExists.return_value = True
    spark.sql.return_value.first.side_effect = [
        {"properties": {"delta.enableChangeDataFeed": "true"}},
        {"properties": expected},
    ]
    monkeypatch.setattr(workflow, "_prediction_columns", Mock(return_value=()))
    check_target = Mock()
    monkeypatch.setattr(workflow, "_check_existing_table", check_target)
    prepared = SimpleNamespace(preflight=SimpleNamespace(ready=True, model_digest="a" * 64))
    with pytest.raises(ValueError, match="another model or workflow"):
        workflow.provision_prediction_table(spark, config, prepared)
    check_target.assert_not_called()
