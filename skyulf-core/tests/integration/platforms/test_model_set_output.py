"""Output selection keeps one physical publication and explicit consumer projections."""

import pytest
from tests.integration.platforms.test_model_set_batch import _saved_set, _transport
from tests.integration.platforms.test_model_set_delta import _SOURCE, _rule


def test_combined_only_publishes_values_without_component_predictions(tmp_path, monkeypatch):
    """Intermediate predictions must be calculated but omitted from a combined-only write."""
    artifact, model, query = _saved_set(tmp_path, _SOURCE, _rule(10))
    batch, commit = _transport(monkeypatch, query)
    result = batch._run_admitted_set(
        None,
        model,
        artifact,
        "source",
        "target",
        "source",
        "target",
        "incremental_append",
        20,
        1024 * 1024,
        publication={"mode": "combined_only"},
    )
    frame = commit.call_args.args[1]
    assert frame.id.tolist() == [3, 1, 2]
    assert frame.adjusted_value.tolist() == pytest.approx([39, 28, 51])
    assert "amount__prediction" not in frame.columns
    assert "adjusted__scoring_status" in frame.columns
    assert result.output_count == 3


def test_combined_only_requires_saved_rules(tmp_path):
    """A result selection must not silently publish only keys when no rule exists."""
    from skyulf.integrations.databricks.model_sets.model_set_output import publication_columns

    artifact, _, _ = _saved_set(tmp_path)
    with pytest.raises(ValueError, match="combined.*rules"):
        publication_columns(artifact, {"mode": "combined_only"})


def test_named_views_select_components_and_combined_results(tmp_path):
    """Consumer names and columns must reference one shared physical prediction table."""
    from skyulf.integrations.databricks.model_sets.model_set_output import publication_views

    artifact, _, _ = _saved_set(tmp_path, _SOURCE, _rule(1))
    views = publication_views(
        artifact,
        "catalog.results.all_predictions",
        {
            "mode": "separate_views",
            "model_views": {"amount": "catalog.results.amount_estimates"},
            "combined_view": "catalog.results.business_results",
        },
    )
    assert [view.name for view in views] == [
        "catalog.results.amount_estimates",
        "catalog.results.business_results",
    ]
    assert "amount__prediction" in views[0].columns
    assert "adjusted_value" not in views[0].columns
    assert "adjusted_value" in views[1].columns
    assert "amount__prediction" not in views[1].columns
    assert all("id" in view.columns and "model_set_digest" in view.columns for view in views)


@pytest.mark.parametrize(
    "options",
    [
        {"model_views": {"unknown": "cat.db.other"}},
        {"model_views": {"amount": "cat.db.predictions"}},
        {"model_views": {"amount": "cat.db.same"}, "combined_view": "CAT.DB.SAME"},
        {"model_views": {"amount": "cat.db.bad;drop"}},
    ],
)
def test_invalid_view_names_fail_before_spark(tmp_path, options):
    """Unknown branches and ambiguous destinations must never reach catalog mutation."""
    from skyulf.integrations.databricks.model_sets.model_set_output import publication_views

    artifact, _, _ = _saved_set(tmp_path, _SOURCE, _rule(1))
    with pytest.raises(ValueError):
        publication_views(artifact, "cat.db.predictions", {"mode": "separate_views", **options})


@pytest.mark.parametrize(
    "policy",
    [
        {"mode": "typo"},
        {"mode": "all", "model_views": {}},
        {"mode": "all", "unknown": True},
        {"mode": "separate_views", "model_views": []},
    ],
)
def test_invalid_publication_policy_is_not_ignored(policy):
    """Misspelled output options must fail rather than accidentally persist extra values."""
    from skyulf.integrations.databricks.model_sets.model_set_output import publication_policy

    with pytest.raises(ValueError):
        publication_policy(policy)


def test_unrelated_existing_view_stops_setup_before_creating_any_views():
    """A later name conflict must not mutate earlier missing view names."""
    from types import SimpleNamespace
    from unittest.mock import Mock

    from skyulf.integrations.databricks.model_sets.model_set_output import (
        PredictionView,
        provision_publication_views,
    )

    spark = Mock()
    spark.catalog.tableExists.side_effect = lambda name: name == "db.unrelated"
    spark.catalog.getTable.return_value = SimpleNamespace(tableType="VIEW")
    spark.sql.return_value.first.return_value = {"value": "other-owner"}
    views = (PredictionView("db.new", ("id",)), PredictionView("db.unrelated", ("id",)))
    with pytest.raises(ValueError, match="another owner"):
        provision_publication_views(spark, "db.predictions", views)
    assert [call.args[0] for call in spark.sql.call_args_list] == [
        "SHOW TBLPROPERTIES `db`.`unrelated` ('prediction.projection')"
    ]


def test_separate_views_without_rules_still_exposes_each_model(tmp_path):
    """Multi-model scoring must remain usable when no cross-model rule is configured."""
    from skyulf.integrations.databricks.model_sets.model_set_output import publication_views

    artifact, _, _ = _saved_set(tmp_path)
    views = publication_views(artifact, "db.output", {"mode": "separate_views"})
    assert [view.name for view in views] == ["db.output_amount"]
    assert "amount__prediction" in views[0].columns
