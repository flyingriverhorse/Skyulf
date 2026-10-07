"""Guard native training lookup lineage before fitting and across job boundaries."""

import json
from dataclasses import replace
from types import SimpleNamespace
from unittest.mock import Mock

import pandas as pd
import pytest

from skyulf.integrations.databricks.feature_store import training
from skyulf.integrations.databricks.feature_store.lifecycle_config import (
    binding_json,
    serialize_feature_spec,
)
from skyulf.integrations.databricks.lifecycle.local_workflow import training_spec


@pytest.fixture
def bound_spec(workflow_config):
    """Pin one numeric feature while retaining record and event keys for splitting."""
    config = {
        **workflow_config,
        "engine": "pandas",
        "inference_mode": "spark",
        "feature_lookup": {
            "lookups": [
                {
                    "table_name": "workspace.features.history",
                    "lookup_key": ["entity"],
                    "feature_names": ["x"],
                }
            ]
        },
    }
    spec = training_spec(config)
    lookup = training.training_lookup(spec)
    assert lookup is not None
    binding = {
        "version": 1,
        "lookup_spec": serialize_feature_spec(lookup),
        "lookup_evidence": {
            "policy": "training_snapshot",
            "feature_tables": [
                {"table_name": "workspace.features.history", "table_id": "fixed", "version": 4}
            ],
        },
    }
    return replace(spec, feature_binding_json=binding_json(binding))


def test_materialization_uses_metadata_preserving_lookup(bound_spec, monkeypatch):
    """SDK exclusions must not remove keys, time or weights required by training."""
    helper = Mock(return_value=SimpleNamespace(load_df=Mock(return_value="enriched")))
    monkeypatch.setattr(training, "create_checked_training_set", helper)
    source = Mock(columns=[c for c in bound_spec.source_columns if c != "x"])
    result = training.enrich_training_source(Mock(), source, bound_spec)
    assert result == "enriched"
    assert helper.call_args.args[1].exclude_columns == ()
    source.select.assert_called_once_with(*[c for c in bound_spec.source_columns if c != "x"])


def test_materialization_rejects_feature_override_before_projection(bound_spec):
    """A stale feature in the source cannot silently override historical lookup values."""
    source = Mock(columns=[*bound_spec.source_columns, "X"])
    with pytest.raises(ValueError, match="override"):
        training.enrich_training_source(Mock(), source, bound_spec)
    source.select.assert_not_called()


def test_prepared_source_requires_exact_pinned_binding(bound_spec):
    """Saved training rows must belong to the prepared feature snapshot."""
    frame = pd.DataFrame({"x": [1.0]})
    with pytest.raises(ValueError, match="feature"):
        training.validate_training_frame(frame, bound_spec)
    frame.attrs["feature_lookup"] = json.loads(bound_spec.feature_binding_json)
    training.validate_training_frame(frame, bound_spec)
    frame.attrs["feature_lookup"]["lookup_evidence"]["feature_tables"][0]["version"] = 5
    with pytest.raises(ValueError, match="feature"):
        training.validate_training_frame(frame, bound_spec)
    assert frame.iloc[0, 0] == 1.0


def test_logging_reconstructs_excluded_set_from_pinned_source(bound_spec, monkeypatch):
    """Separate notebook tasks must log lineage without sending raw identifiers into the model."""
    helper = Mock(return_value="native-set")
    monkeypatch.setattr(training, "create_checked_training_set", helper)
    spark = Mock()
    source = spark.read.format.return_value.option.return_value.table.return_value
    source.columns = [c for c in bound_spec.source_columns if c != "x"]
    result = training.logging_training_set(spark, bound_spec)
    assert result == "native-set"
    spark.read.format.return_value.option.assert_called_once_with("versionAsOf", bound_spec.version)
    assert "entity" in helper.call_args.args[1].exclude_columns
    assert "x" not in helper.call_args.args[1].exclude_columns


def test_feature_frame_fits_real_pipeline_and_keeps_saved_lineage(
    bound_spec, monkeypatch, tmp_path
):
    """Prepared native features must reach actual fitting without losing their snapshot receipt."""
    from skyulf.integrations.databricks.training.fitting import local_retraining as fitting
    from skyulf.integrations.databricks.training.tuning.local_cv import LocalCVSpec

    frame = pd.DataFrame(
        {
            "id": range(6),
            "record_id": range(6),
            "entity": range(6),
            "x": [1.0, 2.0, 3.0, 4.0, 5.0, 6.0],
            "target": [2.0, 4.0, 6.0, 8.0, 10.0, 12.0],
            "event_time": pd.to_datetime(["2026-01-10"] * 4 + ["2026-02-10"] * 2, utc=True),
            "label_at": pd.to_datetime(["2026-02-11"] * 6, utc=True),
        }
    )
    frame.attrs["feature_lookup"] = json.loads(bound_spec.feature_binding_json)
    train, holdout, unavailable = fitting.split_labeled_snapshot(frame, bound_spec)
    monkeypatch.setattr(fitting, "pin_fit_lookup", lambda *args: bound_spec)
    config = {"preprocessing": [], "modeling": {"type": "linear_regression"}}
    fitted = fitting.fit_candidate(
        None,
        bound_spec,
        config,
        run=Mock(),
        pipeline_config=config,
        artifact_path=tmp_path / "model",
        engine="pandas",
        cv=LocalCVSpec(),
        risk_category=None,
        prepared_data=(frame, train, holdout, unavailable),
    )
    assert fitted.spec.feature_binding_json == bound_spec.feature_binding_json
    assert fitted.training_rows == 4 and fitted.holdout_rows == 2
    result = fitting.evaluate_local_holdout(fitted.artifact, fitted.holdout, target_column="target")
    assert result["heldout_rmse"] == pytest.approx(0.0, abs=1e-8)
