"""Shared evidence and result owners preserve the saved lifecycle contracts."""

import hashlib
import json
from copy import deepcopy
from dataclasses import asdict, replace
from datetime import datetime
from unittest.mock import Mock

import pandas as pd
import pytest

from skyulf.integrations.databricks.data.training.training_dates import TrainingDateSpec
from skyulf.integrations.databricks.lifecycle import local_workflow as workflow
from skyulf.integrations.databricks.training.fitting import local_retraining as training
from skyulf.integrations.databricks.training.shared import local_training_evidence as evidence
from skyulf.integrations.mlflow.lifecycle.promotion import AliasChangeReceipt
from skyulf.integrations.mlflow.lifecycle.validation import (
    ModelComparisonReport,
    comparison_payload,
)


def _spec(temporal=False):
    """Build concrete random or offset-aware temporal evidence without external reads."""
    dates = (
        {
            "split_strategy": "temporal",
            "event_column": "event_at",
            "start": datetime.fromisoformat("2026-01-01T00:00:00+03:00"),
            "holdout_start": datetime.fromisoformat("2026-02-01T00:00:00+03:00"),
            "cutoff": datetime.fromisoformat("2026-03-01T00:00:00+03:00"),
            "event_time_parsing": TrainingDateSpec(format="%d/%m/%Y %H:%M", timezone="Asia/Tokyo"),
        }
        if temporal
        else {}
    )
    return training.LocalTrainingSpec(
        table="workspace.test.source",
        version=4,
        record_key_columns=("id",),
        input_columns=("x",),
        target_column="target",
        max_rows=100,
        max_bytes=1024,
        holdout_key_sha256="a" * 64,
        **dates,
    )


@pytest.mark.parametrize("temporal", [False, True])
@pytest.mark.parametrize("engine", ["pandas", "polars"])
def test_saved_spec_roundtrip_preserves_identity_and_input(temporal, engine):
    """Both replay adapters need identical dates, tuples and defaults from immutable JSON."""
    spec = _spec(temporal)
    payload = json.loads(json.dumps(training.training_spec_payload(spec, engine)))
    payload.pop("pre_split_steps")  # Existing approval evidence permits this absent field.
    original = deepcopy(payload)
    restored = training.LocalTrainingSpec.from_payload(payload)
    assert restored == spec and restored.dataset_id == spec.dataset_id
    assert payload == original and isinstance(restored.record_key_columns, tuple)


def _report(spec):
    """Keep first-candidate null champion metrics in a complete comparison receipt."""
    return ModelComparisonReport(
        dataset_id=spec.dataset_id,
        row_count=2,
        code_version="0.9.0",
        model_name="workspace.test.model",
        candidate_version="1",
        candidate_digest="b" * 64,
        champion_version=None,
        champion_digest=None,
        metric="heldout_rmse",
        metric_direction="minimize",
        min_improvement=0.0,
        quality_threshold=2.0,
        candidate_metrics={"heldout_rmse": 1.0},
        champion_metrics=None,
        improvement=None,
        eligible=False,
        reason="no_champion",
    )


@pytest.mark.parametrize("change", ["none", "digest", "model", "version", "dataset", "holdout"])
def test_shared_loader_verifies_named_candidate_evidence(tmp_path, change):
    """A valid digest alone cannot authorize foreign models, snapshots or unproved holdouts."""
    spec = _spec()
    report = _report(spec)
    if change == "model":
        report = replace(report, model_name="workspace.test.foreign")
    elif change == "version":
        report = replace(report, candidate_version="2")
    elif change == "dataset":
        report = replace(report, dataset_id="foreign")
    elif change == "holdout":
        spec = replace(spec, holdout_key_sha256=None)
    digest = hashlib.sha256(
        json.dumps(comparison_payload(report), sort_keys=True, allow_nan=False).encode()
    ).hexdigest()
    if change == "digest":
        digest = "0" * 64
    (tmp_path / "candidate_comparison.json").write_text(
        json.dumps(comparison_payload(report)), encoding="utf-8"
    )
    (tmp_path / "candidate_training_spec.json").write_text(
        json.dumps(training.training_spec_payload(spec, "pandas")), encoding="utf-8"
    )
    client = Mock()
    client.get_model_version.return_value.run_id = "saved-run"
    client.download_artifacts.side_effect = lambda run, name, directory: str(tmp_path / name)
    if change == "none":
        loaded = evidence.load_candidate_evidence(client, "workspace.test.model", "1", digest)
        assert loaded == (report, spec, "pandas", None)
    else:
        messages = {
            "digest": "digest",
            "model": "requested candidate",
            "version": "requested candidate",
            "dataset": "snapshot",
            "holdout": "holdout membership",
        }
        with pytest.raises(ValueError, match=messages[change]):
            evidence.load_candidate_evidence(client, "workspace.test.model", "1", digest)


@pytest.mark.parametrize(
    "action,kind", [("approve", "promotion"), ("reject", "rejection"), ("rollback", "rollback")]
)
@pytest.mark.parametrize("handoff", ["disabled", "after_alias_change"])
def test_shared_result_preserves_operator_receipt_and_score_policy(action, kind, handoff):
    """Moving result construction must not change rollback inputs or trigger rejected scoring."""
    receipt = AliasChangeReceipt(
        event_id="c" * 32,
        kind=kind,
        model_name="workspace.test.model",
        alias="champion",
        prior_version="1",
        new_version="2",
        comparison_sha256="d" * 64,
        parent_event_id=None,
    )
    config = {
        "score_model_selection": "pinned_version",
        "promotion_policy": "manual_approval",
        "score_handoff": handoff,
    }
    result = workflow.build_bundle_result(config, action, receipt)
    assert result.result is receipt and result.action == action
    assert result.score_requested == (handoff == "after_alias_change" and action != "reject")
    if kind == "promotion":
        assert json.loads(result.next_actions["rollback"]["promotion_receipt_json"]) == asdict(
            receipt
        )
    else:
        assert result.next_actions == {}


def test_first_candidate_output_keeps_legacy_result_type():
    """Notebook callers retain their result class and copyable first-model approval inputs."""
    from skyulf.integrations.databricks.jobs.shared.job_runtime import BundleActionResult

    spec = _spec()
    candidate = training.LocalCandidateResult(
        run_id="run",
        model_name="workspace.test.model",
        model_version="1",
        model_digest="b" * 64,
        dataset_id=spec.dataset_id,
        training_rows=8,
        holdout_rows=2,
        unavailable_labels=0,
        engine="pandas",
        comparison=_report(spec),
        comparison_sha256="c" * 64,
        holdout_key_sha256="a" * 64,
    )
    result = workflow.build_bundle_result(
        {
            "score_model_selection": "pinned_version",
            "promotion_policy": "manual_approval",
            "score_handoff": "after_alias_change",
        },
        "train",
        candidate,
    )
    assert isinstance(result, BundleActionResult) and not result.score_requested
    assert result.result.comparison.champion_metrics is None
    assert result.next_actions == {
        action: {
            "lifecycle_action": action,
            "candidate_version": "1",
            "expected_champion_version": "none",
        }
        for action in ("approve", "reject")
    }


@pytest.mark.parametrize(
    "changes,message",
    [
        ({"version": -1, "test_size": 2}, "version must"),
        ({"test_size": 2, "max_rows": 0}, "test_size must"),
        ({"input_columns": ("id",), "max_rows": 0}, "columns must be distinct"),
        ({"max_rows": 0, "holdout_key_sha256": "bad"}, "max_rows must"),
    ],
)
def test_training_validation_preserves_first_error(changes, message):
    """Splitting validation must keep the same first diagnostic for contradictory requests."""
    with pytest.raises(ValueError, match=message):
        replace(_spec(), **changes)


@pytest.mark.parametrize("malformed", ["missing_counts", "typed_count", "foreign_source"])
def test_filter_evidence_rejects_malformed_or_foreign_receipts(malformed):
    """Helper extraction must retain structural error normalization and source identity checks."""
    spec = _spec()
    heldout = pd.DataFrame({"x": [1, 2]})
    heldout.attrs.update(
        pre_filter_key_sha256="b" * 64,
        survivor_key_sha256=None,
        train_key_sha256="c" * 64,
        holdout_key_sha256=spec.holdout_key_sha256,
        sample_key_sha256=None,
        source_rows=10,
        pre_filter_rows=10,
        survivor_rows=10,
        training_rows=8,
        pre_split_filter_counts=[],
    )
    receipt = evidence.build_training_evidence(spec, heldout, project_source_sha256=None)
    if malformed == "missing_counts":
        receipt.pop("filter_counts")
    elif malformed == "typed_count":
        receipt["training_rows"] = "8"
    else:
        receipt["project_source_sha256"] = "d" * 64
    spec = replace(spec, training_evidence_sha256=evidence.evidence_digest(receipt))
    message = "different project source" if malformed == "foreign_source" else "malformed"
    with pytest.raises(ValueError, match=message):
        evidence.validate_training_evidence(receipt, spec, project_source_sha256=None)
