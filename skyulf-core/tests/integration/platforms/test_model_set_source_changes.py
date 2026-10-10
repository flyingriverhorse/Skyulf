"""Source corrections reuse complete model-set publication without stale carry state."""

import json
from copy import deepcopy
from typing import Any, cast
from unittest.mock import Mock

import pandas as pd
import pytest
from tests.integration.platforms.test_model_set_batch import _saved_set, _transport
from tests.integration.platforms.test_model_set_scoring import temporal_set


def _previous(model, history=None):
    """Represent a successful earlier publication of the same set."""
    return {
        "skyulf_mode": "incremental_append",
        "artifact_kind": "model_set",
        "source_table_id": "source",
        "target_table_id": "target",
        "source_end_version": 0,
        "model_set_name": model.name,
        "model_set_version": model.version,
        "model_set_digest": model.digest,
        "set_history": history or {},
    }


def _changed_source(monkeypatch, batch, previous, snapshot, error=None):
    """Simulate the remote CDF boundary while keeping actual scoring and receipts."""
    from skyulf.integrations.databricks.scoring.incremental.incremental_batch import (
        SourceChangeRequiresRebuild,
    )

    def select(spark, source, prior, upper, period, functions):
        """Require the correction fallback to re-read the pinned complete snapshot."""
        assert upper == 1
        if prior is not None:
            raise error or SourceChangeRequiresRebuild("source correction")
        return snapshot

    monkeypatch.setattr(batch, "select_incremental_rows", select)
    monkeypatch.setattr(
        batch,
        "latest_source_version",
        lambda spark, name: {
            "version": 1 if name == "source" else 0,
            "userMetadata": json.dumps(previous) if name == "target" else None,
        },
    )
    monkeypatch.setattr(batch, "bounded_frame", lambda selected, *args: selected)


def _run(batch, model, artifact, policy="rebuild_on_change"):
    """Exercise the admitted scorer with the same model and a changed source version."""
    return batch._run_admitted_set(
        None,
        model,
        artifact,
        "source",
        "target",
        "source",
        "target",
        "incremental_append",
        100,
        1024 * 1024,
        source_change_policy=policy,
    )


def test_source_correction_rebuilds_same_model_and_records_reason(tmp_path, monkeypatch):
    """A model release change must not be required to replace corrected predictions."""
    artifact, model, query = _saved_set(tmp_path)
    previous = _previous(model)
    snapshot = query.iloc[:2].copy()
    snapshot.loc[snapshot.index[0], "amount"] = 100.0
    batch, commit = _transport(monkeypatch, query, previous)
    _changed_source(monkeypatch, batch, previous, snapshot)
    result = _run(batch, model, artifact)
    output = commit.call_args.args[1]
    assert output.id.tolist() == [3, 1]
    assert output.amount__prediction.tolist() == pytest.approx([122, 18])
    assert commit.call_args.args[-1] == "overwrite"
    assert result.manifest["source_rebuilt"] is True
    assert result.manifest["source_start_version"] is None
    assert result.manifest["source_end_version"] == 1
    assert result.manifest["source_change_policy"] == "rebuild_on_change"


def test_delete_all_publishes_an_empty_replacement(tmp_path, monkeypatch):
    """Removing the last source row must clear old predictions and commit progress."""
    artifact, model, query = _saved_set(tmp_path)
    previous = _previous(model)
    batch, commit = _transport(monkeypatch, query, previous)
    _changed_source(monkeypatch, batch, previous, query.iloc[:0])
    result = _run(batch, model, artifact)
    assert not result.noop and result.output_count == 0
    assert commit.call_args.args[1].empty
    assert commit.call_args.args[-1] == "overwrite"
    assert result.manifest["set_history"] == {}


def test_snapshot_limit_failure_preserves_previous_publication(tmp_path, monkeypatch):
    """A rebuild must respect the full-snapshot budget instead of sampling corrections."""
    artifact, model, query = _saved_set(tmp_path)
    previous = _previous(model)
    batch, commit = _transport(monkeypatch, query, previous)
    _changed_source(monkeypatch, batch, previous, query)
    monkeypatch.setattr(
        batch, "bounded_frame", Mock(side_effect=ValueError("Source increment exceeds max_rows."))
    )
    with pytest.raises(ValueError, match="max_rows"):
        _run(batch, model, artifact)
    commit.assert_not_called()


def test_rebuild_policy_keeps_insert_batches_as_appends(tmp_path, monkeypatch):
    """Opting into source correction must not turn ordinary new rows into full rebuilds."""
    artifact, model, query = _saved_set(tmp_path)
    previous = _previous(model)
    batch, commit = _transport(monkeypatch, query, previous)
    monkeypatch.setattr(
        batch,
        "latest_source_version",
        lambda spark, name: {
            "version": 1 if name == "source" else 0,
            "userMetadata": json.dumps(previous) if name == "target" else None,
        },
    )
    result = _run(batch, model, artifact)
    assert commit.call_args.args[-1] == "append"
    assert result.manifest["source_rebuilt"] is False
    assert result.manifest["source_start_version"] == 1


@pytest.mark.parametrize(
    "policy,error",
    [
        ("reject", None),
        ("rebuild_on_change", ValueError("CDF expired")),
        ("rebuild_on_change", PermissionError("CDF denied")),
    ],
)
def test_unapproved_or_unreadable_changes_never_publish(tmp_path, monkeypatch, policy, error):
    """Only an explicitly allowed correction can use full-snapshot recovery."""
    artifact, model, query = _saved_set(tmp_path)
    previous = _previous(model)
    batch, commit = _transport(monkeypatch, query, previous)
    _changed_source(monkeypatch, batch, previous, query, error)
    expected = type(error) if error is not None else ValueError
    with pytest.raises(expected):
        _run(batch, model, artifact, policy)
    commit.assert_not_called()


def test_invalid_policy_fails_before_spark_or_catalog_mutations(tmp_path):
    """Misspelled policies must never create output resources or silently choose recovery."""
    from skyulf.integrations.databricks.model_sets.model_set_batch import run_model_set_batch

    artifact, model, _ = _saved_set(tmp_path)
    spark = Mock()
    with pytest.raises(ValueError, match="source_change_policy"):
        run_model_set_batch(
            spark,
            model,
            artifact,
            source_table="db.source",
            prediction_table="db.results",
            admission=None,
            source_change_policy="typo",
        )
    assert spark.mock_calls == []


def test_temporal_correction_resets_history_and_preserves_next_append(tmp_path, monkeypatch):
    """Corrected history must match fresh scoring and drive the next prediction causally."""
    from skyulf.inference.model_set_scoring import score_model_set
    from skyulf.integrations.mlflow.registration.registry import ResolvedModel

    artifact = cast(Any, temporal_set).__wrapped__(tmp_path)
    model = ResolvedModel("db.set", "1", "models:/db.set/1", None, artifact.manifest.set_sha256)
    original = pd.DataFrame({"id": [0, 1, 2], "t": [0, 1, 2], "v": [0.0, 1.0, 2.0]})
    old = score_model_set(original, artifact, bootstrap_history=True)
    previous = _previous(model, old.history)
    before = deepcopy(previous)
    corrected = original.copy()
    corrected.loc[0, "v"] = 12.0
    batch, commit = _transport(monkeypatch, original, previous)
    _changed_source(monkeypatch, batch, previous, corrected)
    result = _run(batch, model, artifact)
    fresh = score_model_set(corrected, artifact, bootstrap_history=True)
    pd.testing.assert_frame_equal(commit.call_args.args[1], fresh.frame)
    assert result.manifest["set_history"] == fresh.history
    assert previous == before
    next_row = pd.DataFrame({"id": [3], "t": [3], "v": [3.0]})
    continued = score_model_set(next_row, artifact, history_state=result.manifest["set_history"])
    whole = score_model_set(
        pd.concat([corrected, next_row], ignore_index=True), artifact, bootstrap_history=True
    )
    assert continued.frame["a__prediction"].tolist() == pytest.approx(
        whole.frame["a__prediction"].iloc[-1:].tolist()
    )
    assert continued.history == whole.history
