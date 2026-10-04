"""CDF recovery replaces a complete model-set snapshot under the original write guard."""

import json
from unittest.mock import Mock

import pytest
from tests.integration.platforms.test_model_set_batch import _saved_set, _transport

from skyulf.integrations.databricks.admission import BatchConflictError
from skyulf.integrations.databricks.cdf_recovery import CdfHistoryExpired, CdfRecoveryRequired

SOURCE = "workspace.test.source"
TARGET = "workspace.test.predictions"


def setup_batch(tmp_path, monkeypatch):
    """Keep real fitted model predictions while replacing the remote Delta transport."""
    artifact, model, query = _saved_set(tmp_path)
    batch, commit = _transport(monkeypatch, query)
    previous = batch._publication_receipt(model, SOURCE, TARGET, None, 5, 2, 3, {})
    state = {"version": 3, "userMetadata": json.dumps(previous)}
    monkeypatch.setattr(
        batch,
        "latest_source_version",
        lambda spark, name: state if name == TARGET else {"version": 9},
    )
    request = {
        "version": 1,
        "layout": "model_set",
        "source_table": SOURCE,
        "source_table_id": SOURCE,
        "target_table": TARGET,
        "target_table_id": TARGET,
        "target_version": 3,
        "source_start_version": 5,
        "source_end_version": 9,
        "model_name": model.name,
        "model_version": model.version,
        "model_digest": model.digest,
    }
    args = (None, model, artifact, SOURCE, TARGET, SOURCE, TARGET, "incremental_append", 20, 2**20)
    return batch, commit, query, state, request, args


@pytest.mark.parametrize("failure_site", ["select_incremental_rows", "bounded_frame"])
def test_unreadable_set_cdf_proposes_recovery_without_writing(tmp_path, monkeypatch, failure_site):
    """Both eager and deferred CDF failures must preserve output and pin the exact request."""
    batch, commit, _, _, request, args = setup_batch(tmp_path, monkeypatch)
    monkeypatch.setattr(batch, failure_site, Mock(side_effect=CdfHistoryExpired("expired")))
    with pytest.raises(CdfRecoveryRequired) as error:
        batch._run_admitted_set(*args)
    assert error.value.request == request
    commit.assert_not_called()


@pytest.mark.parametrize("empty", [False, True])
def test_recovery_overwrites_complete_pinned_set_and_can_replay(tmp_path, monkeypatch, empty):
    """An empty recovered snapshot clears stale predictions and an exact retry writes nothing."""
    batch, commit, query, state, request, args = setup_batch(tmp_path, monkeypatch)
    if empty:
        query = query.iloc[:0]
    select = Mock(return_value=query)
    monkeypatch.setattr(batch, "select_incremental_rows", select)
    monkeypatch.setattr(batch, "bounded_frame", lambda *args: query)
    result = batch._run_admitted_set(*args, recovery_request=request)
    assert result.input_count == result.output_count == len(query)
    assert not result.noop
    assert commit.call_args.args[-1] == "overwrite"
    assert select.call_args.args[2:4] == (None, 9)
    assert result.manifest["cdf_recovered"] is True
    assert result.manifest["cdf_recovery_request"] == request
    state.update(version=4, userMetadata=json.dumps(result.manifest))
    replay = batch._run_admitted_set(*args, recovery_request=request)
    assert replay.noop and replay.commit_version == 4
    assert select.call_count == commit.call_count == 1


def test_recovered_set_resumes_incremental_after_pinned_watermark(tmp_path, monkeypatch):
    """Recovery is a single replacement, not a permanent switch to full recomputation."""
    batch, commit, query, state, request, args = setup_batch(tmp_path, monkeypatch)
    recovered = batch._run_admitted_set(*args, recovery_request=request)
    state.update(version=4, userMetadata=json.dumps(recovered.manifest))
    monkeypatch.setattr(
        batch,
        "latest_source_version",
        lambda spark, name: state if name == TARGET else {"version": 10},
    )
    select = Mock(return_value=query)
    monkeypatch.setattr(batch, "select_incremental_rows", select)
    result = batch._run_admitted_set(*args)
    assert select.call_args.args[2:4] == (9, 10)
    assert commit.call_args.args[-1] == "append"
    assert "cdf_recovered" not in result.manifest


@pytest.mark.parametrize("mutation", ["version", "source_id", "target_id", "model_digest"])
def test_stale_set_recovery_never_predicts_or_writes(tmp_path, monkeypatch, mutation):
    """A delayed request must not overwrite changed tables or use another saved artifact."""
    batch, commit, _, state, request, args = setup_batch(tmp_path, monkeypatch)
    if mutation == "version":
        state["version"] = 4
    elif mutation == "model_digest":
        request["model_digest"] = "a" * 64
    else:
        request["source_table_id" if mutation == "source_id" else "target_table_id"] = "replaced"
    predictor = Mock()
    monkeypatch.setattr(batch, "_score_increment", predictor)
    with pytest.raises(BatchConflictError):
        batch._run_admitted_set(*args, recovery_request=request)
    predictor.assert_not_called()
    commit.assert_not_called()


def test_failed_set_recovery_preserves_previous_output(tmp_path, monkeypatch):
    """Full snapshot limits remain enforced before any replacement reaches Delta."""
    batch, commit, _, _, request, args = setup_batch(tmp_path, monkeypatch)
    monkeypatch.setattr(batch, "bounded_frame", Mock(side_effect=ValueError("max_rows")))
    with pytest.raises(ValueError, match="max_rows"):
        batch._run_admitted_set(*args, recovery_request=request)
    commit.assert_not_called()


def test_set_overwrite_clears_partitions_even_with_dynamic_session_defaults(monkeypatch):
    """Full recovery must delete absent partitions, including a completely empty snapshot."""
    from skyulf.integrations.databricks import model_set_batch as batch

    receipt = {"source_table_id": SOURCE, "target_table_id": TARGET, "expected_target_version": 3}
    monkeypatch.setattr(batch, "table_identity", lambda spark, name: name)
    monkeypatch.setattr(
        batch, "latest_source_version", Mock(side_effect=[{"version": 3}, {"version": 4}])
    )
    monkeypatch.setattr(batch, "_previous_set_receipt", lambda *args: receipt)
    output = Mock()
    writer = output.write
    writer.format.return_value = writer
    writer.mode.return_value = writer
    writer.option.return_value = writer
    assert batch._commit_set(None, output, SOURCE, TARGET, receipt, "overwrite") == 4
    options = {call.args[0]: call.args[1] for call in writer.option.call_args_list}
    assert options["partitionOverwriteMode"] == "static"
    assert options["overwriteSchema"] == "false"
    writer.mode.assert_called_once_with("overwrite")
