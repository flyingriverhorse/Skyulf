"""Starting after existing rows must survive every later receipt and rebuild."""

import pytest


@pytest.mark.parametrize("mode", ["full_snapshot", "from_now"])
def test_initial_choice_is_persisted_and_reused(mode):
    """A later source version must never move the first scoring boundary."""
    from skyulf.integrations.databricks.scoring.incremental.initial_scoring import start_fields

    first = start_fields(None, mode, 7)
    receipt = first | {"source_end_version": 9}
    assert start_fields(receipt, mode, 11) == first
    assert first == (
        {"initial_scoring": "from_now", "scoring_start_version": 7} if mode == "from_now" else {}
    )


@pytest.mark.parametrize("mode", [None, True, "now", [], 1])
def test_initial_choice_rejects_invalid_values(mode):
    """Unknown choices cannot accidentally trigger an initial full-table prediction."""
    from skyulf.integrations.databricks.scoring.incremental.initial_scoring import start_fields

    with pytest.raises(ValueError, match="initial_scoring"):
        start_fields(None, mode, 7)


@pytest.mark.parametrize(
    "previous,mode",
    [
        ({"source_end_version": 3}, "from_now"),
        (
            {"initial_scoring": "from_now", "scoring_start_version": 2, "source_end_version": 3},
            "full_snapshot",
        ),
    ],
)
def test_existing_target_cannot_change_initial_choice(previous, mode):
    """Editing YAML must not discard the trusted watermark or add excluded history."""
    from skyulf.integrations.databricks.data.admission import BatchConflictError
    from skyulf.integrations.databricks.scoring.incremental.initial_scoring import start_fields

    with pytest.raises(BatchConflictError, match="initial_scoring"):
        start_fields(previous, mode, 8)


@pytest.mark.parametrize("boundary", [None, -1, True, "2", 4, 1.5])
def test_invalid_start_receipt_is_rejected(boundary):
    """A missing, mistyped or future boundary must not expand rebuild membership."""
    from skyulf.integrations.databricks.data.admission import BatchConflictError
    from skyulf.integrations.databricks.scoring.incremental.initial_scoring import start_fields

    with pytest.raises(BatchConflictError, match="scoring_start_version"):
        start_fields(
            {
                "initial_scoring": "from_now",
                "scoring_start_version": boundary,
                "source_end_version": 3,
            },
            "from_now",
            8,
        )


def test_full_snapshot_receipt_cannot_hide_a_boundary():
    """A malformed receipt cannot be interpreted as permission to score history."""
    from skyulf.integrations.databricks.data.admission import BatchConflictError
    from skyulf.integrations.databricks.scoring.incremental.initial_scoring import start_fields

    with pytest.raises(BatchConflictError, match="scoring_start_version"):
        start_fields({"source_end_version": 4, "scoring_start_version": 2}, "full_snapshot", 8)


def test_legacy_receipt_preserves_existing_full_snapshot_behavior():
    """Existing projects do not require a migration or a changed historical digest."""
    from skyulf.integrations.databricks.scoring.incremental.initial_scoring import start_fields

    assert start_fields({"source_end_version": 2}, "full_snapshot", 3) == {}


def test_model_set_first_from_now_commit_is_empty(tmp_path, monkeypatch):
    """The first checkpoint must commit zero predictions even when source history exists."""
    from tests.integration.platforms.test_model_set_batch import _saved_set, _transport

    artifact, model, query = _saved_set(tmp_path)
    batch, commit = _transport(monkeypatch, query)
    monkeypatch.setattr(
        batch, "scoring_snapshot", lambda *args, **kwargs: query.iloc[:0], raising=False
    )
    monkeypatch.setattr(batch, "bounded_frame", lambda selected, *args: selected)
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
        initial_scoring="from_now",
    )
    assert result.input_count == result.output_count == 0
    assert not result.noop
    assert result.manifest["scoring_start_version"] == 0
    assert commit.call_args.args[1].empty


def test_single_initial_read_uses_only_empty_bounded_snapshot(monkeypatch):
    """Skipped historical rows must not undergo model validation or driver collection."""
    from types import SimpleNamespace
    from typing import cast
    from unittest.mock import Mock

    from skyulf.integrations.databricks.scoring.incremental import incremental_batch as batch
    from skyulf.integrations.databricks.scoring.workflow import PreparedWorkflow

    selected = object()
    snapshot = Mock(return_value=selected)
    monkeypatch.setattr(batch, "scoring_snapshot", snapshot, raising=False)
    monkeypatch.setattr(
        batch, "select_incremental_rows", Mock(side_effect=AssertionError("history"))
    )
    monkeypatch.setattr(batch, "_read_scoring_rows", lambda selected, *args: "empty frame")
    prepared = SimpleNamespace(config=SimpleNamespace(source=SimpleNamespace(table="source")))
    actual = batch._read_incremental_input(
        None,
        cast(PreparedWorkflow, prepared),
        ("id",),
        None,
        ("x",),
        "source_id",
        "target_id",
        {"version": 0},
        None,
        7,
        None,
        boundary=7,
    )
    snapshot.assert_called_once_with(None, "source", 7, ("id",), 7, checkpoint=None)
    assert actual == (selected, "empty frame")


@pytest.mark.parametrize("engine", ["pandas", "polars"])
def test_zero_start_checkpoint_has_valid_empty_temporal_context(tmp_path, engine):
    """The first zero-row receipt must not replay training history on the next increment."""
    from types import SimpleNamespace
    from typing import cast

    import numpy as np
    import pandas as pd
    from tests.integration.platforms.test_temporal_history import fitted_temporal_pipeline

    from skyulf.inference.fitted_pipeline import load_pipeline, predict_pipeline, save_pipeline
    from skyulf.integrations.databricks.scoring.incremental.history import (
        history_receipt,
        incremental_history,
    )
    from skyulf.integrations.databricks.scoring.incremental.incremental_batch import (
        _empty_incremental_predictions,
    )
    from skyulf.integrations.databricks.scoring.workflow import PreparedWorkflow

    save_pipeline(fitted_temporal_pipeline(engine), tmp_path / "model")
    artifact = load_pipeline(tmp_path / "model")
    prepared = cast(PreparedWorkflow, SimpleNamespace(artifact=artifact))
    with incremental_history(prepared, None) as first:
        empty = _empty_incremental_predictions(prepared, ("prediction",))
    assert empty.empty
    receipt = history_receipt(first)
    rows = pd.DataFrame({"t": [20, 21, 22], "v": [20.0, 21.0, 22.0]})
    with incremental_history(prepared, receipt):
        actual = predict_pipeline(rows, artifact)
    with incremental_history(prepared, None):
        expected = predict_pipeline(rows, artifact)
    np.testing.assert_allclose(actual["prediction"], expected["prediction"], equal_nan=True)


def test_from_now_rebuild_retains_event_time_validation(monkeypatch):
    """A retained boundary cannot bypass the existing null-event-time publication guard."""
    from types import SimpleNamespace
    from typing import cast
    from unittest.mock import Mock

    from skyulf.integrations.databricks.scoring.incremental import incremental_batch as batch
    from skyulf.integrations.databricks.scoring.workflow import PreparedWorkflow

    frame = Mock()
    frame.schema = {"at": SimpleNamespace(dataType=SimpleNamespace(typeName=lambda: "timestamp"))}
    frame.where.return_value.limit.return_value.count.return_value = 1
    monkeypatch.setattr(batch, "scoring_snapshot", lambda *args, **kwargs: frame)
    read = Mock()
    monkeypatch.setattr(batch, "_read_scoring_rows", read)
    prepared = cast(
        PreparedWorkflow,
        SimpleNamespace(config=SimpleNamespace(source=SimpleNamespace(table="source"))),
    )
    with pytest.raises(ValueError, match="event time must not be null"):
        batch._read_incremental_input(
            None, prepared, ("id",), "at", ("x",), "s", "t", {}, None, 9, Mock(), boundary=7
        )
    read.assert_not_called()


def test_model_set_rebuild_pins_membership_from_previous_output(tmp_path, monkeypatch):
    """Normal model changes must not depend on the old source boundary snapshot."""
    import json
    from unittest.mock import Mock

    from tests.integration.platforms.test_model_set_batch import _saved_set, _transport

    artifact, model, query = _saved_set(tmp_path)
    previous = {
        "skyulf_mode": "incremental_append",
        "artifact_kind": "model_set",
        "source_table_id": "source",
        "target_table_id": "target",
        "model_set_name": model.name,
        "model_set_version": "1",
        "model_set_digest": "older",
        "source_end_version": 8,
        "initial_scoring": "from_now",
        "scoring_start_version": 1,
        "set_history": {},
    }
    batch, _ = _transport(monkeypatch, query, previous)
    monkeypatch.setattr(
        batch,
        "latest_source_version",
        lambda spark, table: {
            "version": 9 if table == "source" else 4,
            "userMetadata": json.dumps(previous),
        },
    )
    snapshot = Mock(return_value=query)
    monkeypatch.setattr(
        batch,
        "prediction_commit",
        lambda *args: {
            "version": 2,
            "userMetadata": json.dumps(previous),
        },
    )
    monkeypatch.setattr(batch, "scoring_snapshot", snapshot)
    result = batch._run_admitted_set(
        None,
        model,
        artifact,
        "source",
        "target",
        "source",
        "target",
        "full_rebuild",
        20,
        1048576,
        initial_scoring="from_now",
    )
    checkpoint = snapshot.call_args.kwargs.get("checkpoint")
    assert checkpoint is not None
    assert (
        checkpoint.table,
        checkpoint.table_id,
        checkpoint.version,
        checkpoint.source_version,
    ) == (
        "target",
        "target",
        4,
        8,
    )
    assert result.manifest["scoring_start_version"] == 1


@pytest.mark.parametrize(
    "condition",
    [
        "DELTA_TRUNCATED_TRANSACTION_LOG",
        "DELTA_FILE_NOT_FOUND",
        "FAILED_READ_FILE.DBR_FILE_NOT_EXIST",
    ],
)
def test_lazy_snapshot_retention_failure_has_actionable_error(monkeypatch, condition):
    """Materialization failures must never escape as an unclassified Spark file error."""
    from types import SimpleNamespace
    from typing import cast
    from unittest.mock import Mock

    from skyulf.integrations.databricks.scoring.incremental import incremental_batch as batch
    from skyulf.integrations.databricks.scoring.workflow import PreparedWorkflow

    class ExpiredSnapshot(RuntimeError):
        """Expose the same structured retention condition as Delta."""

        def getErrorClass(self):
            """Avoid matching arbitrary error message text."""
            return condition

    monkeypatch.setattr(batch, "scoring_snapshot", Mock(return_value=object()))
    monkeypatch.setattr(batch, "_read_scoring_rows", Mock(side_effect=ExpiredSnapshot("expired")))
    prepared = cast(
        PreparedWorkflow,
        SimpleNamespace(config=SimpleNamespace(source=SimpleNamespace(table="source"))),
    )
    with pytest.raises(ValueError, match="from_now rebuild.*scoring_start_version=7"):
        batch._read_incremental_input(
            None, prepared, ("id",), None, ("x",), "s", "t", {}, None, 9, Mock(), boundary=7
        )
