"""Model-set publication preserves complete releases and source watermarks."""

from typing import Any, cast

import pytest


def _saved_set(tmp_path, source="", config=None):
    """Use real fitted component bytes while isolating only the Delta transport."""
    from tests.integration.platforms.test_local_pipeline_artifact import _fitted_pipeline

    from skyulf.inference._manifest import ColumnSpec
    from skyulf.inference.local_pipeline import load_local_pipeline, save_local_pipeline
    from skyulf.inference.model_set import ComponentReference, save_model_set
    from skyulf.integrations.mlflow.registration.registry import ResolvedModel

    pipeline, query = _fitted_pipeline("pandas")
    path = tmp_path / "component"
    save_local_pipeline(pipeline, path)
    digest = load_local_pipeline(path).manifest.pipeline_sha256
    artifact = save_model_set(
        tmp_path / "set",
        {
            "amount": (
                ComponentReference(name="workspace.test.amount", version="1", digest=digest),
                path,
            )
        },
        record_key_schema=(ColumnSpec(name="id", dtype="int64"),),
        composition_source=source,
        composition_config=config,
    )
    model = ResolvedModel(
        "workspace.test.set",
        "1",
        "models:/workspace.test.set/1",
        None,
        artifact.manifest.set_sha256,
    )
    query = query.reset_index(drop=True)
    query.insert(0, "id", [3, 1, 2])
    return artifact, model, query


def _transport(monkeypatch, query, previous=None):
    """Replace remote I/O while keeping actual model and rule computation."""
    import importlib
    from types import SimpleNamespace
    from unittest.mock import Mock

    from skyulf.integrations.databricks.model_sets import model_set_batch as batch

    monkeypatch.setattr(
        batch,
        "importlib",
        SimpleNamespace(
            import_module=lambda name: (
                object() if name == "pyspark.sql.functions" else importlib.import_module(name)
            )
        ),
    )
    monkeypatch.setattr(batch, "table_identity", lambda spark, name: name)
    metadata = __import__("json").dumps(previous) if previous else None
    monkeypatch.setattr(
        batch,
        "latest_source_version",
        lambda spark, name: {"version": 0, "userMetadata": metadata if name == "target" else None},
    )
    monkeypatch.setattr(batch, "check_incremental_bootstrap", lambda *args: None)
    monkeypatch.setattr(batch, "select_incremental_rows", lambda *args: query)
    monkeypatch.setattr(batch, "bounded_frame", lambda *args: query)
    monkeypatch.setattr(batch, "_output_frame", lambda spark, frame, *args: frame)
    commit = Mock(return_value=1)
    monkeypatch.setattr(batch, "_commit_set", commit)
    return batch, commit


def test_publication_contains_all_real_predictions_and_set_identity(tmp_path, monkeypatch):
    """The final Delta writer must receive complete keyed predictions and the exact set pin."""
    import numpy as np

    artifact, model, query = _saved_set(tmp_path)
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
    )
    output = commit.call_args.args[1]
    assert output.id.tolist() == [3, 1, 2]
    np.testing.assert_allclose(output.amount__prediction.to_numpy(dtype=float), [29, 18, 41])
    assert result.manifest["model_set_digest"] == artifact.manifest.set_sha256
    assert result.input_count == result.output_count == 3
    assert not result.noop and result.commit_version == 1


def test_rule_exception_never_calls_delta_commit(tmp_path, monkeypatch):
    """Successful component predictions cannot leak into final output when a rule fails."""
    config = {
        "outputs": [
            {
                "name": "broken",
                "version": "1",
                "function": "fail",
                "params": {},
                "columns": [{"name": "business_value", "dtype": "float64"}],
                "required_components": ["amount"],
            }
        ]
    }
    artifact, model, query = _saved_set(
        tmp_path,
        "def fail(inputs, predictions, params):\n    raise RuntimeError('rule failed')\n",
        config,
    )
    batch, commit = _transport(monkeypatch, query)
    with pytest.raises(RuntimeError, match="rule failed"):
        batch._run_admitted_set(
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
        )
    commit.assert_not_called()


def test_foreign_receipt_is_not_a_set_watermark():
    """An ordinary model's table must not be adopted as a model-set target."""
    import json

    from skyulf.integrations.databricks.data.admission import BatchConflictError
    from skyulf.integrations.databricks.model_sets.model_set_batch import _previous_set_receipt

    previous = {
        "skyulf_mode": "incremental_append",
        "source_table_id": "s",
        "target_table_id": "t",
        "source_end_version": 0,
    }
    with pytest.raises(BatchConflictError, match="another scoring contract"):
        _previous_set_receipt({"userMetadata": json.dumps(previous)}, "s", "t")


def test_empty_full_rebuild_replaces_old_output(tmp_path, monkeypatch):
    """A complete empty replacement must clear old predictions instead of reporting a no-op."""
    artifact, model, query = _saved_set(tmp_path)
    previous = {
        "skyulf_mode": "incremental_append",
        "artifact_kind": "model_set",
        "source_table_id": "source",
        "target_table_id": "target",
        "source_end_version": 0,
        "model_set_name": model.name,
        "model_set_version": "1",
        "model_set_digest": "older",
        "set_history": {},
    }
    import pandas as pd

    batch, commit = _transport(monkeypatch, pd.DataFrame(columns=query.columns), previous)
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
        1024 * 1024,
    )
    assert not result.noop and result.output_count == 0
    assert commit.call_args.args[-1] == "overwrite"


def test_append_cannot_bootstrap_new_temporal_set_from_only_new_rows(tmp_path):
    """Switching from stateless to carry models requires the complete source history."""
    import pandas as pd
    from tests.integration.platforms.test_model_set_scoring import temporal_set

    from skyulf.integrations.databricks.model_sets.model_set_batch import _score_increment

    artifact = cast(Any, temporal_set).__wrapped__(tmp_path)
    previous = {"model_set_digest": "previous-stateless-set", "set_history": {}}
    with pytest.raises(ValueError, match="full_rebuild"):
        _score_increment(
            artifact,
            pd.DataFrame({"id": [22], "t": [22], "v": [0.0]}),
            previous,
            "append",
            10,
            1024 * 1024,
        )


def test_new_set_rebuilds_without_new_source_rows():
    """Changing a complete set must recompute the snapshot in full-rebuild mode."""
    from skyulf.integrations.databricks.model_sets.model_set_batch import _plan_publication

    previous = {"source_end_version": 3, "model_set_digest": "old"}
    assert _plan_publication(previous, set_digest="new", mode="full_rebuild", source_version=3) == (
        None,
        "overwrite",
        False,
    )


def test_same_set_and_snapshot_is_a_noop():
    """A successful repeated scoring invocation must not duplicate output rows."""
    from skyulf.integrations.databricks.model_sets.model_set_batch import _plan_publication

    previous = {"source_end_version": 3, "model_set_digest": "same"}
    assert _plan_publication(
        previous, set_digest="same", mode="full_rebuild", source_version=3
    ) == (3, "append", True)


def test_new_registry_version_rebuilds_identical_package(tmp_path, monkeypatch):
    """A newly selected registry release must update full-rebuild row provenance."""
    artifact, model, query = _saved_set(tmp_path)
    previous = {
        "skyulf_mode": "incremental_append",
        "artifact_kind": "model_set",
        "source_table_id": "source",
        "target_table_id": "target",
        "source_end_version": 0,
        "model_set_name": model.name,
        "model_set_version": "2",
        "model_set_digest": model.digest,
        "set_history": {},
    }
    batch, commit = _transport(monkeypatch, query, previous)
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
        1024 * 1024,
    )
    assert not result.noop and result.output_count == 3
    assert commit.call_args.args[-1] == "overwrite"


def test_append_set_change_keeps_prior_row_provenance():
    """Append mode starts after the committed watermark even when a new set is selected."""
    from skyulf.integrations.databricks.model_sets.model_set_batch import _plan_publication

    previous = {"source_end_version": 3, "model_set_digest": "old"}
    assert _plan_publication(
        previous, set_digest="new", mode="incremental_append", source_version=4
    ) == (3, "append", False)


def test_watermark_cannot_move_backwards():
    """A recreated or rewound source must never silently replace a prior source identity."""
    from skyulf.integrations.databricks.data.admission import BatchConflictError
    from skyulf.integrations.databricks.model_sets.model_set_batch import _plan_publication

    with pytest.raises(BatchConflictError, match="behind"):
        _plan_publication(
            {"source_end_version": 3, "model_set_digest": "same"},
            set_digest="same",
            mode="incremental_append",
            source_version=2,
        )


@pytest.mark.parametrize("mode", ["overwrite", "typo", None])
def test_unknown_publication_policy_fails(mode):
    """Misspelled policies cannot silently select a destructive overwrite."""
    from skyulf.integrations.databricks.model_sets.model_set_batch import _plan_publication

    with pytest.raises(ValueError, match="mode"):
        _plan_publication(None, set_digest="new", mode=mode, source_version=0)


def test_batch_rejects_unpinned_models_before_spark():
    """Scoring cannot provision or read tables before its set identity is concrete."""
    from skyulf.integrations.databricks.model_sets.model_set_batch import run_model_set_batch

    with pytest.raises(TypeError, match="ResolvedModel"):
        run_model_set_batch(
            None,
            cast(Any, None),
            cast(Any, None),
            source_table="workspace.test.source",
            prediction_table="workspace.test.predictions",
            admission=None,
        )
