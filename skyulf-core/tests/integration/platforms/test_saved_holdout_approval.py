"""Saved enriched holdouts survive feature updates without relaxing promotion guards."""

import json
from copy import deepcopy
from dataclasses import replace
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import Mock

import pandas as pd
import pytest

pytest.importorskip("mlflow")

from skyulf.integrations.databricks.feature_store import snapshots, training  # noqa: E402
from skyulf.integrations.databricks.feature_store.lifecycle_config import (  # noqa: E402
    binding_json,
    serialize_feature_spec,
)
from skyulf.integrations.databricks.lifecycle import approval  # noqa: E402
from skyulf.integrations.databricks.lifecycle.workflow import training_spec  # noqa: E402
from skyulf.integrations.databricks.training.shared import holdout_evidence as saved  # noqa: E402
from skyulf.integrations.mlflow.lifecycle.promotion import (
    ExclusiveAliasWriterAdmission,  # noqa: E402
)


@pytest.fixture
def feature_spec(workflow_config):
    """Freeze a native feature source independently of mutable latest Delta values."""
    workflow_config.update(
        engine="pandas",
        inference_mode="spark",
        feature_lookup={
            "lookups": [
                {
                    "table_name": "workspace.features.latest",
                    "lookup_key": ["id"],
                    "feature_names": ["x"],
                }
            ]
        },
    )
    spec = training_spec(workflow_config)
    lookup = training.training_lookup(spec)
    assert lookup is not None
    binding = {
        "version": 1,
        "lookup_spec": serialize_feature_spec(lookup),
        "lookup_evidence": {
            "policy": "training_snapshot",
            "feature_tables": [
                {
                    "table_name": "workspace.features.latest",
                    "table_id": "features-id",
                    "version": 1,
                }
            ],
        },
    }
    return replace(spec, feature_binding_json=binding_json(binding))


def test_approval_after_feature_advance_uses_saved_holdout(feature_spec, monkeypatch):
    """The historical training snapshot check must not prevent saved offline approval."""
    report = SimpleNamespace(candidate_digest="digest", holdout_reference={"version": 1})
    config = {"model_name": "main.ml.model", "max_rows": 20, "max_input_mb": 2}
    monkeypatch.setattr(approval, "require_mlflow", Mock())
    monkeypatch.setattr(approval, "make_registry_client", Mock())
    monkeypatch.setattr(
        approval,
        "load_candidate_evidence",
        lambda *a, **k: (
            report,
            feature_spec,
            "pandas",
            None,
        ),
    )
    monkeypatch.setattr(approval, "_validate_approval_policy", Mock())
    monkeypatch.setattr(approval, "resolve_model", lambda *a, **k: SimpleNamespace(digest="digest"))
    monkeypatch.setattr(approval, "controlled_champion_version", lambda *a, **k: "1")
    monkeypatch.setattr(approval, "verify_staged_challenger", Mock())
    monkeypatch.setattr(
        snapshots,
        "snapshot_table",
        lambda *a: {
            "table_name": "workspace.features.latest",
            "table_id": "features-id",
            "version": 2,
        },
    )
    # Exercise the actual read -> pin -> validate_snapshots failure chain on RED.
    heldout = pd.DataFrame({"x": [1.0, 2.0], "target": [2.0, 4.0]})
    loader = Mock(return_value=heldout)
    monkeypatch.setattr(approval, "load_saved_holdout", loader, raising=False)
    promoter = Mock(return_value="receipt")
    monkeypatch.setattr(approval, "promote_candidate", promoter)
    assert (
        approval.approve_saved_candidate(
            None,
            config,
            candidate_version="2",
            comparison_sha256="a" * 64,
            expected_champion_version="1",
            admission=ExclusiveAliasWriterAdmission(),
        )
        == "receipt"
    )
    assert promoter.call_args.args[1] is heldout
    loader.assert_called_once()


@pytest.fixture
def saved_reference(feature_spec, tmp_path, monkeypatch):
    """Store real Parquet while isolating only remote artifact and Delta I/O."""
    frame = pd.DataFrame({"x": [1.0, 2.0], "target": [2.0, 4.0]})
    frame.attrs.update(
        {
            "feature_lookup": json.loads(feature_spec.feature_binding_json),
            "source_table_id": "source-id",
            "holdout_key_sha256": "a" * 64,
            "train_key_sha256": "b" * 64,
            "group_split": {"column": "group", "holdout_groups": 2},
        }
    )
    spec = replace(feature_spec, holdout_key_sha256="a" * 64)
    report = SimpleNamespace(
        model_name="main.ml.model", candidate_version="2", candidate_digest="digest", row_count=2
    )
    root = tmp_path / "artifacts"
    client = Mock()

    def upload(run_id, local, folder):
        """Persist actual serialized bytes for mutation and readback."""
        dest = root / folder / Path(local).name
        dest.parent.mkdir(parents=True, exist_ok=True)
        dest.write_bytes(Path(local).read_bytes())

    def listing(*args):
        """Expose real file sizes so pre-download budget guards execute."""
        path = root / "lifecycle/data/holdout.parquet"
        return (
            [
                SimpleNamespace(
                    path="lifecycle/data/holdout.parquet",
                    is_dir=False,
                    file_size=path.stat().st_size,
                )
            ]
            if path.exists()
            else []
        )

    client.log_artifact.side_effect = upload
    client.list_artifacts.side_effect = listing
    client.download_artifacts.side_effect = lambda run, path, directory: str(root / path)
    client.get_model_version.return_value = SimpleNamespace(run_id="run-id")
    monkeypatch.setattr(
        saved,
        "table_identity",
        lambda spark, table: "source-id" if table == spec.table else "features-id",
    )
    run = SimpleNamespace(client=client, run_id="run-id")
    report.holdout_reference = saved.save_holdout_reference(run, frame, spec, report)
    return client, run, report, spec, frame, root / "lifecycle/data/holdout.parquet"


def test_reference_reuses_lifecycle_bytes_and_preserves_context(saved_reference):
    """The reference must reuse exact existing Parquet and all split attrs."""
    client, run, report, spec, frame, _ = saved_reference
    first = deepcopy(report.holdout_reference)
    assert saved.save_holdout_reference(run, frame, spec, report) == first
    assert client.log_artifact.call_count == 1
    actual = saved.load_saved_holdout(client, None, report, spec)
    pd.testing.assert_frame_equal(actual, frame)
    assert actual.attrs == frame.attrs


@pytest.mark.parametrize("prepared", [False, True])
@pytest.mark.parametrize(
    "conversions",
    [(), ({"column": "x", "source_dtype": "decimal(10,2)", "target_dtype": "float64"},)],
)
def test_spark_conversion_context_roundtrips_as_json(saved_reference, prepared, conversions):
    """Fresh Spark tuples and saved Parquet lists must bind the same conversion evidence."""
    client, run, report, spec, frame, path = saved_reference
    frame.attrs["source_column_conversions"] = conversions
    original = deepcopy(frame.attrs)
    path.unlink()
    if prepared:
        saved.save_frame_artifact(client, run.run_id, "holdout", frame)
    uploads = client.log_artifact.call_count
    report.holdout_reference = saved.save_holdout_reference(run, frame, spec, report)
    actual = saved.load_saved_holdout(client, None, report, spec)
    pd.testing.assert_frame_equal(actual, frame)
    assert actual.attrs == json.loads(json.dumps(original))
    assert report.holdout_reference["frame"]["attrs"] == actual.attrs
    assert frame.attrs == original
    assert client.log_artifact.call_count == uploads + (not prepared)


def test_changed_spark_conversion_context_still_rejects_existing_holdout(saved_reference):
    """JSON normalization must not accept a conversion that the prepared rows never used."""
    client, run, report, spec, frame, path = saved_reference
    frame.attrs["source_column_conversions"] = ()
    path.unlink()
    saved.save_frame_artifact(client, run.run_id, "holdout", frame)
    frame.attrs["source_column_conversions"] = (
        {"column": "x", "source_dtype": "decimal(10,2)", "target_dtype": "float64"},
    )
    before = path.read_bytes()
    with pytest.raises(ValueError, match="schema or split context"):
        saved.save_holdout_reference(run, frame, spec, report)
    assert path.read_bytes() == before


@pytest.mark.parametrize(
    "change",
    ["run", "path", "split", "dataset", "model", "bytes", "rows", "schema", "attrs", "hash"],
)
def test_tampered_reference_rejects_before_decode(saved_reference, monkeypatch, change):
    """Every comparison binding remains load-bearing when bytes themselves are intact."""
    client, _, report, spec, _, _ = saved_reference
    ref = report.holdout_reference
    if change in {"run", "path", "split", "dataset", "model"}:
        field = {
            "run": "run_id",
            "path": "artifact_path",
            "split": "holdout_key_sha256",
            "dataset": "dataset_id",
            "model": "model_digest",
        }[change]
        ref[field] = "wrong"
    elif change in {"bytes", "rows"}:
        ref["frame"][change] += 1
    elif change == "schema":
        ref["frame"]["arrow_schema_sha256"] = "0" * 64
    elif change == "attrs":
        ref["frame"]["attrs"]["holdout_key_sha256"] = "wrong"
    else:
        ref["frame"]["sha256"] = "0" * 64
    decode = Mock(side_effect=AssertionError("Decoded unverified holdout"))
    monkeypatch.setattr(saved.pd, "read_parquet", decode)
    with pytest.raises(ValueError):
        saved.load_saved_holdout(client, None, report, spec)
    decode.assert_not_called()


@pytest.mark.parametrize(
    "change",
    ["missing", "corrupt", "source_recreated", "feature_recreated", "row_budget", "byte_budget"],
)
def test_changed_storage_identity_and_tightened_limits_fail_closed(
    saved_reference, monkeypatch, change
):
    """Recreated tables, missing bytes and tightened limits must stop approval."""
    client, _, report, spec, _, path = saved_reference
    if change == "missing":
        path.unlink()
    elif change == "corrupt":
        data = bytearray(path.read_bytes())
        data[10] ^= 1
        path.write_bytes(data)
    elif change.endswith("recreated"):
        monkeypatch.setattr(
            saved,
            "table_identity",
            lambda spark, table: (
                "new-id"
                if (table == spec.table) == (change == "source_recreated")
                else "source-id"
                if table == spec.table
                else "features-id"
            ),
        )
    elif change == "row_budget":
        spec = replace(spec, max_rows=1)
    else:
        spec = replace(spec, max_bytes=1)
    with pytest.raises(ValueError):
        saved.load_saved_holdout(client, None, report, spec)


def test_missing_serializable_history_rejected_at_creation(saved_reference):
    """Required dataframe history must never disappear from saved evidence."""
    _, run, _, _, frame, _ = saved_reference
    frame.attrs["history"] = pd.DataFrame({"x": [0.0]})
    with pytest.raises(ValueError, match="history cannot be omitted"):
        saved.save_frame_artifact(run.client, run.run_id, "history", frame)


def test_reference_is_bound_to_comparison_and_preserved_by_fresh_promotion(monkeypatch):
    """New references change evidence digests while absent references retain legacy bytes."""
    from skyulf.integrations.mlflow.lifecycle import promotion, validation

    legacy = validation.ModelComparisonReport(
        dataset_id="dataset",
        row_count=2,
        code_version="version",
        model_name="model",
        candidate_version="2",
        candidate_digest="candidate",
        champion_version="1",
        champion_digest="champion",
        metric="heldout_rmse",
        metric_direction="minimize",
        min_improvement=0.1,
        quality_threshold=10.0,
        candidate_metrics={"heldout_rmse": 1.0},
        champion_metrics={"heldout_rmse": 2.0},
        improvement=1.0,
        eligible=True,
        reason="candidate_improved",
    )
    assert "holdout_reference" not in validation.comparison_payload(legacy)
    saved_report = replace(legacy, holdout_reference={"version": 1, "sha256": "a" * 64})
    assert validation.comparison_digest(saved_report) != validation.comparison_digest(legacy)
    monkeypatch.setattr(promotion, "resolve_model", Mock())
    monkeypatch.setattr(promotion, "compare_registered_pipeline_models", Mock(return_value=legacy))
    result = promotion._validated_report(
        saved_report,
        pd.DataFrame({"x": [1.0, 2.0], "target": [2.0, 4.0]}),
        target_column="target",
        expected_champion_version="1",
        max_rows=20,
        max_bytes=1024 * 1024,
        tracking_uri=None,
        registry_uri=None,
    )
    assert result == validation.comparison_digest(saved_report)


@pytest.mark.parametrize("engine", ["pandas", "polars"])
def test_typed_temporal_context_roundtrips_without_reordering(tmp_path, engine):
    """Nullable and zoned features must preserve dtype and order for offline scoring."""
    import polars as pl

    frame = pd.DataFrame(
        {
            "group": pd.Series([2, 1, 2], dtype="Int64"),
            "event": pd.to_datetime(["2026-01-03", "2026-01-01", "2026-01-02"], utc=True),
            "x": pd.Series([1.0, None, 3.0], dtype="Float64"),
        }
    )
    frame.attrs["history"] = {
        "mode": "whole_frame",
        "group_column": "group",
        "event_column": "event",
    }
    run = SimpleNamespace(client=Mock(), run_id="run")
    run.client.log_artifact.side_effect = lambda run, path, directory: (
        tmp_path / "holdout.parquet"
    ).write_bytes(Path(path).read_bytes())
    receipt = saved.save_frame_artifact(run.client, run.run_id, "holdout", frame)
    actual = saved._read_parquet(
        tmp_path / "holdout.parquet", receipt, SimpleNamespace(max_rows=3, max_bytes=1024 * 1024)
    )
    if engine == "pandas":
        pd.testing.assert_frame_equal(frame, actual)
        assert actual.attrs == frame.attrs
    else:
        assert pl.from_pandas(actual).equals(pl.from_pandas(frame))


def test_tightened_memory_budget_rejects_dictionary_expansion_before_decode(
    saved_reference, monkeypatch
):
    """Parquet dictionary compression must not hide allocations above the approval budget."""
    client, run, report, spec, frame, path = saved_reference
    path.unlink()
    attrs = frame.attrs
    frame = pd.DataFrame({"x": ["x" * 10000] * 1000, "target": [1.0] * 1000})
    frame.attrs = attrs
    spec = replace(spec, max_rows=1000, max_bytes=20 * 1024 * 1024)
    report.row_count = 1000
    report.holdout_reference = saved.save_holdout_reference(run, frame, spec, report)
    spec = replace(spec, max_bytes=100000)
    decode = Mock(side_effect=AssertionError("Allocated oversized decoded frame"))
    monkeypatch.setattr(saved.pd, "read_parquet", decode)
    with pytest.raises(ValueError, match="memory|byte"):
        saved.load_saved_holdout(client, None, report, spec)
    decode.assert_not_called()


def test_first_champion_keeps_saved_reference_in_committed_receipt(monkeypatch):
    """Bootstrap must compare fresh metrics while committing the bound holdout digest."""
    from skyulf.integrations.mlflow.lifecycle import promotion, validation

    fresh = validation.ModelComparisonReport(
        dataset_id="dataset",
        row_count=2,
        code_version="version",
        model_name="model",
        candidate_version="1",
        candidate_digest="candidate",
        champion_version=None,
        champion_digest=None,
        metric="heldout_rmse",
        metric_direction="minimize",
        min_improvement=0.1,
        quality_threshold=10.0,
        candidate_metrics={"heldout_rmse": 1.0},
        champion_metrics=None,
        improvement=None,
        eligible=False,
        reason="no_champion",
    )
    report = replace(fresh, holdout_reference={"version": 1, "sha256": "a" * 64})
    client = Mock()
    client.get_registered_model.return_value = SimpleNamespace(tags={})
    monkeypatch.setattr(promotion, "resolve_model", Mock())
    monkeypatch.setattr(promotion, "compare_registered_pipeline_models", Mock(return_value=fresh))
    monkeypatch.setattr(promotion, "make_registry_client", lambda *args: client)
    monkeypatch.setattr(promotion, "assert_not_rejected", Mock())
    monkeypatch.setattr(promotion, "read_optional_alias", Mock(return_value=None))
    monkeypatch.setattr(promotion, "_initial_alias_updates", Mock(return_value={}))
    commit = Mock()
    monkeypatch.setattr(promotion, "commit_change", commit)
    receipt = promotion.initialize_champion(
        report,
        pd.DataFrame({"x": [1.0, 2.0], "target": [2.0, 4.0]}),
        target_column="target",
        admission=ExclusiveAliasWriterAdmission(),
        max_rows=20,
        max_bytes=1024 * 1024,
    )
    assert receipt.kind == "initial"
    assert receipt.comparison_sha256 == validation.comparison_digest(report)
    assert commit.call_args.args[1] == receipt
