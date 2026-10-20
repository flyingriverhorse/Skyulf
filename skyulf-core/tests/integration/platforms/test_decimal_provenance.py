"""Observed Decimal conversions stay inspectable without weakening input contracts."""

import json
from copy import deepcopy

import pandas as pd
import pytest
from test_project_source_validation import _config, _query
from test_training_summary import ArtifactClient, _documents, _payload

from skyulf.core.schema import SchemaMismatchError
from skyulf.data.dataset import SplitDataset
from skyulf.inference.fitted_pipeline import load_pipeline, save_pipeline
from skyulf.integrations.databricks.observability.reports.training_summary import (
    render_training_summary,
)
from skyulf.integrations.databricks.projects.source_validation import validate_project_sources
from skyulf.integrations.databricks.projects.workflow_config import preview_workflow_config
from skyulf.integrations.mlflow.lifecycle import validation
from skyulf.integrations.mlflow.registration.registry import ResolvedModel
from skyulf.pipeline import SkyulfPipeline

CONVERSION = {
    "column": "x",
    "source_dtype": "decimal(17,6)",
    "target_dtype": "float64",
}


def _fitted(*, dtype="float64", conversions=None):
    """Fit real numeric values under either the legacy or converted input contract."""
    frame = pd.DataFrame(
        {"x": pd.Series([1, 2, 3, 4, 5, 6], dtype=dtype), "y": [2, 4, 6, 8, 10, 12]}
    )
    config = {"preprocessing": [], "modeling": {"type": "linear_regression"}}
    if conversions is not None:
        config["source_column_conversions"] = conversions
    pipeline = SkyulfPipeline(config)
    pipeline.fit(SplitDataset(train=frame, test=frame.head(0)), target_column="y")
    return pipeline, frame


def test_legacy_manifest_omits_unrecorded_conversion_field(tmp_path):
    """Existing JSON and partition proof dictionaries must retain their old shape."""
    pipeline, _ = _fitted()
    save_pipeline(pipeline, tmp_path / "model")
    manifest = load_pipeline(tmp_path / "model").manifest
    serialized = json.loads((tmp_path / "model/manifest.json").read_text())
    assert manifest.source_column_conversions is None
    assert "source_column_conversions" not in manifest.model_dump()
    assert "source_column_conversions" not in serialized
    assert serialized == manifest.model_dump(mode="json")


@pytest.mark.parametrize("conversions", [[], [CONVERSION]])
def test_observed_conversion_manifest_round_trip_is_payload_bound(tmp_path, conversions):
    """Observed empty evidence differs from legacy absence and survives actual saving."""
    pipeline, _ = _fitted(conversions=deepcopy(conversions))
    save_pipeline(pipeline, tmp_path / "model")
    restored = load_pipeline(tmp_path / "model")
    assert restored.manifest.model_dump(mode="json")["source_column_conversions"] == conversions
    assert dict(restored.pipeline.config)["source_column_conversions"] == conversions
    path = tmp_path / "model/manifest.json"
    document = json.loads(path.read_text())
    document["source_column_conversions"] = [] if conversions else [CONVERSION]
    path.write_text(json.dumps(document))
    with pytest.raises(ValueError, match="manifest disagrees"):
        load_pipeline(tmp_path / "model")


def test_wide_decimal_manifest_round_trip_preserves_complete_evidence(tmp_path):
    """Conversion evidence must not make an otherwise bounded wide model unsaveable."""
    columns = [f"feature_{index:03}" for index in range(520)]
    frame = pd.DataFrame({name: [1.0, 2.0, 3.0, 4.0] for name in columns})
    frame["target"] = [2.0, 4.0, 6.0, 8.0]
    conversions = [
        {"column": name, "source_dtype": "decimal(18,2)", "target_dtype": "float64"}
        for name in columns
    ]
    pipeline = SkyulfPipeline(
        {
            "preprocessing": [],
            "modeling": {"type": "linear_regression"},
            "source_column_conversions": conversions,
        }
    )
    pipeline.fit(SplitDataset(train=frame, test=frame.head(0)), target_column="target")

    save_pipeline(pipeline, tmp_path / "wide")
    restored = load_pipeline(tmp_path / "wide")

    assert (tmp_path / "wide/manifest.json").stat().st_size <= 64 * 1024
    assert restored.manifest.model_dump(mode="json")["source_column_conversions"] == conversions
    assert dict(restored.pipeline.config)["source_column_conversions"] == conversions
    assert restored.pipeline.predict(frame[columns]).tolist() == pytest.approx([2.0, 4.0, 6.0, 8.0])
    path = tmp_path / "wide/manifest.json"
    document = json.loads(path.read_text())
    document["source_column_conversions_compact"]["items"].reverse()
    path.write_text(json.dumps(document))
    with pytest.raises(ValueError, match="manifest disagrees"):
        load_pipeline(tmp_path / "wide")


def _compact_manifest(path):
    """Represent existing conversion evidence using the explicit compact wire format."""
    document = json.loads(path.read_text())
    conversions = document.pop("source_column_conversions")
    document["source_column_conversions_compact"] = {
        "version": 1,
        "items": [[item["column"], item["source_dtype"]] for item in conversions],
    }
    return document


@pytest.mark.parametrize(
    "compact",
    [
        None,
        {"version": 1, "items": [], "extra": True},
        {"version": True, "items": []},
        {"version": 2, "items": []},
        {"version": 1, "items": {}},
        {"version": 1, "items": [["x", "decimal(17,6)", "float64"]]},
        {"version": 1, "items": [[True, "decimal(17,6)"]]},
        {"version": 1, "items": [["x", "decimal(17,6)"], ["x", "decimal(17,6)"]]},
        {"version": 1, "items": [["missing", "decimal(17,6)"]]},
        {"version": 1, "items": [["x", "decimal(4,6)"]]},
    ],
)
def test_compact_conversion_manifest_rejects_invalid_evidence_before_unpickle(
    tmp_path, monkeypatch, compact
):
    """Compact records retain strict metadata admission before trusted pickle loading."""
    pipeline, _ = _fitted(conversions=[CONVERSION])
    root = tmp_path / "model"
    save_pipeline(pipeline, root)
    path = root / "manifest.json"
    document = _compact_manifest(path)
    document["source_column_conversions_compact"] = compact
    path.write_text(json.dumps(document))
    monkeypatch.setattr(
        "skyulf.inference.fitted_pipeline.pickle.loads",
        lambda _: pytest.fail("Invalid conversion metadata reached pickle deserialization."),
    )

    with pytest.raises(ValueError):
        load_pipeline(root)


@pytest.mark.parametrize("change", ["dtype", "omission", "ambiguous"])
def test_compact_conversion_manifest_remains_bound_to_payload(tmp_path, change):
    """A shorter representation must not permit rewriting or hiding the saved evidence."""
    pipeline, _ = _fitted(conversions=[CONVERSION])
    root = tmp_path / "model"
    save_pipeline(pipeline, root)
    path = root / "manifest.json"
    document = _compact_manifest(path)
    if change == "dtype":
        document["source_column_conversions_compact"]["items"][0][1] = "decimal(18,2)"
    elif change == "omission":
        document["source_column_conversions_compact"]["items"] = []
    else:
        document["source_column_conversions"] = [CONVERSION]
    path.write_text(json.dumps(document))

    with pytest.raises(ValueError, match="manifest disagrees|Invalid compact"):
        load_pipeline(root)


def test_compact_conversion_manifest_keeps_transport_size_limit(tmp_path):
    """Compact evidence cannot bypass the pre-parse limit by adding excessive JSON bytes."""
    pipeline, _ = _fitted(conversions=[CONVERSION])
    root = tmp_path / "model"
    save_pipeline(pipeline, root)
    path = root / "manifest.json"
    path.write_text(json.dumps(_compact_manifest(path)) + " " * (64 * 1024))

    with pytest.raises(ValueError, match="manifest.json exceeds the size limit"):
        load_pipeline(root)


@pytest.mark.parametrize(
    "conversions",
    [
        [CONVERSION, CONVERSION],
        [CONVERSION | {"column": "missing"}],
        [CONVERSION | {"source_dtype": "double"}],
        [CONVERSION | {"source_dtype": "decimal(39,0)"}],
        [CONVERSION | {"source_dtype": "decimal(4,6)"}],
        [CONVERSION | {"target_dtype": "float32"}],
        [CONVERSION | {"extra": "untrusted"}],
    ],
)
def test_manifest_rejects_malformed_or_unbound_conversion_evidence(tmp_path, conversions):
    """Provenance must describe unique real float64 model inputs and valid Spark Decimal types."""
    pipeline, _ = _fitted(conversions=conversions)
    with pytest.raises(ValueError):
        save_pipeline(pipeline, tmp_path / "model")


def test_manifest_rejects_conversion_evidence_for_object_input(tmp_path):
    """A provenance declaration cannot disguise an unchanged object-typed fit."""
    pipeline, _ = _fitted(dtype="object", conversions=[CONVERSION])
    with pytest.raises(ValueError, match="float64"):
        save_pipeline(pipeline, tmp_path / "model")


def test_training_report_lists_observed_conversions_and_escapes_column_names():
    """The report must expose saved casts without rendering source names as markup."""
    documents = _documents()
    documents["training_data.json"]["source_column_conversions"] = [
        CONVERSION | {"column": "x<script>"}
    ]
    report = render_training_summary(ArtifactClient({"fit": documents}), "fit", _payload())
    assert "x&lt;script&gt;" in report
    assert "decimal(17,6)" in report and "float64" in report
    assert "15-17 significant digits" in report
    assert "training_data.json" in report


@pytest.mark.parametrize("conversions,label", [(None, "not recorded"), ([], "none")])
def test_training_report_distinguishes_legacy_and_observed_no_conversions(conversions, label):
    """Missing historical evidence cannot be presented as proof that no casts occurred."""
    documents = _documents()
    if conversions is not None:
        documents["training_data.json"]["source_column_conversions"] = conversions
    report = render_training_summary(ArtifactClient({"fit": documents}), "fit", _payload())
    assert f"Decimal conversions: {label}" in report


def test_offline_preview_discloses_unknown_decimal_columns(workflow_config):
    """Static config cannot truthfully infer physical source types from feature names."""
    report = preview_workflow_config(workflow_config)
    assert "DECIMAL model inputs are converted to float64" in report
    assert "Converted columns: unknown until the source schema is read" in report
    assert "15-17 significant digits" in report


@pytest.mark.parametrize("key_is_input", [False, True])
def test_source_preview_reports_only_observed_decimal_model_inputs(key_is_input):
    """Explicit online schema checks can show planned casts without claiming training ran."""
    query, _ = _query()

    def decimal_query(sql):
        """Return a source Decimal feature and an unrelated Decimal nonfeature."""
        rows = query(sql)
        if sql.startswith("DESCRIBE TABLE"):
            return [
                row | {"data_type": "decimal(17,6)"} if row["col_name"] in {"id", "x", "y"} else row
                for row in rows
            ]
        return rows

    config = _config()
    if key_is_input:
        config["input_columns"] = ["id", "x"]
    report = validate_project_sources(config, decimal_query)
    for table in ("cat.data.train", "cat.data.score"):
        assert report["tables"][table]["planned_input_conversions"] == [CONVERSION]
    assert any("15-17 significant digits" in warning for warning in report["warnings"])


@pytest.mark.parametrize("control_is_input", [False, True])
def test_source_preview_preserves_decimal_feature_lookup_controls(control_is_input):
    """A native feature lookup identity is never advertised as a floating-point cast."""
    config = _config() | {
        "input_columns": ["x", "fetched"] + (["lookup_id"] if control_is_input else []),
        "feature_lookup": {
            "lookups": [
                {
                    "table_name": "cat.features.values",
                    "lookup_key": ["lookup_id"],
                    "feature_names": ["fetched"],
                }
            ]
        },
    }
    query, _ = _query()

    def lookup_query(sql):
        """Expose a precision-sensitive lookup key distinct from the record key."""
        rows = query(sql)
        if sql.startswith("DESCRIBE TABLE"):
            rows = [*rows, {"col_name": "lookup_id", "data_type": "decimal(20,0)"}]
        return rows

    report = validate_project_sources(config, lookup_query)
    assert all(table["planned_input_conversions"] == [] for table in report["tables"].values())


@pytest.mark.parametrize(
    "recorded,heldout_dtype,migration",
    [(True, "float64", True), (False, "float64", False), (True, "int64", False)],
)
def test_legacy_object_champion_error_is_actionable_only_with_conversion_evidence(
    tmp_path, monkeypatch, recorded, heldout_dtype, migration
):
    """A real schema failure keeps its class/cause; only observed Decimal migration gets advice."""
    candidate, heldout = _fitted(conversions=[CONVERSION] if recorded else None)
    heldout["x"] = heldout["x"].astype(heldout_dtype)
    champion, _ = _fitted(dtype="object")
    artifacts = {}
    references = []
    for version, pipeline in (("2", candidate), ("1", champion)):
        path = tmp_path / version
        save_pipeline(pipeline, path)
        artifacts[version] = load_pipeline(path)
        references.append(
            ResolvedModel(
                "cat.models.decimal",
                version,
                f"models:/cat.models.decimal/{version}",
                None,
                artifacts[version].manifest.pipeline_sha256,
            )
        )
    monkeypatch.setattr(
        validation, "_checked_artifact", lambda reference, **kwargs: artifacts[reference.version]
    )
    with pytest.raises(SchemaMismatchError) as caught:
        validation.compare_registered_pipeline_models(
            references[0],
            references[1],
            heldout,
            target_column="y",
            dataset_id="snapshot-1",
            metric="heldout_rmse",
            min_improvement=0.0,
            max_rows=20,
            max_bytes=10000,
        )
    expected = ("object", "float64") if heldout_dtype == "float64" else ("float64", "int64")
    assert caught.value.dtype_mismatches == {"x": expected}
    if migration:
        assert "new registered model names" in str(caught.value)
        assert "cat.models.decimal" in str(caught.value) and "version 1" in str(caught.value)
        assert isinstance(caught.value.__cause__, SchemaMismatchError)
    else:
        assert "new registered model names" not in str(caught.value)
        assert caught.value.__cause__ is None
