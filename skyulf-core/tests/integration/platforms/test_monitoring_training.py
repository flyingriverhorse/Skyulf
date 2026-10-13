"""Bound saved training summaries without inventing model or dataset ownership."""

import hashlib
import importlib.util
import json
from copy import deepcopy
from pathlib import Path
from types import SimpleNamespace

import pytest


class SavedClient:
    """Model only the bounded artifact and run reads used by the collector."""

    def __init__(self):
        """Keep independent bytes, declared sizes and read failures for boundary probes."""
        self.documents = {}
        self.sizes = {}
        self.runs = {}
        self.denied = set()
        self.downloaded = []
        self.run_reads = []

    def save(self, run_id, path, document):
        """Serialize actual artifact bytes while retaining malformed-byte test support."""
        content = document if isinstance(document, bytes) else json.dumps(document).encode()
        self.documents[run_id, path] = content
        self.sizes[run_id, path] = len(content)

    def list_artifacts(self, run_id, path=None):
        """List immediate file metadata without traversing nested artifact directories."""
        if (run_id, path) in self.denied:
            raise PermissionError("private token must never enter evidence")
        return [
            SimpleNamespace(path=name, is_dir=False, file_size=self.sizes[owner, name])
            for owner, name in self.documents
            if owner == run_id and str(Path(name).parent).replace("\\", "/") == (path or ".")
        ]

    def download_artifacts(self, run_id, path, dst_path):
        """Write only the explicitly requested fixture file into the disposable directory."""
        self.downloaded.append((run_id, path))
        destination = Path(dst_path) / path
        destination.parent.mkdir(parents=True, exist_ok=True)
        destination.write_bytes(self.documents[run_id, path])
        return str(destination)

    def get_run(self, run_id):
        """Read a concrete run or preserve a permission failure without registry access."""
        self.run_reads.append(run_id)
        value = self.runs[run_id]
        if isinstance(value, Exception):
            raise value
        return value


def _run(run_id, parent=None, candidate=None):
    """Retain real MLflow run structure including optional native Inputs."""
    tags = {}
    if parent:
        tags["mlflow.parentRunId"] = parent
    if candidate:
        tags["competition_candidate"] = candidate
    return SimpleNamespace(
        info=SimpleNamespace(
            run_id=run_id, experiment_id="17", status="FINISHED", start_time=1000, end_time=4500
        ),
        data=SimpleNamespace(tags=tags, params={}),
        inputs=SimpleNamespace(dataset_inputs=[]),
    )


@pytest.fixture
def evidence():
    """Bind saved source version zero and exact membership to one registered model."""
    client = SavedClient()
    model = SimpleNamespace(run_id="parent", version="7", name="workspace.models.example")
    artifact = SimpleNamespace(
        manifest=SimpleNamespace(pipeline_sha256="winner-digest"),
        pipeline=SimpleNamespace(config={"modeling": {"type": "ridge_regression"}}),
    )
    run = _run("parent")
    client.runs["parent"] = run
    spec = {
        "table": "workspace.data.training",
        "version": 0,
        "input_columns": ["x"],
        "target_column": "target",
        "feature_binding_json": None,
    }
    filters = {
        "training_rows": 8,
        "holdout_rows": 2,
        "train_key_sha256": "train-members",
        "holdout_key_sha256": "holdout-members",
    }
    provenance = {
        "version": 1,
        "source_table_id": "delta-id",
        "source_uc_table_id": "uc-id",
        "training_spec": deepcopy(spec),
        "training_evidence": deepcopy(filters),
        "source_evidence": {"dataset_id": "dataset-id"},
        "training_selection": None,
        "feature_lookup": None,
    }
    client.save("parent", "dataset_inputs.json", provenance)
    client.save(
        "parent",
        "training_parameters.json",
        {
            "model_type": "ridge_regression",
            "model_params": {"alpha": 1.0, "random_state": None, "fit_intercept": True},
            "ensemble": {},
            "split": {"strategy": "random", "group_column": None, "random_state": 42},
            "preprocessing_recipe": "numeric",
            "pre_split_recipe": "inline",
            "preprocessing_steps": ["scale"],
            "pre_split_steps": [],
            "secret_source": "never expose me",
        },
    )
    return client, model, artifact, run, spec, filters, provenance


def _capture(evidence):
    """Exercise the public collector with an already verified monitoring model."""
    assert (
        importlib.util.find_spec(
            "skyulf.integrations.databricks.observability.monitoring.monitoring_training"
        )
        is not None
    )
    from skyulf.integrations.databricks.observability.monitoring.monitoring_training import (
        capture_training_summary,
    )

    return capture_training_summary(*evidence[:6])


def _native_inputs(run, provenance):
    """Attach genuine-shaped Inputs whose provenance digest binds the exact JSON artifact."""
    digest = hashlib.sha256(
        json.dumps(
            provenance, sort_keys=True, separators=(",", ":"), ensure_ascii=False, allow_nan=False
        ).encode()
    ).hexdigest()
    spec = provenance["training_spec"]
    source = {
        "delta_table_name": spec["table"],
        "delta_table_version": spec["version"],
        "delta_table_id": provenance["source_uc_table_id"],
        "is_databricks_uc_table": True,
        "skyulf": {
            "source_table_id": provenance["source_table_id"],
            "dataset_id": "dataset-id",
            "provenance_artifact": "dataset_inputs.json",
            "provenance_sha256": digest,
        },
    }
    run.inputs.dataset_inputs = [
        SimpleNamespace(
            dataset=SimpleNamespace(
                name=f"{spec['table']}@0/{role}",
                source_type="delta_table",
                source=json.dumps(source),
                digest=f"{role}-digest",
                profile=json.dumps({"num_rows": count}),
            ),
            tags=[SimpleNamespace(key="mlflow.data.context", value=role)],
        )
        for role, count in (("training", 8), ("holdout", 2))
    ]


def _competition(evidence):
    """Save a lower-loss winner on the parent with distinct candidate child ownership."""
    client, _, _, run, _, _, _ = evidence
    run.data.tags["competition_winner"] = "winner"
    rows = []
    for name, score, digest in (("winner", 1.0, "winner-digest"), ("loser", 3.0, "loser-digest")):
        child = _run(name, parent="parent", candidate=name)
        child.data.params["model_digest"] = digest
        client.runs[name] = child
        rows.append(
            {
                "candidate": name,
                "model_type": "ridge_regression",
                "strategy": "fixed",
                "run_id": name,
                "model_uri": f"runs:/{name}/model",
                "model_digest": digest,
                "config_sha256": f"{name}-config",
                "metric": "heldout_rmse",
                "scoring_metric": "neg_root_mean_squared_error",
                "direction": "minimize",
                "mean": score,
                "std": 0.0,
                "fold_scores": [score, score],
                "fold_membership_sha256": "same-folds",
                "evaluation_mode": "fixed_cv",
                "unbiased_estimate": True,
                "split_policy": {"method": "k_fold"},
                "folds": [],
            }
        )
    selection = {
        "winner": "winner",
        "selection_metric": "heldout_rmse",
        "direction": "minimize",
        "tie_rule": "exact score ties use ascending candidate name",
        "candidate_count": 2,
        "leaderboard": rows,
    }
    client.save("parent", "competition/selection.json", selection)
    return selection


def test_single_candidate_has_no_fabricated_competition_score(evidence):
    """Configured models remain visible without inventing competition evaluation evidence."""
    summary = _capture(evidence)
    row = summary["candidates"]["rows"][0]
    assert summary["format_version"] == 1
    assert summary["candidates"]["status"] == "recorded"
    assert row["registered_version"] == "7" and row["selected"] is True
    assert row["score"] is None and row["metric"] is None
    assert row["unbiased_estimate"] is None
    assert row["duration_seconds"] == 3.5
    assert row["source"] == "configured_candidate"


def test_fitted_parameters_keep_defaults_and_explicit_nulls(evidence):
    """Readable fitted constructor defaults must not become absent or leak other payloads."""
    section = _capture(evidence)["parameters"]
    values = {(row["section"], row["name"]): row["value"] for row in section["rows"]}
    assert section["status"] == "recorded"
    assert values["model_params", "random_state"] == "null"
    assert values["model_params", "fit_intercept"] == "true"
    assert values["split", "random_state"] == "42"
    assert "never expose me" not in json.dumps(section)


def test_dataset_links_require_real_inputs_and_preserve_version_zero(evidence):
    """Saved lineage without native Inputs must not acquire a fabricated MLflow link."""
    rows = _capture(evidence)["datasets"]["rows"]
    assert [row["role"] for row in rows] == ["training", "holdout"]
    assert rows[0]["source_version"] == 0 and rows[0]["row_count"] == 8
    assert rows[0]["delta_table_id"] == "delta-id" and rows[0]["uc_table_id"] == "uc-id"
    assert rows[0]["membership_digest"] == "train-members"
    assert rows[0]["run_id"] is None and rows[0]["dataset_digest"] is None
    assert rows[0]["inputs_status"] == "not_recorded"


def test_real_parent_inputs_bind_to_the_saved_source(evidence):
    """A matching native Input yields the actual run link and native dataset digest."""
    _native_inputs(evidence[3], evidence[6])
    rows = _capture(evidence)["datasets"]["rows"]
    assert rows[0]["run_id"] == "parent" and rows[0]["experiment_id"] == "17"
    assert rows[0]["dataset_digest"] == "training-digest"
    assert rows[0]["inputs_status"] == "recorded"


def test_competition_parent_uses_winning_child_inputs_only(evidence):
    """Copied winner JSON cannot turn parent registration into native parent Input ownership."""
    _competition(evidence)
    _native_inputs(evidence[0].runs["winner"], evidence[6])
    summary = _capture(evidence)
    winner, loser = summary["candidates"]["rows"]
    assert winner["registered_version"] == "7" and loser["registered_version"] is None
    assert winner["score"] == 1.0 and loser["selected"] is False
    assert winner["selection_reason"] and loser["selection_reason"]
    assert summary["datasets"]["rows"][0]["run_id"] == "winner"
    assert summary["datasets"]["rows"][0]["inputs_status"] == "recorded"


@pytest.mark.parametrize("change", ["digest", "parent", "candidate", "winner", "duplicate"])
def test_competition_rejects_mismatched_identity(evidence, change):
    """An unrelated candidate must never appear to explain the registered model."""
    selection = _competition(evidence)
    if change == "digest":
        selection["leaderboard"][0]["model_digest"] = "other-model"
    elif change == "parent":
        evidence[0].runs["winner"].data.tags["mlflow.parentRunId"] = "other-parent"
    elif change == "candidate":
        evidence[0].runs["winner"].data.tags["competition_candidate"] = "other-candidate"
    elif change == "winner":
        selection["winner"] = "loser"
    else:
        selection["leaderboard"][1] = deepcopy(selection["leaderboard"][0])
    evidence[0].save("parent", "competition/selection.json", selection)
    summary = _capture(evidence)
    assert summary["candidates"]["status"] == "invalid"
    assert summary["candidates"]["rows"] == []
    assert summary["parameters"]["status"] == "recorded"


def test_missing_competition_receipt_does_not_become_single_candidate(evidence):
    """A competition tag cannot substantiate a configured-single-model explanation."""
    evidence[3].data.tags["competition_winner"] = "winner"
    assert _capture(evidence)["candidates"]["status"] == "not_recorded"


@pytest.mark.parametrize("path", ["training_parameters.json", "dataset_inputs.json"])
def test_missing_artifact_is_explicit_and_other_sections_survive(evidence, path):
    """Old runs without optional summaries remain valid monitoring sources."""
    del evidence[0].documents["parent", path]
    section = "parameters" if path.startswith("training_parameters") else "datasets"
    summary = _capture(evidence)
    assert summary[section]["status"] == "not_recorded"
    assert summary[section]["rows"] == []
    assert summary["candidates"]["status"] == "recorded"


def test_permission_failure_is_redacted_and_independent(evidence):
    """Optional metadata denial cannot expose exception text or hide configured model identity."""
    evidence[0].denied.add(("parent", None))
    summary = _capture(evidence)
    assert summary["parameters"]["status"] == "unavailable"
    assert summary["datasets"]["status"] == "unavailable"
    assert "private token" not in json.dumps(summary)
    assert summary["candidates"]["status"] == "recorded"


@pytest.mark.parametrize("declared", [True, False])
def test_oversized_artifacts_are_bounded_before_and_after_download(evidence, declared):
    """Both advertised and actual artifact size must cap optional evidence reads."""
    client = evidence[0]
    client.save("parent", "training_parameters.json", b" " * (1024 * 1024 + 1))
    if not declared:
        client.sizes["parent", "training_parameters.json"] = 1
    summary = _capture(evidence)
    assert summary["parameters"]["status"] == "too_large"
    assert (("parent", "training_parameters.json") in client.downloaded) is not declared


@pytest.mark.parametrize("content", [b"[]", b"{bad", b'{"model_params":{"alpha":NaN}}'])
def test_malformed_parameters_do_not_hide_valid_lineage(evidence, content):
    """Non-object, malformed and nonfinite artifacts cannot be accepted as display metadata."""
    evidence[0].save("parent", "training_parameters.json", content)
    summary = _capture(evidence)
    assert summary["parameters"]["status"] == "invalid"
    assert summary["datasets"]["status"] == "recorded"


@pytest.mark.parametrize("kind", ["spec", "membership"])
def test_dataset_receipt_must_match_verified_training_evidence(evidence, kind):
    """Saved lineage from another split cannot be attached to a pinned monitored model."""
    provenance = deepcopy(evidence[6])
    if kind == "spec":
        provenance["training_spec"]["version"] = 1
    else:
        provenance["training_evidence"]["train_key_sha256"] = "different"
    evidence[0].save("parent", "dataset_inputs.json", provenance)
    assert _capture(evidence)["datasets"]["status"] == "invalid"


def test_mismatched_native_input_does_not_get_a_link(evidence):
    """A source mismatch invalidates only the optional native Input link."""
    _native_inputs(evidence[3], evidence[6])
    source = json.loads(evidence[3].inputs.dataset_inputs[0].dataset.source)
    source["delta_table_version"] = 1
    evidence[3].inputs.dataset_inputs[0].dataset.source = json.dumps(source)
    summary = _capture(evidence)
    assert summary["datasets"]["status"] == "recorded"
    assert summary["datasets"]["rows"][0]["inputs_status"] == "invalid"
    assert summary["datasets"]["rows"][0]["run_id"] is None


def test_model_set_component_never_traverses_parent(evidence):
    """An already pinned component uses its own artifacts and independent target population."""
    evidence[3].data.tags.update(
        {"mlflow.parentRunId": "set-parent", "skyulf.training.branch": "amount"}
    )
    _native_inputs(evidence[3], evidence[6])
    summary = _capture(evidence)
    assert summary["candidates"]["rows"][0]["candidate"] == "amount"
    assert summary["datasets"]["rows"][0]["run_id"] == "parent"
    assert all(run_id == "parent" for run_id, _ in evidence[0].downloaded)


@pytest.mark.parametrize("kind", ["rows", "bytes"])
def test_summary_output_is_bounded(evidence, kind):
    """Large constructor maps cannot inflate the shared monitoring evidence document."""
    params = {str(index): index for index in range(257)}
    if kind == "bytes":
        params = {"long": "x" * (64 * 1024)}
    evidence[0].save("parent", "training_parameters.json", {"model_params": params})
    assert _capture(evidence)["parameters"]["status"] == "too_large"


def test_feature_snapshot_rows_do_not_claim_native_inputs(evidence):
    """Pinned feature tables are distinct sources with unknown population size and no run link."""
    binding = {
        "version": 1,
        "lookup_spec": {"lookups": [{"table_name": "workspace.features.history"}]},
        "lookup_evidence": {
            "feature_tables": [
                {
                    "table_name": "workspace.features.history",
                    "table_id": "feature-delta",
                    "version": 0,
                }
            ]
        },
    }
    evidence[4]["feature_binding_json"] = json.dumps(binding)
    evidence[6]["training_spec"] = deepcopy(evidence[4])
    evidence[6]["feature_lookup"] = binding
    evidence[0].save("parent", "dataset_inputs.json", evidence[6])
    rows = _capture(evidence)["datasets"]["rows"]
    feature = rows[2]
    assert feature["role"] == "feature" and feature["source_version"] == 0
    assert feature["delta_table_id"] == "feature-delta" and feature["uc_table_id"] is None
    assert feature["row_count"] is None and feature["run_id"] is None
    assert json.loads(rows[0]["feature_binding"])["lookup_evidence"] == binding["lookup_evidence"]
    assert all(row["feature_binding"] is None for row in rows[1:])


@pytest.mark.parametrize("change", ["experiment", "run_digest", "boolean_score", "boolean_count"])
def test_competition_requires_exact_child_and_score_types(evidence, change):
    """Equal-looking booleans and unrelated child runs cannot pass candidate provenance checks."""
    selection = _competition(evidence)
    child = evidence[0].runs["winner"]
    if change == "experiment":
        child.info.experiment_id = "other-experiment"
    elif change == "run_digest":
        child.data.params["model_digest"] = "other-model"
    elif change == "boolean_score":
        selection["leaderboard"][0]["mean"] = True
        selection["leaderboard"][0]["fold_scores"] = [True, True]
    else:
        selection["candidate_count"] = True
        selection["leaderboard"] = selection["leaderboard"][:1]
    evidence[0].save("parent", "competition/selection.json", selection)
    assert _capture(evidence)["candidates"]["status"] == "invalid"


@pytest.mark.parametrize("change", ["profile_bool", "name", "uc_marker"])
def test_native_input_identity_types_are_exact(evidence, change):
    """A native Input link requires the actual population name, count and UC source marker."""
    _native_inputs(evidence[3], evidence[6])
    dataset = evidence[3].inputs.dataset_inputs[0].dataset
    if change == "profile_bool":
        evidence[5]["training_rows"] = 1
        evidence[6]["training_evidence"]["training_rows"] = 1
        evidence[0].save("parent", "dataset_inputs.json", evidence[6])
        _native_inputs(evidence[3], evidence[6])
        dataset = evidence[3].inputs.dataset_inputs[0].dataset
        dataset.profile = json.dumps({"num_rows": True})
    elif change == "name":
        dataset.name = "another population"
    else:
        source = json.loads(dataset.source)
        source["is_databricks_uc_table"] = False
        dataset.source = json.dumps(source)
    row = _capture(evidence)["datasets"]["rows"][0]
    assert row["inputs_status"] == "invalid" and row["run_id"] is None


def test_missing_advertised_size_prevents_download(evidence):
    """Unknown remote size must not authorize an unbounded optional artifact download."""
    evidence[0].sizes["parent", "training_parameters.json"] = None
    section = _capture(evidence)["parameters"]
    assert section["status"] == "unavailable"
    assert ("parent", "training_parameters.json") not in evidence[0].downloaded


def test_child_permission_denial_does_not_remove_saved_dataset_rows(evidence):
    """Candidate-run permissions cannot hide independently verified parent provenance."""
    _competition(evidence)
    evidence[0].runs["winner"] = PermissionError("secret token")
    summary = _capture(evidence)
    assert summary["candidates"]["status"] == "unavailable"
    assert summary["datasets"]["status"] == "recorded"
    assert summary["datasets"]["rows"][0]["run_id"] is None
    assert "secret token" not in json.dumps(summary)


def test_missing_run_metadata_does_not_invent_single_model_or_input_links(evidence):
    """Unreadable run tags cannot distinguish old single training from an incomplete competition."""
    summary = _capture((*evidence[:3], None, *evidence[4:]))
    assert summary["candidates"]["status"] == "unavailable"
    assert summary["parameters"]["status"] == "recorded"
    assert summary["datasets"]["status"] == "recorded"
    assert summary["datasets"]["rows"][0]["inputs_status"] == "unavailable"


def test_missing_experiment_is_null_instead_of_a_fake_link(evidence):
    """Optional experiment metadata cannot become the literal navigation identity None."""
    evidence[3].info.experiment_id = None
    row = _capture(evidence)["candidates"]["rows"][0]
    assert row["experiment_id"] is None


def test_absent_parent_inputs_allow_verified_winner_fallback(evidence):
    """MLflow represents absent Inputs as None, which must preserve child-owned lineage links."""
    _competition(evidence)
    evidence[3].inputs = None
    _native_inputs(evidence[0].runs["winner"], evidence[6])
    rows = _capture(evidence)["datasets"]["rows"]
    assert all(row["run_id"] == "winner" and row["inputs_status"] == "recorded" for row in rows)


@pytest.mark.parametrize("inputs, expected", [(None, "not_recorded"), ({}, "invalid")])
def test_absent_and_malformed_inputs_have_distinct_statuses(evidence, inputs, expected):
    """Missing native Inputs are ordinary absence, while malformed present metadata stays invalid."""
    evidence[3].inputs = inputs
    row = _capture(evidence)["datasets"]["rows"][0]
    assert row["inputs_status"] == expected
    assert row["run_id"] is None


@pytest.mark.parametrize("field", ["source_table_id", "source_uc_table_id"])
@pytest.mark.parametrize("value", [{"private": "do not expose"}, "x" * 1025, 7])
def test_source_table_ids_require_bounded_strings(evidence, field, value):
    """Dataset identifier columns must not carry nested payloads or oversized provenance values."""
    provenance = deepcopy(evidence[6])
    provenance[field] = value
    evidence[0].save("parent", "dataset_inputs.json", provenance)
    section = _capture(evidence)["datasets"]
    assert section["status"] == "invalid" and section["rows"] == []
    assert "do not expose" not in json.dumps(section)


def test_null_source_table_ids_stay_explicitly_null(evidence):
    """Historical provenance without captured table identities remains visibly incomplete."""
    provenance = deepcopy(evidence[6])
    provenance.update(source_table_id=None, source_uc_table_id=None)
    evidence[0].save("parent", "dataset_inputs.json", provenance)
    row = _capture(evidence)["datasets"]["rows"][0]
    assert row["delta_table_id"] is None and row["uc_table_id"] is None


@pytest.mark.parametrize("length, expected", [(10000, "recorded"), (120000, "too_large")])
def test_section_budget_matches_actual_store_json_encoding(evidence, length, expected):
    """ASCII escaping in persisted evidence cannot expand an accepted section past its byte cap."""
    from skyulf.integrations.databricks.observability.monitoring.monitoring_store import (
        _json as store_json,
    )

    evidence[0].save(
        "parent", "training_parameters.json", {"model_params": {"label": "é" * length}}
    )
    section = _capture(evidence)["parameters"]
    assert section["status"] == expected
    assert len(store_json(section).encode("utf-8")) <= 64 * 1024


@pytest.mark.parametrize("count, status", [(32, "recorded"), (33, "too_large")])
def test_competition_bound_is_enforced_before_child_run_reads(evidence, count, status):
    """The producer's candidate cap bounds remote calls before inspecting any child run."""
    selection = _competition(evidence)
    for index in range(2, count):
        name = f"candidate_{index}"
        row = deepcopy(selection["leaderboard"][1])
        row.update(candidate=name, run_id=name, model_uri=f"runs:/{name}/model")
        selection["leaderboard"].append(row)
        child = _run(name, parent="parent", candidate=name)
        child.data.params["model_digest"] = row["model_digest"]
        evidence[0].runs[name] = child
    selection["leaderboard"][1:] = sorted(
        selection["leaderboard"][1:], key=lambda row: row["candidate"]
    )
    selection["candidate_count"] = count
    evidence[0].save("parent", "competition/selection.json", selection)
    section = _capture(evidence)["candidates"]
    assert section["status"] == status
    assert len(evidence[0].run_reads) == (32 if count == 32 else 0)


def test_artifact_listing_budget_is_separate_from_display_rows(evidence):
    """An ordinary run with many artifacts must not lose its small allowlisted summary."""
    for index in range(300):
        evidence[0].save("parent", f"unused_{index}.json", {})
    section = _capture(evidence)["parameters"]
    assert section["status"] == "recorded"
    assert not any(path.startswith("unused_") for _, path in evidence[0].downloaded)
