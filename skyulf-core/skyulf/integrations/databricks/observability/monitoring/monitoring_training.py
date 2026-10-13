"""Capture bounded optional training metadata without loading data or changing lifecycle state."""

import hashlib
import json
import math
from collections.abc import Callable
from dataclasses import dataclass, field
from pathlib import Path
from tempfile import TemporaryDirectory
from typing import Any

from ...training.competition.competition import choose_winner

_ARTIFACT_BYTES = 1024 * 1024
_SECTION_BYTES = 64 * 1024
_MAX_ROWS = 256
_MAX_CANDIDATES = 32
_MAX_LIST_ENTRIES = 1000
_PARAMETER_GROUPS = ("model_params", "ensemble", "split", "training_weights")
_PARAMETER_FIELDS = (
    "model_type",
    "preprocessing_recipe",
    "pre_split_recipe",
    "preprocessing_steps",
    "pre_split_steps",
    "configured_class_weight",
)


class _EvidenceIssue(Exception):
    """Carry a bounded public status without propagating remote exception details."""

    def __init__(self, status: str, reason: str) -> None:
        """Store only fixed collector reason codes for optional evidence failures."""
        super().__init__(reason)
        self.status = status
        self.reason = reason


def _invalid() -> None:
    """Reject malformed or mismatched evidence without copying any source content."""
    raise _EvidenceIssue("invalid", "saved_training_evidence_invalid")


def _text(value: Any) -> str:
    """Require bounded nonempty identity text for artifact and run navigation fields."""
    if type(value) is not str or not value or len(value.encode("utf-8")) > 1024:
        _invalid()
    return value


def _integer(value: Any) -> int:
    """Preserve version zero while rejecting booleans and negative counts."""
    if type(value) is not int or value < 0:
        _invalid()
    return value


def _json(value: Any) -> str:
    """Serialize display values deterministically while retaining explicit JSON nulls."""
    return json.dumps(
        value, sort_keys=True, separators=(",", ":"), ensure_ascii=False, allow_nan=False
    )


def _reject_constant(value: str) -> None:
    """Reject nonstandard nonfinite JSON constants before building display evidence."""
    _invalid()


def _parse(content: str) -> dict:
    """Accept object-shaped finite JSON within the artifact byte budget."""
    if len(content.encode("utf-8")) > _ARTIFACT_BYTES:
        raise _EvidenceIssue("too_large", "saved_training_artifact_too_large")
    value = json.loads(content, parse_constant=_reject_constant)
    if not isinstance(value, dict):
        _invalid()
    return value


@dataclass
class _Capture:
    """Share verified invocation identity and only the winner needed for native Input lookup."""

    client: Any
    model: Any
    artifact: Any
    run: Any
    spec: dict
    filters: dict
    winner: Any = None
    candidate_status: str = "not_recorded"
    listings: dict = field(default_factory=dict)

    def document(self, path: str) -> dict:
        """Check advertised bytes before downloading one known JSON artifact."""
        parent = str(Path(path).parent).replace("\\", "/")
        parent = None if parent == "." else parent
        if parent not in self.listings:
            self.listings[parent] = self.client.list_artifacts(self.model.run_id, parent)
        listing = self.listings[parent]
        if len(listing) > _MAX_LIST_ENTRIES:
            raise _EvidenceIssue("too_large", "saved_training_artifact_listing_too_large")
        matches = [item for item in listing if item.path == path]
        if not matches:
            raise _EvidenceIssue("not_recorded", "saved_training_artifact_missing")
        if len(matches) != 1 or matches[0].is_dir:
            _invalid()
        _artifact_size(getattr(matches[0], "file_size", None))
        return _download(self.client, self.model.run_id, path)


def _artifact_size(value: Any) -> None:
    """Refuse an unbounded remote download when its advertised size is missing."""
    if value is None:
        raise _EvidenceIssue("unavailable", "saved_training_artifact_size_unavailable")
    if _integer(value) > _ARTIFACT_BYTES:
        raise _EvidenceIssue("too_large", "saved_training_artifact_too_large")


def _download(client: Any, run_id: str, path: str) -> dict:
    """Read at most one bounded file from the SDK's disposable download directory."""
    with TemporaryDirectory(prefix="skyulf-training-summary-") as directory:
        downloaded = Path(client.download_artifacts(run_id, path, directory)).resolve()
        if not downloaded.is_relative_to(Path(directory).resolve()) or not downloaded.is_file():
            _invalid()
        with downloaded.open("rb") as stream:
            content = stream.read(_ARTIFACT_BYTES + 1)
        if len(content) > _ARTIFACT_BYTES:
            raise _EvidenceIssue("too_large", "saved_training_artifact_too_large")
        return _parse(content.decode("utf-8"))


def _section(read: Callable[[], list[dict]]) -> dict:
    """Isolate optional failures and enforce the complete serialized section budget."""
    try:
        rows = read()
        _bound_rows(rows)
        return {"status": "recorded", "reason": None, "rows": rows}
    except _EvidenceIssue as exc:
        return {"status": exc.status, "reason": exc.reason, "rows": []}
    except (ValueError, TypeError, KeyError, AttributeError, RecursionError):
        return {"status": "invalid", "reason": "saved_training_evidence_invalid", "rows": []}
    except Exception:  # noqa: BLE001 - optional evidence must not invalidate monitoring
        return {
            "status": "unavailable",
            "reason": "saved_training_evidence_unavailable",
            "rows": [],
        }


def _bound_rows(rows: list[dict]) -> None:
    """Check serialized output incrementally before repeated lookup text inflates a section."""
    # Match monitoring_store._json, including ASCII escaping and separator spaces.
    encoder = json.JSONEncoder(sort_keys=True, allow_nan=False)
    size = len(encoder.encode({"status": "recorded", "reason": None, "rows": []}).encode("utf-8"))
    if len(rows) > _MAX_ROWS:
        raise _EvidenceIssue("too_large", "saved_training_summary_too_large")
    for index, row in enumerate(rows):
        size += len(encoder.encode(row).encode("utf-8")) + (2 if index else 0)
        if size > _SECTION_BYTES:
            raise _EvidenceIssue("too_large", "saved_training_summary_too_large")


def capture_training_summary(
    client: Any, model: Any, artifact: Any, run: Any, saved_spec: dict, filters: dict
) -> dict:
    """Describe one pinned model's saved candidates, fitted parameters and dataset provenance.

    ``run`` is the registered version's producing run, or None if its optional
    metadata is unavailable. Model-set callers pass their already verified
    component model. Each section fails independently and no raw dataset, model
    package, mutable alias or unrelated parent is fetched.
    """
    capture = _Capture(client, model, artifact, run, saved_spec, filters)
    candidates = _section(lambda: _candidates(capture))
    capture.candidate_status = candidates["status"]
    if capture.candidate_status != "recorded":
        capture.winner = None
    return {
        "format_version": 1,
        "candidates": candidates,
        "parameters": _section(lambda: _parameters(capture)),
        "datasets": _section(lambda: _datasets(capture)),
    }


def _algorithm(artifact: Any) -> str:
    """Name the concrete model behind an optional hyperparameter search."""
    modeling = artifact.pipeline.config["modeling"]
    if modeling.get("type") == "hyperparameter_tuner":
        modeling = modeling["base_model"]
    return _text(modeling["type"])


def _run_info(run: Any, expected: str) -> dict:
    """Expose navigation only for the expected run and label its full wall duration."""
    if run is None:
        return {"run_id": expected, "experiment_id": None, "duration_seconds": None}
    if run.info.run_id != expected:
        _invalid()
    experiment = getattr(run.info, "experiment_id", None)
    return {
        "run_id": _text(run.info.run_id),
        "experiment_id": None if experiment is None else _text(str(experiment)),
        "duration_seconds": _duration(run),
    }


def _duration(run: Any) -> float | None:
    """Return complete run wall time only when valid start and end timestamps exist."""
    start, end = getattr(run.info, "start_time", None), getattr(run.info, "end_time", None)
    if not isinstance(start, int | float) or not isinstance(end, int | float):
        return None
    if isinstance(start, bool) or isinstance(end, bool):
        return None
    elapsed = (end - start) / 1000
    return elapsed if math.isfinite(elapsed) and elapsed >= 0 else None


def _candidates(capture: _Capture) -> list[dict]:
    """Use a saved competition receipt, otherwise explicitly describe configured training."""
    try:
        selection = capture.document("competition/selection.json")
    except _EvidenceIssue as exc:
        if exc.status != "not_recorded":
            raise
        if capture.run is None:
            raise _EvidenceIssue("unavailable", "training_run_metadata_unavailable") from None
        tags = capture.run.data.tags
        if tags.get("competition_winner"):
            raise
        return [_configured_candidate(capture, tags)]
    return _competition_candidates(capture, selection)


def _configured_candidate(capture: _Capture, tags: dict) -> dict:
    """Avoid inventing a contest, validation score or search outcome for a single model."""
    return {
        "candidate": _text(tags.get("skyulf.training.branch") or _algorithm(capture.artifact)),
        "model_type": _algorithm(capture.artifact),
        "strategy": "configured",
        **_run_info(capture.run, capture.model.run_id),
        "registered_version": _text(str(capture.model.version)),
        "selected": True,
        "metric": None,
        "direction": None,
        "score": None,
        "score_std": None,
        "evaluation_mode": None,
        "unbiased_estimate": None,
        "selection_reason": "Configured training candidate; no competition selection recorded.",
        "source": "configured_candidate",
    }


def _competition_candidates(capture: _Capture, selection: dict) -> list[dict]:
    """Bind complete comparable rankings and selected digest to the monitored version."""
    leaderboard = _ranked_candidates(selection)
    tags = capture.run.data.tags if capture.run is not None else {}
    if tags.get("competition_winner", selection["winner"]) != selection["winner"]:
        _invalid()
    rows = []
    for row in leaderboard:
        child = _competition_child(capture, row)
        selected = row["candidate"] == selection["winner"]
        if selected:
            _selected_identity(capture, row)
            capture.winner = child
        rows.append(_competition_row(capture, selection, row, child, selected))
    return rows


def _ranked_candidates(selection: dict) -> list[dict]:
    """Require bounded, complete rankings whose winner matches the saved selection policy."""
    leaderboard = selection["leaderboard"]
    if not isinstance(leaderboard, list) or not leaderboard:
        _invalid()
    if len(leaderboard) > _MAX_CANDIDATES:
        raise _EvidenceIssue("too_large", "saved_training_summary_too_large")
    _integer(selection["candidate_count"])
    for row in leaderboard:
        _score_types(row)
    names = {_text(row["candidate"]) for row in leaderboard}
    if choose_winner(leaderboard, names) != selection:
        _invalid()
    return leaderboard


def _score_types(row: dict) -> None:
    """Require actual finite numeric scores rather than truthy JSON scalar substitutes."""
    values = [row["mean"], row["std"], *row["fold_scores"]]
    if any(type(value) not in (int, float) or not math.isfinite(value) for value in values):
        _invalid()


def _competition_child(capture: _Capture, row: dict) -> Any:
    """Require the candidate's finished child run to belong to this registered parent."""
    run_id = _text(row["run_id"])
    child = capture.client.get_run(run_id)
    expected = (
        child.info.run_id == run_id,
        child.info.status == "FINISHED",
        child.data.tags.get("mlflow.parentRunId") == capture.model.run_id,
        child.data.tags.get("competition_candidate") == row["candidate"],
        child.data.params.get("model_digest") == row["model_digest"],
        row["model_uri"] == f"runs:/{run_id}/model",
    )
    if not all(expected):
        _invalid()
    if capture.run is not None and child.info.experiment_id != capture.run.info.experiment_id:
        _invalid()
    return child


def _selected_identity(capture: _Capture, row: dict) -> None:
    """Prevent a valid contest from being attached to a different fitted model."""
    if row["model_digest"] != capture.artifact.manifest.pipeline_sha256 or row[
        "model_type"
    ] != _algorithm(capture.artifact):
        _invalid()


def _competition_row(
    capture: _Capture, selection: dict, row: dict, child: Any, selected: bool
) -> dict:
    """Keep score semantics and source ownership explicit for winners and losing candidates."""
    unbiased = row.get("unbiased_estimate")
    if unbiased is not None and type(unbiased) is not bool:
        _invalid()
    action = "Selected" if selected else "Not selected"
    return {
        "candidate": row["candidate"],
        "model_type": _text(row["model_type"]),
        "strategy": _text(row["strategy"]),
        **_run_info(child, row["run_id"]),
        "registered_version": str(capture.model.version) if selected else None,
        "selected": selected,
        "metric": _text(selection["selection_metric"]),
        "direction": selection["direction"],
        "score": float(row["mean"]),
        "score_std": float(row["std"]),
        "evaluation_mode": _text(row["evaluation_mode"]),
        "unbiased_estimate": unbiased,
        "selection_reason": (
            f"{action}: {selection['direction']} {selection['selection_metric']}; "
            f"{selection['tie_rule']}."
        ),
        "source": "competition/selection.json",
    }


def _parameters(capture: _Capture) -> list[dict]:
    """Read fitted defaults and configured recipe fields without dumping arbitrary metadata."""
    saved = capture.document("training_parameters.json")
    rows = []
    for group in _PARAMETER_GROUPS:
        values = saved.get(group, {})
        if not isinstance(values, dict):
            _invalid()
        rows.extend(_parameter_row(group, name, value) for name, value in values.items())
    rows.extend(
        _parameter_row("recipe", name, saved[name]) for name in _PARAMETER_FIELDS if name in saved
    )
    if not rows:
        _invalid()
    if "model_type" in saved and saved["model_type"] != _algorithm(capture.artifact):
        _invalid()
    return rows


def _parameter_row(section: str, name: str, value: Any) -> dict:
    """Preserve explicit scalar and structured constructor values as finite JSON text."""
    return {
        "section": section,
        "name": _text(name),
        "value": _json(value),
        "source": "training_parameters.json",
    }


def _provenance(capture: _Capture) -> dict:
    """Require dataset derivation metadata to match the already loaded saved split exactly."""
    saved = capture.document("dataset_inputs.json")
    if (
        saved["version"] != 1
        or saved["training_spec"] != capture.spec
        or saved["training_evidence"] != capture.filters
    ):
        _invalid()
    binding = capture.spec.get("feature_binding_json")
    expected_binding = None if binding is None else _parse(binding)
    if saved.get("feature_lookup") != expected_binding:
        _invalid()
    _text(capture.spec["table"])
    _integer(capture.spec["version"])
    for name in ("source_table_id", "source_uc_table_id"):
        if saved.get(name) is not None:
            _text(saved[name])
    return saved


def _datasets(capture: _Capture) -> list[dict]:
    """Describe both saved populations and link only source-verified native dataset Inputs."""
    saved = _provenance(capture)
    links = _input_links(capture, saved)
    binding = _binding_preview(saved.get("feature_lookup"))
    rows = []
    for role, prefix in (("training", "train"), ("holdout", "holdout")):
        rows.append(
            {
                "role": role,
                "name": f"{capture.spec['table']}@{capture.spec['version']}/{role}",
                "source_table": capture.spec["table"],
                "source_version": capture.spec["version"],
                "delta_table_id": saved.get("source_table_id"),
                "uc_table_id": saved.get("source_uc_table_id"),
                "row_count": _integer(capture.filters[f"{role}_rows"]),
                "membership_digest": _text(capture.filters[f"{prefix}_key_sha256"]),
                "feature_binding": binding if role == "training" else None,
                "source": "dataset_inputs.json",
                **links[role],
            }
        )
    return rows + _feature_rows(saved)


def _empty_link(status: str) -> dict:
    """Keep absent native Input identities explicitly null without fabricating run links."""
    return {"run_id": None, "experiment_id": None, "dataset_digest": None, "inputs_status": status}


def _input_links(capture: _Capture, saved: dict) -> dict:
    """Prefer the owning run, then the verified winning child when parent Inputs were not copied."""
    links = _run_inputs(capture.run, capture.model.run_id, saved)
    if capture.winner is not None:
        child_links = _run_inputs(capture.winner, capture.winner.info.run_id, saved)
        for role, link in links.items():
            if link["inputs_status"] == "not_recorded":
                links[role] = child_links[role]
    return links


def _run_inputs(run: Any, expected: str, saved: dict) -> dict:
    """Keep native Input availability independent from valid saved dataset provenance."""
    roles = ("training", "holdout")
    if run is None:
        return {role: _empty_link("unavailable") for role in roles}
    try:
        info = _run_info(run, expected)
        if run.inputs is None:
            return {role: _empty_link("not_recorded") for role in roles}
        inputs = run.inputs.dataset_inputs
        if len(inputs) > _MAX_LIST_ENTRIES:
            return {role: _empty_link("invalid") for role in roles}
        return {role: _role_input(inputs, role, info, saved) for role in roles}
    except Exception:  # noqa: BLE001 - native Input navigation is optional metadata
        return {role: _empty_link("invalid") for role in roles}


def _role_input(inputs: list, role: str, info: dict, saved: dict) -> dict:
    """Reject duplicate or mismatched native Inputs for one concrete population."""
    matches = [
        item
        for item in inputs
        if {tag.key: tag.value for tag in item.tags}.get("mlflow.data.context") == role
    ]
    if not matches:
        return _empty_link("not_recorded")
    try:
        if len(matches) != 1:
            _invalid()
        dataset = matches[0].dataset
        _validate_input(dataset, role, saved)
        return {
            "run_id": info["run_id"],
            "experiment_id": info["experiment_id"],
            "dataset_digest": _text(dataset.digest),
            "inputs_status": "recorded",
        }
    except Exception:  # noqa: BLE001 - malformed navigation cannot invalidate saved lineage
        return _empty_link("invalid")


def _validate_input(dataset: Any, role: str, saved: dict) -> None:
    """Bind native dataset links to source version, both table identities and saved derivation."""
    spec = saved["training_spec"]
    source = _parse(dataset.source)
    skyulf = source["skyulf"]
    provenance_digest = hashlib.sha256(_json(saved).encode("utf-8")).hexdigest()
    checks = (
        dataset.source_type == "delta_table",
        dataset.name == f"{spec['table']}@{spec['version']}/{role}",
        source["delta_table_name"] == spec["table"],
        type(source["delta_table_version"]) is int,
        source["delta_table_version"] == spec["version"],
        source.get("delta_table_id") == saved.get("source_uc_table_id"),
        source.get("is_databricks_uc_table", False) is bool(saved.get("source_uc_table_id")),
        skyulf.get("source_table_id") == saved.get("source_table_id"),
        skyulf["dataset_id"] == saved["source_evidence"]["dataset_id"],
        skyulf["provenance_artifact"] == "dataset_inputs.json",
        skyulf["provenance_sha256"] == provenance_digest,
        _integer(_parse(dataset.profile)["num_rows"]) == saved["training_evidence"][f"{role}_rows"],
    )
    if not all(checks):
        _invalid()


def _binding_preview(binding: dict | None) -> str | None:
    """Expose declarative lookup metadata without copying arbitrary binding fields."""
    if binding is None:
        return None
    return _json(
        {key: binding[key] for key in ("lookup_spec", "lookup_evidence") if key in binding}
    )


def _feature_rows(saved: dict) -> list[dict]:
    """Distinguish pinned feature tables from the actual training and holdout populations."""
    binding = saved.get("feature_lookup")
    if binding is None:
        return []
    records = binding["lookup_evidence"]["feature_tables"]
    if not isinstance(records, list):
        _invalid()
    if len(records) > _MAX_ROWS - 2:
        raise _EvidenceIssue("too_large", "saved_training_summary_too_large")
    return [_feature_row(record) for record in records]


def _feature_row(record: dict) -> dict:
    """Avoid attaching training-row counts or native Input links to feature snapshot metadata."""
    table = _text(record["table_name"])
    return {
        "role": "feature",
        "name": table,
        "source_table": table,
        "source_version": _integer(record["version"]),
        "delta_table_id": _text(record["table_id"]),
        "uc_table_id": None,
        "row_count": None,
        "membership_digest": None,
        "feature_binding": None,
        "source": "dataset_inputs.json:feature_lookup",
        **_empty_link("not_recorded"),
    }
