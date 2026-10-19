"""Link the existing training evidence from one compact final notebook overview."""

from collections.abc import Callable, Iterable
from html import escape
from itertools import islice
from typing import Any
from urllib.parse import quote, urlsplit

from .training_node_output import training_report_document

_UNAVAILABLE = "Unavailable; inspect the fitting run artifacts"
_SHAP_HELP = (
    '<a href="https://www.skyulf.com/manual/user_guide/databricks_explanations.html">'
    "Enable config/training.yml defaults.explainability and the SHAP report task</a>"
)


def _origin(value: str) -> str:
    """Accept only a plain HTTPS workspace origin, never executable or injected URLs."""
    parsed = urlsplit(value)
    if (
        parsed.scheme != "https"
        or not parsed.hostname
        or parsed.username
        or parsed.password
        or parsed.path not in {"", "/"}
        or parsed.query
        or parsed.fragment
    ):
        return ""
    return f"https://{parsed.netloc}"


def _link(origin: str, path: str, label: Any) -> str:
    """Escape labels and attributes while leaving missing workspace links as plain text."""
    text = escape(str(label))
    return f'<a href="{escape(origin + path, quote=True)}">{text}</a>' if origin else text


def _run_path(experiment_id: Any, run_id: Any) -> str:
    """Encode both run identifiers as path components."""
    return (
        f"/ml/experiments/{quote(str(experiment_id), safe='')}/runs/{quote(str(run_id), safe='')}"
    )


def _version_link(origin: str, name: str, version: Any) -> str:
    """Link a concrete registered version without resolving mutable registry aliases."""
    model_path = "/".join(quote(part, safe="") for part in name.split("."))
    path = f"/explore/data/models/{model_path}/version/{quote(str(version), safe='')}"
    return _link(origin, path, f"{name} v{version}")


def _document(client: Any, run_id: str, path: str) -> dict | None:
    """Treat absent or unreadable optional evidence as unavailable, without error details."""
    try:
        value = training_report_document(client, run_id, path)
        return value if isinstance(value, dict) else None
    except Exception:  # noqa: BLE001 - optional presentation must not retry lifecycle mutations
        return None


def _model_identities(payload: dict, run_id: str) -> list[dict]:
    """Retain each fitting run and only associate the registered version with its winner."""
    if "components" in payload:
        return [{**identity, "name": name} for name, identity in payload["components"].items()]
    result = payload.get("result", {})
    candidate = result.get("candidate", result)
    competition = payload.get("competition", {})
    if competition.get("leaderboard"):
        return [
            {
                **(candidate if row["candidate"] == competition["winner"] else {}),
                "name": row["candidate"],
                "run_id": row["run_id"],
                "charts_source": run_id if row["candidate"] == competition["winner"] else None,
            }
            for row in competition["leaderboard"]
        ]
    return [{**candidate, "run_id": candidate.get("run_id", run_id), "name": "Model"}]


def _artifact_link(origin: str, run_path: str, path: str) -> str:
    """Open the established Artifacts tab and name the saved file to select there."""
    return _link(origin, f"{run_path}/artifacts", path)


def _data_cell(documents: dict, origin: str, run_path: str) -> str:
    """Show only source identity and aggregate counts, with links to detailed evidence."""
    data = documents.get("training_data.json") or {}
    spec = documents.get("candidate_training_spec.json") or {}
    evidence = documents.get("training_filter_evidence.json") or {}
    filters = documents.get("pre_split_filters.json") or {}
    table = data.get("train_data_destination", spec.get("table", "Unavailable"))
    version = data.get("train_data_version", spec.get("version", "Unavailable"))
    lines = [escape(f"{table} · Delta version {version}")]
    for label, value in (
        ("Source rows", evidence.get("source_rows")),
        ("Training rows", evidence.get("training_rows")),
        ("Holdout rows", evidence.get("holdout_rows")),
        ("Pre-split removed", filters.get("excluded_rows")),
    ):
        lines.append(escape(f"{label}: {value if value is not None else 'Unavailable'}"))
    lines.append(_conversion_cell(data))
    lines.extend(
        _artifact_link(origin, run_path, path)
        for path in (
            "training_data.json",
            "training_filter_evidence.json",
            "pre_split_filters.json",
        )
        if documents.get(path) is not None
    )
    return "<br>".join(lines)


def _conversion_cell(data: dict) -> str:
    """Show performed casts from saved evidence without inferring missing source types."""
    conversions = data.get("source_column_conversions")
    if not isinstance(conversions, list):
        return "Decimal conversions: not recorded"
    if not conversions:
        return "Decimal conversions: none"
    try:
        lines = [
            escape(f"{str(item['column'])[:120]}: {item['source_dtype']} -> {item['target_dtype']}")
            for item in conversions[:10]
        ]
    except (KeyError, TypeError):
        return "Decimal conversions: not recorded"
    if len(conversions) > 10:
        lines.append("See training_data.json for additional conversions")
    return (
        "Decimal conversions:<br>" + "<br>".join(lines) + "<br>"
        "Converting DECIMAL to float64 may lose precision beyond roughly 15-17 significant digits."
    )


def _optional_cell(
    document: dict | None, enabled: bool | None, path: str, origin: str, run_path: str
) -> str:
    """A missing artifact is off only when the saved configuration explicitly says so."""
    if document is not None:
        status = document.get("status", "Unavailable")
        return escape(str(status)) + "<br>" + _artifact_link(origin, run_path, path)
    return "Off" if enabled is False else _UNAVAILABLE


def _threshold_preview(entries: Iterable[tuple[Any, Any]]) -> str:
    """Bound class previews while retaining the saved artifact as the full authority."""
    selected = list(islice(entries, 11))
    lines = []
    for label, value in selected[:10]:
        fields = [str(label), str(value)]
        lines.append(
            ": ".join(escape(field[:120] + ("…" if len(field) > 120 else "")) for field in fields)
        )
    if len(selected) > 10:
        lines.append("See the full artifact for additional thresholds")
    return "<br>".join(lines)


def _threshold_cell(documents: dict, origin: str, run_path: str) -> str:
    """Expose saved threshold selection without recomputing or inferring probability rules."""
    evidence = documents.get("decision_threshold.json")
    if evidence is not None:
        return (
            escape(str(evidence.get("mode", "Saved")))
            + "<br>"
            + _threshold_preview(
                (item["class"], item["value"]) for item in evidence.get("thresholds", [])
            )
            + "<br>"
            + _artifact_link(origin, run_path, "decision_threshold.json")
        )
    tuning = documents.get("tuning.json") or {}
    if tuning.get("decision_thresholds"):
        return (
            _threshold_preview(tuning["decision_thresholds"].items())
            + "<br>"
            + _artifact_link(origin, run_path, "tuning.json")
        )
    return "No saved threshold evidence"


def _chart_cell(client: Any, identity: dict, run: Any, origin: str, request: dict | None) -> str:
    """Use a completed chart-run tag; optional leaves may still be running at report time."""
    source = identity.get("charts_source")
    try:
        chart_source = client.get_run(source) if source else run
    except Exception:  # noqa: BLE001 - optional chart metadata must not hide the fitting evidence
        return "Unavailable at report time; open generate_charts > Output"
    chart_id = chart_source.data.tags.get("skyulf.charts.run_id")
    if chart_id:
        return _link(
            origin,
            _run_path(chart_source.info.experiment_id, chart_id) + "/model-metrics",
            "Evaluation charts",
        )
    if "charts_source" in identity and source is None:
        return "Winner only; see the selected model"
    if request is not None and not (request.get("config", {}).get("evaluation_charts") or {}).get(
        "enabled"
    ):
        return "Off"
    return "Unavailable at report time; open generate_charts > Output"


def _model_row(client: Any, identity: dict, origin: str, request: dict | None) -> str:
    """Read each fit's saved metadata and render one row without samples or model bytes."""
    run_id = identity["run_id"]
    run = client.get_run(run_id)
    run_path = _run_path(run.info.experiment_id, run_id)
    artifacts = {item.path for item in client.list_artifacts(run_id)}
    documents = {
        path: _document(client, run_id, path)
        for path in (
            "candidate_training_spec.json",
            "training_pipeline_config.json",
            "training_data.json",
            "training_filter_evidence.json",
            "pre_split_filters.json",
            "preprocessing_probe.json",
            "explanations.json",
            "decision_threshold.json",
            "tuning.json",
        )
        if path in artifacts
    }
    spec = documents.get("candidate_training_spec.json")
    pipeline = documents.get("training_pipeline_config.json")
    model = escape(str(identity["name"])) + "<br>" + _link(origin, run_path, f"MLflow run {run_id}")
    if identity.get("model_name") and identity.get("model_version"):
        model += "<br>" + _version_link(origin, identity["model_name"], identity["model_version"])
    probe = _optional_cell(
        documents.get("preprocessing_probe.json"),
        bool(spec.get("preprocessing_probe", False)) if spec is not None else None,
        "preprocessing_probe.json",
        origin,
        run_path,
    )
    shap = _optional_cell(
        documents.get("explanations.json"),
        bool(pipeline.get("explainability")) if pipeline is not None else None,
        "explanations.json",
        origin,
        run_path,
    )
    shap += "<br>" + (
        _artifact_link(origin, run_path, "explanations.html")
        if "explanations.html" in artifacts
        else _SHAP_HELP
    )
    cells = (
        model,
        _data_cell(documents, origin, run_path),
        probe,
        shap,
        _chart_cell(client, identity, run, origin, request),
        _threshold_cell(documents, origin, run_path),
    )
    return "<tr>" + "".join(f"<td>{cell}</td>" for cell in cells) + "</tr>"


def render_training_summary(
    client: Any, run_id: str, payload: dict, *, workspace_host: str = ""
) -> str:
    """Show a per-model index of existing evidence at the top of the final report."""
    origin = _origin(workspace_host)
    request = _document(client, run_id, "lifecycle/request.json")
    sections = ["<h2>Where to look</h2>"]
    candidate = payload.get("model_set_candidate")
    if candidate:
        sections.append(
            "<p>Model set: "
            + _version_link(origin, candidate["name"], candidate["version"])
            + "</p>"
        )
    headers = (
        "Model / run",
        "Training data",
        "Preprocessing probe",
        "SHAP",
        "Evaluation charts",
        "Decision threshold",
    )
    sections.append(
        "<table><thead><tr>"
        + "".join(f"<th>{title}</th>" for title in headers)
        + "</tr></thead><tbody>"
    )
    for identity in _model_identities(payload, run_id):
        try:
            sections.append(_model_row(client, identity, origin, request))
        except Exception:  # noqa: BLE001 - one unreadable fit must not hide other results
            sections.append(
                f'<tr><td>{escape(str(identity["name"]))}</td><td colspan="5">{_UNAVAILABLE}</td></tr>'
            )
    sections.append(
        "</tbody></table><p>Artifact links open the run's Artifacts tab; select the named file. "
        "Links use saved evidence only. Optional chart tasks may still be running. "
        "Inspect the job graph for task states. A requested child score run starts after this report.</p>"
    )
    return (
        '<div style="font-family:system-ui;line-height:1.5;overflow-x:auto">'
        + "".join(sections)
        + "</div>"
    )


def render_final_training_report(
    payload: dict,
    *,
    tracking_uri: str,
    run_id: str,
    workspace_host: str,
    render: Callable[[dict], str],
) -> str:
    """Keep the lifecycle result visible even when optional evidence cannot be retrieved."""
    from skyulf.integrations.mlflow.shared._client import make_tracking_client  # noqa: PLC0415

    try:
        summary = render_training_summary(
            make_tracking_client(tracking_uri), run_id, payload, workspace_host=workspace_host
        )
    except Exception:  # noqa: BLE001 - presentation must not trigger lifecycle retries
        summary = f"<h2>Where to look</h2><p>{_UNAVAILABLE}</p>"
    return summary + render(payload)
