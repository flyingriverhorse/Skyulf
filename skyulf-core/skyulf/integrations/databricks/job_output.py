"""Render shared, escaped operator reports for Bundle notebook results."""

import json
from html import escape
from typing import Any


def _text(value: Any) -> str:
    """Escape dynamic registry, parameter and result values before rendering HTML."""
    return escape(str(value))


def _table(headers: tuple[str, ...], rows: list[tuple[Any, ...]]) -> str:
    """Render small comparison and parameter tables without injecting dynamic markup."""
    head = "".join(f"<th>{_text(value)}</th>" for value in headers)
    body = "".join(
        "<tr>" + "".join(f"<td>{_text(value)}</td>" for value in row) + "</tr>" for row in rows
    )
    return f"<table><thead><tr>{head}</tr></thead><tbody>{body}</tbody></table>"


def render_bundle_output(payload: dict[str, Any]) -> str:
    """Present lifecycle, comparison, scoring and next steps with raw JSON in a disclosure.

    The report describes only this task's completed work. A handoff request
    does not imply success of the separate score job. No-op manifests describe
    the previous prediction write, not the model selected by the current run.
    """
    action = payload["action"]
    result = payload["result"]
    candidate = result.get("candidate", result)
    receipt = result.get("alias_change") or result
    sections = [f"<h2>{_text(action.replace('_', ' ').title())} completed</h2>"]
    kind = receipt.get("kind")
    if kind in {"initial", "promotion", "rollback"}:
        prior = receipt.get("prior_version")
        sections.append(
            "<p><strong>Champion changed:</strong> "
            f"{_text('v' + prior if prior else 'none')} &rarr; "
            f"{_text('v' + receipt['new_version'])}</p>"
        )
    elif kind == "rejection":
        sections.append("<p>Candidate rejected. Champion is unchanged.</p>")
    name = candidate.get("model_name") or receipt.get("model_name")
    if name:
        sections.append(f"<p><strong>Model:</strong> {_text(name)}</p>")
    if action.startswith("train"):
        sections.append(
            f"<p><strong>Candidate version:</strong> {_text(candidate.get('model_version'))}</p>"
        )
        comparison = candidate.get("comparison", {})
        if comparison:
            sections.append(
                f"<p><strong>Decision metric:</strong> {_text(comparison['metric'])}<br>"
                f"<strong>Eligible:</strong> {_text(comparison['eligible'])}<br>"
                f"<strong>Reason:</strong> {_text(comparison['reason'])}</p>"
            )
            challenger = comparison.get("candidate_metrics", {})
            champion = comparison.get("champion_metrics", {})
            sections.append(
                _table(
                    ("Metric", "Candidate", "Champion"),
                    [
                        (key, value, champion.get(key, "No champion"))
                        for key, value in challenger.items()
                    ],
                )
            )
    if action == "score":
        if result.get("selected_model_version"):
            sections.append(
                "<p><strong>Selected model for this run:</strong> "
                f"{_text(result.get('selected_model_name'))} "
                f"v{_text(result['selected_model_version'])}</p>"
            )
        noop = result.get("noop", False)
        sections.append(
            "<p><strong>No new predictions written.</strong></p>"
            if noop
            else "<p><strong>Prediction write completed.</strong></p>"
        )
        sections.append(
            _table(
                ("Result", "Value"),
                [
                    (label, result[key])
                    for key, label in (
                        ("input_count", "Input rows"),
                        ("output_count", "Output rows"),
                        ("commit_version", "Prediction table Delta version"),
                    )
                    if key in result
                ],
            )
        )
        manifest = result.get("manifest", {})
        if manifest:
            label = "Model recorded by the previous write" if noop else "Prediction model"
            sections.append(
                f"<p><strong>{label}:</strong> {_text(manifest.get('model_name'))} "
                f"v{_text(manifest.get('model_version'))}</p>"
            )
    elif payload["score_requested"]:
        sections.append("<p>Scoring requested. Check the child score run for its result.</p>")
    else:
        sections.append("<p>This action did not request scoring.</p>")
    for next_action, parameters in payload.get("next_actions", {}).items():
        rollback = next_action == "rollback"
        if rollback:
            expected = parameters["expected_champion_version"]
            sections.append("<h3>If rollback is needed</h3>")
            sections.append(
                "<p>You can use the information below to restore the previous champion. "
                "Rollback is optional and does not run automatically.</p>"
            )
            sections.append(
                _table(
                    ("Rollback", "Version"),
                    [
                        ("Required current champion", "v" + expected),
                        ("Restore version", "v" + receipt["prior_version"]),
                    ],
                )
            )
            sections.append(
                f"<p>Rollback proceeds only if champion is still v{_text(expected)}. "
                "The expected champion is a safety check, not the restore target.</p>"
                "<details><summary>Show parameters only if you want to roll back</summary>"
            )
        else:
            sections.append(f"<h3>Available action: {_text(next_action)}</h3>")
        sections.append(
            "<p>Use Run with different settings on the same train job. "
            "Clear fields not listed below.</p>"
        )
        rows = list(parameters.items())
        if next_action == "reject":
            rows.append(("rejection_reason", "Enter your reason"))
        sections.append(_table(("Parameter", "Value"), rows))
        if rollback:
            sections.append("</details>")
    raw = json.dumps(payload, indent=2, default=str, allow_nan=False)
    sections.append(
        f"<details><summary>Technical details (JSON)</summary><pre>{_text(raw)}</pre></details>"
    )
    return (
        '<div style="font-family:system-ui;max-width:1000px;line-height:1.5">'
        "<style>td,th{padding:8px;text-align:left;vertical-align:top;"
        "border-bottom:1px solid #ddd;overflow-wrap:anywhere}table{border-collapse:collapse;"
        "width:100%}pre{white-space:pre-wrap;overflow-wrap:anywhere}</style>"
        + "".join(sections)
        + "</div>"
    )
