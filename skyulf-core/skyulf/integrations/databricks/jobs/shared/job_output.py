"""Render shared, escaped operator reports for Bundle notebook results."""

import json
from html import escape
from typing import Any

from ....mlflow.lifecycle.validation import evaluate_quality_gates


def _text(value: Any) -> str:
    """Escape dynamic registry, parameter and result values before rendering HTML."""
    return escape(str(value))


def output_table(headers: tuple[str, ...], rows: list[tuple[Any, ...]]) -> str:
    """Render small comparison and parameter tables without injecting dynamic markup."""
    head = "".join(f"<th>{_text(value)}</th>" for value in headers)
    body = "".join(
        "<tr>" + "".join(f"<td>{_text(value)}</td>" for value in row) + "</tr>" for row in rows
    )
    return f"<table><thead><tr>{head}</tr></thead><tbody>{body}</tbody></table>"


def _receipt_summary(receipt: dict[str, Any]) -> list[str]:
    """Describe the actual committed transition without implying a new mutation."""
    kind = receipt.get("kind")
    if kind in {"initial", "promotion", "rollback"}:
        prior = receipt.get("prior_version")
        return [
            "<p><strong>Champion changed:</strong> "
            f"{_text('v' + prior if prior else 'none')} &rarr; "
            f"{_text('v' + receipt['new_version'])}</p>"
        ]
    if kind == "rejection":
        return ["<p>Candidate rejected. Champion is unchanged.</p>"]
    return []


def _comparison_summary(candidate: dict[str, Any]) -> list[str]:
    """Present the same selection metric and metric panel in phase and final reports."""
    comparison = candidate.get("comparison", {})
    if not comparison:
        return []
    challenger = comparison.get("candidate_metrics", {})
    champion = comparison.get("champion_metrics") or {}
    sections = [
        f"<p><strong>Decision metric:</strong> {_text(comparison['metric'])}<br>"
        f"<strong>Minimum improvement (absolute):</strong> "
        f"{_text(comparison.get('min_improvement', 0))}<br>"
        f"<strong>Eligible:</strong> {_text(comparison['eligible'])}<br>"
        f"<strong>Reason:</strong> {_text(comparison['reason'])}</p>",
        output_table(
            ("Metric", "Candidate", "Champion"),
            [(key, value, champion.get(key, "No champion")) for key, value in challenger.items()],
        ),
    ]
    gates = evaluate_quality_gates(
        challenger,
        comparison["metric"],
        comparison.get("quality_threshold"),
        comparison.get("quality_gates"),
    )
    if gates:
        labels = {
            "passed": "Passed",
            "threshold_not_met": "Failed: threshold not met",
            "metric_unavailable_or_non_finite": "Failed: metric unavailable or non-finite on this holdout",
        }
        sections.append(
            output_table(
                ("Quality metric", "Candidate", "Required", "Result"),
                [
                    (
                        gate["metric"],
                        gate["value"],
                        f"{'<=' if gate['direction'] == 'minimize' else '>='} {gate['threshold']}",
                        labels[gate["reason"]],
                    )
                    for gate in gates
                ],
            )
        )
    return sections


def _nested_search_output(tuning: dict[str, Any]) -> list[str]:
    """Display independent outer evaluation beside the separate final search score."""
    nested = tuning.get("nested_cv")
    if not isinstance(nested, dict) or nested.get("status") != "nested_cv":
        return []
    return [
        "<h3>Nested CV evaluation</h3>",
        output_table(
            ("Setting", "Value"),
            [
                ("Outer folds", nested["outer_folds"]),
                ("Inner folds", nested["inner_folds"]),
                ("Outer mean score", nested["mean_score"]),
                ("Outer score std", nested["std_score"]),
                ("Metric", nested["scoring_metric"]),
                ("Total search trials", nested["total_trials"]),
                ("Split policy", json.dumps(nested.get("split_policy", {}), sort_keys=True)),
                (
                    "Final decision thresholds",
                    json.dumps(tuning.get("decision_thresholds"), sort_keys=True),
                ),
                ("Threshold metric", tuning.get("decision_threshold_metric")),
            ],
        ),
        output_table(
            (
                "Fold",
                "Inner best score",
                "Outer score",
                "Selected parameters",
                "Decision thresholds",
            ),
            [
                (
                    fold["fold"],
                    fold["inner_best_score"],
                    fold["outer_score"],
                    json.dumps(fold["best_params"], sort_keys=True),
                    json.dumps(
                        fold.get("threshold_selection", {}).get("decision_thresholds"),
                        sort_keys=True,
                    ),
                )
                for fold in nested["folds"]
            ],
        ),
        "<p>Each outer score evaluates an independent search on untouched rows. "
        "The saved model comes from a separate final search on all training rows.</p>",
    ]


def render_lifecycle_output(phase: str, payload: dict[str, Any]) -> str:
    """Show a completed phase's useful values while keeping technical receipts folded."""
    titles = {
        "prepare": "Request validated and data pinned",
        "initialize": "Run initialized",
        "load_data": "Source data loaded",
        "prepare_dataset": "Dataset prepared and split",
        "select_best_model": "Model selection completed",
        "validate_model": "Fitted model validated",
        "model_decision": "Model decision completed",
        "train": "Candidate pipeline trained",
        "evaluate_register": "Candidate evaluated and registered",
        "train_register": "Candidate trained, evaluated and registered",
        "compare": "Candidate comparison completed",
        "decide": "Promotion policy evaluated",
        "compare_decide": "Candidate compared and promotion policy evaluated",
        "operator": "Operator action completed",
        "finalize": "Training status recorded",
    }
    rows = [
        (key.replace("_", " ").capitalize(), value)
        for key, value in payload.items()
        if isinstance(value, (str, int, float, bool)) and not key.endswith(("sha256", "digest"))
    ]
    sections = [
        f"<h2>{_text(titles.get(phase, phase))}</h2>",
        output_table(("Result", "Value"), rows),
    ]
    _append_training_output(sections, payload)
    _append_competition_output(sections, payload)
    candidate = payload.get("candidate", payload)
    if "candidate" in payload:
        sections.append(
            f"<p><strong>Candidate:</strong> {_text(candidate.get('model_name', ''))} "
            f"v{_text(candidate.get('model_version', ''))}</p>"
        )
    sections.extend(_comparison_summary(candidate))
    receipt = payload.get("alias_change") or payload.get("result") or {}
    if phase in {"operator", "model_decision"} and receipt.get("model_name"):
        sections.append(f"<p><strong>Model:</strong> {_text(receipt['model_name'])}</p>")
    sections.extend(_receipt_summary(receipt))
    _append_lifecycle_actions(sections, phase, payload)
    raw = json.dumps(payload, indent=2, default=str, allow_nan=False)
    sections.append(
        f"<details><summary>Technical details (JSON)</summary><pre>{_text(raw)}</pre></details>"
    )
    return '<div style="font-family:system-ui;line-height:1.5">' + "".join(sections) + "</div>"


def _append_competition_output(sections: list[str], payload: dict[str, Any]) -> None:
    """Display measured training-side ranks separately from champion approval."""
    competition = payload.get("competition", payload)
    rows = competition.get("leaderboard")
    if not isinstance(rows, list):
        return
    sections.append(f"<h3>Model competition: {_text(competition['selection_metric'])}</h3>")
    sections.append(
        output_table(
            ("Candidate", "Model", "Search", "CV mean", "CV std", "Evaluation", "Training run"),
            [
                (
                    row["candidate"],
                    row.get("model_type", ""),
                    row.get("strategy", ""),
                    row["mean"],
                    row["std"],
                    row["evaluation_mode"],
                    row["run_id"],
                )
                for row in rows
            ],
        )
    )
    sections.append(
        f"<p>Selected model: <strong>{_text(competition['winner'])}</strong>. "
        "Holdout quality checks and champion approval are separate steps.</p>"
    )


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
    status = "recovery requested" if payload.get("recovery_required") else "completed"
    sections = [f"<h2>{_text(action.replace('_', ' ').title())} {status}</h2>"]
    _append_competition_output(sections, payload)
    sections.extend(_receipt_summary(receipt))
    name = candidate.get("model_name") or receipt.get("model_name")
    if name:
        sections.append(f"<p><strong>Model:</strong> {_text(name)}</p>")
    if action.startswith("train"):
        sections.append(
            f"<p><strong>Candidate version:</strong> {_text(candidate.get('model_version'))}</p>"
        )
        sections.extend(_comparison_summary(candidate))
    if action == "score":
        sections.append(render_scoring_summary({**payload, **result}))
    elif payload["score_requested"]:
        sections.append("<p>Scoring requested. Check the child score run for its result.</p>")
    else:
        sections.append("<p>This action did not request scoring.</p>")
    _append_operator_options(sections, payload, receipt)
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


def _append_training_output(sections: list[str], payload: dict[str, Any]) -> None:
    """Render measured training metrics, tuning evidence and optional explanations."""
    metrics = payload.get("metrics")
    if isinstance(metrics, dict):
        sections.append(output_table(("Metric", "Value"), list(metrics.items())))
    tuning = payload.get("tuning")
    if isinstance(tuning, dict):
        sections.append("<h3>Training search</h3>")
        sections.append(
            output_table(
                ("Setting", "Value"),
                [
                    ("Strategy", tuning.get("strategy")),
                    ("Trials", tuning.get("n_trials")),
                    ("Search metric", tuning.get("scoring_metric")),
                    ("Best search score", tuning.get("best_score")),
                    (
                        "Selected parameters",
                        json.dumps(tuning.get("best_params", {}), sort_keys=True),
                    ),
                    ("MLflow artifact", tuning.get("artifact")),
                ],
            )
        )
        sections.append(
            "<p>Search uses training rows only. Negative loss scores remain negative; higher is better.</p>"
        )
        sections.extend(_nested_search_output(tuning))
    explanation = payload.get("explanations")
    if isinstance(explanation, dict):
        sections.append("<h3>Model explanations</h3>")
        sections.append(output_table(("Result", "Value"), list(explanation.items())))


def _append_lifecycle_actions(sections: list[str], phase: str, payload: dict[str, Any]) -> None:
    """Explain unchanged aliases and point operators to the final report."""
    if (
        phase in {"decide", "compare_decide", "model_decision"}
        and "promotion_policy" in payload
        and not payload.get("alias_change")
    ):
        sections.append(
            "<p>Awaiting manual review. Champion is unchanged.</p>"
            if payload.get("promotion_policy") == "manual_approval"
            else "<p>Champion is unchanged. The candidate did not pass promotion gates.</p>"
        )
    if phase in {"decide", "compare_decide", "operator", "model_decision"}:
        report = "training_report" if phase == "model_decision" else "finalize_and_report"
        sections.append(
            f"<p>Open <strong>{report}</strong> for the final decision and operator actions.</p>"
        )


def render_scoring_summary(result: dict[str, Any]) -> str:
    """Render current scoring evidence separately from any previous committed manifest."""
    if result.get("recovery_required"):
        return (
            "<p>CDF history is unavailable. Predictions have not been replaced. "
            "Open <strong>recover_predictions</strong> for the full snapshot recovery "
            "and <strong>scoring_report</strong> for the final result.</p>"
        )
    sections: list[str] = []
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
        output_table(
            ("Result", "Value"),
            [
                (label, result[key])
                for key, label in (
                    ("source_table", "Source table"),
                    ("prediction_table", "Prediction table"),
                    ("source_end_version", "Source end version"),
                    ("input_count", "Input rows"),
                    ("output_count", "Output rows"),
                    ("commit_version", "Prediction table Delta version"),
                )
                if key in result
            ],
        )
    )
    manifest = result.get("manifest", {})
    _append_scoring_provenance(sections, manifest, noop)
    return "".join(sections)


def _append_scoring_provenance(
    sections: list[str],
    manifest: dict[str, Any] | None,
    noop: bool,
) -> None:
    """Distinguish this write's recovery and coverage from historical no-op provenance."""
    if manifest and manifest.get("cdf_recovered") and not noop:
        sections.append(
            "<p><strong>CDF recovery completed.</strong> Predictions were replaced from "
            "the pinned full source snapshot. Later runs resume incremental scoring.</p>"
        )
    if manifest and not noop:
        _append_scoring_coverage(sections, manifest)
    if manifest:
        label = "Model recorded by the previous write" if noop else "Prediction model"
        model_name = manifest.get("model_name", manifest.get("model_set_name"))
        model_version = manifest.get("model_version", manifest.get("model_set_version"))
        sections.append(
            f"<p><strong>{label}:</strong> {_text(model_name)} v{_text(model_version)}</p>"
        )


def _append_operator_options(
    sections: list[str], payload: dict[str, Any], receipt: dict[str, Any]
) -> None:
    """Render explicit next actions and the guarded rollback target."""
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
                output_table(
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
        sections.append(output_table(("Parameter", "Value"), rows))
        if rollback:
            sections.append("</details>")


def _append_scoring_coverage(sections: list[str], manifest: dict[str, Any]) -> None:
    """Expose successful estimates and deliberate exclusions for the current write."""
    rows = [
        (label, manifest[key])
        for key, label in (
            ("predicted_count", "Predicted rows"),
            ("excluded_count", "Excluded rows"),
        )
        if key in manifest
    ]
    if rows:
        sections.append(output_table(("Scoring coverage", "Rows"), rows))
