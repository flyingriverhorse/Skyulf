"""Readable training evidence kept separate from optional SHAP report leaves."""

import json
from pathlib import Path
from tempfile import TemporaryDirectory
from typing import Any

from ...jobs.shared.job_output import output_table, render_lifecycle_output


def training_report_document(client: Any, run_id: str, path: str) -> dict:
    """Read bounded run metadata without interpreting report HTML or model bytes."""
    with TemporaryDirectory(prefix="skyulf-training-report-") as directory:
        downloaded = client.download_artifacts(run_id, path, directory)
        return json.loads(Path(downloaded).read_text(encoding="utf-8"))


def render_training_node(client: Any, run_id: str, output: dict) -> str:
    """Show actual split, preprocessing, model, CV and tuning details for one fit."""
    run = client.get_run(run_id)
    params = run.data.params
    spec = training_report_document(client, run_id, "candidate_training_spec.json")
    pipeline = training_report_document(client, run_id, "training_pipeline_config.json")
    summary = {
        "run_id": run_id,
        "model_name": output.get("model_name"),
        **{
            key: spec.get(key)
            for key in (
                "table",
                "version",
                "target_column",
                "split_strategy",
                "random_state",
                "test_size",
            )
        },
        **output.get("training", {}),
    }
    sections = [
        render_lifecycle_output("train", summary),
        "<h3>Preprocessing and model settings</h3>",
        output_table(
            ("Setting", "Value"),
            [
                (key, json.dumps(pipeline[key], default=str))
                for key in ("preprocessing", "modeling")
                if key in pipeline
            ],
        ),
        "<h3>Cross-validation and training parameters</h3>",
        output_table(("Parameter", "Value"), sorted(params.items())),
    ]
    artifacts = {item.path for item in client.list_artifacts(run_id)}
    for path, title in (
        ("tuning.json", "Training search"),
        ("cross_validation.json", "Cross-validation results"),
        ("competition_evaluation.json", "Candidate selection evaluation"),
    ):
        if path in artifacts:
            document = training_report_document(client, run_id, path)
            sections.extend(
                [
                    f"<h3>{title}</h3>",
                    output_table(
                        ("Setting", "Value"),
                        [(key, json.dumps(value, default=str)) for key, value in document.items()],
                    ),
                ]
            )
    sections.append("<p>Open this model's SHAP task for its separate explanation report.</p>")
    return "".join(sections)
