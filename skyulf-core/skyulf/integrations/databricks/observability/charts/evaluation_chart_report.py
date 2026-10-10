"""Publish genuine saved-model diagnostics as native MLflow logged images."""

import io
import json
import math
from pathlib import Path
from tempfile import TemporaryDirectory
from typing import Any

from ....mlflow.registration.registry import load_run_pipeline
from .evaluation_chart_data import load_chart_sample
from .evaluation_charts import chart_figure, diagnostic_charts, model_charts


def publish_figures(
    client: Any, run_id: str, figures: dict, *, prefix: str, caption: str
) -> list[str]:
    """Use key-based logging, which indexes images for MLflow's Image grid chart."""
    from PIL import Image  # noqa: PLC0415 - installed with optional Matplotlib

    keys = []
    for name, figure in figures.items():
        key = f"{prefix}_{name}"
        try:
            figure.text(0.5, -0.02, caption, ha="center", fontsize=8, color="#475569")
            with io.BytesIO() as buffer:
                figure.savefig(buffer, format="png", dpi=130, bbox_inches="tight")
                buffer.seek(0)
                with Image.open(buffer) as decoded:
                    client.log_image(
                        run_id, decoded.convert("RGB"), key=key, step=0, synchronous=True
                    )
            keys.append(key)
        finally:
            figure.clear()
    return keys


def report_model(
    client: Any, tracking_uri: str, identity: dict, *, destination: str, prefix: str
) -> dict:
    """Plot saved full-holdout predictions without retraining or recomputing temporal features."""
    sample, metadata = load_chart_sample(client, identity["run_id"], identity["model_digest"])
    _validate_sample_identity(metadata, identity)
    report = {
        "run_id": identity["run_id"],
        "destination_run_id": destination,
        "model_digest": identity["model_digest"],
        "sample_rows": metadata["sample_rows"],
        "holdout_rows": metadata["holdout_rows"],
        "image_keys": [],
        "skipped": [],
    }
    if sample is None:
        report["skipped"] = [metadata["reason"]]
        return report
    artifact = load_run_pipeline(
        f"runs:/{identity['run_id']}/model",
        digest=identity["model_digest"],
        tracking_uri=tracking_uri,
    )
    manifest = artifact.manifest
    target, limits = metadata["target_column"], metadata["settings"]
    expected = ["observed", "prediction"]
    if manifest.task == "classification" and manifest.classification_probabilities:
        expected += [f"probability_{index}" for index in range(len(manifest.classes))]
    if list(sample.columns) != expected:
        raise ValueError("Chart predictions differ from the fitted output contract.")
    figures, skipped = diagnostic_charts(
        manifest.task, sample["observed"], sample, manifest.classes, limits["max_classes"]
    )
    extras, notes = _estimator_figures(artifact, limits)
    caption = f"{prefix} | saved holdout: {len(sample)}/{metadata['holdout_rows']} rows | uniform sample, seed 42"
    report["image_keys"] = publish_figures(
        client, destination, {**figures, **extras}, prefix=prefix, caption=caption
    )
    report["image_keys"] += publish_figures(
        client,
        destination,
        evidence_charts(client, identity["run_id"], manifest.task),
        prefix=prefix,
        caption="Recorded training/CV evidence; final holdout was not used for selection.",
    )
    report.update(task=manifest.task, target=target, skipped=skipped + notes)
    return report


def _validate_sample_identity(metadata: dict, identity: dict) -> None:
    """Require the same population and holdout count as the completed training receipt."""
    for key in ("dataset_id", "holdout_key_sha256", "holdout_rows"):
        if metadata[key] != identity[key]:
            raise ValueError(f"Chart sample {key} differs from the completed training receipt.")


def _estimator_figures(artifact: Any, limits: dict) -> tuple[dict, list[str]]:
    """Respect class budgets for potentially wide multiclass coefficient matrices."""
    manifest = artifact.manifest
    if len(manifest.classes) > limits["max_classes"]:
        return {}, ["Model explanation charts exceed max_classes."]
    model = artifact.pipeline.model_estimator._unwrap_tuned_model()
    return model_charts(model, manifest.feature_columns, manifest.classes, limits["max_features"])


def _read_document(client: Any, run_id: str, path: str) -> dict:
    """Download exact optional evidence only after its presence was established."""
    with TemporaryDirectory(prefix="skyulf-chart-evidence-") as directory:
        downloaded = client.download_artifacts(run_id, path, directory)
        return json.loads(Path(downloaded).read_text(encoding="utf-8"))


def evidence_charts(
    client: Any, run_id: str, task: str, *, metrics_run_id: str | None = None
) -> dict:
    """Render existing CV folds and actual search trials without performing additional fits."""
    paths = {item.path for item in client.list_artifacts(run_id)}
    figures = {}
    cv = {}
    if "cross_validation.json" in paths:
        cv = _read_document(client, run_id, "cross_validation.json")
        figures.update(_cv_charts(cv, task))
    figures.update(_metric_summary(client.get_run(metrics_run_id or run_id).data.metrics, cv, task))
    if "tuning.json" in paths:
        tuning = _read_document(client, run_id, "tuning.json")
        trials = [
            (index + 1, trial["score"])
            for index, trial in enumerate(tuning["trials"])
            if trial.get("score") is not None
        ]
        if trials:
            figure, (axis,) = chart_figure("Recorded tuning trials (selection scores)")
            axis.scatter(*zip(*trials, strict=True), color="#a855f7")
            axis.set(xlabel="Trial", ylabel=f"{tuning['scoring_metric']} — higher is better")
            figures["tuning_trials"] = figure
    return figures


def _metric_summary(saved: dict, cv: dict, task: str) -> dict:
    """Compare recorded full-holdout scalars with training CV on separate metric scales."""
    choices = (
        ("rmse", "mae", "r2")
        if task == "regression"
        else ("accuracy", "f1_weighted", "roc_auc", "log_loss")
    )
    names = [name for name in choices if f"heldout_{name}" in saved]
    if not names:
        return {}
    figure, axes = chart_figure(
        "Recorded metrics — full holdout and training CV", panels=len(names)
    )
    for axis, name in zip(axes, names, strict=True):
        values, errors, labels, colors = (
            [saved[f"heldout_{name}"]],
            [0],
            ["Final holdout"],
            ["#22c55e"],
        )
        summary = cv.get("aggregated_metrics", {}).get(name)
        if summary is not None:
            values.insert(0, summary["mean"])
            errors.insert(0, summary["std"])
            labels.insert(0, "Training CV ± SD")
            colors.insert(0, "#a855f7")
        axis.bar(labels, values, yerr=errors, color=colors, capsize=5)
        axis.set_title(name)
    return {"metric_summary": figure}


def _cv_charts(evidence: dict, task: str) -> dict:
    """Keep different metric units on separate axes and disclose post-selection CV."""
    names = (
        ("rmse", "mae", "r2")
        if task == "regression"
        else ("accuracy", "f1_weighted", "roc_auc", "log_loss")
    )
    figures = {}
    for name in names:
        points = [
            (fold["fold"], fold["metrics"][name])
            for fold in evidence.get("folds", [])
            if name in fold["metrics"]
        ]
        if not points or not all(math.isfinite(value) for _, value in points):
            continue
        note = evidence.get("status", "training folds")
        figure, (axis,) = chart_figure(f"Cross-validation: {name} ({note})")
        axis.plot(*zip(*points, strict=True), marker="o", color="#a855f7")
        axis.set(xlabel="Fold", ylabel=name)
        axis.set_xticks([index for index, _ in points])
        figures[f"cv_{name}"] = figure
    return figures


def competition_charts(selection: dict) -> dict:
    """Compare only common-policy training folds, highlighting the already selected winner."""
    rows = selection["leaderboard"]
    figure, (axis,) = chart_figure("Model competition — training CV selection")
    positions = list(range(len(rows)))
    colors = ["#22c55e" if row["candidate"] == selection["winner"] else "#a855f7" for row in rows]
    axis.barh(
        positions, [row["mean"] for row in rows], xerr=[row["std"] for row in rows], color=colors
    )
    axis.set_yticks(positions, [row["candidate"] for row in rows])
    axis.invert_yaxis()
    axis.set_xlabel(f"{selection['selection_metric']} — {selection['direction']} (mean ± fold SD)")
    return {"leaderboard": figure}


def model_set_charts(components: dict) -> dict:
    """Present separate objective rows; never rank or average unrelated target metrics."""
    figure, (axis,) = chart_figure("Model set — independent targets (no common ranking)")
    rows = []
    for name, item in components.items():
        comparison = item["comparison"]
        metric = comparison["metric"]
        rows.append(
            [
                name,
                metric,
                f"{comparison['candidate_metrics'][metric]:.5g}",
                str(item["holdout_rows"]),
            ]
        )
    axis.axis("off")
    table = axis.table(
        cellText=rows,
        colLabels=["Component", "Holdout metric", "Value", "Holdout rows"],
        loc="center",
        cellLoc="left",
    )
    table.auto_set_font_size(False)
    table.set_fontsize(10)
    table.scale(1, 1.7)
    return {"overview": figure}
