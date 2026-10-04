"""Self-contained, bounded SHAP charts for MLflow and Databricks notebooks."""

import base64
import io
import logging
from collections.abc import Callable
from html import escape
from pathlib import Path
from tempfile import TemporaryDirectory
from typing import Any


def _text(value: Any) -> str:
    """Escape feature names, labels and runtime messages as plain HTML text."""
    return escape(str(value), quote=True)


def _figure(count: int) -> Any:
    """Create an isolated headless figure without changing the process plotting backend."""
    from matplotlib.backends.backend_agg import FigureCanvasAgg  # noqa: PLC0415
    from matplotlib.figure import Figure  # noqa: PLC0415

    figure = Figure(figsize=(10, max(3, 0.3 * count + 1.5)), layout="constrained")
    FigureCanvasAgg(figure)
    return figure


def _image(figure: Any, title: str) -> str:
    """Embed a portable PNG rather than requiring network scripts or a plotting server."""
    with io.BytesIO() as stream:
        figure.savefig(stream, format="png", dpi=100)
        encoded = base64.b64encode(stream.getvalue()).decode("ascii")
    figure.clear()
    return (
        f'<img alt="{_text(title)}" style="max-width:100%" src="data:image/png;base64,{encoded}">'
    )


def _importance_chart(importance: dict[str, float]) -> str:
    """Plot mean absolute contributions across all explained training samples."""
    entries = sorted(importance.items(), key=lambda item: item[1], reverse=True)
    figure = _figure(len(entries))
    axes = figure.subplots()
    axes.barh(range(len(entries)), [value for _, value in entries], color="#2563eb")
    axes.set_yticks(range(len(entries)), [name[:80] for name, _ in entries])
    axes.invert_yaxis()
    axes.set_xlabel("Mean absolute SHAP value (model output units)")
    axes.set_title("Global feature importance")
    return _image(figure, "Global feature importance")


def _waterfall(sample: dict[str, Any]) -> str:
    """Accumulate signed contributions from the base value to the explained output."""
    entries = sorted(sample["shap_values"].items(), key=lambda item: abs(item[1]), reverse=True)
    figure = _figure(len(entries))
    axes = figure.subplots()
    running = float(sample["base_value"])
    for position, (_, contribution) in enumerate(entries):
        end = running + contribution
        axes.barh(
            position,
            abs(contribution),
            left=min(running, end),
            color="#dc2626" if contribution >= 0 else "#2563eb",
        )
        running = end
    axes.axvline(sample["base_value"], color="#64748b", linestyle="--", label="Base")
    axes.axvline(running, color="#111827", label="Explained output")
    axes.set_yticks(range(len(entries)), [name[:80] for name, _ in entries])
    axes.invert_yaxis()
    axes.set_xlabel("SHAP output units; red increases, blue decreases")
    axes.legend()
    return _image(figure, "Signed SHAP waterfall")


def _sample_report(sample: dict[str, Any], index: int) -> str:
    """Keep exact transformed values and signed contributions alongside each chart."""
    base = float(sample["base_value"])
    output = base + sum(sample["shap_values"].values())
    rows = "".join(
        f"<tr><td>{_text(name)}</td><td>{_text(sample['feature_values'].get(name))}</td>"
        f"<td>{value:+.6g}</td></tr>"
        for name, value in sample["shap_values"].items()
    )
    return (
        f"<details{' open' if index == 1 else ''}><summary>Sample {index}</summary>"
        f"<p>Base: {base:.6g} &rarr; Explained output: {output:.6g}</p>"
        + _waterfall(sample)
        + "<table><thead><tr><th>Transformed feature</th><th>Value</th>"
        "<th>Signed contribution</th></tr></thead><tbody>" + rows + "</tbody></table></details>"
    )


def _reading_guide() -> str:
    """Explain chart terms without introducing synthetic results into the report."""
    return (
        "<h3>How to read this report</h3>"
        "<p>SHAP helps answer: <strong>Why did the model make this prediction?</strong> "
        "It shows how each input moves the explained output up or down from a reference value.</p>"
        "<ul><li><strong>Base:</strong> the model's reference output for the explanation. "
        "It is not necessarily the average observed target.</li>"
        "<li><strong>Value:</strong> the input seen by the model after preprocessing. "
        "It may be scaled or encoded; its contribution is shown separately.</li>"
        "<li><strong>Signed contribution:</strong> how much that feature moves this output. "
        "Positive (red) raises it; negative (blue) lowers it. A larger absolute contribution "
        "means a stronger effect on this prediction. Higher does not always mean better.</li>"
        "<li><strong>Explained output:</strong> Base + all signed contributions.</li></ul>"
        "<p><strong>Global feature importance:</strong> longer bars mean larger average "
        "absolute contributions across the explained training rows. This chart shows "
        "which features matter most in that sample, but not whether they raise or lower "
        "predictions. <strong>Individual predictions:</strong> each waterfall shows "
        "the direction and size of contributions for one row.</p>"
        "<p><strong>Output units:</strong> regression explanations commonly use the target's "
        "units. Classification explanations may use probabilities or raw scores/log-odds; "
        "do not read a contribution as a percentage unless the explained output is known "
        "to be a probability.</p>"
        "<p>These charts explain model behavior. They do not prove the prediction is correct "
        "or that changing a feature will cause the same change in the real world.</p>"
    )


def render_explanation_report(evidence: dict[str, Any]) -> str:
    """Render saved evidence only; plotting never fits preprocessing or calls the model."""
    sections = [
        '<section style="font-family:system-ui;max-width:1100px;margin:auto">',
        "<h2>Model explanations</h2>",
        f"<p>Status: {_text(evidence.get('status', 'unavailable'))}</p>",
        "<p>Training samples only; holdout rows are not explained. Feature names and values "
        "are after fitted preprocessing. Importance describes model behavior, not causality.</p>",
    ]
    if evidence.get("run_id"):
        url = f"/ml/experiments/{evidence['experiment_id']}/runs/{evidence['run_id']}"
        sections.append(
            f'<p><a href="{_text(url)}" target="_blank" rel="noopener">MLflow run and artifacts</a>'
            f" &middot; Model: {_text(evidence.get('model_uri'))}</p>"
        )
    if evidence.get("status") != "completed":
        sections.append(f"<p>Reason: {_text(evidence.get('reason', 'not_requested'))}</p>")
        return "".join(sections) + "</section>"
    shap = evidence["shap"]
    sections.extend(
        [
            _reading_guide(),
            f"<p>Explained training rows: {evidence['sample_count']}; "
            f"transformed features: {evidence['feature_count']}. "
            "The feature budget is a guard, not top-K feature selection.</p>",
            "<h3>Global feature importance</h3>",
            _importance_chart(shap["mean_abs_importance"]),
            "<h3>Individual predictions</h3><p>Base plus signed contributions gives the explained "
            "model output, not necessarily a probability. Classifiers may use raw scores/log-odds. "
            "For multi-output SHAP, binary explanations use class index 1; multiclass explanations "
            "use each sample's predicted class. Sample numbers are report positions, not record IDs.</p>",
        ]
    )
    sections.extend(
        _sample_report(sample, index) for index, sample in enumerate(shap["samples"], 1)
    )
    if not shap["samples"]:
        sections.append("<p>Per-row display is disabled (max_display_samples=0).</p>")
    return "".join(sections) + "</section>"


def _run_ids(value: Any) -> list[str]:
    """Collect concrete child run references from single, competition and set results."""
    if isinstance(value, list):
        return [run_id for item in value for run_id in _run_ids(item)]
    if not isinstance(value, dict):
        return []
    found = []
    for key, item in value.items():
        found.extend(_reference_run_ids(key, item))
    return list(dict.fromkeys(found))


def _reference_run_ids(key: str, item: Any) -> list[str]:
    """Read only known run reference fields, never arbitrary strings in a result."""
    if key == "run_id" and isinstance(item, str):
        return [item]
    if key == "model_uri" and isinstance(item, str) and item.startswith("runs:/"):
        return [item.split("/")[1]]
    return _run_ids(item)


def display_explanation_reports(
    payload: dict[str, Any],
    client: Any,
    display_html: Callable[[str], Any],
) -> None:
    """Display saved training reports without putting images into task values or exit JSON."""
    for run_id in dict.fromkeys(_run_ids(payload)):
        try:
            if not any(item.path == "explanations.html" for item in client.list_artifacts(run_id)):
                continue
            with TemporaryDirectory(prefix="skyulf-explanation-") as directory:
                path = Path(client.download_artifacts(run_id, "explanations.html", directory))
                if path.stat().st_size > 10 * 1024 * 1024:
                    raise ValueError("Explanation report exceeds the 10 MiB display limit.")
                display_html(path.read_text(encoding="utf-8"))
        except Exception:  # noqa: BLE001 - display failure must not retry completed training
            logging.getLogger(__name__).warning("Explanation report unavailable for run %s", run_id)
            display_html(
                f"<p>Explanation report unavailable; inspect MLflow run {_text(run_id)}.</p>"
            )


def copy_winner_explanations(store: Any, selection: dict[str, Any]) -> None:
    """Retain the selected child's immutable explanation and run provenance on its parent."""
    winner = next(
        row for row in selection["leaderboard"] if row["candidate"] == selection["winner"]
    )
    with TemporaryDirectory(prefix="skyulf-winner-explanation-") as directory:
        for name in ("explanations.json", "explanations.html"):
            downloaded = store.client.download_artifacts(winner["run_id"], name, directory)
            store.client.log_artifact(store.run_id, downloaded)
