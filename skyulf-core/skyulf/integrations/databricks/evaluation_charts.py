"""Lazy, bounded prediction diagnostics for MLflow Image grid reports."""

from itertools import combinations
from typing import Any

import numpy as np
from sklearn import metrics
from sklearn.calibration import calibration_curve
from sklearn.svm import SVC, NuSVC


def chart_figure(title: str, *, panels: int = 1) -> tuple[Any, Any]:
    """Create legible figures without requiring Matplotlib for ordinary training."""
    from matplotlib.figure import Figure  # noqa: PLC0415 - optional visualization dependency

    figure = Figure(figsize=(7 * panels, 5), layout="constrained", facecolor="white")
    axes = np.atleast_1d(figure.subplots(1, panels))
    figure.suptitle(title, fontsize=14, fontweight="bold")
    for axis in axes:
        axis.spines[["top", "right"]].set_visible(False)
        axis.grid(alpha=0.15)
        axis.set_axisbelow(True)
    return figure, axes


def diagnostic_charts(
    task: str,
    observed: Any,
    predictions: Any,
    classes: tuple,
    max_classes: int,
) -> tuple[dict[str, Any], list[str]]:
    """Plot paired saved holdout predictions, preserving the fitted class vocabulary."""
    actual, predicted = np.asarray(observed), predictions["prediction"].to_numpy()
    if len(actual) < 2:
        return {}, ["Holdout sample has fewer than two rows."]
    if task == "regression":
        return _regression(actual.astype(float), predicted.astype(float)), []
    if len(classes) > max_classes:
        return {}, [f"Classification charts exceed max_classes={max_classes}."]
    if not np.isin(actual, classes).all() or not np.isin(predicted, classes).all():
        raise ValueError("Chart labels differ from the fitted class vocabulary.")
    figures = _class_counts(actual, predicted, classes)
    columns = [f"probability_{index}" for index in range(len(classes))]
    if not set(columns) <= set(predictions.columns):
        return figures, ["Model provides no class probabilities; ROC/PR and calibration skipped."]
    probabilities = predictions[columns].to_numpy(dtype=float)
    _validate_probabilities(probabilities)
    probability_figures, skipped = _probability_charts(actual, probabilities, classes)
    return {**figures, **probability_figures}, skipped


def _regression(actual: np.ndarray, predicted: np.ndarray) -> dict[str, Any]:
    """Show fit, heteroscedasticity and the signed prediction-error distribution."""
    if not np.isfinite(actual).all() or not np.isfinite(predicted).all():
        raise ValueError("Regression chart values must be finite.")
    figures = {}
    figure, (axis,) = chart_figure("Actual versus predicted")
    axis.scatter(actual, predicted, color="#22c55e", alpha=0.55, s=14)
    bounds = [min(actual.min(), predicted.min()), max(actual.max(), predicted.max())]
    axis.plot(bounds, bounds, color="#64748b", linestyle="--")
    axis.set(xlabel="Actual", ylabel="Predicted")
    figures["actual_vs_predicted"] = figure
    residuals = predicted - actual
    figure, (axis,) = chart_figure("Residuals versus predicted")
    axis.scatter(predicted, residuals, color="#a855f7", alpha=0.55, s=14)
    axis.axhline(0, color="#64748b", linestyle="--")
    axis.set(xlabel="Predicted", ylabel="Predicted minus actual")
    figures["residuals"] = figure
    figure, (axis,) = chart_figure("Residual distribution")
    axis.hist(residuals, bins=min(30, max(2, int(np.sqrt(len(actual))))), color="#a855f7")
    axis.axvline(0, color="#334155", linestyle="--")
    axis.set(xlabel="Predicted minus actual", ylabel="Rows")
    figures["residual_distribution"] = figure
    return figures


def _class_counts(actual: np.ndarray, predicted: np.ndarray, classes: tuple) -> dict[str, Any]:
    """Include absent classes and show both raw support and class-normalized errors."""
    figure, axes = chart_figure("Confusion matrix", panels=2)
    for axis, normalize, title in zip(
        axes, (None, "true"), ("Row counts", "Fraction of actual class"), strict=True
    ):
        matrix = metrics.confusion_matrix(
            actual, predicted, labels=list(classes), normalize=normalize
        )
        display = metrics.ConfusionMatrixDisplay(matrix, display_labels=classes)
        display.plot(
            ax=axis,
            colorbar=False,
            cmap="Blues",
            xticks_rotation=45,
            include_values=len(classes) <= 10,
        )
        axis.set_title(title)
        axis.grid(False)
    summary, (axis,) = chart_figure("Per-class precision, recall and F1")
    precision, recall, f1, support = metrics.precision_recall_fscore_support(
        actual, predicted, labels=list(classes), zero_division=0
    )
    positions = np.arange(len(classes))
    for offset, values, label, color in zip(
        (-0.25, 0, 0.25),
        (precision, recall, f1),
        ("Precision", "Recall", "F1"),
        ("#3b82f6", "#22c55e", "#a855f7"),
        strict=True,
    ):
        axis.bar(positions + offset, values, width=0.25, label=label, color=color)
    axis.set_xticks(
        positions,
        [f"{label}\nn={count}" for label, count in zip(classes, support, strict=True)],
        rotation=45,
    )
    axis.set(ylim=(0, 1.05), ylabel="Score (undefined values shown as 0)")
    axis.legend()
    return {"confusion_matrix": figure, "class_metrics": summary}


def _validate_probabilities(values: np.ndarray) -> None:
    """Reject invalid probabilities before plotting calibrated-looking curves."""
    if not np.isfinite(values).all() or (values < 0).any() or (values > 1).any():
        raise ValueError("Chart probabilities must be finite and between zero and one.")
    if not np.allclose(values.sum(axis=1), 1, atol=1e-5):
        raise ValueError("Chart class probabilities must sum to one.")


def _probability_charts(
    actual: np.ndarray, probabilities: np.ndarray, classes: tuple
) -> tuple[dict, list[str]]:
    """Use the fitted positive class for binary, and one-versus-rest for multiclass."""
    curves, (roc, pr) = chart_figure("ROC and precision–recall (holdout sample)", panels=2)
    calibration, (axis,) = chart_figure("Probability calibration (holdout sample)")
    skipped = []
    indices = [1] if len(classes) == 2 else range(len(classes))
    plotted = 0
    for index in indices:
        binary = actual == classes[index]
        if binary.all() or not binary.any():
            skipped.append(
                f"Class {classes[index]}: positive or negative examples absent from sample."
            )
            continue
        _plot_probability_class(roc, pr, axis, binary, probabilities[:, index], str(classes[index]))
        plotted += 1
    if not plotted:
        return {}, skipped
    roc.plot([0, 1], [0, 1], "--", color="#94a3b8")
    roc.set(xlabel="False positive rate", ylabel="True positive rate")
    pr.set(xlabel="Recall", ylabel="Precision")
    axis.plot([0, 1], [0, 1], "--", color="#94a3b8", label="Perfect calibration")
    axis.set(xlabel="Mean predicted probability", ylabel="Observed positive fraction")
    for item in (roc, pr, axis):
        item.set(xlim=(0, 1), ylim=(0, 1.05))
        item.legend(fontsize="small")
    return {"roc_pr": curves, "calibration": calibration}, skipped


def _plot_probability_class(
    roc: Any, pr: Any, axis: Any, binary: np.ndarray, scores: np.ndarray, label: str
) -> None:
    """Annotate sample-specific discrimination without overwriting full-holdout metrics."""
    fpr, tpr, _ = metrics.roc_curve(binary, scores)
    precision, recall, _ = metrics.precision_recall_curve(binary, scores)
    roc.plot(fpr, tpr, label=f"{label}: AUC={metrics.auc(fpr, tpr):.3f}")
    pr.plot(
        recall,
        precision,
        label=f"{label}: AP={metrics.average_precision_score(binary, scores):.3f}",
    )
    pr.axhline(binary.mean(), alpha=0.25, linestyle=":")
    fraction, mean = calibration_curve(binary, scores, n_bins=10, strategy="quantile")
    axis.plot(mean, fraction, marker="o", markersize=3, label=label)


def model_charts(
    model: Any, features: tuple, classes: tuple, max_features: int
) -> tuple[dict, list[str]]:
    """Report native importance or signed coefficients only when the estimator exposes them."""
    figures, skipped = {}, []
    if hasattr(model, "feature_importances_"):
        figures["feature_importance"] = _feature_bars(
            model.feature_importances_, features, max_features, "Native tree feature importance"
        )
    elif hasattr(model, "coef_"):
        if len(np.atleast_2d(model.coef_)) > 20:
            skipped.append("Coefficient output exceeds the 20-boundary chart budget.")
        else:
            figures.update(_coefficient_charts(model, features, classes, max_features))
    else:
        skipped.append("Estimator exposes no native feature importance or linear coefficients.")
    losses = getattr(model, "loss_curve_", None)
    if losses is not None and len(losses) > 1:
        figure, (axis,) = chart_figure("Recorded training loss (not holdout performance)")
        axis.plot(np.arange(1, len(losses) + 1), losses, color="#a855f7")
        axis.set(xlabel="Training iteration", ylabel="Training loss")
        figures["training_loss"] = figure
    return figures, skipped


def _coefficient_charts(model: Any, features: tuple, classes: tuple, limit: int) -> dict:
    """Label pairwise SVC boundaries separately from one-versus-rest class coefficients."""
    coefficients = np.atleast_2d(model.coef_)
    labels = _coefficient_labels(model, classes, len(coefficients))
    if len(labels) != len(coefficients):
        raise ValueError("Coefficient rows differ from the fitted class contract.")
    figures = {}
    for index, (label, values) in enumerate(zip(labels, coefficients, strict=True)):
        key = "coefficients" if len(coefficients) == 1 else f"coefficients_{index}"
        figures[key] = _feature_bars(
            values, features, limit, f"Signed coefficients (transformed units) — {label}"
        )
    return figures


def _coefficient_labels(model: Any, classes: tuple, rows: int) -> list[str]:
    """Distinguish pairwise SVM boundaries from ordinary class coefficients."""
    if isinstance(model, SVC | NuSVC) and len(classes) > 2:
        return [f"{left} versus {right}" for left, right in combinations(classes, 2)]
    if len(classes) == 2 and rows == 1:
        return [f"class {classes[1]}"]
    return [f"class {label}" for label in classes] or ["regression"]


def _feature_bars(values: Any, features: tuple, limit: int, title: str) -> Any:
    """Rank actual transformed features while retaining coefficient signs."""
    values = np.asarray(values, dtype=float)
    if values.ndim != 1 or len(values) != len(features) or not np.isfinite(values).all():
        raise ValueError("Chart feature values differ from the transformed feature contract.")
    order = np.argsort(np.abs(values))[-limit:]
    figure, (axis,) = chart_figure(title)
    figure.set_size_inches(9, max(4, len(order) * 0.28 + 1.5))
    axis.barh(np.arange(len(order)), values[order], color="#3b82f6")
    axis.set_yticks(np.arange(len(order)), [features[index] for index in order])
    axis.set_xlabel("Value")
    return figure
