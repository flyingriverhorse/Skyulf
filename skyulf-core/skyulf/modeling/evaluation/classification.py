"""Classification evaluation logic."""

from __future__ import annotations

from typing import Any, List, Optional, Union

import numpy as np
import pandas as pd
from sklearn.metrics import confusion_matrix, precision_recall_curve, roc_curve
from sklearn.preprocessing import label_binarize

from ...engines import SkyulfDataFrame
from ...modeling.sklearn_wrapper import SklearnBridge
from .common import downsample_curve, sanitize_metrics
from .metrics import calculate_classification_metrics
from .schemas import (
    ClassificationEvaluation,
    ConfusionMatrixData,
    CurveData,
    ModelEvaluationReport,
)


def evaluate_classification_model(
    model: Any,
    X_test: Union[pd.DataFrame, SkyulfDataFrame],
    y_test: Union[pd.Series, Any],
    X_train: Optional[Union[pd.DataFrame, SkyulfDataFrame]] = None,
    y_train: Optional[Union[pd.Series, Any]] = None,
    dataset_name: str = "test",
) -> ModelEvaluationReport:
    """Evaluate a classification model and return a structured report."""

    # Convert to Numpy for compatibility
    X_test_np, y_test_np = SklearnBridge.to_sklearn((X_test, y_test))

    # Calculate scalar metrics
    metrics = calculate_classification_metrics(model, X_test, y_test)

    # Generate predictions and probabilities
    y_pred = model.predict(X_test_np)
    y_prob = None
    if hasattr(model, "predict_proba"):
        try:
            y_prob = model.predict_proba(X_test_np)
        except Exception:
            pass

    # Determine classes
    classes = getattr(model, "classes_", None)
    if classes is None:
        classes = np.unique(y_test_np)
    class_names = [str(c) for c in classes]
    n_classes = len(classes)

    # Confusion Matrix
    cm_data = _compute_confusion_matrix(y_test_np, y_pred, class_names)

    # ROC and PR Curves
    roc_curves = []
    pr_curves = []

    if y_prob is not None:
        if n_classes == 2:
            # Binary classification
            # Assuming positive class is at index 1
            pos_label = classes[1]
            pos_probs = y_prob[:, 1]

            # ROC
            fpr, tpr, _ = roc_curve(y_test_np, pos_probs, pos_label=pos_label)
            roc_curves.append(
                CurveData(
                    name=f"ROC (Class {class_names[1]})",
                    points=downsample_curve(fpr, tpr),
                    auc=metrics.get("roc_auc"),
                )
            )

            # PR
            precision, recall, _ = precision_recall_curve(
                y_test_np, pos_probs, pos_label=pos_label
            )
            pr_curves.append(
                CurveData(
                    name=f"PR (Class {class_names[1]})",
                    points=downsample_curve(recall, precision),
                    auc=metrics.get("pr_auc"),
                )
            )
        else:
            # Multiclass classification
            y_test_bin = label_binarize(y_test_np, classes=classes)

            for i, class_name in enumerate(class_names):
                # ROC
                fpr, tpr, _ = roc_curve(y_test_bin[:, i], y_prob[:, i])
                roc_curves.append(
                    CurveData(
                        name=f"ROC (Class {class_name})",
                        points=downsample_curve(fpr, tpr),
                    )
                )

                # PR
                precision, recall, _ = precision_recall_curve(
                    y_test_bin[:, i], y_prob[:, i]
                )
                pr_curves.append(
                    CurveData(
                        name=f"PR (Class {class_name})",
                        points=downsample_curve(recall, precision),
                    )
                )

    classification_eval = ClassificationEvaluation(
        confusion_matrix=cm_data,
        roc_curves=roc_curves,
        pr_curves=pr_curves,
    )

    return ModelEvaluationReport(
        dataset_name=dataset_name,
        metrics=sanitize_metrics(metrics),
        classification=classification_eval,
        regression=None,
    )


def _compute_confusion_matrix(
    y_true: Any, y_pred: Any, labels: List[str]
) -> ConfusionMatrixData:
    """Compute confusion matrix data."""
    cm = confusion_matrix(y_true, y_pred, labels=labels)

    # Convert to list of lists for JSON serialization
    matrix_data = cm.tolist()

    return ConfusionMatrixData(labels=labels, matrix=matrix_data)
