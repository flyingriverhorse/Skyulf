import sys
from pathlib import Path

import numpy as np

REPO_ROOT = Path(__file__).resolve().parents[1]

try:
    from core.feature_engineering.modeling import model_training_tasks as training_tasks
except ImportError:  # pragma: no cover - allow running without installation
    if str(REPO_ROOT) not in sys.path:
        sys.path.insert(0, str(REPO_ROOT))
    from core.feature_engineering.modeling import model_training_tasks as training_tasks


class _PredictiveModel:
    def __init__(self, values: np.ndarray):
        self._values = values

    def predict(self, _):
        return self._values


def test_regression_metrics_handles_metric_exceptions(monkeypatch):
    y_true = np.array([1.0, 2.0, 3.0], dtype=float)
    y_pred = np.array([1.1, 1.9, 3.2], dtype=float)
    model = _PredictiveModel(y_pred)

    def failing_r2(_y_true, _y_pred):
        raise ValueError("cannot compute r2")

    def failing_mape(_y_true, _y_pred):
        raise ValueError("cannot compute mape")

    monkeypatch.setattr(training_tasks, "r2_score", failing_r2)
    monkeypatch.setattr(training_tasks, "mean_absolute_percentage_error", failing_mape)

    metrics = training_tasks._regression_metrics(model, y_true.reshape(-1, 1), y_true)

    assert metrics["mae"] > 0
    assert metrics["rmse"] > 0
    assert np.isnan(metrics["r2"])
    assert np.isnan(metrics["mape"])

