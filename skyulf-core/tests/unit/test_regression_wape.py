"""WAPE adds a finite relative error without disguising an all-zero target population."""

import numpy as np
import pandas as pd
import pytest

from skyulf.modeling._evaluation.metrics import calculate_regression_metrics


def _metrics(truth, prediction, **kwargs):
    """Use saved predictions so the test cannot hide an extra estimator prediction call."""
    return calculate_regression_metrics(
        None,
        pd.DataFrame(),
        truth,
        X_np=np.empty((len(truth), 0)),
        y_np=np.asarray(truth, dtype=float),
        predictions=np.asarray(prediction, dtype=float),
        **kwargs,
    )


def test_wape_stays_finite_with_zero_actuals_and_preserves_mape():
    """Zero outcomes must not dominate the new aggregate relative-error measure."""
    values = _metrics([0, 10, 20], [5, 12, 18])
    assert values["wape"] == pytest.approx(9 / 30)
    assert values["mae"] == 3.0
    assert values["mape"] > 1e12


@pytest.mark.parametrize("prediction", [[0, 0], [1, 2]])
def test_all_zero_actuals_leave_wape_unavailable(prediction):
    """Even perfect all-zero predictions have no defined relative-error denominator."""
    values = _metrics([0, 0], prediction)
    assert "wape" not in values
    assert np.isfinite(list(values.values())).all()


def test_wape_uses_absolute_truth_including_negative_values():
    """Signed outcomes must not cancel the denominator or create a negative loss."""
    assert _metrics([-10, 10], [-8, 13])["wape"] == 0.25


def test_wape_adds_only_one_key_to_existing_unweighted_results():
    """The new diagnostic must not change the meaning or shape of existing regression results."""
    values = _metrics([0, 10, 20], [5, 12, 18])
    assert set(values) == {"mae", "mse", "rmse", "r2", "mape", "explained_variance", "wape"}
    assert values["mse"] == 11.0
