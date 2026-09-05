"""Tests for skyulf.modeling._evaluation.metrics — verified against sklearn/hand-computed values."""

import numpy as np
import pandas as pd
import pytest
from sklearn.linear_model import LinearRegression, LogisticRegression
from sklearn.metrics import (
    accuracy_score,
    f1_score,
    mean_absolute_error,
    mean_squared_error,
    r2_score,
    roc_auc_score,
)
from sklearn.tree import DecisionTreeClassifier

from skyulf.modeling._evaluation import metrics as metrics_mod
from skyulf.modeling._evaluation.metrics import (
    calculate_classification_metrics,
    calculate_regression_metrics,
)


@pytest.fixture
def binary_data():
    """Small, deterministic binary-classification data with a fitted LogisticRegression."""
    rng = np.random.RandomState(0)
    X = pd.DataFrame(
        {"f1": rng.normal(0, 1, 60), "f2": rng.normal(0, 1, 60)},
    )
    y = pd.Series((X["f1"] + X["f2"] > 0).astype(int), name="target")
    model = LogisticRegression().fit(X, y)
    return model, X, y


@pytest.fixture
def multiclass_data():
    """Small deterministic 3-class dataset with a fitted DecisionTreeClassifier."""
    rng = np.random.RandomState(1)
    X = pd.DataFrame({"f1": rng.normal(0, 1, 90), "f2": rng.normal(0, 1, 90)})
    y = pd.Series(np.tile([0, 1, 2], 30), name="target")
    model = DecisionTreeClassifier(random_state=0).fit(X, y)
    return model, X, y


def test_classification_metrics_accuracy_matches_sklearn(binary_data):
    """accuracy metric must equal sklearn's accuracy_score on the same predictions."""
    model, X, y = binary_data
    metrics = calculate_classification_metrics(model, X, y)
    expected = accuracy_score(y, model.predict(X))
    assert metrics["accuracy"] == pytest.approx(expected)


def test_classification_metrics_f1_weighted_matches_sklearn(binary_data):
    """f1_weighted metric must equal sklearn's weighted f1_score."""
    model, X, y = binary_data
    metrics = calculate_classification_metrics(model, X, y)
    expected = f1_score(y, model.predict(X), average="weighted", zero_division=0)
    assert metrics["f1_weighted"] == pytest.approx(expected)


def test_classification_metrics_binary_adds_unweighted_variants(binary_data):
    """Binary classification should include unweighted precision/recall/f1 keys."""
    model, X, y = binary_data
    metrics = calculate_classification_metrics(model, X, y)
    assert "precision" in metrics
    assert "recall" in metrics
    assert "f1" in metrics


def test_classification_metrics_binary_non_01_string_labels_still_produce_precision_recall_f1():
    """Regression test: binary labels that aren't literally {0, 1} (e.g. string
    labels) must not silently drop precision/recall/f1. Previously
    average="binary" relied on sklearn's default pos_label=1, which raises for
    non-{0,1} labels — swallowed by a bare `except Exception: pass`."""
    rng = np.random.RandomState(0)
    X = pd.DataFrame({"f1": rng.normal(0, 1, 60), "f2": rng.normal(0, 1, 60)})
    y = pd.Series(np.where(X["f1"] + X["f2"] > 0, "yes", "no"), name="target")
    model = LogisticRegression().fit(X, y)

    metrics = calculate_classification_metrics(model, X, y)
    assert "precision" in metrics
    assert "recall" in metrics
    assert "f1" in metrics

    predictions = model.predict(X)
    pos_label = model.classes_[1]
    expected_f1 = f1_score(y, predictions, average="binary", pos_label=pos_label)
    assert metrics["f1"] == pytest.approx(expected_f1)


def test_classification_metrics_binary_negative_positive_int_labels():
    """Non-{0,1} integer binary labels (e.g. {-1, 1}) must also produce
    precision/recall/f1 using the correct positive-class label."""
    rng = np.random.RandomState(2)
    X = pd.DataFrame({"f1": rng.normal(0, 1, 60), "f2": rng.normal(0, 1, 60)})
    y = pd.Series(np.where(X["f1"] + X["f2"] > 0, 1, -1), name="target")
    model = LogisticRegression().fit(X, y)

    metrics = calculate_classification_metrics(model, X, y)
    predictions = model.predict(X)
    pos_label = model.classes_[1]
    expected_precision = f1_score(y, predictions, average="binary", pos_label=pos_label)
    assert metrics["f1"] == pytest.approx(expected_precision)


def test_classification_metrics_binary_roc_auc_matches_sklearn(binary_data):
    """roc_auc for binary classification should match sklearn's roc_auc_score."""
    model, X, y = binary_data
    metrics = calculate_classification_metrics(model, X, y)
    proba = model.predict_proba(X)
    expected = roc_auc_score(y, proba[:, 1])
    assert metrics["roc_auc"] == pytest.approx(expected)


def test_classification_metrics_multiclass_has_ovr_and_ovo_variants(multiclass_data):
    """Multiclass predictions should produce OVR/OVO roc_auc variants, not binary keys."""
    model, X, y = multiclass_data
    metrics = calculate_classification_metrics(model, X, y)
    assert "roc_auc_ovr" in metrics
    assert "roc_auc_ovo" in metrics
    assert "roc_auc_ovr_weighted" in metrics
    assert "roc_auc_ovo_weighted" in metrics
    assert "roc_auc" not in metrics


def test_classification_metrics_multiclass_survives_fold_missing_a_trained_class(multiclass_data):
    """Regression test: multiclass roc_auc/pr_auc must not silently disappear
    when the evaluated split doesn't contain every class the model was
    trained on (common with small/imbalanced CV folds). Previously
    roc_auc_score raised "Number of classes in y_true not equal to the number
    of columns in y_score", swallowed by a bare except."""
    model, X, y = multiclass_data
    # Evaluate on a subset containing only 2 of the 3 trained classes.
    mask = y != 2
    X_subset, y_subset = X[mask], y[mask]

    metrics = calculate_classification_metrics(model, X_subset, y_subset)
    assert "roc_auc_ovr" in metrics
    assert "roc_auc_ovo" in metrics
    assert "pr_auc_weighted" in metrics


def test_classification_metrics_multiclass_split_missing_class_keeps_multiclass_only(
    multiclass_data,
):
    """Regression test (OC-35): a 3-class model evaluated on a split that
    contains only two classes must NOT be treated as binary. Previously the
    binary gate looked at the unique labels in y_true, so such a split gained
    unweighted precision/recall/f1 keys that don't belong to a multiclass model."""
    model, X, y = multiclass_data
    mask = y != 2
    X_subset, y_subset = X[mask], y[mask]

    metrics = calculate_classification_metrics(model, X_subset, y_subset)
    assert "precision" not in metrics
    assert "recall" not in metrics
    assert "f1" not in metrics
    # The multiclass metrics that were always computable must still be present.
    assert "precision_weighted" in metrics
    assert "f1_weighted" in metrics


def test_classification_metrics_multiclass_split_missing_class_computes_log_loss(multiclass_data):
    """Regression test (OC-35): log_loss must be computed for a multiclass model
    on a split missing a class, using the full trained label set. Previously
    log_loss raised "2 vs 3. Please provide labels" and the metric was dropped."""
    from sklearn.metrics import log_loss

    model, X, y = multiclass_data
    mask = y != 2
    X_subset, y_subset = X[mask], y[mask]

    metrics = calculate_classification_metrics(model, X_subset, y_subset)
    assert "log_loss" in metrics
    expected = log_loss(y_subset, model.predict_proba(X_subset), labels=[0, 1, 2])
    assert metrics["log_loss"] == pytest.approx(expected)


def test_evaluate_classification_model_split_missing_class_emits_no_null_curve_points(
    multiclass_data,
):
    """Regression test (OC-35): per-class ROC/PR curves for a class absent from
    the evaluated split must be skipped, not emitted with NaN (null) points."""
    import math

    from skyulf.modeling._evaluation.classification import evaluate_classification_model

    model, X, y = multiclass_data
    mask = y != 2
    X_subset, y_subset = X[mask], y[mask]

    report = evaluate_classification_model(model, X_subset, y_subset)
    assert report.classification is not None
    for curve in [*report.classification.roc_curves, *report.classification.pr_curves]:
        assert curve.points, f"curve {curve.name} has no points"
        for point in curve.points:
            assert math.isfinite(point.x), f"curve {curve.name} has non-finite x"
            assert math.isfinite(point.y), f"curve {curve.name} has non-finite y"
    # The absent class (2) must not produce a curve at all.
    curve_names = [c.name for c in report.classification.roc_curves]
    assert "ROC (Class 2)" not in curve_names


def test_classification_metrics_multiclass_pr_auc_weighted_present(multiclass_data):
    """Multiclass predictions should include a weighted PR-AUC computed via label_binarize."""
    model, X, y = multiclass_data
    metrics = calculate_classification_metrics(model, X, y)
    assert "pr_auc_weighted" in metrics
    assert 0.0 <= metrics["pr_auc_weighted"] <= 1.0


# ---------------------------------------------------------------------------
# OC-146: binary pr_auc must score the class the model actually treats as
# positive, not sklearn's pos_label=1 default.
# ---------------------------------------------------------------------------


@pytest.fixture
def strong_signal_binary():
    """Binary data with a learnable signal, so pr_auc is high (~0.97) and an
    inverted computation (~0.32) is unmistakably distinguishable from it."""
    rng = np.random.RandomState(7)
    X = pd.DataFrame({"f1": rng.normal(0, 1, 400), "f2": rng.normal(0, 1, 400)})
    y = (X["f1"] - X["f2"] + rng.normal(0, 0.3, 400) > 0).astype(int).to_numpy()
    return X, y


@pytest.mark.parametrize(
    ("encoding", "relabel"),
    [
        ("{0,1}", lambda y: y),
        ("{1,2}", lambda y: y + 1),
        ("{1,5}", lambda y: np.where(y == 1, 5, 1)),
        ("{-1,1}", lambda y: np.where(y == 1, 1, -1)),
        ("{'no','yes'}", lambda y: np.where(y == 1, "yes", "no")),
    ],
)
def test_classification_metrics_pr_auc_invariant_under_label_reencoding(
    strong_signal_binary, encoding, relabel
):
    """Regression test (OC-146/OC-37): the same data under any binary label
    encoding must report the same pr_auc.

    ``average_precision_score`` defaults to ``pos_label=1`` — unlike
    ``roc_auc_score``, which infers the positive class from the sorted uniques.
    On ``{1,2}``/``{1,5}`` the literal ``1`` is the *negative* class, so PR-AUC
    was computed for the inverted problem (0.32 reported for a 0.97 model) with
    no warning; on string and other arbitrary labels it raised and the metric
    vanished from the report entirely.
    """
    from sklearn.metrics import average_precision_score

    X, y_raw = strong_signal_binary
    y = pd.Series(relabel(y_raw), name="target")
    model = LogisticRegression(max_iter=1000).fit(X, y)

    metrics = calculate_classification_metrics(model, X, y)

    assert "pr_auc" in metrics, f"pr_auc missing for {encoding} labels"
    expected = average_precision_score(y, model.predict_proba(X)[:, 1], pos_label=model.classes_[1])
    assert metrics["pr_auc"] == pytest.approx(expected), encoding


def test_classification_metrics_pr_auc_agrees_across_encodings(strong_signal_binary):
    """The {1,2} encoding must report the same pr_auc as the {0,1} control —
    the two disagreed by 3x (0.32 vs 0.97) before the pos_label fix."""
    X, y_raw = strong_signal_binary
    reported = {}
    for encoding, y in (("{0,1}", y_raw), ("{1,2}", y_raw + 1)):
        y_ser = pd.Series(y, name="target")
        model = LogisticRegression(max_iter=1000).fit(X, y_ser)
        reported[encoding] = calculate_classification_metrics(model, X, y_ser)["pr_auc"]

    assert reported["{1,2}"] == pytest.approx(reported["{0,1}"])
    assert reported["{1,2}"] > 0.9  # a strong model, not the inverted ~0.32


def test_classification_metrics_matthews_corrcoef_bounded(binary_data):
    """matthews_corrcoef should be within its valid [-1, 1] range."""
    model, X, y = binary_data
    metrics = calculate_classification_metrics(model, X, y)
    assert -1.0 <= metrics["matthews_corrcoef"] <= 1.0


def test_classification_metrics_no_predict_proba_skips_probability_metrics():
    """Models without predict_proba should not produce log_loss/roc_auc keys."""
    from sklearn.svm import LinearSVC

    rng = np.random.RandomState(2)
    X = pd.DataFrame({"f1": rng.normal(0, 1, 40), "f2": rng.normal(0, 1, 40)})
    y = pd.Series((X["f1"] > 0).astype(int))
    model = LinearSVC().fit(X, y)
    metrics = calculate_classification_metrics(model, X, y)
    assert "log_loss" not in metrics
    assert "roc_auc" not in metrics
    assert "accuracy" in metrics


def test_regression_metrics_mae_matches_sklearn():
    """mae metric must equal sklearn's mean_absolute_error."""
    rng = np.random.RandomState(3)
    X = pd.DataFrame({"f1": rng.normal(0, 1, 50)})
    y = pd.Series(2 * X["f1"] + 1 + rng.normal(0, 0.1, 50))
    model = LinearRegression().fit(X, y)
    metrics = calculate_regression_metrics(model, X, y)
    expected = mean_absolute_error(y, model.predict(X))
    assert metrics["mae"] == pytest.approx(expected)


def test_regression_metrics_rmse_is_sqrt_of_mse():
    """rmse must equal the square root of mse for consistency."""
    rng = np.random.RandomState(4)
    X = pd.DataFrame({"f1": rng.normal(0, 1, 50)})
    y = pd.Series(3 * X["f1"] - 2 + rng.normal(0, 0.1, 50))
    model = LinearRegression().fit(X, y)
    metrics = calculate_regression_metrics(model, X, y)
    assert metrics["rmse"] == pytest.approx(metrics["mse"] ** 0.5)


def test_regression_metrics_r2_matches_sklearn():
    """r2 metric must equal sklearn's r2_score."""
    rng = np.random.RandomState(5)
    X = pd.DataFrame({"f1": rng.normal(0, 1, 50)})
    y = pd.Series(X["f1"] * 5 + rng.normal(0, 0.2, 50))
    model = LinearRegression().fit(X, y)
    metrics = calculate_regression_metrics(model, X, y)
    expected = r2_score(y, model.predict(X))
    assert metrics["r2"] == pytest.approx(expected)


def test_regression_metrics_mse_matches_sklearn():
    """mse metric must equal sklearn's mean_squared_error."""
    rng = np.random.RandomState(6)
    X = pd.DataFrame({"f1": rng.normal(0, 1, 30)})
    y = pd.Series(X["f1"] + rng.normal(0, 0.05, 30))
    model = LinearRegression().fit(X, y)
    metrics = calculate_regression_metrics(model, X, y)
    expected = mean_squared_error(y, model.predict(X))
    assert metrics["mse"] == pytest.approx(expected)


# ---------------------------------------------------------------------------
# Exception-swallowing branches (all guarded blocks must not propagate errors)
# ---------------------------------------------------------------------------


def test_binary_unweighted_metrics_exception_is_swallowed(monkeypatch, binary_data):
    """If precision_score raises for the binary block, precision/recall/f1 must be absent."""
    model, X, y = binary_data
    original = metrics_mod.precision_score

    def flaky(y_true, y_pred, average="binary", **kwargs):
        if average == "binary":
            raise ValueError("boom")
        return original(y_true, y_pred, average=average, **kwargs)

    monkeypatch.setattr(metrics_mod, "precision_score", flaky)
    result = calculate_classification_metrics(model, X, y)
    assert "precision" not in result
    assert "precision_weighted" in result


def test_geometric_mean_score_exception_is_swallowed(monkeypatch, binary_data):
    """If geometric_mean_score raises, g_score must simply be absent from the result."""
    model, X, y = binary_data

    def flaky(*args, **kwargs):
        raise ValueError("boom")

    monkeypatch.setattr(metrics_mod, "geometric_mean_score", flaky)
    result = calculate_classification_metrics(model, X, y)
    assert "g_score" not in result
    assert "accuracy" in result


def test_log_loss_exception_is_swallowed(monkeypatch, binary_data):
    """If log_loss raises, log_loss must be absent but roc_auc/pr_auc still computed."""
    model, X, y = binary_data

    def flaky(*args, **kwargs):
        raise ValueError("boom")

    monkeypatch.setattr(metrics_mod, "log_loss", flaky)
    result = calculate_classification_metrics(model, X, y)
    assert "log_loss" not in result
    assert "roc_auc" in result


def test_multiclass_pr_auc_weighted_exception_is_swallowed(monkeypatch, multiclass_data):
    """If roc_auc_score raises inside the multiclass block, roc_auc_ovo must be absent."""
    model, X, y = multiclass_data

    def flaky(*args, **kwargs):
        raise ValueError("boom")

    monkeypatch.setattr(metrics_mod, "roc_auc_score", flaky)
    result = calculate_classification_metrics(model, X, y)
    assert "roc_auc_ovo" not in result
    assert "accuracy" in result


def test_predict_proba_outer_exception_is_swallowed(monkeypatch, binary_data):
    """If predict_proba itself raises, the whole probability block must be skipped safely."""
    model, X, y = binary_data

    class _BrokenProbaModel:
        def __init__(self, inner):
            self._inner = inner

        def predict(self, X):
            return self._inner.predict(X)

        def predict_proba(self, X):
            raise RuntimeError("boom")

    broken = _BrokenProbaModel(model)
    result = calculate_classification_metrics(broken, X, y)
    assert "roc_auc" not in result
    assert "accuracy" in result


def test_multiclass_classes_fallback_when_model_lacks_classes_attr(monkeypatch, multiclass_data):
    """When model.classes_ is missing/mismatched, classes must fall back to np.arange."""
    model, X, y = multiclass_data

    class _NoClassesModel:
        def __init__(self, inner):
            self._inner = inner

        def predict(self, X):
            return self._inner.predict(X)

        def predict_proba(self, X):
            return self._inner.predict_proba(X)

    wrapped = _NoClassesModel(model)
    result = calculate_classification_metrics(wrapped, X, y)
    assert "pr_auc_weighted" in result


def test_imblearn_import_failure_leaves_geometric_mean_score_none(monkeypatch):
    """If imblearn.metrics is unimportable, geometric_mean_score must fall back to None."""
    import importlib
    import sys

    monkeypatch.setitem(sys.modules, "imblearn.metrics", None)
    try:
        importlib.reload(metrics_mod)
        assert metrics_mod.geometric_mean_score is None
    finally:
        monkeypatch.delitem(sys.modules, "imblearn.metrics", raising=False)
        importlib.reload(metrics_mod)


def test_regression_metrics_returns_all_expected_keys():
    """calculate_regression_metrics should always return a fixed set of keys."""
    rng = np.random.RandomState(7)
    X = pd.DataFrame({"f1": rng.normal(0, 1, 20)})
    y = pd.Series(X["f1"] + 1)
    model = LinearRegression().fit(X, y)
    metrics = calculate_regression_metrics(model, X, y)
    assert set(metrics.keys()) == {"mae", "mse", "rmse", "r2", "mape", "explained_variance"}


# ---------------------------------------------------------------------------
# F-05: a single failing metric must be isolated and logged, not silently
# swallowed together with the metrics that follow it.
# ---------------------------------------------------------------------------


def test_one_failing_metric_does_not_drop_siblings(caplog, monkeypatch, binary_data):
    """When roc_auc fails, only roc_auc is omitted (and logged); pr_auc,
    log_loss and the base metrics must still be present."""
    import logging

    model, X, y = binary_data

    def boom(*args, **kwargs):
        raise ValueError("simulated roc_auc failure")

    monkeypatch.setattr(metrics_mod, "roc_auc_score", boom)
    with caplog.at_level(logging.WARNING):
        metrics = calculate_classification_metrics(model, X, y)

    assert "roc_auc" not in metrics  # the failing metric is omitted
    assert "pr_auc" in metrics  # its sibling survives
    assert "log_loss" in metrics  # so does log_loss
    assert "accuracy" in metrics  # and the base metrics too
    assert any("roc_auc" in message for message in caplog.messages)  # failure logged


def test_metric_failure_is_omitted_not_nan(caplog, monkeypatch, binary_data):
    """A failed metric must be absent (sanitize_metrics strips non-finite values
    downstream), never recorded as nan which would poison tuning comparisons."""
    import logging
    import math

    model, X, y = binary_data

    def boom(*args, **kwargs):
        raise ValueError("simulated pr_auc failure")

    monkeypatch.setattr(metrics_mod, "average_precision_score", boom)
    with caplog.at_level(logging.WARNING):
        metrics = calculate_classification_metrics(model, X, y)

    assert "pr_auc" not in metrics
    assert not any(isinstance(v, float) and math.isnan(v) for v in metrics.values())
    assert "roc_auc" in metrics


class TestFailureBranches:
    """Direct coverage of the isolated-failure branches (Codecov patch follow-up)."""

    def test_try_add_metric_value_error_omits_and_warns(self, caplog):
        import logging

        def boom(*args, **kwargs):
            raise ValueError("bad input")

        metrics = {}
        with caplog.at_level(logging.WARNING):
            metrics_mod._try_add_metric(metrics, "score", boom, 1, 2)

        assert "score" not in metrics
        assert any("score" in rec.message for rec in caplog.records)

    def test_try_add_metric_type_error_omits_and_warns(self, caplog):
        import logging

        def boom(*args, **kwargs):
            raise TypeError("wrong types")

        metrics = {"other": 1.0}
        with caplog.at_level(logging.WARNING):
            metrics_mod._try_add_metric(metrics, "score", boom)

        assert metrics == {"other": 1.0}

    def test_probability_metrics_predict_proba_failure_is_noop(self):
        class BrokenProba:
            def predict_proba(self, X):
                raise RuntimeError("model-specific failure")

        metrics = {"accuracy": 0.9}
        metrics_mod._add_probability_based_metrics(
            metrics, BrokenProba(), np.zeros((4, 2)), np.array([0, 1, 0, 1])
        )
        assert metrics == {"accuracy": 0.9}

    def test_binary_unweighted_label_resolution_failure_is_noop(self, caplog):
        import logging

        metrics = {"accuracy": 0.8}
        y_arr = np.array([1, "a"], dtype=object)  # np.unique raises TypeError
        with caplog.at_level(logging.WARNING):
            metrics_mod._add_binary_unweighted_metrics(metrics, None, y_arr, y_arr)

        assert metrics == {"accuracy": 0.8}
