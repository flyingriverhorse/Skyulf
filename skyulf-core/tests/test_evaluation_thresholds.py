"""Tests for skyulf.modeling._evaluation.thresholds (optimize_thresholds/apply_thresholds)."""

import numpy as np
import pytest
from sklearn.metrics import f1_score

from skyulf.modeling._evaluation.thresholds import apply_thresholds, optimize_thresholds


def test_apply_thresholds_binary_basic():
    """Binary: predicts positive class when proba[:, 1] >= threshold."""
    y_proba = np.array(
        [
            [0.9, 0.1],
            [0.4, 0.6],
            [0.55, 0.45],
            [0.1, 0.9],
        ]
    )
    preds = apply_thresholds(y_proba, thresholds=0.5, classes=[0, 1])
    np.testing.assert_array_equal(preds, [0, 1, 0, 1])


def test_apply_thresholds_binary_dict_form():
    """Binary thresholds may also be passed as a single-entry dict."""
    y_proba = np.array([[0.9, 0.1], [0.2, 0.8]])
    preds = apply_thresholds(y_proba, thresholds={1: 0.5}, classes=[0, 1])
    np.testing.assert_array_equal(preds, [0, 1])


def test_apply_thresholds_multiclass_equal_thresholds_matches_argmax():
    """Equal thresholds across all classes must reduce to plain argmax."""
    y_proba = np.array(
        [
            [0.2, 0.5, 0.3],
            [0.6, 0.1, 0.3],
            [0.1, 0.1, 0.8],
        ]
    )
    classes = ["a", "b", "c"]
    preds = apply_thresholds(y_proba, thresholds={"a": 0.3, "b": 0.3, "c": 0.3}, classes=classes)
    expected = np.array(classes)[np.argmax(y_proba, axis=1)]
    np.testing.assert_array_equal(preds, expected)


def test_apply_thresholds_multiclass_scaled_argmax():
    """Dividing by a class's own threshold shifts which class 'wins' the row."""
    y_proba = np.array([[0.4, 0.4, 0.2]])
    classes = [0, 1, 2]
    # Class 1 has a much smaller threshold, so 0.4/0.1 (=4.0) beats 0.4/1.0.
    preds = apply_thresholds(y_proba, thresholds={0: 1.0, 1: 0.1, 2: 1.0}, classes=classes)
    np.testing.assert_array_equal(preds, [1])


def test_apply_thresholds_raises_on_incomplete_coverage():
    """Missing a class's threshold in the dict is a caller error, not silently ignored."""
    y_proba = np.array([[0.2, 0.5, 0.3]])
    with pytest.raises(ValueError, match="threshold"):
        apply_thresholds(y_proba, thresholds={0: 0.5, 1: 0.5}, classes=[0, 1, 2])


def test_apply_thresholds_raises_on_non_2d_proba():
    """y_proba must be 2D (n_samples, n_classes)."""
    with pytest.raises(ValueError, match="2D"):
        apply_thresholds(np.array([0.1, 0.9]), thresholds=0.5, classes=[0, 1])


def test_optimize_thresholds_binary_grid_recovers_known_optimum():
    """A synthetic binary case where the F1-optimal threshold is analytically known."""
    rng = np.random.default_rng(0)
    y_true = np.array([0] * 300 + [1] * 100)
    # Positive class scores cluster around 0.7, negative class around 0.3,
    # so F1 is maximized near threshold ~0.5 regardless of exact noise.
    proba_pos = np.concatenate([rng.normal(0.3, 0.05, 300), rng.normal(0.7, 0.05, 100)]).clip(
        0.01, 0.99
    )
    y_proba = np.column_stack([1 - proba_pos, proba_pos])

    thresholds = optimize_thresholds(
        y_true, y_proba, metric=f1_score, classes=[0, 1], strategy="grid", grid_points=101
    )
    assert set(thresholds.keys()) == {0, 1}
    tuned_pred = apply_thresholds(y_proba, thresholds, classes=[0, 1])
    default_pred = apply_thresholds(y_proba, thresholds=0.5, classes=[0, 1])
    assert f1_score(y_true, tuned_pred) >= f1_score(y_true, default_pred)


def test_optimize_thresholds_defaults_to_grid_for_binary():
    """strategy=None must auto-select grid search for exactly 2 classes."""
    y_true = np.array([0, 0, 1, 1])
    y_proba = np.array([[0.8, 0.2], [0.6, 0.4], [0.4, 0.6], [0.2, 0.8]])
    thresholds = optimize_thresholds(y_true, y_proba, metric=f1_score, classes=[0, 1])
    assert set(thresholds.keys()) == {0, 1}


def test_optimize_thresholds_multiclass_nelder_mead_improves_on_argmax():
    """Nelder-Mead-tuned multiclass thresholds must not do worse than plain argmax
    on balanced accuracy for an imbalanced synthetic dataset."""
    rng = np.random.default_rng(1)
    classes = np.array(["a", "b", "c"])
    n_per_class = [300, 50, 50]
    y_true_parts = []
    proba_parts = []
    for i, n in enumerate(n_per_class):
        y_true_parts.append(np.full(n, classes[i]))
        base = rng.dirichlet(alpha=[1, 1, 1], size=n)
        # Bias each row's own-class column upward so there's real signal.
        base[:, i] += 1.5
        base = base / base.sum(axis=1, keepdims=True)
        proba_parts.append(base)
    y_true = np.concatenate(y_true_parts)
    y_proba = np.concatenate(proba_parts)

    def balanced_acc(y_t, y_p):
        from sklearn.metrics import balanced_accuracy_score

        return balanced_accuracy_score(y_t, y_p)


def test_optimize_thresholds_multiclass_nelder_mead_actually_escapes_argmax_start():
    """Regression test for a real bug: scipy's default Nelder-Mead initial
    simplex for a zero-valued starting point (log(1.0) == 0, i.e. plain
    argmax) perturbs by a minuscule ~0.00025 log-units. `apply_thresholds`'s
    scaled-argmax rule is piecewise-constant, so that tiny step almost never
    crosses a real decision boundary -- the optimizer used to see a flat
    plateau in every direction and "converge" immediately at the untouched
    starting point, silently never tuning anything, *regardless of which
    metric was requested*. This dataset has a clear, fixable class-1 bias
    (every row's probabilities are pushed toward class 1), so a real optimum
    strictly better than plain argmax exists and must actually be found.
    """
    rng = np.random.default_rng(0)
    n = 300
    y_true = rng.integers(0, 3, n)
    y_proba = np.empty((n, 3))
    for i in range(n):
        base = rng.dirichlet([1, 1, 1])
        base[1] += 0.3  # systematically over-predict class 1
        y_proba[i] = base / base.sum()
    classes = np.array([0, 1, 2])

    def f1_weighted(y_t, y_p):
        return f1_score(y_t, y_p, average="weighted")

    argmax_pred = classes[np.argmax(y_proba, axis=1)]
    argmax_score = f1_weighted(y_true, argmax_pred)

    thresholds = optimize_thresholds(
        y_true, y_proba, metric=f1_weighted, classes=classes, strategy="nelder-mead"
    )
    tuned_pred = apply_thresholds(y_proba, thresholds, classes=classes)
    tuned_score = f1_weighted(y_true, tuned_pred)

    # Must not just tie the untouched argmax baseline (that's the bug) -- it
    # must find a real, better solution to this clearly fixable bias.
    assert tuned_score > argmax_score + 0.01
    assert not np.allclose(list(thresholds.values()), 1.0)


def test_optimize_thresholds_multiclass_nelder_mead_never_worse_than_argmax():
    """Whatever the search finds, it must never regress below plain argmax."""
    rng = np.random.default_rng(1)
    classes = np.array(["a", "b", "c"])
    n_per_class = [300, 50, 50]
    y_true_parts = []
    proba_parts = []
    for i, n in enumerate(n_per_class):
        y_true_parts.append(np.full(n, classes[i]))
        base = rng.dirichlet(alpha=[1, 1, 1], size=n)
        # Bias each row's own-class column upward so there's real signal.
        base[:, i] += 1.5
        base = base / base.sum(axis=1, keepdims=True)
        proba_parts.append(base)
    y_true = np.concatenate(y_true_parts)
    y_proba = np.concatenate(proba_parts)

    def balanced_acc(y_t, y_p):
        from sklearn.metrics import balanced_accuracy_score

        return balanced_accuracy_score(y_t, y_p)

    thresholds = optimize_thresholds(
        y_true, y_proba, metric=balanced_acc, classes=classes, strategy="nelder-mead"
    )
    assert set(thresholds.keys()) == set(classes)
    tuned_pred = apply_thresholds(y_proba, thresholds, classes=classes)
    argmax_pred = classes[np.argmax(y_proba, axis=1)]
    assert balanced_acc(y_true, tuned_pred) >= balanced_acc(y_true, argmax_pred) - 1e-9


def test_optimize_thresholds_defaults_to_nelder_mead_for_multiclass():
    """strategy=None must auto-select nelder-mead for 3+ classes."""
    y_true = np.array(["a", "b", "c", "a", "b", "c"])
    y_proba = np.array(
        [
            [0.7, 0.2, 0.1],
            [0.1, 0.7, 0.2],
            [0.2, 0.1, 0.7],
            [0.6, 0.3, 0.1],
            [0.2, 0.6, 0.2],
            [0.1, 0.2, 0.7],
        ]
    )
    thresholds = optimize_thresholds(
        y_true,
        y_proba,
        metric=lambda a, b: f1_score(a, b, average="macro"),
        classes=["a", "b", "c"],
    )
    assert set(thresholds.keys()) == {"a", "b", "c"}


def test_optimize_thresholds_raises_on_unknown_strategy():
    y_true = np.array([0, 1])
    y_proba = np.array([[0.6, 0.4], [0.4, 0.6]])
    with pytest.raises(ValueError, match="strategy"):
        optimize_thresholds(y_true, y_proba, metric=f1_score, strategy="bogus")


def test_optimize_thresholds_and_apply_thresholds_exported_from_evaluation_package():
    from skyulf.modeling._evaluation import apply_thresholds as ev_apply
    from skyulf.modeling._evaluation import optimize_thresholds as ev_optimize

    assert ev_optimize is optimize_thresholds
    assert ev_apply is apply_thresholds


def test_optimize_thresholds_and_apply_thresholds_exported_from_modeling_top_level():
    from skyulf.modeling import apply_thresholds as top_apply
    from skyulf.modeling import optimize_thresholds as top_optimize

    assert top_optimize is optimize_thresholds
    assert top_apply is apply_thresholds
