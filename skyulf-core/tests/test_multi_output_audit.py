"""Backlog 4.2 audit: proves what does/doesn't support multi-output `y` today.

`y` here means a DataFrame with *more than one* target column (as opposed to
the usual single `pd.Series`/1-D array). This file is a deliberate audit, not
a feature test suite: every test documents current, unmodified behavior of
`skyulf-core` — some tests show things that already work "for free" (because
no code path forces a 1-D reshape), others use `pytest.raises`/`xfail` to
pin down exactly where and why multi-output `y` breaks.

See `skyulf-core/docs_internal/multi_output_audit.md` for the full narrative
write-up (summary table, effort estimates, recommended next steps) that this
file provides executable evidence for.
"""

import numpy as np
import pandas as pd
import pytest

from skyulf.modeling.classification import (
    LogisticRegressionCalculator,
    RandomForestClassifierApplier,
    RandomForestClassifierCalculator,
)
from skyulf.modeling.regression import (
    LinearRegressionApplier,
    LinearRegressionCalculator,
    RandomForestRegressorApplier,
    RandomForestRegressorCalculator,
)

# ---------------------------------------------------------------------------
# Shared fixtures
# ---------------------------------------------------------------------------


@pytest.fixture
def multi_output_regression_data():
    """Features + a genuinely 2-column numeric target DataFrame."""
    rng = np.random.RandomState(0)
    X = pd.DataFrame({"f1": rng.normal(0, 1, 40), "f2": rng.normal(0, 1, 40)})
    y = pd.DataFrame(
        {
            "target_a": X["f1"] * 2 + rng.normal(0, 0.1, 40),
            "target_b": X["f2"] * -1 + rng.normal(0, 0.1, 40),
        }
    )
    return X, y


@pytest.fixture
def multi_output_classification_data():
    """Features + a genuinely 2-column binary target DataFrame (multilabel-style)."""
    rng = np.random.RandomState(0)
    X = pd.DataFrame({"f1": rng.normal(0, 1, 40), "f2": rng.normal(0, 1, 40)})
    y = pd.DataFrame(
        {
            "target_a": (X["f1"] > 0).astype(int),
            "target_b": (X["f2"] > 0).astype(int),
        }
    )
    return X, y


# ---------------------------------------------------------------------------
# SklearnCalculator.fit — training
# ---------------------------------------------------------------------------


class TestSklearnCalculatorFit:
    """`SklearnBridge.to_sklearn` only ravels a (N, 1) y (sklearn_bridge.py:32-33),
    so a genuinely 2-D `y` reaches `estimator.fit(X, y)` untouched
    (sklearn_wrapper.py:110-112). Whether that succeeds depends entirely on
    whether the wrapped sklearn estimator natively supports multi-output y.
    """

    def test_linear_regression_fit_succeeds_with_multi_output_y(self, multi_output_regression_data):
        """LinearRegression natively supports 2-D y: skyulf adds no coercion that
        would block it, so .fit() succeeds and produces a (n_targets, n_features)
        coefficient matrix."""
        X, y = multi_output_regression_data
        calc = LinearRegressionCalculator()

        model = calc.fit(X, y, {})

        assert model.coef_.shape == (2, 2)  # (n_targets, n_features)

    def test_random_forest_regressor_fit_succeeds_with_multi_output_y(
        self, multi_output_regression_data
    ):
        """RandomForestRegressor also natively supports multi-output regression."""
        X, y = multi_output_regression_data
        calc = RandomForestRegressorCalculator()

        model = calc.fit(X, y, {"n_estimators": 5, "random_state": 0})

        assert model.n_outputs_ == 2

    def test_random_forest_classifier_fit_succeeds_with_multi_output_y(
        self, multi_output_classification_data
    ):
        """RandomForestClassifier natively supports multi-output ("multilabel-indicator"
        style) classification, so training already works without any skyulf changes."""
        X, y = multi_output_classification_data
        calc = RandomForestClassifierCalculator()

        model = calc.fit(X, y, {"n_estimators": 5, "random_state": 0})

        assert model.n_outputs_ == 2

    def test_logistic_regression_fit_raises_with_multi_output_y(
        self, multi_output_classification_data
    ):
        """LogisticRegression does NOT natively support multi-output y, and skyulf
        does not wrap it in sklearn.multioutput.MultiOutputClassifier — so fit()
        fails with sklearn's own 1-D-target error. This documents the gap: fixing
        it would require skyulf to auto-detect 2-D y and wrap non-multi-output
        estimators (see docs_internal/multi_output_audit.md, item 4)."""
        X, y = multi_output_classification_data
        calc = LogisticRegressionCalculator()

        with pytest.raises(ValueError, match="1d array"):
            calc.fit(X, y, {})


# ---------------------------------------------------------------------------
# SklearnApplier.predict — inference (the actual, confirmed gap)
# ---------------------------------------------------------------------------


class TestSklearnApplierPredict:
    """`SklearnApplier.predict` unconditionally does
    `return pd.Series(preds, index=index)` (sklearn_wrapper.py:136). `pd.Series`
    requires 1-D data, so any estimator that produces a (n_samples, n_targets)
    prediction array — i.e. every estimator proven to *train* successfully with
    multi-output y above — fails at prediction time. This is the cleanest,
    highest-impact gap identified in the audit (item 1)."""

    def test_linear_regression_predict_raises_on_multi_output_predictions(
        self, multi_output_regression_data
    ):
        X, y = multi_output_regression_data
        model = LinearRegressionCalculator().fit(X, y, {})
        applier = LinearRegressionApplier()

        # Sanity check: the underlying sklearn model itself predicts 2-D output fine.
        raw_preds = model.predict(X.to_numpy())
        assert raw_preds.shape == (len(X), 2)

        # But skyulf's Applier wrapper cannot represent that as a pd.Series.
        with pytest.raises(ValueError, match="1-dimensional"):
            applier.predict(X, model)

    def test_random_forest_regressor_predict_raises_on_multi_output_predictions(
        self, multi_output_regression_data
    ):
        X, y = multi_output_regression_data
        model = RandomForestRegressorCalculator().fit(X, y, {"n_estimators": 5, "random_state": 0})
        applier = RandomForestRegressorApplier()

        with pytest.raises(ValueError, match="1-dimensional"):
            applier.predict(X, model)

    def test_random_forest_classifier_predict_raises_on_multi_output_predictions(
        self, multi_output_classification_data
    ):
        X, y = multi_output_classification_data
        model = RandomForestClassifierCalculator().fit(X, y, {"n_estimators": 5, "random_state": 0})
        applier = RandomForestClassifierApplier()

        with pytest.raises(ValueError, match="1-dimensional"):
            applier.predict(X, model)

    def test_single_output_predict_still_works(self):
        """Control case: a normal 1-D y target still predicts fine via pd.Series —
        proving the failure above is specific to multi-output shape, not a
        general regression in the Applier."""
        rng = np.random.RandomState(0)
        X = pd.DataFrame({"f1": rng.normal(0, 1, 40), "f2": rng.normal(0, 1, 40)})
        y = pd.Series(X["f1"] * 2 + rng.normal(0, 0.1, 40))

        model = LinearRegressionCalculator().fit(X, y, {})
        preds = LinearRegressionApplier().predict(X, model)

        assert isinstance(preds, pd.Series)
        assert len(preds) == len(X)


# ---------------------------------------------------------------------------
# Resampling (preprocessing) — passthrough behavior
# ---------------------------------------------------------------------------

pytest.importorskip(
    "imblearn", reason="imbalanced-learn not installed — pip install imbalanced-learn"
)

from skyulf.preprocessing.resampling import (  # noqa: E402
    OversamplingApplier,
    OversamplingCalculator,
)


class TestResamplingMultiOutputY:
    """Resampling Appliers route `y` straight into imblearn
    (`sampler.fit_resample(X_pd, y_pd)`, resampling.py:97) with no shape
    coercion of their own (resampling.py:79-80 only forces a `pd.Series` if
    the sampler *didn't* already return one). Whether multi-output `y` works
    is therefore entirely a property of the chosen imblearn sampler."""

    @pytest.fixture
    def multi_output_binary_data(self):
        rng = np.random.RandomState(0)
        n = 40
        X = pd.DataFrame({"f1": rng.normal(0, 1, n), "f2": rng.normal(0, 1, n)})
        y = pd.DataFrame(
            {
                "target_a": rng.randint(0, 2, n),
                "target_b": rng.randint(0, 2, n),
            }
        )
        return X, y

    def test_random_oversampler_passes_through_multi_output_y_unchanged(
        self, multi_output_binary_data
    ):
        """RandomOverSampler only duplicates existing rows (no synthetic
        generation), so it tolerates a 2-column y and returns a 2-column
        DataFrame back — this already "just works" today, incidentally."""
        X, y = multi_output_binary_data
        config = {"method": "random_over", "target_column": "target_a"}

        artifact = OversamplingCalculator().fit((X, y), config)
        X_res, y_res = OversamplingApplier().apply((X, y), artifact)

        assert isinstance(y_res, pd.DataFrame)
        assert list(y_res.columns) == ["target_a", "target_b"]
        assert len(X_res) == len(y_res)

    def test_smote_raises_on_multi_output_y(self, multi_output_binary_data):
        """SMOTE (and its variants) explicitly reject multi-output targets —
        imblearn itself raises this error; skyulf does not catch or translate
        it into a clearer message (see audit doc, recommended next steps)."""
        X, y = multi_output_binary_data
        config = {"method": "smote", "target_column": "target_a", "k_neighbors": 2}

        artifact = OversamplingCalculator().fit((X, y), config)

        with pytest.raises(
            ValueError, match="Multilabel and multioutput targets are not supported"
        ):
            OversamplingApplier().apply((X, y), artifact)
