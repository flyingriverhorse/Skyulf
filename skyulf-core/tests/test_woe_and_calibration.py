"""Tests for the WOE/IV encoder and the calibrated classifier node."""

import numpy as np
import pandas as pd
import polars as pl

from skyulf.preprocessing.encoding import WOEEncoderApplier, WOEEncoderCalculator
from skyulf.registry import NodeRegistry


def _woe_frame() -> pd.DataFrame:
    # "city" is predictive of the binary target; "noise" is not.
    return pd.DataFrame(
        {
            "city": ["A", "A", "A", "B", "B", "B", "C", "C"],
            "target": [1, 1, 1, 0, 0, 0, 1, 0],
        }
    )


def test_woe_fit_produces_mappings_and_iv():
    df = _woe_frame()
    params = WOEEncoderCalculator().fit((df[["city"]], df["target"]), {"columns": ["city"]})
    assert params["type"] == "woe_encoder"
    assert set(params["mappings"]["city"].keys()) == {"A", "B", "C"}
    assert "city" in params["information_value"]
    # Category A (all positive) must have lower WOE than B (all negative).
    assert params["mappings"]["city"]["A"] < params["mappings"]["city"]["B"]


def test_woe_apply_replaces_categories_with_floats():
    df = _woe_frame()
    params = WOEEncoderCalculator().fit((df[["city"]], df["target"]), {"columns": ["city"]})
    out, _ = WOEEncoderApplier().apply((df[["city"]].copy(), df["target"]), params)
    assert out["city"].dtype == float
    assert np.isclose(out.loc[0, "city"], params["mappings"]["city"]["A"])


def test_woe_apply_unseen_category_uses_default():
    df = _woe_frame()
    params = WOEEncoderCalculator().fit((df[["city"]], df["target"]), {"columns": ["city"]})
    new = pd.DataFrame({"city": ["Z"]})
    out = WOEEncoderApplier().apply(new, params)
    assert out.loc[0, "city"] == params["default"]


def test_woe_engine_parity_pandas_vs_polars():
    df = _woe_frame()
    pd_params = WOEEncoderCalculator().fit((df[["city"]], df["target"]), {"columns": ["city"]})
    pl_X = pl.from_pandas(df[["city"]])
    pl_y = pl.from_pandas(df[["target"]])["target"]
    pl_params = WOEEncoderCalculator().fit((pl_X, pl_y), {"columns": ["city"]})
    for cat, woe in pd_params["mappings"]["city"].items():
        assert np.isclose(woe, pl_params["mappings"]["city"][cat])


def test_woe_non_binary_target_is_skipped():
    df = pd.DataFrame({"city": ["A", "B", "C"], "target": [0, 1, 2]})
    params = WOEEncoderCalculator().fit((df[["city"]], df["target"]), {"columns": ["city"]})
    assert params == {}


def test_woe_registered():
    assert "WOEEncoder" in NodeRegistry.get_all_metadata()


def test_calibrated_classifier_registered_and_predicts():
    assert "calibrated_classifier" in NodeRegistry.get_all_metadata()

    rng = np.random.default_rng(0)
    X = pd.DataFrame(
        rng.normal(size=(40, 3)),
        columns=["a", "b", "c"],  # ty: ignore[invalid-argument-type]
    )
    y = pd.Series((X["a"] + X["b"] > 0).astype(int))

    calc = NodeRegistry.get_calculator("calibrated_classifier")()
    applier = NodeRegistry.get_applier("calibrated_classifier")()
    model = calc.fit(X, y, {"cv": 3})
    preds = applier.predict(X, model)
    assert len(preds) == len(X)

    proba = applier.predict_proba(X, model)
    assert proba is not None
    # Calibrated probabilities must lie in [0, 1].
    assert ((proba.to_numpy() >= 0) & (proba.to_numpy() <= 1)).all()


def test_calibrated_classifier_selectable_base_estimator():
    from sklearn.ensemble import RandomForestClassifier

    rng = np.random.default_rng(1)
    X = pd.DataFrame(
        rng.normal(size=(40, 3)),
        columns=["a", "b", "c"],  # ty: ignore[invalid-argument-type]
    )
    y = pd.Series((X["a"] + X["b"] > 0).astype(int))

    calc = NodeRegistry.get_calculator("calibrated_classifier")()
    model = calc.fit(X, y, {"base_estimator": "random_forest", "method": "isotonic", "cv": 3})

    # The string key must be resolved into the matching estimator instance.
    assert isinstance(model.estimator, RandomForestClassifier)
    assert model.method == "isotonic"


def test_calibrated_classifier_unknown_base_estimator_falls_back():
    from sklearn.linear_model import LogisticRegression

    rng = np.random.default_rng(2)
    X = pd.DataFrame(
        rng.normal(size=(30, 2)),
        columns=["a", "b"],  # ty: ignore[invalid-argument-type]
    )
    y = pd.Series((X["a"] > 0).astype(int))

    calc = NodeRegistry.get_calculator("calibrated_classifier")()
    model = calc.fit(X, y, {"base_estimator": "does_not_exist", "cv": 3})

    # Unknown keys fall back to logistic regression rather than raising.
    assert isinstance(model.estimator, LogisticRegression)
