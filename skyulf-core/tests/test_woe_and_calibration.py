"""Tests for the WOE/IV encoder and the calibrated classifier node."""

import importlib

import numpy as np
import pandas as pd
import polars as pl
import pytest
from tests.utils.dataset_loader import load_sample_dataset
from tests.utils.test_case_loader import TestCaseLoader

from skyulf.preprocessing.encoding import WOEEncoderApplier, WOEEncoderCalculator
from skyulf.registry import NodeRegistry

_base_estimator_cases = TestCaseLoader("preprocessing/woe_and_calibration").load()


def _import_from_path(dotted_path: str) -> type:
    """Import a class from a fully-qualified dotted path, e.g. ``sklearn.ensemble.RandomForestClassifier``."""
    module_path, class_name = dotted_path.rsplit(".", 1)
    return getattr(importlib.import_module(module_path), class_name)


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


def test_woe_excludes_target_column_when_explicitly_selected():
    """Regression test: _exclude_target_column previously no-op'd for WOEEncoder.

    If a user misconfigures ``columns`` to include the target column itself,
    WOE would fit/apply a mapping of the target against itself (a degenerate,
    leaky encoding) and silently overwrite the target's own values. The guard
    must strip the target column from ``columns`` just like it does for
    OneHot/Dummy/Hash/TargetEncoder.
    """
    df = _woe_frame()
    params = WOEEncoderCalculator().fit(
        (df, None), {"columns": ["city", "target"], "target_column": "target"}
    )
    assert "target" not in params["mappings"]
    assert "city" in params["mappings"]


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


class TestCalibratedClassifierBaseEstimator:
    """Scenarios loaded from
    ``tests/test_cases/preprocessing/woe_and_calibration.json``.
    """

    @pytest.mark.parametrize(*_base_estimator_cases)
    def test_base_estimator_resolution(
        self,
        base_estimator: str,
        method: str | None,
        expected_estimator_path: str,
        expected_method: str | None,
    ) -> None:
        """A known ``base_estimator`` key resolves to its matching estimator class;
        an unknown key falls back to logistic regression rather than raising.
        """
        rng = np.random.default_rng(1)
        X = pd.DataFrame(
            rng.normal(size=(40, 3)),
            columns=["a", "b", "c"],  # ty: ignore[invalid-argument-type]
        )
        y = pd.Series((X["a"] + X["b"] > 0).astype(int))

        calc = NodeRegistry.get_calculator("calibrated_classifier")()
        config = {"base_estimator": base_estimator, "cv": 3}
        if method is not None:
            config["method"] = method
        model = calc.fit(X, y, config)

        # The string key must be resolved into the matching estimator instance
        # (or fall back to logistic regression for an unknown key).
        assert isinstance(model.estimator, _import_from_path(expected_estimator_path))
        if expected_method is not None:
            assert model.method == expected_method


# ---------------------------------------------------------------------------
# Real-shaped dataset integration check
# ---------------------------------------------------------------------------


class TestRealShapedDataset:
    """Integration-style check against the checked-in ``customers.csv`` sample.

    Verifies that WOEEncoderCalculator handles a real categorical column
    (``plan_type``) with the binary ``churned`` target, producing a valid
    WOE mapping for every category and applying float-valued WOE scores.
    """

    def test_woe_on_plan_type_with_churned_target(self) -> None:
        df = load_sample_dataset("customers")
        X = df[["plan_type"]]
        y = df["churned"]
        params = WOEEncoderCalculator().fit((X, y), {"columns": ["plan_type"]})
        # A binary target must yield a valid WOE artifact.
        assert params.get("type") == "woe_encoder"
        assert set(params["mappings"]["plan_type"].keys()) == {"basic", "premium", "enterprise"}
        # Applying WOE scores must yield float-valued column.
        out = WOEEncoderApplier().apply(X.copy(), params)
        assert out["plan_type"].dtype == float
