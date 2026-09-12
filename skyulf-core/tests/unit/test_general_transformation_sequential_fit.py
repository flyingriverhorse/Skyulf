"""Regressions for fitting general transformation rules in application order."""

import json
from copy import deepcopy
from typing import Any

import numpy as np
import pandas as pd
import polars as pl
import pytest
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import FunctionTransformer, PowerTransformer

from skyulf.preprocessing.pipeline import FeatureEngineer
from skyulf.preprocessing.transformations.general import (
    GeneralTransformationApplier,
    GeneralTransformationCalculator,
)


@pytest.fixture(params=["pandas", "polars"])
def engine(request, monkeypatch) -> str:
    """Keep native frames and pipeline engine selection consistent."""
    monkeypatch.setenv("SKYULF_ENGINE", request.param)
    return request.param


def _frame(values: dict[str, list[float]], engine: str) -> Any:
    """Create equivalent records with nontrivial pandas row labels."""
    frame = pd.DataFrame(values, index=[7] * len(next(iter(values.values()))))
    return pl.from_pandas(frame) if engine == "polars" else frame


@pytest.mark.parametrize("method", ["box-cox", "yeo-johnson"])
@pytest.mark.parametrize("standardize", [False, True])
def test_log_then_power_fits_transformed_training_values(engine, method, standardize):
    """A later fitted rule must learn from the values it receives during replay."""
    train_values = np.array([1.0, 2.0, 4.0, 8.0, 16.0, 32.0])
    heldout_values = np.array([3.0, 64.0])
    train = _frame({"x": train_values.tolist()}, engine)
    heldout = _frame({"x": heldout_values.tolist()}, engine)
    config = {
        "transformations": [
            {"column": "x", "method": "log"},
            {"column": "x", "method": method, "standardize": standardize},
        ]
    }
    original_config = deepcopy(config)
    params = GeneralTransformationCalculator().fit(train, config)
    serialized = json.dumps(params)
    restored = json.loads(serialized)
    applier = GeneralTransformationApplier()
    train_out = applier.apply(train, restored)
    heldout_out = applier.apply(heldout, restored)
    single_out = applier.apply(_frame({"x": [3.0]}, engine), restored)
    reference = PowerTransformer(method=method, standardize=standardize).fit(
        np.log1p(train_values).reshape(-1, 1)
    )

    np.testing.assert_allclose(
        train_out["x"].to_numpy(),
        reference.transform(np.log1p(train_values).reshape(-1, 1)).ravel(),
        atol=1e-10,
    )
    np.testing.assert_allclose(
        heldout_out["x"].to_numpy(),
        reference.transform(np.log1p(heldout_values).reshape(-1, 1)).ravel(),
        atol=1e-10,
    )
    np.testing.assert_allclose(single_out["x"].to_numpy(), heldout_out["x"].to_numpy()[:1])
    np.testing.assert_array_equal(train["x"].to_numpy(), train_values)
    if standardize:
        assert abs(train_out["x"].to_numpy().mean()) < 1e-10
    if engine == "pandas":
        assert train_out.index.tolist() == train.index.tolist()
        assert heldout_out.index.tolist() == heldout.index.tolist()
    assert json.dumps(restored) == serialized
    assert config == original_config


def test_interleaved_repeated_power_rules_match_separate_pipeline_nodes(engine):
    """Combining ordered rules must preserve training and inference pipeline outputs."""
    train = _frame(
        {"x": [1.0, 2.0, 4.0, 8.0, 16.0, 32.0], "z": [2.0, 6.0, 3.0, 9.0, 25.0, 81.0]},
        engine,
    )
    heldout = _frame({"x": [3.0, 64.0], "z": [8.0, 36.0]}, engine)
    rules = [
        {"column": "x", "method": "log"},
        {"column": "z", "method": "sqrt"},
        {"column": "x", "method": "yeo-johnson", "standardize": False},
        {"column": "z", "method": "box-cox", "standardize": False},
        {"column": "x", "method": "yeo-johnson"},
        {"column": "z", "method": "yeo-johnson"},
    ]
    combined = FeatureEngineer(
        [
            {
                "name": "combined",
                "transformer": "GeneralTransformation",
                "params": {"transformations": rules},
            }
        ]
    )
    separate = FeatureEngineer(
        [
            {
                "name": f"rule {index}",
                "transformer": "GeneralTransformation",
                "params": {"transformations": [rule]},
            }
            for index, rule in enumerate(rules)
        ]
    )
    combined_train, _ = combined.fit_transform(train)
    separate_train, _ = separate.fit_transform(train)
    artifact = combined.fitted_steps[0]["artifact"]
    combined.fitted_steps[0]["artifact"] = json.loads(json.dumps(artifact))
    combined_heldout = combined.transform(heldout)
    separate_heldout = separate.transform(heldout)
    references = {
        "x": make_pipeline(
            FunctionTransformer(np.log1p),
            PowerTransformer(method="yeo-johnson", standardize=False),
            PowerTransformer(method="yeo-johnson"),
        ),
        "z": make_pipeline(
            FunctionTransformer(np.sqrt),
            PowerTransformer(method="box-cox", standardize=False),
            PowerTransformer(method="yeo-johnson"),
        ),
    }

    for col, reference in references.items():
        train_values = train[col].to_numpy().reshape(-1, 1)
        heldout_values = heldout[col].to_numpy().reshape(-1, 1)
        np.testing.assert_allclose(
            combined_train[col].to_numpy(), reference.fit_transform(train_values).ravel(), atol=1e-8
        )
        np.testing.assert_allclose(
            combined_heldout[col].to_numpy(), reference.transform(heldout_values).ravel(), atol=1e-8
        )
        np.testing.assert_allclose(
            combined_train[col].to_numpy(), separate_train[col].to_numpy(), atol=1e-8
        )
        np.testing.assert_allclose(
            combined_heldout[col].to_numpy(), separate_heldout[col].to_numpy(), atol=1e-8
        )
    assert len(artifact["transformations"]) == len(rules)


@pytest.mark.parametrize("skip_first", [False, True])
def test_box_cox_uses_positive_values_created_by_an_earlier_rule(engine, skip_first):
    """Box-Cox eligibility must use transformed values even after an earlier skipped fit."""
    values = np.array([-1.0, -2.0, -4.0, -8.0, -16.0])
    train = _frame({"x": values.tolist()}, engine)
    rules = [{"column": "x", "method": "box-cox"}] if skip_first else []
    rules += [{"column": "x", "method": "square"}, {"column": "x", "method": "box-cox"}]
    params = GeneralTransformationCalculator().fit(train, {"transformations": rules})
    out = GeneralTransformationApplier().apply(train, params)
    expected = PowerTransformer(method="box-cox").fit_transform(np.square(values).reshape(-1, 1))

    np.testing.assert_allclose(out["x"].to_numpy(), expected.ravel(), atol=1e-10)
    assert [item["method"] for item in params["transformations"]] == ["square", "box-cox"]


def test_box_cox_skips_nonpositive_values_created_by_an_earlier_power_rule(engine):
    """A rule invalidated by prior standardization must be omitted from the saved artifact."""
    values = np.array([1.0, 2.0, 4.0, 8.0, 16.0])
    train = _frame({"x": values.tolist()}, engine)
    params = GeneralTransformationCalculator().fit(
        train,
        {
            "transformations": [
                {"column": "x", "method": "yeo-johnson"},
                {"column": "x", "method": "box-cox"},
                {"column": "x", "method": "square"},
            ]
        },
    )
    out = GeneralTransformationApplier().apply(train, params)
    expected = np.square(PowerTransformer().fit_transform(values.reshape(-1, 1))).ravel()

    np.testing.assert_allclose(out["x"].to_numpy(), expected, atol=1e-10)
    assert [item["method"] for item in params["transformations"]] == ["yeo-johnson", "square"]


def test_legacy_ordered_power_artifact_keeps_its_saved_statistics(engine):
    """Old artifacts must replay their saved rules without inferring or refitting statistics."""
    params = {
        "type": "general_transformation",
        "transformations": [
            {
                "column": "x",
                "method": "yeo-johnson",
                "lambdas": [1.0],
                "scaler_params": {"mean": [2.0], "scale": [2.0]},
            },
            {"column": "x", "method": "square"},
            {
                "column": "x",
                "method": "yeo-johnson",
                "lambdas": [1.0],
                "scaler_params": {"mean": [1.0], "scale": [2.0]},
            },
        ],
    }
    out = GeneralTransformationApplier().apply(_frame({"x": [0.0, 2.0, 4.0]}, engine), params)

    # Equivalent Yeo-Johnson implementations can leave roundoff near exact zero.
    np.testing.assert_allclose(out["x"].to_numpy(), [0.0, -0.5, 0.0], rtol=0, atol=1e-12)
