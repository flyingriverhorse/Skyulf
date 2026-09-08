"""Explain leakage protection for standalone skyulf-core users.

Run from the repository root after installing skyulf-core:
    python skyulf-core/examples/09_leakage_safety.py

Uses small in-memory pandas data, with no backend, dataset download, or files.
Runtime checks document the expected behavior; deliberately unsafe fits are labeled.
"""

import logging
from copy import deepcopy

import numpy as np
import pandas as pd

from skyulf import SkyulfPipeline, validate_leakage_safety
from skyulf.data.dataset import SplitDataset


def _require(condition: bool, message: str) -> None:
    """Keep tutorial correctness checks active under optimized Python."""
    if not condition:
        raise RuntimeError(message)


def main() -> None:
    """Walk through the leakage policy, safe training, and external splits."""
    logging.basicConfig(level=logging.WARNING, format="%(levelname)s: %(message)s")
    data = pd.DataFrame({"x": [1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 100.0, 200.0, 300.0, 400.0]})
    data["target"] = 2 * data["x"] + 3

    split_step = {
        "name": "split",
        "transformer": "TrainTestSplitter",
        "params": {"test_size": 0.25, "random_state": 42, "target_column": "target"},
    }
    scale_step = {
        "name": "scale",
        "transformer": "StandardScaler",
        "params": {"columns": ["x"]},
    }
    leaking_config = {
        "preprocessing": [scale_step, split_step],
        "modeling": {"type": "linear_regression"},
    }

    print("\n1. raise (default): reject learned preprocessing BEFORE a split")
    # This is an ordinary variable holding a SkyulfPipeline instance.
    # Its name describes the deliberately incorrect order, not a special API.
    leaking_pipeline = SkyulfPipeline(deepcopy(leaking_config))
    try:
        leaking_pipeline.fit(data, target_column="target")
    except ValueError as error:
        print(error)
    else:
        raise AssertionError("Expected the unsafe fit to be rejected")
    _require(not leaking_pipeline.is_fitted(), "Rejection must happen before any fit")

    # get_fitted_split() fits a throwaway preprocessing chain and has the same gate.
    try:
        leaking_pipeline.get_fitted_split(data, target_column="target")
    except ValueError:
        print("get_fitted_split() also rejected this unsafe order.")
    else:
        raise AssertionError("Split extraction must not bypass the leakage gate")

    print("\n2. warn: deliberately allow the unsafe fit, with a warning")
    # A diagnostic returns messages without training or logging them itself.
    messages = validate_leakage_safety(leaking_config, on_leakage="warn")
    _require(bool(messages), "The unsafe configuration must produce a warning diagnostic")
    print("Diagnostic messages:", messages)
    # fit() logs those messages and continues. This DOES NOT make the fit safe.
    warning_pipeline = SkyulfPipeline(deepcopy(leaking_config))
    warning_pipeline.fit(data, target_column="target", on_leakage="warn")
    _require(warning_pipeline.is_fitted(), "Warning mode must permit the explicitly requested fit")

    print("\n3. ignore: deliberately allow the unsafe fit, without leakage warnings")
    _require(
        validate_leakage_safety(leaking_config, on_leakage="ignore") == [],
        "Ignore mode must suppress leakage diagnostics",
    )
    ignored_pipeline = SkyulfPipeline(deepcopy(leaking_config))
    ignored_pipeline.fit(data, target_column="target", on_leakage="ignore")
    _require(ignored_pipeline.is_fitted(), "Ignore mode must permit the explicitly requested fit")
    # ignore affects only leakage diagnostics, not other validation or errors.
    print("The pipeline fitted, but its evaluation is still contaminated.")

    print("\n4. Recommended fix: Split -> StandardScaler -> Model")
    safe_config = deepcopy(leaking_config)
    safe_config["preprocessing"] = [deepcopy(split_step), deepcopy(scale_step)]
    pipeline = SkyulfPipeline(safe_config)
    _require(
        pipeline.validate_leakage_safety() == [], "The split-first pipeline must pass validation"
    )
    pipeline.fit(data, target_column="target")
    print("Predictions:", pipeline.predict(pd.DataFrame({"x": [9.0, 10.0]})))
    X_train, y_train, X_test, y_test = pipeline.get_fitted_split(data, target_column="target")
    np.testing.assert_allclose(X_train["x"].mean(), 0, atol=1e-12)
    _require(
        len(X_train) == len(y_train) and len(X_test) == len(y_test),
        "Preprocessing must keep feature and target row counts aligned",
    )
    print("Preprocessed train/test shapes:", X_train.shape, X_test.shape)
    # These X frames are already transformed. Use them with a raw estimator,
    # not pipeline.predict(), which expects raw data and would transform again.

    print("\n5. Stateless rules and parameter-dependent exceptions")
    stateless_config = {
        "preprocessing": [
            {"name": "drop_empty_rows", "transformer": "DropMissingRows", "params": {}},
            deepcopy(split_step),
        ],
        "modeling": {},
    }
    _require(
        validate_leakage_safety(stateless_config) == [], "Stateless row rules must remain allowed"
    )
    print("DropMissingRows before splitting: no fitted statistic, so allowed.")

    constant_config = {
        "preprocessing": [
            {
                "name": "fill",
                "transformer": "SimpleImputer",
                "params": {"strategy": "constant", "fill_value": 0},
            },
            deepcopy(split_step),
        ],
        "modeling": {},
    }
    _require(
        validate_leakage_safety(constant_config) == [], "Constant imputation must remain allowed"
    )
    mean_config = deepcopy(constant_config)
    mean_config["preprocessing"][0]["params"] = {"strategy": "mean"}
    _require(
        bool(validate_leakage_safety(mean_config, on_leakage="warn")),
        "Mean imputation before a split must produce a leakage diagnostic",
    )
    print("Constant imputation is allowed; mean imputation must follow the split.")

    print("\n6. Supply an external SplitDataset instead of a splitter node")
    # The caller creates disjoint raw partitions BEFORE preprocessing.
    dataset = SplitDataset(train=data.iloc[:8].copy(), test=data.iloc[8:].copy())
    scaler_only_config = {"preprocessing": [deepcopy(scale_step)], "modeling": {}}
    external_pipeline = SkyulfPipeline(scaler_only_config)
    external_pipeline.fit(dataset, target_column="target")
    X_train, _, X_test, _ = external_pipeline.get_fitted_split(dataset, target_column="target")
    if not isinstance(dataset.train, pd.DataFrame):
        raise RuntimeError("This example requires a pandas training partition")
    if not isinstance(dataset.test, pd.DataFrame):
        raise RuntimeError("This example requires a pandas test partition")
    train_mean = dataset.train["x"].mean()
    train_std = dataset.train["x"].std(ddof=0)
    np.testing.assert_allclose(X_train["x"], (dataset.train["x"] - train_mean) / train_std)
    np.testing.assert_allclose(X_test["x"], (dataset.test["x"] - train_mean) / train_std)
    print("Train mean used by the scaler:", train_mean)
    print("Held-out rows use that SAME train mean/std, not their own statistics.")

    print("\n7. No splitter in a config is advisory, not proof of safety")
    # A config-only diagnostic cannot know about the external dataset above.
    messages = validate_leakage_safety(scaler_only_config, on_leakage="raise")
    _require(
        bool(messages) and "No train/test split" in messages[0],
        "Unsplit configurations must retain their advisory",
    )
    print(messages[0])
    # An unsplit DataFrame still gets this advisory during fit(). If you intend
    # to evaluate generalization, supply SplitDataset or configure a splitter.
    print("\nFinished. Prefer the safe order or an externally supplied SplitDataset.")


if __name__ == "__main__":
    main()
