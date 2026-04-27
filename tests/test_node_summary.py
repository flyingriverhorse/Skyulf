"""Unit tests for the per-node summary builder."""

from __future__ import annotations

import pandas as pd

from backend.ml_pipeline.execution.summary import build_summary
from skyulf.data.dataset import SplitDataset


def test_dataframe_summary_uses_shape_with_thousands_separator() -> None:
    df = pd.DataFrame({"a": range(7000), "b": range(7000), "c": range(7000)})
    out = build_summary(step_type="drop_columns", output=df, metrics={})
    assert out == "7,000 rows × 3 cols"


def test_split_dataset_summary_two_way() -> None:
    train = pd.DataFrame({"x": range(7000)})
    test = pd.DataFrame({"x": range(1500)})
    sd = SplitDataset(train=train, test=test)
    out = build_summary(step_type="train_test_splitter", output=sd, metrics={})
    assert out == "7,000 / 1,500"


def test_split_dataset_summary_with_validation() -> None:
    train = pd.DataFrame({"x": range(7000)})
    test = pd.DataFrame({"x": range(1500)})
    val = pd.DataFrame({"x": range(1500)})
    sd = SplitDataset(train=train, test=test, validation=val)
    out = build_summary(step_type="train_val_test_splitter", output=sd, metrics={})
    assert out == "7,000 / 1,500 / 1,500"


def test_split_dataset_with_x_y_tuple_payload() -> None:
    """FeatureTargetSplit produces (X, y) tuples per slot."""
    X_train = pd.DataFrame({"x": range(7000)})
    y_train = pd.Series(range(7000))
    X_test = pd.DataFrame({"x": range(1500)})
    y_test = pd.Series(range(1500))
    sd = SplitDataset(train=(X_train, y_train), test=(X_test, y_test))
    out = build_summary(step_type="feature_target_splitter", output=sd, metrics={})
    assert out == "7,000 / 1,500"


def test_training_summary_prefers_acc_plus_f1() -> None:
    metrics = {"metrics": {"accuracy": 0.873, "f1": 0.842, "precision": 0.91}}
    out = build_summary(step_type="basic_training", output=None, metrics=metrics)
    assert out == "acc 0.87 · f1 0.84"


def test_training_summary_falls_back_to_accuracy_only() -> None:
    out = build_summary(
        step_type="basic_training", output=None, metrics={"metrics": {"accuracy": 0.9}}
    )
    assert out == "acc 0.90"


def test_training_summary_regression_path() -> None:
    out = build_summary(
        step_type="basic_training",
        output=None,
        metrics={"metrics": {"r2": 0.87, "rmse": 0.123}},
    )
    assert out == "r² 0.87 · rmse 0.123"


def test_unknown_step_with_no_recognisable_output_returns_none() -> None:
    assert build_summary(step_type="something_weird", output=None, metrics={}) is None


def test_build_summary_swallows_exceptions() -> None:
    """A pathological metrics dict must never raise."""

    class _Boom:
        def __getitem__(self, key: str) -> object:
            raise RuntimeError("boom")

    # Force the training branch and feed an exploding dict.
    out = build_summary(step_type="training", output=None, metrics=_Boom())  # type: ignore[arg-type]
    assert out is None
