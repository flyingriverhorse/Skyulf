"""Unit tests for the per-node summary builder.

Each family lives in its own focused test so a phrasing change is
quickly traceable.
"""

from __future__ import annotations

import pandas as pd

from backend.ml_pipeline.execution.summary import build_summary
from skyulf.data.dataset import SplitDataset

# ---------------------------------------------------------------------------
# Snapshot / loaders / unknown family — pure shape line
# ---------------------------------------------------------------------------


def test_data_loader_renders_shape_line() -> None:
    df = pd.DataFrame({"a": range(7000), "b": range(7000), "c": range(7000)})
    out = build_summary(step_type="data_loader", output=df, metrics={})
    assert out == "7,000 rows × 3 cols"


def test_unknown_step_with_no_recognisable_output_returns_none() -> None:
    assert build_summary(step_type="something_weird", output=None, metrics={}) is None


# ---------------------------------------------------------------------------
# Splitters
# ---------------------------------------------------------------------------


def test_split_dataset_summary_two_way() -> None:
    train = pd.DataFrame({"x": range(7000), "y": range(7000)})
    test = pd.DataFrame({"x": range(1500), "y": range(1500)})
    sd = SplitDataset(train=train, test=test)
    out = build_summary(step_type="TrainTestSplitter", output=sd, metrics={})
    assert out == "7,000 / 1,500 rows × 2 cols"


def test_split_dataset_summary_with_validation() -> None:
    train = pd.DataFrame({"x": range(7000)})
    test = pd.DataFrame({"x": range(1500)})
    val = pd.DataFrame({"x": range(1500)})
    sd = SplitDataset(train=train, test=test, validation=val)
    out = build_summary(step_type="Split", output=sd, metrics={})
    assert out == "7,000 / 1,500 / 1,500 rows × 1 cols"


def test_split_dataset_with_x_y_tuple_payload() -> None:
    """FeatureTargetSplit produces (X, y) tuples per slot."""
    X_train = pd.DataFrame({"x": range(7000)})
    y_train = pd.Series(range(7000))
    X_test = pd.DataFrame({"x": range(1500)})
    y_test = pd.Series(range(1500))
    sd = SplitDataset(train=(X_train, y_train), test=(X_test, y_test))
    out = build_summary(step_type="feature_target_split", output=sd, metrics={})
    # 1 X col + y → 2 cols reported.
    assert out == "7,000 / 1,500 rows × 2 cols"


# ---------------------------------------------------------------------------
# Drop / outlier / dedup / sampling — row delta dominates
# ---------------------------------------------------------------------------


def test_drop_missing_rows_renders_negative_row_delta_with_pct() -> None:
    in_df = pd.DataFrame({"a": range(7000), "b": range(7000)})
    out_df = pd.DataFrame({"a": range(6873), "b": range(6873)})
    out = build_summary(
        step_type="DropMissingRows",
        output=out_df,
        metrics={},
        input_shape=in_df.shape,
    )
    assert out == "−127 rows (1.8%)"


def test_outlier_iqr_renders_row_delta() -> None:
    in_df = pd.DataFrame({"a": range(1000)})
    out_df = pd.DataFrame({"a": range(982)})
    out = build_summary(step_type="IQR", output=out_df, metrics={}, input_shape=in_df.shape)
    assert out == "−18 rows (1.8%)"


def test_oversampling_renders_positive_row_delta() -> None:
    in_df = pd.DataFrame({"a": range(800), "b": range(800)})
    out_df = pd.DataFrame({"a": range(1600), "b": range(1600)})
    out = build_summary(
        step_type="Oversampling", output=out_df, metrics={}, input_shape=in_df.shape
    )
    assert out == "+800 rows (100.0%)"


# ---------------------------------------------------------------------------
# Encoders / generators / selectors / binning — col delta with target count
# ---------------------------------------------------------------------------


def test_one_hot_encoder_shows_col_growth_with_final_count() -> None:
    in_df = pd.DataFrame({"a": range(100), "b": range(100), "c": range(100)})
    out_df = pd.DataFrame({f"c{i}": range(100) for i in range(8)})
    out = build_summary(
        step_type="OneHotEncoder", output=out_df, metrics={}, input_shape=in_df.shape
    )
    assert out == "+5 cols → 8 cols"


def test_feature_selection_shows_col_loss_with_final_count() -> None:
    in_df = pd.DataFrame({f"c{i}": range(100) for i in range(7)})
    out_df = pd.DataFrame({f"c{i}": range(100) for i in range(4)})
    out = build_summary(
        step_type="VarianceThreshold", output=out_df, metrics={}, input_shape=in_df.shape
    )
    assert out == "−3 cols → 4 cols"


def test_polynomial_features_shows_col_growth() -> None:
    in_df = pd.DataFrame({"a": range(100), "b": range(100)})
    out_df = pd.DataFrame({f"c{i}": range(100) for i in range(6)})
    out = build_summary(
        step_type="PolynomialFeatures", output=out_df, metrics={}, input_shape=in_df.shape
    )
    assert out == "+4 cols → 6 cols"


# ---------------------------------------------------------------------------
# Same-shape transformers — phrasing names the action
# ---------------------------------------------------------------------------


def test_scaler_with_same_shape_names_method() -> None:
    df = pd.DataFrame({"a": range(100), "b": range(100), "c": range(100)})
    out = build_summary(step_type="StandardScaler", output=df, metrics={}, input_shape=df.shape)
    assert out == "standard · 3 cols"


def test_robust_scaler_label() -> None:
    df = pd.DataFrame({"a": range(50)})
    out = build_summary(step_type="RobustScaler", output=df, metrics={}, input_shape=df.shape)
    assert out == "robust · 1 cols"


def test_imputer_with_same_shape_says_imputed() -> None:
    df = pd.DataFrame({"a": range(100), "b": range(100)})
    out = build_summary(step_type="SimpleImputer", output=df, metrics={}, input_shape=df.shape)
    assert out == "imputed · 2 cols"


def test_imputer_with_strategy_param_names_strategy() -> None:
    df = pd.DataFrame({"a": range(100), "b": range(100)})
    out = build_summary(
        step_type="SimpleImputer",
        output=df,
        metrics={},
        input_shape=df.shape,
        params={"strategy": "mean"},
    )
    assert out == "mean imputer · 2 cols"


def test_replacement_with_same_shape_says_cleaned() -> None:
    df = pd.DataFrame({"a": range(50)})
    out = build_summary(step_type="ValueReplacement", output=df, metrics={}, input_shape=df.shape)
    assert out == "cleaned · 1 cols"


def test_binning_with_same_shape_says_binned() -> None:
    df = pd.DataFrame({"a": range(50), "b": range(50)})
    out = build_summary(step_type="KBinsDiscretizer", output=df, metrics={}, input_shape=df.shape)
    assert out == "binned · 2 cols"


def test_binning_with_n_bins_param_includes_count() -> None:
    df = pd.DataFrame({"a": range(50)})
    out = build_summary(
        step_type="KBinsDiscretizer",
        output=df,
        metrics={},
        input_shape=df.shape,
        params={"n_bins": 5},
    )
    assert out == "binned (5 bins) · 1 cols"


# ---------------------------------------------------------------------------
# No baseline available — fall back to plain shape line
# ---------------------------------------------------------------------------


def test_transformer_without_input_shape_falls_back_to_shape_line() -> None:
    df = pd.DataFrame({"a": range(500), "b": range(500)})
    out = build_summary(step_type="StandardScaler", output=df, metrics={})
    assert out == "500 rows × 2 cols"


# ---------------------------------------------------------------------------
# Trainers
# ---------------------------------------------------------------------------


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


def test_classifier_step_type_routes_to_trainer() -> None:
    """Top-level metrics dict (no nested "metrics" key) is also accepted."""
    out = build_summary(
        step_type="random_forest_classifier",
        output=None,
        metrics={"accuracy": 0.91, "f1": 0.88},
    )
    assert out == "acc 0.91 · f1 0.88"


def test_training_summary_handles_test_prefixed_classification_metrics() -> None:
    """Engine flattens evaluation report into ``test_*`` / ``train_*`` keys.

    Make sure the builder finds them — this was previously broken for
    binary f1 only when the bare ``f1`` key was missing.
    """
    out = build_summary(
        step_type="basic_training",
        output=None,
        metrics={
            "train_accuracy": 0.99,
            "train_f1": 0.99,
            "test_accuracy": 0.873,
            "test_f1": 0.842,
        },
    )
    assert out == "acc 0.87 · f1 0.84"


def test_training_summary_handles_multiclass_f1_weighted() -> None:
    """Multiclass evaluation surfaces ``test_f1_weighted`` rather than ``test_f1``."""
    out = build_summary(
        step_type="basic_training",
        output=None,
        metrics={"test_accuracy": 0.85, "test_f1_weighted": 0.83},
    )
    assert out == "acc 0.85 · f1 0.83"


def test_training_summary_handles_test_prefixed_regression_metrics() -> None:
    """Regression training was previously not surfacing — it had no ``r2`` key,
    only ``test_r2`` / ``test_rmse``. Verify both are picked up now.
    """
    out = build_summary(
        step_type="linear_regression",
        output=None,
        metrics={
            "train_r2": 0.95,
            "train_rmse": 0.05,
            "test_r2": 0.87,
            "test_rmse": 0.123,
        },
    )
    assert out == "r² 0.87 · rmse 0.123"


def test_training_summary_falls_back_to_auc_when_no_acc_or_f1() -> None:
    out = build_summary(
        step_type="basic_training",
        output=None,
        metrics={"test_roc_auc": 0.93},
    )
    assert out == "auc 0.93"


# ---------------------------------------------------------------------------
# Defensive
# ---------------------------------------------------------------------------


def test_build_summary_swallows_exceptions() -> None:
    """A pathological metrics dict must never raise."""

    class _Boom:
        def __getitem__(self, key: str) -> object:
            raise RuntimeError("boom")

    out = build_summary(step_type="training", output=None, metrics=_Boom())  # type: ignore[arg-type]
    assert out is None
