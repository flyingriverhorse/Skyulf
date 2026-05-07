"""Unit tests for the per-node summary builder.

Each family lives in its own focused test so a phrasing change is
quickly traceable.
"""

from __future__ import annotations

import pandas as pd

from backend.ml_pipeline._execution.summary import build_summary
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
    assert out == "82/18% · 7,000 / 1,500 × 2 cols"


def test_split_dataset_summary_with_validation() -> None:
    train = pd.DataFrame({"x": range(7000)})
    test = pd.DataFrame({"x": range(1500)})
    val = pd.DataFrame({"x": range(1500)})
    sd = SplitDataset(train=train, test=test, validation=val)
    out = build_summary(step_type="Split", output=sd, metrics={})
    assert out == "70/15/15% · 7,000 / 1,500 / 1,500 × 1 cols"


def test_split_dataset_with_x_y_tuple_payload() -> None:
    """FeatureTargetSplit produces (X, y) tuples per slot."""
    X_train = pd.DataFrame({"x": range(7000)})
    y_train = pd.Series(range(7000))
    X_test = pd.DataFrame({"x": range(1500)})
    y_test = pd.Series(range(1500))
    sd = SplitDataset(train=(X_train, y_train), test=(X_test, y_test))
    out = build_summary(step_type="feature_target_split", output=sd, metrics={})
    # 1 X col + y → 2 cols reported.
    assert out == "82/18% · 7,000 / 1,500 × 2 cols"


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
    binary f1 only when the bare ``f1`` key was missing. Train/test
    here diverge by 0.12 so the overfit-gap badge also appears.
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
    assert out == "acc 0.87 · f1 0.84 · ▲0.12"


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
    assert out == "r² 0.87 · rmse 0.123 · ▲0.08"


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


# ---------------------------------------------------------------------------
# Overfit-gap badge — surfaces train→test divergence on the card
# ---------------------------------------------------------------------------


def test_training_summary_appends_overfit_gap_when_train_diverges() -> None:
    """When ``train_*`` and ``test_*`` differ by ≥ 0.05, append ▲gap."""
    out = build_summary(
        step_type="basic_training",
        output=None,
        metrics={
            "train_accuracy": 0.99,
            "train_f1": 0.99,
            "test_accuracy": 0.80,
            "test_f1": 0.78,
        },
    )
    assert out == "acc 0.80 · f1 0.78 · ▲0.19"


def test_training_summary_omits_gap_when_train_test_close() -> None:
    """A small gap (< 0.05) is not interesting; keep the card clean."""
    out = build_summary(
        step_type="basic_training",
        output=None,
        metrics={
            "train_accuracy": 0.88,
            "test_accuracy": 0.86,
        },
    )
    assert out == "acc 0.86"


def test_regression_overfit_gap_uses_r2_direction() -> None:
    out = build_summary(
        step_type="linear_regression",
        output=None,
        metrics={"train_r2": 0.95, "test_r2": 0.72, "test_rmse": 0.4, "train_rmse": 0.1},
    )
    assert out == "r² 0.72 · rmse 0.4 · ▲0.23"


# ---------------------------------------------------------------------------
# Advanced tuning — best_score fallback + trial count + per-split phrasing
# ---------------------------------------------------------------------------


def test_advanced_tuning_leads_with_best_score_and_scoring_metric() -> None:
    """Tuning headline mirrors the JobsDrawer's "Best Score" line.

    Even when post-tune eval metrics exist, the card leads with the
    tuner's own objective so the number on the card matches the number
    the user sees in the drawer (and updates run-over-run, which the
    rounded test-set numbers often don't on small holdouts).
    """
    out = build_summary(
        step_type="advanced_tuning",
        output=None,
        metrics={
            "best_score": 0.9324,
            "scoring_metric": "f1_weighted",
            "trials": [{}] * 10,
            "test_accuracy": 0.9667,
            "test_f1_weighted": 0.9666,
        },
    )
    assert out == "f1w 0.932 · 10 trials"


def test_advanced_tuning_uses_best_score_when_no_test_metrics() -> None:
    out = build_summary(
        step_type="advanced_tuning",
        output=None,
        metrics={
            "best_score": 0.913,
            "scoring_metric": "f1",
            "trials": [{}, {}, {}, {}, {}],
        },
    )
    assert out == "f1 0.913 · 5 trials"


def test_advanced_tuning_int_trials_is_accepted() -> None:
    """Some tuners report ``trials`` as a count rather than a list."""
    out = build_summary(
        step_type="advanced_tuning",
        output=None,
        metrics={"best_score": 0.5, "scoring_metric": "r2", "trials": 12},
    )
    assert out == "r² 0.500 · 12 trials"


def test_advanced_tuning_neg_loss_is_sign_flipped() -> None:
    """sklearn losses come in as ``neg_*`` (higher-is-better). Show the
    natural magnitude on the card so the user can reason about it."""
    out = build_summary(
        step_type="advanced_tuning",
        output=None,
        metrics={
            "best_score": -12.34,
            "scoring_metric": "neg_mean_squared_error",
            "trials": 20,
        },
    )
    assert out == "mse 12.340 · 20 trials"


def test_advanced_tuning_legacy_eval_only_falls_back_to_test_metrics() -> None:
    """Old job rows without ``best_score`` (pre-tuning-summary refactor)
    still render via the eval headline."""
    out = build_summary(
        step_type="advanced_tuning",
        output=None,
        metrics={
            "trials": [{}] * 40,
            "test_accuracy": 0.873,
            "test_f1": 0.842,
        },
    )
    assert out == "acc 0.87 · f1 0.84 · 40 trials"


# ---------------------------------------------------------------------------
# Loader / snapshot dtype-mix enrichment
# ---------------------------------------------------------------------------


def test_loader_appends_dtype_breakdown_for_mixed_schema() -> None:
    df = pd.DataFrame(
        {
            "n1": range(100),
            "n2": range(100),
            "c1": ["x"] * 100,
            "c2": ["y"] * 100,
        }
    )
    out = build_summary(step_type="data_loader", output=df, metrics={})
    assert out == "100 rows × 4 cols (2 num · 2 cat)"


def test_loader_omits_dtype_breakdown_for_uniform_schema() -> None:
    """Single-bucket dtype mix adds no info; keep the line short."""
    df = pd.DataFrame({"a": range(50), "b": range(50)})
    out = build_summary(step_type="data_loader", output=df, metrics={})
    assert out == "50 rows × 2 cols"


def test_snapshot_node_also_uses_loader_phrasing() -> None:
    df = pd.DataFrame({"n": range(10), "c": ["x"] * 10})
    out = build_summary(step_type="DataSnapshot", output=df, metrics={})
    assert out == "10 rows × 2 cols (1 num · 1 cat)"
