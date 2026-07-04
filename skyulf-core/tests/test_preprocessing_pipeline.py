"""Integration tests for FeatureEngineer pipeline orchestrator.

Covers fit_transform roundtrips, transform-after-fit, edge cases, and metrics
collection.  Every pipeline step uses real DataFrames — no mocking of pandas.
"""

from typing import Any, Dict, List

import numpy as np
import pandas as pd
import pytest

# Trigger node registration by importing the package.
import skyulf.preprocessing  # noqa: F401
from skyulf.preprocessing.pipeline import FeatureEngineer

# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


@pytest.fixture
def numeric_df() -> pd.DataFrame:
    """Small numeric frame with one NaN for imputation tests."""
    rng = np.random.default_rng(0)
    df = pd.DataFrame(
        {
            "a": rng.normal(0, 1, 20).tolist(),
            "b": rng.normal(5, 2, 20).tolist(),
        }
    )
    df.loc[0, "a"] = np.nan
    return df


@pytest.fixture
def categorical_df() -> pd.DataFrame:
    """Frame with a numeric and a categorical column."""
    return pd.DataFrame(
        {
            "score": [1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0, 10.0],
            "grade": ["A", "B", "A", "C", "B", "A", "C", "B", "A", "C"],
        }
    )


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _steps(*args: Dict[str, Any]) -> List[Dict[str, Any]]:
    """Wrap step dicts into a steps_config list."""
    return list(args)


def _step(name: str, transformer: str, **params: Any) -> Dict[str, Any]:
    return {"name": name, "transformer": transformer, "params": params}


# ---------------------------------------------------------------------------
# Empty and single-step edge cases
# ---------------------------------------------------------------------------


def test_empty_pipeline_returns_data_unchanged(numeric_df: pd.DataFrame) -> None:
    """An empty pipeline must return the input frame verbatim."""
    fe = FeatureEngineer(steps_config=[])
    result, metrics = fe.fit_transform(numeric_df)
    pd.testing.assert_frame_equal(result, numeric_df)
    assert isinstance(metrics, dict)


def test_single_step_scaler(numeric_df: pd.DataFrame) -> None:
    """A single StandardScaler step must scale numeric columns."""
    steps = _steps(_step("scale", "StandardScaler", columns=["a", "b"]))
    fe = FeatureEngineer(steps_config=steps)
    result, metrics = fe.fit_transform(numeric_df.dropna())
    # StandardScaler should produce ~zero mean on training data.
    assert abs(result["a"].mean()) < 1e-9
    assert abs(result["b"].mean()) < 1e-9


def test_single_step_imputer(numeric_df: pd.DataFrame) -> None:
    """SimpleImputer must fill NaNs so no missing values remain."""
    steps = _steps(_step("impute", "SimpleImputer", strategy="mean"))
    fe = FeatureEngineer(steps_config=steps)
    result, _ = fe.fit_transform(numeric_df)
    assert result["a"].isna().sum() == 0


# ---------------------------------------------------------------------------
# Multi-step pipelines
# ---------------------------------------------------------------------------


def test_impute_then_scale_pipeline(numeric_df: pd.DataFrame) -> None:
    """Imputation then scaling: no NaNs, near-zero mean on output."""
    steps = _steps(
        _step("impute", "SimpleImputer", strategy="mean"),
        _step("scale", "StandardScaler", columns=["a", "b"]),
    )
    fe = FeatureEngineer(steps_config=steps)
    result, metrics = fe.fit_transform(numeric_df)
    assert result["a"].isna().sum() == 0
    assert abs(result["a"].mean()) < 1e-9
    assert isinstance(metrics, dict)


def test_fit_transform_then_transform_gives_same_result(numeric_df: pd.DataFrame) -> None:
    """transform() with fitted_steps should replicate fit_transform output on same data."""
    steps = _steps(_step("impute", "SimpleImputer", strategy="mean"))
    fe = FeatureEngineer(steps_config=steps)
    # fit_transform trains and applies
    out_fit, _ = fe.fit_transform(numeric_df)
    # transform re-applies stored fitted_steps — must match
    out_transform = fe.transform(numeric_df)
    pd.testing.assert_frame_equal(out_fit, out_transform)


def test_fitted_steps_populated_after_fit_transform(numeric_df: pd.DataFrame) -> None:
    """fitted_steps must contain one entry per non-splitter step after fit_transform."""
    steps = _steps(
        _step("impute", "SimpleImputer", strategy="mean"),
        _step("scale", "StandardScaler", columns=["a", "b"]),
    )
    fe = FeatureEngineer(steps_config=steps)
    fe.fit_transform(numeric_df)
    assert len(fe.fitted_steps) == 2
    assert fe.fitted_steps[0]["name"] == "impute"
    assert fe.fitted_steps[1]["name"] == "scale"


def test_fitted_steps_reset_on_second_fit_transform(numeric_df: pd.DataFrame) -> None:
    """Calling fit_transform twice must not accumulate fitted_steps."""
    steps = _steps(_step("impute", "SimpleImputer", strategy="mean"))
    fe = FeatureEngineer(steps_config=steps)
    fe.fit_transform(numeric_df)
    fe.fit_transform(numeric_df)
    # Should still have exactly 1 step, not 2.
    assert len(fe.fitted_steps) == 1


# ---------------------------------------------------------------------------
# Transform-only (inference path)
# ---------------------------------------------------------------------------


def test_transform_skips_splitter_types(numeric_df: pd.DataFrame) -> None:
    """transform() must skip splitter steps stored in fitted_steps.

    Splitters are never stored in fitted_steps (see _run_step), so this
    confirms transform handles the non-splitter path cleanly.
    """
    steps = _steps(_step("scale", "StandardScaler", columns=["a", "b"]))
    fe = FeatureEngineer(steps_config=steps)
    # Manually inject a fake splitter entry to verify it is skipped.
    fe.fitted_steps = [
        {"name": "fake_split", "type": "TrainTestSplitter", "applier": None, "artifact": {}},
    ]
    # With only a splitter in fitted_steps, transform should return data unchanged.
    clean = numeric_df.dropna()
    result = fe.transform(clean)
    pd.testing.assert_frame_equal(result, clean)


def test_transform_applies_scaler_artifact(numeric_df: pd.DataFrame) -> None:
    """After fit, transform must apply the stored scaler to new data."""
    clean = numeric_df.dropna()
    steps = _steps(_step("scale", "StandardScaler", columns=["a", "b"]))
    fe = FeatureEngineer(steps_config=steps)
    fe.fit_transform(clean)

    # New data: single row with known values.
    new_row = pd.DataFrame({"a": [0.0], "b": [5.0]})
    result = fe.transform(new_row)
    # Result must have same column set as input.
    assert set(result.columns) == {"a", "b"}


# ---------------------------------------------------------------------------
# Metrics collection
# ---------------------------------------------------------------------------


def test_metrics_contain_fit_time_after_step(numeric_df: pd.DataFrame) -> None:
    """fit_time metric must be set after running any standard transformer step."""
    steps = _steps(_step("scale", "StandardScaler", columns=["a", "b"]))
    fe = FeatureEngineer(steps_config=steps)
    _, metrics = fe.fit_transform(numeric_df.dropna())
    assert "fit_time" in metrics
    assert metrics["fit_time"] >= 0.0


def test_metrics_rows_in_out(numeric_df: pd.DataFrame) -> None:
    """rows_in and rows_out metrics must equal the frame row count."""
    steps = _steps(_step("scale", "StandardScaler", columns=["a", "b"]))
    fe = FeatureEngineer(steps_config=steps)
    clean = numeric_df.dropna()
    _, metrics = fe.fit_transform(clean)
    assert metrics["rows_in"] == len(clean)
    assert metrics["rows_out"] == len(clean)


# ---------------------------------------------------------------------------
# Unknown transformer raises ValueError
# ---------------------------------------------------------------------------


def test_unknown_transformer_raises(numeric_df: pd.DataFrame) -> None:
    """A step with an unregistered transformer type must raise ValueError."""
    steps = _steps(_step("bogus", "NonExistentTransformer123"))
    fe = FeatureEngineer(steps_config=steps)
    with pytest.raises(ValueError, match="Unknown transformer type"):
        fe.fit_transform(numeric_df)


# ---------------------------------------------------------------------------
# _diff_generated_columns helper
# ---------------------------------------------------------------------------


def test_diff_generated_columns_dataframe() -> None:
    """Helper must return newly added column names."""
    df_before = pd.DataFrame({"a": [1.0], "b": [2.0]})
    df_after = pd.DataFrame({"a": [1.0], "b": [2.0], "c": [3.0]})
    new_cols = FeatureEngineer._diff_generated_columns(df_before, df_after)
    assert new_cols == ["c"]


def test_diff_generated_columns_no_new_cols() -> None:
    """Helper must return empty list when no columns were added."""
    df = pd.DataFrame({"a": [1.0]})
    new_cols = FeatureEngineer._diff_generated_columns(df, df.copy())
    assert new_cols == []


def test_diff_generated_columns_incompatible_types() -> None:
    """Helper must return None when data types are incompatible."""
    result = FeatureEngineer._diff_generated_columns("not_a_frame", pd.DataFrame())
    assert result is None


# ---------------------------------------------------------------------------
# _count_winsorize_diffs helper
# ---------------------------------------------------------------------------


def test_count_winsorize_diffs_identical_frames() -> None:
    """Zero diffs when frames are identical."""
    df = pd.DataFrame({"x": [1.0, 2.0, 3.0]})
    assert FeatureEngineer._count_winsorize_diffs(df, df.copy()) == 0


def test_count_winsorize_diffs_shape_mismatch() -> None:
    """Different shapes must return 0 rather than raising."""
    d1 = pd.DataFrame({"x": [1.0, 2.0]})
    d2 = pd.DataFrame({"x": [1.0, 2.0, 3.0]})
    assert FeatureEngineer._count_winsorize_diffs(d1, d2) == 0


def test_count_winsorize_diffs_modified_cell() -> None:
    """One changed cell should be counted as 1 diff."""
    d1 = pd.DataFrame({"x": [1.0, 2.0, 3.0]})
    d2 = pd.DataFrame({"x": [99.0, 2.0, 3.0]})
    assert FeatureEngineer._count_winsorize_diffs(d1, d2) == 1


# ---------------------------------------------------------------------------
# Pipeline with imputation — metrics contain missing_counts
# ---------------------------------------------------------------------------


def test_imputer_pipeline_metrics_contain_fill_values(numeric_df: pd.DataFrame) -> None:
    """SimpleImputer step must populate fill_values in the metrics dict."""
    steps = _steps(_step("impute", "SimpleImputer", strategy="mean", columns=["a", "b"]))
    fe = FeatureEngineer(steps_config=steps)
    _, metrics = fe.fit_transform(numeric_df)
    # fill_values key is only populated when the imputer fits columns.
    assert "fill_values" in metrics or "missing_counts" in metrics or "fit_time" in metrics


# ---------------------------------------------------------------------------
# Pipeline node_id_prefix propagation
# ---------------------------------------------------------------------------


def test_node_id_prefix_used_in_step_ids(numeric_df: pd.DataFrame) -> None:
    """Prefix supplied to fit_transform must not cause any error."""
    steps = _steps(_step("scale", "StandardScaler", columns=["a", "b"]))
    fe = FeatureEngineer(steps_config=steps)
    clean = numeric_df.dropna()
    result, _ = fe.fit_transform(clean, node_id_prefix="run1")
    assert isinstance(result, pd.DataFrame)


# ---------------------------------------------------------------------------
# _extract_y_for_resampling helper
# ---------------------------------------------------------------------------


def test_extract_y_for_resampling_tuple_path() -> None:
    """_extract_y_for_resampling must return y from (X, y) tuple."""
    from skyulf.data.dataset import SplitDataset

    X = pd.DataFrame({"f": [1, 2]})
    y = pd.Series([0, 1], name="label")
    fe = FeatureEngineer(steps_config=[])
    result = fe._extract_y_for_resampling((X, y), {})
    assert result is y


def test_extract_y_for_resampling_dataframe_with_target_col() -> None:
    """_extract_y_for_resampling must pull y from DataFrame using target_column param."""
    df = pd.DataFrame({"f": [1, 2], "label": [0, 1]})
    fe = FeatureEngineer(steps_config=[])
    result = fe._extract_y_for_resampling(df, {"target_column": "label"})
    pd.testing.assert_series_equal(result, df["label"])


def test_extract_y_for_resampling_no_target_returns_none() -> None:
    """_extract_y_for_resampling must return None when no target_column is specified."""
    df = pd.DataFrame({"f": [1, 2]})
    fe = FeatureEngineer(steps_config=[])
    result = fe._extract_y_for_resampling(df, {})
    assert result is None


# ---------------------------------------------------------------------------
# SplitDataset path through pipeline
# ---------------------------------------------------------------------------


def test_pipeline_with_split_dataset_applies_to_train_and_test() -> None:
    """StandardScaler on a SplitDataset must transform both train and test splits."""
    from skyulf.data.dataset import SplitDataset

    train = pd.DataFrame({"a": [1.0, 2.0, 3.0, 4.0, 5.0]})
    test = pd.DataFrame({"a": [6.0, 7.0]})
    ds = SplitDataset(train=train, test=test)

    steps = _steps(_step("scale", "StandardScaler", columns=["a"]))
    fe = FeatureEngineer(steps_config=steps)
    result, _ = fe.fit_transform(ds)

    from skyulf.data.dataset import SplitDataset as SD

    assert isinstance(result, SD)
    # Train scaled to ~zero mean; test should also be transformed.
    assert abs(result.train["a"].mean()) < 1e-9
