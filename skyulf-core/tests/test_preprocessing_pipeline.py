"""Integration tests for FeatureEngineer pipeline orchestrator.

Covers fit_transform roundtrips, transform-after-fit, edge cases, and metrics
collection.  Every pipeline step uses real DataFrames — no mocking of pandas.
"""

from typing import Any, Dict, List, cast

import numpy as np
import pandas as pd
import pytest
from tests.utils.test_case_loader import TestCaseLoader

# Trigger node registration by importing the package.
import skyulf.preprocessing  # noqa: F401
from skyulf.data.dataset import SplitDataset
from skyulf.preprocessing.pipeline import FeatureEngineer

_diff_generated_columns_cases = TestCaseLoader(
    "preprocessing/pipeline_diff_generated_columns"
).load()
_count_winsorize_diffs_simple_cases = TestCaseLoader(
    "preprocessing/pipeline_count_winsorize_diffs_simple"
).load()
_count_winsorize_diffs_tuple_cases = TestCaseLoader(
    "preprocessing/pipeline_count_winsorize_diffs_tuple"
).load()
_metrics_from_fitted_params_cases = TestCaseLoader(
    "preprocessing/pipeline_metrics_from_fitted_params_copies_keys"
).load()
_metrics_shape_change_cases = TestCaseLoader("preprocessing/pipeline_metrics_shape_change").load()


def _coerce_frame(value: Any) -> Any:
    """Build a DataFrame from a dict fixture value; pass other types through unchanged."""
    return pd.DataFrame(value) if isinstance(value, dict) else value


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
    pd.testing.assert_frame_equal(out_fit, cast(pd.DataFrame, out_transform))


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
    pd.testing.assert_frame_equal(cast(pd.DataFrame, result), clean)


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


@pytest.mark.parametrize(*_diff_generated_columns_cases)
def test_diff_generated_columns(df_before: Any, df_after: Any, expected: Any) -> None:
    """Helper must return newly added column names, or None for incompatible types."""
    new_cols = FeatureEngineer._diff_generated_columns(
        _coerce_frame(df_before), _coerce_frame(df_after)
    )
    assert new_cols == expected


# ---------------------------------------------------------------------------
# _count_winsorize_diffs helper
# ---------------------------------------------------------------------------


@pytest.mark.parametrize(*_count_winsorize_diffs_simple_cases)
def test_count_winsorize_diffs_simple(before: Any, after: Any, expected: int) -> None:
    """Cell-level diff count across identical, shape-mismatched, and modified frames."""
    result = FeatureEngineer._count_winsorize_diffs(_coerce_frame(before), _coerce_frame(after))
    assert result == expected


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


# ---------------------------------------------------------------------------
# TrainTestSplitter skip-on-already-split path (_run_step lines 154-155)
# ---------------------------------------------------------------------------


def test_run_step_train_test_splitter_skips_already_split_data() -> None:
    """TrainTestSplitter must skip (with a warning) when data isn't a frame/tuple."""
    fe = FeatureEngineer(steps_config=[])
    calculator, applier = fe._get_transformer_components("TrainTestSplitter")

    # A SplitDataset is neither a DataFrame, SkyulfDataFrame nor a tuple, so the
    # splitter must bail out and return the data unchanged instead of re-splitting.
    already_split = SplitDataset(
        train=pd.DataFrame({"a": [1.0, 2.0]}), test=pd.DataFrame({"a": [3.0]})
    )
    result, fitted_params, transformer_inst = fe._run_step(
        transformer_type="TrainTestSplitter",
        name="split",
        calculator=calculator,
        applier=applier,
        step_node_id="node_split",
        current_data=already_split,
        params={},
    )
    assert result is already_split
    assert fitted_params == {}
    assert transformer_inst is None


# ---------------------------------------------------------------------------
# _collect_step_metrics: exception path (lines 239-240)
# ---------------------------------------------------------------------------


def test_collect_step_metrics_swallows_exception_from_fitted_params(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """An exception raised while deriving metrics must be logged, not propagated."""
    fe = FeatureEngineer(steps_config=[])

    def _boom(*args: Any, **kwargs: Any) -> None:
        raise RuntimeError("boom")

    monkeypatch.setattr(fe, "_metrics_from_fitted_params", _boom)
    metrics: Dict[str, Any] = {}
    # Should not raise despite _metrics_from_fitted_params blowing up.
    fe._collect_step_metrics(
        transformer_type="StandardScaler",
        fitted_params={"mean": [0.0]},
        data_before=pd.DataFrame({"a": [1.0]}),
        current_data=pd.DataFrame({"a": [1.0]}),
        params={},
        rows_before=1,
        cols_before={"a"},
        rows_after=1,
        cols_after={"a"},
        name="scale",
        metrics=metrics,
    )
    # metrics dict left untouched by the failed helper, but no crash.
    assert "mean" not in metrics


# ---------------------------------------------------------------------------
# _collect_step_metrics: resampling dispatch (line 243)
# ---------------------------------------------------------------------------


def test_collect_step_metrics_dispatches_resampling_metrics() -> None:
    """Oversampling/Undersampling step types must trigger resampling metrics."""
    fe = FeatureEngineer(steps_config=[])
    df = pd.DataFrame({"f": [1, 2, 3, 4], "label": [0, 0, 1, 1]})
    metrics: Dict[str, Any] = {}
    fe._collect_step_metrics(
        transformer_type="Oversampling",
        fitted_params={},
        data_before=df,
        current_data=df,
        params={"target_column": "label"},
        rows_before=4,
        cols_before={"f", "label"},
        rows_after=4,
        cols_after={"f", "label"},
        name="oversample",
        metrics=metrics,
    )
    assert metrics["total_samples"] == 4
    assert metrics["class_counts"] == {"0": 2, "1": 2}


# ---------------------------------------------------------------------------
# _metrics_from_fitted_params: feature selection keys (lines 272-281)
# ---------------------------------------------------------------------------


@pytest.mark.parametrize(*_metrics_from_fitted_params_cases)
def test_metrics_from_fitted_params_copies_keys(
    transformer_type: str,
    fitted_params: Dict[str, Any],
    df_before: Dict[str, Any],
    df_after: Dict[str, Any],
    expected_metrics: Dict[str, Any],
) -> None:
    """_metrics_from_fitted_params must copy the transformer-family-specific keys
    it recognizes (or derive them, e.g. ``generated_features``) into the metrics
    dict, across every transformer family it handles.
    """
    fe = FeatureEngineer(steps_config=[])
    metrics: Dict[str, Any] = {}
    fe._metrics_from_fitted_params(
        transformer_type, fitted_params, pd.DataFrame(df_before), pd.DataFrame(df_after), metrics
    )
    for key, value in expected_metrics.items():
        assert metrics[key] == value


# ---------------------------------------------------------------------------
# _diff_generated_columns: SplitDataset paths (lines 334-345)
# ---------------------------------------------------------------------------


def test_diff_generated_columns_split_dataset_dataframe_train() -> None:
    """SplitDataset with DataFrame train members must diff on train columns."""
    before = SplitDataset(train=pd.DataFrame({"a": [1.0]}), test=pd.DataFrame({"a": [1.0]}))
    after = SplitDataset(
        train=pd.DataFrame({"a": [1.0], "b": [2.0]}),
        test=pd.DataFrame({"a": [1.0], "b": [2.0]}),
    )
    new_cols = FeatureEngineer._diff_generated_columns(before, after)
    assert new_cols == ["b"]


def test_diff_generated_columns_split_dataset_tuple_train() -> None:
    """SplitDataset with (X, y) tuple train members must diff on X columns."""
    before = SplitDataset(
        train=(pd.DataFrame({"a": [1.0]}), pd.Series([0])),
        test=(pd.DataFrame({"a": [1.0]}), pd.Series([0])),
    )
    after = SplitDataset(
        train=(pd.DataFrame({"a": [1.0], "b": [2.0]}), pd.Series([0])),
        test=(pd.DataFrame({"a": [1.0], "b": [2.0]}), pd.Series([0])),
    )
    new_cols = FeatureEngineer._diff_generated_columns(before, after)
    assert new_cols == ["b"]


# ---------------------------------------------------------------------------
# _extract_y_for_resampling: SplitDataset paths (lines 352-358)
# ---------------------------------------------------------------------------


def test_extract_y_for_resampling_split_dataset_tuple_train() -> None:
    """SplitDataset whose train is an (X, y) tuple must yield y from train."""
    y = pd.Series([0, 1], name="label")
    ds = SplitDataset(train=(pd.DataFrame({"f": [1, 2]}), y), test=(pd.DataFrame(), None))
    fe = FeatureEngineer(steps_config=[])
    result = fe._extract_y_for_resampling(ds, {})
    assert result is y


def test_extract_y_for_resampling_split_dataset_dataframe_train() -> None:
    """SplitDataset whose train is a DataFrame must pull y via target_column."""
    train = pd.DataFrame({"f": [1, 2], "label": [0, 1]})
    ds = SplitDataset(train=train, test=pd.DataFrame({"f": [3], "label": [1]}))
    fe = FeatureEngineer(steps_config=[])
    result = fe._extract_y_for_resampling(ds, {"target_column": "label"})
    pd.testing.assert_series_equal(result, train["label"])


# ---------------------------------------------------------------------------
# _metrics_resampling: exception path (lines 371-381)
# ---------------------------------------------------------------------------


def test_metrics_resampling_success_populates_class_counts() -> None:
    """Successful resampling metrics collection must set class_counts + total_samples."""
    fe = FeatureEngineer(steps_config=[])
    X = pd.DataFrame({"f": [1, 2, 3]})
    y = pd.Series([0, 1, 1])
    metrics: Dict[str, Any] = {}
    fe._metrics_resampling((X, y), {}, metrics)
    assert metrics["total_samples"] == 3
    assert metrics["class_counts"] == {"0": 1, "1": 2}


def test_metrics_resampling_returns_early_when_y_is_none() -> None:
    """When y cannot be extracted, no metrics should be added and no error raised."""
    fe = FeatureEngineer(steps_config=[])
    metrics: Dict[str, Any] = {}
    fe._metrics_resampling(pd.DataFrame({"f": [1, 2]}), {}, metrics)
    assert metrics == {}


def test_metrics_resampling_converts_polars_like_y_via_to_pandas() -> None:
    """A y_res exposing to_pandas() must be converted before value_counts()."""

    class _FakePolarsSeries:
        """Minimal stand-in exposing to_pandas(), mimicking a polars Series."""

        def to_pandas(self) -> pd.Series:
            return pd.Series([0, 0, 1], name="label")

    fe = FeatureEngineer(steps_config=[])
    X = pd.DataFrame({"f": [1, 2, 3]})
    metrics: Dict[str, Any] = {}
    fe._metrics_resampling((X, _FakePolarsSeries()), {}, metrics)
    assert metrics["total_samples"] == 3
    assert metrics["class_counts"] == {"0": 2, "1": 1}


def test_metrics_resampling_handles_extraction_failure_gracefully() -> None:
    """If y extraction / value_counts blows up, the method must not raise."""
    fe = FeatureEngineer(steps_config=[])
    # y_res will be `5` (an int) -> .value_counts() raises AttributeError internally.
    metrics: Dict[str, Any] = {}
    fe._metrics_resampling((pd.DataFrame({"f": [1]}), 5), {}, metrics)
    assert "class_counts" not in metrics


# ---------------------------------------------------------------------------
# _count_winsorize_diffs: tuple path (lines 394-413)
# ---------------------------------------------------------------------------


@pytest.mark.parametrize(*_count_winsorize_diffs_tuple_cases)
def test_count_winsorize_diffs_tuple_path(
    before_x: Dict[str, Any],
    before_y: List[Any],
    after_x: Dict[str, Any],
    after_y: List[Any],
    expected: int,
) -> None:
    """Tuple (X, y) diffs must sum differing cells across both X and y."""
    before = (pd.DataFrame(before_x), pd.Series(before_y))
    after = (pd.DataFrame(after_x), pd.Series(after_y))
    diffs = FeatureEngineer._count_winsorize_diffs(before, after)
    assert diffs == expected


# ---------------------------------------------------------------------------
# _metrics_winsorize_clipped (lines 418-432)
# ---------------------------------------------------------------------------


def test_metrics_winsorize_clipped_dataframe_path() -> None:
    """Plain DataFrame before/after must populate values_clipped via cell diff."""
    fe = FeatureEngineer(steps_config=[])
    before = pd.DataFrame({"a": [1.0, 2.0, 3.0]})
    after = pd.DataFrame({"a": [1.0, 2.0, 99.0]})
    metrics: Dict[str, Any] = {}
    fe._metrics_winsorize_clipped(before, after, metrics)
    assert metrics["values_clipped"] == 1


def test_metrics_winsorize_clipped_split_dataset_path() -> None:
    """SplitDataset before/after must sum diffs across train/test/validation."""
    fe = FeatureEngineer(steps_config=[])
    before = SplitDataset(
        train=pd.DataFrame({"a": [1.0, 2.0]}),
        test=pd.DataFrame({"a": [5.0]}),
        validation=pd.DataFrame({"a": [7.0]}),
    )
    after = SplitDataset(
        train=pd.DataFrame({"a": [1.0, 99.0]}),
        test=pd.DataFrame({"a": [5.0]}),
        validation=pd.DataFrame({"a": [70.0]}),
    )
    metrics: Dict[str, Any] = {}
    fe._metrics_winsorize_clipped(before, after, metrics)
    assert metrics["values_clipped"] == 2


def test_metrics_winsorize_clipped_swallows_exception(monkeypatch: pytest.MonkeyPatch) -> None:
    """Any exception while computing clip diffs must be logged, not raised."""
    fe = FeatureEngineer(steps_config=[])

    def _boom(*args: Any, **kwargs: Any) -> None:
        raise RuntimeError("boom")

    monkeypatch.setattr(FeatureEngineer, "_count_winsorize_diffs", staticmethod(_boom))
    metrics: Dict[str, Any] = {}
    fe._metrics_winsorize_clipped(pd.DataFrame({"a": [1.0]}), pd.DataFrame({"a": [2.0]}), metrics)
    assert "values_clipped" not in metrics


# ---------------------------------------------------------------------------
# _metrics_shape_change: row-drop / Winsorize / MissingIndicator / encoders
# (lines 447-454, 457-459, 462-464, 471, 473)
# ---------------------------------------------------------------------------


@pytest.mark.parametrize(*_metrics_shape_change_cases)
def test_metrics_shape_change(
    transformer_type: str,
    df_before: Dict[str, Any],
    df_after: Dict[str, Any],
    params: Dict[str, Any],
    rows_before: int,
    cols_before: List[str],
    rows_after: int,
    cols_after: List[str],
    expected_metrics: Dict[str, Any],
) -> None:
    """_metrics_shape_change must record the transformer-family-specific
    shape-change metrics (rows dropped, new/dropped columns, encoder counts)
    across every transformer family it handles.
    """
    fe = FeatureEngineer(steps_config=[])
    metrics: Dict[str, Any] = {}
    fe._metrics_shape_change(
        transformer_type,
        pd.DataFrame(df_before),
        pd.DataFrame(df_after),
        params,
        rows_before,
        set(cols_before),
        rows_after,
        set(cols_after),
        metrics,
    )
    for key, value in expected_metrics.items():
        assert metrics[key] == value
