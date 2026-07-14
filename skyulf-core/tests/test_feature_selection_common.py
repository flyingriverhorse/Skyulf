"""Tests for shared helpers in feature_selection/_common.py.

Covers every public function: problem-type inference, score-function/estimator
resolution, drop-list building, target extraction, sklearn-y preparation,
univariate & model selector construction, chi2 rescaling, and artifact helpers.
"""

import numpy as np
import pandas as pd
import polars as pl
import pytest
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor
from sklearn.feature_selection import (
    RFE,
    SelectFdr,
    SelectFpr,
    SelectFromModel,
    SelectFwe,
    SelectKBest,
    SelectPercentile,
    f_classif,
    f_regression,
    mutual_info_regression,
)
from sklearn.linear_model import LinearRegression, LogisticRegression
from tests.utils.dataset_loader import load_sample_dataset
from tests.utils.test_case_loader import TestCaseLoader

from skyulf.preprocessing.feature_selection._common import (
    _build_model_selector,
    _build_univariate_selector,
    _drop_selected_pandas,
    _drop_selected_polars,
    _extract_target,
    _infer_problem_type,
    _maybe_chi2_rescale,
    _model_feature_importances,
    _prepare_sklearn_y,
    _resolve_candidate_columns,
    _resolve_drop_list,
    _resolve_estimator,
    _resolve_generic_param,
    _resolve_problem_type,
    _resolve_score_function,
    _univariate_no_target_artifact,
    _univariate_score_dicts,
)

# Registries mapping JSON-fixture name strings to the real sklearn objects.
_SCORE_FUNC_REGISTRY = {
    "f_classif": f_classif,
    "f_regression": f_regression,
    "mutual_info_regression": mutual_info_regression,
}
_ESTIMATOR_CLASS_REGISTRY = {
    "LogisticRegression": LogisticRegression,
    "LinearRegression": LinearRegression,
    "RandomForestClassifier": RandomForestClassifier,
    "RandomForestRegressor": RandomForestRegressor,
}
_UNIVARIATE_SELECTOR_CLASSES = {
    "SelectKBest": SelectKBest,
    "SelectPercentile": SelectPercentile,
    "SelectFpr": SelectFpr,
    "SelectFdr": SelectFdr,
    "SelectFwe": SelectFwe,
}
_MODEL_SELECTOR_CLASSES = {"SelectFromModel": SelectFromModel, "RFE": RFE}
_ESTIMATOR_FACTORIES = {
    "LogisticRegression": LogisticRegression,
    "LinearRegression": LinearRegression,
}
_IMPORTANCE_ESTIMATOR_FACTORIES = {
    "RandomForestClassifier": lambda: RandomForestClassifier(n_estimators=5, random_state=0),
    "LogisticRegression": lambda: LogisticRegression(max_iter=1000),
}
_DROP_SELECTED_FUNCS = {"pandas": _drop_selected_pandas, "polars": _drop_selected_polars}
_DROP_SELECTED_DF_BUILDERS = {
    "pandas": lambda: pd.DataFrame({"a": [1, 2], "b": [3, 4], "c": [5, 6]}),
    "polars": lambda: pl.DataFrame({"a": [1, 2], "b": [3, 4], "c": [5, 6]}),
}

_infer_problem_type_cases = TestCaseLoader(
    "preprocessing/feature_selection_common", group="infer_problem_type"
).load()
_resolve_score_function_cases = TestCaseLoader(
    "preprocessing/feature_selection_common", group="resolve_score_function"
).load()
_resolve_estimator_cases = TestCaseLoader(
    "preprocessing/feature_selection_common", group="resolve_estimator"
).load()
_resolve_drop_list_cases = TestCaseLoader(
    "preprocessing/feature_selection_common", group="resolve_drop_list"
).load()
_drop_selected_cases = TestCaseLoader(
    "preprocessing/feature_selection_common", group="drop_selected"
).load()
_resolve_problem_type_cases = TestCaseLoader(
    "preprocessing/feature_selection_common", group="resolve_problem_type"
).load()
_resolve_generic_param_cases = TestCaseLoader(
    "preprocessing/feature_selection_common", group="resolve_generic_param"
).load()
_build_univariate_selector_cases = TestCaseLoader(
    "preprocessing/feature_selection_common", group="build_univariate_selector"
).load()
_build_model_selector_cases = TestCaseLoader(
    "preprocessing/feature_selection_common", group="build_model_selector"
).load()
_maybe_chi2_rescale_cases = TestCaseLoader(
    "preprocessing/feature_selection_common", group="maybe_chi2_rescale"
).load()
_model_feature_importances_cases = TestCaseLoader(
    "preprocessing/feature_selection_common", group="model_feature_importances"
).load()

# ---------------------------------------------------------------------------
# _infer_problem_type
# ---------------------------------------------------------------------------


class TestInferProblemType:
    @pytest.mark.parametrize(*_infer_problem_type_cases)
    def test_infer_problem_type(self, values: list, dtype: str | None, expected: str) -> None:
        """Various Series shapes/dtypes must infer the documented problem type."""
        s = pd.Series(values, dtype=dtype)
        assert _infer_problem_type(s) == expected

    def test_infer_problem_type_logs_debug_note_on_classification_heuristic(
        self, caplog: pytest.LogCaptureFixture
    ) -> None:
        """Regression test: the numeric <=10-unique-values classification
        heuristic is a coarse cutoff with no config knob, so a debug note
        must be logged to make the inference visible in diagnostics."""
        import logging

        s = pd.Series([1.0, 2.0, 3.0], dtype="float64")
        with caplog.at_level(
            logging.DEBUG, logger="skyulf.preprocessing.feature_selection._common"
        ):
            result = _infer_problem_type(s)
        assert result == "classification"
        assert any("classification" in rec.message for rec in caplog.records)


# ---------------------------------------------------------------------------
# _resolve_score_function
# ---------------------------------------------------------------------------


class TestResolveScoreFunction:
    @pytest.mark.parametrize(*_resolve_score_function_cases)
    def test_resolve_score_function(
        self, name: str | None, problem_type: str, expected_func_name: str
    ) -> None:
        """An explicit name or problem-type default must resolve to the right score function."""
        assert (
            _resolve_score_function(name, problem_type) is _SCORE_FUNC_REGISTRY[expected_func_name]
        )


# ---------------------------------------------------------------------------
# _resolve_estimator
# ---------------------------------------------------------------------------


class TestResolveEstimator:
    @pytest.mark.parametrize(*_resolve_estimator_cases)
    def test_resolve_estimator(
        self, key: str, problem_type: str, expected_class_name: str | None
    ) -> None:
        """Known estimator keys must resolve to the right estimator class; unknown keys to None."""
        est = _resolve_estimator(key, problem_type)
        if expected_class_name is None:
            assert est is None
        else:
            assert isinstance(est, _ESTIMATOR_CLASS_REGISTRY[expected_class_name])


# ---------------------------------------------------------------------------
# _resolve_drop_list
# ---------------------------------------------------------------------------


class TestResolveDropList:
    @pytest.mark.parametrize(*_resolve_drop_list_cases)
    def test_resolve_drop_list(
        self, config: dict, existing_cols: list, expected_drop: list
    ) -> None:
        """The drop list must be candidates minus selected, filtered by existing_cols."""
        assert set(_resolve_drop_list(config, existing_cols)) == set(expected_drop)


# ---------------------------------------------------------------------------
# _drop_selected_pandas / _drop_selected_polars
# ---------------------------------------------------------------------------


class TestDropSelected:
    @pytest.mark.parametrize(*_drop_selected_cases)
    def test_drop_selected(self, engine: str, drop_columns: bool, expected_columns: list) -> None:
        """Both pandas and polars engines must drop/keep columns identically."""
        df = _DROP_SELECTED_DF_BUILDERS[engine]()
        params = {
            "drop_columns": drop_columns,
            "selected_columns": ["a"],
            "candidate_columns": ["a", "b"],
        }
        X_out, _ = _DROP_SELECTED_FUNCS[engine](df, None, params)
        assert set(X_out.columns) == set(expected_columns)


class TestDropSelectedPandas:
    def _df(self) -> pd.DataFrame:
        """Small DataFrame with three columns."""
        return pd.DataFrame({"a": [1, 2], "b": [3, 4], "c": [5, 6]})

    def test_y_returned_unchanged(self) -> None:
        """y is not modified by the column-drop helper."""
        y = pd.Series([0, 1])
        params = {
            "drop_columns": True,
            "selected_columns": ["a"],
            "candidate_columns": ["a", "b"],
        }
        _, y_out = _drop_selected_pandas(self._df(), y, params)
        pd.testing.assert_series_equal(y, y_out)


# ---------------------------------------------------------------------------
# _extract_target
# ---------------------------------------------------------------------------


class TestExtractTarget:
    def test_explicit_y_returned_directly(self) -> None:
        """When y is provided it takes priority over target_col."""
        df = pd.DataFrame({"a": [1, 2], "target": [0, 1]})
        y = pd.Series([9, 8])
        result = _extract_target(df, y, "target")
        assert result is not None
        pd.testing.assert_series_equal(result, y)

    def test_target_col_extracted_from_df(self) -> None:
        """When y is None, target_col must be pulled from the DataFrame."""
        df = pd.DataFrame({"a": [1, 2], "target": [0, 1]})
        result = _extract_target(df, None, "target")
        assert result is not None
        assert list(result) == [0, 1]

    def test_missing_target_col_returns_none(self) -> None:
        """A target_col that isn't in the DataFrame must return None."""
        df = pd.DataFrame({"a": [1, 2]})
        assert _extract_target(df, None, "missing") is None

    def test_no_y_no_target_returns_none(self) -> None:
        """Neither y nor a valid target_col → None."""
        df = pd.DataFrame({"a": [1, 2]})
        assert _extract_target(df, None, None) is None


# ---------------------------------------------------------------------------
# _prepare_sklearn_y
# ---------------------------------------------------------------------------


class TestPrepareSklearnY:
    def test_numeric_series_passed_through(self) -> None:
        """A numeric Series must be converted to a float numpy array as-is."""
        y = pd.Series([1.0, 2.0, 3.0])
        y_np = _prepare_sklearn_y(y, "regression")
        assert y_np.dtype in (np.float64, np.float32, float)
        np.testing.assert_array_equal(y_np, [1.0, 2.0, 3.0])

    def test_string_classification_target_factorized(self) -> None:
        """String labels for classification must be factorized to integers."""
        y = pd.Series(["cat", "dog", "cat", "bird"])
        y_np = _prepare_sklearn_y(y, "classification")
        assert np.issubdtype(y_np.dtype, np.integer)
        # Factorized output must have the same length.
        assert len(y_np) == 4

    def test_binary_int_target_passes_through(self) -> None:
        """Integer binary labels don't require factorization."""
        y = pd.Series([0, 1, 0, 1])
        y_np = _prepare_sklearn_y(y, "classification")
        assert len(y_np) == 4


# ---------------------------------------------------------------------------
# _resolve_problem_type
# ---------------------------------------------------------------------------


class TestResolveProblemType:
    @pytest.mark.parametrize(*_resolve_problem_type_cases)
    def test_resolve_problem_type(self, mode: str, y_values: list | None, expected: str) -> None:
        """Explicit modes and 'auto' inference must resolve to the right problem type."""
        y = pd.Series(y_values) if y_values is not None else None
        assert _resolve_problem_type(mode, y) == expected


# ---------------------------------------------------------------------------
# _resolve_generic_param
# ---------------------------------------------------------------------------


class TestResolveGenericParam:
    @pytest.mark.parametrize(*_resolve_generic_param_cases)
    def test_resolve_generic_param(self, config: dict, expected: float) -> None:
        """Explicit params, mode-based defaults, and unknown-mode fallback must all resolve."""
        assert _resolve_generic_param(config) == expected


# ---------------------------------------------------------------------------
# _build_univariate_selector
# ---------------------------------------------------------------------------


class TestBuildUnivariateSelector:
    @pytest.mark.parametrize(*_build_univariate_selector_cases)
    def test_build_univariate_selector(
        self,
        method: str,
        config: dict,
        expected_class_name: str | None,
        attr_name: str | None,
        attr_value: object,
    ) -> None:
        """Each method must build the right selector class (or None if unrecognised)."""
        sel = _build_univariate_selector(method, f_classif, config)
        if expected_class_name is None:
            assert sel is None
            return
        assert isinstance(sel, _UNIVARIATE_SELECTOR_CLASSES[expected_class_name])
        if attr_name is not None:
            assert getattr(sel, attr_name) == attr_value

        if attr_name is not None:
            assert getattr(sel, attr_name) == attr_value


# ---------------------------------------------------------------------------
# _build_model_selector
# ---------------------------------------------------------------------------


class TestBuildModelSelector:
    @pytest.mark.parametrize(*_build_model_selector_cases)
    def test_build_model_selector(
        self,
        method: str,
        estimator_name: str,
        config: dict,
        expected_class_name: str | None,
        attr_name: str | None,
        attr_value: object,
    ) -> None:
        """Each method must build the right selector class (or None if unrecognised)."""
        est = _ESTIMATOR_FACTORIES[estimator_name]()
        sel = _build_model_selector(method, est, config)
        if expected_class_name is None:
            assert sel is None
            return
        assert isinstance(sel, _MODEL_SELECTOR_CLASSES[expected_class_name])
        if attr_name is not None:
            assert getattr(sel, attr_name) == attr_value


# ---------------------------------------------------------------------------
# _maybe_chi2_rescale
# ---------------------------------------------------------------------------


class TestMaybeChi2Rescale:
    @pytest.mark.parametrize(*_maybe_chi2_rescale_cases)
    def test_maybe_chi2_rescale(
        self, x: list, score_func: str | None, expect_rescale: bool
    ) -> None:
        """chi2 with negative values must be rescaled; every other case is a passthrough."""
        X = np.array(x)
        result = _maybe_chi2_rescale(X, score_func)
        if expect_rescale:
            assert result.min() >= 0.0
            assert result.max() <= 1.0 + 1e-9
        else:
            np.testing.assert_array_equal(result, X)


# ---------------------------------------------------------------------------
# _fillna_zero_with_warning
# ---------------------------------------------------------------------------


class TestFillnaZeroWithWarning:
    def test_warns_when_missing_values_present(self, caplog: pytest.LogCaptureFixture) -> None:
        """Regression test: a silent fillna(0) before scoring can bias
        univariate/model-based feature scores when 0 is meaningful or
        missingness correlates with the target - must now warn."""
        import logging

        from skyulf.preprocessing.feature_selection._common import _fillna_zero_with_warning

        df = pd.DataFrame({"a": [1.0, None, 3.0], "b": [1.0, 2.0, 3.0]})
        with caplog.at_level(
            logging.WARNING, logger="skyulf.preprocessing.feature_selection._common"
        ):
            result = _fillna_zero_with_warning(df, ["a", "b"])
        assert list(result["a"]) == [1.0, 0.0, 3.0]
        assert any("filling missing values with 0" in rec.message for rec in caplog.records)
        assert any("'a'" in rec.message for rec in caplog.records)

    def test_no_warning_when_no_missing_values(self, caplog: pytest.LogCaptureFixture) -> None:
        """No missing values means fillna(0) is a no-op - must not warn."""
        import logging

        from skyulf.preprocessing.feature_selection._common import _fillna_zero_with_warning

        df = pd.DataFrame({"a": [1.0, 2.0, 3.0], "b": [1.0, 2.0, 3.0]})
        with caplog.at_level(
            logging.WARNING, logger="skyulf.preprocessing.feature_selection._common"
        ):
            result = _fillna_zero_with_warning(df, ["a", "b"])
        assert list(result["a"]) == [1.0, 2.0, 3.0]
        assert not any("filling missing values with 0" in rec.message for rec in caplog.records)


# ---------------------------------------------------------------------------
# _univariate_score_dicts
# ---------------------------------------------------------------------------


class TestUnivariateScoreDicts:
    def _fitted_selector(self) -> tuple:
        """Return a fitted SelectKBest and the columns used."""
        X = np.array([[1.0, 0.0, 0.0], [2.0, 0.0, 1.0], [3.0, 0.0, 0.0], [4.0, 0.0, 1.0]])
        y = np.array([0, 1, 0, 1])
        sel = SelectKBest(f_classif, k=2).fit(X, y)
        return sel, ["feat_a", "feat_b", "feat_c"]

    def test_scores_dict_has_all_columns(self) -> None:
        """scores dict must have one key per candidate column."""
        sel, cols = self._fitted_selector()
        scores, _ = _univariate_score_dicts(sel, cols)
        assert set(scores.keys()) == set(cols)

    def test_pvalues_dict_has_all_columns(self) -> None:
        """pvalues dict must have one key per candidate column."""
        sel, cols = self._fitted_selector()
        _, pvalues = _univariate_score_dicts(sel, cols)
        assert set(pvalues.keys()) == set(cols)

    def test_nan_scores_replaced_by_zero(self) -> None:
        """NaN scores must be normalised to 0.0 to avoid downstream errors."""
        sel, cols = self._fitted_selector()
        # Inject a NaN manually.
        sel.scores_[1] = float("nan")
        scores, _ = _univariate_score_dicts(sel, cols)
        assert scores[cols[1]] == 0.0

    def test_no_scores_attr_returns_empty_dicts(self) -> None:
        """An object without scores_ / pvalues_ must produce two empty dicts."""

        class FakeSel:
            pass

        scores, pvalues = _univariate_score_dicts(FakeSel(), ["a", "b"])
        assert scores == {}
        assert pvalues == {}


# ---------------------------------------------------------------------------
# _univariate_no_target_artifact
# ---------------------------------------------------------------------------


class TestUnivariateNoTargetArtifact:
    def test_returns_passthrough_artifact(self) -> None:
        """No-target artifact must select all candidate columns as-is."""
        cols = ["a", "b", "c"]
        artifact = _univariate_no_target_artifact(cols, "select_k_best", {})
        assert artifact["type"] == "univariate_selection"
        assert artifact["selected_columns"] == cols
        assert artifact["candidate_columns"] == cols
        assert artifact["scores"] == {}
        assert artifact["pvalues"] == {}

    def test_drop_columns_inherited_from_config(self) -> None:
        """The artifact must inherit drop_columns from the config."""
        artifact = _univariate_no_target_artifact(["a"], "select_k_best", {"drop_columns": False})
        assert artifact["drop_columns"] is False


# ---------------------------------------------------------------------------
# _model_feature_importances
# ---------------------------------------------------------------------------


class TestModelFeatureImportances:
    @pytest.mark.parametrize(*_model_feature_importances_cases)
    def test_model_feature_importances(
        self, estimator_name: str, columns: list, check_non_negative: bool
    ) -> None:
        """Both tree-based and coefficient-based estimators must yield a per-column dict."""
        X = np.array([[1.0, 0.0], [2.0, 1.0], [3.0, 0.0], [4.0, 1.0]])
        y = np.array([0, 1, 0, 1])
        est = _IMPORTANCE_ESTIMATOR_FACTORIES[estimator_name]()
        est.fit(X, y)
        # Wrap in a minimal SelectFromModel so estimator_ attribute is set.
        sfm = SelectFromModel(est, threshold="mean").fit(X, y)
        imps = _model_feature_importances(sfm, columns)
        assert set(imps.keys()) == set(columns)
        if check_non_negative:
            assert all(v >= 0 for v in imps.values())

    def test_no_estimator_attr_returns_empty(self) -> None:
        """An object with no estimator_ must return an empty dict."""

        class FakeSel:
            pass

        assert _model_feature_importances(FakeSel(), ["a", "b"]) == {}


# ---------------------------------------------------------------------------
# _resolve_candidate_columns
# ---------------------------------------------------------------------------


class TestResolveCandidateColumns:
    def test_target_col_excluded(self) -> None:
        """The target column must never appear in the candidate list."""
        df = pd.DataFrame({"a": [1.0, 2.0, 3.0], "b": [4.0, 5.0, 6.0], "target": [0, 1, 0]})
        cols = _resolve_candidate_columns(df, {}, "target")
        assert "target" not in cols
        assert "a" in cols

    def test_explicit_columns_config_respected(self) -> None:
        """When 'columns' is specified in config, only those are considered."""
        df = pd.DataFrame({"a": [1.0, 2.0], "b": [3.0, 4.0], "c": [5.0, 6.0]})
        cols = _resolve_candidate_columns(df, {"columns": ["a", "b"]}, None)
        assert set(cols) == {"a", "b"}
        assert "c" not in cols


# ---------------------------------------------------------------------------
# Real-shaped dataset: mixed-dtype customers.csv
# ---------------------------------------------------------------------------


class TestRealShapedDataset:
    """Integration check against customers.csv — verifies that _infer_problem_type
    correctly classifies the binary ``churned`` target and that
    _resolve_candidate_columns correctly excludes non-numeric and target columns
    from a mixed-dtype frame that includes NaN in age/income/lat/lon.
    """

    def test_infer_problem_type_on_binary_churned_column(self) -> None:
        """churned has only 2 unique integer values → must infer as classification."""
        df = load_sample_dataset("customers")
        assert _infer_problem_type(df["churned"]) == "classification"

    def test_resolve_candidate_columns_excludes_target_and_non_numeric(self) -> None:
        """Numeric columns (age, income, lat, lon, customer_id) minus churned
        are candidates; string columns must not appear even with NaN present."""
        df = load_sample_dataset("customers")
        cols = _resolve_candidate_columns(df, {}, "churned")
        # Target must be excluded.
        assert "churned" not in cols
        # Non-numeric columns must not appear.
        assert "city" not in cols
        assert "plan_type" not in cols
        assert "signup_date" not in cols
        # Numeric columns with NaN must still qualify as candidates.
        assert "age" in cols
        assert "income" in cols
