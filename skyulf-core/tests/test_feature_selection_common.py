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
)
from sklearn.linear_model import LinearRegression, LogisticRegression
from tests.utils.dataset_loader import load_sample_dataset

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

# ---------------------------------------------------------------------------
# _infer_problem_type
# ---------------------------------------------------------------------------


class TestInferProblemType:
    def test_bool_column_is_classification(self) -> None:
        """Boolean dtype must infer as classification."""
        s = pd.Series([True, False, True])
        assert _infer_problem_type(s) == "classification"

    def test_object_column_is_classification(self) -> None:
        """String/object dtype is always classification."""
        s = pd.Series(["cat", "dog", "cat"])
        assert _infer_problem_type(s) == "classification"

    def test_few_unique_numeric_is_classification(self) -> None:
        """Ten or fewer unique values → classification even for numeric."""
        s = pd.Series([1, 2, 3, 1, 2])
        assert _infer_problem_type(s) == "classification"

    def test_many_unique_numeric_is_regression(self) -> None:
        """More than ten unique float values → regression."""
        s = pd.Series([float(i) for i in range(20)])
        assert _infer_problem_type(s) == "regression"

    def test_empty_series_defaults_to_classification(self) -> None:
        """An empty Series should default to classification to be safe."""
        assert _infer_problem_type(pd.Series([], dtype=float)) == "classification"


# ---------------------------------------------------------------------------
# _resolve_score_function
# ---------------------------------------------------------------------------


class TestResolveScoreFunction:
    def test_named_function_returned(self) -> None:
        """An explicitly named score function must be returned directly."""
        assert _resolve_score_function("f_classif", "classification") is f_classif

    def test_classification_default_is_f_classif(self) -> None:
        """Unknown name + classification problem → f_classif fallback."""
        assert _resolve_score_function(None, "classification") is f_classif

    def test_regression_default_is_f_regression(self) -> None:
        """Unknown name + regression problem → f_regression fallback."""
        assert _resolve_score_function(None, "regression") is f_regression

    def test_named_mutual_info_regression(self) -> None:
        """mutual_info_regression must be returned when explicitly named."""
        from sklearn.feature_selection import mutual_info_regression

        assert (
            _resolve_score_function("mutual_info_regression", "regression")
            is mutual_info_regression
        )


# ---------------------------------------------------------------------------
# _resolve_estimator
# ---------------------------------------------------------------------------


class TestResolveEstimator:
    def test_auto_classification_gives_logistic(self) -> None:
        """'auto' for classification should produce a LogisticRegression."""
        est = _resolve_estimator("auto", "classification")
        assert isinstance(est, LogisticRegression)

    def test_auto_regression_gives_linear(self) -> None:
        """'auto' for regression should produce a LinearRegression."""
        est = _resolve_estimator("auto", "regression")
        assert isinstance(est, LinearRegression)

    def test_random_forest_classification(self) -> None:
        """'random_forest' for classification must give a RandomForestClassifier."""
        est = _resolve_estimator("random_forest", "classification")
        assert isinstance(est, RandomForestClassifier)

    def test_random_forest_regression(self) -> None:
        """'random_forest' for regression must give a RandomForestRegressor."""
        est = _resolve_estimator("random_forest", "regression")
        assert isinstance(est, RandomForestRegressor)

    def test_unknown_key_returns_none(self) -> None:
        """A completely unknown key should return None, not raise."""
        assert _resolve_estimator("quantum_forest", "classification") is None

    def test_case_insensitive(self) -> None:
        """Estimator key resolution must be case-insensitive."""
        est = _resolve_estimator("LogisticRegression", "classification")
        assert isinstance(est, LogisticRegression)


# ---------------------------------------------------------------------------
# _resolve_drop_list
# ---------------------------------------------------------------------------


class TestResolveDropList:
    def test_drops_candidates_minus_selected(self) -> None:
        """Columns in candidates but not in selected must be in the drop list."""
        params = {"selected_columns": ["a"], "candidate_columns": ["a", "b", "c"]}
        drop = _resolve_drop_list(params, ["a", "b", "c"])
        assert set(drop) == {"b", "c"}

    def test_no_selected_returns_empty(self) -> None:
        """Missing selected_columns key → nothing to drop."""
        params = {"candidate_columns": ["a", "b"]}
        assert _resolve_drop_list(params, ["a", "b"]) == []

    def test_column_not_in_existing_excluded(self) -> None:
        """Drop list is filtered by existing_cols; phantom columns are excluded."""
        params = {"selected_columns": [], "candidate_columns": ["a", "b"]}
        drop = _resolve_drop_list(params, ["a"])
        # b is not in existing_cols, so only a is dropped.
        assert drop == ["a"]

    def test_empty_candidates_empty_drop(self) -> None:
        """No candidate columns → nothing to drop."""
        params = {"selected_columns": [], "candidate_columns": []}
        assert _resolve_drop_list(params, ["a", "b"]) == []


# ---------------------------------------------------------------------------
# _drop_selected_pandas / _drop_selected_polars
# ---------------------------------------------------------------------------


class TestDropSelectedPandas:
    def _df(self) -> pd.DataFrame:
        """Small DataFrame with three columns."""
        return pd.DataFrame({"a": [1, 2], "b": [3, 4], "c": [5, 6]})

    def test_drops_unselected_columns(self) -> None:
        """Columns in candidates but not selected must be removed."""
        params = {
            "drop_columns": True,
            "selected_columns": ["a"],
            "candidate_columns": ["a", "b"],
        }
        X_out, _ = _drop_selected_pandas(self._df(), None, params)
        assert "b" not in X_out.columns
        assert "a" in X_out.columns
        # c was not a candidate so it survives.
        assert "c" in X_out.columns

    def test_drop_columns_false_is_passthrough(self) -> None:
        """drop_columns=False must return the frame unchanged."""
        params = {
            "drop_columns": False,
            "selected_columns": ["a"],
            "candidate_columns": ["a", "b"],
        }
        X_out, _ = _drop_selected_pandas(self._df(), None, params)
        assert list(X_out.columns) == ["a", "b", "c"]

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


class TestDropSelectedPolars:
    def _df(self) -> pl.DataFrame:
        """Small Polars DataFrame with three columns."""
        return pl.DataFrame({"a": [1, 2], "b": [3, 4], "c": [5, 6]})

    def test_drops_unselected_columns_polars(self) -> None:
        """Polars path must drop the same columns as the pandas path."""
        params = {
            "drop_columns": True,
            "selected_columns": ["a"],
            "candidate_columns": ["a", "b"],
        }
        X_out, _ = _drop_selected_polars(self._df(), None, params)
        assert "b" not in X_out.columns
        assert "a" in X_out.columns

    def test_drop_columns_false_passthrough_polars(self) -> None:
        """drop_columns=False must leave the Polars frame intact."""
        params = {
            "drop_columns": False,
            "selected_columns": ["a"],
            "candidate_columns": ["a", "b"],
        }
        X_out, _ = _drop_selected_polars(self._df(), None, params)
        assert list(X_out.columns) == ["a", "b", "c"]


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
    def test_explicit_classification_returned(self) -> None:
        """An explicit 'classification' declaration must be returned unchanged."""
        assert _resolve_problem_type("classification", None) == "classification"

    def test_explicit_regression_returned(self) -> None:
        """An explicit 'regression' declaration must be returned unchanged."""
        assert _resolve_problem_type("regression", pd.Series([1, 2])) == "regression"

    def test_auto_with_none_y_defaults_classification(self) -> None:
        """'auto' with no y available falls back to classification."""
        assert _resolve_problem_type("auto", None) == "classification"

    def test_auto_infers_regression_from_y(self) -> None:
        """'auto' with a continuous y must resolve to regression."""
        y = pd.Series([float(i) for i in range(20)])
        assert _resolve_problem_type("auto", y) == "regression"


# ---------------------------------------------------------------------------
# _resolve_generic_param
# ---------------------------------------------------------------------------


class TestResolveGenericParam:
    def test_explicit_param_returned(self) -> None:
        """An explicit 'param' key in config must be returned directly."""
        assert _resolve_generic_param({"param": 42}) == 42

    def test_k_best_mode_uses_k(self) -> None:
        """k_best mode reads the 'k' key (default 10)."""
        assert _resolve_generic_param({"mode": "k_best", "k": 5}) == 5

    def test_k_best_default(self) -> None:
        """k_best without explicit k defaults to 10."""
        assert _resolve_generic_param({"mode": "k_best"}) == 10

    def test_percentile_mode_uses_percentile(self) -> None:
        """percentile mode reads the 'percentile' key."""
        assert _resolve_generic_param({"mode": "percentile", "percentile": 25}) == 25

    def test_unknown_mode_defaults_to_alpha(self) -> None:
        """An unknown mode falls through to alpha (default 0.05)."""
        val = _resolve_generic_param({"mode": "fpr"})
        assert val == 0.05


# ---------------------------------------------------------------------------
# _build_univariate_selector
# ---------------------------------------------------------------------------


class TestBuildUnivariateSelector:
    def test_select_k_best(self) -> None:
        """select_k_best must produce a SelectKBest instance with the right k."""
        sel = _build_univariate_selector("select_k_best", f_classif, {"k": 3})
        assert isinstance(sel, SelectKBest)
        assert sel.k == 3

    def test_select_percentile(self) -> None:
        """select_percentile must produce a SelectPercentile with the right percentile."""
        sel = _build_univariate_selector("select_percentile", f_classif, {"percentile": 50})
        assert isinstance(sel, SelectPercentile)
        assert sel.percentile == 50

    def test_select_fpr(self) -> None:
        """select_fpr must produce a SelectFpr."""
        sel = _build_univariate_selector("select_fpr", f_classif, {"alpha": 0.05})
        assert isinstance(sel, SelectFpr)

    def test_select_fdr(self) -> None:
        """select_fdr must produce a SelectFdr."""
        sel = _build_univariate_selector("select_fdr", f_classif, {})
        assert isinstance(sel, SelectFdr)

    def test_select_fwe(self) -> None:
        """select_fwe must produce a SelectFwe."""
        sel = _build_univariate_selector("select_fwe", f_classif, {})
        assert isinstance(sel, SelectFwe)

    def test_unknown_method_returns_none(self) -> None:
        """An unrecognised method name must return None."""
        assert _build_univariate_selector("select_magic", f_classif, {}) is None


# ---------------------------------------------------------------------------
# _build_model_selector
# ---------------------------------------------------------------------------


class TestBuildModelSelector:
    def test_select_from_model(self) -> None:
        """select_from_model must produce a SelectFromModel."""
        est = LogisticRegression()
        sel = _build_model_selector("select_from_model", est, {})
        assert isinstance(sel, SelectFromModel)

    def test_select_from_model_float_threshold(self) -> None:
        """A numeric string threshold must be coerced to float."""
        est = LinearRegression()
        sel = _build_model_selector("select_from_model", est, {"threshold": "0.5"})
        assert isinstance(sel, SelectFromModel)
        assert sel.threshold == 0.5

    def test_select_from_model_string_threshold_kept(self) -> None:
        """A non-numeric string threshold such as 'mean' must stay as a string."""
        est = LinearRegression()
        sel = _build_model_selector("select_from_model", est, {"threshold": "mean"})
        assert sel is not None
        assert sel.threshold == "mean"

    def test_rfe(self) -> None:
        """rfe must produce an RFE instance."""
        est = LogisticRegression()
        sel = _build_model_selector("rfe", est, {"n_features_to_select": 2, "step": 1})
        assert isinstance(sel, RFE)
        assert sel.n_features_to_select == 2

    def test_unknown_method_returns_none(self) -> None:
        """An unrecognised model selector method must return None."""
        assert _build_model_selector("telepathy", LinearRegression(), {}) is None


# ---------------------------------------------------------------------------
# _maybe_chi2_rescale
# ---------------------------------------------------------------------------


class TestMaybeChi2Rescale:
    def test_no_rescale_for_non_chi2(self) -> None:
        """Non-chi2 score functions must leave the array unchanged."""
        X = np.array([[-1.0, 2.0], [3.0, -4.0]])
        result = _maybe_chi2_rescale(X, "f_classif")
        np.testing.assert_array_equal(result, X)

    def test_no_rescale_when_all_non_negative(self) -> None:
        """chi2 with no negative values must not trigger rescaling."""
        X = np.array([[0.0, 1.0], [2.0, 3.0]])
        result = _maybe_chi2_rescale(X, "chi2")
        np.testing.assert_array_equal(result, X)

    def test_rescales_when_chi2_and_negatives(self) -> None:
        """chi2 with negative values must be MinMax-scaled to [0, 1]."""
        X = np.array([[-1.0, 2.0], [3.0, 4.0]])
        result = _maybe_chi2_rescale(X, "chi2")
        assert result.min() >= 0.0
        assert result.max() <= 1.0 + 1e-9

    def test_rescale_none_score_func(self) -> None:
        """None score_func (not chi2) must leave the array unchanged."""
        X = np.array([[-1.0, 2.0]])
        result = _maybe_chi2_rescale(X, None)
        np.testing.assert_array_equal(result, X)


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
    def test_tree_based_importances(self) -> None:
        """A fitted forest's feature_importances_ must be returned as a dict."""
        X = np.array([[1.0, 0.0], [2.0, 1.0], [3.0, 0.0], [4.0, 1.0]])
        y = np.array([0, 1, 0, 1])
        rf = RandomForestClassifier(n_estimators=5, random_state=0)
        rf.fit(X, y)

        # Wrap in a minimal SelectFromModel so estimator_ attribute is set.
        sfm = SelectFromModel(rf, threshold="mean").fit(X, y)
        imps = _model_feature_importances(sfm, ["feat_a", "feat_b"])
        assert set(imps.keys()) == {"feat_a", "feat_b"}
        assert all(v >= 0 for v in imps.values())

    def test_coef_based_importances(self) -> None:
        """A fitted linear model's |coef| must be returned as a dict."""
        X = np.array([[1.0, 0.0], [2.0, 1.0], [3.0, 0.0], [4.0, 1.0]])
        y = np.array([0, 1, 0, 1])
        lr = LogisticRegression(max_iter=1000)
        lr.fit(X, y)
        sfm = SelectFromModel(lr, threshold="mean").fit(X, y)
        imps = _model_feature_importances(sfm, ["a", "b"])
        assert set(imps.keys()) == {"a", "b"}

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
