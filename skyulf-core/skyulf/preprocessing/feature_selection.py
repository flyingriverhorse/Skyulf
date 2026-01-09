import logging
from typing import Any, Callable, Dict, Optional, Tuple, Union

import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor
from sklearn.feature_selection import (
    RFE,
    GenericUnivariateSelect,
    SelectFdr,
    SelectFpr,
    SelectFromModel,
    SelectFwe,
    SelectKBest,
    SelectPercentile,
    VarianceThreshold,
    chi2,
    f_classif,
    f_regression,
    mutual_info_classif,
    mutual_info_regression,
    r_regression,
)
from sklearn.linear_model import LinearRegression, LogisticRegression

from ..utils import (
    detect_numeric_columns,
    pack_pipeline_output,
    resolve_columns,
    unpack_pipeline_input,
)
from .base import BaseApplier, BaseCalculator
from ..registry import NodeRegistry
from ..core.meta.decorators import node_meta
from ..engines import SkyulfDataFrame, get_engine
from ..engines.sklearn_bridge import SklearnBridge

logger = logging.getLogger(__name__)

# --- Helpers ---
SCORE_FUNCTIONS: Dict[str, Callable] = {
    "f_classif": f_classif,
    "f_regression": f_regression,
    "mutual_info_classif": mutual_info_classif,
    "mutual_info_regression": mutual_info_regression,
    "chi2": chi2,
    "r_regression": r_regression,
}


def _infer_problem_type(series: pd.Series) -> str:
    if series.empty:
        return "classification"
    if pd.api.types.is_bool_dtype(series) or pd.api.types.is_object_dtype(series):
        return "classification"
    unique_values = series.dropna().unique()
    if len(unique_values) <= 10:
        return "classification"
    return "regression"


def _resolve_score_function(name: Optional[str], problem_type: str) -> Any:
    if name and name in SCORE_FUNCTIONS:
        return SCORE_FUNCTIONS[name]

    if problem_type == "classification":
        return f_classif
    else:
        return f_regression


def _resolve_estimator(key: Optional[str], problem_type: str) -> Any:
    key = (key or "auto").lower()
    if problem_type == "classification":
        if key in {"auto", "logistic_regression", "logisticregression"}:
            return LogisticRegression(max_iter=1000)
        if key in {"random_forest", "randomforest"}:
            return RandomForestClassifier(n_estimators=100, random_state=0, n_jobs=-1)
        if key in {"linear_regression", "linearregression"}:
            return LinearRegression()  # Odd for classification but allowed in V1 logic
    else:
        if key in {"auto", "linear_regression", "linearregression"}:
            return LinearRegression()
        if key in {"random_forest", "randomforest"}:
            return RandomForestRegressor(n_estimators=100, random_state=0, n_jobs=-1)
    return None


# --- Variance Threshold ---


class VarianceThresholdApplier(BaseApplier):
    def apply(
        self,
        df: Union[pd.DataFrame, SkyulfDataFrame, Tuple[Any, ...], Any],
        params: Dict[str, Any],
    ) -> Union[pd.DataFrame, SkyulfDataFrame, Tuple[Any, ...]]:
        X, y, is_tuple = unpack_pipeline_input(df)
        engine = get_engine(X)

        selected_cols = params.get("selected_columns")
        candidate_columns = params.get("candidate_columns", [])
        drop_columns = params.get("drop_columns", True)

        if selected_cols is None:
            return pack_pipeline_output(X, y, is_tuple)

        cols_to_drop_set = set(candidate_columns) - set(selected_cols)
        cols_to_drop_list = [c for c in cols_to_drop_set if c in X.columns]

        if cols_to_drop_list and drop_columns:
            if engine.name == "polars":
                import polars as pl
                X_pl: Any = X
                X = X_pl.drop(cols_to_drop_list)
            else:
                X = X.drop(columns=cols_to_drop_list)
        return pack_pipeline_output(X, y, is_tuple)


@NodeRegistry.register("VarianceThreshold", VarianceThresholdApplier)
@node_meta(
    id="VarianceThreshold",
    name="Variance Threshold",
    category="Feature Selection",
    description="Remove features with low variance.",
    params={"threshold": 0.0}
)
class VarianceThresholdCalculator(BaseCalculator):
    def fit(
        self,
        df: Union[pd.DataFrame, SkyulfDataFrame, Tuple[Any, ...], Any],
        config: Dict[str, Any],
    ) -> Dict[str, Any]:
        X, _, _ = unpack_pipeline_input(df)
        engine = get_engine(X)

        # Config: {"threshold": 0.0, "columns": [...]}
        threshold = config.get("threshold", 0.0)
        drop_columns = config.get("drop_columns", True)

        cols = resolve_columns(
            X,
            config,
            lambda d: detect_numeric_columns(d, exclude_binary=False, exclude_constant=False),
        )

        if not cols:
            return {}

        selector = VarianceThreshold(threshold=threshold)
        
        # Use Bridge for fitting
        if engine.name == "polars":
            X_pl: Any = X
            X_subset = X_pl.select(cols)
        else:
            X_subset = X[cols]

        X_np, _ = SklearnBridge.to_sklearn(X_subset)
        
        selector.fit(X_np)

        support = selector.get_support()
        selected_cols = [c for c, s in zip(cols, support) if s]

        variances = {}
        if hasattr(selector, "variances_"):
            variances = dict(zip(cols, selector.variances_.tolist()))

        return {
            "type": "variance_threshold",
            "selected_columns": selected_cols,
            "candidate_columns": cols,
            "threshold": threshold,
            "drop_columns": drop_columns,
            "variances": variances,
        }


# --- Correlation Threshold ---


class CorrelationThresholdApplier(BaseApplier):
    def apply(
        self,
        df: Union[pd.DataFrame, SkyulfDataFrame, Tuple[Any, ...], Any],
        params: Dict[str, Any],
    ) -> Union[pd.DataFrame, SkyulfDataFrame, Tuple[Any, ...]]:
        X, y, is_tuple = unpack_pipeline_input(df)
        engine = get_engine(X)

        cols_to_drop = params.get("columns_to_drop", [])
        drop_columns = params.get("drop_columns", True)

        cols_to_drop = [c for c in cols_to_drop if c in X.columns]
        if not cols_to_drop:
            return pack_pipeline_output(X, y, is_tuple)

        if drop_columns:
            if engine.name == "polars":
                import polars as pl
                X_pl: Any = X
                X = X_pl.drop(cols_to_drop)
            else:
                X = X.drop(columns=cols_to_drop)
        return pack_pipeline_output(X, y, is_tuple)


@NodeRegistry.register("CorrelationThreshold", CorrelationThresholdApplier)
@node_meta(
    id="CorrelationThreshold",
    name="Correlation Threshold",
    category="Feature Selection",
    description="Remove features highly correlated with others.",
    params={"threshold": 0.95, "method": "pearson"}
)
class CorrelationThresholdCalculator(BaseCalculator):
    def fit(
        self,
        df: Union[pd.DataFrame, SkyulfDataFrame, Tuple[Any, ...], Any],
        config: Dict[str, Any],
    ) -> Dict[str, Any]:
        X, _, _ = unpack_pipeline_input(df)
        engine = get_engine(X)
        
        # Ensure pandas for correlation logic
        if engine.name == "polars":
            # X comes in as Any (SkyulfDataFrame), cast to pandas
            X = X.to_pandas()

        # Config: {"threshold": 0.95, "correlation_method": "pearson"}
        threshold = config.get("threshold", 0.95)
        drop_columns = config.get("drop_columns", True)
        # Prefer "correlation_method" to avoid conflict with facade's "method"
        # FIX: Do NOT fallback to config.get("method") because it might be
        # "correlation_threshold" (the facade method name)
        method = config.get("correlation_method", "pearson")

        cols = resolve_columns(X, config, detect_numeric_columns)

        if len(cols) < 2:
            return {}

        corr_matrix = X[cols].corr(method=method).abs()
        upper = corr_matrix.where(np.triu(np.ones(corr_matrix.shape), k=1).astype(bool))
        to_drop = [column for column in upper.columns if any(upper[column] > threshold)]

        return {
            "type": "correlation_threshold",
            "columns_to_drop": to_drop,
            "threshold": threshold,
            "method": method,
            "drop_columns": drop_columns,
        }


# --- Univariate Selection ---


class UnivariateSelectionApplier(BaseApplier):
    def apply(
        self,
        df: Union[pd.DataFrame, SkyulfDataFrame, Tuple[Any, ...], Any],
        params: Dict[str, Any],
    ) -> Union[pd.DataFrame, SkyulfDataFrame, Tuple[Any, ...]]:
        X, y, is_tuple = unpack_pipeline_input(df)
        engine = get_engine(X)

        selected_cols = params.get("selected_columns")
        candidate_columns = params.get("candidate_columns", [])
        drop_columns = params.get("drop_columns", True)

        if selected_cols is None:
            return pack_pipeline_output(X, y, is_tuple)

        cols_to_drop_set = set(candidate_columns) - set(selected_cols)
        cols_to_drop_list = [c for c in cols_to_drop_set if c in X.columns]
        if cols_to_drop_list and drop_columns:
            if engine.name == "polars":
                import polars as pl
                X_pl: Any = X
                X = X_pl.drop(cols_to_drop_list)
            else:
                X = X.drop(columns=cols_to_drop_list)
        return pack_pipeline_output(X, y, is_tuple)


@NodeRegistry.register("UnivariateSelection", UnivariateSelectionApplier)
@node_meta(
    id="UnivariateSelection",
    name="Univariate Selection",
    category="Feature Selection",
    description="Select best features based on univariate statistical tests.",
    params={"method": "SelectKBest", "score_func": "f_classif", "k": 10}
)
class UnivariateSelectionCalculator(BaseCalculator):
    def fit(  # noqa: C901
        self,
        df: Union[pd.DataFrame, SkyulfDataFrame, Tuple[Any, ...], Any],
        config: Dict[str, Any],
    ) -> Dict[str, Any]:
        # Config: method, k, percentile, alpha, score_func, target_column
        target_col = config.get("target_column")

        X, y, is_tuple = unpack_pipeline_input(df)
        engine = get_engine(X)

        if not is_tuple:
            if not target_col or target_col not in X.columns:
                logger.error(
                    f"UnivariateSelection requires target column '{target_col}' to be present in training data."
                )
                return {}
            # Handle y extraction safely for Polars
            if engine.name == "polars":
                import polars as pl
                y = X.select(target_col).to_series().to_pandas()
            else:
                y = X[target_col]

        cols = resolve_columns(
            X, config, lambda d: detect_numeric_columns(d, exclude_binary=False)
        )

        # Ensure target is not in candidate columns
        if target_col in cols:
            cols = [c for c in cols if c != target_col]

        if not cols:
            return {}

        method = config.get("method", "select_k_best")
        score_func_name = config.get("score_func")
        problem_type = config.get("problem_type", "auto")

        if problem_type == "auto":
            if y is None:
                problem_type = "classification"  # Default fallback
            else:
                problem_type = _infer_problem_type(y)

        score_func = _resolve_score_function(score_func_name, problem_type)

        selector = None
        if method == "select_k_best":
            k = config.get("k", 10)
            # logger.info(f"SelectKBest k={k}")
            selector = SelectKBest(score_func=score_func, k=k)
        elif method == "select_percentile":
            p = config.get("percentile", 10)
            selector = SelectPercentile(score_func=score_func, percentile=p)
        elif method == "select_fpr":
            alpha = config.get("alpha", 0.05)
            selector = SelectFpr(score_func=score_func, alpha=alpha)
        elif method == "select_fdr":
            alpha = config.get("alpha", 0.05)
            selector = SelectFdr(score_func=score_func, alpha=alpha)
        elif method == "select_fwe":
            alpha = config.get("alpha", 0.05)
            selector = SelectFwe(score_func=score_func, alpha=alpha)
        elif method == "generic_univariate_select":
            mode = config.get("mode", "k_best")
            # Prioritize explicit 'param' from config (Frontend sends this)
            if "param" in config:
                param = config.get("param")
            else:
                # Fallback to mapping from specific keys (Legacy/Alternative)
                if mode == "k_best":
                    param = config.get("k", 10)
                elif mode == "percentile":
                    param = config.get("percentile", 10)
                else:
                    param = config.get("alpha", 0.05)

            selector = GenericUnivariateSelect(
                score_func=score_func, mode=mode, param=param
            )

        if not selector:
            return {}

        # Use Bridge for fitting
        if engine.name == "polars":
             # Cast X for safety
             X_pl: Any = X
             X_subset = X_pl.select(cols).fill_null(0)
        else:
             X_subset = X[cols].fillna(0)
             
        X_np, _ = SklearnBridge.to_sklearn(X_subset)

        # Handle Chi2 negative values
        if score_func_name == "chi2" and (X_np < 0).any():
            logger.warning(
                "Chi-squared statistic requires non-negative feature values. "
                "Applying MinMaxScaler to features for selection."
            )
            from sklearn.preprocessing import MinMaxScaler
            X_np = MinMaxScaler().fit_transform(X_np)

        # Handle classification target encoding if needed
        y_fit = y
        
        # Convert y to numpy/pandas for inspection
        if hasattr(y, "to_numpy"):
             y_np = y.to_numpy()
        elif hasattr(y, "to_pandas"):
             y_np = y.to_pandas().to_numpy()
        else:
             y_np = np.array(y)
             
        if (
            problem_type == "classification"
            and y is not None
            # Check if numeric
            and not np.issubdtype(y_np.dtype, np.number)
        ):
            y_factorized, _ = pd.factorize(y_np)
            y_fit = y_factorized
        else:
            y_fit = y_np

        if y is not None:
            selector.fit(X_np, y_fit)
            support = selector.get_support()
            selected_cols = [c for c, s in zip(cols, support) if s]
        else:
            selected_cols = cols  # Fallback if no target
            scores: Dict[str, float] = {}
            pvalues: Dict[str, float] = {}
            return {
                "type": "univariate_selection",
                "selected_columns": selected_cols,
                "candidate_columns": cols,
                "method": method,
                "drop_columns": config.get("drop_columns", True),
                "scores": scores,
                "pvalues": pvalues,
            }

        scores = {}
        pvalues = {}
        if hasattr(selector, "scores_"):
            # Handle potential NaN/Inf in scores
            safe_scores = np.nan_to_num(
                selector.scores_, nan=0.0, posinf=0.0, neginf=0.0
            )
            scores = dict(zip(cols, safe_scores.tolist()))

        if hasattr(selector, "pvalues_"):
            # Handle potential NaN in pvalues
            safe_pvalues = np.nan_to_num(selector.pvalues_, nan=1.0)
            pvalues = dict(zip(cols, safe_pvalues.tolist()))

        return {
            "type": "univariate_selection",
            "selected_columns": selected_cols,
            "candidate_columns": cols,
            "method": method,
            "drop_columns": config.get("drop_columns", True),
            "feature_scores": scores,
            "p_values": pvalues,
        }


# --- Model Based Selection ---


class ModelBasedSelectionApplier(BaseApplier):
    def apply(
        self,
        df: Union[pd.DataFrame, SkyulfDataFrame, Tuple[Any, ...], Any],
        params: Dict[str, Any],
    ) -> Union[pd.DataFrame, SkyulfDataFrame, Tuple[Any, ...]]:
        X, y, is_tuple = unpack_pipeline_input(df)
        engine = get_engine(X)

        selected_cols = params.get("selected_columns")
        candidate_columns = params.get("candidate_columns", [])
        drop_columns = params.get("drop_columns", True)

        if selected_cols is None:
            return pack_pipeline_output(X, y, is_tuple)

        cols_to_drop_set = set(candidate_columns) - set(selected_cols)
        cols_to_drop_list = [c for c in cols_to_drop_set if c in X.columns]
        if cols_to_drop_list and drop_columns:
            if engine.name == "polars":
                import polars as pl
                X_pl: Any = X
                X = X_pl.drop(cols_to_drop_list)
            else:
                X = X.drop(columns=cols_to_drop_list)
        return pack_pipeline_output(X, y, is_tuple)


@NodeRegistry.register("ModelBasedSelection", ModelBasedSelectionApplier)
@node_meta(
    id="ModelBasedSelection",
    name="Model-Based Selection",
    category="Feature Selection",
    description="Select features based on importance weights.",
    params={"estimator": "RandomForest", "threshold": "mean", "max_features": None}
)
class ModelBasedSelectionCalculator(BaseCalculator):
    def fit(  # noqa: C901
        self,
        df: Union[pd.DataFrame, SkyulfDataFrame, Tuple[Any, ...], Any],
        config: Dict[str, Any],
    ) -> Dict[str, Any]:
        # Config: method (select_from_model, rfe), estimator, target_column
        target_col = config.get("target_column")

        X, y, is_tuple = unpack_pipeline_input(df)
        engine = get_engine(X)

        if not is_tuple:
            if not target_col or target_col not in X.columns:
                logger.error(
                    f"ModelBasedSelection requires target column '{target_col}' to be present in training data."
                )
                return {}
            # Handle y extraction safely for Polars
            if engine.name == "polars":
                import polars as pl
                y = X.select(target_col).to_series().to_pandas()
            else:
                y = X[target_col]

        cols = resolve_columns(
            X, config, lambda d: detect_numeric_columns(d, exclude_binary=False)
        )

        # Ensure target is not in candidate columns
        if target_col in cols:
            cols = [c for c in cols if c != target_col]

        if not cols:
            return {}

        method = config.get("method", "select_from_model")
        estimator_name = config.get("estimator", "auto")
        problem_type = config.get("problem_type", "auto")

        if problem_type == "auto":
            if y is None:
                problem_type = "classification"
            else:
                problem_type = _infer_problem_type(y)

        estimator = _resolve_estimator(estimator_name, problem_type)
        if estimator is None:
            logger.error(
                f"Could not resolve estimator '{estimator_name}' for problem type '{problem_type}'"
            )
            return {}



        selector = None
        if method == "select_from_model":
            threshold = config.get("threshold", "mean")
            # Try to convert string number to float
            if isinstance(threshold, str):
                try:
                    threshold = float(threshold)
                except ValueError:
                    pass  # Keep as string (e.g. "mean", "1.25*mean")

            max_features = config.get("max_features", None)
            selector = SelectFromModel(
                estimator=estimator, threshold=threshold, max_features=max_features
            )
        elif method == "rfe":
            n_features_to_select = config.get("n_features_to_select", None)
            step = config.get("step", 1)
            selector = RFE(
                estimator=estimator,
                n_features_to_select=n_features_to_select,
                step=step,
            )

        if not selector:
            return {}

        # Use Bridge for fitting
        if engine.name == "polars":
             X_pl: Any = X
             X_subset = X_pl.select(cols).fill_null(0)
        else:
             X_subset = X[cols].fillna(0)
             
        X_np, _ = SklearnBridge.to_sklearn(X_subset)

        # Handle classification target encoding if needed
        y_fit = y
        
        # Convert y to numpy/pandas for inspection
        if hasattr(y, "to_numpy"):
             y_np = y.to_numpy()
        elif hasattr(y, "to_pandas"):
             y_np = y.to_pandas().to_numpy()
        else:
             y_np = np.array(y)
             
        if (
            problem_type == "classification"
            and y is not None
            # Check if numeric
            and not np.issubdtype(y_np.dtype, np.number)
        ):
            y_factorized, _ = pd.factorize(y_np)
            y_fit = y_factorized
        else:
            y_fit = y_np

        if y is not None:
            selector.fit(X_np, y_fit)
            support = selector.get_support()
            selected_cols = [c for c, s in zip(cols, support) if s]
        else:
            selected_cols = cols  # Fallback
            return {
                "type": "model_based_selection",
                "selected_columns": selected_cols,
                "candidate_columns": cols,
                "method": method,
                "drop_columns": config.get("drop_columns", True),
                "feature_importances": {},
            }

        feature_importances = {}
        if hasattr(selector, "estimator_") and hasattr(
            selector.estimator_, "feature_importances_"
        ):
            feature_importances = dict(
                zip(cols, selector.estimator_.feature_importances_.tolist())
            )
        elif hasattr(selector, "estimator_") and hasattr(selector.estimator_, "coef_"):
            # For linear models, use coef_
            coef = selector.estimator_.coef_
            if coef.ndim > 1:
                coef = coef[0]  # Take first class or flatten
            feature_importances = dict(zip(cols, np.abs(coef).tolist()))

        return {
            "type": "model_based_selection",
            "selected_columns": selected_cols,
            "candidate_columns": cols,
            "method": method,
            "drop_columns": config.get("drop_columns", True),
            "feature_importances": feature_importances,
        }


# --- Unified Feature Selection (Facade) ---
class FeatureSelectionApplier(BaseApplier):
    def apply(
        self,
        df: Union[pd.DataFrame, SkyulfDataFrame, Tuple[Any, ...], Any],
        params: Dict[str, Any],
    ) -> Union[pd.DataFrame, SkyulfDataFrame, Tuple[Any, ...]]:
        # The params returned by the specific calculator will have a "type" field
        # corresponding to the specific calculator's return value.
        type_name = params.get("type")

        applier: Optional[BaseApplier] = None
        if type_name == "variance_threshold":
            applier = VarianceThresholdApplier()
        elif type_name == "correlation_threshold":
            applier = CorrelationThresholdApplier()
        elif type_name == "univariate_selection":
            applier = UnivariateSelectionApplier()
        elif type_name == "model_based_selection":
            applier = ModelBasedSelectionApplier()

        if applier:
            return applier.apply(df, params)  # type: ignore
        return pack_pipeline_output(*unpack_pipeline_input(df))


@NodeRegistry.register("feature_selection", FeatureSelectionApplier)
@node_meta(
    id="feature_selection",
    name="Feature Selection (Wrapper)",
    category="Feature Selection",
    description="General wrapper for feature selection strategies.",
    params={"method": "variance", "threshold": 0.0}
)
class FeatureSelectionCalculator(BaseCalculator):
    def fit(
        self,
        df: Union[pd.DataFrame, SkyulfDataFrame, Tuple[Any, ...], Any],
        config: Dict[str, Any],
    ) -> Dict[str, Any]:
        method = config.get("method", "select_k_best")

        calculator: Optional[BaseCalculator] = None
        if method == "variance_threshold":
            calculator = VarianceThresholdCalculator()
        elif method == "correlation_threshold":
            calculator = CorrelationThresholdCalculator()
        elif method in [
            "select_k_best",
            "select_percentile",
            "generic_univariate_select",
            "select_fpr",
            "select_fdr",
            "select_fwe",
        ]:
            calculator = UnivariateSelectionCalculator()
        elif method in ["select_from_model", "rfe"]:
            calculator = ModelBasedSelectionCalculator()

        if calculator:
            return calculator.fit(df, config)

        logger.warning(f"Unknown feature selection method: {method}")
        return {}

        return df
