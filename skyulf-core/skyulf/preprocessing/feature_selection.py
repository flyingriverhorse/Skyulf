from typing import Dict, Any, List, Optional, Union, Callable, Tuple
import logging
import pandas as pd
import numpy as np
from sklearn.feature_selection import (
    VarianceThreshold, SelectKBest, SelectPercentile, SelectFpr, SelectFdr, SelectFwe,
    GenericUnivariateSelect, SelectFromModel, RFE,
    f_classif, f_regression, chi2, mutual_info_classif, mutual_info_regression, r_regression
)
from sklearn.linear_model import LinearRegression, LogisticRegression
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor
from .base import BaseCalculator, BaseApplier
from ..utils import detect_numeric_columns, resolve_columns, unpack_pipeline_input, pack_pipeline_output

logger = logging.getLogger(__name__)

# --- Helpers ---
SCORE_FUNCTIONS = {
    "f_classif": f_classif,
    "f_regression": f_regression,
    "mutual_info_classif": mutual_info_classif,
    "mutual_info_regression": mutual_info_regression,
    "chi2": chi2,
    "r_regression": r_regression,
}

def _infer_problem_type(series: pd.Series) -> str:
    if series.empty: return "classification"
    if pd.api.types.is_bool_dtype(series) or pd.api.types.is_object_dtype(series):
        return "classification"
    unique_values = series.dropna().unique()
    if len(unique_values) <= 10:
        return "classification"
    return "regression"

def _resolve_score_function(name: Optional[str], problem_type: str) -> Callable:
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
            return LinearRegression() # Odd for classification but allowed in V1 logic
    else:
        if key in {"auto", "linear_regression", "linearregression"}:
            return LinearRegression()
        if key in {"random_forest", "randomforest"}:
            return RandomForestRegressor(n_estimators=100, random_state=0, n_jobs=-1)
    return None

# --- Variance Threshold ---
class VarianceThresholdCalculator(BaseCalculator):
    def fit(self, df: Union[pd.DataFrame, Tuple[pd.DataFrame, pd.Series]], config: Dict[str, Any]) -> Dict[str, Any]:
        X, _, _ = unpack_pipeline_input(df)

        # Config: {"threshold": 0.0, "columns": [...]}
        threshold = config.get("threshold", 0.0)
        drop_columns = config.get("drop_columns", True)
        
        cols = resolve_columns(X, config, lambda d: d.select_dtypes(include=["number"]).columns.tolist())
        
        if not cols:
            return {}
            
        selector = VarianceThreshold(threshold=threshold)
        selector.fit(X[cols])
        
        support = selector.get_support()
        selected_cols = [c for c, s in zip(cols, support) if s]
        
        variances = {}
        if hasattr(selector, 'variances_'):
             variances = dict(zip(cols, selector.variances_.tolist()))

        return {
            "type": "variance_threshold",
            "selected_columns": selected_cols,
            "candidate_columns": cols,
            "threshold": threshold,
            "drop_columns": drop_columns,
            "variances": variances
        }

class VarianceThresholdApplier(BaseApplier):
    def apply(self, df: Union[pd.DataFrame, Tuple[pd.DataFrame, pd.Series]], params: Dict[str, Any]) -> Union[pd.DataFrame, Tuple[pd.DataFrame, pd.Series]]:
        X, y, is_tuple = unpack_pipeline_input(df)
        
        selected_cols = params.get("selected_columns")
        candidate_columns = params.get("candidate_columns", [])
        drop_columns = params.get("drop_columns", True)
        
        if selected_cols is None:
            return pack_pipeline_output(X, y, is_tuple)
            
        cols_to_drop = set(candidate_columns) - set(selected_cols)
        cols_to_drop = [c for c in cols_to_drop if c in X.columns]
        
        if cols_to_drop and drop_columns:
            X = X.drop(columns=cols_to_drop)
        return pack_pipeline_output(X, y, is_tuple)

# --- Correlation Threshold ---
class CorrelationThresholdCalculator(BaseCalculator):
    def fit(self, df: Union[pd.DataFrame, Tuple[pd.DataFrame, pd.Series]], config: Dict[str, Any]) -> Dict[str, Any]:
        X, _, _ = unpack_pipeline_input(df)

        # Config: {"threshold": 0.95, "correlation_method": "pearson"}
        threshold = config.get("threshold", 0.95)
        drop_columns = config.get("drop_columns", True)
        # Prefer "correlation_method" to avoid conflict with facade's "method"
        # FIX: Do NOT fallback to config.get("method") because it might be "correlation_threshold" (the facade method name)
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
            "drop_columns": drop_columns
        }

class CorrelationThresholdApplier(BaseApplier):
    def apply(self, df: Union[pd.DataFrame, Tuple[pd.DataFrame, pd.Series]], params: Dict[str, Any]) -> Union[pd.DataFrame, Tuple[pd.DataFrame, pd.Series]]:
        X, y, is_tuple = unpack_pipeline_input(df)
        
        cols_to_drop = params.get("columns_to_drop", [])
        drop_columns = params.get("drop_columns", True)
        
        cols_to_drop = [c for c in cols_to_drop if c in X.columns]
        if not cols_to_drop:
            return pack_pipeline_output(X, y, is_tuple)
        
        if drop_columns:
            X = X.drop(columns=cols_to_drop)
        return pack_pipeline_output(X, y, is_tuple)

# --- Univariate Selection ---
class UnivariateSelectionCalculator(BaseCalculator):
    def fit(self, df: Union[pd.DataFrame, Tuple[pd.DataFrame, pd.Series]], config: Dict[str, Any]) -> Dict[str, Any]:
        # Config: method, k, percentile, alpha, score_func, target_column
        target_col = config.get("target_column")
        
        X, y, is_tuple = unpack_pipeline_input(df)
        
        if not is_tuple:
            if not target_col or target_col not in X.columns:
                logger.error(f"UnivariateSelection requires target column '{target_col}' to be present in training data.")
                return {}
            y = X[target_col]

        cols = resolve_columns(X, config, detect_numeric_columns)
        
        # Ensure target is not in candidate columns
        if target_col in cols:
            cols = [c for c in cols if c != target_col]
        
        if not cols:
            return {}
            
        method = config.get("method", "select_k_best")
        score_func_name = config.get("score_func")
        problem_type = config.get("problem_type", "auto")
        
        if problem_type == "auto":
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
                if mode == "k_best": param = config.get("k", 10)
                elif mode == "percentile": param = config.get("percentile", 10)
                else: param = config.get("alpha", 0.05)
            
            selector = GenericUnivariateSelect(score_func=score_func, mode=mode, param=param)
            
        if not selector:
            return {}
            
        X_fit = X[cols].fillna(0)
        
        # Handle Chi2 negative values
        if score_func_name == "chi2" and (X_fit < 0).any().any():
            logger.warning("Chi-squared statistic requires non-negative feature values. Applying MinMaxScaler to features for selection.")
            from sklearn.preprocessing import MinMaxScaler
            X_fit = pd.DataFrame(MinMaxScaler().fit_transform(X_fit), columns=X_fit.columns, index=X_fit.index)

        # Handle classification target encoding if needed
        if problem_type == "classification" and not pd.api.types.is_numeric_dtype(y):
            y = pd.factorize(y)[0]
            
        selector.fit(X_fit, y)
        support = selector.get_support()
        selected_cols = [c for c, s in zip(cols, support) if s]
        
        scores = {}
        pvalues = {}
        if hasattr(selector, 'scores_'):
            # Handle potential NaN/Inf in scores
            safe_scores = np.nan_to_num(selector.scores_, nan=0.0, posinf=0.0, neginf=0.0)
            scores = dict(zip(cols, safe_scores.tolist()))
            
        if hasattr(selector, 'pvalues_'):
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
            "p_values": pvalues
        }

class UnivariateSelectionApplier(BaseApplier):
    def apply(self, df: Union[pd.DataFrame, Tuple[pd.DataFrame, pd.Series]], params: Dict[str, Any]) -> Union[pd.DataFrame, Tuple[pd.DataFrame, pd.Series]]:
        X, y, is_tuple = unpack_pipeline_input(df)
        
        selected_cols = params.get("selected_columns")
        candidate_columns = params.get("candidate_columns", [])
        drop_columns = params.get("drop_columns", True)
        
        if selected_cols is None:
            return pack_pipeline_output(X, y, is_tuple)
        
        cols_to_drop = set(candidate_columns) - set(selected_cols)
        cols_to_drop = [c for c in cols_to_drop if c in X.columns]
        if cols_to_drop and drop_columns:
            X = X.drop(columns=cols_to_drop)
        return pack_pipeline_output(X, y, is_tuple)

# --- Model Based Selection ---
class ModelBasedSelectionCalculator(BaseCalculator):
    def fit(self, df: Union[pd.DataFrame, Tuple[pd.DataFrame, pd.Series]], config: Dict[str, Any]) -> Dict[str, Any]:
        # Config: method (select_from_model, rfe), estimator, target_column
        target_col = config.get("target_column")
        
        X, y, is_tuple = unpack_pipeline_input(df)
        
        if not is_tuple:
            if not target_col or target_col not in X.columns:
                logger.error(f"ModelBasedSelection requires target column '{target_col}' to be present in training data.")
                return {}
            y = X[target_col]

        cols = resolve_columns(X, config, detect_numeric_columns)
        
        # Ensure target is not in candidate columns
        if target_col in cols:
            cols = [c for c in cols if c != target_col]
        
        if not cols:
            return {}
            
        method = config.get("method", "select_from_model")
        estimator_name = config.get("estimator", "auto")
        problem_type = config.get("problem_type", "auto")
        
        if problem_type == "auto":
            problem_type = _infer_problem_type(y)
            
        estimator = _resolve_estimator(estimator_name, problem_type)
        if estimator is None:
            logger.error(f"Could not resolve estimator '{estimator_name}' for problem type '{problem_type}'")
            return {}
            
        X_fit = X[cols].fillna(0)
        
        # Handle classification target encoding if needed
        if problem_type == "classification" and not pd.api.types.is_numeric_dtype(y):
            y = pd.factorize(y)[0]
            
        selector = None
        if method == "select_from_model":
            threshold = config.get("threshold", "mean")
            max_features = config.get("max_features", None)
            selector = SelectFromModel(estimator=estimator, threshold=threshold, max_features=max_features)
        elif method == "rfe":
            n_features_to_select = config.get("n_features_to_select", None)
            step = config.get("step", 1)
            selector = RFE(estimator=estimator, n_features_to_select=n_features_to_select, step=step)
            
        if not selector:
            return {}
            
        selector.fit(X_fit, y)
        support = selector.get_support()
        selected_cols = [c for c, s in zip(cols, support) if s]
        
        feature_importances = {}
        if hasattr(selector, 'estimator_') and hasattr(selector.estimator_, 'feature_importances_'):
             feature_importances = dict(zip(cols, selector.estimator_.feature_importances_.tolist()))
        elif hasattr(selector, 'estimator_') and hasattr(selector.estimator_, 'coef_'):
             # For linear models, use coef_
             coef = selector.estimator_.coef_
             if coef.ndim > 1: coef = coef[0] # Take first class or flatten
             feature_importances = dict(zip(cols, np.abs(coef).tolist()))

        return {
            "type": "model_based_selection",
            "selected_columns": selected_cols,
            "candidate_columns": cols,
            "method": method,
            "drop_columns": config.get("drop_columns", True),
            "feature_importances": feature_importances
        }

class ModelBasedSelectionApplier(BaseApplier):
    def apply(self, df: Union[pd.DataFrame, Tuple[pd.DataFrame, pd.Series]], params: Dict[str, Any]) -> Union[pd.DataFrame, Tuple[pd.DataFrame, pd.Series]]:
        X, y, is_tuple = unpack_pipeline_input(df)
        
        selected_cols = params.get("selected_columns")
        candidate_columns = params.get("candidate_columns", [])
        drop_columns = params.get("drop_columns", True)
        
        if selected_cols is None:
            return pack_pipeline_output(X, y, is_tuple)
        
        cols_to_drop = set(candidate_columns) - set(selected_cols)
        cols_to_drop = [c for c in cols_to_drop if c in X.columns]
        if cols_to_drop and drop_columns:
            X = X.drop(columns=cols_to_drop)
        return pack_pipeline_output(X, y, is_tuple)


# --- Unified Feature Selection (Facade) ---
class FeatureSelectionCalculator(BaseCalculator):
    def fit(self, df: Union[pd.DataFrame, Tuple[pd.DataFrame, pd.Series]], config: Dict[str, Any]) -> Dict[str, Any]:
        method = config.get("method", "select_k_best")
        
        calculator = None
        if method == "variance_threshold":
            calculator = VarianceThresholdCalculator()
        elif method == "correlation_threshold":
            calculator = CorrelationThresholdCalculator()
        elif method in ["select_k_best", "select_percentile", "generic_univariate_select", "select_fpr", "select_fdr", "select_fwe"]:
            calculator = UnivariateSelectionCalculator()
        elif method in ["select_from_model", "rfe"]:
            calculator = ModelBasedSelectionCalculator()
        
        if calculator:
            return calculator.fit(df, config)
            
        logger.warning(f"Unknown feature selection method: {method}")
        return {}


class FeatureSelectionApplier(BaseApplier):
    def apply(self, df: Union[pd.DataFrame, Tuple[pd.DataFrame, pd.Series]], params: Dict[str, Any]) -> Union[pd.DataFrame, Tuple[pd.DataFrame, pd.Series]]:
        # The params returned by the specific calculator will have a "type" field
        # corresponding to the specific calculator's return value.
        type_name = params.get("type")
        
        applier = None
        if type_name == "variance_threshold":
            applier = VarianceThresholdApplier()
        elif type_name == "correlation_threshold":
            applier = CorrelationThresholdApplier()
        elif type_name == "univariate_selection":
            applier = UnivariateSelectionApplier()
        elif type_name == "model_based_selection":
            applier = ModelBasedSelectionApplier()
            
        if applier:
            return applier.apply(df, params)
            
        return df
