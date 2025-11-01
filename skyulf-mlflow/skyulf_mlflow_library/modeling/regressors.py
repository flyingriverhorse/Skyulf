"""
Model training utilities for regression tasks.

This module provides wrapper classes for common regression algorithms
with standardized interface and automatic hyperparameter optimization.
"""

from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
from sklearn.ensemble import (
    RandomForestRegressor as SklearnRFR,
    GradientBoostingRegressor as SklearnGBR,
    AdaBoostRegressor as SklearnAdaBoostR,
)
from sklearn.linear_model import (
    LinearRegression as SklearnLR,
    Ridge as SklearnRidge,
    Lasso as SklearnLasso,
    ElasticNet as SklearnElasticNet,
)
from sklearn.tree import DecisionTreeRegressor as SklearnDTR
from sklearn.svm import SVR as SklearnSVR
from sklearn.neighbors import KNeighborsRegressor as SklearnKNR
from sklearn.model_selection import GridSearchCV, RandomizedSearchCV

from skyulf_mlflow_library.core.exceptions import DataProcessingError


class BaseRegressor:
    """
    Base class for regression models with unified interface.
    
    Provides common functionality for all regressors including
    hyperparameter tuning and model evaluation.
    """
    
    def __init__(self, random_state: Optional[int] = None, **kwargs):
        """
        Initialize base regressor.
        
        Parameters
        ----------
        random_state : int, optional
            Random seed for reproducibility.
        **kwargs : dict
            Additional model parameters.
        """
        self.random_state = random_state
        self.params = kwargs
        self.model = None
        self._is_fitted = False
    
    def fit(self, X, y):
        """Fit the model."""
        if not isinstance(X, (pd.DataFrame, np.ndarray)):
            raise DataProcessingError("X must be DataFrame or numpy array")
        
        self.model.fit(X, y)
        self._is_fitted = True
        return self
    
    def predict(self, X):
        """Make predictions."""
        if not self._is_fitted:
            raise DataProcessingError("Model must be fitted before prediction")
        return self.model.predict(X)
    
    def score(self, X, y):
        """Return R² score."""
        if not self._is_fitted:
            raise DataProcessingError("Model must be fitted before scoring")
        return self.model.score(X, y)
    
    def get_params(self) -> Dict[str, Any]:
        """Get model parameters."""
        return self.model.get_params()
    
    def set_params(self, **params):
        """Set model parameters."""
        self.model.set_params(**params)
        return self


class RandomForestRegressor(BaseRegressor):
    """
    Random Forest Regressor.
    
    Ensemble of decision trees for regression tasks.
    Handles non-linear relationships well.
    
    Parameters
    ----------
    n_estimators : int, default=100
        Number of trees.
    max_depth : int, optional
        Maximum tree depth.
    min_samples_split : int, default=2
        Minimum samples to split node.
    min_samples_leaf : int, default=1
        Minimum samples at leaf.
    max_features : str or float, default='sqrt'
        Number of features for best split.
    random_state : int, optional
        Random seed.
    n_jobs : int, default=-1
        Parallel jobs.
    
    Examples
    --------
    >>> from skyulf_mlflow_library.modeling.regressors import RandomForestRegressor
    >>> 
    >>> model = RandomForestRegressor(n_estimators=100)
    >>> model.fit(X_train, y_train)
    >>> y_pred = model.predict(X_test)
    >>> score = model.score(X_test, y_test)
    """
    
    def __init__(
        self,
        n_estimators: int = 100,
        max_depth: Optional[int] = None,
        min_samples_split: int = 2,
        min_samples_leaf: int = 1,
        max_features: str = 'sqrt',
        random_state: Optional[int] = None,
        n_jobs: int = -1,
        **kwargs
    ):
        super().__init__(random_state=random_state, **kwargs)
        
        self.model = SklearnRFR(
            n_estimators=n_estimators,
            max_depth=max_depth,
            min_samples_split=min_samples_split,
            min_samples_leaf=min_samples_leaf,
            max_features=max_features,
            random_state=random_state,
            n_jobs=n_jobs,
            **kwargs
        )
    
    @staticmethod
    def get_param_grid() -> Dict[str, List]:
        """Get hyperparameter grid for tuning."""
        return {
            'n_estimators': [50, 100, 200, 300],
            'max_depth': [None, 10, 20, 30],
            'min_samples_split': [2, 5, 10],
            'min_samples_leaf': [1, 2, 4],
            'max_features': ['sqrt', 'log2', None]
        }


class GradientBoostingRegressor(BaseRegressor):
    """
    Gradient Boosting Regressor.
    
    Sequential ensemble for regression.
    Often achieves high accuracy.
    
    Parameters
    ----------
    n_estimators : int, default=100
        Number of boosting stages.
    learning_rate : float, default=0.1
        Learning rate.
    max_depth : int, default=3
        Maximum tree depth.
    min_samples_split : int, default=2
        Minimum samples to split.
    subsample : float, default=1.0
        Fraction of samples.
    random_state : int, optional
        Random seed.
    
    Examples
    --------
    >>> from skyulf_mlflow_library.modeling.regressors import GradientBoostingRegressor
    >>> 
    >>> model = GradientBoostingRegressor(n_estimators=100, learning_rate=0.1)
    >>> model.fit(X_train, y_train)
    >>> y_pred = model.predict(X_test)
    """
    
    def __init__(
        self,
        n_estimators: int = 100,
        learning_rate: float = 0.1,
        max_depth: int = 3,
        min_samples_split: int = 2,
        min_samples_leaf: int = 1,
        subsample: float = 1.0,
        random_state: Optional[int] = None,
        **kwargs
    ):
        super().__init__(random_state=random_state, **kwargs)
        
        self.model = SklearnGBR(
            n_estimators=n_estimators,
            learning_rate=learning_rate,
            max_depth=max_depth,
            min_samples_split=min_samples_split,
            min_samples_leaf=min_samples_leaf,
            subsample=subsample,
            random_state=random_state,
            **kwargs
        )
    
    @staticmethod
    def get_param_grid() -> Dict[str, List]:
        """Get hyperparameter grid for tuning."""
        return {
            'n_estimators': [50, 100, 200],
            'learning_rate': [0.01, 0.1, 0.3],
            'max_depth': [3, 5, 7],
            'min_samples_split': [2, 5, 10],
            'subsample': [0.8, 1.0]
        }


class LinearRegression(BaseRegressor):
    """
    Linear Regression (Ordinary Least Squares).
    
    Simple linear model. Fast and interpretable.
    Good baseline for regression tasks.
    
    Parameters
    ----------
    fit_intercept : bool, default=True
        Whether to calculate intercept.
    normalize : bool, default=False
        Whether to normalize features (deprecated in sklearn).
    n_jobs : int, optional
        Number of parallel jobs.
    
    Examples
    --------
    >>> from skyulf_mlflow_library.modeling.regressors import LinearRegression
    >>> 
    >>> model = LinearRegression()
    >>> model.fit(X_train, y_train)
    >>> y_pred = model.predict(X_test)
    """
    
    def __init__(
        self,
        fit_intercept: bool = True,
        n_jobs: Optional[int] = None,
        **kwargs
    ):
        super().__init__(**kwargs)
        
        self.model = SklearnLR(
            fit_intercept=fit_intercept,
            n_jobs=n_jobs,
            **kwargs
        )


class Ridge(BaseRegressor):
    """
    Ridge Regression (L2 regularization).
    
    Linear regression with L2 penalty.
    Helps prevent overfitting.
    
    Parameters
    ----------
    alpha : float, default=1.0
        Regularization strength.
    fit_intercept : bool, default=True
        Whether to calculate intercept.
    random_state : int, optional
        Random seed.
    
    Examples
    --------
    >>> from skyulf_mlflow_library.modeling.regressors import Ridge
    >>> 
    >>> model = Ridge(alpha=1.0)
    >>> model.fit(X_train, y_train)
    >>> y_pred = model.predict(X_test)
    """
    
    def __init__(
        self,
        alpha: float = 1.0,
        fit_intercept: bool = True,
        random_state: Optional[int] = None,
        **kwargs
    ):
        super().__init__(random_state=random_state, **kwargs)
        
        self.model = SklearnRidge(
            alpha=alpha,
            fit_intercept=fit_intercept,
            random_state=random_state,
            **kwargs
        )
    
    @staticmethod
    def get_param_grid() -> Dict[str, List]:
        """Get hyperparameter grid for tuning."""
        return {
            'alpha': [0.001, 0.01, 0.1, 1.0, 10.0, 100.0]
        }


class Lasso(BaseRegressor):
    """
    Lasso Regression (L1 regularization).
    
    Linear regression with L1 penalty.
    Performs feature selection by setting coefficients to zero.
    
    Parameters
    ----------
    alpha : float, default=1.0
        Regularization strength.
    fit_intercept : bool, default=True
        Whether to calculate intercept.
    max_iter : int, default=1000
        Maximum iterations.
    random_state : int, optional
        Random seed.
    
    Examples
    --------
    >>> from skyulf_mlflow_library.modeling.regressors import Lasso
    >>> 
    >>> model = Lasso(alpha=0.1)
    >>> model.fit(X_train, y_train)
    >>> y_pred = model.predict(X_test)
    """
    
    def __init__(
        self,
        alpha: float = 1.0,
        fit_intercept: bool = True,
        max_iter: int = 1000,
        random_state: Optional[int] = None,
        **kwargs
    ):
        super().__init__(random_state=random_state, **kwargs)
        
        self.model = SklearnLasso(
            alpha=alpha,
            fit_intercept=fit_intercept,
            max_iter=max_iter,
            random_state=random_state,
            **kwargs
        )
    
    @staticmethod
    def get_param_grid() -> Dict[str, List]:
        """Get hyperparameter grid for tuning."""
        return {
            'alpha': [0.001, 0.01, 0.1, 1.0, 10.0]
        }


class ElasticNet(BaseRegressor):
    """
    ElasticNet Regression (L1 + L2 regularization).
    
    Combines L1 and L2 penalties.
    Good for datasets with many correlated features.
    
    Parameters
    ----------
    alpha : float, default=1.0
        Regularization strength.
    l1_ratio : float, default=0.5
        Mix ratio between L1 and L2 (0=Ridge, 1=Lasso).
    fit_intercept : bool, default=True
        Whether to calculate intercept.
    max_iter : int, default=1000
        Maximum iterations.
    random_state : int, optional
        Random seed.
    
    Examples
    --------
    >>> from skyulf_mlflow_library.modeling.regressors import ElasticNet
    >>> 
    >>> model = ElasticNet(alpha=1.0, l1_ratio=0.5)
    >>> model.fit(X_train, y_train)
    >>> y_pred = model.predict(X_test)
    """
    
    def __init__(
        self,
        alpha: float = 1.0,
        l1_ratio: float = 0.5,
        fit_intercept: bool = True,
        max_iter: int = 1000,
        random_state: Optional[int] = None,
        **kwargs
    ):
        super().__init__(random_state=random_state, **kwargs)
        
        self.model = SklearnElasticNet(
            alpha=alpha,
            l1_ratio=l1_ratio,
            fit_intercept=fit_intercept,
            max_iter=max_iter,
            random_state=random_state,
            **kwargs
        )
    
    @staticmethod
    def get_param_grid() -> Dict[str, List]:
        """Get hyperparameter grid for tuning."""
        return {
            'alpha': [0.001, 0.01, 0.1, 1.0, 10.0],
            'l1_ratio': [0.1, 0.3, 0.5, 0.7, 0.9]
        }


class DecisionTreeRegressor(BaseRegressor):
    """
    Decision Tree Regressor.
    
    Tree-based regression model.
    Interpretable but prone to overfitting.
    
    Parameters
    ----------
    max_depth : int, optional
        Maximum tree depth.
    min_samples_split : int, default=2
        Minimum samples to split.
    min_samples_leaf : int, default=1
        Minimum samples at leaf.
    random_state : int, optional
        Random seed.
    
    Examples
    --------
    >>> from skyulf_mlflow_library.modeling.regressors import DecisionTreeRegressor
    >>> 
    >>> model = DecisionTreeRegressor(max_depth=5)
    >>> model.fit(X_train, y_train)
    >>> y_pred = model.predict(X_test)
    """
    
    def __init__(
        self,
        max_depth: Optional[int] = None,
        min_samples_split: int = 2,
        min_samples_leaf: int = 1,
        random_state: Optional[int] = None,
        **kwargs
    ):
        super().__init__(random_state=random_state, **kwargs)
        
        self.model = SklearnDTR(
            max_depth=max_depth,
            min_samples_split=min_samples_split,
            min_samples_leaf=min_samples_leaf,
            random_state=random_state,
            **kwargs
        )
    
    @staticmethod
    def get_param_grid() -> Dict[str, List]:
        """Get hyperparameter grid for tuning."""
        return {
            'max_depth': [None, 5, 10, 20, 30],
            'min_samples_split': [2, 5, 10],
            'min_samples_leaf': [1, 2, 4]
        }


class KNeighborsRegressor(BaseRegressor):
    """
    K-Nearest Neighbors Regressor.
    
    Non-parametric regression based on k nearest neighbors.
    Simple but can be slow on large datasets.
    
    Parameters
    ----------
    n_neighbors : int, default=5
        Number of neighbors.
    weights : str, default='uniform'
        Weight function.
    algorithm : str, default='auto'
        Algorithm to compute neighbors.
    metric : str, default='minkowski'
        Distance metric.
    n_jobs : int, default=-1
        Parallel jobs.
    
    Examples
    --------
    >>> from skyulf_mlflow_library.modeling.regressors import KNeighborsRegressor
    >>> 
    >>> model = KNeighborsRegressor(n_neighbors=5)
    >>> model.fit(X_train, y_train)
    >>> y_pred = model.predict(X_test)
    """
    
    def __init__(
        self,
        n_neighbors: int = 5,
        weights: str = 'uniform',
        algorithm: str = 'auto',
        metric: str = 'minkowski',
        n_jobs: int = -1,
        **kwargs
    ):
        super().__init__(**kwargs)
        
        self.model = SklearnKNR(
            n_neighbors=n_neighbors,
            weights=weights,
            algorithm=algorithm,
            metric=metric,
            n_jobs=n_jobs,
            **kwargs
        )
    
    @staticmethod
    def get_param_grid() -> Dict[str, List]:
        """Get hyperparameter grid for tuning."""
        return {
            'n_neighbors': [3, 5, 7, 9, 11],
            'weights': ['uniform', 'distance'],
            'metric': ['euclidean', 'manhattan', 'minkowski']
        }


def tune_hyperparameters(
    model: BaseRegressor,
    X_train,
    y_train,
    param_grid: Optional[Dict] = None,
    cv: int = 5,
    scoring: str = 'r2',
    n_iter: Optional[int] = None,
    random_state: Optional[int] = None,
    n_jobs: int = -1
) -> Tuple[BaseRegressor, Dict[str, Any]]:
    """
    Perform hyperparameter tuning for regression models.
    
    Parameters
    ----------
    model : BaseRegressor
        Model to tune.
    X_train : array-like
        Training features.
    y_train : array-like
        Training target.
    param_grid : dict, optional
        Parameter grid.
    cv : int, default=5
        Cross-validation folds.
    scoring : str, default='r2'
        Scoring metric ('r2', 'neg_mean_squared_error', 'neg_mean_absolute_error').
    n_iter : int, optional
        For RandomizedSearchCV.
    random_state : int, optional
        Random seed.
    n_jobs : int, default=-1
        Parallel jobs.
    
    Returns
    -------
    best_model : BaseRegressor
        Model with best parameters.
    best_params : dict
        Best parameters found.
    
    Examples
    --------
    >>> from skyulf_mlflow_library.modeling.regressors import RandomForestRegressor, tune_hyperparameters
    >>> 
    >>> model = RandomForestRegressor()
    >>> best_model, best_params = tune_hyperparameters(
    ...     model, X_train, y_train, cv=5, scoring='r2'
    ... )
    """
    if param_grid is None:
        if hasattr(model, 'get_param_grid'):
            param_grid = model.get_param_grid()
        else:
            raise ValueError("param_grid must be provided")
    
    if n_iter is not None:
        search = RandomizedSearchCV(
            model.model,
            param_distributions=param_grid,
            n_iter=n_iter,
            cv=cv,
            scoring=scoring,
            random_state=random_state,
            n_jobs=n_jobs,
            verbose=1
        )
    else:
        search = GridSearchCV(
            model.model,
            param_grid=param_grid,
            cv=cv,
            scoring=scoring,
            n_jobs=n_jobs,
            verbose=1
        )
    
    search.fit(X_train, y_train)
    
    model.model = search.best_estimator_
    model._is_fitted = True
    
    return model, search.best_params_
