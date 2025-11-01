"""
Model training utilities for classification tasks.

This module provides wrapper classes for common classification algorithms
with standardized interface and automatic hyperparameter optimization.
"""

from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
from sklearn.ensemble import (
    RandomForestClassifier as SklearnRFC,
    GradientBoostingClassifier as SklearnGBC,
    AdaBoostClassifier as SklearnAdaBoost,
)
from sklearn.linear_model import LogisticRegression as SklearnLR
from sklearn.tree import DecisionTreeClassifier as SklearnDTC
from sklearn.svm import SVC as SklearnSVC
from sklearn.naive_bayes import GaussianNB as SklearnGNB
from sklearn.neighbors import KNeighborsClassifier as SklearnKNN
from sklearn.model_selection import GridSearchCV, RandomizedSearchCV

from skyulf_mlflow_library.core.exceptions import DataProcessingError


class BaseClassifier:
    """
    Base class for classification models with unified interface.
    
    Provides common functionality for all classifiers including
    hyperparameter tuning and model evaluation.
    """
    
    def __init__(self, random_state: Optional[int] = None, **kwargs):
        """
        Initialize base classifier.
        
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
    
    def predict_proba(self, X):
        """Predict class probabilities."""
        if not self._is_fitted:
            raise DataProcessingError("Model must be fitted before prediction")
        
        if not hasattr(self.model, 'predict_proba'):
            raise DataProcessingError(f"{self.__class__.__name__} doesn't support predict_proba")
        
        return self.model.predict_proba(X)
    
    def get_params(self) -> Dict[str, Any]:
        """Get model parameters."""
        return self.model.get_params()
    
    def set_params(self, **params):
        """Set model parameters."""
        self.model.set_params(**params)
        return self


class RandomForestClassifier(BaseClassifier):
    """
    Random Forest Classifier with hyperparameter tuning support.
    
    Ensemble of decision trees with bootstrap aggregating.
    Good for handling non-linear relationships and feature interactions.
    
    Parameters
    ----------
    n_estimators : int, default=100
        Number of trees in the forest.
    max_depth : int, optional
        Maximum depth of trees.
    min_samples_split : int, default=2
        Minimum samples required to split a node.
    min_samples_leaf : int, default=1
        Minimum samples required at leaf node.
    max_features : str or int, default='sqrt'
        Number of features to consider for best split.
    random_state : int, optional
        Random seed.
    n_jobs : int, default=-1
        Number of parallel jobs.
    
    Examples
    --------
    >>> from skyulf_mlflow_library.modeling.classifiers import RandomForestClassifier
    >>> 
    >>> model = RandomForestClassifier(n_estimators=100, max_depth=10)
    >>> model.fit(X_train, y_train)
    >>> y_pred = model.predict(X_test)
    >>> y_proba = model.predict_proba(X_test)
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
        
        self.model = SklearnRFC(
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
            'max_features': ['sqrt', 'log2']
        }


class GradientBoostingClassifier(BaseClassifier):
    """
    Gradient Boosting Classifier.
    
    Sequential ensemble that builds trees to correct previous errors.
    Often provides better accuracy than Random Forest.
    
    Parameters
    ----------
    n_estimators : int, default=100
        Number of boosting stages.
    learning_rate : float, default=0.1
        Learning rate shrinks contribution of each tree.
    max_depth : int, default=3
        Maximum depth of trees.
    min_samples_split : int, default=2
        Minimum samples to split node.
    subsample : float, default=1.0
        Fraction of samples for fitting trees.
    random_state : int, optional
        Random seed.
    
    Examples
    --------
    >>> from skyulf_mlflow_library.modeling.classifiers import GradientBoostingClassifier
    >>> 
    >>> model = GradientBoostingClassifier(n_estimators=100, learning_rate=0.1)
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
        
        self.model = SklearnGBC(
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


class LogisticRegression(BaseClassifier):
    """
    Logistic Regression for binary and multi-class classification.
    
    Linear model with logistic function. Fast and interpretable.
    Good baseline model.
    
    Parameters
    ----------
    penalty : str, default='l2'
        Regularization norm ('l1', 'l2', 'elasticnet', None).
    C : float, default=1.0
        Inverse of regularization strength.
    solver : str, default='lbfgs'
        Algorithm to use ('lbfgs', 'liblinear', 'saga', etc.).
    max_iter : int, default=100
        Maximum iterations for convergence.
    random_state : int, optional
        Random seed.
    
    Examples
    --------
    >>> from skyulf_mlflow_library.modeling.classifiers import LogisticRegression
    >>> 
    >>> model = LogisticRegression(C=1.0, max_iter=1000)
    >>> model.fit(X_train, y_train)
    >>> y_pred = model.predict(X_test)
    """
    
    def __init__(
        self,
        penalty: str = 'l2',
        C: float = 1.0,
        solver: str = 'lbfgs',
        max_iter: int = 100,
        random_state: Optional[int] = None,
        **kwargs
    ):
        super().__init__(random_state=random_state, **kwargs)
        
        self.model = SklearnLR(
            penalty=penalty,
            C=C,
            solver=solver,
            max_iter=max_iter,
            random_state=random_state,
            **kwargs
        )
    
    @staticmethod
    def get_param_grid() -> Dict[str, List]:
        """Get hyperparameter grid for tuning."""
        return {
            'C': [0.001, 0.01, 0.1, 1.0, 10.0, 100.0],
            'penalty': ['l1', 'l2'],
            'solver': ['liblinear', 'saga']
        }


class SupportVectorClassifier(BaseClassifier):
    """
    Support Vector Machine Classifier.
    
    Finds optimal hyperplane to separate classes.
    Effective in high-dimensional spaces.
    
    Parameters
    ----------
    C : float, default=1.0
        Regularization parameter.
    kernel : str, default='rbf'
        Kernel type ('linear', 'poly', 'rbf', 'sigmoid').
    gamma : str or float, default='scale'
        Kernel coefficient.
    random_state : int, optional
        Random seed.
    
    Examples
    --------
    >>> from skyulf_mlflow_library.modeling.classifiers import SupportVectorClassifier
    >>> 
    >>> model = SupportVectorClassifier(C=1.0, kernel='rbf')
    >>> model.fit(X_train, y_train)
    >>> y_pred = model.predict(X_test)
    """
    
    def __init__(
        self,
        C: float = 1.0,
        kernel: str = 'rbf',
        gamma: str = 'scale',
        probability: bool = True,
        random_state: Optional[int] = None,
        **kwargs
    ):
        super().__init__(random_state=random_state, **kwargs)
        
        self.model = SklearnSVC(
            C=C,
            kernel=kernel,
            gamma=gamma,
            probability=probability,
            random_state=random_state,
            **kwargs
        )
    
    @staticmethod
    def get_param_grid() -> Dict[str, List]:
        """Get hyperparameter grid for tuning."""
        return {
            'C': [0.1, 1.0, 10.0, 100.0],
            'kernel': ['linear', 'rbf', 'poly'],
            'gamma': ['scale', 'auto', 0.001, 0.01, 0.1]
        }


class DecisionTreeClassifier(BaseClassifier):
    """
    Decision Tree Classifier.
    
    Tree-based model that makes decisions based on feature thresholds.
    Interpretable but prone to overfitting.
    
    Parameters
    ----------
    max_depth : int, optional
        Maximum depth of tree.
    min_samples_split : int, default=2
        Minimum samples to split node.
    min_samples_leaf : int, default=1
        Minimum samples at leaf node.
    criterion : str, default='gini'
        Split quality measure ('gini' or 'entropy').
    random_state : int, optional
        Random seed.
    
    Examples
    --------
    >>> from skyulf_mlflow_library.modeling.classifiers import DecisionTreeClassifier
    >>> 
    >>> model = DecisionTreeClassifier(max_depth=5)
    >>> model.fit(X_train, y_train)
    >>> y_pred = model.predict(X_test)
    """
    
    def __init__(
        self,
        max_depth: Optional[int] = None,
        min_samples_split: int = 2,
        min_samples_leaf: int = 1,
        criterion: str = 'gini',
        random_state: Optional[int] = None,
        **kwargs
    ):
        super().__init__(random_state=random_state, **kwargs)
        
        self.model = SklearnDTC(
            max_depth=max_depth,
            min_samples_split=min_samples_split,
            min_samples_leaf=min_samples_leaf,
            criterion=criterion,
            random_state=random_state,
            **kwargs
        )
    
    @staticmethod
    def get_param_grid() -> Dict[str, List]:
        """Get hyperparameter grid for tuning."""
        return {
            'max_depth': [None, 5, 10, 20, 30],
            'min_samples_split': [2, 5, 10],
            'min_samples_leaf': [1, 2, 4],
            'criterion': ['gini', 'entropy']
        }


class KNeighborsClassifier(BaseClassifier):
    """
    K-Nearest Neighbors Classifier.
    
    Non-parametric method that classifies based on k nearest neighbors.
    Simple but can be slow on large datasets.
    
    Parameters
    ----------
    n_neighbors : int, default=5
        Number of neighbors to consider.
    weights : str, default='uniform'
        Weight function ('uniform' or 'distance').
    algorithm : str, default='auto'
        Algorithm to compute nearest neighbors.
    metric : str, default='minkowski'
        Distance metric.
    
    Examples
    --------
    >>> from skyulf_mlflow_library.modeling.classifiers import KNeighborsClassifier
    >>> 
    >>> model = KNeighborsClassifier(n_neighbors=5)
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
        
        self.model = SklearnKNN(
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
    model: BaseClassifier,
    X_train,
    y_train,
    param_grid: Optional[Dict] = None,
    cv: int = 5,
    scoring: str = 'accuracy',
    n_iter: Optional[int] = None,
    random_state: Optional[int] = None,
    n_jobs: int = -1
) -> Tuple[BaseClassifier, Dict[str, Any]]:
    """
    Perform hyperparameter tuning using Grid or Random search.
    
    Parameters
    ----------
    model : BaseClassifier
        Model to tune.
    X_train : array-like
        Training features.
    y_train : array-like
        Training labels.
    param_grid : dict, optional
        Parameter grid. If None, uses model's default grid.
    cv : int, default=5
        Cross-validation folds.
    scoring : str, default='accuracy'
        Scoring metric.
    n_iter : int, optional
        If specified, uses RandomizedSearchCV with n_iter iterations.
        Otherwise uses GridSearchCV.
    random_state : int, optional
        Random seed for RandomizedSearchCV.
    n_jobs : int, default=-1
        Number of parallel jobs.
    
    Returns
    -------
    best_model : BaseClassifier
        Model with best parameters.
    best_params : dict
        Best parameters found.
    
    Examples
    --------
    >>> from skyulf_mlflow_library.modeling.classifiers import RandomForestClassifier, tune_hyperparameters
    >>> 
    >>> model = RandomForestClassifier()
    >>> best_model, best_params = tune_hyperparameters(
    ...     model, X_train, y_train, cv=5, scoring='f1'
    ... )
    >>> print(f"Best params: {best_params}")
    """
    if param_grid is None:
        if hasattr(model, 'get_param_grid'):
            param_grid = model.get_param_grid()
        else:
            raise ValueError("param_grid must be provided or model must have get_param_grid method")
    
    if n_iter is not None:
        # Randomized search
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
        # Grid search
        search = GridSearchCV(
            model.model,
            param_grid=param_grid,
            cv=cv,
            scoring=scoring,
            n_jobs=n_jobs,
            verbose=1
        )
    
    search.fit(X_train, y_train)
    
    # Update model with best parameters
    model.model = search.best_estimator_
    model._is_fitted = True
    
    return model, search.best_params_
