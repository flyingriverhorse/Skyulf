"""
Model Metrics Calculator for Skyulf MLflow.

This module provides functions for calculating classification and regression metrics.
"""

from typing import Any, Dict, List, Optional, Union

import numpy as np
import pandas as pd
from sklearn.metrics import (
    accuracy_score,
    average_precision_score,
    f1_score,
    mean_absolute_error,
    mean_absolute_percentage_error,
    mean_squared_error,
    precision_score,
    recall_score,
    roc_auc_score,
    r2_score,
    confusion_matrix,
    classification_report,
)
from sklearn.preprocessing import label_binarize

# Try to import imblearn for geometric mean
try:
    from imblearn.metrics import geometric_mean_score
    HAS_IMBLEARN = True
except ImportError:
    HAS_IMBLEARN = False
    geometric_mean_score = None


class MetricsCalculator:
    """
    Calculate performance metrics for classification and regression models.
    
    This class provides a unified interface for computing various metrics
    for both classification and regression tasks, including support for
    multi-class problems and probability-based metrics.
    
    Parameters
    ----------
    problem_type : {'classification', 'regression'}
        Type of machine learning problem.
    
    Examples
    --------
    >>> from skyulf_mlflow_library.modeling import MetricsCalculator
    >>> import numpy as np
    >>> 
    >>> # Classification metrics
    >>> y_true = [0, 1, 1, 0, 1]
    >>> y_pred = [0, 1, 0, 0, 1]
    >>> calc = MetricsCalculator('classification')
    >>> metrics = calc.calculate(y_true, y_pred)
    >>> print(metrics['accuracy'])
    0.8
    >>> 
    >>> # Regression metrics
    >>> y_true = [3.0, -0.5, 2.0, 7.0]
    >>> y_pred = [2.5, 0.0, 2.0, 8.0]
    >>> calc = MetricsCalculator('regression')
    >>> metrics = calc.calculate(y_true, y_pred)
    >>> print(metrics['mae'])
    0.5
    """
    
    def __init__(self, problem_type: str):
        """
        Initialize the metrics calculator.
        
        Parameters
        ----------
        problem_type : {'classification', 'regression'}
            Type of machine learning problem.
        """
        if problem_type not in {'classification', 'regression'}:
            raise ValueError(
                f"problem_type must be 'classification' or 'regression', got '{problem_type}'"
            )
        self.problem_type = problem_type
    
    def calculate(
        self,
        y_true: Union[np.ndarray, pd.Series, List],
        y_pred: Union[np.ndarray, pd.Series, List],
        y_prob: Optional[Union[np.ndarray, pd.DataFrame]] = None,
        average: str = 'weighted',
        **kwargs
    ) -> Dict[str, float]:
        """
        Calculate metrics for the given predictions.
        
        Parameters
        ----------
        y_true : array-like
            True target values.
        y_pred : array-like
            Predicted target values.
        y_prob : array-like, optional
            Predicted probabilities for classification (for ROC AUC, PR AUC).
        average : str, default='weighted'
            Averaging strategy for multi-class classification metrics.
            Options: 'micro', 'macro', 'weighted', 'samples'.
        **kwargs : dict
            Additional keyword arguments for specific metrics.
        
        Returns
        -------
        dict
            Dictionary containing computed metrics.
        """
        # Convert to numpy arrays
        y_true = np.asarray(y_true)
        y_pred = np.asarray(y_pred)
        
        if self.problem_type == 'classification':
            return self._classification_metrics(y_true, y_pred, y_prob, average, **kwargs)
        else:
            return self._regression_metrics(y_true, y_pred, **kwargs)
    
    def _classification_metrics(
        self,
        y_true: np.ndarray,
        y_pred: np.ndarray,
        y_prob: Optional[np.ndarray],
        average: str,
        **kwargs
    ) -> Dict[str, float]:
        """
        Compute classification metrics.
        
        Parameters
        ----------
        y_true : np.ndarray
            True labels.
        y_pred : np.ndarray
            Predicted labels.
        y_prob : np.ndarray, optional
            Predicted probabilities.
        average : str
            Averaging strategy.
        **kwargs : dict
            Additional arguments.
        
        Returns
        -------
        dict
            Classification metrics.
        """
        metrics: Dict[str, float] = {}
        
        # Basic metrics
        metrics['accuracy'] = float(accuracy_score(y_true, y_pred))
        metrics['precision'] = float(precision_score(
            y_true, y_pred, average=average, zero_division=0
        ))
        metrics['recall'] = float(recall_score(
            y_true, y_pred, average=average, zero_division=0
        ))
        metrics['f1'] = float(f1_score(
            y_true, y_pred, average=average, zero_division=0
        ))
        
        # Geometric mean score (if imblearn is available)
        if HAS_IMBLEARN and geometric_mean_score is not None:
            try:
                metrics['g_score'] = float(geometric_mean_score(
                    y_true, y_pred, average=average
                ))
            except Exception:
                # Some class distributions can make geometric mean undefined
                pass
        
        # Probability-based metrics
        if y_prob is not None:
            try:
                unique_classes = len(np.unique(y_true))
                
                if unique_classes == 2:
                    # Binary classification
                    if y_prob.ndim == 2:
                        # Use probability for positive class
                        prob_positive = y_prob[:, 1]
                    else:
                        prob_positive = y_prob
                    
                    metrics['roc_auc'] = float(roc_auc_score(y_true, prob_positive))
                    metrics['pr_auc'] = float(average_precision_score(y_true, prob_positive))
                
                elif unique_classes > 2:
                    # Multi-class classification
                    if y_prob.ndim == 2 and y_prob.shape[1] >= unique_classes:
                        # ROC AUC with one-vs-rest
                        metrics['roc_auc_ovr'] = float(roc_auc_score(
                            y_true, y_prob, multi_class='ovr', average=average
                        ))
                        
                        # PR AUC with binarized labels
                        classes = np.unique(y_true)
                        y_bin = label_binarize(y_true, classes=classes)
                        metrics['pr_auc'] = float(average_precision_score(
                            y_bin, y_prob, average=average
                        ))
            except Exception as e:
                # Skip probability-based metrics if they fail
                pass
        
        return metrics
    
    def _regression_metrics(
        self,
        y_true: np.ndarray,
        y_pred: np.ndarray,
        **kwargs
    ) -> Dict[str, Optional[float]]:
        """
        Compute regression metrics.
        
        Parameters
        ----------
        y_true : np.ndarray
            True values.
        y_pred : np.ndarray
            Predicted values.
        **kwargs : dict
            Additional arguments.
        
        Returns
        -------
        dict
            Regression metrics.
        """
        metrics: Dict[str, Optional[float]] = {}
        
        # Mean Absolute Error
        metrics['mae'] = float(mean_absolute_error(y_true, y_pred))
        
        # Mean Squared Error and Root Mean Squared Error
        mse = mean_squared_error(y_true, y_pred)
        metrics['mse'] = float(mse)
        metrics['rmse'] = float(np.sqrt(mse))
        
        # R-squared
        try:
            metrics['r2'] = float(r2_score(y_true, y_pred))
        except Exception:
            metrics['r2'] = None
        
        # Mean Absolute Percentage Error
        try:
            metrics['mape'] = float(mean_absolute_percentage_error(y_true, y_pred))
        except Exception:
            metrics['mape'] = None
        
        return metrics
    
    def get_confusion_matrix(
        self,
        y_true: Union[np.ndarray, pd.Series, List],
        y_pred: Union[np.ndarray, pd.Series, List],
        labels: Optional[List] = None,
        normalize: Optional[str] = None
    ) -> np.ndarray:
        """
        Get confusion matrix for classification problems.
        
        Parameters
        ----------
        y_true : array-like
            True labels.
        y_pred : array-like
            Predicted labels.
        labels : list, optional
            List of label values to order the matrix. If None, uses sorted unique labels.
        normalize : {'true', 'pred', 'all'}, optional
            Normalize confusion matrix.
        
        Returns
        -------
        np.ndarray
            Confusion matrix.
        
        Raises
        ------
        ValueError
            If called on regression problem.
        """
        if self.problem_type != 'classification':
            raise ValueError("Confusion matrix is only available for classification problems")
        
        y_true = np.asarray(y_true)
        y_pred = np.asarray(y_pred)
        
        return confusion_matrix(y_true, y_pred, labels=labels, normalize=normalize)
    
    def get_classification_report(
        self,
        y_true: Union[np.ndarray, pd.Series, List],
        y_pred: Union[np.ndarray, pd.Series, List],
        target_names: Optional[List[str]] = None,
        output_dict: bool = False
    ) -> Union[str, Dict]:
        """
        Get detailed classification report.
        
        Parameters
        ----------
        y_true : array-like
            True labels.
        y_pred : array-like
            Predicted labels.
        target_names : list of str, optional
            Display names matching the labels.
        output_dict : bool, default=False
            If True, return output as dict instead of string.
        
        Returns
        -------
        str or dict
            Classification report.
        
        Raises
        ------
        ValueError
            If called on regression problem.
        """
        if self.problem_type != 'classification':
            raise ValueError("Classification report is only available for classification problems")
        
        y_true = np.asarray(y_true)
        y_pred = np.asarray(y_pred)
        
        return classification_report(
            y_true, y_pred, target_names=target_names, output_dict=output_dict, zero_division=0
        )


def calculate_metrics(
    y_true: Union[np.ndarray, pd.Series, List],
    y_pred: Union[np.ndarray, pd.Series, List],
    problem_type: str,
    y_prob: Optional[Union[np.ndarray, pd.DataFrame]] = None,
    **kwargs
) -> Dict[str, float]:
    """
    Convenience function to calculate metrics.
    
    Parameters
    ----------
    y_true : array-like
        True target values.
    y_pred : array-like
        Predicted target values.
    problem_type : {'classification', 'regression'}
        Type of machine learning problem.
    y_prob : array-like, optional
        Predicted probabilities for classification.
    **kwargs : dict
        Additional keyword arguments.
    
    Returns
    -------
    dict
        Dictionary containing computed metrics.
    
    Examples
    --------
    >>> from skyulf_mlflow_library.modeling import calculate_metrics
    >>> y_true = [0, 1, 1, 0]
    >>> y_pred = [0, 1, 0, 0]
    >>> metrics = calculate_metrics(y_true, y_pred, 'classification')
    >>> print(metrics['accuracy'])
    0.75
    """
    calculator = MetricsCalculator(problem_type)
    return calculator.calculate(y_true, y_pred, y_prob, **kwargs)
