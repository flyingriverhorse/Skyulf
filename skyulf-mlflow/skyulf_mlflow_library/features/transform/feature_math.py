"""
Advanced feature engineering through mathematical operations.

This module provides the FeatureMath transformer for creating new features
through arithmetic, ratio, statistical, similarity, and datetime operations.
"""

from typing import Any, Dict, List, Literal, Optional, Union
import warnings

import numpy as np
import pandas as pd

from ...core.base import BaseTransformer
from ...core.exceptions import FeatureEngineeringError

# Try to import optional dependency
try:
    from rapidfuzz import fuzz
    _HAS_RAPIDFUZZ = True
except ImportError:
    fuzz = None
    _HAS_RAPIDFUZZ = False

from difflib import SequenceMatcher


DEFAULT_EPSILON = 1e-9


class FeatureMath(BaseTransformer):
    """
    Create new features through mathematical operations.
    
    Supports arithmetic, ratio, statistical, text similarity, and datetime
    feature extraction operations. Each operation can involve multiple columns
    and produces one or more output features.
    
    Parameters
    ----------
    operations : list of dict
        List of operation configurations. Each dict must have:
        - 'type': Operation type ('arithmetic', 'ratio', 'stat', 'similarity', 'datetime')
        - 'columns': Input column(s)
        - 'output': Output column name
        
        Additional parameters vary by operation type.
    error_handling : {'skip', 'raise'}, default='skip'
        How to handle operation errors:
        - 'skip': Skip failed operations and continue
        - 'raise': Raise exception on first error
    epsilon : float, default=1e-9
        Small value to prevent division by zero.
        
    Examples
    --------
    >>> # Arithmetic operations
    >>> df = pd.DataFrame({'price': [100, 200], 'tax': [10, 20]})
    >>> ops = [{
    ...     'type': 'arithmetic',
    ...     'method': 'add',
    ...     'columns': ['price', 'tax'],
    ...     'output': 'total'
    ... }]
    >>> fm = FeatureMath(operations=ops)
    >>> result = fm.fit_transform(df)
    >>> print(result['total'])
    0    110
    1    220
    
    >>> # Ratio operations
    >>> df = pd.DataFrame({'sales': [100, 200], 'cost': [80, 150]})
    >>> ops = [{
    ...     'type': 'ratio',
    ...     'numerator': ['sales'],
    ...     'denominator': ['cost'],
    ...     'output': 'margin'
    ... }]
    >>> fm = FeatureMath(operations=ops)
    >>> result = fm.fit_transform(df)
    >>> print(result['margin'].round(2))
    0    1.25
    1    1.33
    
    >>> # Statistical operations
    >>> df = pd.DataFrame({'q1': [10, 20], 'q2': [15, 25], 'q3': [20, 30]})
    >>> ops = [{
    ...     'type': 'stat',
    ...     'method': 'mean',
    ...     'columns': ['q1', 'q2', 'q3'],
    ...     'output': 'avg_quarterly'
    ... }]
    >>> fm = FeatureMath(operations=ops)
    >>> result = fm.fit_transform(df)
    >>> print(result['avg_quarterly'])
    0    15.0
    1    25.0
    
    >>> # Text similarity (requires rapidfuzz or uses fallback)
    >>> df = pd.DataFrame({
    ...     'name': ['John Doe', 'Jane Smith'],
    ...     'alt_name': ['John D', 'Jane S']
    ... })
    >>> ops = [{
    ...     'type': 'similarity',
    ...     'columns': ['name', 'alt_name'],
    ...     'output': 'name_similarity'
    ... }]
    >>> fm = FeatureMath(operations=ops)
    >>> result = fm.fit_transform(df)
    
    >>> # DateTime feature extraction
    >>> df = pd.DataFrame({
    ...     'order_date': pd.to_datetime(['2024-01-15', '2024-06-20'])
    ... })
    >>> ops = [{
    ...     'type': 'datetime',
    ...     'columns': ['order_date'],
    ...     'features': ['year', 'month', 'quarter', 'is_weekend'],
    ...     'prefix': 'order_'
    ... }]
    >>> fm = FeatureMath(operations=ops)
    >>> result = fm.fit_transform(df)
    
    Notes
    -----
    - Arithmetic methods: add, subtract, multiply, divide
    - Statistical methods: sum, mean, min, max, std, median, count, range
    - Similarity methods: token_sort_ratio, token_set_ratio, ratio
    - DateTime features: year, quarter, month, month_name, week, day, day_name,
      weekday, is_weekend, hour, minute, second, season, time_of_day
    - Division by zero is handled using epsilon value
    - Missing values can be filled with fillna parameter per operation
    - Rapidfuzz library recommended for faster text similarity (optional)
    """
    
    def __init__(
        self,
        operations: List[Dict[str, Any]],
        error_handling: Literal['skip', 'raise'] = 'skip',
        epsilon: float = DEFAULT_EPSILON,
    ):
        super().__init__()
        self.operations = operations
        self.error_handling = error_handling
        self.epsilon = epsilon
        self._created_features: List[str] = []
        
    def fit(self, X: pd.DataFrame, y: Optional[pd.Series] = None) -> 'FeatureMath':
        """
        Fit the transformer (validates operations).
        
        Parameters
        ----------
        X : pd.DataFrame
            Input data.
        y : pd.Series, optional
            Target variable (not used, present for API consistency).
            
        Returns
        -------
        self
            Fitted transformer.
        """
        if not isinstance(X, pd.DataFrame):
            raise FeatureEngineeringError("X must be a pandas DataFrame")
        
        # Validate operations
        if not isinstance(self.operations, list):
            raise FeatureEngineeringError("operations must be a list")
        
        if not self.operations:
            raise FeatureEngineeringError("operations list cannot be empty")
        
        # Validate each operation has required keys
        for i, op in enumerate(self.operations):
            if not isinstance(op, dict):
                raise FeatureEngineeringError(
                    f"Operation {i} must be a dictionary"
                )
            if 'type' not in op:
                raise FeatureEngineeringError(
                    f"Operation {i} missing required 'type' key"
                )
            if op['type'] not in {'arithmetic', 'ratio', 'stat', 'similarity', 'datetime'}:
                raise FeatureEngineeringError(
                    f"Operation {i} has invalid type: {op['type']}"
                )
        
        from ...core.types import TransformerState
        self.state = TransformerState.FITTED
        return self
    
    def transform(self, X: pd.DataFrame) -> pd.DataFrame:
        """
        Transform data by applying all operations.
        
        Parameters
        ----------
        X : pd.DataFrame
            Input data to transform.
            
        Returns
        -------
        pd.DataFrame
            Transformed data with new features.
        """
        self._check_is_fitted()
        if not isinstance(X, pd.DataFrame):
            raise FeatureEngineeringError("X must be a pandas DataFrame")
        
        X_transformed = X.copy()
        self._created_features = []
        
        for i, op in enumerate(self.operations):
            try:
                op_type = op['type']
                
                if op_type == 'arithmetic':
                    self._apply_arithmetic(X_transformed, op)
                elif op_type == 'ratio':
                    self._apply_ratio(X_transformed, op)
                elif op_type == 'stat':
                    self._apply_stat(X_transformed, op)
                elif op_type == 'similarity':
                    self._apply_similarity(X_transformed, op)
                elif op_type == 'datetime':
                    self._apply_datetime(X_transformed, op)
                    
            except Exception as e:
                if self.error_handling == 'raise':
                    raise FeatureEngineeringError(
                        f"Operation {i} ({op.get('type')}) failed: {e}"
                    )
                # Skip this operation and continue
                warnings.warn(
                    f"Operation {i} ({op.get('type')}) failed: {e}. Skipping.",
                    UserWarning
                )
        
        return X_transformed
    
    def _apply_arithmetic(self, df: pd.DataFrame, op: Dict[str, Any]) -> None:
        """Apply arithmetic operation (add, subtract, multiply, divide)."""
        method = op.get('method', 'add')
        columns = op.get('columns', [])
        constants = op.get('constants', [])
        output = op['output']
        fillna = op.get('fillna')
        round_digits = op.get('round')
        
        if not columns and not constants:
            raise FeatureEngineeringError("Arithmetic requires columns or constants")
        
        # Prepare numeric series
        series_list = []
        for col in columns:
            if col not in df.columns:
                raise FeatureEngineeringError(f"Column '{col}' not found")
            s = pd.to_numeric(df[col], errors='coerce')
            if fillna is not None:
                s = s.fillna(fillna)
            series_list.append(s)
        
        # Apply operation
        if method == 'add':
            result = pd.Series(0.0, index=df.index)
            for s in series_list:
                result = result + s
            for c in constants:
                result = result + c
                
        elif method == 'subtract':
            if series_list:
                result = series_list[0].copy()
                for s in series_list[1:]:
                    result = result - s
            else:
                result = pd.Series(0.0, index=df.index)
            for c in constants:
                result = result - c
                
        elif method == 'multiply':
            result = pd.Series(1.0, index=df.index)
            for s in series_list:
                result = result * s
            for c in constants:
                result = result * c
                
        elif method == 'divide':
            if series_list:
                result = series_list[0].copy()
                for s in series_list[1:]:
                    # Prevent division by zero
                    denominator = s.replace(0, self.epsilon)
                    result = result / denominator
            else:
                if not constants:
                    raise FeatureEngineeringError("Division requires operands")
                result = pd.Series(constants[0], index=df.index)
                constants = constants[1:]
            for c in constants:
                result = result / (c if c != 0 else self.epsilon)
        else:
            raise FeatureEngineeringError(f"Unknown arithmetic method: {method}")
        
        # Apply rounding
        if round_digits is not None:
            result = result.round(round_digits)
        
        # Fill NA if specified
        if fillna is not None:
            result = result.fillna(fillna)
        
        df[output] = result
        self._created_features.append(output)
    
    def _apply_ratio(self, df: pd.DataFrame, op: Dict[str, Any]) -> None:
        """Apply ratio operation (numerator / denominator)."""
        numerator_cols = op.get('numerator', op.get('columns', []))
        denominator_cols = op.get('denominator', op.get('secondary_columns', []))
        output = op['output']
        fillna = op.get('fillna')
        round_digits = op.get('round')
        
        if not numerator_cols or not denominator_cols:
            raise FeatureEngineeringError("Ratio requires numerator and denominator")
        
        # Sum numerator columns
        numerator = pd.Series(0.0, index=df.index)
        for col in numerator_cols:
            if col not in df.columns:
                raise FeatureEngineeringError(f"Column '{col}' not found")
            s = pd.to_numeric(df[col], errors='coerce')
            if fillna is not None:
                s = s.fillna(fillna)
            numerator = numerator + s
        
        # Sum denominator columns
        denominator = pd.Series(0.0, index=df.index)
        for col in denominator_cols:
            if col not in df.columns:
                raise FeatureEngineeringError(f"Column '{col}' not found")
            s = pd.to_numeric(df[col], errors='coerce')
            if fillna is not None:
                s = s.fillna(fillna)
            denominator = denominator + s
        
        # Prevent division by zero
        denominator = denominator.replace(0, self.epsilon)
        result = numerator / denominator
        
        if round_digits is not None:
            result = result.round(round_digits)
        
        if fillna is not None:
            result = result.fillna(fillna)
        
        df[output] = result
        self._created_features.append(output)
    
    def _apply_stat(self, df: pd.DataFrame, op: Dict[str, Any]) -> None:
        """Apply statistical operation (sum, mean, min, max, etc.)."""
        method = op.get('method', 'sum')
        columns = op.get('columns', [])
        output = op['output']
        fillna = op.get('fillna')
        round_digits = op.get('round')
        
        if not columns:
            raise FeatureEngineeringError("Statistical operations require columns")
        
        # Prepare series
        series_list = []
        for col in columns:
            if col not in df.columns:
                raise FeatureEngineeringError(f"Column '{col}' not found")
            s = pd.to_numeric(df[col], errors='coerce')
            if fillna is not None:
                s = s.fillna(fillna)
            series_list.append(s)
        
        # Combine into dataframe for easier operations
        combined = pd.concat(series_list, axis=1)
        
        # Apply statistical method
        if method == 'sum':
            result = combined.sum(axis=1, skipna=True)
        elif method == 'mean':
            result = combined.mean(axis=1, skipna=True)
        elif method == 'min':
            result = combined.min(axis=1, skipna=True)
        elif method == 'max':
            result = combined.max(axis=1, skipna=True)
        elif method == 'std':
            result = combined.std(axis=1, ddof=0)
        elif method == 'median':
            result = combined.median(axis=1, skipna=True)
        elif method == 'count':
            result = combined.count(axis=1)
        elif method == 'range':
            result = combined.max(axis=1, skipna=True) - combined.min(axis=1, skipna=True)
        else:
            raise FeatureEngineeringError(f"Unknown stat method: {method}")
        
        if round_digits is not None:
            result = result.round(round_digits)
        
        if fillna is not None:
            result = result.fillna(fillna)
        
        df[output] = result
        self._created_features.append(output)
    
    def _compute_similarity(self, a: Any, b: Any, method: str = 'ratio') -> float:
        """Compute text similarity score."""
        text_a = '' if pd.isna(a) else str(a)
        text_b = '' if pd.isna(b) else str(b)
        
        if not text_a and not text_b:
            return 100.0
        if not text_a or not text_b:
            return 0.0
        
        if _HAS_RAPIDFUZZ:
            if method == 'token_sort_ratio':
                return float(fuzz.token_sort_ratio(text_a, text_b))
            elif method == 'token_set_ratio':
                return float(fuzz.token_set_ratio(text_a, text_b))
            else:  # 'ratio'
                return float(fuzz.ratio(text_a, text_b))
        
        # Fallback to difflib
        return SequenceMatcher(None, text_a, text_b).ratio() * 100.0
    
    def _apply_similarity(self, df: pd.DataFrame, op: Dict[str, Any]) -> None:
        """Apply text similarity operation."""
        columns = op.get('columns', [])
        method = op.get('method', 'ratio')
        output = op['output']
        normalize = op.get('normalize', False)  # Divide by 100 to get 0-1 range
        round_digits = op.get('round')
        
        if len(columns) < 2:
            raise FeatureEngineeringError("Similarity requires at least 2 columns")
        
        col_a, col_b = columns[0], columns[1]
        if col_a not in df.columns or col_b not in df.columns:
            raise FeatureEngineeringError(f"Columns not found: {col_a}, {col_b}")
        
        # Compute similarity for each row
        result = df[col_a].combine(
            df[col_b],
            lambda a, b: self._compute_similarity(a, b, method)
        )
        
        if normalize:
            result = result / 100.0
        
        if round_digits is not None:
            result = result.round(round_digits)
        
        df[output] = result
        self._created_features.append(output)
    
    def _apply_datetime(self, df: pd.DataFrame, op: Dict[str, Any]) -> None:
        """Apply datetime feature extraction."""
        columns = op.get('columns', [])
        features = op.get('features', ['year', 'month', 'day'])
        prefix = op.get('prefix', '')
        
        if not columns:
            raise FeatureEngineeringError("Datetime extraction requires columns")
        
        for col in columns:
            if col not in df.columns:
                raise FeatureEngineeringError(f"Column '{col}' not found")
            
            # Convert to datetime
            if pd.api.types.is_datetime64_any_dtype(df[col]):
                dt_series = df[col]
            else:
                dt_series = pd.to_datetime(df[col], errors='coerce')
            
            # Extract features
            for feature in features:
                output_name = f"{prefix}{feature}"
                
                if feature == 'year':
                    df[output_name] = dt_series.dt.year
                elif feature == 'quarter':
                    df[output_name] = dt_series.dt.quarter
                elif feature == 'month':
                    df[output_name] = dt_series.dt.month
                elif feature == 'month_name':
                    df[output_name] = dt_series.dt.month_name()
                elif feature == 'week':
                    df[output_name] = dt_series.dt.isocalendar().week
                elif feature == 'day':
                    df[output_name] = dt_series.dt.day
                elif feature == 'day_name':
                    df[output_name] = dt_series.dt.day_name()
                elif feature == 'weekday':
                    df[output_name] = dt_series.dt.weekday
                elif feature == 'is_weekend':
                    df[output_name] = dt_series.dt.weekday.isin([5, 6]).astype(int)
                elif feature == 'hour':
                    df[output_name] = dt_series.dt.hour
                elif feature == 'minute':
                    df[output_name] = dt_series.dt.minute
                elif feature == 'second':
                    df[output_name] = dt_series.dt.second
                elif feature == 'season':
                    df[output_name] = dt_series.dt.month.map(self._get_season)
                elif feature == 'time_of_day':
                    df[output_name] = dt_series.dt.hour.map(self._get_time_of_day)
                else:
                    warnings.warn(f"Unknown datetime feature: {feature}", UserWarning)
                    continue
                
                self._created_features.append(output_name)
    
    @staticmethod
    def _get_season(month):
        """Map month to season."""
        if pd.isna(month):
            return None
        if month in [12, 1, 2]:
            return 'winter'
        elif month in [3, 4, 5]:
            return 'spring'
        elif month in [6, 7, 8]:
            return 'summer'
        elif month in [9, 10, 11]:
            return 'autumn'
        return None
    
    @staticmethod
    def _get_time_of_day(hour):
        """Map hour to time of day."""
        if pd.isna(hour):
            return None
        if 5 <= hour < 12:
            return 'morning'
        elif 12 <= hour < 17:
            return 'afternoon'
        elif 17 <= hour < 21:
            return 'evening'
        else:
            return 'night'
    
    def get_feature_names_out(self) -> List[str]:
        """
        Get names of created features.
        
        Returns
        -------
        list of str
            Names of features created during transform.
        """
        return self._created_features.copy()
