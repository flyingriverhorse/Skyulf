"""
Sampling methods for handling imbalanced datasets.

This module provides various sampling techniques including:
- SMOTE (Synthetic Minority Over-sampling Technique)
- Random over-sampling
- Random under-sampling
- Combined sampling strategies
"""

from typing import Optional

import pandas as pd

from ...core.base import BaseTransformer
from ...core.exceptions import DataProcessingError
from ...core.types import TransformerState

# Try to import imbalanced-learn
try:
    from imblearn.over_sampling import SMOTE as ImbSMOTE
    from imblearn.over_sampling import RandomOverSampler as ImbRandomOverSampler
    from imblearn.under_sampling import RandomUnderSampler as ImbRandomUnderSampler
    _HAS_IMBLEARN = True
except ImportError:
    ImbSMOTE = None
    ImbRandomOverSampler = None
    ImbRandomUnderSampler = None
    _HAS_IMBLEARN = False


class SMOTE(BaseTransformer):
    """
    Synthetic Minority Over-sampling Technique for imbalanced datasets.
    
    SMOTE creates synthetic samples for the minority class by interpolating
    between existing minority samples. This helps balance class distributions.
    
    **Requires**: imbalanced-learn library (`pip install imbalanced-learn`)
    
    Parameters
    ----------
    sampling_strategy : float, str, or dict, default='auto'
        Sampling information:
        - float: Ratio of minority to majority class after resampling
        - 'auto': Balance all classes
        - dict: {class_label: n_samples}
    k_neighbors : int, default=5
        Number of nearest neighbors to use for generating synthetic samples.
    random_state : int, optional
        Random seed for reproducibility.
    
    Examples
    --------
    >>> import pandas as pd
    >>> from skyulf_mlflow_library.preprocessing import SMOTE
    >>> 
    >>> # Imbalanced dataset
    >>> X = pd.DataFrame({
    ...     'feature1': [1, 2, 3, 4, 5, 6, 7, 8, 9, 10],
    ...     'feature2': [10, 20, 30, 40, 50, 60, 70, 80, 90, 100]
    ... })
    >>> y = pd.Series([0, 0, 0, 0, 0, 0, 0, 1, 1, 1])  # Imbalanced: 7 vs 3
    >>> 
    >>> smote = SMOTE(random_state=42)
    >>> X_resampled, y_resampled = smote.fit_resample(X, y)
    >>> print(f"Original: {len(y)}, Resampled: {len(y_resampled)}")
    >>> print(f"Class distribution: {y_resampled.value_counts().to_dict()}")
    """
    
    def __init__(
        self,
        sampling_strategy: str = 'auto',
        k_neighbors: int = 5,
        random_state: Optional[int] = None,
        **kwargs
    ):
        """
        Initialize SMOTE sampler.
        
        Parameters
        ----------
        sampling_strategy : str, float, or dict, default='auto'
            Sampling strategy.
        k_neighbors : int, default=5
            Number of nearest neighbors.
        random_state : int, optional
            Random seed.
        **kwargs : dict
            Additional parameters.
        """
        super().__init__(**kwargs)
        
        if not _HAS_IMBLEARN:
            raise ImportError(
                "SMOTE requires imbalanced-learn. "
                "Install it with: pip install imbalanced-learn"
            )
        
        self.sampling_strategy = sampling_strategy
        self.k_neighbors = k_neighbors
        self.random_state = random_state
        
        self._sampler = ImbSMOTE(
            sampling_strategy=sampling_strategy,
            k_neighbors=k_neighbors,
            random_state=random_state
        )
    
    def fit(self, X: pd.DataFrame, y: pd.Series) -> "SMOTE":
        """
        Fit the SMOTE sampler.
        
        Parameters
        ----------
        X : pd.DataFrame
            Input features.
        y : pd.Series
            Target variable.
        
        Returns
        -------
        self
            Fitted sampler.
        """
        if not isinstance(X, pd.DataFrame):
            raise DataProcessingError("X must be a pandas DataFrame")
        
        if y is None:
            raise DataProcessingError("y is required for SMOTE")
        
        # SMOTE is fit during transform, but we validate here
        self.state = TransformerState.FITTED
        return self
    
    def transform(self, X: pd.DataFrame) -> pd.DataFrame:
        """
        Not used for sampling - use fit_resample instead.
        """
        raise NotImplementedError(
            "Use fit_resample() method for SMOTE instead of transform()"
        )
    
    def fit_resample(self, X: pd.DataFrame, y: pd.Series):
        """
        Fit and resample the data.
        
        Parameters
        ----------
        X : pd.DataFrame
            Input features.
        y : pd.Series
            Target variable.
        
        Returns
        -------
        X_resampled : pd.DataFrame
            Resampled features.
        y_resampled : pd.Series
            Resampled target.
        """
        self._check_is_fitted()
        
        if not isinstance(X, pd.DataFrame):
            raise DataProcessingError("X must be a pandas DataFrame")
        
        X_res, y_res = self._sampler.fit_resample(X, y)
        
        # Convert back to DataFrame/Series
        X_resampled = pd.DataFrame(X_res, columns=X.columns)
        y_resampled = pd.Series(y_res, name=y.name if hasattr(y, 'name') else 'target')
        
        return X_resampled, y_resampled


class RandomOverSampler(BaseTransformer):
    """
    Random over-sampling by duplicating samples from minority classes.
    
    This is a simple oversampling technique that randomly duplicates samples
    from the minority class until the desired class distribution is achieved.
    
    **Requires**: imbalanced-learn library (`pip install imbalanced-learn`)
    
    Parameters
    ----------
    sampling_strategy : float, str, or dict, default='auto'
        Sampling information:
        - float: Ratio of minority to majority class after resampling
        - 'auto': Balance all classes
        - dict: {class_label: n_samples}
    random_state : int, optional
        Random seed for reproducibility.
    
    Examples
    --------
    >>> import pandas as pd
    >>> from skyulf_mlflow_library.preprocessing import RandomOverSampler
    >>> 
    >>> X = pd.DataFrame({'feature': [1, 2, 3, 4, 5]})
    >>> y = pd.Series([0, 0, 0, 1, 1])  # Imbalanced: 3 vs 2
    >>> 
    >>> ros = RandomOverSampler(random_state=42)
    >>> X_resampled, y_resampled = ros.fit_resample(X, y)
    >>> print(y_resampled.value_counts())
    """
    
    def __init__(
        self,
        sampling_strategy: str = 'auto',
        random_state: Optional[int] = None,
        **kwargs
    ):
        """
        Initialize random over-sampler.
        
        Parameters
        ----------
        sampling_strategy : str, float, or dict, default='auto'
            Sampling strategy.
        random_state : int, optional
            Random seed.
        **kwargs : dict
            Additional parameters.
        """
        super().__init__(**kwargs)
        
        if not _HAS_IMBLEARN:
            raise ImportError(
                "RandomOverSampler requires imbalanced-learn. "
                "Install it with: pip install imbalanced-learn"
            )
        
        self.sampling_strategy = sampling_strategy
        self.random_state = random_state
        
        self._sampler = ImbRandomOverSampler(
            sampling_strategy=sampling_strategy,
            random_state=random_state
        )
    
    def fit(self, X: pd.DataFrame, y: pd.Series) -> "RandomOverSampler":
        """Fit the sampler."""
        if not isinstance(X, pd.DataFrame):
            raise DataProcessingError("X must be a pandas DataFrame")
        
        if y is None:
            raise DataProcessingError("y is required for RandomOverSampler")
        
        self.state = TransformerState.FITTED
        return self
    
    def transform(self, X: pd.DataFrame) -> pd.DataFrame:
        """Not used - use fit_resample instead."""
        raise NotImplementedError(
            "Use fit_resample() method instead of transform()"
        )
    
    def fit_resample(self, X: pd.DataFrame, y: pd.Series):
        """
        Fit and resample the data.
        
        Parameters
        ----------
        X : pd.DataFrame
            Input features.
        y : pd.Series
            Target variable.
        
        Returns
        -------
        X_resampled : pd.DataFrame
            Resampled features.
        y_resampled : pd.Series
            Resampled target.
        """
        self.fit(X, y)
        
        X_res, y_res = self._sampler.fit_resample(X, y)
        
        X_resampled = pd.DataFrame(X_res, columns=X.columns)
        y_resampled = pd.Series(y_res, name=y.name if hasattr(y, 'name') else 'target')
        
        return X_resampled, y_resampled


class RandomUnderSampler(BaseTransformer):
    """
    Random under-sampling by removing samples from majority classes.
    
    This technique randomly removes samples from the majority class to balance
    the class distribution. May result in loss of information.
    
    **Requires**: imbalanced-learn library (`pip install imbalanced-learn`)
    
    Parameters
    ----------
    sampling_strategy : float, str, or dict, default='auto'
        Sampling information:
        - float: Ratio of minority to majority class after resampling
        - 'auto': Balance all classes
        - dict: {class_label: n_samples}
    random_state : int, optional
        Random seed for reproducibility.
    replacement : bool, default=False
        Whether to sample with replacement.
    
    Examples
    --------
    >>> import pandas as pd
    >>> from skyulf_mlflow_library.preprocessing import RandomUnderSampler
    >>> 
    >>> X = pd.DataFrame({'feature': [1, 2, 3, 4, 5, 6, 7]})
    >>> y = pd.Series([0, 0, 0, 0, 0, 1, 1])  # Imbalanced: 5 vs 2
    >>> 
    >>> rus = RandomUnderSampler(random_state=42)
    >>> X_resampled, y_resampled = rus.fit_resample(X, y)
    >>> print(y_resampled.value_counts())  # Should be 2 vs 2
    """
    
    def __init__(
        self,
        sampling_strategy: str = 'auto',
        random_state: Optional[int] = None,
        replacement: bool = False,
        **kwargs
    ):
        """
        Initialize random under-sampler.
        
        Parameters
        ----------
        sampling_strategy : str, float, or dict, default='auto'
            Sampling strategy.
        random_state : int, optional
            Random seed.
        replacement : bool, default=False
            Sample with replacement.
        **kwargs : dict
            Additional parameters.
        """
        super().__init__(**kwargs)
        
        if not _HAS_IMBLEARN:
            raise ImportError(
                "RandomUnderSampler requires imbalanced-learn. "
                "Install it with: pip install imbalanced-learn"
            )
        
        self.sampling_strategy = sampling_strategy
        self.random_state = random_state
        self.replacement = replacement
        
        self._sampler = ImbRandomUnderSampler(
            sampling_strategy=sampling_strategy,
            random_state=random_state,
            replacement=replacement
        )
    
    def fit(self, X: pd.DataFrame, y: pd.Series) -> "RandomUnderSampler":
        """Fit the sampler."""
        if not isinstance(X, pd.DataFrame):
            raise DataProcessingError("X must be a pandas DataFrame")
        
        if y is None:
            raise DataProcessingError("y is required for RandomUnderSampler")
        
        self.state = TransformerState.FITTED
        return self
    
    def transform(self, X: pd.DataFrame) -> pd.DataFrame:
        """Not used - use fit_resample instead."""
        raise NotImplementedError(
            "Use fit_resample() method instead of transform()"
        )
    
    def fit_resample(self, X: pd.DataFrame, y: pd.Series):
        """
        Fit and resample the data.
        
        Parameters
        ----------
        X : pd.DataFrame
            Input features.
        y : pd.Series
            Target variable.
        
        Returns
        -------
        X_resampled : pd.DataFrame
            Resampled features.
        y_resampled : pd.Series
            Resampled target.
        """
        self.fit(X, y)
        
        X_res, y_res = self._sampler.fit_resample(X, y)
        
        X_resampled = pd.DataFrame(X_res, columns=X.columns)
        y_resampled = pd.Series(y_res, name=y.name if hasattr(y, 'name') else 'target')
        
        return X_resampled, y_resampled
