"""
Pipeline module for Skyulf MLflow.

This module provides a convenient pipeline interface for chaining transformers
and building end-to-end ML workflows, compatible with scikit-learn.
"""

from typing import List, Tuple

from sklearn.pipeline import Pipeline as SklearnPipeline

from ..core.base import BaseTransformer


class Pipeline(SklearnPipeline):
    """
    Pipeline for chaining transformers.
    
    This is a thin wrapper around sklearn's Pipeline that works seamlessly
    with Skyulf MLflow transformers. All transformers must implement fit()
    and transform() methods.
    
    Parameters
    ----------
    steps : list of tuples
        List of (name, transformer) tuples that are chained in order.
    memory : str or object, optional
        Used to cache the fitted transformers of the pipeline.
    verbose : bool, default=False
        If True, print elapsed time while fitting each step.
    
    Examples
    --------
    >>> from skyulf_mlflow_library.pipeline import Pipeline
    >>> from skyulf_mlflow_library.preprocessing import StandardScaler
    >>> from skyulf_mlflow_library.features import OneHotEncoder
    >>> import pandas as pd
    >>> 
    >>> # Create sample data
    >>> df = pd.DataFrame({
    ...     'age': [25, 30, 35, 40],
    ...     'city': ['NYC', 'LA', 'NYC', 'LA']
    ... })
    >>> 
    >>> # Build pipeline
    >>> pipeline = Pipeline([
    ...     ('encoder', OneHotEncoder(columns=['city'])),
    ...     ('scaler', StandardScaler(columns=['age']))
    ... ])
    >>> 
    >>> # Fit and transform
    >>> result = pipeline.fit_transform(df)
    >>> print(result.head())
    """
    
    def __init__(
        self,
        steps: List[Tuple[str, BaseTransformer]],
        memory=None,
        verbose: bool = False
    ):
        """
        Initialize the pipeline.
        
        Parameters
        ----------
        steps : list of tuples
            List of (name, transformer) tuples.
        memory : str or object, optional
            Caching parameter.
        verbose : bool, default=False
            Verbosity flag.
        """
        super().__init__(steps=steps, memory=memory, verbose=verbose)


def make_pipeline(*steps, **kwargs) -> Pipeline:
    """
    Construct a Pipeline from transformers.
    
    This is a convenience function that creates a Pipeline with automatic
    naming of steps based on transformer class names.
    
    Parameters
    ----------
    *steps : BaseTransformer
        Variable number of transformers.
    **kwargs : dict
        Additional keyword arguments for Pipeline (memory, verbose).
    
    Returns
    -------
    Pipeline
        A Pipeline object.
    
    Examples
    --------
    >>> from skyulf_mlflow_library.pipeline import make_pipeline
    >>> from skyulf_mlflow_library.preprocessing import StandardScaler, SimpleImputer
    >>> 
    >>> # Create pipeline with automatic naming
    >>> pipeline = make_pipeline(
    ...     SimpleImputer(strategy='mean'),
    ...     StandardScaler()
    ... )
    """
    # Generate automatic names
    named_steps = []
    for i, step in enumerate(steps):
        name = step.__class__.__name__.lower()
        # Add counter if name already exists
        base_name = name
        counter = 1
        while any(n == name for n, _ in named_steps):
            name = f"{base_name}_{counter}"
            counter += 1
        named_steps.append((name, step))
    
    return Pipeline(steps=named_steps, **kwargs)


__all__ = ['Pipeline', 'make_pipeline']
