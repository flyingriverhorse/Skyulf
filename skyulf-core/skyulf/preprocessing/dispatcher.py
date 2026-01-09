import logging
from typing import Any, Callable, Dict, Optional, Tuple, Union

import pandas as pd

from ..utils import unpack_pipeline_input, pack_pipeline_output
from ..engines import get_engine, SkyulfDataFrame

logger = logging.getLogger(__name__)

# Type definitions for the processing functions
# They receive (X, y, params)
# Apply returns (X_transformed, y_transformed)
ApplyFunction = Callable[[Any, Optional[Any], Dict[str, Any]], Tuple[Any, Optional[Any]]]
# Fit returns configuration dict
FitFunction = Callable[[Any, Optional[Any], Dict[str, Any]], Dict[str, Any]]


def apply_dual_engine(
    df: Union[pd.DataFrame, SkyulfDataFrame, Tuple[Any, ...], Any],
    params: Dict[str, Any],
    polars_func: ApplyFunction,
    pandas_func: ApplyFunction,
) -> Union[pd.DataFrame, SkyulfDataFrame, Tuple[Any, ...]]:
    """
    Dispatcher to handle boilerplate for dual-engine Appliers.

    Args:
        df: Input data (DataFrame or Tuple).
        params: Configuration parameters.
        polars_func: Function to execute if engine is Polars. 
                     Signature: (X, y, params) -> (X_out, y_out)
        pandas_func: Function to execute if engine is Pandas.
                     Signature: (X, y, params) -> (X_out, y_out)
                     Note: Input X is guaranteed to be a Pandas DataFrame/Series here.

    Returns:
        Packed output matching the input format.
    """
    X, y, is_tuple = unpack_pipeline_input(df)
    engine = get_engine(X)

    if engine.name == "polars":
        # Polars path
        # We pass X directly. The func should handle typing (X_pl: Any = X)
        try:
            X_out, y_out = polars_func(X, y, params)
        except Exception as e:
            # logger.error(f"Polars Engine Apply Failed: {e}")
            raise e
    else:
        # Pandas path
        # Ensure X is pandas
        if hasattr(X, "to_pandas"):
            X_pd = X.to_pandas()
        else:
            X_pd = X
            
        try:
            X_out, y_out = pandas_func(X_pd, y, params)
        except Exception as e:
            # logger.error(f"Pandas Engine Apply Failed: {e}")
            raise e

    return pack_pipeline_output(X_out, y_out, is_tuple)


def fit_dual_engine(
    df: Union[pd.DataFrame, SkyulfDataFrame, Tuple[Any, ...], Any],
    params: Dict[str, Any],
    polars_func: FitFunction,
    pandas_func: FitFunction,
) -> Dict[str, Any]:
    """
    Dispatcher to handle boilerplate for dual-engine Calculators.

    Args:
        df: Inputs.
        params: Config.
        polars_func: (X, y, params) -> Dict[Result]
        pandas_func: (X, y, params) -> Dict[Result]

    Returns:
        Dictionary of fitted parameters.
    """
    X, y, _ = unpack_pipeline_input(df)
    engine = get_engine(X)

    if engine.name == "polars":
        try:
            return polars_func(X, y, params)
        except Exception as e:
            # logger.error(f"Polars Engine Fit Failed: {e}")
            raise e
    else:
        if hasattr(X, "to_pandas"):
            X_pd = X.to_pandas()
        else:
            X_pd = X
        try:
            return pandas_func(X_pd, y, params)
        except Exception as e:
            # logger.error(f"Pandas Engine Fit Failed: {e}")
            raise e
