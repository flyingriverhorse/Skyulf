from typing import Any, Optional, Tuple
import numpy as np
from .registry import get_engine


class SklearnBridge:
    """
    Bridge between Skyulf DataFrames (Pandas/Polars) and Scikit-Learn (Numpy).
    """

    @staticmethod
    def to_sklearn(X: Any) -> Tuple[np.ndarray, Any]:
        """
        Convert input to Numpy array for Scikit-Learn.

        Args:
            X: Input data (Pandas, Polars, Wrapper, or (X, y) tuple).

        Returns:
            Tuple (X_numpy, y_numpy_or_None)
        """
        y = None

        # Handle tuple (X, y)
        if isinstance(X, tuple):
            X_data, y_data = X
            y = SklearnBridge._convert_single(y_data)

            # Flatten y if it is (N, 1) - common requirement for Sklearn targets
            if y is not None and isinstance(y, np.ndarray) and y.ndim == 2 and y.shape[1] == 1:
                y = y.ravel()

            X = X_data

        X_numpy = SklearnBridge._convert_single(X)
        assert X_numpy is not None  # X is not None at this point
        return X_numpy, y

    @staticmethod
    def _convert_single(data: Any) -> Optional[np.ndarray]:
        if data is None:
            return None

        # If it's already numpy, return it
        if isinstance(data, np.ndarray):
            return data

        # Use engine to convert
        engine = get_engine(data)
        return engine.to_numpy(data)
