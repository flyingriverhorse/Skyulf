"""Preprocessing module for Skyulf-MLFlow library."""

from skyulf_mlflow_library.preprocessing.cleaning import (
    drop_missing_rows,
    drop_missing_columns,
    remove_duplicates,
    remove_outliers,
    fill_missing,
)
from skyulf_mlflow_library.preprocessing.scaling import (
    StandardScaler,
    MinMaxScaler,
    RobustScaler,
    MaxAbsScaler,
)
from skyulf_mlflow_library.preprocessing.imputation import (
    SimpleImputer,
    KNNImputer,
)

# Import sampling methods (optional dependency)
try:
    from skyulf_mlflow_library.preprocessing.sampling import (
        SMOTE,
        RandomOverSampler,
        RandomUnderSampler,
    )
    _HAS_SAMPLING = True
except ImportError:
    _HAS_SAMPLING = False

# Import consistency module
from skyulf_mlflow_library.preprocessing import consistency

# Convenience aliases
drop_missing = drop_missing_rows

__all__ = [
    # Cleaning functions
    "drop_missing_rows",
    "drop_missing_columns",
    "drop_missing",
    "remove_duplicates",
    "remove_outliers",
    "fill_missing",
    # Scalers
    "StandardScaler",
    "MinMaxScaler",
    "RobustScaler",
    "MaxAbsScaler",
    # Imputers
    "SimpleImputer",
    "KNNImputer",
    # Consistency module
    "consistency",
]

# Add sampling methods to __all__ if available
if _HAS_SAMPLING:
    __all__.extend([
        "SMOTE",
        "RandomOverSampler",
        "RandomUnderSampler",
    ])
