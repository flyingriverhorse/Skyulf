"""Edit this file to choose Skyulf steps and define your own fit/apply logic.

Training snapshots this self-contained file into the model artifact. Keep imports
to installed packages; sibling project modules are not packaged automatically.
Never train at module import time. Keep learned values in returned fit state.
"""

import polars as pl

from skyulf.inference.project_code import custom_step
from skyulf.preprocessing.base import BaseApplier, BaseCalculator, apply_method, fit_method


def build_preprocessing():
    """Return steps in execution order; change column names to match your input."""
    return [
        # {"name": "impute", "transformer": "SimpleImputer",
        #  "params": {"columns": ["feature_value"], "strategy": "mean"}},
        # {"name": "scale", "transformer": "StandardScaler",
        #  "params": {"columns": ["feature_value"]}},
        # custom_step("center", CenterCalculator, CenterApplier,
        #             params={"column": "feature_value"}),
    ]


class CenterCalculator(BaseCalculator):
    """Example custom fit: learn a numeric mean from this training partition only."""

    @fit_method
    def fit(self, X, y, config):
        """Return learned state; CV calls this separately inside every fold."""
        frame = X.to_native() if hasattr(X, "to_native") else X
        column = config["column"]
        return {"column": column, "mean": float(frame[column].mean())}


class CenterApplier(BaseApplier):
    """Example custom apply: reuse the saved mean without learning from new rows."""

    @apply_method
    def apply(self, X, y, params):
        """Preserve engine, row order and row count while transforming one column."""
        frame = X.to_native() if hasattr(X, "to_native") else X
        column = params["column"]
        if isinstance(frame, pl.DataFrame):
            return frame.with_columns((pl.col(column) - params["mean"]).alias(column))
        return frame.assign(**{column: frame[column] - params["mean"]})


def example_custom_step(column):
    """Build the example without enabling it in the default preprocessing chain."""
    return custom_step("center", CenterCalculator, CenterApplier, params={"column": column})
