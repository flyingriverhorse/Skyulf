"""Edit this file to choose Skyulf steps and define your own fit/apply logic.

Training snapshots this self-contained file into the model artifact. Keep imports
to installed packages; sibling project modules are not packaged automatically.
Never train at module import time. Keep learned values in returned fit state.
"""

import polars as pl

from skyulf.inference.project_code import custom_step
from skyulf.preprocessing.base import BaseApplier, BaseCalculator, apply_method, fit_method


def build_pre_split_steps():
    """Declare fixed cleanup and training eligibility before the final split.

    Use explicit source columns. Fixed feature edits are saved and replayed once
    on raw model inputs; row filters only select training/evaluation rows. Keep
    learned preprocessing in build_preprocessing() below.
    """
    return [
        # {"name": "missing_sentinel", "transformer": "ValueReplacement",
        #  "params": {"columns": ["feature_value"], "to_replace": -999, "value": None}},
        # {"name": "known_target", "transformer": "DropMissingRows",
        #  "params": {"subset": ["target"], "how": "any"}},
        # {"name": "valid_age", "transformer": "ManualBounds",
        #  "params": {"bounds": {"age": {"lower": 0, "upper": 120}}}},
        # {"name": "unique_observations", "transformer": "Deduplicate",
        #  "params": {"subset": ["feature_value"], "keep": "first"}},
        # example_custom_pre_split("is_test"),
    ]


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


class EligibilityCalculator(BaseCalculator):
    """Example fixed rule: retain the declared value without learning statistics."""

    @fit_method
    def fit(self, X, y, config):
        """Return only the developer's fixed rule, never a statistic from the data."""
        return dict(config)


class EligibilityApplier(BaseApplier):
    """Example training filter: keep rows whose flag equals the declared value."""

    @apply_method
    def apply(self, X, y, params):
        """Keep existing rows in order without editing any column or target value."""
        frame = X.to_native() if hasattr(X, "to_native") else X
        column, value = params["column"], params["keep_value"]
        if isinstance(frame, pl.DataFrame):
            return frame.filter(pl.col(column) == value)
        return frame.loc[frame[column] == value]


def example_custom_pre_split(column):
    """Opt in to retaining known non-test rows; null or true flags are excluded.

    This declaration is the developer's assertion that no statistics are learned.
    Runtime guards check row/value preservation; they do not audit arbitrary code.
    Custom value transformations belong in build_preprocessing().
    """
    return custom_step(
        "non_test_rows",
        EligibilityCalculator,
        EligibilityApplier,
        params={"column": column, "keep_value": False},
        pre_split={
            "effect": "filter",
            "required_columns": [column],
            "learns_from_data": False,
        },
    )
