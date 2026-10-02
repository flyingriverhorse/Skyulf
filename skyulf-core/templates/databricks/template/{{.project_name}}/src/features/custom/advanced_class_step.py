"""ADVANCED: a custom step written as a Calculator + Applier class pair.

You normally do NOT need this. preprocessing_custom.py has the exact same
step, rare_categories(), written with two plain functions (learn + apply).
Compare both to see the difference:

  Function version (preprocessing_custom.py)   Class version (this file)
  ------------------------------------------   -----------------------------------
  learn(df, y)  -> dict                        Calculator.fit(X, y, config) -> dict
  apply(df, state) -> the changed column       Applier.apply(X, y, state) -> whole table
  df is always pandas                          X can be pandas OR polars: handle both
  Skyulf puts the column back for you          you rebuild the table yourself

What the class gives you in return: native Polars code (see _apply_polars),
which avoids a pandas copy on large data. Use a class only when you need that,
or when the functions cannot do the job:
  - the learned state is not a small dict (e.g. a fitted scikit-learn object),
  - the step must remove columns (like one-hot encoding) or add/remove rows.

How to use: in preprocessing.py import class_rare_categories and add
class_rare_categories("city", min_share=0.05) to a recipe list.
"""

import polars as pl

from skyulf.inference.project_code import custom_step
from skyulf.preprocessing.base import BaseApplier, BaseCalculator, apply_method, fit_method


class RareCategoryCalculator(BaseCalculator):
    """fit(): runs on the training rows of every CV fold and returns what it learned."""

    @fit_method
    def fit(self, X, y, config):
        """Learn the categories covering at least min_share of the training rows."""
        native = X.to_native() if hasattr(X, "to_native") else X
        values = native[config["column"]]
        values = values.to_pandas() if isinstance(values, pl.Series) else values
        shares = values.value_counts() / len(values)
        common = shares[shares >= config["min_share"]].index
        return {
            "column": config["column"],
            "other": config["other"],
            "keep": sorted(str(category) for category in common),
        }


class RareCategoryApplier(BaseApplier):
    """apply(): runs on train, test and scoring rows using only the saved fit() result."""

    @apply_method
    def apply(self, X, y, state):
        """Replace categories not in the saved list with `other`; nulls stay null."""
        native = X.to_native() if hasattr(X, "to_native") else X
        if isinstance(native, pl.DataFrame):
            return _apply_polars(native, state)
        return _apply_pandas(native, state)


def _apply_polars(frame, state):
    """Native Polars expression: no conversion to pandas."""
    column = pl.col(state["column"])
    keep = column.is_null() | column.cast(pl.Utf8).is_in(state["keep"])
    replaced = pl.when(keep).then(column.cast(pl.Utf8)).otherwise(pl.lit(state["other"]))
    return frame.with_columns(replaced.alias(state["column"]))


def _apply_pandas(frame, state):
    """Pandas version; returns a copy so the caller's table is never changed."""
    values = frame[state["column"]].astype(object)
    keep = values.isna() | values.astype(str).isin(state["keep"])
    result = frame.copy()
    result[state["column"]] = values.where(keep, state["other"])
    return result


def class_rare_categories(column, min_share=0.05, other="Other"):
    """Build the step: replace rare categories in column with `other`."""
    if not 0 < min_share < 1:
        raise ValueError("min_share must be between 0 and 1, e.g. 0.05 for 5%.")
    return custom_step(
        f"class_rare_{column}",
        RareCategoryCalculator,
        RareCategoryApplier,
        params={"column": column, "min_share": min_share, "other": other},
    )
