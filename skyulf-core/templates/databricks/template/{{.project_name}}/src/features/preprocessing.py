"""YOUR OWN PREPROCESSING STEPS. Select them in config/preprocessing.yml.

A step is one or two plain pandas functions plus a small "factory" function
that wraps them. YAML selects the factory and passes params as keyword arguments:

  - custom: preprocessing.frequency_encoding
    params: {columns: [category]}

Built-in steps and ordered lists live only in config/preprocessing.yml.
These functions learn from training rows after the split; scoring reuses saved state.
See PREPROCESSING.md for a complete example.

Two kinds of step:

  column_step  - nothing to learn. One function:
                   fn(df) -> the new column
                 Example below: log_feature.

  fitted_step  - something to learn from training data (a mean, a mapping...).
                 Two functions:
                   learn(df, y)     -> dict   runs on TRAINING rows only
                   apply(df, state) -> column runs on train, test AND scoring rows,
                                       state is the dict learn() returned
                 Skyulf saves the dict with the model, so scoring uses exactly
                 what was learned in training. Examples: frequency_encoding,
                 rare_categories.

Rules:
  - Write normal top-level `def` functions (no lambda).
  - df is a pandas copy (also for Polars models); return one value per row.
  - learn() returns a plain dict: string keys, numbers/strings/lists, no NaN.
  - params={...} is passed to your functions as their last argument.
  - replace=True overwrites an existing column; otherwise output must be new.

custom/advanced_class_step.py shows rare_categories written as a class pair
(Calculator/Applier), so you can compare both ways.
"""

import numpy as np
import pandas as pd

from skyulf.preprocessing import column_step, fitted_step

# ---------------------------------------------------------------------------
# Example 1 - column_step: a new column, nothing learned.
#   log_feature("income")  ->  adds log_income = log(1 + income)
# ---------------------------------------------------------------------------


def log1p_value(df, params):
    """Return log(1 + value); negative values count as 0."""
    return np.log1p(df[params["column"]].clip(lower=0))


def log_feature(column):
    """Add log_<column> next to the original column."""
    return column_step(
        f"log_{column}", log1p_value, output=f"log_{column}", params={"column": column}
    )


# ---------------------------------------------------------------------------
# Example 2 - fitted_step without the target: frequency encoding.
#   training city: A, A, B, null  ->  learns {"A": 0.5, "B": 0.25}
#   scoring  city: A, NEW, null   ->  0.5, 0.0, 0.0  (replaces the column)
# ---------------------------------------------------------------------------


def learn_frequencies(df, y, params):
    """Learn: share of training rows per category (null is not a category)."""
    if len(df) == 0:
        raise ValueError("Frequency encoding requires nonempty training rows.")
    return {column: (df[column].value_counts() / len(df)).to_dict() for column in params["columns"]}


def apply_frequencies(df, state, params):
    """Apply: look up the saved share; unseen or null categories get 0."""
    return pd.DataFrame(
        {
            column: df[column].map(state[column]).astype(float).fillna(0.0)
            for column in params["columns"]
        }
    )


def frequency_encoding(columns):
    """Replace string category columns with their training frequency."""
    if not isinstance(columns, (list, tuple)) or not columns:
        raise ValueError("Frequency columns must be a nonempty list of column names.")
    if any(not isinstance(column, str) or not column for column in columns):
        raise ValueError("Frequency columns must be nonempty column names.")
    if len(set(columns)) != len(columns):
        raise ValueError("Frequency columns must be unique.")
    columns = list(columns)
    return fitted_step(
        "frequency_encoding",
        learn_frequencies,
        apply_frequencies,
        output=columns,
        replace=True,
        params={"columns": columns},
    )


# ---------------------------------------------------------------------------
# Example 3 - fitted_step that cleans a text column: group rare categories.
# The same step is written as a class pair in custom/advanced_class_step.py.
#   rare_categories("city", min_share=0.05)
#   training city: 60% London, 38% Paris, 2% Oslo  ->  learns ["London", "Paris"]
#   scoring  city: London, Oslo, Tokyo, null        ->  London, Other, Other, null
#   Fewer, more stable categories help one-hot or frequency encoding afterwards,
#   and a city never seen in training cannot break scoring.
# ---------------------------------------------------------------------------


def learn_common_categories(df, y, params):
    """Learn: categories that cover at least min_share of the training rows."""
    shares = df[params["column"]].value_counts() / len(df)
    common = shares[shares >= params["min_share"]].index
    return {"keep": sorted(str(category) for category in common)}


def apply_common_categories(df, state, params):
    """Apply: keep learned categories and nulls; everything else becomes `other`."""
    values = df[params["column"]].astype(object)  # object: "Other" is allowed as a value
    keep = values.isna() | values.astype(str).isin(state["keep"])
    return values.where(keep, params["other"])


def rare_categories(column, min_share=0.05, other="Other"):
    """Replace categories seen in less than min_share of training rows with `other`."""
    if not 0 < min_share < 1:
        raise ValueError("min_share must be between 0 and 1, e.g. 0.05 for 5%.")
    params = {"column": column, "min_share": min_share, "other": other}
    return fitted_step(
        f"rare_{column}",
        learn_common_categories,
        apply_common_categories,
        output=column,
        replace=True,
        params=params,
    )


# ---------------------------------------------------------------------------
# Example 4 - using a data file (asset). Inactive: uncomment to try.
#   1. Create src/features/assets/city_region.json  {"London": "UK", "Vilnius": "LT"}
#   2. In src/features/assets.json set   "files": ["assets/city_region.json"]
#   3. Uncomment the code below, then select custom: preprocessing.city_region in YAML.
#   The file is saved with the model; editing it later needs a new training run.
# ---------------------------------------------------------------------------

# import json
#
# from skyulf.inference.project_package import read_project_asset
#
#
# def region_of_city(df):
#     """Map each city to its region with the saved lookup file."""
#     regions = json.loads(read_project_asset(__package__, "assets/city_region.json"))
#     return df["city"].map(regions)
#
#
# def city_region():
#     """Add a region column from the city column."""
#     return column_step("city_region", region_of_city, output="region")
