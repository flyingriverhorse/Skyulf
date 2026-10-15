"""Fitted preprocessing steps selected in config/preprocessing.yml as preprocessing.<factory>."""

import numpy as np
import pandas as pd

from skyulf.preprocessing import column_step, fitted_step


def log1p_value(df, params):
    """Return log(1 + value); negative values count as 0."""
    return np.log1p(df[params["column"]].clip(lower=0))


def log_feature(column):
    """Add log_<column> next to the original column."""
    return column_step(
        f"log_{column}", log1p_value, output=f"log_{column}", params={"column": column}
    )


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
