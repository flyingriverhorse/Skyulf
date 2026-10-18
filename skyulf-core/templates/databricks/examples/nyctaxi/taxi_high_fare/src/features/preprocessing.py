"""Fitted preprocessing steps selected in config/preprocessing.yml as preprocessing.<factory>."""

import pandas as pd

from skyulf.preprocessing import fitted_step


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
