"""YOUR OWN ROW FILTERS. Use them in features/pre_split.py.

A filter decides, row by row, which rows may be used for training.
Write one function that returns True for rows to KEEP, then wrap it:

  filter_step(name, fn, columns=[...])
      fn(df) -> True/False for every row
      columns = the columns fn reads (Skyulf checks they exist)

Rules:
  - Look only at the row itself (no df.mean(), df.duplicated() etc.).
    Something learned from data belongs in preprocessing, not here.
  - Return exactly one True/False per row. Missing values give no answer,
    so decide them yourself: (df["x"] > 0).fillna(False)
  - Never change values; filters only remove rows.
  - Write normal top-level `def` functions (no lambda).
  - params={...} is passed to your function as its last argument.

scoring.py can reuse these filters for scoring rows (SCORING_MODE="pre_split")
or "combined" in features/scoring.py
"""

from skyulf.preprocessing import filter_step

# ---------------------------------------------------------------------------
# Example 1 - keep rows with enough filled-in fields.
#   minimum_completeness(["a", "b", "c"], min_present=2)
#   a=1, b=null, c=3  -> kept (2 values)      a=null, b=null, c=3 -> removed
#   null/NaN count as missing; empty text and infinity count as values.
# ---------------------------------------------------------------------------


def has_enough_values(df, params):
    """True when at least min_present of the selected columns are not null."""
    return df[params["columns"]].notna().sum(axis=1) >= params["min_present"]


def minimum_completeness(columns, min_present=1):
    """Keep rows where at least min_present of the given columns have a value."""
    if not isinstance(columns, (list, tuple)) or not columns:
        raise ValueError("Completeness columns must be a nonempty list of column names.")
    if any(not isinstance(column, str) or not column for column in columns):
        raise ValueError("Completeness columns must be nonempty column names.")
    if len(set(columns)) != len(columns):
        raise ValueError("Completeness columns must be unique.")
    if type(min_present) is not int or not 1 <= min_present <= len(columns):
        raise ValueError("min_present must be an integer between 1 and the number of columns.")
    params = {"columns": list(columns), "min_present": min_present}
    return filter_step(
        "minimum_completeness", has_enough_values, columns=list(columns), params=params
    )


# ---------------------------------------------------------------------------
# Example 2 - keep values inside a known valid range (missing values are kept).
#   value_range("age", 0, 120)   age = 35 kept, -1 removed, 130 removed, null kept
# ---------------------------------------------------------------------------


def in_range(df, params):
    """True when the value is within [minimum, maximum] or missing."""
    values = df[params["column"]]
    return values.between(params["minimum"], params["maximum"]) | values.isna()


def value_range(column, minimum, maximum):
    """Remove rows whose column is outside [minimum, maximum]."""
    params = {"column": column, "minimum": minimum, "maximum": maximum}
    return filter_step(f"{column}_range", in_range, columns=[column], params=params)


# ---------------------------------------------------------------------------
# Example 3 - keep only listed categories (null is removed).
#   allowed_values("country", ["NL", "DE"])   NL kept, FR removed, null removed
# ---------------------------------------------------------------------------


def is_allowed(df, params):
    """True when the value is one of the allowed values."""
    return df[params["column"]].isin(params["values"])


def allowed_values(column, values):
    """Keep rows whose column is one of the given values."""
    params = {"column": column, "values": list(values)}
    return filter_step(f"{column}_allowed", is_allowed, columns=[column], params=params)


# ---------------------------------------------------------------------------
# Example 4 - the allowed list read from a data file (asset). Inactive.
#   1. Create src/features/assets/countries.json   ["NL", "DE", "BE"]
#   2. In src/features/assets.json set   "files": ["assets/countries.json"]
#   3. Uncomment the code below, then add allowed_countries() to a pre-split recipe.
#   The file is saved with the model; editing it later needs a new training run.
# ---------------------------------------------------------------------------

# import json
#
# from skyulf.inference.project_package import read_project_asset
#
#
# def is_known_country(df):
#     """True when the country is listed in the saved countries file."""
#     countries = json.loads(read_project_asset(__package__, "assets/countries.json"))
#     return df["country"].isin(countries)
#
#
# def allowed_countries():
#     """Keep rows whose country is in assets/countries.json."""
#     return filter_step("allowed_countries", is_known_country, columns=["country"])
