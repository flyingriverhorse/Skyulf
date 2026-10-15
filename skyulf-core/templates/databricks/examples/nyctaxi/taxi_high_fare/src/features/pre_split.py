"""Row filters selected in config/pre_split.yml as pre_split.<factory>."""

from skyulf.preprocessing import filter_step


def in_range(df, params):
    """True when the value is within [minimum, maximum] or missing."""
    values = df[params["column"]]
    return values.between(params["minimum"], params["maximum"]) | values.isna()


def value_range(column, minimum, maximum):
    """Remove rows whose column is outside [minimum, maximum]."""
    params = {"column": column, "minimum": minimum, "maximum": maximum}
    return filter_step(f"{column}_range", in_range, columns=[column], params=params)
