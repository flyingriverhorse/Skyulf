"""Fixed eligibility; missing inputs survive for the fitted imputer."""

from skyulf.preprocessing import filter_step


def in_range(df, params):
    """Accept values in the inclusive range and leave missing inputs for imputation."""
    values = df[params["column"]]
    return values.between(params["minimum"], params["maximum"]) | values.isna()


def value_range(column, minimum, maximum):
    """Build a saved row filter with no learned statistics."""
    return filter_step(
        f"{column}_range",
        in_range,
        columns=[column],
        params={"column": column, "minimum": minimum, "maximum": maximum},
    )
