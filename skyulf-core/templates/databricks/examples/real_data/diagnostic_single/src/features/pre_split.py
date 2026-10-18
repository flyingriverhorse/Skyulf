"""Training row filters that also explain excluded scoring rows."""

from skyulf.preprocessing import filter_step


def positive(frame, params):
    """Keep rows whose recorded dimensions are positive; leave null handling to eligibility."""
    values = frame.loc[:, params["columns"]]
    return (values.gt(0) | values.isna()).all(axis=1)


def positive_measurements(columns):
    """Declare precisely which raw inputs the reusable filter reads."""
    return filter_step(
        "positive_measurements", positive, columns=columns, params={"columns": columns}
    )
