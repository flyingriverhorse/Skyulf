"""Training row filters for the air-quality example; scoring reuses them as eligibility."""

from skyulf.preprocessing import filter_step

from .sensors import MISSING_READING


def online(frame, params):
    """Keep hours where every listed instrument reported; -200 marks an offline instrument."""
    return frame.loc[:, params["columns"]].ne(MISSING_READING).all(axis=1)


def sensors_online(columns):
    """Declare the instrument columns the reusable outage filter reads."""
    return filter_step("sensors_online", online, columns=columns, params={"columns": columns})
