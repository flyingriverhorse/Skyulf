"""Reusable raw-input clock validity filter."""

from skyulf.preprocessing import filter_step


def in_clock_range(frame):
    """Exclude impossible hours while preserving nulls for an explicit eligibility reason."""
    return frame["hr"].between(0, 23) | frame["hr"].isna()


def valid_clock():
    """Save the input dependency so the same filter can run without training labels."""
    return filter_step("valid_clock", in_clock_range, columns=["hr"])
