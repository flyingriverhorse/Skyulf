"""Derive the target and a string record key from raw bronze observations.

Input: entity_id, observed_date (DATE), birth_date (DATE, empty when unknown),
region_code, employee_count, revenue. The features job passes the pinned copy of
the streaming table and publishes the returned rows to config/features.yml
output_table. Keep this to fixed formulas: learned values such as means belong in
config/preprocessing.yml.
"""

from pyspark.sql import functions as F


def prepare(frame):
    """Return keys, the DATE grain, the age target and the raw model inputs."""
    return frame.select(
        F.col("entity_id").cast("string").alias("entity_id"),
        # Record keys cannot be DATE; keep a yyyy-MM-dd text key next to the DATE grain.
        F.date_format("observed_date", "yyyy-MM-dd").alias("observed_on"),
        "observed_date",
        # Empty when birth_date is unknown; skip_missing_targets leaves those rows out.
        F.round(F.months_between("observed_date", "birth_date") / 12, 2).alias("age"),
        "region_code",
        "employee_count",
        "revenue",
    )
