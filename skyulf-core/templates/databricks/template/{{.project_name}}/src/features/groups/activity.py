"""Example completed-month activity features; enable them in config/features.yml.

Input: company_id, observed_at, amount; several transactions per key/date allowed.
observed_at must already identify the completed month's feature availability,
not each transaction's event timestamp. Apply your cutoff/window upstream if
your raw input uses event dates. Do not label future totals with an earlier date.
Output: one row per company/date, monthly_amount and transaction_count.
Sum ignores null amounts (all-null totals stay null); count includes those rows.
A company with no input rows has no output row; no missing month is invented.
"""

from pyspark.sql import functions as F


def compute_features(frame):
    """Aggregate the supplied Spark rows into two features at the declared grain."""
    return frame.groupBy("company_id", "observed_at").agg(
        F.sum("amount").alias("monthly_amount"),
        F.count(F.lit(1)).alias("transaction_count"),
    )
