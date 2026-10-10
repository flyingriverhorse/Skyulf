"""Generate isolated Spark data and read explicit Delta snapshots."""

import re


def validate_namespace(namespace):
    """Keep generated SQL identifiers separate from expressions and existing defaults."""
    if not re.fullmatch(r"[A-Za-z_]\w*\.[A-Za-z_]\w*", namespace, flags=re.ASCII):
        raise ValueError("namespace must be catalog.schema using simple identifiers")
    return namespace


def create_raw(spark, namespace, config):
    """Create a fresh schema and deterministic synthetic raw customers with nulls."""
    validate_namespace(namespace)
    train, predict = config["training_rows"], config["prediction_rows"]
    if any(type(n) is not int or not 32 <= n <= 10000 for n in (train, predict)):
        raise ValueError("Demo cohort sizes must be integers between 32 and 10000")
    # CREATE deliberately fails on an existing schema: this demo never overwrites one.
    spark.sql(f"CREATE SCHEMA {namespace}").collect()
    spark.sql(f"""
        CREATE TABLE {namespace}.raw_customers USING DELTA AS
        WITH complete AS (
          SELECT id, CAST(18 + pmod(id * 7, 60) AS DOUBLE) age,
            CAST(20000 + pmod(id * 7919, 100000) AS DOUBLE) income,
            CAST(1 + pmod(id * 11, 72) AS DOUBLE) tenure,
            element_at(array('Mass', 'Affluent', 'Business'),
                       CAST(pmod(id, 3) + 1 AS INT)) segment
          FROM range({train + predict})
        )
        SELECT CAST(9007199254740993 + id AS BIGINT) customer_id,
          CASE WHEN pmod(id, 23) = 0 THEN NULL ELSE age END age,
          CASE WHEN pmod(id, 17) = 0 THEN NULL ELSE income END income,
          CASE WHEN pmod(id, 19) = 0 THEN NULL ELSE tenure END tenure,
          CASE WHEN pmod(id, 29) = 0 THEN NULL
               WHEN id >= {train} AND pmod(id, 7) = 0 THEN 'NewSegment'
               ELSE segment END segment,
          CAST(age + income * 0.0003 - tenure * 0.3 + sin(id) * 8 > 55 AS BIGINT) churn,
          CASE WHEN id < {train} THEN 'train' ELSE 'score' END cohort
        FROM complete
    """).collect()
    version = spark.sql(f"DESCRIBE HISTORY {namespace}.raw_customers LIMIT 1").first().version
    return {"source_version": int(version), "raw_rows": train + predict}


def read_cohort(spark, namespace, source_version, cohort):
    """Every consumer reads the same immutable source version and explicit population."""
    validate_namespace(namespace)
    if type(source_version) is not int or source_version < 0 or cohort not in {"train", "score"}:
        raise ValueError("Explicit Delta version and train/score cohort required")
    return (
        spark.read.option("versionAsOf", source_version)
        .table(f"{namespace}.raw_customers")
        .where(f"cohort = '{cohort}'")
        .orderBy("customer_id")
    )


def bounded_pandas(frame, expected_rows):
    """Reject oversized demo input before materializing pandas on the driver."""
    result = frame.limit(expected_rows + 1).toPandas()
    if len(result) != expected_rows:
        raise ValueError(f"Expected exactly {expected_rows} demo rows; got {len(result)}")
    return result
