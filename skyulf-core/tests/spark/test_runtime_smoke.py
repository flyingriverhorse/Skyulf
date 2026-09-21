"""Exercise the JVM and Python worker boundaries of the optional Spark test lane."""

import importlib


def test_spark_executes_multiple_partitions(spark):
    """A successful import must not substitute for distributed JVM execution."""
    functions = importlib.import_module("pyspark.sql.functions")

    frame = spark.range(100).repartition(2).withColumn("partition", functions.spark_partition_id())
    result = frame.agg(
        functions.sum("id").alias("total"),
        functions.countDistinct("partition").alias("partitions"),
    ).first()

    assert result["total"] == 4950
    assert result["partitions"] == 2


def test_spark_executes_python_arrow_batches(spark):
    """The configured Python worker must exchange Arrow batches with the driver."""
    functions = importlib.import_module("pyspark.sql.functions")

    def double_batches(batches):
        """Double each batch without moving the input dataset to the driver."""
        for batch in batches:
            yield batch.assign(doubled=batch["id"] * 2)

    frame = spark.range(10).repartition(2)
    result = frame.mapInPandas(double_batches, "id long, doubled long")
    total = result.agg(functions.sum("doubled").alias("total")).first()["total"]

    assert total == 90
