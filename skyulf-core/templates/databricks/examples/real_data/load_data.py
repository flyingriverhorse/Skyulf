# Databricks notebook source
"""Create the two example datasets from checksum-pinned public UCI archives.

Run once before either Bundle. Tables are create-only: this notebook never
overwrites observations or predictions. Source timestamps in the bike archive
are local clock labels without an offset; UTC is used as an ordering convention,
not a claim that the original observations were recorded in UTC.
"""

import hashlib
import io
import json
import zipfile
from pathlib import Path
from urllib.request import urlopen

import pandas as pd
from pyspark.sql import SparkSession
from pyspark.sql import functions as F
from sklearn.model_selection import train_test_split

spark = SparkSession.builder.getOrCreate()
dbutils.widgets.text("catalog", "workspace")  # noqa: F821
dbutils.widgets.text("archive_directory", "")  # noqa: F821
catalog = dbutils.widgets.get("catalog")  # noqa: F821
archive_directory = dbutils.widgets.get("archive_directory")  # noqa: F821
if not catalog.replace("_", "").isalnum():
    raise ValueError("Use a simple catalog identifier.")

SOURCES = {
    "diagnostic": (
        "https://archive.ics.uci.edu/static/public/17/breast+cancer+wisconsin+diagnostic.zip",
        "bc154869ef13f753f9e2b5a17e248cfe1ba4b6721db7c4da9f4880e40b05d3af",
    ),
    "bike": (
        "https://archive.ics.uci.edu/static/public/275/bike+sharing+dataset.zip",
        "b70182d0d0508e9abbb79306ce5c0cec34869000f8220175ac83d11dbe845401",
    ),
}


def source_archive(name):
    """Verify either a downloaded archive or the same archive staged in a UC volume."""
    url, expected = SOURCES[name]
    if archive_directory:
        payload = (Path(archive_directory) / f"{name}.zip").read_bytes()
    else:
        with urlopen(url, timeout=60) as response:
            payload = response.read()
    if hashlib.sha256(payload).hexdigest() != expected:
        raise ValueError(f"{name}: source archive checksum changed; inspect it before using.")
    return zipfile.ZipFile(io.BytesIO(payload))


def save_dataset(schema, training, scoring, targets):
    """Keep training, unlabelled scoring inputs and delayed-outcome labels separate."""
    spark.sql(f"CREATE SCHEMA IF NOT EXISTS `{catalog}`.`{schema}`")
    training_frame = spark.createDataFrame(training)
    scoring_frame = spark.createDataFrame(scoring.drop(columns=targets))
    labels = spark.createDataFrame(scoring[["record_id", *targets]]).withColumn(
        "result_available_at", F.current_timestamp()
    )
    for name, frame in (
        ("observations_train", training_frame),
        ("observations_score", scoring_frame),
        ("labels", labels),
    ):
        frame.writeTo(f"{catalog}.{schema}.{name}").using("delta").tableProperty(
            "delta.enableChangeDataFeed", "true"
        ).create()
    return {"training_rows": len(training), "scoring_rows": len(scoring), "targets": targets}


with source_archive("diagnostic") as archive:
    raw = pd.read_csv(archive.open("wdbc.data"), header=None)
diagnostic = raw.iloc[:, [0, 2, 3, 4, 5, 6, 7]].copy()
diagnostic.columns = [
    "record_id",
    "mean_radius",
    "mean_texture",
    "mean_perimeter",
    "mean_area",
    "mean_smoothness",
    "mean_compactness",
]
diagnostic["malignant"] = raw[1].eq("M").astype("int64")
train, score = train_test_split(
    diagnostic, test_size=0.25, random_state=42, stratify=diagnostic["malignant"]
)
evidence = {"diagnostic_single": save_dataset("diagnostic_single", train, score, ["malignant"])}

with source_archive("bike") as archive:
    bike = pd.read_csv(archive.open("hour.csv"))
assert len(bike) == 17379 and (bike["cnt"] == bike["casual"] + bike["registered"]).all()
bike = bike.rename(columns={"instant": "record_id"})
bike["observed_at"] = pd.to_datetime(bike["dteday"], utc=True) + pd.to_timedelta(
    bike["hr"], unit="h"
)
inputs = [
    "season",
    "yr",
    "mnth",
    "hr",
    "holiday",
    "weekday",
    "workingday",
    "weathersit",
    "temp",
    "atemp",
    "hum",
    "windspeed",
]
bike[inputs] = bike[inputs].astype("float64")
bike = bike[["record_id", "observed_at", *inputs, "casual", "registered", "cnt"]]
training = bike.loc[bike["observed_at"] < pd.Timestamp("2012-10-01", tz="UTC")]
scoring = bike.loc[bike["observed_at"] >= pd.Timestamp("2012-10-01", tz="UTC")]
evidence["bike_demand_set"] = save_dataset(
    "bike_demand_set", training, scoring, ["casual", "registered", "cnt"]
)
evidence["sources"] = {
    name: {"url": url, "sha256": digest} for name, (url, digest) in SOURCES.items()
}
dbutils.notebook.exit(json.dumps(evidence))  # noqa: F821
