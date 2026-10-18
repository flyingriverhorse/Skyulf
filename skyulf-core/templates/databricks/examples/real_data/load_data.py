# Databricks notebook source
"""Create example datasets from checksum-pinned public UCI archives.

Run before the selected Bundles. The default layout creates tables once; this
notebook never overwrites observations or predictions. Source timestamps in the bike archive
are local clock labels without an offset; UTC is used as an ordering convention,
not a claim that the original observations were recorded in UTC.

For the single-table air-quality walkthrough, select only air_quality_no2,
set air_quality_layout=single_table and leave air_quality_day empty to seed
summer data. Then supply one winter YYYY-MM-DD per invocation to append it.
Run these invocations serially; the example is a single-writer loader.
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
# The default layout creates tables once; select only new projects when rerunning it.
dbutils.widgets.text("datasets", "diagnostic_single,bike_demand_set,air_quality_no2")  # noqa: F821
dbutils.widgets.dropdown("air_quality_layout", "separate", ["separate", "single_table"])  # noqa: F821
dbutils.widgets.text("air_quality_day", "")  # noqa: F821
catalog = dbutils.widgets.get("catalog")  # noqa: F821
archive_directory = dbutils.widgets.get("archive_directory")  # noqa: F821
datasets = {name.strip() for name in dbutils.widgets.get("datasets").split(",")}  # noqa: F821
air_quality_layout = dbutils.widgets.get("air_quality_layout")  # noqa: F821
air_quality_day = dbutils.widgets.get("air_quality_day")  # noqa: F821
if not catalog.replace("_", "").isalnum():
    raise ValueError("Use a simple catalog identifier.")
if not datasets <= {"diagnostic_single", "bike_demand_set", "air_quality_no2"}:
    raise ValueError(f"Unknown datasets: {sorted(datasets)}")
if air_quality_layout not in {"separate", "single_table"}:
    raise ValueError("Unknown air_quality_layout.")
if air_quality_layout == "single_table" and datasets != {"air_quality_no2"}:
    raise ValueError("The single-table walkthrough requires datasets=air_quality_no2 only.")
if air_quality_day and air_quality_layout != "single_table":
    raise ValueError("air_quality_day requires air_quality_layout=single_table.")

SOURCES = {
    "diagnostic": (
        "https://archive.ics.uci.edu/static/public/17/breast+cancer+wisconsin+diagnostic.zip",
        "bc154869ef13f753f9e2b5a17e248cfe1ba4b6721db7c4da9f4880e40b05d3af",
    ),
    "bike": (
        "https://archive.ics.uci.edu/static/public/275/bike+sharing+dataset.zip",
        "b70182d0d0508e9abbb79306ce5c0cec34869000f8220175ac83d11dbe845401",
    ),
    "air_quality": (
        "https://archive.ics.uci.edu/static/public/360/air+quality.zip",
        "d4a64013fb385288a8a48d9d193ca7079b2e1bbddf6f8d458feb8c08ab2b8a2a",
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


def save_dataset(schema, training, scoring, targets, labelled=None):
    """Keep training, unlabelled scoring inputs and delayed-outcome labels separate.

    ``labelled`` optionally selects the scoring rows whose outcome was measured.
    """
    spark.sql(f"CREATE SCHEMA IF NOT EXISTS `{catalog}`.`{schema}`")
    training_frame = spark.createDataFrame(training)
    scoring_frame = spark.createDataFrame(scoring.drop(columns=targets))
    outcomes = scoring if labelled is None else scoring.loc[labelled]
    labels = spark.createDataFrame(outcomes[["record_id", *targets]]).withColumn(
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


def load_diagnostic():
    """Split the Wisconsin diagnostic rows into stratified training and scoring sets."""
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
    return save_dataset("diagnostic_single", train, score, ["malignant"])


def load_bike():
    """Train on hours before October 2012 and score the later hours."""
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
    return save_dataset("bike_demand_set", training, scoring, ["casual", "registered", "cnt"])


AIR_QUALITY_COLUMNS = {
    "CO(GT)": "co_gt",
    "PT08.S1(CO)": "pt08_s1_co",
    "PT08.S2(NMHC)": "pt08_s2_nmhc",
    "PT08.S3(NOx)": "pt08_s3_nox",
    "PT08.S4(NO2)": "pt08_s4_no2",
    "PT08.S5(O3)": "pt08_s5_o3",
    "T": "temperature",
    "RH": "rel_humidity",
    "AH": "abs_humidity",
    "NO2(GT)": "no2_gt",
}


def air_quality_frame(archive):
    """Rename the station columns and keep its -200 missing-reading marker unchanged.

    Reference analyzers other than CO (NMHC, C6H6, NOx) are left out: NOx contains the
    NO2 target and C6H6 is derived from a sensor, so they would leak or duplicate inputs.
    """
    raw = pd.read_csv(archive.open("AirQualityUCI.csv"), sep=";", decimal=",", encoding="latin-1")
    raw = raw.dropna(axis=1, how="all").dropna(how="all").reset_index(drop=True)
    assert len(raw) == 9357
    observed = pd.to_datetime(raw["Date"] + " " + raw["Time"], format="%d/%m/%Y %H.%M.%S", utc=True)
    frame = raw[list(AIR_QUALITY_COLUMNS)].rename(columns=AIR_QUALITY_COLUMNS).astype("float64")
    frame.insert(0, "observed_at", observed)
    frame.insert(0, "record_id", range(1, len(frame) + 1))
    frame["hour"] = observed.dt.hour.astype("float64")
    frame["weekday"] = observed.dt.dayofweek.astype("float64")
    return frame


def load_air_quality():
    """Train on March-October 2004 and score the winter that follows, so drift is real."""
    with source_archive("air_quality") as archive:
        frame = air_quality_frame(archive)
    if air_quality_layout == "single_table":
        return save_air_quality_single_table(air_quality_batch(frame, air_quality_day))
    boundary = pd.Timestamp("2004-11-01", tz="UTC")
    training = frame.loc[frame["observed_at"] < boundary]
    scoring = frame.loc[frame["observed_at"] >= boundary]
    # Labels hold only measured hours; requests keep the raw -200 marker in their inputs.
    return save_dataset(
        "air_quality_no2", training, scoring, ["no2_gt"], labelled=scoring["no2_gt"].ne(-200)
    )


def air_quality_batch(frame, day):
    """Select summer initialization or exactly one existing winter day without mutating truth."""
    boundary = pd.Timestamp("2004-11-01", tz="UTC")
    if not day:
        return frame.loc[frame["observed_at"] < boundary].copy()
    try:
        start = pd.Timestamp(day, tz="UTC")
        if start.strftime("%Y-%m-%d") != day or start < boundary:
            raise ValueError("Invalid day.")
    except (ValueError, TypeError) as error:
        raise ValueError("air_quality_day must be a winter YYYY-MM-DD from the archive.") from error
    selected = frame.loc[
        (frame["observed_at"] >= start) & (frame["observed_at"] < start + pd.Timedelta(days=1))
    ]
    if selected.empty:
        raise ValueError("air_quality_day has no observations in the archive.")
    return selected.copy()


def append_missing_rows(table, frame):
    """Recover a serial loader retry with inserts only, including partially written batches."""
    missing = frame.join(spark.table(table).select("record_id"), "record_id", "left_anti")
    count = missing.count()
    if count:
        missing.writeTo(table).append()
    return count


def save_air_quality_single_table(batch):
    """Seed a separate example schema, then append inputs and known outcomes independently."""
    schema = f"{catalog}.air_quality_no2_append"
    observations = spark.createDataFrame(batch)
    labels = spark.createDataFrame(batch[["record_id", "no2_gt"]]).where(F.col("no2_gt") != -200)
    labels = labels.withColumn("result_available_at", F.current_timestamp())
    if air_quality_day:
        # New requests have unknown targets; outcomes never UPDATE the input table.
        observations = observations.withColumn("no2_gt", F.lit(None).cast("double"))
    tables = {"observations": observations, "labels": labels}
    if not air_quality_day:
        spark.sql(f"CREATE SCHEMA IF NOT EXISTS {schema}")
        if any(spark.catalog.tableExists(f"{schema}.{name}") for name in tables):
            raise ValueError("Single-table example already exists; use air_quality_day to append.")
        for name, frame in tables.items():
            frame.writeTo(f"{schema}.{name}").using("delta").tableProperty(
                "delta.enableChangeDataFeed", "true"
            ).tableProperty("skyulf.example", "air_quality_no2_single_table").create()
        return {"schema": schema, "action": "seed", "rows": len(batch)}
    for name in tables:
        properties = {
            row["key"]: row["value"]
            for row in spark.sql(f"SHOW TBLPROPERTIES {schema}.{name}").collect()
        }
        if properties.get("skyulf.example") != "air_quality_no2_single_table":
            raise ValueError(f"{schema}.{name} is not owned by this example loader.")
        if properties.get("delta.enableChangeDataFeed") != "true":
            raise ValueError(f"{schema}.{name} requires change data feed.")
    return {
        "schema": schema,
        "action": "append",
        "day": air_quality_day,
        "inserted": {
            name: append_missing_rows(f"{schema}.{name}", frame) for name, frame in tables.items()
        },
    }


LOADERS = {
    "diagnostic_single": load_diagnostic,
    "bike_demand_set": load_bike,
    "air_quality_no2": load_air_quality,
}
evidence = {name: LOADERS[name]() for name in LOADERS if name in datasets}
evidence["sources"] = {
    name: {"url": url, "sha256": digest} for name, (url, digest) in SOURCES.items()
}
dbutils.notebook.exit(json.dumps(evidence))  # noqa: F821
