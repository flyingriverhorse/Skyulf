# Databricks notebook source
# MAGIC %md
# MAGIC # Raw customer model: training, REST and SQL
# MAGIC Each job task selects one stage. Training does not publish predictions by default.
# MAGIC REST and SQL stages read the same raw Delta snapshot and call the same model version.

# COMMAND ----------
import json
import sys
import time
from pathlib import Path

import yaml

dbutils = globals()["dbutils"]
spark = globals()["spark"]
for name in (
    "stage",
    "config_path",
    "namespace",
    "endpoint_name",
    "source_version",
    "model_version",
    "sql_function_ddl_path",
):
    dbutils.widgets.text(name, "")
stage = dbutils.widgets.get("stage")
config_path = Path(dbutils.widgets.get("config_path"))
sys.path.insert(0, str(config_path.parent / "src"))

from databricks.sdk import WorkspaceClient
from databricks.sdk.core import Config
from raw_serving_demo.data import create_raw, validate_namespace
from raw_serving_demo.scoring import deploy, prepare, rest_predictions, sql_predictions, verify
from raw_serving_demo.training import train

config = yaml.safe_load(config_path.read_text(encoding="utf-8"))
namespace = validate_namespace(dbutils.widgets.get("namespace"))
client = WorkspaceClient(config=Config(retry_timeout_seconds=15))
source_version = int(dbutils.widgets.get("source_version") or 0)
started = time.monotonic()
print(json.dumps({"stage": stage, "namespace": namespace, "status": "STARTED"}), flush=True)

# COMMAND ----------
try:
    if stage == "raw_data":
        result = create_raw(spark, namespace, config)
    elif stage == "train":
        experiment_path = str(config_path.parent).replace("/Workspace/", "/", 1) + "/experiment"
        result = train(spark, namespace, source_version, config, experiment_path)
    else:
        plan = prepare(
            namespace, dbutils.widgets.get("endpoint_name"), dbutils.widgets.get("model_version")
        )
        if stage == "deploy":
            result = deploy(client, plan)
        elif stage == "predict_rest":
            result = rest_predictions(spark, client, plan, namespace, source_version, config)
        elif stage == "predict_sql":
            ddl_path = dbutils.widgets.get("sql_function_ddl_path")
            function_ddl = Path(ddl_path).read_text(encoding="utf-8") if ddl_path else None
            result = sql_predictions(
                spark, client, plan, namespace, source_version, function_ddl=function_ddl
            )
        elif stage == "verify":
            result = verify(spark, client, plan, namespace, source_version, config)
        else:
            raise ValueError(f"Unknown demo stage: {stage}")
    for key in ("source_version", "model_version"):
        if key in result:
            dbutils.jobs.taskValues.set(key=key, value=result[key])
    result.update({"stage": stage, "seconds": round(time.monotonic() - started, 2)})
    print(json.dumps(result, indent=2), flush=True)
except Exception as exc:
    print(
        json.dumps(
            {
                "stage": stage,
                "status": "FAILED",
                "namespace": namespace,
                "error_type": type(exc).__name__,
                "message": str(exc),
            }
        ),
        flush=True,
    )
    raise

# COMMAND ----------
dbutils.notebook.exit(json.dumps(result))
