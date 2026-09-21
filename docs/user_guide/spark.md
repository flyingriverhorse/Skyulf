# Spark: development setup and current support

Spark support is under development for 0.9.0. The optional dependency and
runtime tests are available in the development checkout. **Skyulf's Spark
engine, native nodes and distributed model inference are not available yet.**
Installing the extra does not convert a pandas/Polars pipeline to Spark.

## Why a separate environment?

`.venv-spark` isolates PySpark and its dependencies from the web application's
existing `.venv`. It is a local development convention, not a runtime requirement
or a directory shipped with Skyulf. Both environments are excluded from Git.
Use your own environment name if preferred.

The initial test target is Python 3.12, PySpark 4.0.3 and Java 17. Java must be
installed separately. `JAVA_HOME` should point to the JDK directory. Consult
the [Spark installation documentation](https://spark.apache.org/docs/4.0.3/api/python/getting_started/install.html)
for runtime requirements. Databricks Runtime and Connect require their own
validated dependency profiles; do not replace a managed runtime's PySpark
installation with this development environment.

## Install from the repository

Run these commands from the repository root. Create the environment only once.

```powershell
uv venv .venv-spark --python 3.12
uv pip install --python .venv-spark/Scripts/python.exe -r requirements-spark.txt
$env:JAVA_HOME = 'C:\path\to\jdk-17'
$env:SKYULF_REQUIRE_SPARK = '1'
.venv-spark/Scripts/python.exe -m pytest skyulf-core/tests/spark -q -o addopts=
```

On Linux/macOS:

```bash
uv venv .venv-spark --python 3.12
uv pip install --python .venv-spark/bin/python -r requirements-spark.txt
export JAVA_HOME=/path/to/jdk-17
SKYULF_REQUIRE_SPARK=1 .venv-spark/bin/python -m pytest skyulf-core/tests/spark -q -o addopts=
```

The tests execute a distributed aggregate and an Arrow/Python worker transform.
The mandatory lane fails when PySpark is missing. In the ordinary core suite,
the optional runtime tests skip if PySpark is not installed. A skip does not
prove Spark compatibility.

## Minimal runtime example

This is a plain PySpark example, not a Skyulf feature-engineering pipeline.
Run it using the Spark environment's Python interpreter.

```python
import os
import sys

from pyspark.sql import SparkSession, functions as F

os.environ["PYSPARK_PYTHON"] = sys.executable
spark = SparkSession.builder.master("local[2]").appName("spark-check").getOrCreate()
try:
    frame = spark.range(100).repartition(2)
    total = frame.agg(F.sum("id").alias("total")).first()["total"]
    assert total == 4950
finally:
    spark.stop()
```

Only one aggregate row is returned to Python. Future native FE operations will
keep the dataset in Spark and transfer bounded learned state when supported.
Python model inference on Spark workers is a separate capability from native
Spark model training.

## Troubleshooting and support boundaries

- **Java gateway fails:** check Java 17 and the current shell's `JAVA_HOME`.
- **Worker cannot find Python:** use the same interpreter for driver and workers;
  the test fixture sets `PYSPARK_PYTHON` explicitly.
- **Windows access denied at shutdown:** a process sandbox can prevent Spark's
  child-process cleanup. A passing calculation does not prove clean shutdown;
  use an environment that permits the owned JVM processes to terminate.
- **Engine not found:** this is expected before the Spark engine implementation
  is delivered. Do not collect an entire distributed frame into pandas as a workaround.

MLflow tracking, Unity Catalog registration, Databricks jobs, serving endpoints
and Bundle templates are separate integration stages. This page will gain
tested fit/apply and inference examples as those capabilities are implemented.
