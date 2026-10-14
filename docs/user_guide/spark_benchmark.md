# Measure Spark inference capacity

This page covers the reproducible Spark inference benchmark, how to read its
measurements, and the separate local PySpark environment with its runtime
smoke check. Read it when you size worker scoring on your own compute or set up
a local Spark development environment; choosing an execution path and the
native feature-engineering walkthroughs stay on the parent page.

[Back to Spark: choose an execution path](spark.md)

## Measuring inference capacity

The reproducible notebook `skyulf-core/benchmarks/bench_spark_inference.py`
compares `native_features`, `python_pipeline` and certified MLflow pyfunc UDF
on the same fitted SimpleImputer/StandardScaler/LinearRegression pipeline.
It uses Spark-generated data, including nulls, and verifies every prediction
against an independent expression. Only aggregate diagnostics reach the driver.

Notebook parameters:

| Parameter | Purpose |
| --- | --- |
| `schema` | Dedicated `catalog.schema` for benchmark models and report volume |
| `experiment` | MLflow experiment path for benchmark artifacts |
| `compute` | `classic` or `serverless`; must match the actual job compute |
| `env_manager` | `virtualenv` for validated classic environments; `local` for the declared serverless worker environment |
| `suite` | `full` for the scale matrix; `smoke` for a small runner verification |

The full matrix covers 1 and 5 million rows, 2 and 64 input features, 8 and 32
requested partitions, and 1,000/10,000-row model calls. It reports observed
source partitions and both first/repeat action timings. Timings include source
generation, inference and a correctness aggregate, not Delta publication.
Preparation and model registration are outside action throughput. Fixed route
order and managed serverless allocation mean these are workload observations,
not a universal ranking or latency guarantee.

Full reports are saved under `/Volumes/<catalog>/<schema>/benchmark/<id>.json`
and returned from the notebook. Action checkpoints survive later scoring errors;
the job still fails. Separate worker probes measure package restore time and
Python-process RSS. Their process lifetime high-water mark is not an incremental
allocation measurement or a complete executor-memory measurement.

On classic compute, enable `spark.executor.processTreeMetrics.enabled=true`
and `spark.executor.metrics.pollingInterval=1000` before starting the cluster.
Executor peaks are cumulative across the application. Disabled or unobserved
process counters are unknown, not zero. Serverless does not expose the executor
status store or caller-controlled Arrow allocation; those metrics remain
unavailable. Model-call batch size alone cannot bound worker memory.

### Interpreting measurements

Compare preparation, first action and repeated action separately. Changing
partition count or model-call size can increase overhead as well as improve
throughput; measure the actual feature width, model size and input distribution.
Include publication when estimating end-to-end score-job latency.

Keep benchmark outputs in your own schema or volume and record the runtime,
package versions and parameters alongside the results. Worker process RSS includes
libraries and the Python environment; its lifetime peak is not a per-call memory
ceiling. A passing benchmark does not expand fitted-recipe admission.

## Local development environment and runtime check

### Why a separate environment?

`.venv-spark` isolates PySpark and its dependencies from the web application's
existing `.venv`. It is a local development convention, not a runtime requirement
or a directory shipped with Skyulf. Both environments are excluded from Git.
Use your own environment name if preferred.

The initial test target is Python 3.12, PySpark 4.0.3 and Java 17. Java must be installed
separately. `JAVA_HOME` should point to the JDK directory. Consult the
[Spark installation documentation](https://spark.apache.org/docs/4.0.3/api/python/getting_started/install.html)
for runtime requirements. Databricks Runtime and Connect require their own validated dependency
profiles; do not replace a managed runtime's PySpark installation with this development environment.

### Install from the repository

Run these commands from the repository root. Create the environment only once.

```powershell
uv venv .venv-spark --python 3.12
uv pip install --python .venv-spark/Scripts/python.exe -r requirements-spark.txt
$env:JAVA_HOME = 'C:\path\to\jdk-17'
$env:SKYULF_REQUIRE_SPARK = '1'
.venv-spark/Scripts/python.exe -m pytest skyulf-core/tests/spark/test_runtime_smoke.py -q -o addopts=
```

On Linux/macOS:

```bash
uv venv .venv-spark --python 3.12
uv pip install --python .venv-spark/bin/python -r requirements-spark.txt
export JAVA_HOME=/path/to/jdk-17
SKYULF_REQUIRE_SPARK=1 .venv-spark/bin/python -m pytest skyulf-core/tests/spark/test_runtime_smoke.py -q -o addopts=
```

The tests execute a distributed aggregate and an Arrow/Python worker transform.
The mandatory lane fails when PySpark is missing. In the ordinary core suite,
the optional runtime tests skip if PySpark is not installed. A skip does not
prove Spark compatibility.

### Minimal runtime example

This is a plain PySpark example, not a Skyulf feature-engineering pipeline.
Run it using the Spark environment's Python interpreter.

<!-- skyulf-example: spark_runtime local-jvm-startup -->
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

Only one aggregate row is returned to Python. Supported native FE operations
keep the dataset in Spark and transfer bounded learned state when supported.
Python model inference on Spark workers is a separate capability from native
Spark model training.
