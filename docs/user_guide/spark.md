# Spark: development setup and current support

Spark support is under development for 0.9.0. The optional dependency and
runtime tests, Spark engine adapter and keyed FeatureEngineer entry point are
available in the development checkout.
**Native nodes and distributed model inference are not available yet.**
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

## Spark engine adapter

The adapter detects real Spark dataframes without changing the default local
engine. Importing Skyulf does not import PySpark or MLflow. Requesting the Spark
engine without its optional dependency raises an explicit installation error.

After creating your own `spark` session, you can run:

```python
from skyulf.engines import EngineRegistry, SparkEngine, get_engine

local_default = get_engine()
frame = spark.range(10).repartition(2)
assert get_engine(frame) is SparkEngine
wrapped = EngineRegistry.wrap(frame)
selected = wrapped.select(["id"])
assert selected.columns == ["id"]
assert selected.schema == frame.schema
assert selected.to_native().count() == 10  # Explicit native Spark action.
assert get_engine() is local_default
```

The separate `DistributedDataFrame` protocol exposes `columns`, `schema`,
`select` and `to_native`. Projection treats names literally, including dots and
backticks; an empty projection keeps the rows. Schema access can require query
analysis, but these adapter operations do not collect rows or count them.

The local `SkyulfDataFrame` protocol remains for pandas/Polars. Spark wrappers
reject `len`, `shape`, `to_pandas`, `to_numpy` and `to_arrow` with
`DistributedMaterializationError`. `SparkEngine.to_numpy` and `SklearnBridge`
also reject distributed features or targets. Worker inference will have its own
explicit entry point; it is not enabled by this adapter.

Skyulf neither creates nor retains a Spark session. Use your own
`spark.createDataFrame(...)` to create input; `SparkEngine.from_pandas` and
`create_dataframe` reject calls with this guidance. `to_native()` returns the
original distributed frame: native actions you call explicitly remain your
responsibility. In particular, Spark's
[`toPandas`](https://spark.apache.org/docs/4.0.3/api/python/reference/pyspark.sql/api/pyspark.sql.DataFrame.toPandas.html)
collects data into driver memory.

Validation currently covers classic PySpark 4.0.3 on the local test runtime.
The adapter uses public dataframe APIs, but Databricks and Spark Connect have
not been integration-tested yet. Engine detection does not authorize a Spark
frame to run through existing local-only FE nodes.

## Execution contracts

The development API can validate a future execution request without starting
Spark or changing the current pandas/Polars engine:

```python
from skyulf.core.execution import ExecutionOptions, FrameSpec

options = ExecutionOptions.from_config({
    "engine": "spark",
    "python_batch_rows": 4096,
    "state_max_bytes": 8 * 1024 * 1024,
})
identity = FrameSpec(row_keys=("customer_id", "event_id"), target="label")
```

Engine values are `pandas`, `polars` and `spark`; `databricks` names a platform,
not an engine. Unknown configuration fields and nonpositive/noninteger limits
are rejected. These immutable objects declare requirements; they do not apply
memory limits themselves. The Spark FeatureEngineer entry point validates actual
key values as described below; state/model budgets belong to later execution stages.

## Keyed Spark FeatureEngineer entry point

Use a single Spark dataframe with explicit identity columns. The initial path
supports flat scalar schemas and native operations that preserve rows. Nested
array/map/struct, Decimal and other unsupported types are rejected. Column names
must be unambiguous under the session's case-sensitivity setting.

This working example exercises the input boundary with an empty pipeline;
it does not perform feature transformations:

```python
from skyulf.core.execution import ExecutionOptions, FrameSpec
from skyulf.preprocessing.pipeline import FeatureEngineer

data = spark.createDataFrame(
    [(1, 10.0, 0), (2, 20.0, 1)],
    ["customer_id", "amount", "label"],
).repartition(2)
engineer = FeatureEngineer(
    [],
    frame_spec=FrameSpec(row_keys=("customer_id",), target="label"),
    execution_options=ExecutionOptions(engine="spark"),
)
training, metrics = engineer.fit_transform(data)
inference = engineer.transform(data.drop("label"))
assert inference.columns == ["customer_id", "amount"]
assert metrics["summary"]["rows_in"] is None
```

Both options are required for Spark. The target must exist during fit when
declared; it can be absent during transform. Composite keys must be unique and
non-null on each input. Separate Spark `(X, y)` pairs and `SplitDataset` inputs
are rejected; label alignment never depends on partition or row position.
`target_column`, if passed to fit, must agree with `FrameSpec.target`.
The `on_split` callback is unavailable on this single-frame entry point.

Automatic feature selection excludes keys and target. Explicit `columns` cannot
select either. Names are exact and literal, including dots/backticks. Raw inputs
produce raw Spark outputs; wrapped inputs retain their wrapper convention.

Every step must declare both fit and apply support before fitting starts. Apply
also checks all fitted steps before execution. This path accepts native,
row-preserving operations with row-local apply behavior. Worker Python, window,
filtering and expansion paths require later implementations. No built-in Spark
FE node is enabled yet; unsupported nodes fail before any validation action or fit.
Custom native nodes use the existing engine-keyed dispatcher mapping with a
`"spark"` implementation. It receives the full native frame, `y=None` and the
resolved feature names in `config["columns"]`; the returned frame retains keys
and any target. Fitted parameters must contain what its applier needs.

### Validation actions and metrics

Key validation performs a distributed group/count and returns at most one
invalid group to the driver. After each step, distributed multiset comparisons
check that the protected key/target projection is unchanged. These checks detect
changed labels, keys, dropped rows and duplicates even when a custom node claims
to preserve rows. They can scan/shuffle the data and execute the lazy plan again;
this initial correctness path is not a claim of optimized cluster performance.
Only a bounded validation result is collected, never the whole dataset.

The default metrics do not run a separate dataframe `count()` or local memory
profiler. `rows_in`, `rows_out`, `fit_time` and `peak_memory_bytes` are `None`,
meaning unknown. Per-step `driver_elapsed_seconds` measures caller-observed time
including validation; it is not distributed CPU time or cluster peak memory.
Explicit distributed aggregate metrics are not implemented yet.
The local `get_data_stats` helper rejects Spark input; use the Spark FE metrics.

Transform preserves key identity, not physical row order, including when
`preserve_rows=True`. Consumers must match outputs by keys. A successful refit
replaces fitted steps; a failed refit leaves the last successful fitted steps
available. Saving portable learned state follows in SM-04.

### Inspecting declared support

A capability check distinguishes fit and apply and never falls back to pandas:

```python
from skyulf.core.capabilities import UnsupportedExecutionError, require_capability

try:
    require_capability("SimpleImputer", "fit", "spark", config={"strategy": "mean"})
except UnsupportedExecutionError as error:
    print(error.node_type, error.operation, error.engine, error.reason)
```

This currently reports unsupported: no built-in Spark node has been enabled.
Existing local pipelines continue to work; this new preflight is not yet wired
into their execution. A local node without an explicit declaration also fails
this new query, even though its existing local pipeline path remains available.

### Declaring custom operation support

Node authors can pass an immutable `execution_capabilities` tuple to
`NodeRegistry.register`. Each `ExecutionCapability` records:

| Field | Meaning |
| --- | --- |
| `engine`, `operation` | Exact engine and `fit` or `apply` |
| `execution_kind` | `native`, `python_batch` or `local` |
| `row_effect` | `preserve`, `filter` or `expand` |
| `context` | `row`, `group`, `window` or `global` |
| `codec_version` | Optional positive version of the fitted-state codec |
| `config_match` | Immutable tuple of required key/scalar-value pairs |

For example, `config_match=(("strategy", "mean"),)` allows only an explicit
mean strategy; absent keys do not match. Normalize node defaults before querying.
Selectors compare types as well as values, so `True` is not treated as integer 1.
Aliases of the same calculator share declarations; subclasses must declare
their own support. Metadata is an implementation promise: it does not generate
Spark code, certify compatibility or make a window transform batch-independent.

## Troubleshooting and support boundaries

- **Java gateway fails:** check Java 17 and the current shell's `JAVA_HOME`.
- **Worker cannot find Python:** use the same interpreter for driver and workers;
  the test fixture sets `PYSPARK_PYTHON` explicitly.
- **Windows access denied at shutdown:** a process sandbox can prevent Spark's
  child-process cleanup. A passing calculation does not prove clean shutdown;
  use an environment that permits the owned JVM processes to terminate.
- **Spark dependency unavailable:** install the optional `skyulf-core[spark]`
  dependency in a suitable environment, or use the repository requirements above.
- **Unsupported node:** the keyed entry point is available, but built-in native
  FE nodes are still being implemented. Installing Spark does not enable them.

MLflow tracking, Unity Catalog registration, Databricks jobs, serving endpoints
and Bundle templates are separate integration stages. This page will gain
tested fit/apply and inference examples as those capabilities are implemented.
