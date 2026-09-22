# Spark: development setup and current support

Spark support is under development for 0.9.0. The optional dependency and
runtime tests, Spark engine adapter and keyed FeatureEngineer entry point are
available in the development checkout.
**Native SimpleImputer (mean/constant) and StandardScaler support fit and apply.**
Their fitted pipeline can be exported and restored across pandas, Polars and Spark.
The [standalone inference bundle](inference_bundles.md) packages this state with
a fitted model and validates local raw/features prediction. The Spark runner
applies native FE and runs a Python regression model in worker batches.
Other native nodes and distributed classification remain under development.
Installing the extra does not convert a pandas/Polars pipeline to Spark.

[How inference works](inference_flow.md) explains how the saved FE and
model are reused, with diagrams for local, native Spark and worker FE paths.

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

Only one aggregate row is returned to Python. Supported native FE operations
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

Validation covers classic PySpark 4.0.3 locally and a limited live Databricks
serverless Spark Connect 4.2.0 inference path. The latter loads a Polars-trained
regression bundle from Unity Catalog and applies mean imputation/StandardScaler
in both inference modes. It does not certify every Spark fit operation or compute
type. Engine detection does not authorize local-only FE nodes on Spark input.

Serverless hides `spark.sql.caseSensitive`. Skyulf handles that specific
unavailable-configuration condition with conservative case-insensitive collision
checks; unrelated connection and permission failures still propagate.

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
key values as described below and enforces `state_max_bytes` for nodes declaring
the portable codec. Model budgets belong to later inference stages.

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
filtering and expansion paths require later implementations. SimpleImputer
mean/constant and StandardScaler are available; unsupported nodes or strategies fail before any
validation action or fit. The runner normalizes registered node defaults before
checking capabilities, so an omitted SimpleImputer strategy means `mean`, and
StandardScaler defaults to `with_mean=True, with_std=True`.
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
available. The portable learned-state codec is described below.

## Native SimpleImputer: fit once, apply without relearning

The following pipeline learns the mean on training data and applies that same
value to an unlabeled batch. Use a caller-owned Spark session as above:

```python
from skyulf.core.execution import ExecutionOptions, FrameSpec
from skyulf.preprocessing.pipeline import FeatureEngineer

train = spark.createDataFrame(
    [(1, 1.0, 0), (2, None, 1), (3, 3.0, 0)],
    "customer_id long, amount double, label long",
).repartition(2)
engineer = FeatureEngineer(
    [{"name": "fill", "transformer": "SimpleImputer",
      "params": {"columns": ["amount"], "strategy": "mean"}}],
    frame_spec=FrameSpec(row_keys=("customer_id",), target="label"),
    execution_options=ExecutionOptions(engine="spark", state_max_bytes=8192),
)
training, metrics = engineer.fit_transform(train)
batch = spark.createDataFrame(
    [(10, 100.0), (11, None)], "customer_id long, amount double",
)
output = engineer.transform(batch)
# Collect only this tiny demonstration; production output remains a Spark frame.
assert [row.amount for row in output.orderBy("customer_id").collect()] == [100.0, 2.0]
```

`mean` fit aggregates means and missing counts together, returning one statistics
row to the driver. `constant` fit also aggregates missing counts for the artifact;
the fill value itself is supplied in `fill_value`. Both null and floating NaN
count as missing. Direct node apply constructs native Spark expressions without
a data action, Python UDF or pandas conversion. FeatureEngineer additionally
performs the key/target validation actions described above.

Selection and dtype rules:

- Explicit `columns=[]` is a no-op. Omitted columns select numeric mean
  candidates automatically, excluding all-missing, constant and binary columns.
  Automatic detection adds min/max, distinct-count and binary checks to the same
  aggregate query; Spark may shuffle data internally. Explicit columns avoid
  this selection cost and can include binary or constant numeric columns.
- The pipeline protects keys and target. A direct calculator has no `FrameSpec`;
  specify its feature columns explicitly to avoid learning from identifiers.
- Mean supports Spark byte/short/integer/long/float/double columns. Mean filling
  can promote integers to double; it does not promise exact large-integer output.
  Explicitly selected all-missing numeric columns retain an undefined mean and
  remain missing. Spark/Polars use `None` for that mean; pandas can use NaN.
- Numeric constant defaults to zero. String and boolean columns require an
  explicit compatible string or boolean value. Integer constants must fit
  signed 64-bit integers; integer-only filling preserves large integer values.
  Decimal, date/time and nested imputation are not supported.
- Non-finite computed means are rejected. Median and most-frequent/mode remain
  unsupported on Spark; there is no local fallback. Exact feature names and
  Spark's case-sensitivity setting must not produce ambiguous columns.

Learned state can also cross engines explicitly. This example fits on pandas and
applies to Spark; swapping the fit/apply engines follows the same codec contract:

```python
import pandas as pd
from skyulf.core.portable_state import encode_state, decode_state
from skyulf.preprocessing.imputation.simple import (
    SimpleImputerCalculator, SimpleImputerApplier,
)

state = SimpleImputerCalculator().fit(
    pd.DataFrame({"amount": [1.0, None, 3.0]}),
    {"columns": ["amount"], "strategy": "mean"},
)
payload = encode_state("SimpleImputer", state, max_bytes=8192)
_, restored = decode_state(payload, max_bytes=8192)
batch = spark.createDataFrame([(100.0,), (None,)], "amount double")
output = SimpleImputerApplier().apply(batch, restored)
assert [row.amount for row in output.collect()] == [100.0, 2.0]
```

Only the small artifact crosses engines. The Spark dataset remains distributed.
Existing input column order is retained; direct apply appends missing learned
columns when a usable fill value exists. The keyed pipeline requires configured
input feature columns to be present. Output rows must still be matched by keys.

## Native StandardScaler: center and scale

StandardScaler learns per-column statistics on Spark and applies them with native
expressions. Its population variance matches the local node's `ddof=0` convention,
as documented by [scikit-learn](https://scikit-learn.org/stable/modules/generated/sklearn.preprocessing.StandardScaler.html).
It uses Spark's native
[`var_pop`](https://spark.apache.org/docs/latest/api/python/reference/pyspark.sql/api/pyspark.sql.functions.var_pop.html)
on shifted values to reduce numerical error.

```python
from skyulf.core.execution import ExecutionOptions, FrameSpec
from skyulf.preprocessing.pipeline import FeatureEngineer

train = spark.createDataFrame(
    [(1, 1.0, 0), (2, 3.0, 1)], "id long, amount double, label long",
).repartition(2)
scaler = FeatureEngineer(
    [{"name": "scale", "transformer": "StandardScaler",
      "params": {"columns": ["amount"], "with_mean": True, "with_std": True}}],
    frame_spec=FrameSpec(row_keys=("id",), target="label"),
    execution_options=ExecutionOptions(engine="spark", state_max_bytes=8192),
)
scaled_train, _ = scaler.fit_transform(train)
assert [r.amount for r in scaled_train.orderBy("id").collect()] == [-1.0, 1.0]
batch = spark.createDataFrame([(10, 5.0)], "id long, amount double")
assert scaler.transform(batch).first().amount == 3.0
```

The same mean/variance/scale artifact can be encoded with
`encode_state("StandardScaler", state, max_bytes=...)` and applied using
`StandardScalerApplier` on pandas, Polars or Spark. State list positions follow
`columns`; apply matches names and retains input column order. Row order is not
an identity guarantee.

| `with_mean` | `with_std` | Apply formula | Learned statistics |
| --- | --- | --- | --- |
| True | True | `(x - mean) / scale` | mean, population variance, scale |
| True | False | `x - mean` | mean; variance/scale are None |
| False | True | `x / scale` | mean, population variance, scale |
| False | False | unchanged | mean/variance/scale are None |

Fit performs at most two distributed aggregate queries. The first finds numeric
reference values and validates selected input; the second calculates means and
population variances of deviations from those references. This avoids accumulating
the same large offset in every addition. A range spanning zero keeps a zero
reference; a same-sign range uses its endpoint nearest zero. Each query returns one O(columns) row,
never training samples. With both flags disabled, only the first query is needed.
Automatic selection adds the same binary/constant checks as the imputer. These
queries and the pipeline's identity checks can scan or shuffle data; this is not
a single-pass training algorithm.

- Null/NaN observations are excluded from fitting. Apply preserves missing
  values. Explicit all-missing columns learn NaN statistics; later observed
  values become NaN when an enabled operation uses those statistics.
- Constant and numerically near-constant features receive unit scale using the
  local sklearn error-bound rule. A single observation has zero variance and
  unit scale. Imported zero scales are also treated as one during apply.
- `columns=[]` is a no-op. Explicit feature selection on zero training rows
  raises an error; automatic selection on empty data has no eligible features.
  Direct apply skips absent learned columns, matching the local scaler; the
  keyed pipeline requires its configured feature columns to exist.
- Primitive numeric Spark types and explicitly selected booleans are supported.
  Enabled operations produce double columns; disabling both retains original
  types. Automatic selection skips booleans. Strings, Decimal and nested feature
  types are rejected, without casting text or falling back to pandas.
- Selected training infinities and overflowing aggregates are rejected. Double
  conversion can lose precision for very large integers. Distributed reduction
  order can still cause floating-point differences: ordinary parity fixtures use
  `rtol=1e-10, atol=1e-12`; the large-offset/small-spread fixture allows `rtol=0.002`
  because representable spacing near `1e12` is significant relative to its spread.

Direct apply is lazy, with no Python UDF or data action. FeatureEngineer still
executes the identity checks described above. This implementation has been tested
on the local PySpark runtime; Databricks/Connect validation remains a later gate.

## Portable learned state

The development codec transports a small learned-parameter dictionary as
versioned UTF-8 JSON bytes. It currently supports `StandardScaler` and
`SimpleImputer` with `mean` or `constant` strategies, including empty no-op
artifacts. This example works in the base environment without Spark:

```python
import pandas as pd

from skyulf.core.execution import ExecutionOptions
from skyulf.core.portable_state import decode_state, encode_state
from skyulf.preprocessing.imputation.simple import (
    SimpleImputerApplier,
    SimpleImputerCalculator,
)

training = pd.DataFrame({"amount": [1.0, None, 5.0]})
params = SimpleImputerCalculator().fit(
    training, {"columns": ["amount"], "strategy": "mean"},
)
budget = ExecutionOptions(engine="spark").state_max_bytes
payload = encode_state("SimpleImputer", params, max_bytes=budget)
node_type, restored = decode_state(payload, max_bytes=budget)
assert node_type == "SimpleImputer"
result = SimpleImputerApplier().apply(pd.DataFrame({"amount": [None]}), restored)
assert result["amount"].tolist() == [3.0]
```

The v1 envelope contains `format_version`, `codec_version`, `node_type`,
`ordered_columns`, `learned_parameters` and `semantic_digest`. The original
artifact retains its options, such as scaler centering/scaling flags. Unknown
versions, nodes, fields, malformed tags, duplicate columns and inconsistent
array lengths/counts are rejected. Actual input/output schemas and producer
metadata are not inferred from statistics; they belong to later bundle metadata.

Scalar tags distinguish integers, floats, strings, booleans and nulls. Integer
tags use decimal strings to preserve large category values. Float tags preserve
finite values and negative zero with hexadecimal notation, and represent NaN
and positive/negative infinity explicitly. NumPy scalar equivalents normalize
to Python scalars; arrays, estimators, sessions and arbitrary objects are rejected.
NaN bit patterns are not separate learned semantics. Column/list order is
preserved; dictionary insertion order and JSON whitespace are not significant.

The semantic SHA-256 checksum detects changed content; it is not a signature
or proof of a trusted producer. It uses canonical tagged values, with no
arbitrary-object `repr` or pickle fallback. Existing pipeline pickle save/load
and fingerprint algorithms remain unchanged.

`encode_state` requires an explicit positive `max_bytes` budget.
`decode_state` defaults to 8 MiB and checks received byte length before parsing.
Pass the same custom budget at both ends when changing it. The limit applies to
wire bytes; it is not a bound on total Python process memory. The codec performs
no filesystem or network I/O. Callers own storage and transport of these bytes.

The Spark runner validates each declared v1 artifact through the codec and
enforces `ExecutionOptions.state_max_bytes` after fit and before transform
validation actions. Direct node calls validate state structure but leave wire
byte limits to explicit encode/decode callers. Use the FeatureEngineer methods
below to package an ordered pipeline. Worker model loading remains a later stage.
Codec tests also run without starting a JVM.

## Save and restore a fitted feature pipeline

`FeatureEngineer.export_state()` returns bytes for the complete supported FE
chain. `FeatureEngineer.from_state(...)` validates those bytes and restores the
learned transformations without fitting. Both methods work without Spark installed
when the destination is local. With a caller-owned `spark` session:

```python
import math
import pandas as pd

from skyulf.core.execution import ExecutionOptions, FrameSpec
from skyulf.preprocessing.pipeline import FeatureEngineer

engineer = FeatureEngineer([
    {"name": "fill", "transformer": "SimpleImputer",
     "params": {"columns": ["amount"], "strategy": "mean"}},
    {"name": "scale", "transformer": "StandardScaler",
     "params": {"columns": ["amount"]}},
])
engineer.fit_transform(pd.DataFrame({"amount": [1.0, None, 3.0]}))
payload = engineer.export_state()
# Application code may persist these bytes with Path(...).write_bytes(payload).
restored = FeatureEngineer.from_state(
    payload,
    frame_spec=FrameSpec(row_keys=("customer_id",)),
    execution_options=ExecutionOptions(engine="spark"),
)
batch = spark.createDataFrame(
    [(20, 3.0), (10, 1.0), (30, None)], "customer_id long, amount double",
)
output = restored.transform(batch.repartition(7))
# Collect only this three-row example; a production caller can write output as a table.
actual = {row.customer_id: row.amount for row in output.collect()}
expected = {10: -math.sqrt(1.5), 20: math.sqrt(1.5), 30: 0.0}
assert actual.keys() == expected.keys()
assert all(math.isclose(actual[key], value, abs_tol=1e-12)
           for key, value in expected.items())
assert restored.export_state() == payload
```

For pandas or Polars apply, omit `frame_spec` and either omit `execution_options`
or select that local engine explicitly. For Spark fit, pass `FrameSpec` and
`ExecutionOptions("spark")` to the initial FeatureEngineer as in the native
examples above. The training engine does not lock the saved rules to that engine.
Only learned parameters move; Spark apply keeps the batch distributed.

The envelope retains step order, names, resolved configuration and each node's
validated state. Learned column selection is frozen, including selections originally
made automatically. Restoring and then refitting uses these explicit columns;
create a new FeatureEngineer from the original configuration to rediscover features.
Statistics follow the saved column names/order; transforms retain input column order.
Match output rows by keys, since Spark does not guarantee physical row order.

Keys, target declarations and runtime objects are not saved. Spark loading requires
fresh row keys, optionally a target, and explicit execution options. These columns
cannot overlap learned features. Transform validates actual keys and the required
input schema before applying the saved rules. Neither loading nor exporting creates
a session, reads files, contacts a registry or runs a model.

The current supported chain contains only SimpleImputer `mean`/`constant` and
StandardScaler, including intentionally fitted empty/no-op pipelines. Unsupported
nodes, custom appliers, partial local fits and arbitrary Python objects fail
explicitly. This API exports feature transformations only; it is not a model bundle
or a general replacement for existing pipeline pickle persistence.

`ExecutionOptions.state_max_bytes` limits the entire pipeline envelope, not only
individual node artifacts. The default is 8 MiB; loading checks received byte length
before parsing. Compact UTF-8 wire encoding and escaped JSON have the same semantic
checksum. Unknown versions, malformed fields, configuration/state disagreement and
checksum mismatches are rejected. The checksum covers step order and configuration;
it detects corruption but does not authenticate a producer.

The runnable repository example also exercises Spark fit and application-owned
file persistence. From the repository root, with Java configured as above:

```powershell
.venv-spark/Scripts/python.exe skyulf-core/examples/spark_feature_engineering.py --state-path features.json
```

`skyulf-core/examples/spark_feature_engineering.py` owns and closes its local Spark
session. Its `run_example(spark, state_path)` function accepts an existing session.
The automated gate covers all nine fit/apply engine combinations, file round-trips,
repartitioning to 1/2/7 partitions, reversed inputs, repeated apply and a 10,000-row
fixture that rejects unbounded driver collection. Validation currently uses local
PySpark; Databricks/Connect validation is still pending.

## Model inference after native FE

Use `predict_spark(..., mode="native_features")` with a raw-input inference
bundle to apply the supported saved transformations natively and run a Python
regression model on Spark workers. The result is a Spark DataFrame containing
row keys and predictions. The full dataset remains distributed; pandas/NumPy
are used for individual worker model batches.

The [inference bundle guide](inference_bundles.md#native-spark-fe-and-worker-model-inference)
contains a working example, input/order checks and the distinction between
model chunk size and Arrow transport batches. Its
[legacy artifact section](inference_bundles.md#current-support-and-legacy-adapters)
explains how existing pipeline pickle and backend joblib artifacts relate to
the new bundle. Saving fitted FE with a model already existed; distributed
execution requires explicit support for the transformations in that artifact.

Distributed inference supports raw regression and classification in both modes,
with the documented portable FE restrictions. Local training can use pandas or
Polars. MLflow packaging and registry loading are available. Live Databricks
serverless regression parity, monthly Delta publication, writer coordination
and restricted-principal write denial have passed the selected SM-16 workflow.
This does not extend cloud validation to every node or classification path.

## Capability declarations

### Inspecting declared support

A capability check distinguishes fit and apply and never falls back to pandas:

```python
from skyulf.core.capabilities import UnsupportedExecutionError, require_capability

require_capability("SimpleImputer", "fit", "spark", config={"strategy": "mean"})
require_capability("StandardScaler", "apply", "spark", config={
    "with_mean": True, "with_std": True,
})
try:
    require_capability("SimpleImputer", "fit", "spark", config={"strategy": "median"})
except UnsupportedExecutionError as error:
    print(error.node_type, error.operation, error.engine, error.reason)
```

Mean and StandardScaler succeed; median reports unsupported. The query requires explicit defaults.
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
- **Unsupported node:** SimpleImputer mean/constant and StandardScaler are available. Other built-in
  native FE nodes are still being implemented; installing Spark does not enable them.

MLflow tracking, Unity Catalog registration, Databricks jobs, serving endpoints
and Bundle templates are separate integration stages. This page will gain
tested fit/apply and inference examples as those capabilities are implemented.
