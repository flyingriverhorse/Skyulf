# Adding a compute engine

Skyulf has pandas and Polars frame engines and an explicit distributed Spark
path. Adding an engine requires an input adapter and node implementations for
the operations it will support. Registering an engine does not make existing
preprocessing or sklearn estimators run on it automatically.

Start with [engine mechanics](engine_mechanics.md),
[engine data paths](engine_data_paths.md) and the existing
[Spark execution contract](../user_guide/spark.md). Use the Spark implementation
as the reference for a distributed engine; it preserves distributed input and
rejects implicit materialization into pandas/NumPy.

## Register the adapter

`BaseEngine` and `EngineRegistry` live in `skyulf.engines.registry`.
`EngineRegistry.register(name, engine_cls)` registers an engine class.
Implement its actual class-method contract: `ensure_available`, `is_compatible`,
`wrap`, `from_pandas`, `to_numpy` and `create_dataframe`. A method that cannot
safely support a distributed operation should reject it explicitly.

Use `SkyulfDataFrame` for bounded frame wrappers and `DistributedDataFrame` for
distributed wrappers, both from `skyulf.engines.protocol`. The distributed
contract exposes columns, schema, projection and `to_native`; it does not promise
an eager row count or conversion to a driver array. The caller owns the runtime
session and explicit data actions.

## Implement supported operations

Nodes use Calculator/Applier pairs: fit returns learned state, and apply reuses
that state. Add the new engine's implementation to the node's existing
engine-keyed dispatcher. Do not introduce an unconditional pandas fallback for
an unknown or unsupported engine.

For distributed preprocessing, specify separate fit/apply capabilities, row and
key preservation, context requirements, and bounded learned-state transport.
A transformation requiring complete groups or ordered history needs an execution
path that supplies that context. Independent worker batches cannot establish it
by themselves. See [preprocessing context](../user_guide/preprocessing_context.md).

`SklearnBridge` converts supported bounded frame data for sklearn. It rejects
Spark data rather than collecting the full table into driver memory. Distributed
model prediction is a separate adapter with explicit worker dependency, input
schema and memory contracts; adding a preprocessing engine does not add model
training or prediction support automatically.

## Verify and document the boundary

Use focused checks for engine detection, schema/projection, fit/save/apply parity,
keys, nulls, empty requests and unsupported operations. For a distributed engine,
include multiple partitions and assert that the adapter does not collect the
whole input. Validate the actual runtime and dependencies used in deployment.

Update optional dependency declarations and the public support guide for the
specific implemented nodes. Keep unsupported requests explicit while adding
further native implementations incrementally. Backend ingestion or frontend
engine controls require their own changes only when those consumers need to
expose the new engine.
