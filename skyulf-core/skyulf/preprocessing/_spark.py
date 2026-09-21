"""Keyed native Spark FE execution, isolated from local profiling and X/y alignment."""

import importlib
import time
from copy import deepcopy
from functools import reduce
from typing import Any

from ..core.capabilities import UnsupportedExecutionError, require_capability
from ..core.execution import ExecutionOptions, FrameSpec
from ..data.dataset import SplitDataset
from ..engines import SkyulfSparkWrapper, SparkEngine, get_engine
from ..registry import NodeRegistry
from ..utils import contains_spark_input


def use_spark(data: Any, options: ExecutionOptions | None, spec: FrameSpec | None) -> bool:
    """Require explicit Spark context and reject engine overrides that contradict input."""
    requested = options is not None and options.engine == "spark"
    if contains_spark_input(data) or requested:
        if not requested or spec is None:
            raise ValueError("Spark requires frame_spec and execution_options(engine='spark').")
        if not SparkEngine.is_compatible(data):
            raise TypeError(
                "FeatureEngineer requires a single Spark dataframe containing keys/target."
            )
        return True
    if spec is not None:
        raise ValueError("frame_spec is currently supported only for Spark execution.")
    _validate_local_engine(data, options)
    return False


def _validate_local_engine(data: Any, options: ExecutionOptions | None) -> None:
    """Check opt-in local engine selection without changing legacy defaults."""
    if options is None:
        return
    if isinstance(data, SplitDataset):
        for split in (data.train, data.test, data.validation):
            if split is not None:
                _validate_local_engine(split, options)
        return
    features = data[0] if isinstance(data, tuple) else data
    if get_engine(features).name != options.engine:
        raise ValueError("execution_options engine conflicts with the input engine.")


def _column(frame: Any, name: str) -> Any:
    """Resolve a literal identifier, including dots and embedded backticks."""
    return frame["`" + name.replace("`", "``") + "`"]


def _native(data: Any) -> Any:
    """Validate and unwrap a Spark dataframe without materializing it."""
    if not SparkEngine.is_compatible(data):
        raise TypeError("Expected a single Spark dataframe from each native FE step.")
    return data.to_native() if isinstance(data, SkyulfSparkWrapper) else data


def _validate_schema(frame: Any, spec: FrameSpec, *, training: bool) -> None:
    """Reject ambiguous or unsupported schemas before any validation action."""
    names = frame.columns
    resolved_names = _resolved_names(frame)
    if len(set(resolved_names)) != len(names):
        raise ValueError("Duplicate Spark column names are unsupported.")
    required = set(spec.row_keys)
    if training and spec.target is not None:
        required.add(spec.target)
    missing = required.difference(names)
    if missing:
        raise ValueError(f"Missing Spark columns: {sorted(missing)}")
    supported = {
        "byte",
        "short",
        "integer",
        "long",
        "float",
        "double",
        "string",
        "boolean",
        "date",
        "timestamp",
        "timestamp_ntz",
    }
    for field in frame.schema.fields:
        if field.dataType.typeName() not in supported:
            raise TypeError(
                f"Unsupported Spark dtype for {field.name}: {field.dataType.simpleString()}"
            )


def _validate_keys(frame: Any, spec: FrameSpec) -> None:
    """Check composite key uniqueness/nulls with one bounded validation result."""
    functions = importlib.import_module("pyspark.sql.functions")
    count_name = "__skyulf_count"
    existing = _resolved_names(frame)
    while count_name in existing:
        count_name += "_"
    grouped = frame.groupBy(*[_column(frame, name) for name in spec.row_keys]).agg(
        functions.count(functions.lit(1)).alias(count_name)
    )
    invalid = reduce(
        lambda a, b: a | b, [_column(grouped, name).isNull() for name in spec.row_keys]
    )
    if grouped.filter(invalid | (_column(grouped, count_name) > 1)).limit(1).collect():
        raise ValueError("Spark row_keys must be unique and non-null.")


def _resolved_names(frame: Any) -> list[str]:
    """Use session identifier rules for schema collisions and internal aliases."""
    if frame.sparkSession.conf.get("spark.sql.caseSensitive") == "true":
        return frame.columns
    return [name.lower() for name in frame.columns]


def _params(config: Any, spec: FrameSpec, frame: Any = None) -> dict[str, Any]:
    """Protect identity/target columns while resolving explicit or automatic features."""
    params = deepcopy(dict(config))
    protected = set(spec.row_keys) | {spec.target}
    columns = params.get("columns")
    _validate_feature_names(columns)
    if columns and any(name in protected for name in columns):
        raise ValueError("Spark feature columns cannot include row_keys or target.")
    if frame is not None:
        selected = columns or [name for name in frame.columns if name not in protected]
        if set(selected).difference(frame.columns):
            raise ValueError("Spark feature columns are missing or duplicated.")
        params["columns"] = selected
    if spec.target is not None:
        params["target_column"] = spec.target
    return params


def _validate_feature_names(columns: Any) -> None:
    """Reject malformed and duplicate explicit column declarations before actions."""
    if columns is None:
        return
    if not isinstance(columns, list):
        raise ValueError("Spark columns must be a list of feature names.")
    if any(not isinstance(name, str) for name in columns):
        raise ValueError("Spark feature names must be strings.")
    if len(set(columns)) != len(columns):
        raise ValueError("Spark feature columns are missing or duplicated.")


def _supports_native(declarations: Any, operation: str, params: dict) -> bool:
    """Match the implemented native preserving path, excluding worker/window paths."""
    for item in declarations:
        if (item.engine, item.operation, item.execution_kind, item.row_effect) != (
            "spark",
            operation,
            "native",
            "preserve",
        ):
            continue
        if operation == "apply" and item.context != "row":
            continue
        if item.matches_config(params):
            return True
    return False


def _preflight(steps: Any, spec: FrameSpec, *, training: bool) -> None:
    """Check every operation before any fit or distributed key-validation action."""
    operations = ("fit", "apply") if training else ("apply",)
    for step in steps:
        node_type = step["transformer"] if training else step["type"]
        params = _params(step.get("params", {}), spec)
        for operation in operations:
            require_capability(node_type, operation, "spark", config=params)
            declarations = vars(NodeRegistry.get_calculator(node_type))[
                "__execution_capabilities__"
            ]
            if not _supports_native(declarations, operation, params):
                raise UnsupportedExecutionError(
                    node_type,
                    operation,
                    "spark",
                    "This entry point requires native row-preserving FE.",
                )


def _check_output(before: Any, output: Any, spec: FrameSpec) -> Any:
    """Verify protected values and row identity after each declared preserving step."""
    after = _native(output)
    _validate_schema(after, spec, training=spec.target in before.columns)
    protected = list(spec.row_keys)
    if spec.target is not None and spec.target in before.columns:
        protected.append(spec.target)
    left = before.select(*[_column(before, name) for name in protected])
    right = after.select(*[_column(after, name) for name in protected])
    # Nullable metadata can change harmlessly, but identity/label types cannot.
    if left.dtypes != right.dtypes:
        raise ValueError("Spark step changed row_keys/target dtypes.")
    if left.exceptAll(right).unionByName(right.exceptAll(left)).limit(1).collect():
        raise ValueError("Spark step changed row_keys/target values or row membership.")
    return after


def fit_spark(data: Any, steps: Any, spec: FrameSpec) -> tuple[Any, dict, list[dict]]:
    """Fit/apply native steps transactionally, reporting unknown distributed metrics."""
    _preflight(steps, spec, training=True)
    current = _native(data)
    _validate_schema(current, spec, training=True)
    _validate_keys(current, spec)
    records = []
    step_metrics = {}
    for index, step in enumerate(steps):
        params = _params(step.get("params", {}), spec, current)
        node_type = step["transformer"]
        calculator = NodeRegistry.get_calculator(node_type)()
        applier = NodeRegistry.get_applier(node_type)()
        start = time.perf_counter()
        artifact = dict(calculator.fit(current, params))
        current = _check_output(current, applier.apply(current, artifact), spec)
        step_metrics[f"{index}:{step['name']}"] = {
            "driver_elapsed_seconds": time.perf_counter() - start,
            "rows_in": None,
            "rows_out": None,
            "peak_memory_bytes": None,
        }
        records.append(
            {
                "name": step["name"],
                "type": node_type,
                "applier": applier,
                "artifact": artifact,
                "params": params,
            }
        )
    summary = {"rows_in": None, "rows_out": None, "peak_memory_bytes": None, "fit_time": None}
    metrics = {"summary": summary, "steps": step_metrics, **summary}
    return _restore(current, data), metrics, records


def transform_spark(data: Any, steps: Any, spec: FrameSpec) -> Any:
    """Apply fitted native steps to labeled or unlabeled data using key identity."""
    _preflight(steps, spec, training=False)
    current = _native(data)
    _validate_schema(current, spec, training=False)
    _validate_keys(current, spec)
    for step in steps:
        _params(step["params"], spec, current)
        current = _check_output(current, step["applier"].apply(current, step["artifact"]), spec)
    return _restore(current, data)


def _restore(frame: Any, original: Any) -> Any:
    """Preserve the caller's raw-versus-wrapper return convention."""
    return SkyulfSparkWrapper(frame) if isinstance(original, SkyulfSparkWrapper) else frame
