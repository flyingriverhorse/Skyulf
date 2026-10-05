"""Internal certified Spark output transport without changing local pyfunc results."""

from collections.abc import Callable
from typing import Any

import pandas as pd

SPARK_OUTPUT_PARAM = "skyulf_spark_output"
SPARK_BATCH_ROWS_PARAM = "skyulf_spark_batch_rows"
DEFAULT_PREDICTION_BATCH_ROWS = 10_000


def spark_output_params() -> Any:
    """Declare an opt-in transport flag whose ordinary local prediction default is false."""
    from mlflow.types import (  # noqa: PLC0415  # ty: ignore[unresolved-import]
        ParamSchema,
        ParamSpec,
    )

    return ParamSchema(
        [
            ParamSpec(SPARK_OUTPUT_PARAM, "boolean", False),
            ParamSpec(SPARK_BATCH_ROWS_PARAM, "long", DEFAULT_PREDICTION_BATCH_ROWS),
        ]
    )


def validate_prediction_batch_rows(value: Any) -> int:
    """Reject ambiguous model-call limits, including booleans masquerading as integers."""
    if type(value) is not int or value <= 0:
        raise ValueError("prediction_batch_rows must be a positive integer.")
    return value


def score_prediction_batches(
    frame: pd.DataFrame,
    score: Callable[[pd.DataFrame], pd.DataFrame],
    params: dict[str, Any] | None,
    enabled: bool,
) -> pd.DataFrame:
    """Bound certified Spark model calls without claiming to bound Arrow allocation.

    Serverless manages the incoming Arrow batch. Its existing input and complete
    output still occupy worker memory; only calls into the fitted scorer are
    limited here. Ordinary local calls, including their empty-frame behavior,
    pass through unchanged. Positional slicing retains duplicate/nondefault
    pandas indices and concatenation preserves the original row order.
    """
    if not enabled:
        return score(frame)
    rows = validate_prediction_batch_rows(
        (params or {}).get(SPARK_BATCH_ROWS_PARAM, DEFAULT_PREDICTION_BATCH_ROWS)
    )
    if frame.empty:
        return score(frame)
    outputs = []
    for start in range(0, len(frame), rows):
        chunk = frame.iloc[start : start + rows]
        output = score(chunk)
        if not output.index.equals(chunk.index):
            raise ValueError("Spark prediction chunk changed row identity or order.")
        outputs.append(output)
    return pd.concat(outputs, axis=0)


def require_spark_output(
    params: dict[str, Any] | None,
    certificate: dict[str, Any] | None,
) -> bool:
    """Check transport opt-in before scoring; it cannot authorize an uncertified artifact."""
    enabled = params.get(SPARK_OUTPUT_PARAM, False) if params else False
    if type(enabled) is not bool:
        raise ValueError("Spark output transport parameter must be a boolean.")
    if enabled and certificate is None:
        raise ValueError("Spark output transport requires a certified partition-safe artifact.")
    return enabled


def prepare_spark_output(frame: pd.DataFrame, schema: tuple, enabled: bool) -> pd.DataFrame:
    """Preserve absent string outcomes across MLflow's Spark struct conversion.

    MLflow 3.16 recognizes all-None columns but stringifies pandas.NA to '<NA>'.
    Currently certified inference admits no mixed-null string outcomes: custom
    exclusions/composition are rejected by the safety gate. Reject that future
    case rather than silently converting missing outcomes into literal strings.
    """
    if not enabled:
        return frame
    result = frame.copy()
    for column in schema:
        if column.dtype != "string":
            continue
        missing = result[column.name].isna()
        if missing.all():
            result[column.name] = pd.Series(None, index=result.index, dtype=object)
        elif missing.any():
            raise ValueError("Spark output transport does not support mixed-null string outputs.")
    return result
