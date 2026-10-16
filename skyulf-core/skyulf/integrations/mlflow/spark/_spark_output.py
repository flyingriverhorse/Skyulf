"""Internal certified Spark output transport without changing local pyfunc results."""

from collections.abc import Callable
from typing import Any

import numpy as np
import pandas as pd

from ....inference.fitted_pipeline import has_outlier_steps

SPARK_OUTPUT_PARAM = "spark_output"
# Marks a null in a mixed string column; the driver turns it back into SQL null.
SPARK_NULL_TEXT = "␀null␀"
SPARK_BATCH_ROWS_PARAM = "spark_batch_rows"
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


def spark_udf_params(prediction_batch_rows: int) -> dict[str, Any]:
    """Enable the Spark output transport with the configured model-call batch size."""
    return {SPARK_OUTPUT_PARAM: True, SPARK_BATCH_ROWS_PARAM: prediction_batch_rows}


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
    enabled = (params or {}).get(SPARK_OUTPUT_PARAM, False)
    if type(enabled) is not bool:
        raise ValueError("Spark output transport parameter must be a boolean.")
    if enabled and certificate is None:
        raise ValueError("Spark output transport requires a certified partition-safe artifact.")
    return enabled


def nullable_outputs(artifact: Any) -> bool:
    """Report whether saved scoring rules can leave excluded rows null in the output.

    A pipeline can exclude rows through ``project_scoring`` or fitted outlier
    filters. A set carries exclusions when a component schema has ``scoring_status``.
    """
    pipeline = getattr(artifact, "pipeline", None)
    if pipeline is not None:
        return pipeline.config.get("project_scoring") is not None or has_outlier_steps(artifact)
    return any(
        column.name == "scoring_status"
        for component in artifact.manifest.components
        for column in component.output_schema
    )


def spark_transport_dtype(dtype: str, nullable: bool) -> str:
    """Return the dtype one output column uses between Spark workers and the driver.

    Nullable int64 and bool columns (models with saved scoring rules) travel as
    float64 with NaN for null, because MLflow's struct conversion cannot carry
    their missing values; the driver casts them back.
    """
    return "float64" if nullable and dtype in ("int64", "bool") else dtype


def prepare_spark_output(
    frame: pd.DataFrame, schema: tuple, enabled: bool, *, nullable: bool = False
) -> pd.DataFrame:
    """Preserve absent outcomes across MLflow's Spark struct conversion.

    MLflow 3.16 recognizes all-None columns but stringifies pandas.NA and None
    in mixed columns. Mixed-null strings therefore carry ``SPARK_NULL_TEXT``,
    and nullable numeric/bool columns carry NaN; the driver turns both back
    into SQL nulls. Rule float outputs and predictions are always finite, so NaN
    only ever means null. A real string equal to ``SPARK_NULL_TEXT`` is rejected,
    because the driver would read it as null.
    """
    if not enabled:
        return frame
    result = frame.copy()
    for column in schema:
        values = result[column.name]
        missing = values.isna()
        if column.dtype == "string":
            if bool(values.eq(SPARK_NULL_TEXT).fillna(False).any()):
                raise ValueError(
                    f"Output column {column.name!r} contains the reserved null marker "
                    f"{SPARK_NULL_TEXT!r}; Spark output cannot carry this value."
                )
            if missing.all():
                result[column.name] = pd.Series(None, index=result.index, dtype=object)
            elif missing.any():
                result[column.name] = values.astype(object).where(~missing, SPARK_NULL_TEXT)
        elif nullable and column.dtype in ("float64", "int64", "bool"):
            result[column.name] = _float_transport(values, column.dtype)
    return result


def _float_transport(values: pd.Series, dtype: str) -> pd.Series:
    """Carry nullable numbers and booleans as float64, rejecting integers floats cannot hold."""
    # Check the integers before the cast: float64 already rounds values beyond 2**53.
    if dtype == "int64" and any(abs(int(value)) > 2**53 for value in values.dropna()):
        raise ValueError("Spark output cannot carry nullable int64 values beyond 2**53 exactly.")
    numbers = values.astype("Float64")
    return pd.Series(
        numbers.to_numpy(dtype="float64", na_value=np.nan), index=values.index, name=values.name
    )
