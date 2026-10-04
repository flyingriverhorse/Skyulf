"""Small evaluation-population records shared by preprocessing and model scoring."""

from copy import deepcopy
from typing import Any


def record_coverage(
    input_rows: int | None,
    scored_rows: int,
    steps: list[dict[str, Any]] | None = None,
    *,
    reason: str | None = None,
    step_name: str | None = None,
) -> dict[str, Any]:
    """Describe evaluation exclusions without treating new synthetic rows as observations."""
    if scored_rows < 0 or (input_rows is not None and scored_rows > input_rows):
        if step_name is not None:
            raise ValueError(
                f"Evaluation preprocessing step {step_name!r} changed row count "
                f"from {input_rows} to {scored_rows}; steps cannot add evaluation rows."
            )
        raise ValueError("Evaluation coverage requires 0 <= scored_rows <= input_rows.")
    result: dict[str, Any] = {
        "input_rows": input_rows,
        "scored_rows": scored_rows,
        "excluded_rows": None if input_rows is None else input_rows - scored_rows,
    }
    if input_rows is None:
        result["reason"] = reason or "Original evaluation population unavailable."
    if steps is not None:
        result["steps"] = deepcopy(steps)
    return result


def transform_evaluation(
    preprocessor: Any,
    X: Any,
    y: Any,
    *,
    coverage_out: dict[str, Any] | None = None,
    allow_empty: bool = False,
) -> tuple[Any, Any, dict[str, Any]]:
    """Transform an evaluation pair once and disclose the actual eligible population."""
    input_rows = len(X)
    X_t, y_t = (X, y) if preprocessor is None else preprocessor.transform(X, y)
    recorded = getattr(preprocessor, "last_transform_coverage_", {})
    coverage = record_coverage(input_rows, len(X_t), recorded.get("steps"))
    if coverage_out is not None:
        coverage_out.update(coverage)
    if y_t is None or len(y_t) != len(X_t):
        raise ValueError("Evaluation preprocessing must retain aligned feature and target rows.")
    if not len(X_t) and not allow_empty:
        raise ValueError("No eligible rows remain for evaluation after configured preprocessing.")
    return X_t, y_t, coverage
