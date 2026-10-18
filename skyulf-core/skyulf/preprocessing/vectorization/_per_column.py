"""Independent Count/TF-IDF vocabularies built from their existing flat artifacts."""

import logging
from collections.abc import Callable, Mapping
from typing import Any

import pandas as pd
import polars as pl

from .._fitted_validation import _columns, local_boolean, local_state_fields
from ._common import (
    _drop_and_concat,
    _drop_and_concat_polars,
    _join_text_columns_polars,
    _vectorizer_dense,
    _vectorizer_transform_to_frame,
    _vectorizer_uses_callbacks,
    _warn_large_output,
    validate_text_output_names,
)

logger = logging.getLogger(__name__)


def column_mode(config: Mapping[str, Any]) -> str:
    """Require a supported explicit mode while preserving the joined default."""
    mode = config.get("column_mode", "joined")
    if not isinstance(mode, str) or mode not in ("joined", "per_column"):
        raise ValueError("Text vectorizer column_mode must be 'joined' or 'per_column'.")
    return mode


def build_per_column_artifact(
    config: dict, frame: pd.DataFrame, columns: list[str], build: Callable
) -> dict:
    """Fit ordinary single-source artifacts and combine only their saved feature layout."""
    children = {}
    output_columns = []
    for col in columns:
        child = build(config, frame, [col])
        if not child:
            raise ValueError(
                f"Text column {col!r} produced an empty vocabulary in per_column mode; "
                "check its text, stop words and n-gram settings."
            )
        children[col] = child
        output_columns.extend(child["output_columns"])
    state = {
        "type": children[columns[0]]["type"],
        "column_mode": "per_column",
        "columns": columns,
        "output_columns": output_columns,
        "drop_original": bool(config.get("drop_original", False)),
        "column_artifacts": children,
    }
    warning = _warn_large_output(len(output_columns))
    if warning:
        logger.warning(warning)
    return state


def validate_per_column_state(raw: dict, kind: str, validate_child: Callable) -> bool:
    """Inspect exact nonrecursive child bindings without invoking fitted estimators."""
    if not isinstance(raw, dict) or "column_mode" not in raw:
        return False
    fields = {
        "type",
        "column_mode",
        "columns",
        "output_columns",
        "drop_original",
        "column_artifacts",
    }
    local_state_fields(raw, kind, fields)
    if raw["column_mode"] != "per_column":
        raise ValueError("Unsupported saved text column_mode.")
    columns = _columns(raw["columns"])
    if not columns:
        raise ValueError("Per-column vectorization requires saved source columns.")
    local_boolean(raw["drop_original"], "drop_original")
    children = raw["column_artifacts"]
    if type(children) is not dict or set(children) != set(columns):
        raise ValueError("Saved per-column artifacts must match the source columns.")
    _validate_child_bindings(raw, validate_child)
    return True


def _validate_child_bindings(raw: dict, validate_child: Callable) -> None:
    """Keep each flat vocabulary, drop flag and feature block attached to its source."""
    output = []
    for col in raw["columns"]:
        child = raw["column_artifacts"][col]
        if type(child) is not dict or not child or "column_mode" in child:
            raise ValueError("Saved per-column vocabularies must be nonempty flat artifacts.")
        validate_child(child)
        if child["columns"] != [col] or child["drop_original"] != raw["drop_original"]:
            raise ValueError("Saved per-column source or drop setting disagrees.")
        output.extend(child["output_columns"])
    if raw["output_columns"] != output:
        raise ValueError("Saved per-column output order disagrees with child vocabularies.")
    validate_text_output_names(pd.DataFrame(columns=raw["columns"]), raw, raw["columns"])


def per_column_uses_callbacks(state: dict) -> bool:
    """Apply the existing callback exclusion to every independent saved estimator."""
    return any(
        _vectorizer_uses_callbacks(child["vectorizer_object"])
        for child in state["column_artifacts"].values()
    )


def _require_sources(X: Any, state: dict) -> None:
    """Do not silently change a fitted source's meaning when a request omits a field."""
    missing = [col for col in state["columns"] if col not in X.columns]
    if missing:
        raise ValueError(f"Per-column vectorizer requires fitted source columns: {missing}.")
    validate_text_output_names(X, state, state["columns"])


def apply_per_column_pandas(X: pd.DataFrame, y: Any, state: dict) -> tuple[pd.DataFrame, Any]:
    """Compute every block from original sources before dropping and attaching once."""
    _require_sources(X, state)
    blocks = []
    for col in state["columns"]:
        child = state["column_artifacts"][col]
        blocks.append(
            _vectorizer_transform_to_frame(
                X, child["vectorizer_object"], [col], child["output_columns"]
            )
        )
    encoded = pd.concat(blocks, axis=1)
    return _drop_and_concat(X, encoded, state["columns"], state["drop_original"]), y


def apply_per_column_polars(X: Any, state: dict) -> Any:
    """Keep native string columns in Polars, using the existing fallback for other dtypes."""
    _require_sources(X, state)
    blocks = []
    for col in state["columns"]:
        text = _join_text_columns_polars(X, [col])
        if text is None:
            return None
        child = state["column_artifacts"][col]
        dense = _vectorizer_dense(
            text.to_list(), child["vectorizer_object"], len(child["output_columns"])
        )
        blocks.append(pl.from_numpy(dense, schema=child["output_columns"]))
    encoded = blocks[0]
    for block in blocks[1:]:
        encoded = encoded.hstack(block)
    return _drop_and_concat_polars(X, encoded, state["columns"], state["drop_original"])
