"""Common prediction runtime copied unchanged into exported projects as scoring.py."""

import hashlib
import json
import math
import pickle
import re
from contextlib import AbstractContextManager, suppress
from datetime import date, datetime
from decimal import Decimal, InvalidOperation
from pathlib import Path
from typing import Any, cast

import numpy as np
import pandas as pd
from sklearn import config_context
from sklearn.base import is_classifier

from skyulf.engines.sklearn_bridge import SklearnBridge
from skyulf.modeling import apply_thresholds
from skyulf.preprocessing._exclusion import is_exclusion_step


def _dtype_spec(dtype: str) -> tuple[str, str]:
    """Recognize recorded pandas/Polars scalars without guessing object column contents."""
    lower = dtype.lower().removesuffix("[pyarrow]")
    if re.fullmatch(r"u?int(8|16|32|64)", lower):
        return "integer", lower
    if lower in {"float", "float32", "float64", "double"}:
        return "float", "float32" if lower in {"float", "float32"} else "float64"
    if lower.startswith("datetime"):
        return "datetime", _datetime_dtype(dtype)
    if lower.startswith("decimal"):
        return "decimal", dtype
    kind = {
        "bool": "boolean",
        "boolean": "boolean",
        "str": "string",
        "string": "string",
        "string[python]": "string",
        "date": "date",
    }.get(lower, "object")
    return kind, dtype


def _datetime_dtype(dtype: str) -> str:
    """Translate the explicit Polars time unit/time zone into a pandas datetime dtype."""
    match = re.fullmatch(r"Datetime\(time_unit='(ns|us|ms)', time_zone=(None|'[^']+')\)", dtype)
    if match:
        unit, zone = match.groups()
        suffix = "" if zone == "None" else ", " + zone.strip("'")
        return f"datetime64[{unit}{suffix}]"
    return "datetime64[ns]" if dtype.lower() == "datetime" else dtype


def _missing(value: Any) -> bool:
    """Recognize scalar missing sentinels without treating false or empty text as null."""
    return (
        value is None
        or value is pd.NA
        or value is pd.NaT
        or isinstance(value, (float, np.floating))
        and math.isnan(value)
    )


def _number(value: Any) -> Decimal:
    """Parse exact numeric text before choosing integer or floating representation."""
    if isinstance(value, (bool, np.bool_)):
        raise ValueError("Booleans are not numeric input.")
    parsed = Decimal(str(value))
    if not parsed.is_finite():
        raise ValueError("Numeric input must be finite.")
    return parsed


def _integer(value: Any, dtype: str) -> int:
    """Reject fractions, integer overflow and floating inputs whose precision is ambiguous."""
    if isinstance(value, (float, np.floating)) and abs(value) >= 2**53:
        raise ValueError("Large integers must be supplied as integer values or decimal text.")
    parsed = _number(value)
    if parsed != parsed.to_integral_value():
        raise ValueError("Integer input must not contain a fractional part.")
    limits = np.iinfo(np.dtype(dtype))
    if not limits.min <= parsed <= limits.max:
        raise ValueError("Integer input is outside the recorded dtype range.")
    return int(parsed)


def _floating(value: Any, dtype: str) -> float:
    """Reject nonfinite input and values which overflow the recorded floating width."""
    parsed = float(_number(value))
    if not math.isfinite(parsed) or abs(parsed) > np.finfo(np.dtype(dtype)).max:
        raise ValueError("Floating input is outside the recorded dtype range.")
    return parsed


def _boolean(value: Any) -> bool:
    """Accept explicit booleans and their conventional scalar encodings, never truthiness."""
    if isinstance(value, (bool, np.bool_)):
        return bool(value)
    if isinstance(value, (int, np.integer)) and value in (0, 1):
        return bool(value)
    if isinstance(value, str) and value.lower() in ("true", "false", "0", "1"):
        return value.lower() in ("true", "1")
    raise ValueError("Boolean input must be true, false, 0 or 1.")


def _date(value: Any) -> date:
    """Require an actual calendar date rather than interpreting numbers as timestamps."""
    if isinstance(value, date) and not isinstance(value, datetime):
        return value
    if isinstance(value, str):
        return date.fromisoformat(value)
    raise ValueError("Date input must be an ISO calendar date.")


def _datetime(value: Any, dtype: str) -> pd.Timestamp:
    """Preserve recorded time zones and reject numeric epoch guesses."""
    if not isinstance(value, (str, date, np.datetime64)):
        raise ValueError("Datetime input must be an ISO datetime.")
    parsed = pd.Timestamp(value)
    if pd.isna(parsed):
        raise ValueError("Datetime input must not be NaT text.")
    target = pd.api.types.pandas_dtype(dtype)
    zone = target.tz if isinstance(target, pd.DatetimeTZDtype) else None
    if zone is not None:
        return parsed.tz_localize(zone) if parsed.tz is None else parsed.tz_convert(zone)
    if parsed.tz is not None:
        raise ValueError("Timezone-aware input does not match the naive training dtype.")
    return parsed


def _decimal(value: Any, dtype: str) -> Decimal:
    """Keep decimal precision and reject values outside explicit Polars precision/scale."""
    parsed = _number(value)
    match = re.fullmatch(r"Decimal\(precision=(\d+), scale=(\d+)\)", dtype)
    if match:
        precision, scale = map(int, match.groups())
        parts = parsed.as_tuple()
        digits = "".join(map(str, parts.digits))
        trailing_zeros = len(digits) - len(digits.rstrip("0"))
        fractional_digits = max(0, -int(parts.exponent) - trailing_zeros) if parsed else 0
        if fractional_digits > scale or parsed.copy_abs() >= Decimal(10) ** (precision - scale):
            raise ValueError("Decimal input is outside the recorded precision or scale.")
    return parsed


def _value(value: Any, kind: str, dtype: str) -> Any:
    """Apply one explicitly recorded scalar type while preserving real nulls."""
    if _missing(value):
        return None
    if kind == "integer":
        return _integer(value, dtype)
    if kind == "float":
        return _floating(value, dtype)
    if kind == "boolean":
        return _boolean(value)
    if kind == "date":
        return _date(value)
    if kind == "datetime":
        return _datetime(value, dtype)
    if kind == "decimal":
        return _decimal(value, dtype)
    if kind == "string" and not isinstance(value, str):
        raise ValueError("String input must be text.")
    return value


def _input_value(value: Any, name: str, dtype: str) -> Any:
    """Attach actionable raw-column context to conversion failures in API and batch."""
    try:
        kind, target = _dtype_spec(dtype)
        return _value(value, kind, target)
    except (ValueError, TypeError, OverflowError, InvalidOperation) as exc:
        raise ValueError(
            f"Invalid input for column {name!r} (training dtype {dtype}): {exc}"
        ) from exc


def _typed_series(values: list, index: pd.Index, dtype: str) -> pd.Series:
    """Build nullable integer/bool columns without an intermediate floating conversion."""
    kind, target = _dtype_spec(dtype)
    if kind == "integer" and None in values:
        target = target.replace("uint", "UInt").replace("int", "Int")
    if kind == "boolean":
        target = "boolean" if None in values else "bool"
    if kind not in {"integer", "float", "boolean", "datetime"}:
        target = "object"
    return pd.Series(values, index=index, dtype=target)


def coerce_inputs(frame: pd.DataFrame, schema: list[dict[str, str]]) -> pd.DataFrame:
    """Validate required raw columns and convert only explicitly recorded training types."""
    if not isinstance(frame, pd.DataFrame) or not frame.columns.is_unique:
        raise ValueError("Input must be a DataFrame with unique columns.")
    required = [column["name"] for column in schema]
    missing = [column for column in required if column not in frame.columns]
    if missing:
        raise ValueError(f"Missing required input columns: {missing}")
    result = frame[required].copy()
    for column in schema:
        name, dtype = column["name"], column["dtype"]
        values = [_input_value(value, name, dtype) for value in frame[name]]
        try:
            result[name] = _typed_series(values, frame.index, dtype)
        except (ValueError, TypeError, OverflowError) as exc:
            raise ValueError(
                f"Invalid input for column {name!r} (training dtype {dtype}): {exc}"
            ) from exc
    return result


def build_request_model(directory: str | Path = Path(__file__).resolve().parent) -> Any:
    """Expose required nullable raw fields from schema.json without loading the model pickle."""
    from pydantic import (  # noqa: PLC0415 - API only
        ConfigDict,
        Field,
        create_model,
        field_validator,
    )

    schema = json.loads((Path(directory) / "schema.json").read_text(encoding="utf-8"))
    columns = {f"column_{index}": column for index, column in enumerate(schema)}
    annotations = {
        "integer": int | None,
        "float": float | None,
        "boolean": bool | None,
        "string": str | None,
        "date": date | None,
        "datetime": datetime | None,
        "decimal": Decimal | None,
        "object": Any,
    }

    def validate_value(value: Any, info: Any) -> Any:
        """Apply the same exact conversion used by batch before Pydantic casts scalar types."""
        column = columns[info.field_name]
        return _input_value(value, column["name"], column["dtype"])

    fields: dict[str, Any] = {
        name: (
            annotations[_dtype_spec(column["dtype"])[0]],
            Field(..., alias=column["name"], description=f"Training dtype: {column['dtype']}"),
        )
        for name, column in columns.items()
    }
    validators: dict[str, Any] = {
        "validate_value": field_validator("*", mode="before")(validate_value)
    }
    row = create_model(
        "InputRow",
        __config__=ConfigDict(extra="allow"),
        __validators__=validators,
        **fields,
    )
    return create_model("PredictionRequest", data=(list[row], ...))


class Predictor:
    """Load a trusted exported model once and share its scoring path across API and batch."""

    def __init__(self, directory: str | Path = Path(__file__).resolve().parent):
        """Verify the payload checksum before loading trusted pickle state."""
        directory = Path(directory)
        self.manifest = json.loads((directory / "manifest.json").read_text(encoding="utf-8"))
        self.schema = json.loads((directory / "schema.json").read_text(encoding="utf-8"))
        payload = (directory / "model.pkl").read_bytes()
        if hashlib.sha256(payload).hexdigest() != self.manifest["model_sha256"]:
            raise ValueError("Model checksum mismatch. Export the project again.")
        # Only load projects from a trusted producer; a checksum is not authentication.
        self.state = pickle.loads(payload)

    def predict(self, frame: pd.DataFrame) -> dict[str, Any]:
        """Score raw rows in their original order using the saved decision policy."""
        frame = coerce_inputs(frame, self.schema)
        labels = self.probability_labels()
        values, probabilities, reasons = self._score(frame, labels)
        result = {
            "predictions": values,
            "model_version": self.manifest["job_id"],
            "thresholds_applied": self.state["thresholds"],
        }
        if labels is not None:
            result["probabilities"] = probabilities
        if self.has_exclusions():
            result["scoring_status"] = [
                "predicted" if reason is None else "excluded" for reason in reasons
            ]
            result["exclusion_reason"] = reasons
        # Fail explicitly instead of emitting NaN/Infinity or an invalid JSON response.
        json.dumps(result, allow_nan=False)
        return result

    def _score(self, frame: pd.DataFrame, labels: list[str] | None) -> tuple[list, list, list]:
        """Predict eligible rows and place results back at their input positions."""
        size = len(frame)
        values: list = [None] * size
        probabilities: list = [None] * size
        if not size:
            return values, probabilities, [None] * size
        features, positions, reasons = self._features(frame)
        if not positions:
            return values, probabilities, reasons
        matrix = None if labels is None else self._probabilities(features, labels)
        predicted = self._predictions(features, matrix)
        if len(predicted) != len(positions):
            raise ValueError("Model prediction changed row count.")
        for index, position in enumerate(positions):
            values[position] = predicted[index]
            if labels is not None and matrix is not None:
                probabilities[position] = dict(zip(labels, matrix[index].tolist(), strict=True))
        return values, probabilities, reasons

    def has_exclusions(self) -> bool:
        """Report whether fitted outlier/invalid filters can exclude rows from scoring."""
        return any(is_exclusion_step(step) for step in self.state["feature_engineer"].fitted_steps)

    def _features(self, frame: pd.DataFrame) -> tuple[pd.DataFrame, list[int], list]:
        """Return model features, their input positions and a reason for each excluded row."""
        engineer = self.state["feature_engineer"]
        # sklearn exposes this generator-based context manager without a return
        # annotation; state its documented protocol at the package boundary.
        with cast(AbstractContextManager[None], config_context(transform_output="pandas")):
            if not self.has_exclusions():
                features = engineer.transform(frame, preserve_rows=True)
                return self._model_features(features), list(range(len(frame))), [None] * len(frame)
            # Same semantics as Databricks saved scoring: excluded rows stay with a reason.
            features, positions, reasons = engineer.transform_with_exclusions(frame)
        reasons = [None if pd.isna(reason) else str(reason) for reason in reasons]
        positions = [int(position) for position in positions]
        return (self._model_features(features) if positions else features), positions, reasons

    def _model_features(self, features: Any) -> pd.DataFrame:
        """Enforce the model's exact feature order on transformed rows."""
        if not isinstance(features, pd.DataFrame):
            raise ValueError("Exported preprocessing must produce a pandas DataFrame.")
        columns = self.state["feature_columns"]
        missing = [column for column in columns if column not in features.columns]
        if missing:
            raise ValueError(f"Missing model features after preprocessing: {missing}")
        features = features[columns]
        SklearnBridge.validate_features(features)
        return features

    def prediction_classes(self) -> list | None:
        """Return decoded class values so batch output keeps one type across chunks."""
        model = self.state["model"]
        if not is_classifier(model):
            return None
        classes = np.asarray(model.classes_)
        decoder = self.state["target_encoder"]
        if decoder is not None:
            # Same fallback as _predictions when the decoder cannot map estimator classes.
            with suppress(ValueError, TypeError, OverflowError):
                classes = decoder.inverse_transform(classes.astype(int))
        return np.asarray(classes).tolist()

    def probability_labels(self) -> list[str] | None:
        """Return decoded probability keys in estimator order, including for empty batches."""
        model = self.state["model"]
        if not is_classifier(model) or not callable(getattr(model, "predict_proba", None)):
            return None
        classes = np.asarray(model.classes_)
        if classes.ndim != 1:
            raise ValueError("Probability outputs require a single classification target.")
        decoder = self.state["target_encoder"]
        if decoder is not None:
            classes = decoder.inverse_transform(classes.astype(int))
        labels = [str(label) for label in classes]
        if len(set(labels)) != len(labels):
            raise ValueError("Decoded probability class labels must have unique string keys.")
        return labels

    def _probabilities(self, features: pd.DataFrame, labels: list[str]) -> np.ndarray:
        """Validate class probabilities before sharing them between labels and response output."""
        matrix = np.asarray(self.state["model"].predict_proba(features), dtype=float)
        if matrix.shape != (len(features), len(labels)):
            raise ValueError("Model probability shape disagrees with rows or fitted classes.")
        if not np.isfinite(matrix).all() or (matrix < 0).any() or (matrix > 1).any():
            raise ValueError("Model probabilities must be finite and between zero and one.")
        if not np.allclose(matrix.sum(axis=1), 1):
            raise ValueError("Model class probabilities must sum to one.")
        return matrix

    def _predictions(self, features: pd.DataFrame, probabilities: np.ndarray | None = None) -> list:
        """Apply saved thresholds and the original target decoder when present."""
        model = self.state["model"]
        thresholds = self.state["thresholds"]
        if thresholds is None:
            values = model.predict(features)
        else:
            weights = {label: thresholds[str(label)] for label in model.classes_}
            values = apply_thresholds(probabilities, weights, classes=model.classes_)
        decoder = self.state["target_encoder"]
        if decoder is not None:
            # Match serving's fallback for regression or otherwise undecodable outputs.
            with suppress(ValueError, TypeError, OverflowError):
                values = decoder.inverse_transform(np.asarray(values).astype(int))
        return np.asarray(values).tolist()
