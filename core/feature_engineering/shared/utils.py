"""Shared utility helpers for feature engineering nodes."""

from __future__ import annotations

import math
import re
import string
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from typing import Any, Dict, Iterable, List, Optional, Set

import numpy as np
import pandas as pd
from pandas.api import types as pd_types

PENDING_CONFIRMATION_FLAG = "__pending_confirmation__"

ALIAS_PUNCTUATION_TABLE = str.maketrans("", "", string.punctuation)
COMMON_BOOLEAN_ALIASES: Dict[str, str] = {
    "y": "Yes",
    "yes": "Yes",
    "true": "Yes",
    "1": "Yes",
    "on": "Yes",
    "t": "Yes",
    "affirmative": "Yes",
    "n": "No",
    "no": "No",
    "false": "No",
    "0": "No",
    "off": "No",
    "f": "No",
    "negative": "No",
}
COUNTRY_ALIAS_MAP: Dict[str, str] = {
    "usa": "USA",
    "us": "USA",
    "unitedstates": "USA",
    "unitedstatesofamerica": "USA",
    "states": "USA",
    "america": "USA",
    "unitedkingdom": "United Kingdom",
    "uk": "United Kingdom",
    "greatbritain": "United Kingdom",
    "england": "United Kingdom",
    "uae": "United Arab Emirates",
    "unitedarabemirates": "United Arab Emirates",
    "prc": "China",
    "peoplesrepublicofchina": "China",
    "southkorea": "South Korea",
    "republicofkorea": "South Korea",
    "sk": "South Korea",
}
TWO_DIGIT_YEAR_PIVOT = 50

BOOLEAN_TRUE_VALUES: set[str] = {"true", "1", "yes", "y", "on", "t"}
BOOLEAN_FALSE_VALUES: set[str] = {"false", "0", "no", "n", "off", "f"}


DEDUP_KEEP_MODES: Set[str] = {"first", "last", "none"}


@dataclass
class NormalizedRemoveDuplicatesConfig:
    columns: List[str]
    keep: str


def _normalize_alias_key(value: str) -> str:
    return value.translate(ALIAS_PUNCTUATION_TABLE).replace(" ", "").lower()


def _parse_custom_alias_pairs(raw_pairs: Any) -> Dict[str, str]:
    if not raw_pairs:
        return {}

    result: Dict[str, str] = {}

    if isinstance(raw_pairs, dict):
        for key, value in raw_pairs.items():
            alias = str(key or "").strip()
            replacement = str(value or "").strip()
            if alias and replacement:
                result[_normalize_alias_key(alias)] = replacement
        return result

    if isinstance(raw_pairs, str):
        entries = re.split(r"[\n;,]", raw_pairs)
    elif isinstance(raw_pairs, Iterable):
        entries = list(raw_pairs)
    else:
        entries = []

    for entry in entries:
        if not entry:
            continue
        text = str(entry).strip()
        if not text:
            continue
        if "=>" in text:
            alias, replacement = text.split("=>", 1)
        elif ":" in text:
            alias, replacement = text.split(":", 1)
        else:
            continue
        alias = alias.strip()
        replacement = replacement.strip()
        if alias and replacement:
            result[_normalize_alias_key(alias)] = replacement

    return result


def _extract_clean_string(value: Any) -> Optional[str]:
    if value is None:
        return None
    if isinstance(value, str):
        text = value.strip()
        return text or None
    if isinstance(value, Mapping):
        for key in ("column", "name", "label", "value", "id"):
            if key in value:
                candidate = _extract_clean_string(value[key])
                if candidate:
                    return candidate
    if isinstance(value, Sequence) and not isinstance(value, (str, bytes, bytearray)):
        for item in reversed(value):
            candidate = _extract_clean_string(item)
            if candidate:
                return candidate
        return None
    try:
        text = str(value).strip()
    except Exception:  # pragma: no cover - defensive fallback
        return None
    return text or None


def _coerce_string_list(value: Any) -> List[str]:
    if isinstance(value, str):
        return [segment.strip() for segment in value.split(",") if segment.strip()]
    if isinstance(value, Sequence) and not isinstance(value, (str, bytes, bytearray)):
        result: List[str] = []
        for item in value:
            normalized = _extract_clean_string(item)
            if normalized:
                result.append(normalized)
        return result
    normalized = _extract_clean_string(value)
    return [normalized] if normalized else []


def _auto_detect_text_columns(frame: pd.DataFrame) -> List[str]:
    detected: List[str] = []
    for column in frame.columns:
        series = frame[column]
        if (
            pd_types.is_string_dtype(series)
            or pd_types.is_object_dtype(series)
            or isinstance(series.dtype, pd.CategoricalDtype)
        ):
            detected.append(str(column))
    return detected


def _auto_detect_numeric_columns(frame: pd.DataFrame) -> List[str]:
    detected: List[str] = []
    for column in frame.columns:
        series = frame[column]
        if pd_types.is_numeric_dtype(series):
            detected.append(str(column))
    return detected


def _auto_detect_datetime_columns(frame: pd.DataFrame) -> List[str]:
    detected: List[str] = []
    for column in frame.columns:
        series = frame[column]
        if pd_types.is_datetime64_any_dtype(series):
            detected.append(str(column))
    return detected


def _is_node_pending(node: Dict[str, Any]) -> bool:
    data = node.get("data") or {}
    config = data.get("config")
    if isinstance(config, dict):
        return bool(config.get(PENDING_CONFIRMATION_FLAG))
    return False


def _coerce_config_boolean(value: Any, default: bool = False) -> bool:
    if isinstance(value, bool):
        return value
    if value is None:
        return default
    if isinstance(value, (int, np.integer)):
        return bool(value)
    if isinstance(value, (float, np.floating)):
        if math.isnan(value):
            return default
        return bool(int(value))
    if isinstance(value, str):
        normalized = value.strip().lower()
        if not normalized:
            return default
        if normalized in BOOLEAN_TRUE_VALUES:
            return True
        if normalized in BOOLEAN_FALSE_VALUES:
            return False
    return default


def _coerce_boolean_value(value: Any) -> Optional[bool]:
    if value is None:
        return None
    if isinstance(value, (bool, np.bool_)):
        return bool(value)
    if isinstance(value, (int, np.integer)):
        if value == 1:
            return True
        if value == 0:
            return False
        return None
    if isinstance(value, (float, np.floating)):
        if math.isnan(value):
            return None
        if value == 1.0:
            return True
        if value == 0.0:
            return False
        return None
    if isinstance(value, str):
        normalized = value.strip().lower()
        if not normalized:
            return None
        if normalized in BOOLEAN_TRUE_VALUES:
            return True
        if normalized in BOOLEAN_FALSE_VALUES:
            return False
        return None
    return None


def _format_interval_endpoint(value: Any, precision: int) -> str:
    try:
        numeric = float(value)
    except (TypeError, ValueError):
        return str(value)
    if math.isinf(numeric):
        return "-inf" if numeric < 0 else "inf"
    precision_digits = max(0, precision)
    formatted = f"{numeric:.{precision_digits}f}"
    if "." in formatted:
        formatted = formatted.rstrip("0").rstrip(".")
    return formatted


def _format_interval_value(interval: pd.Interval, precision: int) -> str:
    if interval.closed == "both":
        left_bracket, right_bracket = "[", "]"
    elif interval.closed == "left":
        left_bracket, right_bracket = "[", ")"
    elif interval.closed == "right":
        left_bracket, right_bracket = "(", "]"
    else:
        left_bracket, right_bracket = "(", ")"

    left_value = _format_interval_endpoint(interval.left, precision)
    right_value = _format_interval_endpoint(interval.right, precision)
    return f"{left_bracket}{left_value}, {right_value}{right_bracket}"


def _normalize_remove_duplicates_config(config: Any) -> NormalizedRemoveDuplicatesConfig:
    columns: List[str] = []
    keep_value = "first"

    if isinstance(config, dict):
        raw_columns = config.get("columns")
        if not raw_columns:
            raw_columns = config.get("subset")

        if isinstance(raw_columns, list):
            columns = [str(column).strip() for column in raw_columns if str(column).strip()]
        elif isinstance(raw_columns, str):
            columns = [segment.strip() for segment in raw_columns.split(",") if segment.strip()]

        raw_keep = config.get("keep")
        if raw_keep is None:
            raw_keep = config.get("keep_duplicates") or config.get("keep_strategy")

        if isinstance(raw_keep, str):
            normalized_keep = raw_keep.strip().lower()
            if normalized_keep in {"drop", "remove", "none"}:
                keep_value = "none"
            elif normalized_keep in {"first", "last"}:
                keep_value = normalized_keep
        elif raw_keep is False:
            keep_value = "none"
        elif raw_keep is True:
            keep_value = "first"

    normalized_columns = [column for column in dict.fromkeys(columns)]
    if keep_value not in DEDUP_KEEP_MODES:
        keep_value = "first"

    return NormalizedRemoveDuplicatesConfig(columns=normalized_columns, keep=keep_value)


def _is_binary_indicator(series: pd.Series) -> bool:
    if series.empty:
        return False

    if not pd.api.types.is_numeric_dtype(series):
        return False

    unique_values = pd.unique(series.dropna())
    if unique_values.size == 0:
        return False

    normalized: Set[float] = set()
    for value in unique_values:
        try:
            numeric = float(value)
        except (TypeError, ValueError):
            return False
        if not math.isfinite(numeric):
            return False
        if math.isclose(numeric, 0.0, rel_tol=0.0, abs_tol=1e-9):
            normalized.add(0.0)
        elif math.isclose(numeric, 1.0, rel_tol=0.0, abs_tol=1e-9):
            normalized.add(1.0)
        else:
            return False

    return 0 < len(normalized) <= 2
