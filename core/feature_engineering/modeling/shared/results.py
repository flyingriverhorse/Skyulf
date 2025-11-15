"""Result summarization helpers reused across modeling workflows."""

from __future__ import annotations

from typing import Any, Dict, List

import numpy as np


def _serialize_value(value: Any) -> Any:
    """Convert numpy scalars and complex values into JSON-friendly types."""

    if isinstance(value, (str, bool)):
        return value
    if value is None:
        return None
    if isinstance(value, (int, float)):
        return value
    if isinstance(value, np.generic):
        return value.item()
    if isinstance(value, (list, dict)):
        return value
    return repr(value)


def _summarize_results(cv_results_: Dict[str, Any], limit: int = 10) -> List[Dict[str, Any]]:
    """Summarize CV results into a compact, serialized representation."""

    if not cv_results_:
        return []

    params_list = cv_results_.get("params", [])
    mean_test = cv_results_.get("mean_test_score", [])
    std_test = cv_results_.get("std_test_score", [])
    mean_train = cv_results_.get("mean_train_score", [])
    mean_fit_time = cv_results_.get("mean_fit_time", [])
    rank_test = cv_results_.get("rank_test_score", [])

    records: List[Dict[str, Any]] = []
    for idx, params in enumerate(params_list):
        record = {
            "rank": int(rank_test[idx]) if idx < len(rank_test) else idx + 1,
            "mean_test_score": float(mean_test[idx]) if idx < len(mean_test) else None,
            "std_test_score": float(std_test[idx]) if idx < len(std_test) else None,
            "mean_train_score": float(mean_train[idx]) if idx < len(mean_train) else None,
            "mean_fit_time": float(mean_fit_time[idx]) if idx < len(mean_fit_time) else None,
            "params": {key: _serialize_value(value) for key, value in (params or {}).items()},
        }
        records.append(record)

    records.sort(key=lambda item: (item.get("rank") or 0, -(item.get("mean_test_score") or 0)), reverse=False)
    return records[:limit]


__all__ = ["_serialize_value", "_summarize_results"]
