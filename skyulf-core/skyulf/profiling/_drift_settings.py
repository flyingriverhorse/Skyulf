"""Shared drift settings validation for frame calculations and monitoring enrollment."""

import math

DEFAULT_TOP_K = 50
DEFAULT_CATEGORY_SUMMARIES = 10_000
DEFAULT_UNIQUENESS = 0.5
MAX_MONITOR_ROWS = 1_000_000
MAX_MONITOR_BYTES = 1024**3

_THRESHOLDS = {"psi": 0.2, "ks_statistic": 0.1, "wasserstein": 0.1, "kl_divergence": 0.1}
_INTEGER_LIMITS = {
    "categorical_top_k": 1_000_000,
    "max_category_summaries": 1_000_000,
    "monitor_max_rows": MAX_MONITOR_ROWS,
    "monitor_max_bytes": MAX_MONITOR_BYTES,
}
_EXTRA_FLOATS = {"id_like_uniqueness", "psi_moderate", "psi_critical"}


def _validate_value(name: str, value: float, allow_zero_thresholds: bool) -> None:
    """Reject coercion and nonfinite values before any data-dependent work."""
    if name in _INTEGER_LIMITS:
        maximum = _INTEGER_LIMITS[name]
        if type(value) is not int or not 1 <= value <= maximum:
            guidance = (
                " For larger observations use Spark monitoring."
                if name.startswith("monitor_")
                else ""
            )
            raise ValueError(f"{name} must be a positive integer at most {maximum}.{guidance}")
    else:
        minimum = 0.0 if allow_zero_thresholds and name in _THRESHOLDS else math.nextafter(0.0, 1.0)
        if type(value) not in (int, float) or not math.isfinite(value) or value < minimum:
            raise ValueError(f"Drift thresholds must be finite and positive: {name}.")


def merge_drift_settings(
    value: dict[str, float] | None, *, allow_zero_thresholds: bool = False
) -> dict[str, float]:
    """Validate overrides while preserving absent optional keys for legacy behavior."""
    if value is None:
        return dict(_THRESHOLDS)
    if type(value) is not dict or set(value) - (
        _THRESHOLDS.keys() | _INTEGER_LIMITS.keys() | _EXTRA_FLOATS
    ):
        raise ValueError("Unknown monitoring drift thresholds.")
    for name, setting in value.items():
        _validate_value(name, setting, allow_zero_thresholds)
    if value.get("id_like_uniqueness", DEFAULT_UNIQUENESS) > 1:
        raise ValueError("id_like_uniqueness must be in (0, 1].")
    if value.get("categorical_top_k", DEFAULT_TOP_K) > value.get(
        "max_category_summaries", DEFAULT_CATEGORY_SUMMARIES
    ):
        raise ValueError("categorical_top_k must not exceed max_category_summaries.")
    if value.get("psi_moderate", 0.1) >= value.get("psi_critical", 0.25):
        raise ValueError("psi_moderate must be less than psi_critical.")
    return _THRESHOLDS | value
