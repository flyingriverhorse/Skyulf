"""Revisit a bounded horizon of mature label windows before evaluating today's policy."""

from datetime import datetime
from typing import Any

from .monitoring_config import MonitorConfig
from .monitoring_performance import observe_performance_safely
from .monitoring_store import persist_report, result_row
from .performance_policy import completed_performance_window
from .spark_monitoring_reference import load_spark_monitoring_reference


def revisit_performance_windows(
    spark: Any, namespace: str, configs: list[MonitorConfig], now: datetime, *, windows: int = 3
) -> None:
    """Repair late-label history chronologically while keeping historical actions disabled.

    The horizon includes the latest window, measured by the ordinary observation
    immediately after this function. Retention failures persist unavailable
    evidence and therefore break, rather than silently satisfy, failure streaks.
    """
    if type(windows) is not int or not 1 <= windows <= 100:
        raise ValueError("monitoring_revisit_windows must be an integer between 1 and 100.")
    for config in configs:
        policy = config.performance_policy
        if not policy or policy["mode"] == "off" or windows == 1:
            continue
        _revisit_model(spark, namespace, config, now, windows)


def _revisit_model(
    spark: Any, namespace: str, config: MonitorConfig, now: datetime, windows: int
) -> None:
    """Reuse one prepared model across the selected older windows."""
    try:
        artifact, spec, _, evidence = load_spark_monitoring_reference(
            spark, config, tracking_uri="databricks", registry_uri="databricks-uc"
        )
    except Exception:  # noqa: BLE001 - latest observation persists the reference failure
        return
    policy = config.performance_policy
    assert policy is not None
    start, end = completed_performance_window(now, policy)
    span = end - start
    for offset in range(windows - 1, 0, -1):
        ending = end - offset * span
        performance = observe_performance_safely(
            spark, namespace, config, artifact, spec, evidence, now, window_end=ending
        )
        report = {
            "status": "no_data",
            "metrics": [],
            "performance": performance,
            "notes": [
                "Historical performance window revisited for late labels; no training action."
            ],
        }
        row = result_row(
            config,
            evidence["model_version"],
            now,
            ending - span,
            ending,
            report,
            evidence | {"performance": performance, "execution_engine": "spark"},
        )
        persist_report(spark, namespace, row)
