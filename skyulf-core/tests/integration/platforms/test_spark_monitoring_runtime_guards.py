"""Runtime monitoring invariants remain enforced with optimized Python bytecode."""

from datetime import UTC, datetime
from unittest.mock import MagicMock, Mock

import pytest

from skyulf.integrations.databricks.observability.monitoring.monitoring_config import MonitorConfig
from skyulf.integrations.databricks.observability.monitoring.spark import (
    spark_monitoring_drift as drift,
)
from skyulf.integrations.databricks.observability.monitoring.spark import (
    spark_monitoring_windows as windows,
)
from skyulf.profiling._drift_evidence import DriftEvidence


def test_revisit_requires_performance_policy(monkeypatch):
    """An invalid internal revisit cannot silently proceed without its validated policy."""
    monkeypatch.setattr(
        windows, "load_spark_monitoring_reference", Mock(return_value=(None, None, None, {}))
    )
    persist = Mock()
    monkeypatch.setattr(windows, "persist_report", persist)
    config = MonitorConfig(
        environment="test",
        project="runtime",
        model_name="a.b.model",
        model_version="1",
        source_table="a.b.source",
        prediction_table="a.b.predictions",
    )
    with pytest.raises(ValueError, match="Performance window revisit requires a policy"):
        windows._revisit_model(None, "a.monitor", config, datetime.now(UTC), 3)
    persist.assert_not_called()


def test_numeric_ks_result_requires_probability(monkeypatch):
    """Malformed KS evidence must fail before it becomes a report metric, including with -O."""
    frame = MagicMock()
    frame.count.return_value = 4
    frame.agg.return_value.first.return_value = {"origin": 0.0, "std": 1.0}
    frame.schema.__getitem__.return_value.dataType.typeName.return_value = "double"
    monkeypatch.setattr(drift, "_functions", MagicMock)
    monkeypatch.setattr(drift, "_column", MagicMock())
    monkeypatch.setattr(drift, "_exists", lambda value: False)
    monkeypatch.setattr(drift, "_numeric_frame", lambda *args: frame)
    monkeypatch.setattr(drift, "_cdf_statistics", lambda *args: (0.5, 1.0))
    monkeypatch.setattr(drift, "_histogram_metrics", lambda *args: (0.5, 0.5))
    monkeypatch.setattr(
        drift,
        "_ks_evidence",
        lambda *args: DriftEvidence(
            test="ks_2samp", p_value=None, reference_count=4, current_count=4
        ),
    )
    thresholds = {"wasserstein": 0.1, "ks_statistic": 0.1, "psi": 0.2, "kl_divergence": 0.1}
    with pytest.raises(ValueError, match="KS evidence requires a p-value"):
        drift._numeric_result(frame, frame, "x", thresholds)
