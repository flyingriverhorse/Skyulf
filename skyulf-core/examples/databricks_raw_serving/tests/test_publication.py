"""Training must not publish batch predictions unless explicitly requested."""

import sys
from pathlib import Path
from unittest.mock import Mock

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from raw_serving_demo.training import write_optional_batch_predictions


def test_disabled_publication_does_not_score_or_touch_spark():
    """Endpoint-only training must not invoke a scorer or create a prediction table."""
    spark, artifact = Mock(), Mock()
    result = write_optional_batch_predictions(False, spark, artifact, "workspace.demo", 0)
    assert result == {"status": "skipped", "reason": "write_batch_predictions is false"}
    assert spark.mock_calls == []
    assert artifact.mock_calls == []


@pytest.mark.parametrize("enabled", ["false", "true", 0, 1, None])
def test_publication_requires_an_actual_boolean(enabled):
    """String truthiness must never silently enable table writes."""
    spark = Mock()
    with pytest.raises(ValueError, match="boolean"):
        write_optional_batch_predictions(enabled, spark, Mock(), "workspace.demo", 0)
    assert spark.mock_calls == []
