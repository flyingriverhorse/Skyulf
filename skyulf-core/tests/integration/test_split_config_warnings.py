"""Misspelled split settings must be visible without changing valid split behavior."""

import logging
from typing import Any

import pandas as pd
import polars as pl
import pytest

from skyulf.data.dataset import SplitDataset
from skyulf.preprocessing.pipeline import FeatureEngineer
from skyulf.preprocessing.split import SplitApplier, SplitCalculator


@pytest.fixture(params=["pandas", "polars"])
def frame(request: pytest.FixtureRequest) -> Any:
    """Use balanced enough data to separate configuration warnings from rare-class warnings."""
    data = {"feature": list(range(200)), "target": [0] * 160 + [1] * 40}
    return pd.DataFrame(data) if request.param == "pandas" else pl.DataFrame(data)


@pytest.mark.parametrize(
    "unknown",
    [{"stratify_col": "target"}, {"validation_fraction": 0.2, "test_szie": 0.3}],
)
def test_unknown_split_config_keys_warn_without_leaking_values(
    frame: Any, unknown: dict[str, Any], caplog: pytest.LogCaptureFixture
) -> None:
    """A typo must produce one actionable warning while preserving the artifact contract."""
    config = {"test_size": 0.2, "random_state": 42, **unknown}
    with caplog.at_level(logging.WARNING, logger="skyulf.preprocessing.split"):
        params = SplitCalculator().fit(frame, config)
    assert params == {"type": "split", "test_size": 0.2, "random_state": 42}
    assert len(caplog.records) == 1
    assert str(sorted(unknown)) in caplog.text
    assert "ignored unrecognized config keys" in caplog.text
    assert "Supported keys" in caplog.text
    assert "stratify" in caplog.text and "target_column" in caplog.text
    assert config == {"test_size": 0.2, "random_state": 42, **unknown}


def test_split_config_warning_contains_names_only(
    frame: Any, caplog: pytest.LogCaptureFixture
) -> None:
    """Diagnostics need setting names, never arbitrary objects or values from the config."""
    value = "private-dataset-location"
    SplitCalculator().fit(frame, {"stratify_col": value})
    assert "stratify_col" in caplog.text and value not in caplog.text


def test_supported_split_settings_and_reserved_metadata_are_quiet(
    frame: Any, caplog: pytest.LogCaptureFixture
) -> None:
    """Valid Canvas settings and private routing metadata must not create false warnings."""
    config = {
        "test_size": 0.2,
        "validation_size": 0.2,
        "random_state": 42,
        "shuffle": True,
        "stratify": True,
        "target_column": "target",
    }
    params = SplitCalculator().fit(
        frame,
        {**config, "type": "split", "_display_name": "Split", "_merge_strategy": "first_wins"},
    )
    result = SplitApplier().apply(frame, dict(params))
    assert params == {"type": "split", **config}
    sizes = []
    for part in (result.train, result.test, result.validation):
        assert isinstance(part, tuple)
        features, target = part
        assert isinstance(target, (pd.Series, pl.Series))
        sizes.append(len(features))
        assert target.mean() == pytest.approx(0.2)
    assert sizes == [120, 40, 40]
    assert not caplog.records


def test_pipeline_exposes_split_typo_warning_without_inventing_stratification(
    frame: Any, caplog: pytest.LogCaptureFixture
) -> None:
    """The real pipeline must surface the typo and retain existing split semantics."""
    config = {
        "test_size": 0.2,
        "validation_size": 0.2,
        "random_state": 42,
        "target_column": "target",
        "stratify_col": "target",
    }
    pipeline = FeatureEngineer(
        [{"name": "split", "transformer": "TrainTestSplitter", "params": config}]
    )
    result, _ = pipeline.fit_transform(frame)
    assert isinstance(result, SplitDataset)
    assert "stratify_col" in caplog.text
    assert "ignored unrecognized config keys" in caplog.text
    assert [float(part[1].mean()) for part in (result.train, result.test, result.validation)] == [
        0.2,
        0.175,
        0.225,
    ]
