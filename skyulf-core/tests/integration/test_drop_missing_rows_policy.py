"""Reject invalid missing-row policies consistently before engine-specific filtering."""

from typing import Any

import pandas as pd
import polars as pl
import pytest

from skyulf.preprocessing.drop_and_missing.drop_rows import (
    DropMissingRowsApplier,
    DropMissingRowsCalculator,
)


@pytest.mark.parametrize("engine", ["pandas", "polars"])
@pytest.mark.parametrize("how", ["bad", None, "ANY", "", 7, [], {}])
@pytest.mark.parametrize("stage", ["fit", "replay"])
def test_invalid_how_fails_at_public_boundaries(
    engine: str, how: Any, stage: str, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Invalid options must not silently select opposite row policies on different engines."""
    monkeypatch.setenv("SKYULF_ENGINE", engine)
    data = {"a": [1.0, None, 3.0], "b": [1.0, 2.0, None]}
    frame = pd.DataFrame(data) if engine == "pandas" else pl.DataFrame(data)
    config = {"how": how}
    with pytest.raises(ValueError, match="how.*any.*all"):
        if stage == "fit":
            DropMissingRowsCalculator().fit(frame, config)
        else:
            DropMissingRowsApplier().apply(frame, {"type": "drop_missing_rows", **config})


@pytest.mark.parametrize("engine", ["pandas", "polars"])
@pytest.mark.parametrize("threshold", [{"threshold": 1}, {"missing_threshold": 50}])
def test_invalid_how_is_rejected_even_when_a_threshold_takes_precedence(
    engine: str, threshold: dict, monkeypatch: pytest.MonkeyPatch
) -> None:
    """A temporarily inactive invalid policy must not survive in a saved artifact."""
    monkeypatch.setenv("SKYULF_ENGINE", engine)
    data = {"a": [1.0, None]}
    frame = pd.DataFrame(data) if engine == "pandas" else pl.DataFrame(data)
    with pytest.raises(ValueError, match="how.*any.*all"):
        DropMissingRowsCalculator().fit(frame, {"how": "bad", **threshold})


@pytest.mark.parametrize("engine", ["pandas", "polars"])
@pytest.mark.parametrize(
    "config,expected",
    [
        ({}, [0]),
        ({"how": "any"}, [0]),
        ({"how": "all"}, [0, 1, 2]),
        ({"how": "any", "threshold": 1}, [0, 1, 2]),
    ],
)
def test_supported_policies_keep_the_same_rows_and_targets(
    engine: str, config: dict, expected: list[int], monkeypatch: pytest.MonkeyPatch
) -> None:
    """Validation must preserve the default and explicit policies with aligned targets."""
    monkeypatch.setenv("SKYULF_ENGINE", engine)
    data = {"a": [1.0, None, 3.0, None], "b": [1.0, 2.0, None, None]}
    frame = pd.DataFrame(data) if engine == "pandas" else pl.DataFrame(data)
    target = pd.Series([0, 1, 2, 3]) if engine == "pandas" else pl.Series([0, 1, 2, 3])
    params = DropMissingRowsCalculator().fit((frame, target), config)

    result, result_y = DropMissingRowsApplier().apply((frame, target), params)

    assert len(result) == len(expected)
    assert result_y.to_list() == expected
