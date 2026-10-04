"""Regression tests: transient scoring diagnostics must not change a fitted-state digest."""

from typing import Any

import numpy as np
import pandas as pd

from skyulf.pipeline.seal import artifact_digest
from skyulf.preprocessing.pipeline import FeatureEngineer


class _FittedStep:
    """Hold learned values plus the diagnostic attribute a transform may record."""

    last_transform_coverage_: dict[str, Any]

    def __init__(self) -> None:
        """Start with only learned state."""
        self.mean_ = np.array([1.0, 2.0])


def test_recorded_transform_coverage_does_not_change_digest() -> None:
    """Recording evaluation coverage after a transform must keep the artifact identity."""
    step = _FittedStep()
    before = artifact_digest([step])

    step.last_transform_coverage_ = {"input_rows": 5, "output_rows": 4, "steps": []}

    assert artifact_digest([step]) == before


def test_learned_state_still_changes_digest() -> None:
    """Excluding diagnostics must not hide a change to genuinely learned values."""
    step = _FittedStep()
    before = artifact_digest([step])

    step.mean_ = np.array([1.0, 3.0])

    assert artifact_digest([step]) != before


def test_feature_engineer_digest_survives_evaluation_transform() -> None:
    """Scoring held-out rows with a fitted engineer must not mutate its sealed identity."""
    engineer = FeatureEngineer(
        [
            {
                "name": "bounds",
                "transformer": "ManualBounds",
                "params": {"bounds": {"a": {"lower": 0, "upper": 5}}},
            }
        ]
    )
    frame = pd.DataFrame({"a": [1.0, 2.0, 3.0]})
    engineer.fit_transform(frame)
    before = artifact_digest(engineer)

    engineer.transform(pd.DataFrame({"a": [-1.0, 1.0, 9.0]}))

    assert artifact_digest(engineer) == before
