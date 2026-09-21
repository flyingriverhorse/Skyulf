"""Keep explicit execution selection compatible with existing local input shapes."""

import pandas as pd
import polars as pl
import pytest

from skyulf.core.execution import ExecutionOptions, FrameSpec
from skyulf.data.dataset import SplitDataset
from skyulf.preprocessing.pipeline import FeatureEngineer


@pytest.mark.parametrize("engine", ["pandas", "polars"])
@pytest.mark.parametrize("shape", ["frame", "tuple", "split"])
def test_explicit_local_engine_accepts_existing_input_shapes(engine, shape):
    """Opting into an engine must preserve valid local frame and split APIs."""
    frame = pd.DataFrame({"x": [1, 2]}) if engine == "pandas" else pl.DataFrame({"x": [1, 2]})
    data = frame
    if shape == "tuple":
        data = (frame, [0, 1])
    elif shape == "split":
        data = SplitDataset(train=frame, test=frame)
    engineer = FeatureEngineer([], execution_options=ExecutionOptions(engine))
    output, _ = engineer.fit_transform(data)
    assert output is data


@pytest.mark.parametrize("engine", ["pandas", "polars"])
def test_explicit_engine_checks_each_split(engine):
    """A local split must not conceal a frame conflicting with the requested engine."""
    data = SplitDataset(train=pd.DataFrame({"x": [1]}), test=pl.DataFrame({"x": [2]}))
    with pytest.raises(ValueError, match="conflicts"):
        FeatureEngineer([], execution_options=ExecutionOptions(engine)).fit_transform(data)


def test_frame_spec_is_not_silently_ignored_for_local_input():
    """Local row-key protection must not be advertised before its implementation."""
    with pytest.raises(ValueError, match="only for Spark"):
        FeatureEngineer([], frame_spec=FrameSpec(("id",))).fit_transform(pd.DataFrame({"id": [1]}))
