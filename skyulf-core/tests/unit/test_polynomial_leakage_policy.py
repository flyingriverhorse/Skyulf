"""Polynomial auto-selection must obey the learned preprocessing boundary."""

from typing import Any

import pandas as pd
import polars as pl
import pytest

from skyulf.data.dataset import SplitDataset
from skyulf.leakage import step_learns_from_data, validate_leakage_safety
from skyulf.preprocessing.base import StatefulTransformer
from skyulf.registry import NodeRegistry

_ALIASES = ["PolynomialFeatures", "PolynomialFeaturesNode"]
_AUTO_SELECTIONS = [{"auto_detect": True}, {"auto_detect": True, "columns": []}]
_FIXED_SELECTIONS = [
    {},
    {"columns": []},
    {"columns": ["x"]},
    {"auto_detect": True, "columns": ["x"]},
    {"auto_detect": False, "columns": []},
]


def _pipeline_config(
    alias: str, params: dict[str, Any], splitter: str = "TrainTestSplitter", *, before: bool = True
) -> dict[str, Any]:
    """Place one polynomial operation on either side of an actual row split."""
    candidate = {"name": "polynomial", "transformer": alias, "params": params.copy()}
    split = {
        "name": "split",
        "transformer": splitter,
        "params": {"target_column": "target", "test_size": 0.25, "random_state": 42},
    }
    return {"preprocessing": [candidate, split] if before else [split, candidate], "modeling": {}}


@pytest.mark.parametrize("alias", _ALIASES)
@pytest.mark.parametrize("params", _AUTO_SELECTIONS, ids=["omitted", "empty"])
@pytest.mark.parametrize("splitter", ["TrainTestSplitter", "Split"])
@pytest.mark.parametrize("policy", ["raise", "warn", "ignore"])
@pytest.mark.parametrize("before", [True, False], ids=["before-split", "after-split"])
def test_polynomial_auto_selection_obeys_split_policy(
    alias: str, params: dict[str, Any], splitter: str, policy: Any, before: bool
) -> None:
    """Automatic numeric discovery is learned before, but safe after, a row split."""
    config = _pipeline_config(alias, params, splitter, before=before)

    if before and policy == "raise":
        with pytest.raises(ValueError, match="Data leakage risk"):
            validate_leakage_safety(config, on_leakage=policy)
    else:
        issues = validate_leakage_safety(config, on_leakage=policy)
        assert bool(issues) is (before and policy == "warn")


@pytest.mark.parametrize("alias", _ALIASES)
@pytest.mark.parametrize("params", _FIXED_SELECTIONS)
@pytest.mark.parametrize("policy", ["raise", "warn", "ignore"])
def test_polynomial_fixed_and_noop_modes_remain_admitted(
    alias: str, params: dict[str, Any], policy: Any
) -> None:
    """A coarse learned flag must not reject fixed polynomial math or default no-ops."""
    issues = validate_leakage_safety(_pipeline_config(alias, params), on_leakage=policy)

    assert not step_learns_from_data(alias, params)
    assert issues == []


@pytest.mark.parametrize("alias", _ALIASES)
@pytest.mark.parametrize("params", _AUTO_SELECTIONS, ids=["omitted", "empty"])
@pytest.mark.parametrize("policy", ["raise", "warn", "ignore"])
def test_polynomial_auto_selection_accepts_already_split_input(
    alias: str, params: dict[str, Any], policy: Any
) -> None:
    """A caller-provided training boundary must remain sufficient for automatic selection."""
    issues = validate_leakage_safety(
        _pipeline_config(alias, params), on_leakage=policy, already_split=True
    )

    assert issues == []


@pytest.mark.parametrize("alias", _ALIASES)
@pytest.mark.parametrize("params", _AUTO_SELECTIONS, ids=["omitted", "empty"])
@pytest.mark.parametrize("engine", ["pandas", "polars"])
def test_polynomial_auto_selection_artifact_uses_only_training_rows(
    alias: str, params: dict[str, Any], engine: str
) -> None:
    """Heldout nonbinary values cannot activate a feature rejected by training-only discovery."""
    train: Any = pd.DataFrame({"x": [0.0, 1.0, 0.0, 1.0]})
    heldout: Any = pd.DataFrame({"x": [5.0, 9.0]})
    if engine == "polars":
        train = pl.from_pandas(train)
        heldout = pl.from_pandas(heldout)
    transformer = StatefulTransformer(
        NodeRegistry.get_calculator(alias)(), NodeRegistry.get_applier(alias)(), "polynomial"
    )

    output = transformer.fit_transform(SplitDataset(train=train, test=heldout), params)

    assert list(output.train.columns) == ["x"]
    assert list(output.test.columns) == ["x"]


@pytest.mark.parametrize("alias", _ALIASES)
def test_polynomial_null_selection_is_not_a_classifier_escape(alias: str) -> None:
    """Conservative admission must not equate null with explicit columns; runtime may reject null."""
    assert step_learns_from_data(alias, {"auto_detect": True, "columns": None})
