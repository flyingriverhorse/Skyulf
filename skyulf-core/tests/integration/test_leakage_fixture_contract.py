"""JSON-driven leakage contracts for standalone SDK consumers and every library node."""

import json
import logging
from copy import deepcopy
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import pytest

from skyulf import SkyulfPipeline, validate_leakage_safety
from skyulf.data.dataset import SplitDataset
from skyulf.registry import NodeRegistry

_FIXTURE_PATH = Path(__file__).resolve().parents[1] / "test_cases/leakage/registry_nodes.json"
_FIXTURE = json.loads(_FIXTURE_PATH.read_text(encoding="utf-8"))
_TRANSFORMER_CASES = (
    [(node_type, params, True) for node_type, params in _FIXTURE["learned_transformers"].items()]
    + [
        (node_type, params, False)
        for node_type, params in _FIXTURE["stateless_transformers"].items()
    ]
    + [(node_type, {}, False) for node_type in _FIXTURE["splitters"]]
)


def _config(node_type: str, params: dict[str, Any], placement: str) -> dict[str, Any]:
    """Materialize a public SDK configuration from the shared JSON template."""
    config = deepcopy(_FIXTURE["core_config"])
    config["preprocessing"][0].update(transformer=node_type, params=deepcopy(params))
    if placement == "after_split":
        config["preprocessing"].reverse()
    elif placement == "no_split":
        config["preprocessing"] = config["preprocessing"][:1]
    return config


def test_json_fixture_covers_all_registered_node_contracts() -> None:
    """Registry additions and safety reclassifications need intentional fixture updates."""
    metadata = NodeRegistry.get_all_metadata()
    expected = (
        set(_FIXTURE["learned_transformers"])
        | set(_FIXTURE["stateless_transformers"])
        | set(_FIXTURE["splitters"])
        | set(_FIXTURE["models"])
    )
    registered = {
        node_type
        for node_type in metadata
        if NodeRegistry.get_calculator(node_type).__module__.startswith("skyulf.")
    }
    assert expected == registered
    for node_type, _params, learned in _TRANSFORMER_CASES:
        assert metadata[node_type]["learns_from_data"] is learned
        assert metadata[node_type]["is_splitter"] is (node_type in _FIXTURE["splitters"])
    for model_type in _FIXTURE["models"]:
        assert metadata[model_type]["learns_from_data"] is True


@pytest.mark.parametrize(
    "node_type,params,learned", _TRANSFORMER_CASES, ids=lambda value: str(value)
)
@pytest.mark.parametrize("placement", ["before_split", "after_split", "no_split"])
@pytest.mark.parametrize("mode", ["raise", "warn", "ignore"])
def test_json_transformer_sdk_matrix(node_type, params, learned, placement, mode) -> None:
    """Every node must receive the same safety decision through the public SDK facade."""
    config = _config(node_type, params, placement)
    pipeline = SkyulfPipeline(config)
    violation = learned and placement == "before_split"

    if violation and mode == "raise":
        with pytest.raises(ValueError, match=node_type):
            pipeline.validate_leakage_safety()
    else:
        messages = pipeline.validate_leakage_safety(on_leakage=mode)
        advisory = placement == "no_split" and node_type not in _FIXTURE["splitters"]

        assert bool(messages) is (mode != "ignore" and (violation or advisory))
        if violation and mode == "warn":
            assert node_type in messages[0]


@pytest.mark.parametrize("case", _FIXTURE["parameter_exemptions"], ids=lambda case: case["id"])
def test_json_parameter_exemptions_match_standalone_sdk(case) -> None:
    """The SDK must accept the same stateless parameter modes as backend submission."""
    pipeline = SkyulfPipeline(_config(case["transformer"], case["params"], "before_split"))

    assert pipeline.validate_leakage_safety() == []


@pytest.mark.parametrize("model_type", _FIXTURE["models"])
def test_every_model_is_blocked_before_learning_from_a_leaking_pipeline(model_type) -> None:
    """A model choice must not bypass the SDK's automatic pre-fit leakage gate."""
    config = _config("StandardScaler", {"columns": ["x"]}, "before_split")
    config["modeling"] = {"type": model_type}
    pipeline = SkyulfPipeline(config)

    with pytest.raises(ValueError, match="Data leakage risk"):
        pipeline.fit(pd.DataFrame(_FIXTURE["data"]), target_column="target")

    assert not pipeline.is_fitted()


def test_default_fit_blocks_before_any_preprocessing_is_fitted() -> None:
    """Core-only users must not have to remember to call the diagnostic manually."""
    pipeline = SkyulfPipeline(_config("StandardScaler", {"columns": ["x"]}, "before_split"))

    with pytest.raises(ValueError, match="Data leakage risk"):
        pipeline.fit(pd.DataFrame(_FIXTURE["data"]), target_column="target")

    assert not pipeline.is_fitted()


def test_split_extraction_cannot_bypass_the_pre_fit_gate() -> None:
    """The helper that fits preprocessing must enforce the same boundary as fit()."""
    pipeline = SkyulfPipeline(_config("StandardScaler", {"columns": ["x"]}, "before_split"))

    with pytest.raises(ValueError, match="Data leakage risk"):
        pipeline.get_fitted_split(pd.DataFrame(_FIXTURE["data"]), target_column="target")

    assert not pipeline.is_fitted()


@pytest.mark.parametrize("mode", ["warn", "ignore"])
def test_fit_supports_explicit_nonblocking_modes(mode, caplog) -> None:
    """Intentional legacy runs need an explicit escape hatch with predictable diagnostics."""
    pipeline = SkyulfPipeline(_config("StandardScaler", {"columns": ["x"]}, "before_split"))

    with caplog.at_level(logging.WARNING):
        metrics = pipeline.fit(pd.DataFrame(_FIXTURE["data"]), "target", on_leakage=mode)

    assert "preprocessing" in metrics
    assert pipeline.is_fitted()
    assert any("before the train/test split" in record.message for record in caplog.records) is (
        mode == "warn"
    )


def test_external_split_keeps_held_out_rows_out_of_scaler_statistics() -> None:
    """Externally split SDK inputs must stay safe without requiring a splitter in JSON."""
    frame = pd.DataFrame(_FIXTURE["data"])
    dataset = SplitDataset(train=frame.iloc[:8].copy(), test=frame.iloc[8:].copy())
    pipeline = SkyulfPipeline(_config("StandardScaler", {"columns": ["x"]}, "no_split"))

    pipeline.fit(dataset, "target")
    x_train, _, x_test, _ = pipeline.get_fitted_split(dataset, "target")

    train_mean = np.mean(np.arange(1.0, 9.0))
    train_std = np.std(np.arange(1.0, 9.0))
    np.testing.assert_allclose(x_train["x"], (np.arange(1.0, 9.0) - train_mean) / train_std)
    np.testing.assert_allclose(
        x_test["x"], (np.array([100, 200, 300, 400]) - train_mean) / train_std
    )
    assert x_test["x"].min() > 40


@pytest.mark.parametrize("transformer", ["LabelEncoder", "OrdinalEncoder"])
def test_target_only_encoding_accepts_the_fit_target_argument(transformer) -> None:
    """Encoding only y must not be mistaken for learning a feature vocabulary."""
    config = _config(transformer, {"columns": ["target"]}, "before_split")
    config["preprocessing"][1]["params"].pop("target_column")

    assert validate_leakage_safety(config, target_column="target") == []


def test_unknown_pre_split_transformer_still_fails_closed() -> None:
    """Allowing known stateless nodes must not accidentally allow unknown plugin behavior."""
    config = _config("UnknownPluginTransformer", {}, "before_split")

    with pytest.raises(ValueError, match="UnknownPluginTransformer"):
        validate_leakage_safety(config)
