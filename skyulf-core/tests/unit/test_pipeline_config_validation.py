"""Regression tests for pipeline configuration validation."""

from copy import deepcopy
from typing import Any

import pandas as pd
import pytest

from skyulf.pipeline import SkyulfPipeline
from skyulf.preprocessing.pipeline import FeatureEngineer


def test_typoed_transformer_key_reports_step_index_and_missing_key() -> None:
    """A misspelled transformer key fails at construction with helpful context."""
    config = {
        "preprocessing": [
            {
                "name": "imputer",
                "transfomer": "SimpleImputer",
                "params": {"strategy": "mean"},
            }
        ]
    }

    with pytest.raises(ValueError) as exc_info:
        SkyulfPipeline(config)

    message = str(exc_info.value)
    assert "preprocessing[0]" in message
    assert "transformer" in message
    assert "transfomer" in message
    assert "KeyError" not in message


def test_unknown_transformer_name_lists_available_options() -> None:
    """An unknown transformer retains the registry's available-node hint."""
    engineer = FeatureEngineer(
        [{"name": "invalid-step", "transformer": "NotARealTransformer", "params": {}}]
    )

    with pytest.raises(ValueError) as exc_info:
        engineer.fit_transform(pd.DataFrame({"feature": [1, 2]}))

    message = str(exc_info.value)
    assert "NotARealTransformer" in message
    assert "Available nodes:" in message
    assert "SimpleImputer" in message


def test_unknown_model_name_lists_available_options() -> None:
    """An unknown model retains the pipeline's available-node hint."""
    with pytest.raises(ValueError) as exc_info:
        SkyulfPipeline({"modeling": {"type": "not_a_real_model"}})

    message = str(exc_info.value)
    assert "not_a_real_model" in message
    assert "Available" in message
    assert "logistic_regression" in message


def test_existing_valid_config_initializes_without_behavior_change() -> None:
    """A real supported pipeline configuration still initializes normally."""
    pipeline = SkyulfPipeline(
        {
            "preprocessing": [
                {
                    "name": "imputer",
                    "transformer": "SimpleImputer",
                    "params": {"strategy": "mean"},
                }
            ],
            "modeling": {"type": "logistic_regression", "params": {"C": 1.0}},
        }
    )

    assert pipeline.preprocessing_steps[0]["transformer"] == "SimpleImputer"
    assert pipeline.model_estimator is not None


def test_unknown_params_are_not_rejected_at_config_validation() -> None:
    """Node-specific params remain permissive until the node handles them."""
    pipeline = SkyulfPipeline(
        {
            "preprocessing": [
                {
                    "name": "imputer",
                    "transformer": "SimpleImputer",
                    "params": {"custom_node_parameter": "preserved"},
                }
            ]
        }
    )

    assert pipeline.preprocessing_steps[0]["params"]["custom_node_parameter"] == "preserved"


@pytest.mark.parametrize(
    "config,diagnostic",
    [
        (None, "config: must be a dictionary"),
        ([], "config: must be a dictionary"),
        ({"preprocessing": "scale"}, "preprocessing: must be a list or sequence"),
        ({"preprocessing": 42}, "preprocessing: Input should be an instance of Sequence"),
        ({"preprocessing": [42]}, "preprocessing[0]: must be a dictionary"),
        (
            {"preprocessing": [{"name": 42, "transformer": "StandardScaler"}]},
            "preprocessing[0].name: must be a string",
        ),
        ({"modeling": []}, "modeling: must be a dictionary"),
        ({"modeling": {"type": 42}}, "modeling.type: must be a string"),
    ],
)
def test_invalid_config_shapes_fail_at_pipeline_construction(config: Any, diagnostic: str) -> None:
    """Malformed wire configurations must fail with a location before any node can run."""
    with pytest.raises(ValueError) as error:
        SkyulfPipeline(config)

    assert diagnostic in str(error.value)


def test_structural_errors_are_aggregated_without_mutating_config() -> None:
    """Users must see every malformed field and the correct later-step index in one error."""
    config: Any = {
        "preprocessing": [
            {"name": "valid", "transformer": "StandardScaler"},
            {"name": 42, "unrelated": "value"},
        ],
        "modeling": {"type": False},
    }
    original = deepcopy(config)

    with pytest.raises(ValueError) as error:
        SkyulfPipeline(config)

    message = str(error.value)
    assert "3 problems found" in message
    assert "preprocessing[1].name: must be a string" in message
    assert "preprocessing[1]: missing required key 'transformer'" in message
    assert "modeling.type: must be a string" in message
    assert "did you mean" not in message
    assert config == original


def test_standalone_preprocessing_validates_before_fitting() -> None:
    """FeatureEngineer must reject malformed steps even without a surrounding pipeline."""
    steps: Any = [
        {"name": "valid", "transformer": "StandardScaler"},
        {"name": "invalid", "transformer": False},
    ]

    with pytest.raises(ValueError) as error:
        FeatureEngineer(steps)

    assert "preprocessing[1].transformer: must be a string" in str(error.value)


def test_validation_preserves_extension_fields_and_optional_sections() -> None:
    """Structural validation must preserve plugin data and allow omitted modeling sections."""
    config: Any = {
        "plugin_options": {"custom": ["value"]},
        "preprocessing": [
            {
                "name": "imputer",
                "transformer": "SimpleImputer",
                "params": {"strategy": "mean", "custom": [1, 2]},
                "plugin_metadata": {"key": "value"},
            }
        ],
    }
    original = deepcopy(config)
    pipeline = SkyulfPipeline(config)

    assert pipeline.model_estimator is None
    assert pipeline.config == original
