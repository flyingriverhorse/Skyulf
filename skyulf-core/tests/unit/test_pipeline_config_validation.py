"""Regression tests for pipeline configuration validation."""

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
