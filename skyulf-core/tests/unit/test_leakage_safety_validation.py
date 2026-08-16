"""Tests for the standalone preprocessing leakage-safety diagnostic."""

import skyulf
from skyulf.pipeline import SkyulfPipeline


def test_validate_leakage_safety_reports_learned_step_before_splitter():
    """A learned preprocessing step before a split should be reported."""
    config = {
        "preprocessing": [
            {"name": "fill missing", "transformer": "SimpleImputer", "params": {}},
            {"name": "split", "transformer": "TrainTestSplitter", "params": {}},
        ],
        "modeling": {},
    }

    warnings = skyulf.validate_leakage_safety(config)

    assert len(warnings) == 1
    assert "Step 0 ('SimpleImputer')" in warnings[0]
    assert "step 1, 'TrainTestSplitter'" in warnings[0]


def test_validate_leakage_safety_allows_splitter_first_or_absent():
    """A safe order or no train/test boundary should produce no warnings."""
    splitter_first = {
        "preprocessing": [
            {"name": "split", "transformer": "Split", "params": {}},
            {"name": "scale", "transformer": "StandardScaler", "params": {}},
        ],
        "modeling": {},
    }
    no_splitter = {
        "preprocessing": [
            {"name": "scale", "transformer": "StandardScaler", "params": {}},
        ],
        "modeling": {},
    }

    assert skyulf.validate_leakage_safety(splitter_first) == []
    assert skyulf.validate_leakage_safety(no_splitter) == []


def test_pipeline_validate_leakage_safety_delegates_to_module_function():
    """The pipeline convenience method should return the module result."""
    config = {
        "preprocessing": [
            {"name": "encode", "transformer": "OneHotEncoder", "params": {}},
            {"name": "split", "transformer": "TrainTestSplitter", "params": {}},
        ],
        "modeling": {},
    }

    assert SkyulfPipeline(config).validate_leakage_safety() == skyulf.validate_leakage_safety(
        config
    )


def test_validate_leakage_safety_is_importable_from_package_top_level():
    """The diagnostic should be available from the package root."""
    from skyulf.leakage import validate_leakage_safety

    assert skyulf.validate_leakage_safety is validate_leakage_safety
