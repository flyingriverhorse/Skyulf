"""Tests for pipeline fingerprint, model card, and structured warnings."""

import pandas as pd

from skyulf.core import SkyulfWarning, WarningCategory
from skyulf.pipeline import SkyulfPipeline

_CONFIG = {
    "preprocessing": [
        {"name": "scale", "transformer": "StandardScaler", "params": {"columns": ["a", "b"]}},
    ],
    "modeling": {"type": "logistic_regression", "node_id": "m1", "C": 1.0},
}


def _frame() -> pd.DataFrame:
    return pd.DataFrame(
        {
            "a": [1.0, 2.0, 3.0, 4.0, 5.0, 6.0],
            "b": [6.0, 5.0, 4.0, 3.0, 2.0, 1.0],
            "target": [0, 1, 0, 1, 0, 1],
        }
    )


def test_fingerprint_is_deterministic_for_same_config():
    assert SkyulfPipeline(_CONFIG).fingerprint() == SkyulfPipeline(_CONFIG).fingerprint()


def test_fingerprint_changes_with_config():
    other = {
        "preprocessing": _CONFIG["preprocessing"],
        "modeling": {"type": "logistic_regression", "node_id": "m1", "C": 2.0},
    }
    assert SkyulfPipeline(_CONFIG).fingerprint() != SkyulfPipeline(other).fingerprint()


def test_fingerprint_changes_after_fit():
    pipe = SkyulfPipeline(_CONFIG)
    before = pipe.fingerprint()
    pipe.fit(_frame(), target_column="target")
    assert pipe.fingerprint() != before


def test_is_fitted_flag():
    pipe = SkyulfPipeline(_CONFIG)
    assert pipe.is_fitted() is False
    pipe.fit(_frame(), target_column="target")
    assert pipe.is_fitted() is True


def test_export_model_card_structure():
    card = SkyulfPipeline(_CONFIG).export_model_card()
    assert card["schema_version"] == "1.0"
    assert card["fitted"] is False
    assert card["metrics"] is None
    assert card["model"]["type"] == "logistic_regression"
    assert card["model"]["params"]["C"] == 1.0
    assert card["preprocessing"][0]["transformer"] == "StandardScaler"
    assert isinstance(card["fingerprint"], str) and len(card["fingerprint"]) == 64


def test_export_model_card_after_fit_has_metrics():
    pipe = SkyulfPipeline(_CONFIG)
    pipe.fit(_frame(), target_column="target")
    card = pipe.export_model_card()
    assert card["fitted"] is True
    assert card["metrics"] is not None


def test_skyulf_warning_to_dict():
    warn = SkyulfWarning(
        category=WarningCategory.DEGENERATE,
        code="onehot.single_category",
        message="Column 'x' has a single category.",
        context={"column": "x", "n_categories": 1},
    )
    payload = warn.to_dict()
    assert payload == {
        "category": "degenerate",
        "code": "onehot.single_category",
        "message": "Column 'x' has a single category.",
        "context": {"column": "x", "n_categories": 1},
    }


def test_skyulf_warning_default_context_is_isolated():
    a = SkyulfWarning(WarningCategory.CONFIG, "c1", "m1")
    b = SkyulfWarning(WarningCategory.CONFIG, "c2", "m2")
    a.context["k"] = "v"
    assert b.context == {}
