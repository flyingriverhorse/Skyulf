"""Tests for ``SkyulfPipeline.describe()`` and ``SkyulfPipeline.to_mermaid()``."""

from skyulf.pipeline import SkyulfPipeline

_CONFIG = {
    "preprocessing": [
        {"name": "scale", "transformer": "StandardScaler", "params": {"columns": ["a", "b"]}},
        {"name": "impute", "transformer": "SimpleImputer", "params": {"strategy": "mean"}},
    ],
    "modeling": {"type": "logistic_regression", "node_id": "m1", "C": 1.0},
}


def test_describe_lists_steps_and_model():
    text = SkyulfPipeline(_CONFIG).describe()
    assert "Preprocessing (2 steps):" in text
    assert "1. scale [StandardScaler]" in text
    assert "2. impute [SimpleImputer]" in text
    assert "- strategy: mean" in text
    assert "type: logistic_regression" in text
    assert "- C: 1.0" in text


def test_describe_handles_empty_pipeline():
    text = SkyulfPipeline({"preprocessing": [], "modeling": {}}).describe()
    assert "Preprocessing (0 steps):" in text
    assert "Modeling:" in text
    assert text.count("(none)") == 2


def test_to_mermaid_renders_flowchart():
    diagram = SkyulfPipeline(_CONFIG).to_mermaid()
    assert diagram.startswith("flowchart TD")
    assert "data[Input Data]" in diagram
    assert "data --> pp0" in diagram
    assert "pp0 --> pp1" in diagram
    assert "pp1 --> model" in diagram
    assert "model([logistic_regression])" in diagram


def test_to_mermaid_without_model_has_no_model_node():
    diagram = SkyulfPipeline(
        {"preprocessing": _CONFIG["preprocessing"], "modeling": {}}
    ).to_mermaid()
    assert "model(" not in diagram
    assert "data --> pp0" in diagram


def test_to_mermaid_escapes_brackets_in_labels():
    cfg = {
        "preprocessing": [{"name": "weird[1]", "transformer": 'a"b', "params": {}}],
        "modeling": {},
    }
    diagram = SkyulfPipeline(cfg).to_mermaid()
    label_line = next(line for line in diagram.splitlines() if line.strip().startswith("pp0["))
    inner = label_line.strip()[len("pp0[") : -1]
    assert "[" not in inner and "]" not in inner
    assert '"' not in inner
