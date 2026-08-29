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
    assert 'data["Input Data"]' in diagram
    assert "data --> pp0" in diagram
    assert "pp0 --> pp1" in diagram
    assert "pp1 --> model" in diagram
    assert 'pp0["scale (StandardScaler)<br/>columns: a, b"]' in diagram
    assert 'pp1["impute (SimpleImputer)<br/>strategy: mean"]' in diagram
    assert 'model(["Logistic Regression<br/>C: 1.0"])' in diagram


def test_to_mermaid_markdown_wraps_the_diagram_in_a_fence():
    pipe = SkyulfPipeline(_CONFIG)
    md = pipe.to_mermaid_markdown()
    assert md.startswith("# Pipeline topology\n\n```mermaid\n")
    assert md.rstrip().endswith("```")
    assert pipe.to_mermaid() in md
    assert pipe.to_mermaid_markdown(heading=None).startswith("```mermaid\n")
    assert pipe.to_mermaid_markdown(heading="My flow").startswith("# My flow\n\n")


def test_to_mermaid_humanizes_model_type():
    from skyulf.pipeline.diagram import build_mermaid_diagram

    for raw, expected in [
        ("random_forest_classifier", "Random Forest Classifier"),
        ("RandomForestClassifier", "Random Forest Classifier"),
        ("xgb", "Xgb"),
    ]:
        diagram = build_mermaid_diagram([], {"type": raw})
        assert f'model(["{expected}"])' in diagram


def test_to_mermaid_details_prefer_explicit_details_over_params():
    from skyulf.pipeline.diagram import build_mermaid_diagram

    diagram = build_mermaid_diagram(
        [
            {
                "name": "impute",
                "transformer": "SimpleImputer",
                "params": {"strategy": "mean"},
                "details": "filled 12 values",
            }
        ],
        {},
    )
    assert 'pp0["impute (SimpleImputer)<br/>filled 12 values"]' in diagram
    assert "strategy" not in diagram


def test_to_mermaid_param_summary_skips_internal_and_nested():
    from skyulf.pipeline.diagram import build_mermaid_diagram

    diagram = build_mermaid_diagram(
        [
            {
                "name": "s",
                "transformer": "T",
                "params": {"_display_name": "hidden", "nested": {"a": 1}, "keep": 1},
            }
        ],
        {},
    )
    assert 'pp0["s (T)<br/>keep: 1"]' in diagram
    assert "hidden" not in diagram
    assert "nested" not in diagram


def test_to_mermaid_without_model_has_no_model_node():
    diagram = SkyulfPipeline(
        {"preprocessing": _CONFIG["preprocessing"], "modeling": {}}
    ).to_mermaid()
    assert "model" not in diagram.replace("flowchart", "")
    assert "data --> pp0" in diagram


def test_to_mermaid_labels_are_quoted_and_quotes_escaped():
    # Unquoted parens/brackets break mermaid's flowchart parser
    # (labels like "node-id (DropMissingColumns)"), so every label must be
    # double-quoted; embedded double quotes are neutralised.
    cfg = {
        "preprocessing": [{"name": "weird[1]", "transformer": 'a"b (x)', "params": {}}],
        "modeling": {},
    }
    diagram = SkyulfPipeline(cfg).to_mermaid()
    assert '    pp0["weird[1] (a\'b (x))"]' in diagram


def test_param_summary_truncates_long_lists_and_values():
    from skyulf.pipeline.diagram import params_summary

    # More than six list items collapse to six + an ellipsis marker.
    text = params_summary({"cols": list("abcdefgh")})
    assert text == "cols: a, b, c, d, e, f, …"

    # A single value longer than 24 chars is cut with an ellipsis.
    text = params_summary({"pattern": "x" * 40})
    assert text == "pattern: " + "x" * 23 + "…"


def test_param_summary_caps_at_three_entries():
    from skyulf.pipeline.diagram import params_summary

    text = params_summary({"a": 1, "b": 2, "c": 3, "d": 4})
    assert text == "a: 1 · b: 2 · c: 3"
    assert "d:" not in text


def test_param_summary_returns_none_when_everything_is_skipped():
    from skyulf.pipeline.diagram import params_summary

    assert params_summary({"_hidden": 1, "empty": None, "nested": {"a": 1}}) is None


def test_param_summary_truncates_overlong_detail_lines():
    from skyulf.pipeline.diagram import params_summary

    text = params_summary({"first": "x" * 24, "second": "y" * 24, "third": "z" * 24})
    assert len(text) == 72
    assert text.endswith("…")
