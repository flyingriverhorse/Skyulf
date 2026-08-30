"""Unit tests for the job-level Mermaid topology diagram builder."""

import unittest
from typing import Any

from backend.ml_pipeline._execution.diagram import build_pipeline_diagram
from backend.ml_pipeline._execution.schemas import NodeExecutionResult
from backend.ml_pipeline._internal._routers._notebook_builders import (
    _NodeIn,
    pipeline_diagram_md,
)


def _node(
    node_id: str, step_type: str, metadata: dict[str, Any] | None = None
) -> NodeExecutionResult:
    return NodeExecutionResult(
        node_id=node_id, status="success", step_type=step_type, metadata=metadata or {}
    )


class TestBuildPipelineDiagram(unittest.TestCase):
    def test_linear_chain_renders_steps_and_model(self):
        results = {
            "load": _node("load", "data_loader"),
            "scale": _node("scale", "standard_scaler"),
            "split": _node("split", "train_test_splitter"),
            "train": _node("train", "training"),
        }
        diagram = build_pipeline_diagram(results, model_type="logistic_regression")

        assert diagram is not None
        assert diagram.startswith("flowchart TD")
        assert 'data["Input Data"]' in diagram
        assert 'pp0["standard_scaler"]' in diagram
        assert 'pp1["train_test_splitter"]' in diagram
        # The loader is hidden (the core diagram already starts at Input Data)
        # and the trainer becomes the model stage labeled with the algorithm.
        assert "load" not in diagram
        assert 'model(["Logistic Regression"])' in diagram
        assert diagram.splitlines()[-1].endswith("--> model")

    def test_display_name_and_summary_are_used_instead_of_node_id(self):
        results = {
            "3-a415-6ee3f5cf0b4d": _node(
                "3-a415-6ee3f5cf0b4d",
                "label_encoder",
                {"display_name": "Encoding", "summary": "3 categories"},
            ),
            "7-b902-11aa": _node("7-b902-11aa", "training", {"summary": "acc 0.87 · f1 0.84"}),
        }
        diagram = build_pipeline_diagram(results, model_type="logistic_regression")

        assert diagram is not None
        assert 'pp0["Encoding (label_encoder)<br/>3 categories"]' in diagram
        assert 'model(["Logistic Regression<br/>acc 0.87 · f1 0.84"])' in diagram
        # Internal node ids never leak into the diagram.
        assert "3-a415" not in diagram
        assert "7-b902" not in diagram

    def test_display_name_equal_to_step_type_is_not_duplicated(self):
        results = {"n": _node("n", "simple_imputer", {"display_name": "simple_imputer"})}
        diagram = build_pipeline_diagram(results)
        assert diagram is not None
        assert 'pp0["simple_imputer"]' in diagram
        assert "(simple_imputer)" not in diagram

    def test_modeling_falls_back_to_step_type_without_model_type(self):
        results = {"train": _node("train", "training")}
        diagram = build_pipeline_diagram(results)
        assert diagram is not None
        assert 'model(["Training"])' in diagram

    def test_returns_none_when_nothing_renderable_ran(self):
        results = {"load": _node("load", "data_loader")}
        assert build_pipeline_diagram(results) is None
        assert build_pipeline_diagram({}) is None

    def test_missing_step_type_is_labeled_unknown(self):
        results = {"n1": NodeExecutionResult(node_id="n1", status="success")}
        diagram = build_pipeline_diagram(results)
        assert diagram is not None
        assert 'pp0["unknown"]' in diagram
        assert "n1" not in diagram

    def test_second_modeling_node_is_rendered_as_a_step(self):
        results = {
            "train": _node("train", "training"),
            "tune": _node("tune", "tuning"),
        }
        diagram = build_pipeline_diagram(results, model_type="random_forest")
        assert diagram is not None
        assert 'model(["Random Forest"])' in diagram
        assert 'pp0["tuning"]' in diagram

    def test_params_digest_fills_in_when_no_runtime_summary(self):
        results = {"impute": _node("impute", "simple_imputer")}
        diagram = build_pipeline_diagram(
            results,
            node_params={"impute": {"strategy": "mean", "_display_name": "Impute"}},
        )
        assert diagram is not None
        assert 'pp0["simple_imputer<br/>strategy: mean"]' in diagram
        # Internal underscore params never leak into the digest.
        assert "Impute" not in diagram

    def test_runtime_summary_wins_over_params_digest(self):
        results = {"impute": _node("impute", "simple_imputer", {"summary": "filled 12 values"})}
        diagram = build_pipeline_diagram(results, node_params={"impute": {"strategy": "mean"}})
        assert diagram is not None
        assert "filled 12 values" in diagram
        assert "strategy" not in diagram

    def test_build_failure_returns_none_instead_of_blocking_the_run(self):
        from unittest.mock import patch

        results = {"scale": _node("scale", "standard_scaler")}
        with patch(
            "backend.ml_pipeline._execution.diagram.build_mermaid_diagram",
            side_effect=RuntimeError("mermaid broke"),
        ):
            assert build_pipeline_diagram(results) is None


class TestNotebookDiagramCell(unittest.TestCase):
    def test_diagram_md_wraps_a_mermaid_fence_with_steps_and_model(self):
        preprocess = [
            _NodeIn(
                node_id="scale",
                step_type="standard_scaler",
                params={"columns": ["a", "b"]},
            )
        ]
        model = _NodeIn(
            node_id="train",
            step_type="training",
            params={"algorithm": "logistic_regression"},
        )
        md = pipeline_diagram_md(preprocess, model)

        assert md.startswith("## Pipeline topology\n\n```mermaid\n")
        assert md.rstrip().endswith("```")
        assert 'pp0["standard_scaler<br/>columns: a, b"]' in md
        assert 'model(["Logistic Regression"])' in md

    def test_diagram_md_uses_display_name_not_node_id(self):
        preprocess = [
            _NodeIn(
                node_id="9f8e7d6c",
                step_type="simple_imputer",
                params={"strategy": "mean", "_display_name": "Impute"},
            )
        ]
        md = pipeline_diagram_md(preprocess, None)
        assert 'pp0["Impute (simple_imputer)<br/>strategy: mean"]' in md
        assert "9f8e7d6c" not in md
        assert "_display_name" not in md
        assert "model" not in md


if __name__ == "__main__":
    unittest.main()
