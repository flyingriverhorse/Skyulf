"""QW-83: node metadata must remain data when exported into Python comments."""

import ast

import pytest

from backend.ml_pipeline._internal._routers import notebook_export as ne
from backend.ml_pipeline._internal._routers._notebook_builders import _NodeIn, _PipelineIn


@pytest.mark.parametrize("field", ["node_id", "step_type"])
@pytest.mark.parametrize("separator", ["\n", "\r", "\r\n"])
def test_full_export_node_comments_cannot_add_statements(field, separator):
    """User metadata must not inject an assignment into an executable export cell."""
    values = {"node_id": "scaler", "step_type": "StandardScaler"}
    values[field] += f"{separator}qw83_marker = 1{separator}#"
    cfg = _PipelineIn(nodes=[_NodeIn(**values)])
    notebook = ne._build_full_notebook(cfg, "probe", "data.csv")
    code = ["".join(cell["source"]) for cell in notebook["cells"] if cell["cell_type"] == "code"]
    for source in code:
        tree = ast.parse(source)
        compile(tree, "export.ipynb", "exec")
        assert not any(
            isinstance(node, ast.Name) and node.id == "qw83_marker" for node in ast.walk(tree)
        )
    step_tree = ast.parse(next(source for source in code if "step01_calc =" in source))
    registry_calls = [
        node
        for node in ast.walk(step_tree)
        if isinstance(node, ast.Call)
        and isinstance(node.func, ast.Attribute)
        and node.func.attr in {"get_calculator", "get_applier"}
    ]
    assert [ast.literal_eval(call.args[0]) for call in registry_calls] == [values["step_type"]] * 2


def test_normal_preprocessing_export_keeps_executable_steps():
    """Escaping comment metadata must preserve the normal fit/apply code and parameters."""
    cfg = _PipelineIn(
        nodes=[_NodeIn(node_id="scaler", step_type="StandardScaler", params={"with_mean": False})]
    )
    notebook = ne._build_full_notebook(cfg, "probe", "data.csv")
    sources = ["".join(cell["source"]) for cell in notebook["cells"] if cell["cell_type"] == "code"]
    for source in sources:
        compile(source, "export.ipynb", "exec")
    tree = ast.parse(next(source for source in sources if "step01_calc =" in source))
    assignments = [node for node in tree.body if isinstance(node, ast.Assign)]
    assert [ast.unparse(node.targets[0]) for node in assignments] == [
        "step01_calc",
        "step01_apply",
        "step01_config",
        "step01_artifact",
        "df",
    ]
    assert ast.literal_eval(assignments[2].value) == {"with_mean": False}
