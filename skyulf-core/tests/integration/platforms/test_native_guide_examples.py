"""Execute explicitly marked native documentation chains without cloud access."""

import re
from pathlib import Path

import pytest

GUIDES = Path(__file__).resolve().parents[4] / "docs" / "user_guide"
EXAMPLES = (
    pytest.param(
        "pipeline_quickstart",
        "quickstart",
        False,
        lambda ns: ns["accuracy"] == 0.5 and ns["predictions"].tolist() == [0, 0],
        id="pipeline-quickstart",
    ),
    pytest.param(
        "step_by_step_no_config",
        "manual",
        False,
        lambda ns: ns["new_predictions"].tolist() == [0, 0],
        id="manual-calculator-applier",
    ),
    pytest.param(
        "inference_flow",
        "fitted-flow",
        False,
        lambda ns: ns["result"]["prediction"].tolist() == pytest.approx([400, 200]),
        id="fitted-inference-flow",
    ),
    pytest.param(
        "inference_bundles",
        "portable-local",
        False,
        lambda ns: ns["output"]["prediction"].tolist() == pytest.approx([40, 20]),
        id="portable-bundle",
    ),
    pytest.param(
        "preprocessing_context",
        "context-probe",
        False,
        lambda ns: ns["report"]["status"] == "passed",
        id="preprocessing-probe",
    ),
    pytest.param(
        "preprocessing_context",
        "history",
        False,
        lambda ns: ns["context_report"]["status"] == "requires_context",
        id="continued-history",
    ),
    pytest.param(
        "spark",
        "spark-contracts",
        False,
        lambda ns: ns["result"]["amount"].tolist() == [3.0],
        id="spark-independent-contracts",
    ),
    pytest.param(
        "mlflow_tracking",
        "tracking",
        True,
        lambda ns: ns["stored"].info.status == "FINISHED" and ns["offline"].run_id is None,
        id="local-mlflow-tracking",
    ),
    pytest.param(
        "mlflow_models",
        "models",
        True,
        lambda ns: ns["pyfunc_predictions"]["prediction"].tolist() == pytest.approx([18, 20]),
        id="local-mlflow-models-and-nullable-transport",
    ),
    pytest.param(
        "mlflow_registry",
        "registry",
        True,
        lambda ns: ns["selected"].version == ns["baseline"].version,
        id="local-registry-promotion-and-rollback",
    ),
    pytest.param(
        "databricks_sdk",
        "sdk-local",
        False,
        lambda ns: ns["predictions"]["prediction"].tolist() == pytest.approx([18, 20]),
        id="standalone-sdk-without-mlflow",
    ),
    pytest.param(
        "databricks_sdk",
        "sdk-tracking",
        True,
        lambda ns: ns["metrics"]["heldout_rmse"] < 1e-10,
        id="sdk-local-tracking",
    ),
)
PAGES = (
    "pipeline_quickstart",
    "step_by_step_no_config",
    "inference_flow",
    "inference_bundles",
    "preprocessing_context",
    "spark",
    "spark_benchmark",
    "mlflow_tracking",
    "mlflow_models",
    "mlflow_registry",
    "databricks_sdk",
    "databricks_batch",
    "databricks_model_sets",
)
MARKED_BLOCK = re.compile(
    r"<!-- skyulf-example: ([a-z_]+) ([a-z0-9-]+) -->\s*\n```python\n(.*?)```",
    re.DOTALL,
)


@pytest.mark.parametrize("page", PAGES)
def test_python_examples_are_classified(page):
    """New snippets need an explicit local, runtime, cloud or illustrative boundary."""
    text = (GUIDES / f"{page}.md").read_text(encoding="utf-8")
    marked = MARKED_BLOCK.findall(text)
    assert len(marked) == len(re.findall(r"```python\n", text))
    assert all(
        scope in {"native", "cloud", "spark_runtime", "illustrative"} for scope, _, _ in marked
    )
    expected = {case.values[1] for case in EXAMPLES if case.values[0] == page}
    assert {chain for scope, chain, _ in marked if scope == "native"} == expected


@pytest.mark.parametrize("page,chain,requires_mlflow,check", EXAMPLES)
def test_complete_native_example(page, chain, requires_mlflow, check, tmp_path, monkeypatch):
    """Literal guide code must fit, save and replay successfully in an isolated directory."""
    # Skip optional MLflow before executing any application imports from the page.
    mlflow = pytest.importorskip("mlflow") if requires_mlflow else None
    previous_tracking = mlflow.get_tracking_uri() if mlflow is not None else None
    previous_registry = mlflow.get_registry_uri() if mlflow is not None else None
    text = (GUIDES / f"{page}.md").read_text(encoding="utf-8")
    wanted = {chain, "sdk-local"} if chain == "sdk-tracking" else {chain}
    blocks = [
        code
        for scope, name, code in MARKED_BLOCK.findall(text)
        if scope == "native" and name in wanted
    ]
    assert blocks, f"Missing native chain: {page}/{chain}"
    monkeypatch.chdir(tmp_path)
    monkeypatch.setenv("MLFLOW_DISABLE_AGENT_HINT", "1")
    monkeypatch.setenv("MLFLOW_ENABLE_ARTIFACTS_PROGRESS_BAR", "false")
    namespace = {"__name__": "__skyulf_documentation__"}
    try:
        for index, code in enumerate(blocks):
            exec(compile(code, f"{page}.md:{chain}:{index}", "exec"), namespace)
    finally:
        if mlflow is not None:
            mlflow.set_tracking_uri(previous_tracking)
            mlflow.set_registry_uri(previous_registry)
    assert check(namespace)
