"""Execute documented function helpers through captured project fit/save/reload."""

import ast
import json
import re
import subprocess
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest
from template_sources import copy_template

yaml = pytest.importorskip("yaml")
pytest.importorskip("polars")

from skyulf.data.dataset import SplitDataset
from skyulf.inference.fitted_pipeline import load_pipeline
from skyulf.integrations.databricks.projects.project import load_project_workflow
from skyulf.integrations.databricks.scoring.batch.frame_batch import fit_workflow

ROOT = Path(__file__).resolve().parents[4]
TEMPLATE = ROOT / "skyulf-core/templates/databricks/template/{{.project_name}}"
PAGE = ROOT / "docs/user_guide/databricks_custom_steps.md"


def _example(name, language):
    """Read the literal marked example so prose changes cannot bypass execution."""
    matches = re.findall(
        rf"<!-- example:custom-{name} -->\s*```{language}\n(.*?)```",
        PAGE.read_text(encoding="utf-8"),
        re.DOTALL,
    )
    assert len(matches) == 1, f"Missing or duplicate custom-{name} example"
    return matches[0]


def _project(tmp_path, kind):
    """Install the documented callbacks and recipe into a real generated feature package."""
    features = tmp_path / "src/features"
    copy_template(TEMPLATE / "src/features", features)
    locations = {"budget": "budget.py", "target-mean": "target_mean.py"}
    module = features / locations.get(kind, "preprocessing.py")
    code = _example(f"{kind}-code", "python")
    if kind == "rare":
        code += "\n" + _example("frequency-code", "python")
    module.write_text(code, encoding="utf-8")
    declaration = yaml.safe_load(_example(f"{kind}-recipe", "yaml"))
    config = tmp_path / "config"
    config.mkdir()
    copy_template(TEMPLATE / "config/scoring.yml", config / "scoring.yml")
    (config / "preprocessing.yml").write_text(yaml.safe_dump(declaration), encoding="utf-8")
    (config / "pre_split.yml").write_text("version: 1\nrecipes:\n  default: []\n", encoding="utf-8")
    selected = next(name for name in declaration["recipes"] if name != "default")
    return features, selected


def _case(kind):
    """Use hand-calculated examples including nulls, unseen categories and zero income."""
    cases = {
        "log": (
            {"income": [0.0, 3.0, 8.0, 15.0]},
            {"income": [-2.0, 3.0, np.nan]},
            {"income": [-2.0, 3.0, np.nan], "log_income": [0.0, 1.3862943611198906, np.nan]},
        ),
        "budget": (
            {"income": [100.0, 80.0, 60.0, 40.0], "expenses": [25.0, 40.0, 20.0, 10.0]},
            {"income": [100.0, 80.0, 0.0, np.nan], "expenses": [25.0, 40.0, 10.0, 5.0]},
            {
                "income": [100.0, 80.0, 0.0, np.nan],
                "expenses": [25.0, 40.0, 10.0, 5.0],
                "net_income": [75.0, 40.0, -10.0, np.nan],
                "expense_share": [0.25, 0.5, np.nan, np.nan],
            },
        ),
        "frequency": (
            {"category": ["A", "A", "B", None]},
            {"category": ["A", "B", "NEW", None]},
            {"category": [0.5, 0.25, 0.0, 0.0]},
        ),
        "rare": (
            {"city": ["London", "London", "London", "Paris", "Oslo"]},
            {"city": ["London", "Paris", "Tokyo", None]},
            {"city": [0.6, 0.4, 0.4, 0.0]},
        ),
        "target-mean": (
            {"category": ["A", "A", "B", "B"]},
            {"category": ["A", "B", "NEW", None]},
            {"category": [0.5, 2.5, 1.5, 1.5]},
        ),
    }
    return tuple(pd.DataFrame(data) for data in cases[kind])


def _reload_in_fresh_process(tmp_path, features, query):
    """Restore captured callbacks even after current project files become unusable."""
    for module in features.rglob("*.py"):
        module.write_text("raise RuntimeError('editable source must not run')\n", encoding="utf-8")
    for declaration in (tmp_path / "config").glob("*.yml"):
        declaration.write_text("version: 999\n", encoding="utf-8")
    query.to_json(tmp_path / "query.json", orient="table")
    script = """
import sys
import pandas as pd
from skyulf.inference.fitted_pipeline import load_pipeline
artifact = load_pipeline(sys.argv[1])
query = pd.read_json(sys.argv[2], orient="table")
result = artifact.pipeline.feature_engineer.transform(query)
print(result.to_json(orient="split"))
"""
    completed = subprocess.run(
        [sys.executable, "-c", script, str(tmp_path / "artifact"), str(tmp_path / "query.json")],
        cwd=tmp_path,
        capture_output=True,
        text=True,
        timeout=60,
    )
    assert completed.returncode == 0, completed.stdout + completed.stderr
    return pd.DataFrame(**json.loads(completed.stdout))


@pytest.mark.parametrize("kind", ["log", "budget", "frequency", "rare", "target-mean"])
def test_function_examples_fit_save_and_restore_without_refitting(tmp_path, kind):
    """Literal examples must preserve inputs and reuse saved outputs for null/unseen rows."""
    features, recipe = _project(tmp_path, kind)
    train, query, expected = _case(kind)
    original_train, original_query = train.copy(deep=True), query.copy(deep=True)
    inputs = list(train.columns)
    train["claim_amount"] = np.arange(len(train), dtype=float)
    workflow = load_project_workflow(
        {
            "engine": "pandas",
            "target_column": "claim_amount",
            "input_columns": inputs,
            "pipeline": {"preprocessing": [], "modeling": {"type": "linear_regression"}},
        },
        features,
        pre_split_recipe="default",
        preprocessing_recipe=recipe,
    )
    artifact = fit_workflow(
        workflow["pipeline"],
        SplitDataset(train=train, test=train.iloc[:0]),
        target_column="claim_amount",
        artifact_path=tmp_path / "artifact",
        max_rows=100,
        max_bytes=100_000,
    )
    restored = load_pipeline(tmp_path / "artifact")
    actual = restored.pipeline.feature_engineer.transform(query)
    pd.testing.assert_frame_equal(actual, expected)
    # A different scoring population must not change a learned category mapping.
    repeated = query.iloc[[0, 0]].reset_index(drop=True)
    pd.testing.assert_frame_equal(
        restored.pipeline.feature_engineer.transform(repeated),
        expected.iloc[[0, 0]].reset_index(drop=True),
    )
    pd.testing.assert_frame_equal(train[inputs], original_train)
    pd.testing.assert_frame_equal(query, original_query)
    replayed = _reload_in_fresh_process(tmp_path, features, query)
    pd.testing.assert_frame_equal(replayed.astype(expected.dtypes.to_dict()), expected)
    assert artifact.manifest.project_source_sha256 == restored.manifest.project_source_sha256


def test_shipped_function_excerpts_match_the_generated_callbacks():
    """The guide must show the actual shipped functions while runtime tests check their effects."""
    source = ast.parse((TEMPLATE / "src/features/preprocessing.py").read_text(encoding="utf-8"))
    shipped = {
        node.name: ast.dump(node, include_attributes=False)
        for node in source.body
        if isinstance(node, ast.FunctionDef)
    }
    documented = {}
    for kind in ("log", "frequency", "rare"):
        code = ast.parse(_example(f"{kind}-code", "python"))
        documented.update(
            (node.name, ast.dump(node, include_attributes=False))
            for node in code.body
            if isinstance(node, ast.FunctionDef)
        )
    assert documented == {name: shipped[name] for name in documented}


def test_local_feature_engineer_example_executes_actual_callbacks(tmp_path):
    """The local trial must learn from its training frame before transforming unseen/null rows."""
    features = tmp_path / "src/features"
    copy_template(TEMPLATE / "src/features", features)
    (features / "income_rules.py").write_text(_example("clipping-code", "python"), encoding="utf-8")
    code = _example("local-try", "python")
    completed = subprocess.run(
        [sys.executable, "-c", code + '\nprint(transformed_score.to_json(orient="split"))'],
        cwd=tmp_path,
        capture_output=True,
        text=True,
        timeout=60,
    )
    assert completed.returncode == 0, completed.stdout + completed.stderr
    actual = pd.DataFrame(**json.loads(completed.stdout.splitlines()[-1]))
    pd.testing.assert_frame_equal(
        actual,
        pd.DataFrame({"city": [0.5, 0.0, 0.0, 0.25], "income": [10.3, 39.7, 25.0, np.nan]}),
    )
