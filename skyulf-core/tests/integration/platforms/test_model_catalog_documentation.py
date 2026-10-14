"""Keep the model guide tied to real selectors, defaults and executable recipes."""

import importlib.util
import json
import re
import subprocess
import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[4]
GENERATOR = ROOT / "skyulf-core/templates/databricks/build_model_catalog.py"
DOCUMENT = ROOT / "docs/user_guide/databricks_model_catalog.md"
SCHEMA = GENERATOR.with_name("databricks_template_schema.json")


def _generator():
    """Load the stdlib-only documentation generator without modifying import paths."""
    spec = importlib.util.spec_from_file_location("model_catalog", GENERATOR)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def _examples():
    """Extract the exact snippets users copy, retaining their model and mode."""
    return re.findall(
        r"<!-- model-example:([a-z_]+):(plain|tuned) -->\s*```yaml\n(.*?)```",
        DOCUMENT.read_text(encoding="utf-8"),
        re.DOTALL,
    )


def test_catalog_covers_every_wizard_model_and_two_examples():
    """New selector options must gain a complete card and both usable entry forms."""
    generator = _generator()
    models = generator.wizard_models()
    registry = generator.load_registry()
    properties = json.loads(SCHEMA.read_text(encoding="utf-8"))["properties"]
    selectable = {
        value
        for field in properties.values()
        for value in field.get("enum", [])
        if value in registry.MODEL_HYPERPARAMETERS
    }
    document = DOCUMENT.read_text(encoding="utf-8")
    cards = re.findall(r"^## ([a-z_]+)$", document, re.MULTILINE)
    assert set(cards) == selectable == set(models)
    assert len(cards) == len(set(cards)) == 34
    assert list(models.values()).count("regression") == 16
    assert list(models.values()).count("classification") == 18
    cases = [(key, mode) for key, mode, _ in _examples()]
    assert len(cases) == 68
    assert set(cases) == {(key, mode) for key in models for mode in ("plain", "tuned")}
    for key in models:
        card = document.split(f"## {key}\n", 1)[1].split("\n## ", 1)[0]
        for required in (
            "Task:",
            "**Sample weights:**",
            "**Optional package:**",
            "**Spark workers / serving:**",
            "**Halving resource:**",
            "**Common mistake:**",
        ):
            assert required in card, (key, required)


def test_catalog_tables_match_registry_and_runtime_constructor_sources():
    """Catalog controls, search axes and explicit runtime defaults cannot silently drift."""
    generator = _generator()
    source = DOCUMENT.read_text(encoding="utf-8")
    assert generator.render_document(source) == source


def test_catalog_check_runs_without_site_packages():
    """Maintainers can detect stale generated tables without installing optional ML libraries."""
    result = subprocess.run(
        [sys.executable, "-I", "-S", str(GENERATOR), "--check"],
        capture_output=True,
        text=True,
        check=False,
    )
    assert result.returncode == 0, result.stdout + result.stderr
    assert "up to date" in result.stdout


def _optional_model_package(key):
    """Skip only examples whose explicitly optional runtime package is unavailable."""
    if key.startswith("xgboost_"):
        pytest.importorskip("xgboost")
    if key.startswith("lgbm_"):
        pytest.importorskip("lightgbm")


def _loaded_pipeline(tmp_path, key, source):
    """Use the real YAML loader and Bundle-to-Core model/search builders."""
    pytest.importorskip("yaml")
    pytest.importorskip("polars")
    _optional_model_package(key)
    from skyulf.integrations.databricks.projects.yaml_config import read_training_config
    from skyulf.integrations.databricks.projects.yaml_models import model_entries, model_pipeline
    from skyulf.integrations.databricks.training.tuning.cv import CVSpec
    from skyulf.integrations.databricks.training.tuning.search import prepare_search_pipeline

    task = _generator().wizard_models()[key]
    directory = tmp_path / key
    directory.mkdir()
    (directory / "training.yml").write_text(
        f"version: 1\ndefaults:\n  task: {task}\n" + source, encoding="utf-8"
    )
    document = read_training_config(directory)
    assert document is not None
    entries = model_entries(document)
    assert len(entries) == 1
    entry = entries["candidate"]
    assert entry["model"]["type"] == key
    cv = CVSpec(
        enabled=True, folds=3, method="stratified_k_fold" if task == "classification" else "k_fold"
    )
    pipeline = prepare_search_pipeline(
        model_pipeline(entry), cv, target_column="target", event_column=None
    )
    return pipeline, task, entry["model"]["params"]


@pytest.mark.parametrize(
    "key,mode,source", _examples(), ids=[f"{key}-{mode}" for key, mode, _ in _examples()]
)
def test_catalog_examples_load_build_and_selected_recipes_fit(tmp_path, key, mode, source):
    """All snippets build; every untuned model and representative tuned ensembles really predict."""
    prepared, task, params = _loaded_pipeline(tmp_path, key, source)
    from skyulf.modeling.capabilities import model_supports_sample_weight
    from skyulf.pipeline import SkyulfPipeline

    pipeline = SkyulfPipeline(prepared)
    assert pipeline.model_estimator is not None
    assert model_supports_sample_weight(key, params) is not key.startswith("k_neighbors_")
    fitted_tuners = {
        "ridge_regression",
        "logistic_regression",
        "voting_regressor",
        "voting_classifier",
        "stacking_regressor",
        "stacking_classifier",
    }
    if mode == "plain" or key in fitted_tuners:
        _fit_and_predict(pipeline, task)


def _fit_and_predict(pipeline, task):
    """Fit useful small positive-valued data, including NB and internal ensemble CV."""
    import numpy as np
    import pandas as pd

    from skyulf.data.dataset import SplitDataset

    rng = np.random.default_rng(42)
    frame = pd.DataFrame(rng.uniform(0.1, 2.0, size=(90, 3)), columns=["x", "z", "w"])
    signal = 20.0 * frame["x"] - 10.0 * frame["z"] + 5.0 * frame["w"]
    frame["target"] = (signal > signal.median()).astype(int) if task == "classification" else signal
    train, test = frame.iloc[:72], frame.iloc[72:]
    pipeline.fit(SplitDataset(train=train, test=test), target_column="target")
    predictions = pipeline.predict(test.drop(columns="target"))
    assert len(predictions) == len(test)
    assert np.isfinite(np.asarray(predictions, dtype=float)).all()
    assert set(predictions) <= {0, 1} if task == "classification" else np.std(predictions) > 0
