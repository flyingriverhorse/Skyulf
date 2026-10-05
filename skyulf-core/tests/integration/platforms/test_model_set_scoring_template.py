"""Shared scoring templates expose per-model and combined rules in one package."""

from pathlib import Path

from skyulf.inference.project_code import load_project_module
from skyulf.integrations.databricks.projects._project_files import project_source

TEMPLATE = Path(__file__).resolve().parents[3] / "templates/databricks/template/{{.project_name}}"


def test_shared_scoring_exports_both_rule_stages_without_active_business_examples():
    """Consolidation must preserve normal pre-split scoring and keep business rules opt-in."""
    module = load_project_module(project_source(TEMPLATE / "src/features"))
    assert module.build_scoring() == {"reuse_pre_split": True, "skip_target_steps": False}
    assert module.build_combined_rules() == []
    assert module.build_model_rules() == module.build_scoring()
    assert not (TEMPLATE / "src/composition").exists()


def test_profit_callback_preserves_negative_margins():
    """The sample business rule must subtract costs and retain loss-making results."""
    import pandas as pd

    module = load_project_module(project_source(TEMPLATE / "src/features"))
    inputs = pd.DataFrame({"id": [1, 2]}, index=[8, 3])
    predictions = pd.DataFrame(
        {"revenue__prediction": [100.0, 80.0], "cost__prediction": [60.0, 90.0]}, index=inputs.index
    )
    result = module.scoring.profit(inputs, predictions, {})
    assert result.profit.tolist() == [40.0, -10.0]
    assert result.index.tolist() == [8, 3]


def test_shared_feature_callback_replays_with_two_saved_models(tmp_path):
    """The unified scoring package must resolve callbacks after original source changes."""
    import shutil

    import pytest
    from tests.integration.platforms.test_model_set_batch import _saved_set

    from skyulf.inference.model_set import ComponentReference, save_model_set
    from skyulf.inference.model_set_scoring import predict_model_set
    from skyulf.integrations.databricks.model_sets.model_set_project import capture_set_rules

    original, _, query = _saved_set(tmp_path / "original")
    project = tmp_path / "project"
    features = project / "src/features"
    shutil.copytree(TEMPLATE / "src/features", features)
    (project / "config").mkdir()
    rule = {
        "name": "profit",
        "version": "1",
        "function": "scoring.profit",
        "params": {},
        "columns": [{"name": "profit", "dtype": "float64"}],
        "required_components": ["revenue", "cost"],
    }
    with (features / "scoring.py").open("a", encoding="utf-8") as stream:
        stream.write(f"\ndef build_combined_rules():\n    return {[rule]!r}\n")
    config, source = capture_set_rules(
        {"config_path": str(project / "config/workflow.json")},
        {"combined_rules_path": "../features"},
    )
    reference = original.manifest.components[0].reference
    saved = save_model_set(
        tmp_path / "combined",
        {
            name: (
                ComponentReference(
                    name=f"workspace.test.{name}", version="1", digest=reference.digest
                ),
                tmp_path / "original/component",
            )
            for name in ("revenue", "cost")
        },
        record_key_schema=original.manifest.record_key_schema,
        composition_source=source,
        composition_config=config["composition_config"],
    )
    (features / "scoring.py").write_text("raise RuntimeError('changed source must not execute')")
    result = predict_model_set(query, saved)
    assert result.revenue__prediction.tolist() == pytest.approx([29, 18, 41])
    assert result.cost__prediction.tolist() == pytest.approx([29, 18, 41])
    assert result.profit.tolist() == pytest.approx([0, 0, 0])
