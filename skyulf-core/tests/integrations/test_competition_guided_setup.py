"""Initializer choices must produce the actual competing recipes without hand edits."""

import json
from pathlib import Path

import pytest
from jsonschema import Draft7Validator

ROOT = Path(__file__).resolve().parents[2] / "templates/databricks"


def test_ensemble_questions_are_independent_per_candidate():
    """Each selected ensemble exposes only its own applicable base and strategy controls."""
    props, shown = _visible(
        task="classification",
        competition_candidate_count="2",
        competition_classification_model_1="voting_classifier",
        competition_classification_model_2="stacking_classifier",
        competition_ensemble_1_classification_base_count="3",
        competition_ensemble_2_classification_base_count="2",
    )
    assert "competition_ensemble_1_classification_base_3" in shown
    assert "competition_ensemble_2_classification_base_3" not in shown
    assert "competition_ensemble_1_voting" in shown
    assert "competition_ensemble_2_voting" not in shown
    assert "competition_ensemble_2_classification_final" in shown
    assert "competition_ensemble_1_classification_final" not in shown
    assert "competition_ensemble_1_classification_weight_3" in shown
    assert "competition_ensemble_2_classification_weight_1" not in shown
    assert not any(name.startswith("competition_ensemble_3_") for name in shown)
    assert "logistic_regression" in props["competition_ensemble_1_classification_base_1"]["enum"]


def _visible(**overrides):
    """Evaluate real initializer predicates with all defaults filled as the CLI does."""
    props = json.loads((ROOT / "databricks_template_schema.json").read_text())["properties"]
    values = {name: prop["default"] for name, prop in props.items()}
    values.update({"training_layout": "model_competition", **overrides})
    shown = {
        name
        for name, prop in props.items()
        if not Draft7Validator(prop.get("skip_prompt_if", False)).is_valid(values)
    }
    return props, shown


@pytest.mark.parametrize("task", ["regression", "classification"])
@pytest.mark.parametrize("count", ["2", "3", "8"])
def test_candidate_questions_follow_count_and_task(task, count):
    """Only requested task-compatible model slots should appear during setup."""
    props, shown = _visible(task=task, competition_candidate_count=count)
    assert {
        "competition_candidate_count",
        "competition_preprocessing_recipe",
        "competition_max_trials",
        "search_strategy",
        "search_n_trials",
    } <= shown
    for position in range(1, 9):
        name = f"competition_{task}_model_{position}"
        assert (name in shown) == (position <= int(count))
        assert props[name]["enum"] == props[f"{task}_model"]["enum"]
    other = "classification" if task == "regression" else "regression"
    assert not any(name.startswith(f"competition_{other}_model_") for name in shown)
    assert not {"regression_search_metric", "classification_search_metric"} & shown


def test_inactive_ensemble_slots_do_not_open_questions():
    """A hidden third voting default must not activate ensemble setup for two normal models."""
    _, shown = _visible(competition_candidate_count="2")
    assert not any(name.startswith("competition_ensemble_") for name in shown)
    _, shown = _visible(
        task="classification",
        competition_candidate_count="2",
        competition_classification_model_2="stacking_classifier",
    )
    assert {
        "competition_ensemble_2_classification_base_count",
        "competition_ensemble_2_cv",
        "competition_ensemble_2_classification_final",
        "competition_ensemble_2_calibrate",
    } <= shown
    assert "competition_ensemble_2_voting" not in shown


@pytest.mark.parametrize("layout", ["single_model", "multi_target"])
def test_other_layouts_hide_competition_questions(layout):
    """Existing model and model-set setup must not gain unrelated questions."""
    _, shown = _visible(training_layout=layout)
    assert not any(name.startswith("competition_") for name in shown)


@pytest.mark.parametrize("task,count", [("classification", "13"), ("regression", "14")])
def test_cli_renders_all_base_positions_for_eighth_candidate(tmp_path, task, count):
    """Two-digit base positions and the final candidate slot must retain every selected model."""
    import runpy

    from test_databricks_bundle_generation import CLI, PROFILE, _generate_project

    if not CLI or not PROFILE:
        pytest.skip("Set SKYULF_BUNDLE_CLI_TEST_PROFILE for real CLI generation.")
    model = "voting_classifier" if task == "classification" else "voting_regressor"
    project = _generate_project(
        tmp_path,
        training_layout="model_competition",
        task=task,
        competition_candidate_count="8",
        **{
            f"competition_{task}_model_8": model,
            f"competition_ensemble_8_{task}_base_count": count,
        },
    )
    candidates = runpy.run_path(str(project / "src/modeling/model_competition.py"))[
        "build_candidates"
    ](task)
    params = candidates[f"{model}_8"]["modeling"]["base_model"]["params"]
    assert len(candidates) == 8
    assert len(params["base_estimators"]) == len(set(params["base_estimators"])) == int(count)
    assert params["base_estimators"][-1] == "lightgbm"
    assert params["weights"] == [1] * int(count)


@pytest.mark.parametrize("task", ["regression", "classification"])
@pytest.mark.parametrize("strategy", ["random", "grid", "halving_random", "halving_grid", "optuna"])
def test_cli_builds_selected_recipes_and_common_search(tmp_path, task, strategy):
    """Real CLI output must freeze selected models, strategy controls and ensemble composition."""
    from test_databricks_bundle_generation import CLI, PROFILE, _generate_project

    from skyulf.integrations.databricks.local_cv import LocalCVSpec
    from skyulf.integrations.databricks.local_search import prepare_search_pipeline
    from skyulf.integrations.databricks.project import load_project_workflow

    if not CLI or not PROFILE:
        pytest.skip("Set SKYULF_BUNDLE_CLI_TEST_PROFILE for real CLI generation.")
    classifier = task == "classification"
    first = "logistic_regression" if classifier else "ridge_regression"
    ensemble = "voting_classifier" if classifier else "stacking_regressor"
    overrides = {f"competition_{task}_model_1": first, f"competition_{task}_model_2": ensemble}
    project = _generate_project(
        tmp_path,
        training_layout="model_competition",
        task=task,
        competition_candidate_count="2",
        competition_preprocessing_recipe="example_imputer",
        competition_max_trials="2000",
        search_strategy=strategy,
        search_n_trials="3",
        search_settings="custom",
        search_timeout="12",
        search_factor="2",
        search_max_candidates="2000",
        search_sampler="random",
        search_pruning="false",
        search_pruner="none",
        competition_ensemble_2_classification_base_count="2",
        competition_ensemble_2_voting="hard",
        competition_ensemble_2_cv="4",
        competition_ensemble_2_regression_final="linear_regression",
        **overrides,
    )
    config = json.loads((project / "config/workflow.json").read_text())
    assert config["competition_max_trials"] == 2000
    loaded = load_project_workflow(config, project / "src/features")
    candidates = loaded["competition"]["candidates"]
    assert list(candidates) == [f"{first}_1", f"{ensemble}_2"]
    for entry in candidates.values():
        pipeline = entry["pipeline"]
        assert pipeline["preprocessing"][0]["transformer"] == "SimpleImputer"
        model = pipeline["modeling"]
        assert model["strategy"] == strategy and model["n_trials"] == 3
        assert model["search_space"]
        assert model["metric"] == config["metric"].removeprefix("heldout_")
        if strategy == "optuna":
            assert model["timeout"] == 12 and model["strategy_params"]["sampler"] == "random"
        if strategy.startswith("halving"):
            assert model["strategy_params"]["factor"] == 2
        if strategy in {"random", "optuna", "halving_random"}:
            effective = prepare_search_pipeline(
                pipeline,
                LocalCVSpec.from_workflow(config),
                target_column="target",
                event_column=None,
            )
            assert effective["modeling"]["search_space"]
    params = candidates[f"{ensemble}_2"]["pipeline"]["modeling"]["base_model"]["params"]
    assert params["tune_base_models"] is True
    if classifier:
        assert params["voting"] == "hard"
    else:
        assert params["cv"] == 4 and params["final_estimator"] == "linear_regression"


@pytest.mark.parametrize("task", ["classification", "regression"])
@pytest.mark.parametrize("engine", ["pandas", "polars"])
def test_generated_candidates_fit_and_select_without_python_edits(tmp_path, task, engine):
    """Setup-selected standalone and ensemble recipes must actually train and yield comparable scores."""
    import numpy as np
    import pandas as pd
    import polars as pl
    from test_databricks_bundle_generation import CLI, PROFILE, _generate_project

    from skyulf.data.dataset import SplitDataset
    from skyulf.integrations.databricks.competition_evaluation import evaluate_competition_candidate
    from skyulf.integrations.databricks.local_batch import fit_local_workflow
    from skyulf.integrations.databricks.local_competition import choose_winner
    from skyulf.integrations.databricks.local_cv import LocalCVSpec
    from skyulf.integrations.databricks.local_search import prepare_search_pipeline
    from skyulf.integrations.databricks.local_workflow import resolve_target_config
    from skyulf.integrations.databricks.project import load_project_workflow
    from skyulf.integrations.databricks.workflow_config import validate_workflow_config

    if not CLI or not PROFILE:
        pytest.skip("Set SKYULF_BUNDLE_CLI_TEST_PROFILE for real CLI generation.")
    classification = task == "classification"
    first = "logistic_regression" if classification else "ridge_regression"
    second = "voting_classifier" if classification else "voting_regressor"
    third = "stacking_classifier" if classification else "stacking_regressor"
    project = _generate_project(
        tmp_path,
        training_layout="model_competition",
        task=task,
        engine=engine,
        competition_candidate_count="3",
        search_n_trials="1",
        cv_folds="2",
        cv_type="stratified_k_fold" if classification else "k_fold",
        competition_ensemble_2_classification_base_2="decision_tree",
        competition_ensemble_3_classification_base_2="decision_tree",
        competition_ensemble_3_cv="2",
        **{
            f"competition_{task}_model_1": first,
            f"competition_{task}_model_2": second,
            f"competition_{task}_model_3": third,
        },
    )
    config = load_project_workflow(
        json.loads((project / "config/workflow.json").read_text()), project / "src/features"
    )
    resolved = resolve_target_config(
        config,
        {
            "catalog": "workspace",
            "input_schema": "inputs",
            "output_schema": "outputs",
            "metadata_schema": "models",
            "resource_suffix": "",
        },
    )
    validate_workflow_config(resolved, action="train")
    cv = LocalCVSpec.from_workflow(config)
    x = np.linspace(-2, 2, 60)
    frame = pd.DataFrame(
        {"feature_value": x, "target": (x > 0).astype(int) if classification else 2 * x + 1}
    )
    if engine == "polars":
        frame = pl.from_pandas(frame)
    rows = []
    for name, item in config["competition"]["candidates"].items():
        effective = prepare_search_pipeline(
            item["pipeline"], cv, target_column="target", event_column=None
        )
        artifact = fit_local_workflow(
            effective,
            SplitDataset(train=frame, test=frame.head(0)),
            target_column="target",
            artifact_path=tmp_path / name,
            max_rows=100,
            max_bytes=1_000_000,
        )
        row = evaluate_competition_candidate(
            frame,
            artifact,
            cv,
            target_column="target",
            metric=config["metric"],
            max_rows=100,
            max_bytes=1_000_000,
        )
        rows.append({**row, "candidate": name})
    selection = choose_winner(rows, set(config["competition"]["candidates"]))
    assert selection["candidate_count"] == 3
    assert selection["winner"] in {f"{first}_1", f"{second}_2", f"{third}_3"}
    assert len({row["fold_membership_sha256"] for row in rows}) == 1


@pytest.mark.parametrize("task", ["classification", "regression"])
def test_cli_keeps_each_ensemble_composition_and_controls(tmp_path, task):
    """Two voting and two stacking candidates must not inherit each other's answers."""
    from test_databricks_bundle_generation import CLI, PROFILE, _generate_project

    from skyulf.integrations.databricks.local_cv import LocalCVSpec
    from skyulf.integrations.databricks.local_search import prepare_search_pipeline
    from skyulf.integrations.databricks.project import load_project_workflow

    if not CLI or not PROFILE:
        pytest.skip("Set SKYULF_BUNDLE_CLI_TEST_PROFILE for real CLI generation.")
    classifier = task == "classification"
    suffix = "classifier" if classifier else "regressor"
    first = "logistic_regression" if classifier else "linear_regression"
    second = "gaussian_nb" if classifier else "lasso"
    overrides = {}
    for slot, family in enumerate(["voting", "voting", "stacking", "stacking"], 1):
        overrides[f"competition_{task}_model_{slot}"] = f"{family}_{suffix}"
        overrides[f"competition_ensemble_{slot}_{task}_base_1"] = first if slot % 2 else second
        overrides[f"competition_ensemble_{slot}_{task}_base_2"] = "decision_tree"
    project = _generate_project(
        tmp_path,
        training_layout="model_competition",
        task=task,
        competition_candidate_count="4",
        competition_ensemble_1_classification_weight_1="2.5",
        competition_ensemble_1_regression_weight_1="2.5",
        competition_ensemble_1_voting="hard",
        competition_ensemble_2_voting="soft",
        competition_ensemble_2_calibrate="true",
        competition_ensemble_2_calibration_cv="2",
        competition_ensemble_3_cv="2",
        competition_ensemble_4_cv="4",
        competition_ensemble_4_passthrough="true",
        competition_ensemble_4_classification_final="decision_tree",
        competition_ensemble_4_regression_final="decision_tree",
        **overrides,
    )
    loaded = load_project_workflow(
        json.loads((project / "config/workflow.json").read_text()), project / "src/features"
    )
    candidates = loaded["competition"]["candidates"]
    params = [
        entry["pipeline"]["modeling"]["base_model"]["params"] for entry in candidates.values()
    ]
    assert params[0]["base_estimators"] == params[2]["base_estimators"] == [first, "decision_tree"]
    assert params[1]["base_estimators"] == params[3]["base_estimators"] == [second, "decision_tree"]
    assert params[0]["weights"] == [2.5, 1] and params[1]["weights"] == [1, 1]
    assert params[2]["cv"] == 2 and params[3]["cv"] == 4
    assert params[2]["passthrough"] is False and params[3]["passthrough"] is True
    assert params[3]["final_estimator"] == "decision_tree"
    if classifier:
        assert params[0]["voting"] == "hard" and params[1]["voting"] == "soft"
        assert params[0]["calibrate_base_models"] is False
        assert params[1]["calibrate_base_models"] is True
        assert params[1]["calibration_cv"] == 2
    for index, entry in enumerate(candidates.values()):
        effective = prepare_search_pipeline(
            entry["pipeline"],
            LocalCVSpec(enabled=True, folds=2),
            target_column="target",
            event_column=None,
        )
        space = effective["modeling"]["search_space"]
        selected = first if index % 2 == 0 else second
        unselected = second if index % 2 == 0 else first
        assert any(key.startswith(f"{selected}__") for key in space)
        assert not any(key.startswith(f"{unselected}__") for key in space)
