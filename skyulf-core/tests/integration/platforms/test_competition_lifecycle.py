"""A single target competition registers only its verified complete winner."""

import importlib.metadata
import json
import shutil
import subprocess
import sys
from copy import deepcopy
from pathlib import Path
from typing import Any

import pytest
from test_databricks_lifecycle_tasks import _call, staged  # noqa: F401 - shared registry fixture


def _competition(config, engine="pandas"):
    """Build deliberately different candidates with a shared CV and heldout policy."""
    config.update(
        engine=engine,
        task="regression",
        training_layout="model_competition",
        cv_enabled=True,
        cv_folds=2,
        competition_max_trials=100,
        competition_max_candidates=8,
    )
    config["competition"] = {
        "candidates": {
            "weak": {
                "pipeline": {
                    "preprocessing": [],
                    "modeling": {"type": "ridge_regression", "params": {"alpha": 1e6}},
                }
            },
            "strong": {"pipeline": deepcopy(config["pipeline"])},
        }
    }
    return config


@pytest.mark.parametrize("engine", ["pandas", "polars"])
def test_only_winner_registered_after_shared_cv(staged, engine):
    """The heldout population cannot select losers or create their registry versions."""
    from skyulf.integrations.databricks.jobs.shared.job_output import render_lifecycle_output

    _, client, config, _, _ = staged
    _competition(config, engine)
    prepared = _call(staged, "initialize", config=config, action="train", experiment_name="staged")
    loaded = _call(staged, "load_data", prepared.reference)
    split = _call(staged, "prepare_dataset", loaded.reference)
    trained = _call(staged, "train", split.reference)
    assert trained.output["competition"]["winner"] == "strong"
    assert not client.search_registered_models()
    with pytest.raises(ValueError, match="fresh run"):
        _call(staged, "train", split.reference)
    selected = _call(staged, "select_best_model", trained.reference)
    assert selected.output["candidate_count"] == 2
    assert "weak" in render_lifecycle_output("select_best_model", selected.output)
    registered = _call(staged, "evaluate_register", selected.reference)
    with pytest.raises(ValueError, match="fresh run"):
        _call(staged, "evaluate_register", selected.reference)
    _call(staged, "compare", registered.reference)
    decided = _call(staged, "model_decision", prepared.reference)
    assert decided.output["alias_change"]["new_version"] == "1"
    versions = client.search_model_versions(f"name='{config['model_name']}'")
    assert len(versions) == 1
    parent = prepared.reference["run_id"]
    runs = client.search_runs(
        [client.get_experiment_by_name("staged").experiment_id],
        filter_string=f"tags.`mlflow.parentRunId` = '{parent}'",
    )
    assert len(runs) == 2
    assert all(run.info.status == "FINISHED" for run in runs)
    assert all(not any(k.startswith("heldout_") for k in run.data.metrics) for run in runs)
    receipt = json.loads(
        Path(client.download_artifacts(parent, "competition/selection.json")).read_text()
    )
    assert receipt["winner"] == "strong"


def test_failed_candidate_prevents_registration(staged, monkeypatch):
    """A partial competition must never silently register its successful candidate."""
    from skyulf.integrations.databricks.training.fitting import local_retraining

    _, client, config, _, _ = staged
    _competition(config)
    fit = local_retraining.fit_candidate

    def fail_one(*args, **kwargs):
        """Fail the weak fit after the strong candidate can finish."""
        if kwargs["pipeline_config"]["modeling"].get("params", {}).get("alpha") == 1e6:
            raise ValueError("deliberate candidate failure")
        return fit(*args, **kwargs)

    monkeypatch.setattr(local_retraining, "fit_candidate", fail_one)
    prepared = _call(staged, "prepare", config=config, action="train", experiment_name="staged")
    with pytest.raises(ValueError, match="deliberate candidate failure"):
        _call(staged, "train", prepared.reference)
    assert not client.search_registered_models()


def test_shap_competition_preserves_winner_and_child_reports(staged):
    """Opt-in competition explanations must survive winner adoption and notebook rendering."""
    pytest.importorskip("shap")
    pytest.importorskip("matplotlib")
    _, client, config, _, _ = staged
    _competition(config)
    for candidate in config["competition"]["candidates"].values():
        candidate["pipeline"]["explainability"] = {
            "method": "shap",
            "max_samples": 3,
            "max_display_samples": 1,
        }
    prepared = _call(staged, "prepare", config=config, action="train", experiment_name="shap")
    trained = _call(staged, "train", prepared.reference)
    assert trained.output["explanations"]["status"] == "completed"
    parent = prepared.reference["run_id"]
    evidence = json.loads(Path(client.download_artifacts(parent, "explanations.json")).read_text())
    winner = next(
        row for row in trained.output["competition"]["leaderboard"] if row["candidate"] == "strong"
    )
    assert evidence["run_id"] == winner["run_id"]
    for row in trained.output["competition"]["leaderboard"]:
        report = Path(client.download_artifacts(row["run_id"], "explanations.html")).read_text()
        assert "data:image/png;base64," in report


def test_cv_required_before_reading_source(staged, monkeypatch):
    """Competition cannot rank candidates by the independent final holdout."""
    from skyulf.integrations.databricks.lifecycle import local_workflow

    _, client, config, _, _ = staged
    _competition(config)
    config["cv_enabled"] = False
    monkeypatch.setattr(
        local_workflow, "prepare_training", lambda *a, **kw: pytest.fail("remote preparation")
    )
    with pytest.raises(ValueError, match="cv_enabled"):
        _call(staged, "prepare", config=config, action="train", experiment_name="staged")
    assert not client.search_registered_models()


def _mixed_candidates(config, frame, engine, task, strategy, nested):
    """Keep three real model families small while exercising each search implementation."""
    _competition(config, engine)
    for index in range(len(frame), 80):
        frame.loc[index] = [index, index, 2 * index]
    classification = task == "classification"
    if classification:
        frame["target"] = (frame["x"] >= 40).astype(int)
    fixed = (
        {"type": "logistic_regression", "params": {"C": 1e-9}}
        if classification
        else {"type": "ridge_regression", "params": {"alpha": 1e6}}
    )
    config.update(
        task=task,
        metric="heldout_accuracy" if classification else "heldout_rmse",
        quality_threshold=0.0 if classification else 1.0,
        cv_type="nested_cv" if nested else "k_fold",
        cv_inner_folds=2 if nested else None,
        cv_nested_type="stratified_k_fold" if nested and classification else "auto",
        competition_max_trials=40,
        promotion_policy="manual_approval",
    )
    tuner = {
        "type": "hyperparameter_tuner",
        "base_model": {"type": "logistic_regression" if classification else "ridge_regression"},
        "strategy": strategy,
        "n_trials": 2,
        "search_space": {"C": [0.1, 1.0]} if classification else {"alpha": [0.01, 0.1]},
    }
    if strategy.startswith("halving"):
        tuner["strategy_params"] = {
            "factor": 2,
            "min_resources": 8,
            "max_resources": 16,
        }
    ensemble_params: dict[str, Any] = {
        "base_estimators": (
            ["logistic_regression", "gaussian_nb"]
            if classification
            else ["linear_regression", "ridge"]
        ),
    }
    if not classification:
        ensemble_params["base_estimator_params"] = {"ridge": {"alpha": 100.0}}
    ensemble = {
        "type": "voting_classifier" if classification else "voting_regressor",
        "params": ensemble_params,
    }
    config["pipeline"] = {"preprocessing": [], "modeling": fixed}
    config["competition"]["candidates"] = {
        name: {"pipeline": {"preprocessing": [], "modeling": model}}
        for name, model in {"fixed": fixed, "tuned": tuner, "voting": ensemble}.items()
    }


@pytest.mark.parametrize(
    ("engine", "task", "strategy", "nested"),
    [
        ("pandas", "regression", "grid", False),
        ("polars", "classification", "random", True),
        ("pandas", "classification", "halving_grid", False),
        ("polars", "regression", "halving_random", True),
        ("pandas", "regression", "optuna", False),
    ],
)
def test_mixed_model_families_and_search_strategies_register_one_winner(
    staged, engine, task, strategy, nested
):
    """Every search family must compete with fixed and voting models through real registry I/O."""
    if strategy == "optuna":
        pytest.importorskip("optuna_integration.sklearn")
    _, client, config, _, frame = staged
    _mixed_candidates(config, frame, engine, task, strategy, nested)
    prepared = _call(staged, "prepare", config=config, action="train", experiment_name="staged")
    trained = _call(staged, "train", prepared.reference)
    selection = trained.output["competition"]
    if task == "regression":
        assert selection["winner"] == "tuned"
    rows = {row["candidate"]: row for row in selection["leaderboard"]}
    assert set(rows) == {"fixed", "tuned", "voting"}
    assert len({row["fold_membership_sha256"] for row in rows.values()}) == 1
    assert all(len(row["fold_scores"]) == 2 for row in rows.values())
    assert rows["tuned"]["evaluation_mode"] == ("nested_cv" if nested else "post_selection_cv")
    assert not client.search_registered_models()
    registered = _call(staged, "evaluate_register", trained.reference)
    versions = client.search_model_versions(f"name='{config['model_name']}'")
    assert len(versions) == 1
    assert registered.output["model_digest"] == rows[selection["winner"]]["model_digest"]
    for row in rows.values():
        run = client.get_run(row["run_id"])
        assert run.info.status == "FINISHED"
        assert not any(name.startswith("heldout_") for name in run.data.metrics)


def test_winner_quality_failure_never_evaluates_runner_up(staged, monkeypatch):
    """A failed winner gate must leave the champion untouched without trying another candidate."""
    from skyulf.integrations.databricks.training.fitting import local_retraining

    _, client, config, _, frame = staged
    frame["target"] += [0, 2] * 10
    _competition(config)
    config["quality_threshold"] = 0.0
    evaluated = []
    evaluate = local_retraining.evaluate_candidate

    def record_evaluation(artifact, *args, **kwargs):
        """Record real heldout evaluations without changing their computed metrics."""
        evaluated.append(artifact.manifest.pipeline_sha256)
        return evaluate(artifact, *args, **kwargs)

    monkeypatch.setattr(local_retraining, "evaluate_candidate", record_evaluation)
    prepared = _call(staged, "prepare", config=config, action="train", experiment_name="staged")
    trained = _call(staged, "train", prepared.reference)
    assert evaluated == []
    registered = _call(staged, "evaluate_register", trained.reference)
    decision = _call(staged, "compare_decide", registered.reference)
    assert decision.output["alias_change"] is None
    assert evaluated and set(evaluated) == {trained.output["model_digest"]}
    assert len(client.search_model_versions(f"name='{config['model_name']}'")) == 1
    assert "champion" not in client.get_registered_model(config["model_name"]).aliases


def test_modified_selection_receipt_blocks_registration(staged):
    """Saved winner evidence cannot be edited between training and registry publication."""
    _, client, config, _, _ = staged
    _competition(config)
    prepared = _call(staged, "prepare", config=config, action="train", experiment_name="staged")
    trained = _call(staged, "train", prepared.reference)
    run_id = prepared.reference["run_id"]
    path = "competition/selection.json"
    payload = json.loads(Path(client.download_artifacts(run_id, path)).read_text())
    payload["winner"] = "weak"
    client.log_dict(run_id, payload, path)
    with pytest.raises(ValueError, match="selection|receipt"):
        _call(staged, "evaluate_register", trained.reference)
    assert not client.search_registered_models()
    assert "skyulf.lifecycle.registration_intent" not in client.get_run(run_id).data.tags


def test_winner_scores_with_captured_custom_recipe_after_project_edit(staged, tmp_path):
    """Winner registration and scoring must restore its selected package without editable code."""
    import polars as pl

    from skyulf.inference.local_pipeline import predict_local_pipeline
    from skyulf.inference.project_scoring import run_project_scoring
    from skyulf.integrations.databricks.projects.project import load_project_workflow
    from skyulf.integrations.mlflow.registration.registry import load_run_local_pipeline

    _, client, config, _, frame = staged
    _competition(config, "polars")
    frame["x"] = frame["x"].astype(float)
    frame.loc[0, "x"] = float("nan")
    frame["category"] = "known"
    config["input_columns"] = ["x", "category"]
    features = tmp_path / "features"
    features.mkdir()
    template = (
        Path(__file__).resolve().parents[3]
        / "templates/databricks/template/{{.project_name}}/src/features/preprocessing.py"
    )
    shutil.copyfile(template, features / "custom.py")
    shutil.copyfile(template.with_name("pre_split.py"), features / "shared_filter.py")
    pin = f"numpy=={importlib.metadata.version('numpy')}"
    features.joinpath("requirements.txt").write_text(pin, encoding="utf-8")
    features.joinpath("assets.json").write_text('["filter.json"]', encoding="utf-8")
    features.joinpath("filter.json").write_text('{"min_present": 1}', encoding="utf-8")
    features.joinpath("__init__.py").write_text(
        "import json\n"
        "from skyulf.inference.project_package import read_project_asset\n"
        "from .custom import frequency_encoding\n"
        "from .shared_filter import minimum_completeness\n"
        "FILTER = json.loads(read_project_asset(__package__, 'filter.json'))\n"
        "def build_preprocessing(recipe='default'):\n"
        "    steps = [frequency_encoding(['category'])]\n"
        "    if recipe == 'scaled':\n"
        "        steps.append({'name': 'scale', 'transformer': 'StandardScaler', "
        "'params': {'columns': ['x']}})\n"
        "    return steps\n"
        "def build_pre_split_steps():\n"
        "    return [minimum_completeness(['x'], min_present=FILTER['min_present'])]\n"
        "def build_scoring():\n"
        "    return {'reuse_pre_split': True, 'skip_target_steps': False}\n",
        encoding="utf-8",
    )
    modeling = tmp_path / "modeling"
    modeling.mkdir()
    modeling.joinpath("candidates.py").write_text(
        "def build_candidates(task):\n"
        "    return {'strong': {'modeling': {'type': 'linear_regression'}, "
        "'preprocessing_recipe': 'scaled'}, 'weak': {'modeling': "
        "{'type': 'ridge_regression', 'params': {'alpha': 1000000}}}}\n",
        encoding="utf-8",
    )
    resolved = load_project_workflow(config, features)
    prepared = _call(staged, "prepare", config=resolved, action="train", experiment_name="staged")
    trained = _call(staged, "train", prepared.reference)
    assert trained.output["competition"]["winner"] == "strong"
    features.joinpath("__init__.py").write_text("raise RuntimeError('edited')\n", encoding="utf-8")
    modeling.joinpath("candidates.py").write_text(
        "raise RuntimeError('edited')\n", encoding="utf-8"
    )
    registered = _register_in_fresh_process(tmp_path, config, frame, trained.reference)
    artifact = load_run_local_pipeline(
        trained.output["model_uri"],
        digest=registered["model_digest"],
        tracking_uri=config["tracking_uri"],
    )
    pipeline: dict[str, Any] = dict(artifact.pipeline.config)
    scores = run_project_scoring(
        pl.DataFrame({"x": [20.0, 21.0, None], "category": ["known", "unseen", "known"]}),
        lambda rows: predict_local_pipeline(rows, artifact),
        source=pipeline["project_python_source"],
        config=pipeline["project_scoring"],
        row_keys=[],
        prediction_dtypes={"prediction": "float64"},
    )
    assert scores["prediction"].iloc[:2].tolist() == pytest.approx([40.0, 42.0])
    assert scores["scoring_status"].tolist() == ["predicted", "predicted", "excluded"]
    assert scores["exclusion_reason"].iloc[2] == "pre_split:minimum_completeness"
    assert len(artifact.pipeline.config["preprocessing"]) == 2
    assert artifact.manifest.project_requirements == (pin,)
    assert len(client.search_model_versions(f"name='{config['model_name']}'")) == 1


def _register_in_fresh_process(tmp_path, config, frame, reference):
    """Replay saved custom filters in an empty Python registry before publishing the winner."""
    data = tmp_path / "rows.json"
    frame.to_json(data, orient="table")
    saved = tmp_path / "reference.json"
    saved.write_text(json.dumps(reference), encoding="utf-8")
    code = (
        "import json, sys, pandas as pd\n"
        "from pathlib import Path\n"
        "from skyulf.integrations.databricks.training.fitting import local_retraining\n"
        "from skyulf.integrations.databricks.jobs.lifecycle.lifecycle_tasks import LifecycleContext, run_lifecycle_phase\n"
        "frame=pd.read_json(sys.argv[1], orient='table')\n"
        "local_retraining.read_training_snapshot=lambda spark,spec: frame.copy()\n"
        "reference=json.loads(Path(sys.argv[2]).read_text())\n"
        "result=run_lifecycle_phase(None, phase='evaluate_register', "
        "context=LifecycleContext(job_id='10',job_run_id='20'), "
        "reference=reference, tracking_uri=sys.argv[3])\n"
        "print('REGISTERED:' + json.dumps(result.output))\n"
    )
    replay = subprocess.run(
        [sys.executable, "-c", code, str(data), str(saved), config["tracking_uri"]],
        cwd=tmp_path,
        capture_output=True,
        text=True,
        timeout=60,
    )
    assert replay.returncode == 0, replay.stderr
    output = next(line for line in replay.stdout.splitlines() if line.startswith("REGISTERED:"))
    return json.loads(output.removeprefix("REGISTERED:"))
