"""Execute the Bundle tuning guide's literal YAML through admission and saved fits."""

import json
import math
import re
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

pytest.importorskip("yaml")
pytest.importorskip("polars")

from skyulf.data.dataset import SplitDataset
from skyulf.inference.fitted_pipeline import load_pipeline
from skyulf.integrations.databricks.projects.yaml_config import (
    TUNING_FIELDS,
    read_training_config,
)
from skyulf.integrations.databricks.projects.yaml_models import model_entries, model_pipeline
from skyulf.integrations.databricks.scoring.batch.frame_batch import fit_workflow
from skyulf.integrations.databricks.training.competition.competition import (
    validate_competition_budget,
)
from skyulf.integrations.databricks.training.tuning.cv import CVSpec
from skyulf.integrations.databricks.training.tuning.search import prepare_search_pipeline
from skyulf.integrations.databricks.training.tuning.search_results import (
    tuning_evidence,
    validate_search_membership,
)

ROOT = Path(__file__).resolve().parents[4]
GUIDE = ROOT / "docs/user_guide/databricks_tuning.md"
ENSEMBLE_GUIDE = ROOT / "docs/user_guide/databricks_ensembles.md"
ENSEMBLE_EXAMPLES = {"voting", "stacking", "calibrated"}
SCHEMA = ROOT / "skyulf-core/templates/databricks/schema/tuning.json"
STRATEGIES = ["grid", "random", "halving_grid", "halving_random", "optuna"]


def _document(guide: Path = GUIDE) -> str:
    """Fail explicitly when a public guide is missing from a checkout."""
    assert guide.is_file(), f"The Bundle guide {guide.name} must exist."
    return guide.read_text(encoding="utf-8")


def _example(name: str) -> str:
    """Read one marked executable block without duplicating its settings in tests."""
    guide = ENSEMBLE_GUIDE if name in ENSEMBLE_EXAMPLES else GUIDE
    matches = re.findall(
        rf"<!-- tuning-example:{name} -->\s*```yaml\n(.*?)```", _document(guide), re.DOTALL
    )
    assert len(matches) == 1, f"Expected exactly one executable {name} example."
    return matches[0]


def _load_example(tmp_path: Path, name: str) -> dict:
    """Append the advertised model block to the common header using the real YAML reader."""
    (tmp_path / "training.yml").write_text(
        _example("header") + "\n" + _example(name), encoding="utf-8"
    )
    document = read_training_config(tmp_path)
    assert document is not None
    return document


def _effective(document: dict) -> tuple[dict, CVSpec]:
    """Resolve the same model declarations and authoritative CV contract as a Bundle."""
    entry = next(iter(model_entries(document).values()))
    cv = CVSpec.from_workflow(entry)
    pipeline = prepare_search_pipeline(
        model_pipeline(entry), cv, target_column=entry["target_column"], event_column=None
    )
    return pipeline, cv


def test_strategy_cards_and_field_table_cover_the_shipped_contract():
    """New wizard strategies or YAML controls must receive a discoverable guide card."""
    document = _document()
    schema = json.loads(SCHEMA.read_text(encoding="utf-8"))
    cards = re.findall(r'<a id="strategy-([a-z_]+)"></a>', document)
    assert sorted(cards) == sorted(schema["search_strategy"]["enum"])
    table = document.split("<!-- tuning-fields:start -->")[1].split("<!-- tuning-fields:end -->")[0]
    assert set(re.findall(r"^\| `([a-z_]+)` \|", table, re.MULTILINE)) == TUNING_FIELDS
    assert set(cards) == set(STRATEGIES)


@pytest.mark.parametrize("strategy", STRATEGIES)
def test_strategy_example_fits_saves_reloads_and_predicts(tmp_path, strategy):
    """Copying a strategy card must yield a usable saved model with bounded selection evidence."""
    if strategy == "optuna":
        pytest.importorskip("optuna")
        pytest.importorskip("optuna_integration.sklearn")
    document = _load_example(tmp_path, strategy)
    effective, cv = _effective(document)
    rows = pd.DataFrame(
        {"income": [float(i) for i in range(40)], "revenue": [2.0 * i + 5 for i in range(40)]}
    )
    training, holdout = rows.iloc[:32], rows.iloc[32:]
    validate_search_membership(training, effective, cv, target_column="revenue", event_column=None)
    location = tmp_path / "artifact"
    artifact = fit_workflow(
        effective,
        SplitDataset(train=training, test=holdout),
        target_column="revenue",
        artifact_path=location,
        max_rows=40,
        max_bytes=100_000,
    )
    evidence = tuning_evidence(artifact)
    assert evidence is not None and evidence["status"] == "completed"
    assert math.isfinite(evidence["best_score"])
    assert evidence["modeling"]["strategy"] == strategy
    assert evidence["best_params"]["alpha"] in [0.1, 1.0, 10.0, 100.0]
    expected_trials = 7 if strategy.startswith("halving") else 4
    assert evidence["n_trials"] == expected_trials
    query = pd.DataFrame({"income": [4.0, 18.0]})
    predicted = artifact.pipeline.predict(query)
    restored = load_pipeline(location)
    np.testing.assert_allclose(restored.pipeline.predict(query), predicted)
    assert np.isfinite(predicted).all() and np.asarray(predicted).shape == (2,)


@pytest.mark.parametrize("name", ["voting", "stacking", "calibrated"])
def test_ensemble_example_resolves_selected_members_and_nested_axes(tmp_path, name):
    """Member overrides and calibrated prefixes must survive admission, fitting and persistence."""
    document = _load_example(tmp_path, name)
    effective, _ = _effective(document)
    modeling = effective["modeling"]
    space = modeling["search_space"]
    if name == "calibrated":
        assert space["logistic_regression__estimator__C"] == [0.1, 1.0]
        assert "logistic_regression__C" not in space
    else:
        assert space["ridge__alpha"] == [0.1, 1.0]
        assert space["lasso__alpha"] == [0.01]
    selected = modeling["base_model"]["params"]
    if name == "stacking":
        assert selected["final_estimator"] == "ridge"
        assert space["final_estimator__alpha"] == [0.1, 1.0]
    assert len(selected["base_estimators"]) == 2
    target = "churn" if name == "calibrated" else "revenue"
    values = (
        [i % 2 for i in range(40)] if name == "calibrated" else [2.0 * i + 5 for i in range(40)]
    )
    rows = pd.DataFrame({"income": [float(i) for i in range(40)], target: values})
    artifact = fit_workflow(
        effective,
        SplitDataset(train=rows.iloc[:32], test=rows.iloc[32:]),
        target_column=target,
        artifact_path=tmp_path / "ensemble",
        max_rows=40,
        max_bytes=100_000,
    )
    evidence = tuning_evidence(artifact)
    assert evidence is not None and math.isfinite(evidence["best_score"])
    predicted = artifact.pipeline.predict(pd.DataFrame({"income": [2.0, 9.0]}))
    assert np.isfinite(predicted).all() and np.asarray(predicted).shape == (2,)


@pytest.mark.parametrize(
    "changes,message",
    [
        ({"n_trials": 1001}, "n_trials must be an integer from 1 to 1000"),
        ({"max_candidates": 3}, "4 candidates, above max_candidates=3"),
        ({"strategy_params": {"factor": 2}}, "unsupported for grid and random"),
        ({"timeout": 5}, "supported only by optuna"),
    ],
)
def test_documented_search_limits_reject_invalid_edits(tmp_path, changes, message):
    """Changing the copied YAML beyond its admitted strategy limits must fail before fitting."""
    document = _load_example(tmp_path, "grid")
    document["models"]["revenue"]["tuning"].update(changes)
    with pytest.raises(ValueError, match=message):
        _effective(document)


def test_fixed_parameter_cannot_become_a_conflicting_search_axis(tmp_path):
    """An edited model parameter must not silently lose to the card's search choices."""
    document = _load_example(tmp_path, "grid")
    document["models"]["revenue"]["model"]["params"]["alpha"] = 1.0
    with pytest.raises(ValueError, match="Fixed base parameter conflicts with search_space: alpha"):
        _effective(document)


def test_estimator_halving_resource_requires_a_separate_explicit_budget(tmp_path):
    """Resource budgets must not silently overwrite fixed constructor or searched values."""
    document = _load_example(tmp_path, "halving_grid")
    entry = document["models"]["revenue"]
    entry["model"] = {"type": "random_forest_regressor", "params": {"n_estimators": 10}}
    entry["tuning"]["search_space"] = {"max_depth": [2, 4]}
    entry["tuning"]["strategy_params"] = {
        "resource": "n_estimators",
        "min_resources": 10,
        "max_resources": 40,
    }
    with pytest.raises(ValueError, match="cannot also be a fixed or searched parameter"):
        _effective(document)
    entry["model"]["params"] = {}
    effective, _ = _effective(document)
    assert "n_estimators" not in effective["modeling"]["search_space"]
    entry["tuning"]["strategy_params"]["max_resources"] = "auto"
    with pytest.raises(ValueError, match="requires an explicit max_resources integer"):
        _effective(document)


def test_nested_cv_example_saves_independent_outer_evaluation(tmp_path):
    """The documented nested settings must save outer evaluation separately from final search."""
    document = _load_example(tmp_path, "grid")
    document["defaults"].update(cv_type="nested_cv", cv_folds=3, cv_inner_folds=2)
    effective, cv = _effective(document)
    rows = pd.DataFrame(
        {"income": [float(i) for i in range(40)], "revenue": [2.0 * i + 5 for i in range(40)]}
    )
    validate_search_membership(
        rows.iloc[:32], effective, cv, target_column="revenue", event_column=None
    )
    artifact = fit_workflow(
        effective,
        SplitDataset(train=rows.iloc[:32], test=rows.iloc[32:]),
        target_column="revenue",
        artifact_path=tmp_path / "nested",
        max_rows=40,
        max_bytes=100_000,
    )
    evidence = tuning_evidence(artifact)
    assert evidence is not None and evidence["n_trials"] == 4
    outer = evidence["nested_cv"]
    assert outer["outer_folds"] == 3 and outer["inner_folds"] == 2
    assert len(outer["folds"]) == 3 and outer["aggregated_metrics"]


def test_competition_budget_counts_halving_rounds_and_nested_searches(tmp_path):
    """The documented four-candidate halving budget must reserve all rounds and outer searches."""
    document = _load_example(tmp_path, "halving_grid")
    effective, _ = _effective(document)
    config = {
        **document["defaults"],
        "competition": {"candidates": {"ridge": {"pipeline": effective}}},
        "competition_max_trials": 7,
    }
    validate_competition_budget(config)
    config["competition_max_trials"] = 6
    with pytest.raises(ValueError, match="competition_max_trials: 7"):
        validate_competition_budget(config)
    config.update(cv_type="nested_cv", cv_folds=2, competition_max_trials=21)
    config["competition"]["candidates"]["ridge"]["pipeline"] = model_pipeline(
        next(iter(model_entries(document).values()))
    )
    validate_competition_budget(config)
    config["competition_max_trials"] = 20
    with pytest.raises(ValueError, match="competition_max_trials: 21"):
        validate_competition_budget(config)
