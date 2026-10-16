"""Model-set combined function rules run on Spark only with sealed lint and probe evidence."""

import json

import pandas as pd
import pytest

from skyulf.data.dataset import SplitDataset
from skyulf.inference._manifest import ColumnSpec
from skyulf.inference.fitted_pipeline import load_pipeline, save_pipeline
from skyulf.inference.model_set import ComponentReference, load_model_set, save_model_set
from skyulf.inference.model_set_partition_safety import require_partition_safe_model_set
from skyulf.inference.model_set_scoring import predict_model_set
from skyulf.inference.probed_composition import sealed_composition_config
from skyulf.integrations.databricks.projects.project import load_project_workflow
from skyulf.integrations.databricks.scoring.batch.frame_batch import fit_workflow
from skyulf.integrations.databricks.training.shared.preprocessing_checks import seal_partition_probe
from skyulf.pipeline import SkyulfPipeline

SOURCE = '''
import numpy as np
import pandas as pd


def colour(raw, predictions, params):
    """Combine both component predictions for the same row."""
    total = __TOTAL__
    return pd.DataFrame({"colour": np.where(total > params["cut"], "red", "green")},
                        index=raw.index)
'''
TOTAL = {
    "row": 'predictions["a__prediction"] + predictions["b__prediction"]',
    "lint": 'predictions["a__prediction"] - predictions["a__prediction"].mean()',
    # Hidden from the lint: a batch maximum that only matters above the sample's range.
    "threshold": 'predictions["a__prediction"] - 1000.0 * (getattr(predictions['
    '"a__prediction"], "max")() > 40.0)',
}
RULES = {
    "outputs": [
        {
            "name": "colour",
            "version": "1",
            "function": "colour",
            "params": {"cut": 0.0},
            "columns": [{"name": "colour", "dtype": "string"}],
            "required_components": ["a", "b"],
        }
    ]
}
SAMPLE = pd.DataFrame({"id": range(20), "x": [float((7 * i) % 20) for i in range(20)]})


def _set(tmp_path, mode="row", config=None, name="set"):
    """Save two real components and a combined function rule as one model set."""
    data = pd.DataFrame(
        {"x": [0.0, 5.0, 10.0, 15.0, 19.0], "target": [0.0, 10.0, 20.0, 30.0, 38.0]}
    )
    paths = {}
    for branch in ("a", "b"):
        path = tmp_path / branch
        if not path.exists():
            pipeline = SkyulfPipeline(
                {"preprocessing": [], "modeling": {"type": "linear_regression"}}
            )
            pipeline.fit(SplitDataset(train=data, test=data.iloc[:0]), target_column="target")
            save_pipeline(pipeline, path)
        digest = load_pipeline(path).manifest.pipeline_sha256
        paths[branch] = (ComponentReference(name=branch, version="1", digest=digest), path)
    saved = save_model_set(
        tmp_path / name,
        paths,
        record_key_schema=(ColumnSpec(name="id", dtype="int64"),),
        composition_source=SOURCE.replace("__TOTAL__", TOTAL[mode]) + f"# {tmp_path.name}\n",
        composition_config=config or RULES,
    )
    return load_model_set(saved.directory), paths


def _sealed(tmp_path, mode="row"):
    """Probe the unsealed set and save it again with the evidence in its config."""
    unsealed, _ = _set(tmp_path, mode)
    config = sealed_composition_config(unsealed, SAMPLE)
    sealed, _ = _set(tmp_path, mode, config=config, name="sealed")
    return sealed


def test_row_local_combined_rule_is_admitted_with_rule_evidence(tmp_path):
    """A rule reading only the same row's component predictions is certified."""
    artifact = _sealed(tmp_path)
    probe = artifact.manifest.composition_config["partition_probe"]
    assert [rule["name"] for rule in probe["rules"]] == ["colour"]
    assert {c["name"] for c in probe["rules"][0]["checks"]} >= {"lint", "perturb", "empty"}
    certificate = require_partition_safe_model_set(artifact)
    assert certificate["composition_contract"] == "probed_row_local_functions_v1"
    assert certificate["rules_probed"] == ["colour"]
    whole = predict_model_set(SAMPLE, artifact)
    split = pd.concat(
        [predict_model_set(SAMPLE.iloc[i : i + 3], artifact) for i in range(0, 20, 3)]
    )
    pd.testing.assert_frame_equal(whole, split)


def test_combined_rule_reading_a_batch_statistic_fails_lint(tmp_path):
    """The lint names the rule, file line and call before any probe runs."""
    unsealed, _ = _set(tmp_path, "lint")
    with pytest.raises(ValueError, match=r"Combined rule 'colour' reads a batch statistic"):
        sealed_composition_config(unsealed, SAMPLE)


def test_perturbing_one_rows_predictions_catches_hidden_batch_dependence(tmp_path):
    """Changing one row's component predictions must not move other rows' outputs."""
    unsealed, _ = _set(tmp_path, "threshold")
    with pytest.raises(ValueError, match=r"Combined rule 'colour' .*\(perturb: output_mismatch\)"):
        sealed_composition_config(unsealed, SAMPLE)


def test_unprobed_or_tampered_combined_rules_are_rejected(tmp_path):
    """Function composition without matching sealed evidence stays rejected."""
    unsealed, _ = _set(tmp_path)
    with pytest.raises(ValueError, match="Combined rule 'colour'.*No training probe evidence"):
        require_partition_safe_model_set(unsealed)
    config = sealed_composition_config(unsealed, SAMPLE)
    config["outputs"][0]["params"]["cut"] = 5.0
    changed, _ = _set(tmp_path, config=config, name="changed")
    with pytest.raises(ValueError, match="Combined rule 'colour'.*does not match"):
        require_partition_safe_model_set(changed)


def test_sealed_config_keeps_the_rules_and_identity(tmp_path):
    """Evidence is sealed by the set digest, beside unchanged rule declarations."""
    artifact = _sealed(tmp_path)
    config = artifact.manifest.composition_config
    assert config["outputs"] == RULES["outputs"]
    assert json.loads(artifact.manifest.composition_config_json)["partition_probe"]["version"] == 2
    unsealed, _ = _set(tmp_path, name="plain")
    assert unsealed.manifest.set_sha256 != artifact.manifest.set_sha256


@pytest.mark.parametrize(
    "field,value",
    [("version", 1), ("float_tolerance", None), ("float_tolerance", {"rtol": 1e-6, "atol": 1e-12})],
)
def test_old_or_changed_comparison_policy_is_rejected(tmp_path, field, value):
    """Every admission route must require the trusted versioned tolerance policy."""
    artifact = _sealed(tmp_path)
    config = artifact.manifest.composition_config
    assert config["partition_probe"]["version"] == 2
    assert config["partition_probe"]["float_tolerance"] == {"rtol": 1e-9, "atol": 1e-12}
    config["partition_probe"][field] = value
    changed, _ = _set(tmp_path, config=config, name="policy-changed")
    with pytest.raises(ValueError, match="comparison policy"):
        require_partition_safe_model_set(changed)


def _amplified_set(tmp_path, mode="function", weight=1e13):
    """Build an actual probed component whose tiny batch noise reaches composition."""
    project = tmp_path / "features"
    project.mkdir()
    (project / "__init__.py").write_text(
        "from skyulf.preprocessing import column_step\n"
        "def tiny_noise(df):\n"
        "    return df['x'] + 1e-13 * getattr(df['x'], 'mean')()\n"
        "def build_preprocessing(recipe='default'):\n"
        "    return [column_step('noise', tiny_noise, output='x', replace=True, "
        "inference_context='row')]\n",
        encoding="utf-8",
    )
    config = load_project_workflow(
        {"pipeline": {"preprocessing": [], "modeling": {"type": "linear_regression"}}},
        project,
    )["pipeline"]
    config["admit_probed_custom_steps"] = True
    train = pd.DataFrame({"x": [-1.0, 1.0] * 16, "target": [-1.0, 1.0] * 16})
    path = tmp_path / "component"
    fitted = fit_workflow(
        config,
        SplitDataset(train=train, test=train.head(0)),
        target_column="target",
        artifact_path=path,
        max_rows=100,
        max_bytes=100000,
    )
    sample = pd.DataFrame({"id": range(16), "x": [-2.5e-14, 1.0] * 8})
    component = seal_partition_probe(fitted, sample, path, enabled=True)
    source = (
        "import pandas as pd\n"
        "def outputs(raw, predictions, params):\n"
        "    return pd.DataFrame({'flag': (predictions.a__prediction > 0).astype('int64')}, "
        "index=raw.index)\n"
    )
    rules = {
        "outputs": [
            {
                "name": "flag",
                "version": "1",
                "function": "outputs",
                "params": {},
                "columns": [{"name": "flag", "dtype": "int64"}],
                "required_components": ["a"],
            }
        ]
    }
    if mode == "declarative":
        source = "raise AssertionError('inactive composition source executed')"
        rules = {
            "outputs": [
                {
                    "name": "flag",
                    "version": "1",
                    "operation": "weighted_sum",
                    "params": {"inputs": ["a__prediction"], "weights": [weight]},
                    "columns": [{"name": "flag", "dtype": "float64"}],
                    "required_components": ["a"],
                }
            ]
        }
    saved = save_model_set(
        tmp_path / "set",
        {
            "a": (
                ComponentReference(
                    name="a", version="1", digest=component.manifest.pipeline_sha256
                ),
                path,
            )
        },
        record_key_schema=(ColumnSpec(name="id", dtype="int64"),),
        composition_source=source,
        composition_config=rules,
    )
    artifact = load_model_set(saved.directory)
    return artifact, sample


@pytest.mark.parametrize("mode", ["function", "declarative"])
def test_combined_decision_cannot_amplify_tolerated_component_noise(tmp_path, mode):
    """Component float agreement must not hide a changed final business decision."""
    artifact, sample = _amplified_set(tmp_path, mode)
    whole = predict_model_set(sample, artifact)
    singleton = pd.concat([predict_model_set(sample.iloc[i : i + 1], artifact) for i in range(16)])
    assert whole.flag.iloc[0] > 0
    assert singleton.flag.iloc[0] <= 0
    with pytest.raises(
        ValueError, match="Model-set partition probe failed chunks:1: output_mismatch"
    ):
        sealed_composition_config(artifact, sample)


def test_declarative_empirical_components_require_final_proof(tmp_path):
    """Arithmetic on empirically admitted components needs its own final parity evidence."""
    artifact, sample = _amplified_set(tmp_path, "declarative", weight=1.0)
    with pytest.raises(ValueError, match="probe evidence"):
        require_partition_safe_model_set(artifact)
    config = sealed_composition_config(artifact, sample)
    assert config["partition_probe"]["pipeline"]["rows"] == 16
    saved = save_model_set(
        tmp_path / "sealed",
        {
            component.branch: (
                component.reference,
                artifact.directory / "components" / component.branch,
            )
            for component in artifact.manifest.components
        },
        record_key_schema=artifact.manifest.record_key_schema,
        composition_source=(artifact.directory / "composition.py").read_text(),
        composition_config=config,
    )
    assert (
        require_partition_safe_model_set(load_model_set(saved.directory))["composition_contract"]
        == "row_local_operations_v1"
    )


@pytest.mark.parametrize("tamper", ["missing", "checks", "rows", "row_type"])
def test_combined_rules_require_final_output_evidence(tmp_path, tamper):
    """Rule-local evidence alone cannot authorize composed outputs under float tolerance."""
    artifact = _sealed(tmp_path)
    config = artifact.manifest.composition_config
    evidence = config["partition_probe"]
    if tamper == "missing":
        evidence.pop("pipeline", None)
    else:
        detail = evidence.setdefault("pipeline", {"rows": len(SAMPLE), "checks": []})
        if tamper == "checks":
            detail["checks"] = [{"name": "full", "status": "passed"}]
        else:
            detail["rows"] = 19 if tamper == "rows" else 20.0
    changed, _ = _set(tmp_path, config=config, name="missing-final-proof")
    with pytest.raises(ValueError, match="pipeline partition probe"):
        require_partition_safe_model_set(changed)
