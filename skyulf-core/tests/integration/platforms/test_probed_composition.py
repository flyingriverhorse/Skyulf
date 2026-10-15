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
    assert json.loads(artifact.manifest.composition_config_json)["partition_probe"]["version"] == 1
    unsealed, _ = _set(tmp_path, name="plain")
    assert unsealed.manifest.set_sha256 != artifact.manifest.set_sha256
