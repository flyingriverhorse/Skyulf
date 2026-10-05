"""Company-style fitted preprocessing must remain invariant on independent workers."""

import numpy as np
import pandas as pd
import pytest

pytest.importorskip("xgboost")

from skyulf.core.capabilities import UnsupportedExecutionError
from skyulf.data.dataset import SplitDataset
from skyulf.inference.local_pipeline import (
    load_local_pipeline,
    predict_local_pipeline,
    save_local_pipeline,
)
from skyulf.inference.partition_safety import require_partition_safe_pipeline
from skyulf.pipeline import SkyulfPipeline


@pytest.fixture
def company_artifact(tmp_path, request):
    """Reload a real fitted tree and group/category state as a worker would."""
    data = pd.DataFrame(
        {
            "group": ["A", "B", "A", "B", "A", "B", "A", "B"],
            "region": [2.0, 10.0, 2.0, 10.0, np.nan, 10.0, 2.0, 10.0],
            "amount": [1.0, 9.0, np.nan, 8.0, 3.0, 7.0, 2.0, 10.0],
            "target": [0.1, 0.9, 0.2, 0.8, 0.3, 0.7, 0.2, 0.9],
        }
    )
    mode = getattr(request, "param", "reg:logistic")
    modeling = {
        "type": "xgboost_regressor",
        "params": {
            "n_estimators": 3,
            "max_depth": 2,
            "n_jobs": 1,
            "objective": "reg:squarederror" if mode == "tuned" else mode,
            "random_state": 42,
        },
    }
    if mode == "tuned":
        modeling = {
            "type": "hyperparameter_tuner",
            "base_model": modeling,
            "strategy": "grid",
            "metric": "rmse",
            "search_space": {},
            "cv_folds": 2,
            "n_trials": 1,
            "n_jobs": 1,
        }
    pipeline = SkyulfPipeline(
        {
            "preprocessing": [
                {
                    "name": "clip",
                    "transformer": "ClipValues",
                    "params": {"bounds": {"amount": {"lower": 0, "upper": 10}}},
                },
                {
                    "name": "group_mode",
                    "transformer": "SimpleImputer",
                    "params": {"columns": ["group"], "strategy": "most_frequent"},
                },
                {
                    "name": "means",
                    "transformer": "GroupImputer",
                    "params": {"columns": ["amount"], "group_by": "group", "strategy": "mean"},
                },
                {
                    "name": "modes",
                    "transformer": "GroupImputer",
                    "params": {
                        "columns": ["region"],
                        "group_by": "group",
                        "strategy": "most_frequent",
                    },
                },
                {
                    "name": "encode",
                    "transformer": "OneHotEncoder",
                    "params": {
                        "columns": ["group", "region"],
                        "drop_first": True,
                        "handle_unknown": "ignore",
                        "max_categories": None,
                    },
                },
            ],
            "modeling": modeling,
        }
    )
    pipeline.fit(SplitDataset(train=data.iloc[:6], test=data.iloc[6:]), target_column="target")
    save_local_pipeline(pipeline, tmp_path / "company")
    return load_local_pipeline(tmp_path / "company")


@pytest.mark.parametrize(
    "company_artifact", ["reg:logistic", "reg:squarederror", "tuned"], indirect=True
)
def test_company_pipeline_replays_frozen_state_across_partitions(company_artifact):
    """Unknown groups and nulls must use training state regardless of surrounding rows."""
    certificate = require_partition_safe_pipeline(company_artifact)
    query = pd.DataFrame(
        {
            "group": ["NEW", "B", None, "A", "B"],
            "region": [99.0, np.nan, 2.0, 10.0, np.nan],
            "amount": [np.nan, 200.0, np.nan, -10.0, 5.0],
        },
        index=[90, 1, 15, 3, 4],
    )
    whole = predict_local_pipeline(query, company_artifact)
    singles = pd.concat(
        [predict_local_pipeline(query.iloc[[i]], company_artifact) for i in range(len(query))]
    )
    uneven = pd.concat(
        [
            predict_local_pipeline(query.iloc[:2], company_artifact),
            predict_local_pipeline(query.iloc[2:], company_artifact),
        ]
    )
    pd.testing.assert_frame_equal(whole, singles)
    pd.testing.assert_frame_equal(whole, uneven)
    assert np.isfinite(whole["prediction"]).all()
    assert require_partition_safe_pipeline(company_artifact) == certificate


def test_company_empty_and_null_partitions_preserve_features(company_artifact):
    """Empty and all-null partitions must reuse the fitted feature schema and fills."""
    require_partition_safe_pipeline(company_artifact)
    query = pd.DataFrame(
        {"group": [None, None], "region": [np.nan, np.nan], "amount": [np.nan, np.nan]}
    )
    engineer = company_artifact.pipeline.feature_engineer
    empty = engineer.transform(query.iloc[:0], preserve_rows=True)
    populated = engineer.transform(query, preserve_rows=True)
    assert empty.columns.tolist() == populated.columns.tolist()
    assert populated.notna().all().all()
    assert np.isfinite(predict_local_pipeline(query, company_artifact)["prediction"]).all()


@pytest.mark.parametrize("change", ["clip", "group", "mode", "encoder", "callback", "dart"])
def test_company_admission_rejects_changed_serving_contract(company_artifact, change):
    """State/config drift and custom estimator execution must fail before publication."""
    records = company_artifact.pipeline.feature_engineer.fitted_steps
    if change == "clip":
        records[0]["artifact"]["bounds"]["amount"]["upper"] = 500
    elif change == "group":
        records[2]["artifact"]["group_by"] = "region"
    elif change == "mode":
        records[1]["artifact"]["strategy"] = "constant"
    elif change == "encoder":
        records[4]["artifact"]["encoder_object"].handle_unknown = "error"
    elif change == "callback":
        records[4]["artifact"]["encoder_object"]._transform = lambda *args: None
    else:
        company_artifact.pipeline.model_estimator.model.booster = "dart"
    with pytest.raises(UnsupportedExecutionError):
        require_partition_safe_pipeline(company_artifact)


@pytest.mark.parametrize(
    "change",
    [
        "group_pairs",
        "group_extra",
        "encoder_names",
        "encoder_widths",
        "encoder_indices",
        "encoder_extra",
        "applier",
        "booster_callback",
        "model_callback",
        "custom_objective",
        "training_callback",
        "unknown_config",
        "numeric_fallback",
    ],
)
def test_company_malformed_or_custom_state_is_not_executed(company_artifact, change):
    """Nested callbacks and corrupt saved schemas cannot receive a worker certificate."""
    records = company_artifact.pipeline.feature_engineer.fitted_steps
    encoder = records[4]["artifact"]["encoder_object"]
    model = company_artifact.pipeline.model_estimator.model
    if change == "group_pairs":
        records[2]["artifact"]["group_values"]["amount"].append(["A", 100.0])
    elif change == "group_extra":
        records[2]["artifact"]["batch_mean"] = True
    elif change == "encoder_names":
        records[4]["artifact"]["feature_names"][0] = "wrong"
    elif change == "encoder_widths":
        encoder._n_features_outs[0] += 1
    elif change == "encoder_indices":
        encoder.drop_idx_[0] = 1
    elif change == "encoder_extra":
        encoder.callback = lambda *args: None
    elif change == "applier":
        records[2]["applier"] = object()
    elif change == "booster_callback":
        model.get_booster().predict = lambda *args: None
    elif change == "model_callback":
        model._can_use_inplace_predict = lambda: True
    elif change == "custom_objective":
        model.objective = lambda *args: None
    elif change == "training_callback":
        model.callbacks = [lambda *args: None]
    elif change == "numeric_fallback":
        records[2]["artifact"]["fill_values"]["amount"] = "not-a-number"
    else:
        records[2]["params"]["unexpected"] = 1
    with pytest.raises(UnsupportedExecutionError):
        require_partition_safe_pipeline(company_artifact)


@pytest.mark.parametrize("registry_part", ["_calculators", "_appliers"])
def test_company_registry_replacements_fail_closed(company_artifact, monkeypatch, registry_part):
    """A registered substitution must not inherit trusted batch capabilities."""
    from skyulf.registry import NodeRegistry

    monkeypatch.setitem(getattr(NodeRegistry, registry_part), "GroupImputer", object)
    with pytest.raises(UnsupportedExecutionError, match="registration"):
        require_partition_safe_pipeline(company_artifact)


def test_group_learned_value_changes_detach_evidence(company_artifact):
    """Workers detect altered learned fills even when the new state is structurally valid."""
    before = require_partition_safe_pipeline(company_artifact)
    record = company_artifact.pipeline.feature_engineer.fitted_steps[2]
    record["artifact"]["fill_values"]["amount"] += 0.5
    after = require_partition_safe_pipeline(company_artifact)
    assert before.state_sha256 != after.state_sha256
    assert before.steps[2].state_sha256 != after.steps[2].state_sha256


def test_batch_mode_admission_does_not_expand_portable_codec(company_artifact):
    """Categorical mode support must remain confined to Python-batch inference."""
    from skyulf.core.portable_state import validate_state

    require_partition_safe_pipeline(company_artifact)
    state = company_artifact.pipeline.feature_engineer.fitted_steps[1]["artifact"]
    with pytest.raises(ValueError, match="mean and constant"):
        validate_state("SimpleImputer", state)
