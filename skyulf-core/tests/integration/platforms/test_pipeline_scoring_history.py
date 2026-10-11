"""Local scoring exposes explicit temporal continuation without owning persistence."""

import json
import pickle
from copy import deepcopy
from types import SimpleNamespace

import numpy as np
import pandas as pd
import polars as pl
import pytest

from skyulf.data.dataset import SplitDataset
from skyulf.inference import pipeline_scoring
from skyulf.inference.fitted_pipeline import load_pipeline, save_pipeline
from skyulf.inference.project_code import load_project_module
from skyulf.integrations.databricks.projects.project import load_project_workflow
from skyulf.integrations.databricks.scoring.incremental.history import prediction_history
from skyulf.pipeline import SkyulfPipeline

SCORING_SOURCE = '''
import pandas as pd

def eligible(frame, params):
    """Exclude unavailable observations before estimating."""
    return pd.Series(["negative" if value < 0 else None for value in frame.v],
                     index=frame.index, dtype="string")

def build_scoring():
    """Save one explicit eligibility rule."""
    return {"eligibility": [{"name": "nonnegative", "version": "1",
            "function": "eligible", "params": {}}], "outputs": []}
'''

GROUP_SOURCE = '''
from skyulf.preprocessing import column_step

def total(frame):
    """Use complete supplied groups and retain caller positions."""
    return frame.groupby("g")["v"].transform("sum")

def build_preprocessing():
    """Declare group context without claiming singleton execution."""
    return [column_step("group_total", total, output="total", inference_context="group"),
            {"name": "drop", "transformer": "DropMissingColumns",
             "params": {"columns": ["g", "v"], "missing_threshold": None}}]
'''


def _native(frame, engine):
    """Retain equivalent caller data in each supported native engine."""
    return pl.from_pandas(frame) if engine == "polars" else frame


def _artifact(
    tmp_path, engine, *, kind="RollingAggregate", chained=False, classifier=False, scoring=False
):
    """Transport a real grouped temporal model with the clock removed before modeling."""
    frame = pd.DataFrame(
        {
            "t": np.arange(20, dtype=np.int64) // 2,
            "g": ["a", "b"] * 10,
            "v": np.arange(20, dtype=float),
            "target": np.arange(20, dtype=float) * 2,
        }
    )
    if classifier:
        frame["target"] = (np.arange(20) // 2 % 2).astype(np.int64)
    params = {
        "columns": ["v"],
        "group_by": ["g"],
        "sort_by": "t",
        "history_mode": "carry",
        "window": 3,
        "lags": [1],
    }
    steps = [{"name": "temporal", "transformer": kind, "params": params}]
    if chained:
        steps.append(
            {
                "name": "second",
                "transformer": "RollingAggregate",
                "params": params | {"columns": ["v_lag_1"]},
            }
        )
    steps.extend(
        [
            {"name": "fill", "transformer": "SimpleImputer", "params": {"strategy": "mean"}},
            {
                "name": "drop_keys",
                "transformer": "DropMissingColumns",
                "params": {"columns": ["t", "g", "v"], "missing_threshold": None},
            },
        ]
    )
    config = {
        "preprocessing": steps,
        "modeling": {"type": "logistic_regression" if classifier else "linear_regression"},
    }
    if scoring:
        config["project_python_source"] = SCORING_SOURCE
        config["project_scoring"] = load_project_module(SCORING_SOURCE).build_scoring()
    pipeline = SkyulfPipeline(config)
    native = _native(frame, engine)
    pipeline.fit(SplitDataset(train=native[:16], test=native[16:]), target_column="target")
    save_pipeline(pipeline, tmp_path / "model")
    return load_pipeline(tmp_path / "model")


def _request(engine):
    """Supply complete causal observations with deliberately nonunique pandas indices."""
    frame = pd.DataFrame(
        {"t": [10, 10, 11, 11], "g": ["a", "b", "a", "b"], "v": [20.0, 21.0, 22.0, 23.0]},
        index=[4, 4, 2, 9],
    )
    return _native(frame, engine)


def _score(frame, artifact, **kwargs):
    """Keep the missing API regression as an assertion rather than a collection failure."""
    function = getattr(pipeline_scoring, "score_pipeline_with_history", None)
    assert callable(function), "Local scoring must expose caller-owned temporal continuation."
    return function(frame, artifact, **kwargs)


@pytest.mark.parametrize("engine", ["pandas", "polars"])
@pytest.mark.parametrize("kind,chained", [("RollingAggregate", False), ("LagFeatures", True)])
def test_saved_grouped_history_matches_sequential_reload(tmp_path, engine, kind, chained):
    """Per-step tails must reproduce one complete request across JSON state and model reloads."""
    artifact = _artifact(tmp_path, engine, kind=kind, chained=chained)
    frame = _request(engine)
    before = pickle.dumps(artifact.pipeline.feature_engineer.fitted_steps)
    expected = _score(frame, artifact)
    state = None
    outputs = []
    for batch in (frame[:2], frame[2:]):
        artifact = load_pipeline(tmp_path / "model")
        result = _score(batch, artifact, history_state=state)
        outputs.append(result.frame)
        state = json.loads(json.dumps(result.history))
    np.testing.assert_allclose(pd.concat(outputs).prediction, expected.frame.prediction)
    assert state == expected.history
    assert pickle.dumps(artifact.pipeline.feature_engineer.fitted_steps) == before
    assert expected.frame.index.tolist() == (list(range(4)) if engine == "polars" else [4, 4, 2, 9])


@pytest.mark.parametrize("engine", ["pandas", "polars"])
def test_bootstrap_uses_supplied_snapshot_without_double_seed(tmp_path, engine):
    """Explicit bootstrap must replace the saved training tail, never append it twice."""
    artifact = _artifact(tmp_path, engine)
    early = _native(pd.DataFrame({"t": [0, 1], "g": ["a", "a"], "v": [0.0, 2.0]}), engine)
    with pytest.raises(ValueError, match="late, overlapping or replayed"):
        _score(early, artifact)
    first = _score(early[:1], artifact, bootstrap_history=True)
    continued = _score(early[1:], artifact, history_state=first.history)
    together = _score(early, artifact, bootstrap_history=True)
    np.testing.assert_allclose(
        pd.concat([first.frame, continued.frame]).prediction, together.frame.prediction
    )
    assert continued.history == together.history


@pytest.mark.parametrize("engine", ["pandas", "polars"])
def test_empty_requests_preserve_history_and_output_schema(tmp_path, monkeypatch, engine):
    """An empty batch must keep continuation without invoking the fitted estimator."""
    artifact = _artifact(tmp_path, engine)
    frame = _request(engine)
    prior = _score(frame[:2], artifact)

    def forbidden(*args, **kwargs):
        """Fail if empty scoring reaches the estimator."""
        raise AssertionError("Empty prediction invoked the model.")

    monkeypatch.setattr(artifact.pipeline, "predict", forbidden)
    result = _score(frame[:0], artifact, history_state=prior.history)
    assert result.history == prior.history
    assert result.history is not prior.history
    assert list(result.frame.columns) == ["prediction"]
    assert result.frame.empty
    assert str(result.frame.prediction.dtype) == "Float64"


@pytest.mark.parametrize("engine", ["pandas", "polars"])
def test_prediction_failure_leaves_continuation_reusable(tmp_path, monkeypatch, engine):
    """A model failure after temporal preprocessing cannot advance the caller's state."""
    artifact = _artifact(tmp_path, engine)
    frame = _request(engine)
    first = _score(frame[:2], artifact)
    state = deepcopy(first.history)
    predict = artifact.pipeline.predict

    def fail_after_transform(data, **kwargs):
        """Expose a pending proposal before the model raises."""
        artifact.pipeline.feature_engineer.transform(data, preserve_rows=True)
        raise RuntimeError("model failed")

    monkeypatch.setattr(artifact.pipeline, "predict", fail_after_transform)
    with pytest.raises(RuntimeError, match="model failed"):
        _score(frame[2:], artifact, history_state=first.history)
    monkeypatch.setattr(artifact.pipeline, "predict", predict)
    retry = _score(frame[2:], artifact, history_state=first.history)
    assert first.history == state
    assert retry.history != state
    assert len(retry.frame) == 2


@pytest.mark.parametrize("engine", ["pandas", "polars"])
def test_replay_and_tied_entity_times_are_rejected(tmp_path, engine):
    """Continuation rejects already committed and ambiguous entity/time observations."""
    artifact = _artifact(tmp_path, engine)
    frame = _request(engine)
    first = _score(frame[:2], artifact)
    with pytest.raises(ValueError, match="late, overlapping or replayed"):
        _score(frame[:2], artifact, history_state=first.history)
    tied = _native(pd.DataFrame({"t": [12, 12], "g": ["a", "a"], "v": [24.0, 25.0]}), engine)
    with pytest.raises(ValueError, match="unique entity/time"):
        _score(tied, artifact, history_state=first.history)
    assert first.history is not None


@pytest.mark.parametrize("engine", ["pandas", "polars"])
def test_parallel_proposals_are_detached_and_leave_commit_to_caller(tmp_path, engine):
    """Independent callers may branch from one state; publishing requires caller-owned CAS."""
    artifact = _artifact(tmp_path, engine)
    frame = _request(engine)
    first = _score(frame[:2], artifact)
    before = deepcopy(first.history)
    left = _score(frame[2:], artifact, history_state=first.history)
    right = _score(frame[2:], artifact, history_state=first.history)
    assert left.history == right.history
    assert left.history is not right.history
    assert first.history == before
    left.history["steps"].clear()
    assert right.history["steps"]


@pytest.mark.parametrize("engine", ["pandas", "polars"])
def test_classification_repeat_passes_share_original_history(tmp_path, engine):
    """Predict and predict_proba must read the same original tail within each request."""
    artifact = _artifact(tmp_path, engine, classifier=True)
    frame = _request(engine)
    expected = _score(frame, artifact)
    first = _score(frame[:2], artifact)
    last = _score(frame[2:], artifact, history_state=first.history)
    np.testing.assert_allclose(pd.concat([first.frame, last.frame]), expected.frame)
    assert list(expected.frame.columns) == ["prediction", "probability_0", "probability_1"]
    assert last.history == expected.history


def test_conflicting_bootstrap_and_state_rejected_in_local_and_databricks(tmp_path):
    """Reset cannot silently discard explicitly supplied committed continuation."""
    artifact = _artifact(tmp_path, "pandas")
    state = _score(_request("pandas")[:2], artifact).history
    with pytest.raises(ValueError, match="bootstrap"):
        _score(_request("pandas")[2:], artifact, history_state=state, bootstrap_history=True)
    with pytest.raises(ValueError, match="bootstrap"):
        prediction_history(SimpleNamespace(artifact=artifact), state, bootstrap=True)


def test_wrong_model_and_missing_step_state_rejected(tmp_path):
    """Continuation must bind the model identity and every fitted temporal step."""
    artifact = _artifact(tmp_path, "pandas")
    state = _score(_request("pandas")[:2], artifact).history
    wrong = state | {"model_id": "different"}
    with pytest.raises(ValueError, match="different model"):
        _score(_request("pandas")[2:], artifact, history_state=wrong)
    with pytest.raises(ValueError, match="missing a fitted step"):
        _score(_request("pandas")[2:], artifact, history_state=state | {"steps": {}})


@pytest.mark.parametrize("engine", ["pandas", "polars"])
def test_complete_project_groups_execute_as_one_request_without_history(tmp_path, engine):
    """The wrapper preserves whole-group callback inputs rather than chunking them."""
    source = tmp_path / "preprocessing.py"
    source.write_text(GROUP_SOURCE, encoding="utf-8")
    workflow = load_project_workflow(
        {"pipeline": {"preprocessing": [], "modeling": {"type": "linear_regression"}}}, source
    )
    frame = _native(
        pd.DataFrame(
            {
                "g": ["a", "a", "b", "b", "c", "c", "d", "d"],
                "v": [1.0, 2.0, 4.0, 5.0, 10.0, 11.0, 20.0, 21.0],
                "target": [3.0, 3.0, 9.0, 9.0, 21.0, 21.0, 41.0, 41.0],
            }
        ),
        engine,
    )
    pipeline = SkyulfPipeline(workflow["pipeline"])
    pipeline.fit(SplitDataset(train=frame[:6], test=frame[6:]), target_column="target")
    save_pipeline(pipeline, tmp_path / "model")
    source.write_text("raise AssertionError('Source changed after saving')", encoding="utf-8")
    artifact = load_pipeline(tmp_path / "model")
    incoming = _native(
        pd.DataFrame({"g": ["a", "b", "a", "b"], "v": [2.0, 10.0, 4.0, 15.0]}), engine
    )
    result = _score(incoming, artifact)
    np.testing.assert_allclose(result.frame.prediction, [6.0, 25.0, 6.0, 25.0])
    assert result.history is None
    with pytest.raises(ValueError, match="requires a model with carry"):
        _score(incoming, artifact, history_state={})


@pytest.mark.parametrize("engine", ["pandas", "polars"])
def test_eligibility_exclusions_preserve_previous_history(tmp_path, engine):
    """Excluded observations remain visible outcomes without replacing causal history."""
    artifact = _artifact(tmp_path, engine, scoring=True)
    initial = _score(_request(engine)[:0], artifact)
    excluded = _native(pd.DataFrame({"t": [12], "g": ["a"], "v": [-1.0]}), engine)
    result = _score(excluded, artifact, history_state=initial.history)
    assert result.frame.scoring_status.tolist() == ["excluded"]
    assert result.frame.exclusion_reason.tolist() == ["negative"]
    assert result.history == initial.history
    assert list(initial.frame.columns) == ["prediction", "scoring_status", "exclusion_reason"]


@pytest.mark.parametrize("engine", ["pandas", "polars"])
def test_empty_classification_preserves_probability_columns_and_validates_schema(tmp_path, engine):
    """Zero-row results retain class labels while invalid raw schemas still fail."""
    artifact = _artifact(tmp_path, engine, classifier=True)
    result = _score(_request(engine)[:0], artifact)
    assert list(result.frame.columns) == ["prediction", "probability_0", "probability_1"]
    assert str(result.frame.prediction.dtype) == "Int64"
    with pytest.raises(ValueError, match="columns|schema"):
        _score(_native(pd.DataFrame({"v": pd.Series(dtype=float)}), engine), artifact)


@pytest.mark.parametrize("request_kind", ["empty", "excluded"])
@pytest.mark.parametrize(
    "malformation", ["not_rows", "wrong_schema", "duplicate_keys", "oversized"]
)
def test_preserved_history_validates_rows_keys_and_budgets(tmp_path, request_kind, malformation):
    """No-estimate requests cannot publish malformed continuation as successful state."""
    artifact = _artifact(tmp_path, "pandas", scoring=request_kind == "excluded")
    initial = _score(_request("pandas")[:0], artifact)
    state = deepcopy(initial.history)
    params = artifact.pipeline.feature_engineer.fitted_steps[0]["artifact"]
    row = state["steps"][params["history_id"]][0]
    malformed = {
        "not_rows": "not rows",
        "wrong_schema": [{"unexpected": 1}],
        "duplicate_keys": [row, row],
        "oversized": [row] * (params["history_max_rows"] + 1),
    }[malformation]
    state["steps"][params["history_id"]] = malformed
    original = deepcopy(state)
    frame = (
        _request("pandas")[:0]
        if request_kind == "empty"
        else pd.DataFrame({"t": [12], "g": ["a"], "v": [-1.0]})
    )
    with pytest.raises(ValueError, match="Temporal history"):
        _score(frame, artifact, history_state=state)
    assert state == original
