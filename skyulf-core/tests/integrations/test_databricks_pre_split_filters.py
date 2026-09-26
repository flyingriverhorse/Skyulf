"""Training eligibility filters run on bounded rows before the final split."""

import hashlib
import json
from dataclasses import asdict
from datetime import UTC, datetime
from types import SimpleNamespace
from typing import Any

import numpy as np
import pandas as pd
import polars as pl
import pytest

from skyulf.integrations.databricks import local_retraining as training


def _spec(*, steps=(), **changes):
    """Keep source roles explicit so extra filter columns stay out of the model."""
    values: dict[str, Any] = {
        "table": "workspace.test.labels",
        "version": 3,
        "record_key_columns": ("id",),
        "input_columns": ("x",),
        "target_column": "target",
        "max_rows": 20,
        "max_bytes": 100000,
        "pre_split_steps": steps,
    }
    values.update(changes)
    return training.LocalTrainingSpec(**values)


def _frame():
    """Supply an explicit filter-only column and duplicate pandas index labels."""
    return pd.DataFrame(
        {
            "id": [1, 2, 3, 4, 5, 6, 7, 8],
            "x": [1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0],
            "age": [20.0, -1.0, 40.0, 50.0, 60.0, 70.0, 80.0, 90.0],
            "target": [2.0, 4.0, np.nan, 8.0, 10.0, 12.0, 14.0, 16.0],
        },
        index=[0, 0, 1, 1, 2, 2, 3, 3],
    )


@pytest.mark.parametrize("engine", ["pandas", "polars"])
def test_candidate_training_filters_before_split_and_model_fit(monkeypatch, tmp_path, engine):
    """Rejected rows and filter-only columns cannot enter model fit or evaluation."""
    steps = (
        {
            "name": "known_target",
            "transformer": "DropMissingRows",
            "params": {"subset": ["target"]},
        },
        {
            "name": "valid_age",
            "transformer": "ManualBounds",
            "params": {"bounds": {"age": {"lower": 0}}},
        },
    )
    spec = _spec(steps=steps)
    observed = {}
    monkeypatch.setattr(training, "read_training_snapshot", lambda spark, request: _frame())

    def capture_fit(config, dataset, **kwargs):
        """Capture the actual fit input while allowing the holdout path to run."""
        observed["train"] = dataset.train
        return SimpleNamespace(manifest=SimpleNamespace(pipeline_sha256="a" * 64))

    def capture_holdout(artifact, frame, **kwargs):
        """Stop before MLflow after checking the real split and Core filters."""
        observed["holdout"] = frame
        raise RuntimeError("captured filtered holdout")

    monkeypatch.setattr(training, "fit_local_workflow", capture_fit)
    monkeypatch.setattr(training, "evaluate_local_holdout", capture_holdout)
    with pytest.raises(RuntimeError, match="captured filtered holdout"):
        training.train_local_candidate(
            object(),
            spec,
            {"preprocessing": [], "modeling": {"type": "linear_regression"}},
            model_name="workspace.test.model",
            tracking_uri="unused",
            registry_uri="unused",
            experiment_name="test",
            run_name="test",
            artifact_path=tmp_path,
            metric="heldout_rmse",
            min_improvement=0.0,
            engine=engine,
        )
    fit = observed["train"].to_pandas() if engine == "polars" else observed["train"]
    heldout = observed["holdout"].to_pandas() if engine == "polars" else observed["holdout"]
    assert len(fit) + len(heldout) == 6
    assert list(fit.columns) == ["x", "target"]
    assert list(heldout.columns) == ["x", "target"]
    assert set(fit["x"]) | set(heldout["x"]) == {1.0, 4.0, 5.0, 6.0, 7.0, 8.0}


@pytest.mark.parametrize(
    "transformer,params",
    [
        ("StandardScaler", {"columns": ["x"]}),
        ("SimpleImputer", {"columns": ["x"], "strategy": "mean"}),
        ("unknown_custom", {}),
        ("DropMissingRows", {}),
        ("ManualBounds", {"bounds": {"age": {"lower": 10, "upper": 0}}}),
    ],
)
def test_training_rejects_unsafe_recipe_before_source_read(monkeypatch, transformer, params):
    """A bad pre-split recipe must fail before a candidate reads or fits data."""
    from skyulf.integrations.databricks import local_workflow

    config = {
        "training_table": "workspace.test.labels",
        "training_version": 3,
        "record_key_columns": ["id"],
        "input_columns": ["x"],
        "target_column": "target",
        "max_rows": 20,
        "max_input_mb": 1,
        "engine": "pandas",
        "split_strategy": "random",
        "training_window_mode": "full_snapshot",
        "promotion_policy": "manual_approval",
        "score_model_selection": "pinned_version",
        "model_name": "workspace.test.model",
        "metric": "heldout_rmse",
        "min_improvement": 0.0,
        "pipeline": {"preprocessing": [], "modeling": {"type": "linear_regression"}},
        "pre_split_steps": [{"name": "bad", "transformer": transformer, "params": params}],
    }
    monkeypatch.setattr(local_workflow, "read_training_snapshot", lambda *args: pytest.fail("read"))
    with pytest.raises(ValueError, match="pre.split|column|bounds"):
        local_workflow.run_action(
            object(), config, "train", experiment_name="test", artifact_path="unused"
        )


def test_filter_replay_preserves_holdout_membership_and_rejects_changed_recipe():
    """Saved membership must refer to the filtered population on every replay."""
    steps = ({"name": "known", "transformer": "DropMissingRows", "params": {"subset": ["target"]}},)
    spec = _spec(steps=steps)
    frame = _frame()
    train, holdout, excluded = training.split_labeled_snapshot(frame, spec)
    assert len(train) + len(holdout) == 7
    assert excluded == 0
    assert holdout.attrs["pre_split_filter_counts"][0]["excluded_rows"] == 1
    pinned = training.replace(spec, holdout_key_sha256=holdout.attrs["holdout_key_sha256"])
    replay = training.split_labeled_snapshot(frame.iloc[::-1], pinned)
    pd.testing.assert_frame_equal(holdout, replay[1])
    changed = (
        *steps,
        {
            "name": "valid_age",
            "transformer": "ManualBounds",
            "params": {"bounds": {"age": {"lower": 0}}},
        },
    )
    with pytest.raises(ValueError, match="Holdout membership"):
        training.split_labeled_snapshot(frame, training.replace(pinned, pre_split_steps=changed))


def test_empty_recipe_preserves_legacy_dataset_identity():
    """Old saved candidates without a recipe must retain their comparison digest."""
    spec = _spec()
    old = asdict(spec)
    for key in ("pre_split_steps", "max_rows", "max_bytes"):
        old.pop(key)
    for key in ("start", "holdout_start", "cutoff", "result_cutoff"):
        old[key] = None
    expected = hashlib.sha256(json.dumps(old, sort_keys=True).encode()).hexdigest()
    saved = asdict(spec)
    saved.pop("pre_split_steps")
    saved["event_time_parsing"] = spec.event_time_parsing
    saved["result_time_parsing"] = spec.result_time_parsing
    restored = training.LocalTrainingSpec(**saved)
    assert spec.dataset_id == restored.dataset_id == f"{spec.table}@3/random/{expected}"
    assert (
        training.replace(
            spec,
            pre_split_steps=(
                {
                    "name": "known",
                    "transformer": "DropMissingRows",
                    "params": {"subset": ["target"]},
                },
            ),
        ).dataset_id
        != spec.dataset_id
    )


@pytest.mark.parametrize("engine", ["pandas", "polars"])
def test_filter_only_column_and_sample_counts_are_preserved(engine):
    """Sampling pins selected raw keys while filters can shrink the final cohort."""
    steps = (
        {
            "name": "valid_age",
            "transformer": "ManualBounds",
            "params": {"bounds": {"age": {"lower": 0}}},
        },
    )
    spec = _spec(steps=steps, training_sample_rows=8)
    assert spec.source_columns == ("id", "x", "target", "age")
    frame = _frame().assign(target=lambda values: values.target.fillna(6.0))
    frame.attrs["training_selection"] = {
        "source_rows": 8,
        "eligible_rows": 8,
        "unavailable_labels": 0,
    }
    train, holdout, missing = training.split_labeled_snapshot(frame, spec, engine=engine)
    assert len(train) + len(holdout) == 7
    assert holdout.attrs["sample_key_sha256"] is not None
    assert holdout.attrs["pre_split_filter_counts"] == [
        {
            "name": "valid_age",
            "transformer": "ManualBounds",
            "input_rows": 8,
            "excluded_rows": 1,
            "output_rows": 7,
        }
    ]
    assert missing == 0


@pytest.mark.parametrize("engine", ["pandas", "polars"])
def test_missing_filter_column_and_empty_population_fail_before_fit(engine):
    """Silent Core no-ops and empty split partitions must be visible errors."""
    missing = _spec(
        steps=(
            {
                "name": "bad",
                "transformer": "ManualBounds",
                "params": {"bounds": {"absent": {"lower": 0}}},
            },
        )
    )
    with pytest.raises(ValueError, match="missing required source columns.*absent"):
        training.split_labeled_snapshot(_frame(), missing, engine=engine)
    rejected = _spec(
        steps=(
            {
                "name": "reject",
                "transformer": "ManualBounds",
                "params": {"bounds": {"age": {"lower": 1000}}},
            },
        )
    )
    with pytest.raises(ValueError, match="fewer than four eligible rows"):
        training.split_labeled_snapshot(
            _frame().assign(target=lambda df: df.target.fillna(6.0)), rejected, engine=engine
        )


def test_pre_split_recipe_does_not_hide_invalid_keys_or_mutation():
    """Filters cannot conceal duplicate identities or bypass admission after construction."""
    steps = ({"name": "known", "transformer": "DropMissingRows", "params": {"subset": ["target"]}},)
    spec = _spec(steps=steps)
    duplicate = _frame().copy()
    duplicate.loc[duplicate.index[-1], "id"] = 1
    with pytest.raises(ValueError, match="unique"):
        training.split_labeled_snapshot(duplicate, spec)
    spec.pre_split_steps[0]["transformer"] = "StandardScaler"
    with pytest.raises(ValueError, match="before split"):
        training.split_labeled_snapshot(_frame(), spec)


def test_project_loader_owns_optional_recipe_and_returns_fresh_steps(tmp_path):
    """Removing the Python recipe clears it and JSON cannot override it."""
    from skyulf.integrations.databricks.project import load_project_workflow

    source = tmp_path / "preprocessing.py"
    config = {"pipeline": {"preprocessing": [], "modeling": {"type": "linear_regression"}}}
    source.write_text("def build_preprocessing():\n    return []\n", encoding="utf-8")
    assert load_project_workflow(config, source)["pre_split_steps"] == []
    source.write_text(
        "def build_preprocessing():\n    return []\n"
        "def build_pre_split_steps():\n"
        "    return [{'name': 'known', 'transformer': 'DropMissingRows', "
        "'params': {'subset': ['target']}}]\n",
        encoding="utf-8",
    )
    first = load_project_workflow(config, source)
    first["pre_split_steps"][0]["params"]["subset"].append("x")
    second = load_project_workflow(config, source)
    assert second["pre_split_steps"][0]["params"]["subset"] == ["target"]
    with pytest.raises(ValueError, match="pre_split_steps"):
        load_project_workflow({**config, "pre_split_steps": first["pre_split_steps"]}, source)


@pytest.mark.parametrize("engine", ["pandas", "polars"])
def test_temporal_filtering_precedes_fold_local_cv(engine):
    """Time-series folds see only surviving training rows and retain event metadata."""
    from skyulf.integrations.databricks.local_cv import LocalCVSpec, evaluate_training_cv

    frame = pd.DataFrame(
        {
            "id": range(30),
            "x": [float(i) for i in range(30)],
            "age": [float(i) for i in range(30)],
            "target": [float(i * 2) for i in range(30)],
            "event": pd.date_range("2026-01-01", periods=30, tz="UTC"),
        }
    ).iloc[::-1]
    frame.loc[frame.id == 5, "age"] = -1.0
    frame.loc[frame.id == 6, "target"] = np.nan
    steps = (
        {"name": "known", "transformer": "DropMissingRows", "params": {"subset": ["target"]}},
        {"name": "age", "transformer": "ManualBounds", "params": {"bounds": {"age": {"lower": 0}}}},
    )
    spec = _spec(
        steps=steps,
        max_rows=30,
        split_strategy="temporal",
        event_column="event",
        start=datetime(2026, 1, 1, tzinfo=UTC),
        holdout_start=datetime(2026, 1, 25, tzinfo=UTC),
        cutoff=datetime(2026, 2, 1, tzinfo=UTC),
    )
    train, holdout, _ = training.split_labeled_snapshot(
        frame, spec, keep_training_event=True, engine=engine
    )
    assert len(train) == 22 and len(holdout) == 6
    assert list(train.columns) == ["x", "target", "event"]
    assert "event" not in holdout
    native = pl.from_pandas(train) if engine == "polars" else train
    result = evaluate_training_cv(
        native,
        {"preprocessing": [], "modeling": {"type": "linear_regression"}},
        LocalCVSpec(enabled=True, folds=2, method="time_series_split", shuffle=False),
        target_column="target",
        event_column="event",
    )
    assert result is not None and len(result["folds"]) == 2


def test_offline_preview_names_filter_phase_and_project_recipe(workflow_config):
    """Users can inspect filter order offline without a source read or model fit."""
    from skyulf.integrations.databricks.workflow_config import preview_workflow_config

    config = dict(workflow_config)
    config["pre_split_steps"] = [
        {"name": "known", "transformer": "DropMissingRows", "params": {"subset": ["target"]}}
    ]
    preview = preview_workflow_config(config, action="train")
    assert "known -> DropMissingRows" in preview
    assert "training filters -> final split" in preview
    assert "build_pre_split_steps()" in preview
    assert "fold-local preprocessing/model" in preview


@pytest.mark.parametrize("engine", ["pandas", "polars"])
@pytest.mark.parametrize(
    "params",
    [
        {"subset": ["x", "target"], "threshold": 2, "missing_threshold": 100, "how": "all"},
        {"subset": ["x", "target"], "missing_threshold": 0, "how": "all"},
    ],
)
def test_drop_missing_uses_core_threshold_precedence_and_clears_missing_targets(engine, params):
    """Explicit row thresholds dominate how and remove missing training labels."""
    frame = _frame()
    frame.loc[frame.id == 2, "x"] = np.nan
    spec = _spec(steps=({"name": "complete", "transformer": "DropMissingRows", "params": params},))
    train, holdout, _ = training.split_labeled_snapshot(frame, spec, engine=engine)
    assert len(train) + len(holdout) == 6
    assert holdout.attrs["pre_split_filter_counts"][0]["excluded_rows"] == 2
    assert not train["target"].isna().any() and not holdout["target"].isna().any()


@pytest.mark.parametrize("engine", ["pandas", "polars"])
def test_manual_bounds_rejects_boolean_source_values(engine):
    """Boolean comparisons must not disagree across pandas and Polars."""
    frame = _frame().assign(flag=[True, False] * 4)
    spec = _spec(
        steps=(
            {
                "name": "flag",
                "transformer": "ManualBounds",
                "params": {"bounds": {"flag": {"lower": 0}}},
            },
        )
    )
    with pytest.raises(ValueError, match="numeric column flag"):
        training.split_labeled_snapshot(frame, spec, engine=engine)


@pytest.mark.parametrize("engine", ["pandas", "polars"])
def test_automatic_promotion_replays_saved_filter_spec(monkeypatch, engine):
    """Promotion rechecks the filtered holdout rather than the raw population."""
    from skyulf.integrations.databricks import local_workflow

    steps = ({"name": "known", "transformer": "DropMissingRows", "params": {"subset": ["target"]}},)
    original = _spec(steps=steps)
    saved = asdict(original)
    saved["event_time_parsing"] = original.event_time_parsing
    saved["result_time_parsing"] = original.result_time_parsing
    saved["pre_split_steps"] = tuple(saved["pre_split_steps"])
    restored = training.LocalTrainingSpec(**saved)
    expected = training.split_labeled_snapshot(_frame(), restored)[1]
    candidate = SimpleNamespace(
        holdout_key_sha256=expected.attrs["holdout_key_sha256"],
        comparison=SimpleNamespace(champion_version=None, candidate_version="2"),
    )
    monkeypatch.setattr(local_workflow, "read_training_snapshot", lambda spark, spec: _frame())
    observed = {}

    def capture_stage(report, heldout, **kwargs):
        """Inspect the actual holdout passed into challenger staging."""
        observed["heldout"] = heldout

    monkeypatch.setattr(local_workflow, "stage_challenger", capture_stage)
    assert (
        local_workflow._automatic_promotion(
            object(), {"engine": engine}, restored, candidate, promote=False
        )
        is None
    )
    actual = observed["heldout"].to_pandas() if engine == "polars" else observed["heldout"]
    assert actual["x"].tolist() == expected["x"].tolist()


def test_polars_filter_chain_converts_only_at_split_boundary(monkeypatch):
    """Each Core filter must receive the same selected engine without pandas round trips."""
    steps = (
        {"name": "known", "transformer": "DropMissingRows", "params": {"subset": ["target"]}},
        {"name": "age", "transformer": "ManualBounds", "params": {"bounds": {"age": {"lower": 0}}}},
    )
    calls = []
    original = training.pl.from_pandas

    def record_conversion(frame, *args, **kwargs):
        """Count engine conversion during the shared split path."""
        calls.append(len(frame))
        return original(frame, *args, **kwargs)

    monkeypatch.setattr(training.pl, "from_pandas", record_conversion)
    training.split_labeled_snapshot(_frame(), _spec(steps=steps), engine="polars")
    assert calls == [8]


@pytest.mark.parametrize("in_place", [False, True])
def test_target_pairing_guard_keeps_large_integer_precision(monkeypatch, in_place):
    """A row filter that changes a 53-bit-plus label cannot evade pairing checks."""
    frame = _frame().assign(target=[2**53 + 1 + i for i in range(8)])
    spec = _spec(
        steps=(
            {
                "name": "age",
                "transformer": "ManualBounds",
                "params": {"bounds": {"age": {"lower": -10}}},
            },
        )
    )

    class MutatingApplier:
        """Model a broken Core applier that changes one label without changing keys."""

        def apply(self, native, artifact):
            """Change only one integer target so the pairing guard must reject it."""
            result = native if in_place else native.copy()
            result.loc[0, "target"] = 2**53
            return result

    monkeypatch.setattr(training.NodeRegistry, "get_applier", lambda name: MutatingApplier)
    with pytest.raises(ValueError, match="preserve target pairing"):
        training.split_labeled_snapshot(frame, spec)


@pytest.mark.parametrize("engine", ["pandas", "polars"])
def test_filtering_reports_stratification_failure_before_fit(engine):
    """A filter cannot leave one class too small for the declared stratified split."""
    frame = _frame().assign(target=[0] * 4 + [1] * 4, age=[10.0] * 5 + [-1.0] * 3)
    step = {
        "name": "eligible_age",
        "transformer": "ManualBounds",
        "params": {"bounds": {"age": {"lower": 0}}},
    }
    spec = _spec(steps=(step,), stratify=True)
    with pytest.raises(ValueError, match="stratification requires at least two rows per class"):
        training.split_labeled_snapshot(frame, spec, engine=engine)
