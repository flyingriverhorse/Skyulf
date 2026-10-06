"""Bundle source weights remain aligned and absent from prediction features."""

import hashlib
from dataclasses import replace
from typing import Any
from unittest.mock import MagicMock

import numpy as np
import pandas as pd
import pytest
from sklearn.linear_model import LinearRegression, LogisticRegression

from skyulf.integrations.databricks.training.fitting.local_retraining import (
    LocalTrainingSpec,
    split_labeled_snapshot,
    training_spec_payload,
)


def _spec(**changes):
    """Build a bounded source contract with an explicit row weight."""
    values: dict[str, Any] = {
        "table": "main.test.source",
        "version": 1,
        "record_key_columns": ("id",),
        "input_columns": ("x",),
        "target_column": "y",
        "max_rows": 100,
        "max_bytes": 100000,
        "weight_column": "w",
        "reserved_weight_columns": ("w",),
    }
    return LocalTrainingSpec(**(values | changes))


def test_split_keeps_weights_aligned_after_missing_labels():
    """Branch label masks must affect weights and model rows identically."""
    frame = pd.DataFrame(
        {"id": range(20), "x": range(20), "y": range(20), "w": [value + 1.0 for value in range(20)]}
    )
    frame.loc[3, "y"] = None
    train, holdout, skipped = split_labeled_snapshot(frame, _spec(drop_missing_labels=True))
    assert train.w.tolist() == (train.x + 1).tolist()
    assert "w" not in holdout
    assert holdout.attrs["training_weights"]["count"] == len(train)
    assert skipped == 1


def test_spec_roundtrip_and_inactive_source_preserve_identity():
    """Frozen replay restores tuples and generated inactive hooks preserve legacy IDs."""
    spec = _spec()
    assert LocalTrainingSpec.from_payload(training_spec_payload(spec, "pandas")) == spec
    inactive = _spec(weight_column=None, reserved_weight_columns=())
    source = "DEFAULT_WEIGHTS = {'weight_column': None}\n"
    captured = replace(
        inactive,
        weights_python_source=source,
        weights_python_sha256=hashlib.sha256(source.encode()).hexdigest(),
    )
    assert inactive.dataset_id == captured.dataset_id


@pytest.mark.parametrize("layout", ["single_model", "model_competition", "multi_target"])
@pytest.mark.parametrize("strategy", ["random", "temporal"])
@pytest.mark.parametrize("class_weight,column", [("balanced", "w"), (None, "importance")])
def test_three_layouts_fit_real_weights_and_score_without_them(
    tmp_path, workflow_config, monkeypatch, layout, strategy, class_weight, column
):
    """Every editable layout must deliver aligned user and native class weights to sklearn."""
    from test_simple_bundle_weight_config import _load, _project

    from skyulf.inference.local_pipeline import predict_local_pipeline
    from skyulf.integrations.databricks.lifecycle.local_workflow import training_spec
    from skyulf.integrations.databricks.training.fitting.local_retraining import fit_candidate
    from skyulf.integrations.databricks.training.tuning.local_cv import LocalCVSpec

    workflow_config.update(
        training_window_mode="fixed_window" if strategy == "temporal" else "full_snapshot",
        split_strategy=strategy,
        filter_unavailable_results=False,
        result_available_at_column=None,
        result_cutoff=None,
        record_key_columns=["id"],
        engine="pandas",
        monthly_lookback_months=None,
        window_timezone=None,
    )
    if strategy == "random":
        workflow_config.update(start=None, holdout_start=None, cutoff=None, event_column=None)
    source = f"WEIGHT_COLUMN = {column!r}\n"
    project = _project(
        tmp_path / "project",
        workflow_config,
        layout,
        source,
        branch_overlays={name: {"weight_column": column} for name in ("risk", "revenue")},
    )
    for path in project[2].glob("*.py"):
        path.write_text(
            path.read_text().replace("'balanced'", repr(class_weight)), encoding="utf-8"
        )
    if layout == "multi_target":
        project[0]["pipeline"]["modeling"]["params"] = {"class_weight": class_weight}
        import json

        workflow_path = project[3]["config_path"]
        from pathlib import Path

        Path(workflow_path).write_text(json.dumps(project[0]), encoding="utf-8")
    loaded = _load(project)
    seen = []
    original = LogisticRegression.fit

    def fit(self, X, y, sample_weight=None):
        """Inspect actual estimator inputs while retaining the real training behavior."""
        assert self.class_weight == class_weight
        assert sample_weight is not None
        np.testing.assert_array_equal(sample_weight, np.asarray(X)[:, 0] + 1)
        seen.append(len(X))
        return original(self, X, y, sample_weight=sample_weight)

    monkeypatch.setattr(LogisticRegression, "fit", fit)
    for name, config in loaded.items():
        spec = training_spec(config)
        frame = pd.DataFrame(
            {
                "id": range(60),
                "x": np.arange(60, dtype=float),
                "target": np.arange(60) % 2,
                column: np.arange(60) + 1.0,
            }
        )
        if strategy == "temporal":
            frame["event_time"] = pd.date_range("2026-01-01", periods=60, tz="UTC")
            frame = frame.iloc[:59]
        if layout == "multi_target":
            spec = replace(spec, drop_missing_labels=True)
            frame.loc[3, "target"] = None
        train, heldout, skipped = split_labeled_snapshot(frame, spec)
        cv = LocalCVSpec(enabled=True, folds=3)
        fitted = fit_candidate(
            None,
            spec,
            config["pipeline"],
            run=MagicMock(),
            pipeline_config=config["pipeline"].copy(),
            artifact_path=tmp_path / name,
            engine="pandas",
            cv=cv,
            risk_category=None,
            prepared_data=(frame, train, heldout, skipped),
        )
        if layout == "model_competition":
            from skyulf.integrations.databricks.training.competition.competition_training import (
                fit_training_pipeline,
            )
            from skyulf.integrations.databricks.training.fitting import local_retraining

            monkeypatch.setattr(local_retraining, "log_fitted_candidate", MagicMock())
            monkeypatch.setattr(
                local_retraining, "log_local_model", MagicMock(return_value="runs:/test/model")
            )
            store = MagicMock()
            store.request = {"config": {**config, "tracking_uri": "unused"}}
            for candidate, recipe in config["competition"]["candidates"].items():
                fitted, row = fit_training_pipeline(
                    None,
                    store,
                    spec,
                    (frame, train, heldout, skipped),
                    tmp_path / candidate,
                    candidate,
                    {**recipe, "effective_config": recipe["pipeline"].copy()},
                )
                assert len(row["fold_scores"]) == 3
        assert fitted.artifact.manifest.input_columns == ("x",)
        predicted = predict_local_pipeline(pd.DataFrame({"x": [1.0, 2.0]}), fitted.artifact)
        assert len(predicted) == 2
        assert fitted.artifact.pipeline.config["training_weights"]["count"] == len(train)
    assert len(seen) >= len(loaded) * 4


@pytest.mark.parametrize("nested", [False, True])
@pytest.mark.parametrize("engine", ["pandas", "polars"])
def test_competition_temporal_folds_use_actual_weight_permutation(
    tmp_path, monkeypatch, nested, engine
):
    """Competition ordinary and nested fits must reorder weights exactly with temporal rows."""
    import polars as pl

    from skyulf.data.dataset import SplitDataset
    from skyulf.integrations.databricks.scoring.batch.local_batch import fit_local_workflow
    from skyulf.integrations.databricks.training.competition.competition_evaluation import (
        evaluate_competition_candidate,
    )
    from skyulf.integrations.databricks.training.tuning.local_cv import LocalCVSpec

    frame = pd.DataFrame({"x": np.arange(80, dtype=float), "y": np.arange(80) * 2.0})
    config = {"preprocessing": [], "modeling": {"type": "linear_regression", "params": {}}}
    artifact = fit_local_workflow(
        config,
        SplitDataset(train=frame, test=frame.head(0)),
        target_column="y",
        artifact_path=tmp_path / "artifact",
        max_rows=100,
        max_bytes=100000,
    )
    frame["event"] = pd.date_range("2026-01-01", periods=80, tz="UTC")
    frame = frame.sample(frac=1, random_state=37).reset_index(drop=True)
    weights = frame.x.to_numpy() + 1
    cv = LocalCVSpec(
        enabled=True,
        folds=3,
        shuffle=False,
        method="nested_cv" if nested else "time_series_split",
        nested_type="time_series_split" if nested else "auto",
        inner_folds=2 if nested else None,
    )
    seen = []
    original = LinearRegression.fit

    def fit(self, X, y, sample_weight=None):
        """Compare each genuine fold's rows with the vector actually passed to sklearn."""
        assert sample_weight is not None
        np.testing.assert_array_equal(sample_weight, np.asarray(X)[:, 0] + 1)
        seen.append(len(X))
        return original(self, X, y, sample_weight=sample_weight)

    monkeypatch.setattr(LinearRegression, "fit", fit)
    native = pl.from_pandas(frame) if engine == "polars" else frame
    report = evaluate_competition_candidate(
        native,
        artifact,
        cv,
        target_column="y",
        metric="heldout_rmse",
        max_rows=100,
        max_bytes=100000,
        event_column="event",
        sample_weight=weights,
    )
    assert len(seen) >= 3
    assert report["mean"] == pytest.approx(0, abs=1e-8)
    from skyulf.integrations.databricks.training.tuning.local_cv import evaluate_training_cv

    diagnostics = evaluate_training_cv(
        native, config, cv, target_column="y", event_column="event", sample_weight=weights
    )
    assert diagnostics is not None
    assert len(seen) >= 6


@pytest.mark.parametrize(
    "changes",
    [
        {"input_columns": ("W",)},
        {"target_column": "w"},
        {"weight_column": None, "input_columns": ("w",)},
        {"weights_python_source": "raise RuntimeError('must not execute')"},
        {
            "weights_python_source": "raise RuntimeError('must not execute')",
            "weights_python_sha256": "0" * 64,
        },
    ],
)
def test_spec_rejects_bad_roles_or_unverified_source(changes):
    """Frozen requests must validate metadata without ever executing captured hooks."""
    with pytest.raises(ValueError):
        _spec(**changes)


def test_presplit_cannot_write_inactive_reserved_weight():
    """Disabled branch weights remain immutable source roles for pre-split normalization."""
    step = {
        "name": "cast",
        "transformer": "Casting",
        "params": {"columns": ["w"], "target_type": "float"},
    }
    with pytest.raises(ValueError, match="protected"):
        _spec(weight_column=None, pre_split_steps=(step,))


def test_presplit_filter_aligns_training_weights():
    """Filtering on source weights retains original row values and matching weight vectors."""
    step = {
        "name": "keep",
        "transformer": "ManualBounds",
        "params": {"bounds": {"w": {"lower": 5}}},
    }
    frame = pd.DataFrame(
        {"id": range(30), "x": range(30), "y": range(30), "w": np.arange(30) + 1.0}
    )
    train, _, _ = split_labeled_snapshot(frame, _spec(pre_split_steps=(step,)))
    assert train.w.min() >= 5
    np.testing.assert_array_equal(train.w, train.x + 1)


def test_weight_only_changes_are_not_fresh_but_manual_split_uses_new_weights(monkeypatch):
    """Weight edits must not trigger automatic data freshness but must affect manual training."""
    from test_retraining_data import _assess, freshness

    from skyulf.integrations.databricks.lifecycle.local_workflow import training_spec

    fixture: Any = freshness
    state = fixture.__wrapped__(monkeypatch)
    state["config"].update(weight_column="w", reserved_weight_columns=["w"])
    state["old"]["w"] = 1.0
    state["current"]["w"] = np.arange(20) + 2.0
    result = _assess(state)
    spec = training_spec({**state["config"], "training_version": 1})
    train, _, _ = split_labeled_snapshot(state["current"], spec)
    assert result["status"] == "no_new_training_data"
    assert result["changed_rows"] == 0
    np.testing.assert_array_equal(train.w, train.x + 2)


def test_legacy_dataset_id_omits_all_new_defaults():
    """Unweighted dataset IDs must match the old payload hash byte for byte."""
    import json
    from dataclasses import asdict

    spec = _spec(weight_column=None, reserved_weight_columns=())
    settings = asdict(spec)
    for field in (
        "weight_column",
        "reserved_weight_columns",
        "weights_python_source",
        "weights_python_sha256",
        "drop_missing_labels",
        "group_column",
        "survivor_key_sha256",
        "training_evidence_sha256",
        "pre_split_steps",
        "max_rows",
        "max_bytes",
    ):
        settings.pop(field)
    expected = hashlib.sha256(json.dumps(settings, sort_keys=True).encode()).hexdigest()
    assert spec.dataset_id == f"{spec.table}@{spec.version}/random/{expected}"


@pytest.mark.parametrize("nested", [False, True])
def test_bundle_search_transports_weights_through_temporal_refits(tmp_path, monkeypatch, nested):
    """Bundle search and final refit must receive row weights without exposing the source column."""
    from datetime import UTC, datetime

    from skyulf.integrations.databricks.training.fitting.local_retraining import fit_candidate
    from skyulf.integrations.databricks.training.tuning.local_cv import LocalCVSpec
    from skyulf.integrations.databricks.training.tuning.local_search import prepare_search_pipeline

    spec = _spec(
        split_strategy="temporal",
        event_column="event",
        start=datetime(2026, 1, 1, tzinfo=UTC),
        holdout_start=datetime(2026, 3, 1, tzinfo=UTC),
        cutoff=datetime(2026, 4, 1, tzinfo=UTC),
    )
    frame = pd.DataFrame(
        {
            "id": range(80),
            "x": np.arange(80, dtype=float),
            "y": np.arange(80) * 2.0,
            "w": np.arange(80) + 1.0,
            "event": pd.date_range("2026-01-01", periods=80, tz="UTC"),
        }
    )
    cv = LocalCVSpec(
        enabled=True,
        folds=3,
        shuffle=False,
        method="nested_cv" if nested else "time_series_split",
        nested_type="time_series_split" if nested else "auto",
        inner_folds=2 if nested else None,
    )
    config = {
        "preprocessing": [],
        "modeling": {
            "type": "hyperparameter_tuner",
            "base_model": {"type": "linear_regression", "params": {}},
            "metric": "rmse",
            "strategy": "grid",
            "search_space": {"fit_intercept": [True, False]},
        },
    }
    effective = prepare_search_pipeline(config, cv, target_column="y", event_column="event")
    train, holdout, skipped = split_labeled_snapshot(frame, spec, keep_training_event=True)
    seen = []
    original = LinearRegression.fit

    def fit(self, X, y, sample_weight=None):
        """Observe the final estimator call, including genuine nested inner and outer fits."""
        assert sample_weight is not None
        np.testing.assert_array_equal(sample_weight, np.asarray(X)[:, 0] + 1)
        seen.append(len(X))
        return original(self, X, y, sample_weight=sample_weight)

    monkeypatch.setattr(LinearRegression, "fit", fit)
    fitted = fit_candidate(
        None,
        spec,
        config,
        run=MagicMock(),
        pipeline_config=effective,
        artifact_path=tmp_path / "artifact",
        engine="pandas",
        cv=cv,
        risk_category=None,
        prepared_data=(frame, train, holdout, skipped),
    )
    assert len(seen) >= 7
    assert fitted.artifact.manifest.input_columns == ("x",)


def test_legacy_pinned_request_accepts_new_inactive_spec_defaults():
    """Resuming an old frozen request must tolerate newly introduced inactive optional fields."""
    from skyulf.integrations.databricks.jobs.lifecycle.lifecycle_tasks import _validate_pinned_spec

    current = training_spec_payload(_spec(weight_column=None, reserved_weight_columns=()), "pandas")
    current["reserved_weight_columns"] = []
    legacy = dict(current)
    for name in (
        "weight_column",
        "reserved_weight_columns",
        "weights_python_source",
        "weights_python_sha256",
    ):
        legacy.pop(name)
    _validate_pinned_spec(current, legacy)
    with pytest.raises(ValueError, match="pinned"):
        _validate_pinned_spec({**current, "weight_column": "w"}, legacy)


def test_restore_verified_hook_source_never_executes_it():
    """Frozen training requests use resolved values even when captured Python would raise."""
    source = "raise RuntimeError('mutable hook execution is forbidden')\n"
    spec = _spec(
        weights_python_source=source,
        weights_python_sha256=hashlib.sha256(source.encode()).hexdigest(),
    )
    restored = LocalTrainingSpec.from_payload(training_spec_payload(spec, "pandas"))
    assert restored.weight_column == "w"
    assert restored.weights_python_source == source
