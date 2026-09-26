"""Exercise complete local Bundle recipes for previously unproven node families."""

from pathlib import Path

import numpy as np
import pandas as pd
import polars as pl
import pytest

from skyulf.data.dataset import SplitDataset
from skyulf.inference.local_evaluation import evaluate_local_holdout
from skyulf.inference.local_pipeline import load_local_pipeline, predict_local_pipeline
from skyulf.inference.project_code import load_project_module
from skyulf.integrations.databricks.local_batch import fit_local_workflow
from skyulf.integrations.databricks.local_cv import LocalCVSpec, evaluate_training_cv
from skyulf.integrations.databricks.local_retraining import (
    LocalTrainingSpec,
    split_labeled_snapshot,
)
from skyulf.integrations.databricks.project import load_project_workflow
from skyulf.registry import NodeRegistry

CASES = [
    ("count_vectorizer", {"columns": ["text"], "drop_original": True}, "text"),
    ("tfidf_vectorizer", {"columns": ["text"], "drop_original": True}, "text"),
    (
        "hashing_vectorizer",
        {"columns": ["text"], "drop_original": True, "n_features": 8},
        "text",
    ),
    (
        "tokenizer",
        {"columns": ["text"], "drop_original": True, "add_token_count": True},
        "text",
    ),
    (
        "GeoDistance",
        {"lat1_col": "x", "lon1_col": "z", "lat2_col": "lat", "lon2_col": "lon"},
        "geo",
    ),
    ("LagFeatures", {"columns": ["x"], "lags": [1], "sort_by": "x"}, "temporal"),
    (
        "RollingAggregate",
        {"columns": ["x"], "window": 2, "min_periods": 1, "sort_by": "x"},
        "temporal",
    ),
    ("DatasetProfile", {}, "inspection"),
    ("DataSnapshot", {"n_rows": 2}, "inspection"),
    ("Oversampling", {"method": "random_over", "random_state": 7}, "resampling"),
    (
        "Undersampling",
        {"method": "random_under_sampling", "random_state": 7},
        "resampling",
    ),
]


def _recipe_frame(family):
    """Keep heldout rows distinct and include a token never seen during fitting."""
    x = np.arange(48, dtype=float) + 1
    frame = pd.DataFrame({"x": x, "z": np.sin(x), "target": 2 * x + 1})
    if family == "text":
        frame["text"] = ["red apple" if index % 2 else "green pear" for index in range(48)]
        frame.loc[40:, "text"] = "unseenword apple"
    elif family == "geo":
        frame["lat"] = x + 0.5
        frame["lon"] = frame["z"] + 0.5
    elif family == "resampling":
        frame["target"] = (np.arange(48) % 4 == 0).astype(int)
    return frame


@pytest.mark.parametrize("engine", ["pandas", "polars"])
@pytest.mark.parametrize("node_id,params,family", CASES, ids=[case[0] for case in CASES])
def test_python_recipe_preserves_raw_predictions_after_artifact_reload(
    tmp_path, engine, node_id, params, family
):
    """A registered node must work in a saved full model, not only as a direct applier."""
    if family == "resampling":
        pytest.importorskip("imblearn")
    steps = [{"name": "candidate", "transformer": node_id, "params": params}]
    if node_id == "tokenizer":
        steps.append(
            {
                "name": "numeric_token_count_only",
                "transformer": "DropMissingColumns",
                "params": {"columns": ["text__tokens"]},
            }
        )
    steps.append({"name": "fill", "transformer": "SimpleImputer", "params": {"strategy": "mean"}})
    source_path = tmp_path / "preprocessing.py"
    source_path.write_text(f"def build_preprocessing():\n    return {steps!r}\n", encoding="utf-8")
    config = load_project_workflow(
        {
            "pipeline": {
                "preprocessing": [],
                "modeling": {
                    "type": "logistic_regression" if family == "resampling" else "linear_regression"
                },
            }
        },
        source_path,
    )["pipeline"]
    raw = _recipe_frame(family)
    train = raw.iloc[:40].copy()
    heldout = raw.iloc[40:].copy()
    native_train = pl.from_pandas(train) if engine == "polars" else train
    path = tmp_path / "artifact"
    artifact = fit_local_workflow(
        config,
        SplitDataset(train=native_train, test=native_train.head(0)),
        target_column="target",
        artifact_path=path,
        max_rows=100,
        max_bytes=1_000_000,
    )
    query = heldout.drop(columns="target")
    expected = predict_local_pipeline(query, artifact)
    source_path.write_text("raise RuntimeError('edited source must not be used')", encoding="utf-8")
    restored = load_local_pipeline(path)
    actual = predict_local_pipeline(pl.from_pandas(query), restored)
    assert restored.manifest.fitted_engine == engine
    assert len(actual) == len(query) == 8
    np.testing.assert_allclose(actual["prediction"], expected["prediction"], atol=1e-10)
    state = restored.pipeline.feature_engineer.fitted_steps[0]["artifact"]
    assert state
    if node_id in {"count_vectorizer", "tfidf_vectorizer"}:
        assert "unseenword" not in state["vocabulary"]
    if node_id == "DatasetProfile":
        assert state["profile"]["rows"] == 40
    elif node_id == "DataSnapshot":
        assert len(state["snapshot"]) == 2
    if family == "temporal":
        with pytest.raises(ValueError, match="row order"):
            predict_local_pipeline(query.iloc[::-1], restored)
    evaluation = evaluate_local_holdout(restored, heldout, target_column="target")
    metric = "heldout_accuracy" if family == "resampling" else "heldout_rmse"
    assert np.isfinite(evaluation[metric])
    if family != "temporal":
        cv = evaluate_training_cv(
            native_train, config, LocalCVSpec(enabled=True, folds=2), target_column="target"
        )
        assert cv is not None and cv["aggregated_metrics"]


@pytest.mark.parametrize("engine", ["pandas", "polars"])
@pytest.mark.parametrize("node_id", ["count_vectorizer", "tfidf_vectorizer", "hashing_vectorizer"])
def test_empty_text_partition_retains_fitted_columns_and_dtypes(engine, node_id):
    """Empty evaluation data must have the same feature schema as nonempty data."""
    raw = pd.DataFrame({"text": ["red apple", "green pear"], "x": [1.0, 2.0]})
    native = pl.from_pandas(raw) if engine == "polars" else raw
    calculator = NodeRegistry.get_calculator(node_id)()
    applier = NodeRegistry.get_applier(node_id)()
    state = calculator.fit(native, {"columns": ["text"], "drop_original": True, "n_features": 8})
    full = applier.apply(native, state)
    empty = applier.apply(native.head(0), state)
    assert len(empty) == 0
    assert list(empty.columns) == list(full.columns)
    assert list(empty.dtypes) == list(full.dtypes)


@pytest.mark.parametrize("engine", ["pandas", "polars"])
def test_template_custom_eligibility_example_keeps_only_known_false_flags(engine):
    """The editable generated example must work on both engines without target or key edits."""
    source = (
        Path(__file__).resolve().parents[2]
        / "templates/databricks/template/{{.project_name}}/src/preprocessing.py"
    ).read_text(encoding="utf-8")
    module = load_project_module(source)
    spec = LocalTrainingSpec(
        table="workspace.example.source",
        version=0,
        record_key_columns=("id",),
        input_columns=("x",),
        target_column="target",
        max_rows=20,
        max_bytes=100000,
        pre_split_steps=(module.example_custom_pre_split("is_test"),),
    )
    frame = pd.DataFrame(
        {
            "id": range(16),
            "x": np.arange(16, dtype=float),
            "target": np.arange(16) * 2,
            "is_test": [True, None, *([False] * 14)],
        }
    )
    train, heldout, _ = split_labeled_snapshot(frame, spec, engine=engine)
    assert set(train.x) | set(heldout.x) == set(range(2, 16))
    assert "is_test" in spec.source_columns and "is_test" not in train.columns
    assert heldout.attrs["pre_split_filter_counts"][0]["excluded_rows"] == 2
