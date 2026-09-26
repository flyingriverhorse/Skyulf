"""Pre-split custom eligibility and deduplication keep saved rows reproducible."""

import json
import logging
import subprocess
import sys
from pathlib import Path
from typing import Any

import pandas as pd
import pytest

from skyulf.inference.project_code import load_project_module
from skyulf.integrations.databricks import local_retraining as training
from skyulf.integrations.databricks.project import load_project_workflow
from skyulf.registry import NodeRegistry

SOURCE = """
import pandas as pd
import polars as pl
from skyulf.inference.project_code import custom_step
from skyulf.preprocessing.base import BaseCalculator, BaseApplier, fit_method, apply_method

MODE = "filter"
DECLARATION = {"effect": "filter", "required_columns": ["is_test"], "learns_from_data": False}

class EligibleApplier(BaseApplier):
    @apply_method
    def apply(self, X, y, params):
        frame = X.to_native() if hasattr(X, "to_native") else X
        if MODE == "edit":
            if isinstance(frame, pl.DataFrame):
                return frame.with_columns((pl.col("x") + 1).alias("x"))
            return frame.assign(x=frame["x"] + 1)
        if MODE == "reverse":
            return frame.reverse() if isinstance(frame, pl.DataFrame) else frame.iloc[::-1]
        if MODE == "new_column":
            return frame.with_columns(pl.lit(1).alias("extra")) if isinstance(frame, pl.DataFrame) else frame.assign(extra=1)
        if MODE == "wrong_type":
            return frame.to_pandas() if isinstance(frame, pl.DataFrame) else pl.from_pandas(frame)
        if MODE == "dtype":
            if isinstance(frame, pl.DataFrame):
                return frame.with_columns(pl.col("x").cast(pl.Int64))
            return frame.assign(x=frame["x"].astype(int))
        if MODE == "null_fill":
            frame.loc[frame["x"].isna(), "x"] = 0
            return frame
        if MODE == "inplace_drop":
            if isinstance(frame, pl.DataFrame):
                frame.drop_in_place("is_test")
            else:
                frame.drop(columns=["is_test"], inplace=True)
            return frame
        if MODE == "inplace_add":
            frame["extra"] = 1
            return frame
        if MODE == "inplace_edit":
            frame["x"] = frame["x"] + 1
            return frame
        if MODE in {"fit_drop", "fit_edit"}:
            return frame
        if isinstance(frame, pl.DataFrame):
            return frame.filter(pl.col("is_test") == "eligible")
        return frame.loc[frame["is_test"] == "eligible"]

class EligibleCalculator(BaseCalculator):
    @fit_method
    def fit(self, X, y, config):
        frame = X.to_native() if hasattr(X, "to_native") else X
        if MODE == "fit_drop":
            if isinstance(frame, pl.DataFrame):
                frame.drop_in_place("is_test")
            else:
                frame.drop(columns=["is_test"], inplace=True)
        if MODE == "fit_edit":
            frame["x"] = frame["x"] + 1
        return config

def build_pre_split_steps():
    return [
        {"name": "fixed_x", "transformer": "ValueReplacement",
         "params": {"columns": ["x"], "to_replace": 1.0, "value": 1.25}},
        {"name": "canonical_flag", "transformer": "TextCleaning",
         "params": {"columns": ["is_test"], "operations": [{"op": "trim"}]}},
        custom_step("eligible", EligibleCalculator, EligibleApplier,
                    pre_split=DECLARATION),
    ]

def build_preprocessing():
    return [{"name": "scale", "transformer": "StandardScaler",
             "params": {"columns": ["x"]}}]
"""


def _spec(*, steps: tuple[dict[str, Any], ...] = (), **changes: Any) -> training.LocalTrainingSpec:
    """Make model features distinct from eligibility and record identity fields."""
    values: dict[str, Any] = {
        "table": "workspace.test.labels",
        "version": 3,
        "record_key_columns": ("id",),
        "input_columns": ("x",),
        "target_column": "target",
        "max_rows": 30,
        "max_bytes": 100000,
        "pre_split_steps": steps,
    }
    values.update(changes)
    return training.LocalTrainingSpec(**values)


def _frame() -> pd.DataFrame:
    """Give the custom filter an extra source column with whitespace to normalize."""
    return pd.DataFrame(
        {
            "id": range(10),
            "x": [float(i + 1) for i in range(10)],
            "target": [float(2 * (i + 1)) for i in range(10)],
            "is_test": [" eligible ", " excluded ", *[" eligible "] * 8],
        }
    )


def _project(tmp_path: Any, source: str = SOURCE) -> dict[str, Any]:
    """Resolve the real project recipe from one generated preprocessing file."""
    path = tmp_path / "preprocessing.py"
    path.write_text(source, encoding="utf-8")
    return load_project_workflow(
        {"pipeline": {"preprocessing": [], "modeling": {"type": "linear_regression"}}},
        path,
    )


@pytest.mark.parametrize("engine", ["pandas", "polars"])
def test_fixed_normalization_then_custom_filter_keeps_raw_model_features(tmp_path, engine):
    """Eligibility sees normalized flags while model inputs remain raw and unsullied."""
    workflow = _project(tmp_path)
    spec = _spec(steps=tuple(workflow["pre_split_steps"]))
    train, holdout, _ = training.split_labeled_snapshot(_frame(), spec, engine=engine)
    assert spec.source_columns == ("id", "x", "target", "is_test")
    assert [step["transformer"] for step in workflow["pipeline"]["preprocessing"]] == [
        "StandardScaler"
    ]
    assert set(train.x) | set(holdout.x) == set(range(1, 11)) - {2}
    assert list(train.columns) == list(holdout.columns) == ["x", "target"]
    assert holdout.attrs["pre_split_filter_counts"][2]["excluded_rows"] == 1


def test_repeated_custom_factory_reuses_exact_registration_without_warning(caplog, monkeypatch):
    """Saved source replay must not overwrite its own registered class pair."""
    module = load_project_module(SOURCE)
    with caplog.at_level(logging.WARNING):
        first = module.build_pre_split_steps()
        second = module.build_pre_split_steps()
    identity = first[-1]["transformer"]
    assert first == second
    assert NodeRegistry.get_calculator(identity) is module.EligibleCalculator
    assert NodeRegistry.get_applier(identity) is module.EligibleApplier
    assert "re-registered" not in caplog.text
    monkeypatch.setitem(NodeRegistry._calculators, identity, object)
    with pytest.raises(ValueError, match="custom|registered|identity|conflict"):
        module.build_pre_split_steps()


@pytest.mark.parametrize(
    "declaration",
    [
        None,
        {},
        {"effect": "normalize", "required_columns": ["is_test"], "learns_from_data": False},
        {"effect": "filter", "required_columns": [], "learns_from_data": False},
        {"effect": "filter", "required_columns": ["is_test", "is_test"], "learns_from_data": False},
        {"effect": "filter", "required_columns": ["is_test", "IS_TEST"], "learns_from_data": False},
        {"effect": "filter", "required_columns": [""], "learns_from_data": False},
        {"effect": "filter", "required_columns": ["is_test", 1], "learns_from_data": False},
        {"effect": "filter", "required_columns": "is_test", "learns_from_data": False},
        {"effect": "filter", "required_columns": ["is_test"], "learns_from_data": True},
        {"effect": "filter", "required_columns": ["is_test"], "learns_from_data": 0},
        {
            "effect": "filter",
            "required_columns": ["is_test"],
            "learns_from_data": False,
            "extra": 1,
        },
    ],
)
def test_custom_eligibility_requires_exact_explicit_declaration(declaration):
    """A custom node cannot gain pre-split access through an incomplete assertion."""
    source = SOURCE.replace(
        'DECLARATION = {"effect": "filter", "required_columns": ["is_test"], "learns_from_data": False}',
        f"DECLARATION = {declaration!r}",
    )
    module = load_project_module(source)
    if declaration is None:
        with pytest.raises(ValueError, match="pre.split|custom|declaration"):
            _spec(steps=tuple(module.build_pre_split_steps()))
    else:
        with pytest.raises(ValueError, match="pre.split|custom|declaration|filter"):
            module.build_pre_split_steps()


@pytest.mark.parametrize("mode", ["edit", "reverse", "new_column", "wrong_type"])
@pytest.mark.parametrize("engine", ["pandas", "polars"])
def test_custom_filter_must_preserve_survivor_values_schema_and_order(tmp_path, mode, engine):
    """A declared filter cannot alter rows or columns under cover of eligibility."""
    workflow = _project(tmp_path, SOURCE.replace('MODE = "filter"', f'MODE = "{mode}"'))
    spec = _spec(steps=tuple(workflow["pre_split_steps"]))
    with pytest.raises(ValueError, match="pre_split_steps"):
        training.split_labeled_snapshot(_frame(), spec, engine=engine)


def test_custom_required_column_is_projected_without_a_fixed_step():
    """The custom declaration itself must pull its filter input from the source."""
    custom = load_project_module(SOURCE).build_pre_split_steps()[-1]
    spec = _spec(steps=(custom,))
    assert spec.source_columns == ("id", "x", "target", "is_test")


def test_unknown_custom_and_builtin_spoof_cannot_claim_filter_eligibility():
    """Only a registered project custom step can make the narrow filter claim."""
    custom = load_project_module(SOURCE).build_pre_split_steps()[-1].copy()
    custom.pop("pre_split")
    with pytest.raises(ValueError, match="pre_split_steps"):
        _spec(steps=(custom,))
    builtin = {
        "name": "spoof",
        "transformer": "StandardScaler",
        "params": {"columns": ["x"]},
        "pre_split": {
            "effect": "filter",
            "required_columns": ["x"],
            "learns_from_data": False,
        },
    }
    with pytest.raises(ValueError, match="pre_split_steps"):
        _spec(steps=(builtin,))


@pytest.mark.parametrize(
    "mode", ["inplace_drop", "inplace_add", "inplace_edit", "fit_drop", "fit_edit"]
)
def test_custom_fit_or_apply_cannot_mutate_input_frame_in_place(tmp_path, mode):
    """Schema and value guards compare against the frame as it was before custom code."""
    workflow = _project(tmp_path, SOURCE.replace('MODE = "filter"', f'MODE = "{mode}"'))
    with pytest.raises(ValueError, match="pre_split_steps|column|frame"):
        training.split_labeled_snapshot(
            _frame(), _spec(steps=tuple(workflow["pre_split_steps"])), engine="pandas"
        )


def test_custom_filter_rejects_nullable_value_edit_with_clear_error():
    """A nullable survivor value cannot turn into a non-null value silently."""
    source = SOURCE.replace('MODE = "filter"', 'MODE = "null_fill"')
    custom = load_project_module(source).build_pre_split_steps()[-1]
    frame = _frame()
    frame["x"] = frame["x"].astype(object)
    frame.at[0, "x"] = pd.NA
    with pytest.raises(ValueError, match="pre_split_steps changed undeclared column x"):
        training.split_labeled_snapshot(frame, _spec(steps=(custom,)))


@pytest.mark.parametrize("engine", ["pandas", "polars"])
def test_custom_filter_rejects_dtype_change_even_when_values_compare_equal(engine):
    """Filter-only custom code must retain each survivor column's schema."""
    source = SOURCE.replace('MODE = "filter"', 'MODE = "dtype"')
    custom = load_project_module(source).build_pre_split_steps()[-1]
    with pytest.raises(ValueError, match="pre_split_steps|schema|dtype"):
        training.split_labeled_snapshot(_frame(), _spec(steps=(custom,)), engine=engine)


@pytest.mark.parametrize(
    "keep,expected",
    [
        ("first", {0, 2, 4, 5, 6, 7, 8, 9}),
        ("last", {1, 3, 4, 5, 6, 7, 8, 9}),
        ("none", {4, 5, 6, 7, 8, 9}),
        (False, {4, 5, 6, 7, 8, 9}),
    ],
)
@pytest.mark.parametrize("engine", ["pandas", "polars"])
def test_deduplicate_admits_explicit_subset_and_keep_with_engine_parity(keep, expected, engine):
    """Stable input order fixes the survivor for each duplicate feature group."""
    frame = _frame().assign(
        x=[1.0, 1.0, 2.0, 2.0, *range(5, 11)], target=[2.0, 2.0, 4.0, 4.0, *range(10, 22, 2)]
    )
    step = {
        "name": "dedup",
        "transformer": "Deduplicate",
        "params": {"subset": ["x"], "keep": keep},
    }
    spec = _spec(steps=(step,))
    train, holdout, _ = training.split_labeled_snapshot(frame, spec, engine=engine)
    assert set(train.x) | set(holdout.x) == set(frame.loc[list(expected), "x"])
    assert holdout.attrs["survivor_key_sha256"] == training._key_digest(
        frame.loc[sorted(expected)], ("id",)
    )
    assert holdout.attrs["pre_split_filter_counts"][0]["excluded_rows"] == 10 - len(expected)


@pytest.mark.parametrize("engine", ["pandas", "polars"])
def test_deduplicate_rejects_conflicting_targets_before_selecting_survivor(engine):
    """An identical feature group with different labels has no valid representative."""
    frame = _frame().assign(x=[1.0, 1.0, *range(3, 11)])
    step = {
        "name": "dedup",
        "transformer": "Deduplicate",
        "params": {"subset": ["x"], "keep": "first"},
    }
    with pytest.raises(ValueError, match="conflict|target|label"):
        training.split_labeled_snapshot(frame, _spec(steps=(step,)), engine=engine)


def test_deduplicate_cannot_hide_duplicate_source_keys():
    """Deduplication must not turn an invalid source identity into valid training data."""
    frame = _frame()
    frame.loc[1, "id"] = frame.loc[0, "id"]
    step = {
        "name": "dedup",
        "transformer": "Deduplicate",
        "params": {"subset": ["x"], "keep": "first"},
    }
    with pytest.raises(ValueError, match="row keys must be unique"):
        training.split_labeled_snapshot(frame, _spec(steps=(step,)))


@pytest.mark.parametrize(
    "params",
    [
        {},
        {"subset": []},
        {"subset": ["x", "x"]},
        {"subset": ["x"], "keep": "middle"},
        {"subset": ["x"], "keep": True},
        {"subset": ["x"], "keep": "first", "unknown": 1},
    ],
)
def test_deduplicate_rejects_implicit_or_ambiguous_policy(params):
    """Deduplication must pin exact comparison columns and survivor policy."""
    with pytest.raises(ValueError, match="pre_split_steps|Deduplicate|subset|keep"):
        _spec(steps=({"name": "dedup", "transformer": "Deduplicate", "params": params},))


@pytest.mark.parametrize("engine", ["pandas", "polars"])
def test_deduplicate_groups_on_prior_normalized_values(engine):
    """Ordered normalization defines which rows are duplicate before splitting."""
    frame = _frame().assign(
        x=[" a ", "a", *[str(i) for i in range(2, 10)]],
        target=[2.0, 2.0, *[float(i * 2) for i in range(2, 10)]],
    )
    steps = (
        {
            "name": "trim",
            "transformer": "TextCleaning",
            "params": {"columns": ["x"], "operations": [{"op": "trim"}]},
        },
        {
            "name": "dedup",
            "transformer": "Deduplicate",
            "params": {"subset": ["x"], "keep": "first"},
        },
    )
    train, holdout, _ = training.split_labeled_snapshot(frame, _spec(steps=steps), engine=engine)
    assert len(train) + len(holdout) == 9
    assert " a " in set(train.x) | set(holdout.x)
    assert "a" not in set(train.x) | set(holdout.x)
    assert holdout.attrs["pre_split_filter_counts"][1]["excluded_rows"] == 1


def test_registered_custom_recipe_reloads_from_saved_source_in_fresh_process(tmp_path, monkeypatch):
    """Approval and score replay must work after project code changes on disk."""
    mlflow = pytest.importorskip("mlflow")
    from skyulf.inference.local_pipeline import load_local_pipeline
    from skyulf.integrations.databricks.local_approval import _load_evidence

    workflow = _project(tmp_path)
    spec = _spec(steps=tuple(workflow["pre_split_steps"]))
    monkeypatch.setattr(training, "read_training_snapshot", lambda spark, request: _frame())
    uri = f"sqlite:///{(tmp_path / 'tracking.db').as_posix()}"
    candidate = training.train_local_candidate(
        object(),
        spec,
        workflow["pipeline"],
        model_name="custom_eligibility",
        tracking_uri=uri,
        registry_uri=uri,
        experiment_name="custom_eligibility",
        run_name="combined_prefix",
        artifact_path=tmp_path / "artifact",
        metric="heldout_rmse",
        min_improvement=0.0,
        quality_threshold=100.0,
    )
    artifact = load_local_pipeline(tmp_path / "artifact")
    assert [step["transformer"] for step in artifact.pipeline.config["preprocessing"]] == [
        "ValueReplacement",
        "StandardScaler",
    ]
    assert "is_test" not in artifact.manifest.input_columns
    (tmp_path / "preprocessing.py").write_text(
        "raise RuntimeError('edited source must not run')", encoding="utf-8"
    )
    code = """
import json, sys, mlflow, pandas as pd
from skyulf.integrations.databricks.local_approval import _load_evidence
mlflow.set_tracking_uri(sys.argv[1])
mlflow.set_registry_uri(sys.argv[1])
client = mlflow.MlflowClient(tracking_uri=sys.argv[1], registry_uri=sys.argv[1])
report, spec, engine, evidence = _load_evidence(
    client, sys.argv[2], sys.argv[3], sys.argv[4], registry_uri=sys.argv[1]
)
model = mlflow.pyfunc.load_model(f"models:/{sys.argv[2]}/{sys.argv[3]}")
predicted = model.predict(pd.DataFrame({"x": [1.0, 11.0]}))
print(json.dumps({"engine": engine, "steps": len(spec.pre_split_steps),
                  "evidence": evidence is not None,
                  "predictions": predicted["prediction"].tolist()}))
"""
    result = subprocess.run(
        [
            sys.executable,
            "-c",
            code,
            uri,
            candidate.model_name,
            candidate.model_version,
            candidate.comparison_sha256,
        ],
        cwd=tmp_path,
        capture_output=True,
        text=True,
        timeout=90,
    )
    assert result.returncode == 0, result.stderr
    replay = json.loads(result.stdout)
    assert replay["engine"] == "pandas"
    assert replay["steps"] == 3 and replay["evidence"]
    assert len(replay["predictions"]) == 2

    client = mlflow.MlflowClient(tracking_uri=uri, registry_uri=uri)
    original = json.loads(
        Path(client.download_artifacts(candidate.run_id, "candidate_training_spec.json")).read_text(
            encoding="utf-8"
        )
    )
    original["pre_split_steps"][-1]["pre_split"]["required_columns"] = ["changed_flag"]
    client.log_dict(candidate.run_id, original, "candidate_training_spec.json")
    with pytest.raises(ValueError, match="pre.split|source|recipe|evidence|dataset"):
        _load_evidence(
            client,
            candidate.model_name,
            candidate.model_version,
            candidate.comparison_sha256,
            registry_uri=uri,
        )
