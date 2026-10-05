"""Pre-split reuse must preserve recipe semantics and explicit prediction outcomes."""

import re
import shutil
from pathlib import Path

import pandas as pd
import polars as pl
import pytest

from skyulf.inference.project_scoring import run_project_scoring
from skyulf.integrations.databricks.projects.project import load_project_workflow

TEMPLATE = (
    Path(__file__).resolve().parents[3]
    / "templates/databricks/template/{{.project_name}}/src/features"
)


def _project(tmp_path, *, skip=False, mode="pre_split", steps=None, engine="pandas", examples=True):
    """Exercise actual generated switches with an editable pre-split recipe."""
    root = tmp_path / "features"
    shutil.copytree(TEMPLATE, root)
    scoring = root / "scoring.py"
    source = scoring.read_text(encoding="utf-8")
    if skip:
        source = source.replace(
            "SKIP_TARGET_PRE_SPLIT_STEPS = False", "SKIP_TARGET_PRE_SPLIT_STEPS = True"
        )
    source = source.replace('SCORING_MODE = "pre_split"', f"SCORING_MODE = {mode!r}")
    if examples:
        source = re.sub(r'(?m)^(        )# (?=[{}" ])', r"\1", source)
    scoring.write_text(source, encoding="utf-8")
    if steps is not None:
        (root / "pre_split.py").write_text(
            f"def build_pre_split_steps():\n    return {steps!r}\n", encoding="utf-8"
        )
    config = {
        "engine": engine,
        "target_column": "target",
        "input_columns": ["feature_value"],
        "pipeline": {"preprocessing": [], "modeling": {"type": "linear_regression"}},
    }
    return load_project_workflow(config, root)


@pytest.mark.parametrize("mode", ["custom", "combined"])
def test_template_custom_examples_are_inactive_by_default(tmp_path, mode):
    """Choosing a mode alone must not activate sample business limits or output columns."""
    pipeline = _project(tmp_path, mode=mode, examples=False)["pipeline"]
    policy = pipeline["project_scoring"]
    rows = pd.DataFrame({"feature_value": [-1.0, 121.0]})
    result = run_project_scoring(
        rows,
        lambda frame: pd.DataFrame({"prediction": frame.feature_value}),
        source=pipeline["project_python_source"],
        config=policy,
        row_keys=[],
        prediction_dtypes={"prediction": "float64"},
    )
    assert policy["eligibility"] == policy["outputs"] == []
    assert result.prediction.tolist() == [-1.0, 121.0]
    assert result.scoring_status.tolist() == ["predicted", "predicted"]
    assert "band" not in result


def _missing(column, name="observed"):
    """Build the existing Core missing-value filter without a second scoring callback."""
    return {
        "name": name,
        "transformer": "DropMissingRows",
        "params": {"subset": [column], "how": "any"},
    }


@pytest.mark.parametrize("engine", ["pandas", "polars"])
def test_reuse_skips_only_target_filters_and_preserves_input_positions(tmp_path, engine):
    """Turning off the target rule must retain feature filtering and every output record."""
    config = _project(
        tmp_path,
        skip=True,
        steps=[_missing("target", "labels"), _missing("feature_value")],
        engine=engine,
    )
    pipeline = config["pipeline"]
    policy = pipeline["project_scoring"]
    assert policy["pre_split"]["skipped_target_steps"] == ["labels"]
    rows = pd.DataFrame({"feature_value": [None, 2.0, 7.0]}, index=[8, 8, 1])
    native = pl.from_pandas(rows) if engine == "polars" else rows
    result = run_project_scoring(
        native,
        lambda frame: pd.DataFrame({"prediction": frame["feature_value"].to_list()}),
        source=pipeline["project_python_source"],
        config=policy,
        row_keys=[],
        prediction_dtypes={"prediction": "float64"},
    )
    assert result.scoring_status.tolist() == ["excluded", "predicted", "predicted"]
    assert result.exclusion_reason.iloc[0] == "pre_split:observed"
    assert result.prediction.iloc[1:].tolist() == [2.0, 7.0]
    if engine == "pandas":
        assert result.index.tolist() == [8, 8, 1]


def test_target_reuse_requires_explicit_skip_switch(tmp_path):
    """A target-dependent recipe must not silently remove unlabeled scoring rows."""
    with pytest.raises(ValueError, match="SKIP_TARGET_PRE_SPLIT_STEPS"):
        _project(tmp_path, steps=[_missing("target")])


def test_custom_switch_ignores_pre_split_filters(tmp_path):
    """Custom scoring must replace reuse rather than combine two independent filters."""
    pipeline = _project(tmp_path, mode="custom", steps=[_missing("target")])["pipeline"]
    policy = pipeline["project_scoring"]
    assert "pre_split" not in policy
    assert len(policy["eligibility"]) == 2
    assert len(policy["outputs"]) == 1


def test_reuse_rejects_missing_scoring_input_at_configuration_time(tmp_path):
    """Training-only source fields require an explicit scoring input contract."""
    with pytest.raises(ValueError, match="input_columns"):
        _project(tmp_path, steps=[_missing("age")])


@pytest.mark.parametrize("engine", ["pandas", "polars"])
def test_reuse_normalizes_for_selection_but_model_receives_original_values(tmp_path, engine):
    """Fixed edits may decide membership but must not be applied twice to prediction inputs."""
    steps = [
        {
            "name": "replace",
            "transformer": "ValueReplacement",
            "params": {"columns": ["feature_value"], "replacements": [{"old": -1.0, "new": None}]},
        },
        _missing("feature_value"),
    ]
    pipeline = _project(tmp_path, steps=steps, engine=engine)["pipeline"]
    rows = pd.DataFrame({"feature_value": [-1.0, 4.0]})
    result = run_project_scoring(
        pl.from_pandas(rows) if engine == "polars" else rows,
        lambda frame: pd.DataFrame({"prediction": frame["feature_value"].to_list()}),
        source=pipeline["project_python_source"],
        config=pipeline["project_scoring"],
        row_keys=[],
        prediction_dtypes={"prediction": "float64"},
    )
    assert result.scoring_status.tolist() == ["excluded", "predicted"]
    assert result.prediction.iloc[1] == 4.0


@pytest.mark.parametrize("engine", ["pandas", "polars"])
def test_reuse_custom_filter_all_excluded_and_saved_source(tmp_path, engine):
    """The saved project filter works without original files and never fits an empty model batch."""
    config = _project(tmp_path, engine=engine)
    root = tmp_path / "features"
    (root / "pre_split.py").write_text(
        "from .custom.pre_split_custom import minimum_completeness\n"
        "def build_pre_split_steps():\n"
        "    return [minimum_completeness(['feature_value'])]\n",
        encoding="utf-8",
    )
    config["pre_split_steps"] = []
    config["pipeline"]["preprocessing"] = []
    pipeline = load_project_workflow(config, root)["pipeline"]
    shutil.rmtree(root)
    rows = pd.DataFrame({"feature_value": [float("nan"), float("nan")]})

    def predict(frame):
        """A batch excluded by the reused filter must not reach model execution."""
        pytest.fail("model called for an all-excluded batch")

    result = run_project_scoring(
        pl.from_pandas(rows) if engine == "polars" else rows,
        predict,
        source=pipeline["project_python_source"],
        config=pipeline["project_scoring"],
        row_keys=[],
        prediction_dtypes={"prediction": "float64"},
    )
    assert result.scoring_status.tolist() == ["excluded", "excluded"]
    assert result.exclusion_reason.tolist() == ["pre_split:minimum_completeness"] * 2


@pytest.mark.parametrize("engine", ["pandas", "polars"])
@pytest.mark.parametrize("kind", ["ManualBounds", "Deduplicate"])
def test_reuse_core_filters_apply_original_survivor_policy(tmp_path, engine, kind):
    """Bounds and duplicate decisions must use Core and report the removed source positions."""
    params = (
        {"bounds": {"feature_value": {"lower": 2.0, "upper": 4.0}}}
        if kind == "ManualBounds"
        else {"subset": ["feature_value"], "keep": "first"}
    )
    pipeline = _project(
        tmp_path,
        engine=engine,
        steps=[
            {
                "name": "selected",
                "transformer": kind,
                "params": params,
            }
        ],
    )["pipeline"]
    rows = pd.DataFrame({"feature_value": [1.0, 3.0, 3.0, 5.0]})
    result = run_project_scoring(
        pl.from_pandas(rows) if engine == "polars" else rows,
        lambda frame: pd.DataFrame({"prediction": frame["feature_value"].to_list()}),
        source=pipeline["project_python_source"],
        config=pipeline["project_scoring"],
        row_keys=[],
        prediction_dtypes={"prediction": "float64"},
    )
    expected = (
        ["excluded", "predicted", "predicted", "excluded"]
        if kind == "ManualBounds"
        else ["predicted", "predicted", "excluded", "predicted"]
    )
    assert result.scoring_status.tolist() == expected


def test_target_skip_does_not_partially_rewrite_a_mixed_filter(tmp_path):
    """Removing one field from a completeness rule could change its meaning."""
    step = _missing("target")
    step["params"]["subset"] = ["target", "feature_value"]
    policy = _project(tmp_path, skip=True, steps=[step])["pipeline"]["project_scoring"]
    assert policy["pre_split"]["steps"] == []
    assert policy["pre_split"]["skipped_target_steps"] == ["observed"]


@pytest.mark.parametrize("engine", ["pandas", "polars"])
@pytest.mark.parametrize("mode", ["pre_split", "combined"])
def test_saved_reused_custom_filter_loads_in_fresh_process(tmp_path, engine, mode):
    """Scoring must restore custom registration and saved parameters without the project folder."""
    import json
    import subprocess
    import sys

    from skyulf.data.dataset import SplitDataset
    from skyulf.integrations.databricks.scoring.batch.local_batch import fit_local_workflow

    config = _project(tmp_path, engine=engine, mode=mode)
    root = tmp_path / "features"
    (root / "pre_split.py").write_text(
        "from .custom.pre_split_custom import minimum_completeness\n"
        "def build_pre_split_steps():\n"
        "    return [minimum_completeness(['feature_value'])]\n",
        encoding="utf-8",
    )
    pipeline = load_project_workflow(config, root)["pipeline"]
    train = pd.DataFrame({"feature_value": [1.0, 2.0, 3.0, 4.0], "target": [3.0, 5.0, 7.0, 9.0]})
    native = pl.from_pandas(train) if engine == "polars" else train
    path = tmp_path / "artifact"
    fit_local_workflow(
        pipeline,
        SplitDataset(train=native, test=native[:0]),
        target_column="target",
        artifact_path=path,
        max_rows=10,
        max_bytes=10000,
    )
    shutil.rmtree(root)
    code = (
        "import json,sys,pandas as pd\n"
        "from skyulf.inference.local_pipeline import load_local_pipeline\n"
        "from skyulf.inference.local_scoring import score_local_pipeline\n"
        "model=load_local_pipeline(sys.argv[1])\n"
        "result=score_local_pipeline(pd.DataFrame({'feature_value':[None,5.0]}),model)\n"
        "assert result.scoring_status.tolist()==['excluded','predicted']\n"
        "assert result.exclusion_reason.iloc[0]=='pre_split:minimum_completeness'\n"
        "assert abs(result.prediction.iloc[1]-11.0)<1e-8\n"
        "if sys.argv[2] == 'combined': assert result.band.iloc[1] == 'medium'\n"
        "print(json.dumps({'passed':True}))\n"
    )
    child = subprocess.run(
        [sys.executable, "-I", "-c", code, str(path), mode],
        cwd=tmp_path,
        capture_output=True,
        text=True,
        timeout=60,
    )
    assert child.returncode == 0, child.stderr
    assert json.loads(child.stdout) == {"passed": True}


@pytest.mark.parametrize("engine", ["pandas", "polars"])
def test_reuse_fixed_changes_do_not_double_transform_saved_model(tmp_path, engine):
    """The real candidate recipe applies a non-idempotent replacement exactly once per prediction."""
    from skyulf.data.dataset import SplitDataset
    from skyulf.inference.local_scoring import score_local_pipeline
    from skyulf.integrations.databricks.scoring.batch.local_batch import fit_local_workflow
    from skyulf.integrations.databricks.training.fitting.local_retraining import (
        LocalTrainingSpec,
        candidate_config,
    )
    from skyulf.integrations.databricks.training.tuning.local_cv import LocalCVSpec

    steps = [
        {
            "name": "replace",
            "transformer": "ValueReplacement",
            "params": {
                "columns": ["feature_value"],
                "replacements": [{"old": 1.0, "new": 2.0}, {"old": 2.0, "new": 3.0}],
            },
        },
        _missing("feature_value"),
    ]
    config = _project(tmp_path, engine=engine, steps=steps)
    spec = LocalTrainingSpec(
        table="a.b.c",
        version=0,
        record_key_columns=("id",),
        input_columns=("feature_value",),
        target_column="target",
        max_rows=10,
        max_bytes=10000,
        pre_split_steps=tuple(config["pre_split_steps"]),
    )
    pipeline = candidate_config(
        spec,
        config["pipeline"],
        engine=engine,
        cv=LocalCVSpec(),
        metric="heldout_rmse",
        min_improvement=0.0,
        champion_version=None,
        quality_threshold=100.0,
        risk_category=None,
    )
    train = pd.DataFrame({"feature_value": [4.0, 5.0, 6.0, 7.0], "target": [8.0, 10.0, 12.0, 14.0]})
    native = pl.from_pandas(train) if engine == "polars" else train
    artifact = fit_local_workflow(
        pipeline,
        SplitDataset(train=native, test=native[:0]),
        target_column="target",
        artifact_path=tmp_path / "artifact",
        max_rows=10,
        max_bytes=10000,
    )
    result = score_local_pipeline(pd.DataFrame({"feature_value": [1.0]}), artifact)
    assert result.prediction.iloc[0] == pytest.approx(4.0)


@pytest.mark.parametrize("engine", ["pandas", "polars"])
def test_combined_scoring_orders_filters_and_preserves_first_reason(tmp_path, engine):
    """Pre-split exclusions must precede custom eligibility and retain their own reasons."""
    steps = [
        {
            "name": "positive",
            "transformer": "ManualBounds",
            "params": {"bounds": {"feature_value": {"lower": 0.0}}},
        }
    ]
    config = _project(
        tmp_path,
        mode="combined",
        engine=engine,
        skip=True,
        steps=[_missing("target", "labels"), *steps],
    )
    root = tmp_path / "features"
    custom = root / "custom/scoring_custom.py"
    source = custom.read_text(encoding="utf-8").replace(
        '    columns = params["columns"]',
        '    assert (frame.feature_value >= 0).all(), "pre-split must run first"\n'
        "    assert frame.index.equals(pd.RangeIndex(len(frame)))\n"
        '    columns = params["columns"]',
    )
    custom.write_text(source, encoding="utf-8")
    config["pre_split_steps"] = []
    pipeline = load_project_workflow(config, root)["pipeline"]
    policy = pipeline["project_scoring"]
    assert policy["pre_split"]["skipped_target_steps"] == ["labels"]
    rows = pd.DataFrame(
        {"record_id": [101, 102, 103, 104], "feature_value": [-1.0, 10.0, 121.0, 70.0]},
        index=[4, 4, 0, 8],
    )

    def predict(frame):
        """Only survivors of both filters can reach the model."""
        values = frame["feature_value"].to_list()
        assert values == [10.0, 70.0]
        return pd.DataFrame({"prediction": values})

    result = run_project_scoring(
        pl.from_pandas(rows) if engine == "polars" else rows,
        predict,
        source=pipeline["project_python_source"],
        config=policy,
        row_keys=["record_id"],
        prediction_dtypes={"prediction": "float64"},
    )
    assert result.record_id.tolist() == [101, 102, 103, 104]
    assert result.scoring_status.tolist() == ["excluded", "predicted", "excluded", "predicted"]
    assert result.exclusion_reason.iloc[0] == "pre_split:positive"
    assert result.exclusion_reason.iloc[2] == "outside_range:feature_value"
    assert result.band.dropna().tolist() == ["medium", "high"]
    assert result.prediction.iloc[0:1].isna().all()
    if engine == "pandas":
        assert result.index.tolist() == [4, 4, 0, 8]


@pytest.mark.parametrize("engine", ["pandas", "polars"])
def test_combined_all_excluded_skips_custom_callbacks_and_model(tmp_path, engine):
    """No custom rule or model should run when pre-split already excludes the batch."""
    config = _project(tmp_path, mode="combined", engine=engine, steps=[_missing("feature_value")])
    root = tmp_path / "features"
    custom = root / "custom/scoring_custom.py"
    source = custom.read_text(encoding="utf-8").replace(
        '    columns = params["columns"]', '    raise AssertionError("custom callback ran")'
    )
    custom.write_text(source, encoding="utf-8")
    config["pre_split_steps"] = []
    pipeline = load_project_workflow(config, root)["pipeline"]
    rows = pd.DataFrame({"feature_value": [float("nan"), float("nan")]})

    def predict(frame):
        """All-excluded batches do not need predictions."""
        pytest.fail("model ran")

    result = run_project_scoring(
        pl.from_pandas(rows) if engine == "polars" else rows,
        predict,
        source=pipeline["project_python_source"],
        config=pipeline["project_scoring"],
        row_keys=[],
        prediction_dtypes={"prediction": "float64"},
    )
    assert result.scoring_status.tolist() == ["excluded", "excluded"]
    assert result.band.isna().all()


def test_combined_without_pre_split_keeps_custom_rules(tmp_path):
    """An empty shared recipe must not silently disable custom eligibility or outputs."""
    policy = _project(tmp_path, mode="combined")["pipeline"]["project_scoring"]
    assert len(policy["eligibility"]) == 2
    assert len(policy["outputs"]) == 1
    assert "pre_split" not in policy


def test_combined_requires_target_skip_when_needed(tmp_path):
    """Combined mode must enforce the same target-availability guard as pure reuse."""
    with pytest.raises(ValueError, match="SKIP_TARGET_PRE_SPLIT_STEPS"):
        _project(tmp_path, mode="combined", steps=[_missing("target")])


@pytest.mark.parametrize("mode", ["typo", "", None, True])
def test_invalid_scoring_mode_fails_before_training(tmp_path, mode):
    """Typos must not silently select a different prediction policy."""
    with pytest.raises(ValueError, match="SCORING_MODE"):
        _project(tmp_path, mode=mode)
