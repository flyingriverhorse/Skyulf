"""Review fixes for the admission probe and Spark admission of reused pre-split recipes."""

from copy import deepcopy
from dataclasses import asdict
from types import SimpleNamespace
from typing import Any, cast

import numpy as np
import pandas as pd
import polars as pl
import pytest

from skyulf.core.capabilities import UnsupportedExecutionError
from skyulf.data.dataset import SplitDataset
from skyulf.inference.partition_safety import require_partition_safe_pipeline
from skyulf.inference.pipeline_scoring import scoring_output_schema
from skyulf.inference.probed_rules import probe_rules
from skyulf.inference.project_code import fresh_project_module
from skyulf.integrations.databricks.projects.project import load_project_workflow
from skyulf.integrations.databricks.scoring.batch.frame_batch import fit_workflow
from skyulf.integrations.databricks.training.shared.preprocessing_checks import (
    seal_partition_probe,
)

TRAIN = pd.DataFrame({"x": np.linspace(1.0, 24.0, 24), "target": np.linspace(3.0, 49.0, 24)})
HOLDOUT = pd.DataFrame(
    {"x": [100.0 + ((7 * i) % 20) * 1.25 for i in range(20)], "target": np.arange(20.0)}
)


def _config(artifact) -> dict[str, Any]:
    """Return the pipeline config itself, typed for the keys training adds."""
    return cast(dict[str, Any], artifact.pipeline.config)


def _fit(tmp_path, source, train=TRAIN, *, engine="pandas", extra=None):
    """Load a one-file project package through the real loader and fit it."""
    root = tmp_path / "features"
    root.mkdir()
    exports = ["build_preprocessing", *(extra or [])]
    (root / "__init__.py").write_text(
        f"from .steps import {', '.join(exports)}\n", encoding="utf-8"
    )
    (root / "steps.py").write_text(source + f"\n# {tmp_path.name}\n", encoding="utf-8")
    base = {
        "pipeline": {"preprocessing": [], "modeling": {"type": "linear_regression"}},
        "target_column": "target",
        "input_columns": [c for c in train.columns if c != "target"],
        "engine": engine,
    }
    config = load_project_workflow(base, root)
    pipeline = config["pipeline"]
    pipeline["admit_probed_custom_steps"] = True
    rows = pl.from_pandas(train) if engine == "polars" else train
    path = tmp_path / "artifact"
    artifact = fit_workflow(
        pipeline,
        SplitDataset(train=rows, test=rows[:0]),
        target_column="target",
        artifact_path=path,
        max_rows=10000,
        max_bytes=10_000_000,
    )
    return artifact, path


COLUMN = """
import numpy as np
import pandas as pd
from skyulf.preprocessing import column_step, fitted_step


__BODY__


def build_preprocessing(recipe="default"):
    return [__STEP__]
"""


def _column(body, step='column_step("s", fn, output="z")'):
    """Render a one-step project source."""
    return COLUMN.replace("__BODY__", body).replace("__STEP__", step)


def test_date_text_step_passes_perturb_by_retrying_without_text(tmp_path):
    """A row-local parser may reject unknown text; perturb retries with text unchanged."""
    days = pd.date_range("2024-01-01", periods=24).strftime("%Y-%m-%d")
    train = pd.DataFrame({"ts": days, "target": np.arange(24.0)})
    holdout = pd.DataFrame(
        {"ts": pd.date_range("2025-01-01", periods=20).strftime("%Y-%m-%d"), "target": 0.0}
    )
    body = (
        'def fn(df):\n    return pd.to_datetime(df["ts"], format="%Y-%m-%d").astype("int64") / 1e9'
    )
    artifact, path = _fit(
        tmp_path, _column(body, 'column_step("s", fn, output="ts", replace=True)'), train
    )
    sealed = seal_partition_probe(artifact, holdout, path, enabled=True)
    assert require_partition_safe_pipeline(sealed).steps[0].action == "apply_probed"


def test_step_raising_on_a_changed_number_names_the_cause(tmp_path):
    """Raising on a valid-looking changed value is reported as such, not as row mixing."""
    body = (
        'def fn(df):\n    if (df["x"] > 140).any():\n        raise ValueError("too big")\n'
        '    return df["x"] * 2.0'
    )
    artifact, path = _fit(tmp_path, _column(body))
    with pytest.raises(
        ValueError, match=r"raised on a changed input value \(perturb: apply_error\)"
    ):
        seal_partition_probe(artifact, HOLDOUT, path, enabled=True)


def test_small_integer_columns_keep_their_dtype_under_perturb(tmp_path):
    """Perturbing a uint8 value near its maximum must not widen the column."""
    train = pd.DataFrame({"x": np.linspace(10, 200, 24).astype("uint8"), "target": np.arange(24.0)})
    holdout = pd.DataFrame({"x": np.linspace(10, 250, 20).astype("uint8"), "target": 0.0})
    body = 'def fn(df):\n    return df["x"].astype("float64") * 2.0'
    artifact, path = _fit(tmp_path, _column(body), train)
    sealed = seal_partition_probe(artifact, holdout, path, enabled=True)
    assert require_partition_safe_pipeline(sealed).steps[0].action == "apply_probed"


def test_fresh_module_catches_state_learned_into_a_class(tmp_path):
    """A value set on a class during learn() is missing in a fresh worker process."""
    body = (
        "class Cfg:\n    mean = 0.0\n\n\n"
        'def learn(df, y):\n    setattr(Cfg, "mean", float(df["x"].mean()))\n    return {}\n\n\n'
        'def apply(df, state):\n    return df["x"] - Cfg.mean'
    )
    step = 'fitted_step("s", learn, apply, output="z")'
    artifact, path = _fit(tmp_path, _column(body, step))
    with pytest.raises(
        ValueError, match=r"freshly loaded copy .*\(fresh_module: output_mismatch\)"
    ):
        seal_partition_probe(artifact, HOLDOUT, path, enabled=True)


def test_polars_fitted_models_are_not_probed(tmp_path):
    """Spark and endpoints reject polars-fitted models, so training skips the probe."""
    body = 'def fn(df):\n    return df["x"] * 2.0'
    artifact, path = _fit(tmp_path, _column(body), engine="polars")
    assert seal_partition_probe(artifact, pl.from_pandas(HOLDOUT), path, enabled=True) is artifact


RULES = """
import numpy as np
import pandas as pd
from skyulf.preprocessing import column_step, filter_step


def doubled(df):
    return df["x"] * 2.0


def keep(df):
    __FILTER__


def build_preprocessing(recipe="default"):
    return [column_step("s", doubled, output="z")]


def build_pre_split_steps():
    return [
        {"name": "complete", "transformer": "DropMissingRows", "params": {"subset": ["x"]}},
        filter_step("in_range", keep, columns=["x"]),
__EXTRA__
    ]


def build_scoring():
    return {"reuse_pre_split": True, "skip_target_steps": False}
"""


def _reuse(tmp_path, keep='return df["x"] < 1000.0', extra=""):
    """Fit a project whose scoring reuses its pre-split recipe."""
    source = RULES.replace("__FILTER__", keep).replace("__EXTRA__", extra)
    return _fit(tmp_path, source, extra=["build_pre_split_steps", "build_scoring"])


def test_reused_row_local_pre_split_is_admitted_with_evidence(tmp_path, monkeypatch):
    """Row-local filters reused at scoring are probed, sealed and certified."""
    artifact, path = _reuse(tmp_path, keep='return df["x"] < 120.0')
    sealed = seal_partition_probe(artifact, HOLDOUT, path, enabled=True)
    rules = _config(sealed)["partition_probe"]["rules"]
    assert [(rule["name"], rule["kind"]) for rule in rules] == [("pre_split", "pre_split")]
    steps = asdict(require_partition_safe_pipeline(sealed))["steps"]
    assert ("pre_split", "scoring_rule:pre_split", "rules_probed") in [
        (s["name"], s["node_type"], s["action"]) for s in steps
    ]
    names = [column.name for column in scoring_output_schema(sealed)]
    assert names[-2:] == ["scoring_status", "exclusion_reason"]
    _assert_spark_output_carries_reasons(sealed, path, monkeypatch)


def _assert_spark_output_carries_reasons(sealed, path, monkeypatch):
    """Worker output keeps pre-split exclusion reasons and marks the predicted rows' nulls."""
    pytest.importorskip("mlflow")
    from skyulf.integrations.mlflow.models.pipeline_model import SkyulfPipelinePythonModel
    from skyulf.integrations.mlflow.spark import spark_model
    from skyulf.integrations.mlflow.spark._spark_output import SPARK_NULL_TEXT

    model = SkyulfPipelinePythonModel(
        None, spark_model.partition_safety_certificate(sealed), spark_model.runtime_source_digest()
    )
    model.load_context(SimpleNamespace(artifacts={"local_pipeline": str(path)}))
    worker = model.predict(None, HOLDOUT.loc[:, ["x"]], params={"spark_output": True})
    excluded = worker["scoring_status"].eq("excluded")
    assert excluded.any() and (~excluded).any()
    assert worker.loc[excluded, "exclusion_reason"].eq("pre_split:in_range").all()
    assert worker.loc[~excluded, "exclusion_reason"].eq(SPARK_NULL_TEXT).all()


def test_reused_filter_reading_a_batch_statistic_fails_lint(tmp_path):
    """A reused filter comparing with the batch mean is named with its file and line."""
    artifact, path = _reuse(tmp_path, keep='return df["x"] < df["x"].mean()')
    with pytest.raises(ValueError, match=r"Pre-split filter 'in_range' reads a batch statistic"):
        seal_partition_probe(artifact, HOLDOUT, path, enabled=True)


def test_reused_deduplicate_is_rejected(tmp_path):
    """Deduplicate needs other rows, so reusing it on Spark is rejected with a clear fix."""
    extra = '        {"name": "dedup", "transformer": "Deduplicate", "params": {"subset": ["x"]}},'
    artifact, path = _reuse(tmp_path, extra=extra)
    sealed = seal_partition_probe(artifact, HOLDOUT, path, enabled=True)
    with pytest.raises(UnsupportedExecutionError, match="'dedup' is Deduplicate, which depends"):
        require_partition_safe_pipeline(sealed)


def test_changed_reused_steps_invalidate_evidence(tmp_path):
    """Evidence binds the saved pre-split steps; editing them is no longer admitted."""
    artifact, path = _reuse(tmp_path)
    sealed = seal_partition_probe(artifact, HOLDOUT, path, enabled=True)
    sealed.pipeline.config = deepcopy(sealed.pipeline.config)
    _config(sealed)["project_scoring"]["pre_split"]["steps"].pop(0)
    with pytest.raises(
        UnsupportedExecutionError, match="Reused pre-split eligibility: .*does not match"
    ):
        require_partition_safe_pipeline(sealed)


OUTPUT_RULES = """
import numpy as np
import pandas as pd
from skyulf.preprocessing import column_step


def doubled(df):
    return df["x"] * 2.0


def small(frame, params):
    return pd.Series(np.where(frame["x"] < params["limit"], "small", None), index=frame.index)


def band(frame, predictions, params):
    return pd.DataFrame({"band": np.where(predictions["prediction"] > 0, "high", "low")},
                        index=frame.index)


def build_preprocessing(recipe="default"):
    return [column_step("s", doubled, output="z")]


def build_scoring():
    return {
        "eligibility": [{"name": "small", "version": "1", "function": "steps.small",
                         "params": {"limit": __LIMIT__}}],
        "outputs": [{"name": "band", "version": "1", "function": "steps.band", "params": {},
                     "columns": [{"name": "band", "dtype": "string"}]}],
    }
"""


@pytest.mark.parametrize(("limit", "admitted"), [(280.0, True), (297.0, False)])
def test_output_rules_probe_eligible_rows_from_the_whole_holdout(tmp_path, limit, admitted):
    """Output rules use all eligible rows after a representative raw probe clears its guard."""
    order = np.r_[np.arange(280.0, 296.0), np.arange(280.0), np.arange(296.0, 300.0)]
    holdout = pd.DataFrame({"x": order, "target": 0.0})
    artifact, path = _fit(
        tmp_path, OUTPUT_RULES.replace("__LIMIT__", str(limit)), extra=["build_scoring"]
    )
    if admitted:
        sealed = seal_partition_probe(artifact, holdout, path, enabled=True)
        probe = _config(sealed)["partition_probe"]
        assert [r["name"] for r in probe["rules"]] == ["small", "band"]
        assert probe["pipeline"]["rows"] == 256
        assert probe["pipeline"]["scored_rows"] == 16
    else:
        with pytest.raises(ValueError, match=r"Only 3 holdout rows reach this rule; widen"):
            seal_partition_probe(artifact, holdout, path, enabled=True)


@pytest.mark.parametrize("leading_eligible", [0, 15])
def test_raw_pipeline_probe_rejects_too_few_predictions_in_first_256(tmp_path, leading_eligible):
    """Eligible rows later in the holdout cannot conceal an unrepresentative raw probe sample."""
    eligible = np.arange(280.0, 300.0)
    order = np.r_[eligible[:leading_eligible], np.arange(280.0), eligible[leading_eligible:]]
    holdout = pd.DataFrame({"x": order, "target": 0.0})
    artifact, path = _fit(
        tmp_path, OUTPUT_RULES.replace("__LIMIT__", "280.0"), extra=["build_scoring"]
    )
    with fresh_project_module(_config(artifact)["project_python_source"]) as fresh:
        details = probe_rules(
            artifact, holdout.loc[:, list(artifact.manifest.input_columns)], fresh
        )
    assert {rule["name"]: rule["status"] for rule in details} == {
        "small": "passed",
        "band": "passed",
    }
    with pytest.raises(ValueError, match="Raw pipeline partition probe needs at least 16"):
        seal_partition_probe(artifact, holdout, path, enabled=True)


PACKAGE_INIT = """
from .scoring import build_scoring


def build_preprocessing(recipe="default"):
    from .preprocessing import centred

    return [centred()]


def build_pre_split_steps():
    # Imported only when the builder runs, as in generated multi-file projects.
    from .pre_split import keep_range

    return [keep_range()]
"""
PACKAGE_FILES = {
    "pre_split.py": """
from skyulf.preprocessing import filter_step


def in_range(df, params):
    return df["x"].between(params["low"], params["high"]) | df["x"].isna()


def keep_range():
    return filter_step("x_range", in_range, columns=["x"], params={"low": 0.0, "high": 500.0})
""",
    "preprocessing.py": """
import numpy as np
from skyulf.preprocessing import fitted_step


def learn(df, y):
    return {"cuts": [float(v) for v in np.quantile(df["x"], [0.25, 0.5, 0.75])]}


def bucket(df, state):
    return np.searchsorted(state["cuts"], df["x"].to_numpy()).astype("float64")


def centred():
    return fitted_step("bucket", learn, bucket, output="x_bucket")
""",
    "scoring.py": """
import pandas as pd


def known(frame, params):
    return pd.Series([None] * len(frame.index), index=frame.index, dtype=object)


def build_scoring():
    return {
        "reuse_pre_split": True,
        "skip_target_steps": False,
        "eligibility": [{"name": "known", "version": "1", "function": "scoring.known",
                         "params": {}}],
        "outputs": [],
    }
""",
}


def test_fresh_module_works_for_multi_file_project_packages(tmp_path):
    """Lazily imported submodules (pre_split.py) must load in the fresh copy too."""
    root = tmp_path / "features"
    root.mkdir()
    (root / "__init__.py").write_text(PACKAGE_INIT, encoding="utf-8")
    for name, text in PACKAGE_FILES.items():
        (root / name).write_text(text + f"\n# {tmp_path.name}\n", encoding="utf-8")
    base = {
        "pipeline": {"preprocessing": [], "modeling": {"type": "linear_regression"}},
        "target_column": "target",
        "input_columns": ["x"],
        "engine": "pandas",
    }
    pipeline = load_project_workflow(base, root)["pipeline"]
    pipeline["admit_probed_custom_steps"] = True
    path = tmp_path / "artifact"
    artifact = fit_workflow(
        pipeline,
        SplitDataset(train=TRAIN, test=TRAIN[:0]),
        target_column="target",
        artifact_path=path,
        max_rows=10000,
        max_bytes=10_000_000,
    )
    sealed = seal_partition_probe(artifact, HOLDOUT, path, enabled=True)
    rules = [rule["name"] for rule in _config(sealed)["partition_probe"]["rules"]]
    assert rules == ["pre_split", "known"]
    actions = [step.action for step in require_partition_safe_pipeline(sealed).steps]
    assert actions == ["apply_probed", "rules_probed", "rules_probed"]


def test_fresh_module_exception_is_reported_with_type_and_message():
    """A failure in the fresh copy says the fresh code raised, with the exception."""
    from skyulf.inference.probed_steps import REQUIRED_CHECKS, require_passed

    checks = [{"name": name, "status": "passed"} for name in REQUIRED_CHECKS[:-1]]
    checks.append(
        {
            "name": "fresh_module",
            "status": "failed",
            "reason": "apply_error",
            "error_type": "ValueError",
            "error_message": "boom",
        }
    )
    with pytest.raises(
        ValueError,
        match=r"raised in a freshly loaded copy of the project code \(fresh_module: "
        r"ValueError: boom\)",
    ):
        require_passed("Custom step 's'", {"checks": checks}, {})
