"""Project-owned custom steps must demonstrate distinct training and feature contracts."""

import json
import shutil
import subprocess
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import polars as pl
import pytest

from skyulf.inference.project_code import load_project_module

CUSTOM = (
    Path(__file__).resolve().parents[3]
    / "templates/databricks/template/{{.project_name}}/src/features/custom"
)


def _custom_module(filename):
    """Load exactly the source shipped to users with isolated custom registrations."""
    return load_project_module((CUSTOM / filename).read_text(encoding="utf-8"))


def _native(frame, engine):
    """Exercise each native engine without changing the source row order."""
    return pl.from_pandas(frame) if engine == "polars" else frame


def _fit_apply(step, train, score):
    """Fit a step on training rows and apply it, exactly as the pipeline resolves it."""
    from skyulf.registry import NodeRegistry

    state = NodeRegistry.get_calculator(step["transformer"])().fit(train, step["params"])
    return state, NodeRegistry.get_applier(step["transformer"])().apply(score, state)


@pytest.mark.parametrize("engine", ["pandas", "polars"])
def test_completeness_preserves_rows_with_enough_observed_fields(engine):
    """The general filter counts only selected fields and preserves survivor values/order."""
    module = _custom_module("pre_split_custom.py")
    rows = pd.DataFrame(
        {
            "a": [1.0, np.nan, 2.0, np.nan],
            "b": [None, "x", "y", None],
            "c": [3.0, 4.0, np.nan, np.nan],
            "untouched": [9, 8, 7, 6],
        },
        index=[8, 3, 5, 1],
    )
    source = _native(rows, engine)
    step = module.minimum_completeness(["a", "b", "c"], min_present=2)
    state, result = _fit_apply(step, source, source)
    assert state == step["params"]
    result = result.to_pandas() if engine == "polars" else result
    pd.testing.assert_frame_equal(
        result.reset_index(drop=True), rows.iloc[:3].reset_index(drop=True)
    )
    if engine == "pandas":
        assert list(result.index) == [8, 3, 5]
    pd.testing.assert_frame_equal(
        source.to_pandas() if engine == "polars" else source.reset_index(drop=True),
        rows.reset_index(drop=True),
    )
    assert step["pre_split"] == {
        "effect": "filter",
        "required_columns": ["a", "b", "c"],
        "learns_from_data": False,
    }


@pytest.mark.parametrize("engine", ["pandas", "polars"])
def test_template_range_and_allowed_filters_keep_expected_rows(engine):
    """The beginner filter examples must keep exactly the documented rows."""
    module = _custom_module("pre_split_custom.py")
    rows = pd.DataFrame({"age": [5.0, -1.0, None, 130.0], "country": ["NL", "FR", "DE", None]})
    _, kept = _fit_apply(
        module.value_range("age", 0, 120), _native(rows, engine), _native(rows, engine)
    )
    kept = kept.to_pandas() if engine == "polars" else kept
    assert kept["country"].tolist() == ["NL", "DE"]
    allowed = module.allowed_values("country", ["NL", "DE"])
    _, kept = _fit_apply(allowed, _native(rows, engine), _native(rows, engine))
    kept = kept.to_pandas() if engine == "polars" else kept
    assert kept["country"].tolist() == ["NL", "DE"]


def _check_frequency_scoring(result, score, engine):
    """Saved frequencies map A/B, give unseen and null zero and keep other columns/order."""
    result = result.to_pandas() if engine == "polars" else result
    assert result.category.tolist() == [0.5, 0.0, 0.0, 0.25]
    assert result.other.tolist() == [4, 3, 2, 1]
    if engine == "pandas":
        assert result.index.equals(score.index)


_FREQ_TRAIN = pd.DataFrame({"category": ["A", "A", "B", None], "other": [1, 2, 3, 4]})
_FREQ_SCORE = pd.DataFrame(
    {"category": ["A", "NEW", None, "B"], "other": [4, 3, 2, 1]}, index=[8, 3, 5, 1]
)


@pytest.mark.parametrize("engine", ["pandas", "polars"])
def test_frequency_encoding_reuses_training_mapping_and_handles_unseen(engine):
    """Score batch frequencies must never replace the saved training distribution."""
    module = _custom_module("preprocessing_custom.py")
    step = module.frequency_encoding(["category"])
    state, result = _fit_apply(step, _native(_FREQ_TRAIN, engine), _native(_FREQ_SCORE, engine))
    assert state["state"] == {"category": {"A": 0.5, "B": 0.25}}
    _check_frequency_scoring(result, _FREQ_SCORE, engine)
    _, empty = _fit_apply(step, _native(_FREQ_TRAIN, engine), _native(_FREQ_SCORE.iloc[:0], engine))
    assert len(empty) == 0 and list(empty.columns) == list(_FREQ_SCORE.columns)


_RARE_TRAIN = pd.DataFrame({"city": ["A"] * 6 + ["B"] * 3 + ["C"], "other": range(10)})
_RARE_SCORE = pd.DataFrame({"city": ["A", "C", "NEW", None, "B"]}, index=[7, 3, 9, 1, 0])


@pytest.mark.parametrize("categorical", [False, True])
@pytest.mark.parametrize("engine", ["pandas", "polars"])
def test_rare_categories_function_and_class_versions_match(engine, categorical):
    """The learn/apply example and its Calculator/Applier twin must produce the same column."""
    train, score = _RARE_TRAIN.copy(), _RARE_SCORE.copy()
    if categorical:  # category dtype must still accept the new "Other" value
        train["city"], score["city"] = train.city.astype("category"), score.city.astype("category")
    function_step = _custom_module("preprocessing_custom.py").rare_categories("city", 0.2)
    class_step = _custom_module("advanced_class_step.py").class_rare_categories("city", 0.2)
    for step in (function_step, class_step):
        state, result = _fit_apply(step, _native(train, engine), _native(score, engine))
        learned = state.get("state", state)  # fitted_step nests what learn() returned
        assert learned["keep"] == ["A", "B"]
        result = result.to_pandas() if engine == "polars" else result
        values = [None if pd.isna(value) else value for value in result["city"]]
        assert values == ["A", "Other", "Other", None, "B"]
        assert list(result.columns) == ["city"]
        if engine == "pandas":
            assert result.index.equals(score.index)


def test_rare_categories_rejects_invalid_share():
    """A share outside (0, 1) is a typo (5 instead of 0.05) and must fail early."""
    for filename, factory in [
        ("preprocessing_custom.py", "rare_categories"),
        ("advanced_class_step.py", "class_rare_categories"),
    ]:
        with pytest.raises(ValueError, match="min_share"):
            getattr(_custom_module(filename), factory)("city", min_share=5)


@pytest.mark.parametrize("engine", ["pandas", "polars"])
def test_template_log_feature_example(engine):
    """log_feature adds a stateless column and keeps the original one."""
    module = _custom_module("preprocessing_custom.py")
    frame = _native(pd.DataFrame({"value": [0.0, 1.0, -3.0]}), engine)
    _, logged = _fit_apply(module.log_feature("value"), frame, frame)
    logged = logged.to_pandas() if engine == "polars" else logged
    np.testing.assert_allclose(logged["log_value"], np.log1p([0.0, 1.0, 0.0]))
    assert logged["value"].tolist() == [0.0, 1.0, -3.0]


def _uncomment_asset_example(path):
    """Activate the commented Example 4 block exactly as a user would."""
    source = path.read_text(encoding="utf-8")
    head, _, tail = source.partition("# import json")
    lines = ("# import json" + tail).splitlines()
    active = [line[2:] if line.startswith("# ") else line.lstrip("#") for line in lines]
    path.write_text(head + "\n".join(active) + "\n", encoding="utf-8")


def _enable_asset_examples(root):
    """Create both asset files, declare them and switch on both commented examples."""
    (root / "assets").mkdir()
    (root / "assets/city_region.json").write_text('{"London": "UK", "Vilnius": "LT"}')
    (root / "assets/countries.json").write_text('["NL", "DE", "BE"]')
    manifest = json.loads((root / "assets.json").read_text(encoding="utf-8"))
    manifest["files"] = ["assets/city_region.json", "assets/countries.json"]
    (root / "assets.json").write_text(json.dumps(manifest), encoding="utf-8")
    for name, factory in (("preprocessing", "city_region"), ("pre_split", "allowed_countries")):
        _uncomment_asset_example(root / "custom" / f"{name}_custom.py")
        path = root / f"{name}.py"
        source = path.read_text(encoding="utf-8")
        line = f"from .custom.{name}_custom import {factory}"
        source = source.replace(f"# {line}", line).replace(f"# {factory}(),", f"{factory}(),")
        path.write_text(source, encoding="utf-8")


def test_commented_asset_examples_work_when_enabled(tmp_path):
    """The inactive asset examples must run once a user follows their three steps."""
    from skyulf.integrations.databricks.projects.project import load_project_workflow

    root = tmp_path / "features"
    shutil.copytree(CUSTOM.parent, root, ignore=shutil.ignore_patterns("__pycache__"))
    _enable_asset_examples(root)
    _configure(root, pre_split=False, preprocessing=False)
    config = {"pipeline": {"preprocessing": [], "modeling": {"type": "linear_regression"}}}
    loaded = load_project_workflow(config, root)
    (region,) = loaded["pipeline"]["preprocessing"]
    (countries,) = loaded["pre_split_steps"]
    rows = pd.DataFrame({"city": ["London", "Paris"], "country": ["NL", "FR"]})
    _, added = _fit_apply(region, rows, rows)
    assert added["region"].iloc[0] == "UK" and pd.isna(added["region"].iloc[1])
    _, kept = _fit_apply(countries, rows, rows)
    assert kept["country"].tolist() == ["NL"]


@pytest.mark.parametrize(
    "columns, minimum", [([], 1), (["a", "a"], 1), (["a"], 0), (["a"], 2), (["a"], True)]
)
def test_completeness_rejects_ambiguous_rules(columns, minimum):
    """Malformed selections must fail before a Spark read or row filter is started."""
    with pytest.raises(ValueError):
        _custom_module("pre_split_custom.py").minimum_completeness(columns, minimum)


@pytest.mark.parametrize("columns", [[], ["a", "a"], [""]])
def test_frequency_rejects_ambiguous_columns(columns):
    """A custom encoder must not silently infer or duplicate feature columns."""
    with pytest.raises(ValueError):
        _custom_module("preprocessing_custom.py").frequency_encoding(columns)


def _enabled_project(tmp_path):
    """Enable the two separate builders exactly as a generated-project user would."""
    from skyulf.integrations.databricks.projects.project import load_project_workflow

    root = tmp_path / "features"
    shutil.copytree(CUSTOM.parent, root, ignore=shutil.ignore_patterns("__pycache__"))
    _configure(root, pre_split=True, preprocessing=True)
    config = {"pipeline": {"preprocessing": [], "modeling": {"type": "linear_regression"}}}
    return root, load_project_workflow(config, root)


def _configure(root, *, pre_split, preprocessing):
    """Configure the actual shipped builders by activating their inline custom step entries."""
    # These training-only quality fields are deliberately absent from prediction input.
    # Scoring-policy reuse and its required inputs have separate integration coverage.
    (root / "scoring.py").write_text(
        '"""Keep this fixture focused on fitted custom preprocessing."""\n\n'
        'def build_scoring():\n    """Opt out of prediction eligibility rules."""\n'
        "    return None\n\n"
        "build_model_rules = build_scoring\n\n"
        'def build_combined_rules():\n    """Disable composition for this fixture."""\n'
        "    return []\n",
        encoding="utf-8",
    )
    recipes = [
        (
            "pre_split.py",
            pre_split,
            "minimum_completeness",
            'minimum_completeness(columns=["field_a", "field_b", "field_c"], min_present=2)',
            'minimum_completeness(columns=["quality_a", "quality_b"], min_present=1)',
        ),
        (
            "preprocessing.py",
            preprocessing,
            "frequency_encoding",
            'frequency_encoding(columns=["category"])',
            'frequency_encoding(columns=["category"])',
        ),
    ]
    for filename, selected, factory, shown, configured in recipes:
        if selected:
            path = root / filename
            source = path.read_text(encoding="utf-8")
            module = filename.removesuffix(".py") + "_custom"
            source = source.replace(
                f"# from .custom.{module} import {factory}",
                f"from .custom.{module} import {factory}",
            )
            assert f"# {shown}," in source
            path.write_text(source.replace(f"# {shown},", f"{configured},"), encoding="utf-8")


@pytest.mark.parametrize("pre_split, preprocessing", [(False, False), (True, False), (False, True)])
def test_custom_builders_use_inline_steps(tmp_path, pre_split, preprocessing):
    """The real parent recipes configure custom steps independently without a demo toggle."""
    from skyulf.integrations.databricks.projects.project import load_project_workflow

    root = tmp_path / "features"
    shutil.copytree(CUSTOM.parent, root, ignore=shutil.ignore_patterns("__pycache__"))
    _configure(root, pre_split=pre_split, preprocessing=preprocessing)
    config = {"pipeline": {"preprocessing": [], "modeling": {"type": "linear_regression"}}}
    loaded = load_project_workflow(config, root)
    assert len(loaded["pre_split_steps"]) == int(pre_split)
    assert len(loaded["pipeline"]["preprocessing"]) == int(preprocessing)


@pytest.mark.parametrize("engine", ["pandas", "polars"])
@pytest.mark.parametrize("transport", ["local", "mlflow"])
def test_custom_steps_train_and_reload_without_editable_code(
    tmp_path, monkeypatch, engine, transport
):
    """The completeness filter, learned frequencies and saved package must compose end to end."""
    from skyulf.data.dataset import SplitDataset
    from skyulf.inference.local_pipeline import predict_local_pipeline
    from skyulf.integrations.databricks.scoring.batch.local_batch import fit_local_workflow
    from skyulf.integrations.databricks.training.fitting.local_retraining import (
        LocalTrainingSpec,
        split_labeled_snapshot,
    )

    if transport == "mlflow":
        pytest.importorskip("mlflow")
    monkeypatch.chdir(tmp_path)
    root, config = _enabled_project(tmp_path)
    rows = pd.DataFrame(
        {
            "order_id": range(16),
            "category": ["A", "B", "C", "D"] * 4,
            "amount": np.arange(1, 17, dtype=float) * 10,
            "quality_a": [1.0] * 12 + [None] * 4,
            "quality_b": [None] * 16,
            "target": np.arange(1, 17, dtype=float) * 20 + 1,
        }
    )
    spec = LocalTrainingSpec(
        table="workspace.test.records",
        version=0,
        record_key_columns=("order_id",),
        input_columns=("category", "amount"),
        target_column="target",
        max_rows=30,
        max_bytes=100000,
        pre_split_steps=tuple(config["pre_split_steps"]),
    )
    train, heldout, _ = split_labeled_snapshot(rows, spec, engine=engine)
    assert set(train.amount) | set(heldout.amount) == set(range(10, 121, 10))
    assert heldout.attrs["pre_split_filter_counts"][0]["excluded_rows"] == 4
    data = SplitDataset(train=_native(train, engine), test=_native(heldout, engine))
    artifact = fit_local_workflow(
        config["pipeline"],
        data,
        target_column="target",
        artifact_path=tmp_path / "artifact",
        max_rows=30,
        max_bytes=100000,
    )
    state = artifact.pipeline.feature_engineer.fitted_steps[0]["artifact"]
    assert state["state"]["category"] == train.category.value_counts(normalize=True).to_dict()
    score = pd.DataFrame({"category": ["A", "NEW"], "amount": [55.0, 105.0]})
    expected = predict_local_pipeline(score, artifact)["prediction"].tolist()
    np.testing.assert_allclose(expected, [111.0, 211.0], atol=1e-8)
    saved_path = str(tmp_path / "artifact")
    if transport == "mlflow":
        saved_path = _log_model(tmp_path)
    for path in root.rglob("*.py"):
        path.write_text("raise RuntimeError('edited project')\n", encoding="utf-8")
    code = (
        "import json, sys, pandas as pd\n"
        "from skyulf.inference.local_pipeline import load_local_pipeline, predict_local_pipeline\n"
        "rows = pd.DataFrame({'category':['A','NEW'], 'amount':[55.,105.]})\n"
        "if sys.argv[2] == 'mlflow':\n"
        "    import mlflow\n    result = mlflow.pyfunc.load_model(sys.argv[1]).predict(rows)\n"
        "else:\n    result = predict_local_pipeline(rows, load_local_pipeline(sys.argv[1]))\n"
        "print(json.dumps(result['prediction'].tolist()))\n"
    )
    loaded = subprocess.run(
        [sys.executable, "-c", code, saved_path, transport],
        cwd=tmp_path,
        text=True,
        capture_output=True,
        timeout=60,
    )
    assert loaded.returncode == 0, loaded.stderr
    np.testing.assert_allclose(json.loads(loaded.stdout), expected)


def _log_model(tmp_path):
    """Publish to a temporary local MLflow store through the real Core integration."""
    import mlflow

    from skyulf.integrations.mlflow.models.local_model import log_local_model
    from skyulf.integrations.mlflow.runs.tracking import TrackingConfig, track_run

    uri = f"sqlite:///{(tmp_path / 'tracking.db').as_posix()}"
    with track_run(
        TrackingConfig(enabled=True, tracking_uri=uri, experiment_name="custom_steps"),
        run_name="custom_steps",
    ) as run:
        assert run.run_id is not None
        model_uri = log_local_model(
            tmp_path / "artifact", run_id=run.run_id, artifact_path="model", tracking_uri=uri
        )
    return mlflow.artifacts.download_artifacts(artifact_uri=model_uri, tracking_uri=uri)


@pytest.mark.parametrize("engine", ["pandas", "polars"])
def test_frequencies_are_relearned_inside_each_cv_fold(tmp_path, monkeypatch, engine):
    """Each validation partition must use only its own training category frequencies."""
    from sklearn.model_selection import KFold

    from skyulf.integrations.databricks.training.tuning.local_cv import (
        LocalCVSpec,
        evaluate_training_cv,
    )
    from skyulf.preprocessing.base import BaseCalculator
    from skyulf.registry import NodeRegistry

    _, config = _enabled_project(tmp_path)
    calculator = NodeRegistry.get_calculator(config["pipeline"]["preprocessing"][0]["transformer"])
    assert issubclass(calculator, BaseCalculator)
    original = calculator.fit
    observed = []

    def record_fit(self, X, params):
        """Observe real fitted priors without replacing the custom calculation."""
        state = original(self, X, params)
        observed.append(state["state"]["category"]["A"])
        return state

    monkeypatch.setattr(calculator, "fit", record_fit)
    rows = pd.DataFrame(
        {
            "category": ["A"] * 8 + ["B"] * 4,
            "amount": np.arange(1, 13, dtype=float),
            "target": np.arange(1, 13, dtype=float) * 2,
        }
    )
    expected = [(rows.category.iloc[train] == "A").mean() for train, _ in KFold(3).split(rows)]
    report = evaluate_training_cv(
        _native(rows, engine),
        config["pipeline"],
        LocalCVSpec(enabled=True, folds=3, shuffle=False),
        target_column="target",
    )
    assert report is not None
    assert sorted(observed) == pytest.approx(sorted(expected))
