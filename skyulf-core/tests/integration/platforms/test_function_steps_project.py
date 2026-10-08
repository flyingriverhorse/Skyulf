"""Function steps must survive the real Bundle project lifecycle like custom classes."""

import json
import subprocess
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import polars as pl
import pytest

from skyulf.preprocessing import FeatureEngineer

SOURCE = '''
from skyulf.preprocessing import column_step, fitted_step


def income_per_age(df):
    """Income divided by age."""
    return df["income"] / df["age"]


def learn_city_median(df, y):
    """Learn the training median income per city."""
    return {"medians": df.groupby("city")["income"].median().to_dict()}


def apply_city_median(df, state):
    """Use the saved medians; unseen cities get 0."""
    return df["city"].map(state["medians"]).fillna(0.0).astype(float)


def drop_city(df):
    """Encode city away so the model only sees numbers."""
    return df["age"] * 0.0


def build_preprocessing(recipe="default"):
    """Two named recipes so different models can choose different features."""
    recipes = {
        "default": lambda: [
            column_step("ratio", income_per_age, output="income_per_age"),
            fitted_step("city_median", learn_city_median, apply_city_median,
                        output="city", replace=True),
        ],
        "ratio_only": lambda: [
            column_step("ratio", income_per_age, output="income_per_age"),
            column_step("no_city", drop_city, output="city", replace=True),
        ],
    }
    return recipes[recipe]()
'''


def _frame(engine):
    """Training rows with a numeric target."""
    frame = pd.DataFrame(
        {
            "income": [100.0, 200.0, 300.0, 400.0, 500.0, 600.0],
            "age": [10.0, 20.0, 30.0, 40.0, 50.0, 60.0],
            "city": ["A", "A", "A", "B", "B", "B"],
            "target": [1.0, 2.0, 3.0, 4.0, 5.0, 6.0],
        }
    )
    return pl.from_pandas(frame) if engine == "polars" else frame


def _project(tmp_path, recipe=None):
    """Resolve the editable project file exactly as a Bundle job does."""
    from skyulf.integrations.databricks.projects.project import load_project_workflow

    source = tmp_path / "preprocessing.py"
    source.write_text(SOURCE, encoding="utf-8")
    config = {"pipeline": {"preprocessing": [], "modeling": {"type": "linear_regression"}}}
    return load_project_workflow(config, source, preprocessing_recipe=recipe), source


@pytest.mark.parametrize("engine", ["pandas", "polars"])
def test_function_steps_predict_in_a_fresh_process_without_the_project_file(tmp_path, engine):
    """Saved source and learned state must be enough for inference."""
    from skyulf.data.dataset import SplitDataset
    from skyulf.inference.local_pipeline import predict_local_pipeline
    from skyulf.integrations.databricks.scoring.batch.local_batch import fit_local_workflow

    config, source = _project(tmp_path)
    frame = _frame(engine)
    artifact = fit_local_workflow(
        config["pipeline"],
        SplitDataset(train=frame, test=frame[:0]),
        target_column="target",
        artifact_path=tmp_path / "artifact",
        max_rows=100,
        max_bytes=100000,
    )
    fitted = artifact.pipeline.feature_engineer.fitted_steps
    assert fitted[1]["artifact"]["state"] == {"medians": {"A": 200.0, "B": 500.0}}
    assert artifact.manifest.project_source_sha256
    rows = pd.DataFrame({"income": [700.0, 50.0], "age": [70.0, 5.0], "city": ["B", "NEW"]})
    expected = predict_local_pipeline(rows, artifact)["prediction"].tolist()
    source.write_text("raise RuntimeError('edited project must not run')", encoding="utf-8")
    code = """
import json, sys, pandas as pd
from skyulf.inference.local_pipeline import load_local_pipeline, predict_local_pipeline
artifact = load_local_pipeline(sys.argv[1])
rows = pd.DataFrame({"income": [700., 50.], "age": [70., 5.], "city": ["B", "NEW"]})
print(json.dumps(predict_local_pipeline(rows, artifact)["prediction"].tolist()))
"""
    loaded = subprocess.run(
        [sys.executable, "-c", code, str(tmp_path / "artifact")],
        cwd=tmp_path,
        capture_output=True,
        text=True,
        timeout=120,
    )
    assert loaded.returncode == 0, loaded.stderr
    np.testing.assert_allclose(json.loads(loaded.stdout), expected)


def test_named_recipes_give_models_different_features(tmp_path):
    """Recipes stay the way to choose a feature set per model."""
    configs = []
    for recipe in ("default", "ratio_only"):
        (tmp_path / recipe).mkdir()
        configs.append(_project(tmp_path / recipe, recipe=recipe)[0])
    names = [[s["name"] for s in c["pipeline"]["preprocessing"]] for c in configs]
    assert names == [["ratio", "city_median"], ["ratio", "no_city"]]


def test_fitted_step_learns_inside_every_cv_fold(tmp_path, monkeypatch):
    """Learned state must come from fold training rows, never all rows."""
    from skyulf.integrations.databricks.training.tuning.local_cv import (
        LocalCVSpec,
        evaluate_training_cv,
    )
    from skyulf.preprocessing.function_steps import FittedFunctionCalculator

    config, _ = _project(tmp_path)
    seen = []
    original = FittedFunctionCalculator.fit

    def record(self, frame, params):
        """Record the row count each learn call saw."""
        state = original(self, frame, params)
        X = frame[0] if isinstance(frame, tuple) else frame
        seen.append(len(X))
        return state

    monkeypatch.setattr(FittedFunctionCalculator, "fit", record)
    frame = pd.DataFrame(
        {
            "income": np.arange(1, 13, dtype=float) * 100,
            "age": np.arange(1, 13, dtype=float) * 10,
            "city": ["A", "B"] * 6,
            "target": np.arange(12, dtype=float),
        }
    )
    result = evaluate_training_cv(
        frame,
        config["pipeline"],
        LocalCVSpec(enabled=True, folds=3, shuffle=False),
        target_column="target",
    )
    assert result is not None
    assert seen == [8, 8, 8]


PRE_SPLIT = '''
from skyulf.preprocessing import filter_step


def adult(df):
    """Keep adults only."""
    return df["age"] >= 18


def known_target(df):
    """Training-only rule that reads the target."""
    return df["target"].notna()


def build_pre_split_steps(recipe="default"):
    """Function filters admitted by the same rules as custom filter classes."""
    return [
        filter_step("adult_only", adult, columns=["age"]),
        filter_step("known_target", known_target, columns=["target"]),
    ]
'''


def _pre_split_project(tmp_path):
    """Write a project package with a function-based pre-split recipe."""
    from skyulf.integrations.databricks.projects.project import load_project_workflow

    root = tmp_path / "features"
    root.mkdir()
    (root / "__init__.py").write_text(
        "from .preprocessing import build_preprocessing\n"
        "from .pre_split import build_pre_split_steps\n",
        encoding="utf-8",
    )
    (root / "preprocessing.py").write_text(
        "def build_preprocessing(recipe='default'):\n    return []\n", encoding="utf-8"
    )
    (root / "pre_split.py").write_text(PRE_SPLIT, encoding="utf-8")
    config = {"pipeline": {"preprocessing": [], "modeling": {"type": "linear_regression"}}}
    return load_project_workflow(config, root)


@pytest.mark.parametrize("engine", ["pandas", "polars"])
def test_function_filter_passes_pre_split_admission_and_keeps_survivors(tmp_path, engine):
    """Function filters obey the fixed pre-split contract: select rows, never edit values."""
    from skyulf.integrations.databricks.training.fitting.local_retraining import (
        apply_pre_split_step,
        validate_pre_split_step,
    )

    steps = _pre_split_project(tmp_path)["pre_split_steps"]
    assert [list(validate_pre_split_step(s, i, "target", ())) for i, s in enumerate(steps)] == [
        ["age"],
        ["target"],
    ]
    frame = pd.DataFrame(
        {"id": [1, 2, 3, 4], "age": [10, 20, 17, 40], "target": [1.0, None, 3.0, 4.0]}
    )
    native = pl.from_pandas(frame) if engine == "polars" else frame
    for step in steps:
        native = apply_pre_split_step(native, step, keys=["id"], target_column="target")
    result = native.to_pandas() if isinstance(native, pl.DataFrame) else native
    assert isinstance(result, pd.DataFrame)
    assert result["id"].tolist() == [4]
    assert result["age"].tolist() == [40]


def test_function_filter_is_reused_for_scoring_with_reasons(tmp_path):
    """Scoring reuses the filter, skips the target rule and explains each exclusion."""
    from skyulf.integrations.databricks.scoring.shared.scoring_pre_split import (
        pre_split_exclusion_reasons,
        resolve_pre_split_scoring,
    )

    workflow = _pre_split_project(tmp_path)
    workflow = {**workflow, "target_column": "target", "input_columns": ["age"]}
    request = {"reuse_pre_split": True, "skip_target_steps": True}
    scoring = resolve_pre_split_scoring(request, workflow)
    assert scoring is not None
    resolved = scoring["pre_split"]
    assert resolved["skipped_target_steps"] == ["known_target"]
    assert [s["name"] for s in resolved["steps"]] == ["adult_only"]
    rows = pd.DataFrame({"age": [30, 12, 18]})
    reasons = pre_split_exclusion_reasons(rows, resolved)
    assert reasons.isna().tolist() == [True, False, True]
    assert reasons.iloc[1] == "pre_split:adult_only"


def test_function_filter_from_unloaded_source_is_rejected():
    """A filter must belong to loaded project source, not an arbitrary module."""
    from skyulf.integrations.databricks.training.fitting.local_retraining import (
        validate_pre_split_step,
    )
    from skyulf.preprocessing import filter_step

    step = filter_step("adult_only", adult_outside_project, columns=["age"])
    with pytest.raises(ValueError, match="only fixed normalization or row filters"):
        validate_pre_split_step(step, 0, "target", ())


def adult_outside_project(df):
    """Same rule defined in an ordinary module instead of project source."""
    return df["age"] >= 18


FULL = '''
from skyulf.preprocessing import column_step, filter_step, fitted_step


def has_category(df):
    """Train and score only rows with a known category."""
    return df["category"].notna()


def amount_squared(df):
    """A simple derived numeric feature."""
    return df["amount"] ** 2


def learn_category_share(df, y):
    """Learn each category's share of the training rows."""
    return {"share": df["category"].value_counts(normalize=True).to_dict()}


def apply_category_share(df, state):
    """Replace category with its saved share; unseen categories become 0."""
    return df["category"].map(state["share"]).fillna(0.0).astype(float)


def build_pre_split_steps(recipe="default"):
    """Function filter reused by scoring."""
    return [filter_step("has_category", has_category, columns=["category"])]


def build_preprocessing(recipe="default"):
    """Derived and learned features for the model."""
    return [
        column_step("amount_sq", amount_squared, output="amount_sq"),
        fitted_step("category_share", learn_category_share, apply_category_share,
                    output="category", replace=True),
    ]


def build_scoring():
    """Reuse the pre-split filter as scoring eligibility."""
    return {"reuse_pre_split": True, "skip_target_steps": False}
'''


@pytest.mark.parametrize("engine", ["pandas", "polars"])
@pytest.mark.parametrize("transport", ["local", "mlflow"])
def test_function_recipe_trains_and_scores_in_a_fresh_process(
    tmp_path, monkeypatch, engine, transport
):
    """The whole Bundle path works with function steps only: split, fit, save, score."""
    if transport == "mlflow":
        pytest.importorskip("mlflow")
    from skyulf.data.dataset import SplitDataset
    from skyulf.integrations.databricks.projects.project import load_project_workflow
    from skyulf.integrations.databricks.scoring.batch.local_batch import fit_local_workflow
    from skyulf.integrations.databricks.training.fitting.local_retraining import (
        LocalTrainingSpec,
        split_labeled_snapshot,
    )

    monkeypatch.chdir(tmp_path)
    root = tmp_path / "features"
    root.mkdir()
    (root / "__init__.py").write_text(FULL, encoding="utf-8")
    config = load_project_workflow(
        {
            "engine": engine,
            "input_columns": ["category", "amount"],
            "target_column": "target",
            "pipeline": {"preprocessing": [], "modeling": {"type": "linear_regression"}},
        },
        root,
    )
    rows = pd.DataFrame(
        {
            "id": range(20),
            "category": ["A"] * 10 + ["B"] * 8 + [None] * 2,
            "amount": np.arange(1, 21, dtype=float),
            "target": np.arange(1, 21) * 2.0 + 1,
        }
    )
    spec = LocalTrainingSpec(
        table="workspace.test.rows",
        version=0,
        record_key_columns=("id",),
        input_columns=("category", "amount"),
        target_column="target",
        max_rows=30,
        max_bytes=100000,
        pre_split_steps=tuple(config["pre_split_steps"]),
    )
    train, heldout, _ = split_labeled_snapshot(rows, spec, engine=engine)
    assert set(train.amount) | set(heldout.amount) == set(range(1, 19))
    data = SplitDataset(
        train=pl.from_pandas(train) if engine == "polars" else train,
        test=pl.from_pandas(heldout) if engine == "polars" else heldout,
    )
    artifact = fit_local_workflow(
        config["pipeline"],
        data,
        target_column="target",
        artifact_path=tmp_path / "artifact",
        max_rows=30,
        max_bytes=100000,
    )
    state = artifact.pipeline.feature_engineer.fitted_steps[1]["artifact"]["state"]
    assert state["share"] == pytest.approx(train.category.value_counts(normalize=True).to_dict())
    saved_path = str(tmp_path / "artifact")
    if transport == "mlflow":
        saved_path = _log_model(tmp_path)
    (root / "__init__.py").write_text("raise RuntimeError('edited recipe')\n", encoding="utf-8")
    code = (
        "import json, sys, pandas as pd\n"
        "from skyulf.inference.local_pipeline import load_local_pipeline\n"
        "from skyulf.inference.local_scoring import score_local_pipeline\n"
        "rows = pd.DataFrame({'category':['A','NEW',None], 'amount':[5.,10.,15.]})\n"
        "if sys.argv[2] == 'mlflow':\n"
        "    import mlflow\n    result = mlflow.pyfunc.load_model(sys.argv[1]).predict(rows)\n"
        "else:\n    result = score_local_pipeline(rows, load_local_pipeline(sys.argv[1]))\n"
        "print(json.dumps({'status': result.scoring_status.tolist()}))\n"
    )
    loaded = subprocess.run(
        [sys.executable, "-c", code, saved_path, transport],
        cwd=tmp_path,
        capture_output=True,
        text=True,
        timeout=120,
    )
    assert loaded.returncode == 0, loaded.stderr
    assert json.loads(loaded.stdout)["status"] == ["predicted", "predicted", "excluded"]


def _log_model(tmp_path):
    """Package the artifact through MLflow and download it for a fresh pyfunc load."""
    import mlflow

    from skyulf.integrations.mlflow.models.local_model import log_local_model
    from skyulf.integrations.mlflow.runs.tracking import TrackingConfig, track_run

    uri = f"sqlite:///{(tmp_path / 'tracking.db').as_posix()}"
    with track_run(
        TrackingConfig(enabled=True, tracking_uri=uri, experiment_name="functions"),
        run_name="functions",
    ) as run:
        assert run.run_id is not None
        model_uri = log_local_model(
            tmp_path / "artifact", run_id=run.run_id, artifact_path="model", tracking_uri=uri
        )
    return mlflow.artifacts.download_artifacts(artifact_uri=model_uri, tracking_uri=uri)


PARITY = '''
from skyulf.preprocessing import filter_step


def age_present(df):
    """Same rows as DropMissingRows(subset=["age"])."""
    return df["age"].notna()


def age_in_range(df, params):
    """Same rows as ManualBounds: inclusive bounds, missing ages stay."""
    age = df["age"]
    return age.between(params["lower"], params["upper"]) | age.isna()


def careless_adult(df):
    """Writes into its input before answering; must not edit the snapshot."""
    df["age"] = df["age"].fillna(0)
    return df["age"] >= 18


def nobody(df):
    """Rejects every row."""
    return df["age"] > 1000


def build_preprocessing(recipe="default"):
    """No model features are needed for these filter checks."""
    return []


def build_pre_split_steps(recipe="default"):
    """Function equivalents of Core pre-split filters plus edge cases."""
    bounds = {"lower": 18.0, "upper": 65.0}
    return {
        "present": [filter_step("age_rule", age_present, columns=["age"])],
        "bounds": [filter_step("age_rule", age_in_range, columns=["age"], params=bounds)],
        "careless": [filter_step("age_rule", careless_adult, columns=["age"])],
        "nobody": [filter_step("age_rule", nobody, columns=["age"])],
    }[recipe]
'''

CORE_EQUIVALENT = {
    "present": {
        "name": "age_rule",
        "transformer": "DropMissingRows",
        "params": {"subset": ["age"]},
    },
    "bounds": {
        "name": "age_rule",
        "transformer": "ManualBounds",
        "params": {"bounds": {"age": {"lower": 18.0, "upper": 65.0}}},
    },
}


def _parity_steps(tmp_path, recipe):
    """Load one function pre-split recipe from real project source."""
    from skyulf.integrations.databricks.projects.project import load_project_workflow

    root = tmp_path / "features"
    root.mkdir(exist_ok=True)
    (root / "__init__.py").write_text(PARITY, encoding="utf-8")
    config = {"pipeline": {"preprocessing": [], "modeling": {"type": "linear_regression"}}}
    workflow = load_project_workflow(config, root, pre_split_recipe=recipe)
    return list(workflow["pre_split_steps"])


def _ages(engine):
    """Snapshot with missing ages and values on both bounds."""
    frame = pd.DataFrame(
        {
            "id": range(7),
            "age": [None, 17.0, 18.0, 40.0, 65.0, 66.0, None],
            "target": [1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0],
        }
    )
    return pl.from_pandas(frame) if engine == "polars" else frame


@pytest.mark.parametrize("engine", ["pandas", "polars"])
@pytest.mark.parametrize("recipe", ["present", "bounds"])
def test_function_pre_split_matches_core_filters_for_training_and_scoring(tmp_path, engine, recipe):
    """Same survivors at training and the same exclusion reasons at scoring as Core."""
    from skyulf.integrations.databricks.scoring.shared.scoring_pre_split import (
        pre_split_exclusion_reasons,
    )
    from skyulf.integrations.databricks.training.fitting.local_retraining import (
        apply_pre_split_step,
        validate_pre_split_step,
    )

    function_step = _parity_steps(tmp_path, recipe)[0]
    core_step = CORE_EQUIVALENT[recipe]
    survivors, reasons = [], []
    for step in (core_step, function_step):
        assert list(validate_pre_split_step(step, 0, "target", ())) == ["age"]
        kept = apply_pre_split_step(_ages(engine), step, keys=["id"], target_column="target")
        survivors.append(kept["id"].to_list())
        config = {"engine": engine, "steps": [step], "target_column": "target"}
        scoring = pd.DataFrame({"age": [None, 10.0, 30.0, 99.0]})
        reasons.append(pre_split_exclusion_reasons(scoring, config).tolist())
    assert survivors[1] == survivors[0]
    assert reasons[1] == reasons[0]


@pytest.mark.parametrize("engine", ["pandas", "polars"])
def test_function_filter_that_edits_its_input_cannot_edit_the_snapshot(tmp_path, engine):
    """A careless function sees a copy, so pre-split still only selects rows."""
    from skyulf.integrations.databricks.training.fitting.local_retraining import (
        apply_pre_split_step,
    )

    step = _parity_steps(tmp_path, "careless")[0]
    source = _ages(engine)
    kept = apply_pre_split_step(source, step, keys=["id"], target_column="target")
    assert kept["id"].to_list() == [2, 3, 4, 5]
    assert source["age"].to_list()[0] is None or np.isnan(source["age"].to_list()[0])


def test_function_filter_rejecting_every_row_stops_training_clearly(tmp_path):
    """An empty training snapshot must fail before a model is fitted."""
    from skyulf.integrations.databricks.training.fitting.local_retraining import (
        LocalTrainingSpec,
        split_labeled_snapshot,
    )

    spec = LocalTrainingSpec(
        table="workspace.test.rows",
        version=0,
        record_key_columns=("id",),
        input_columns=("age",),
        target_column="target",
        max_rows=30,
        max_bytes=100000,
        pre_split_steps=tuple(_parity_steps(tmp_path, "nobody")),
    )
    with pytest.raises(ValueError, match="fewer than four eligible rows"):
        split_labeled_snapshot(_ages("pandas"), spec)


TEMPLATE_FEATURES = next(
    (Path(__file__).resolve().parents[3] / "templates" / "databricks" / "template").glob(
        "*/src/features"
    )
)

TEMPLATE_PARITY = '''
import pandas as pd

from skyulf.preprocessing import filter_step, fitted_step

from .pre_split import minimum_completeness
from .preprocessing import frequency_encoding

COLUMNS = ["category", "region"]


def learn_frequencies(df, y, params):
    """Share of training rows per category; missing values are not categories."""
    return {c: (df[c].value_counts() / len(df)).to_dict() for c in params["columns"]}


def apply_frequencies(df, state, params):
    """Replace categories with training shares; missing and unseen become 0."""
    return pd.DataFrame({c: df[c].map(state[c]).fillna(0.0).astype(float) for c in state})


def enough_present(df, params):
    """Keep rows with at least min_present observed fields."""
    return df[params["columns"]].notna().sum(axis=1) >= params["min_present"]


def build_preprocessing(recipe="class"):
    """Same frequency encoding as a 90-line class pair or two short functions."""
    return {
        "class": [frequency_encoding(columns=COLUMNS)],
        "function": [
            fitted_step("frequency_encoding", learn_frequencies, apply_frequencies,
                        output=COLUMNS, replace=True, params={"columns": COLUMNS})
        ],
    }[recipe]


def build_pre_split_steps(recipe="class"):
    """Same completeness filter as a class pair or one short function."""
    params = {"columns": COLUMNS, "min_present": 2}
    return {
        "class": [minimum_completeness(**params)],
        "function": [filter_step("minimum_completeness", enough_present,
                                 columns=COLUMNS, params=params)],
    }[recipe]
'''


def _template_project(tmp_path, recipe):
    """Copy the real template custom steps beside function equivalents and load one recipe."""
    from skyulf.integrations.databricks.projects.project import load_project_workflow

    root = tmp_path / recipe / "features"
    root.mkdir(parents=True)
    for name in ("preprocessing.py", "pre_split.py"):
        source = TEMPLATE_FEATURES / name
        (root / name).write_text(source.read_text(encoding="utf-8"), encoding="utf-8")
    (root / "__init__.py").write_text(TEMPLATE_PARITY, encoding="utf-8")
    config = {"pipeline": {"preprocessing": [], "modeling": {"type": "linear_regression"}}}
    return load_project_workflow(config, root, preprocessing_recipe=recipe, pre_split_recipe=recipe)


def _template_rows(engine):
    """Categories with missing values, an unseen value and incomplete rows."""
    frame = pd.DataFrame(
        {
            "id": range(8),
            "category": ["a", "a", "b", None, "a", "c", None, "b"],
            "region": ["x", None, "y", None, "x", "x", "y", None],
            "target": np.arange(8, dtype=float),
        }
    )
    return pl.from_pandas(frame) if engine == "polars" else frame


@pytest.mark.parametrize("engine", ["pandas", "polars"])
def test_template_class_steps_and_function_steps_agree(tmp_path, engine):
    """The template's class-based steps and their short function versions give equal results."""
    from skyulf.integrations.databricks.scoring.shared.scoring_pre_split import (
        pre_split_exclusion_reasons,
    )
    from skyulf.integrations.databricks.training.fitting.local_retraining import (
        apply_pre_split_step,
    )

    batch = pd.DataFrame({"category": ["a", "zzz", None], "region": [None, "y", None]})
    outcomes = []
    for recipe in ("class", "function"):
        workflow = _template_project(tmp_path, recipe)
        engineer = FeatureEngineer(workflow["pipeline"]["preprocessing"])
        train, _ = engineer.fit_transform(_template_rows(engine))
        scored = engineer.transform(pl.from_pandas(batch) if engine == "polars" else batch)
        step = workflow["pre_split_steps"][0]
        kept = apply_pre_split_step(
            _template_rows(engine), step, keys=["id"], target_column="target"
        )
        config = {"engine": engine, "steps": [step], "target_column": "target"}
        reasons = pre_split_exclusion_reasons(batch, config).tolist()
        outcomes.append((_as_pandas(train), _as_pandas(scored), kept["id"].to_list(), reasons))
    (class_train, class_scored, class_kept, class_reasons) = outcomes[0]
    (fn_train, fn_scored, fn_kept, fn_reasons) = outcomes[1]
    pd.testing.assert_frame_equal(fn_train, class_train)
    pd.testing.assert_frame_equal(fn_scored, class_scored)
    assert fn_scored["category"].tolist() == [0.375, 0.0, 0.0]
    assert fn_kept == class_kept == [0, 2, 4, 5]
    assert fn_reasons == class_reasons


def _as_pandas(frame):
    """Compare engine results as pandas."""
    return frame.to_pandas() if isinstance(frame, pl.DataFrame) else frame
