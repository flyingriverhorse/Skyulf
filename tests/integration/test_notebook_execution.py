"""Execute exported notebooks with real Core models and isolated local artifacts."""

from pathlib import Path

import numpy as np
import pandas as pd
import pytest
from sklearn.datasets import make_classification
from sklearn.model_selection import train_test_split

from backend.ml_pipeline._internal._routers import notebook_export as ne
from backend.ml_pipeline._internal._routers._notebook_builders import _NodeIn, _PipelineIn
from skyulf.pipeline import SkyulfPipeline


@pytest.fixture
def export_data(tmp_path):
    """Keep deterministic train/test means different to expose full-frame fitting."""
    features, target = make_classification(
        n_samples=80, n_features=4, n_informative=3, n_redundant=0, random_state=17
    )
    frame = pd.DataFrame(features, columns=list("abcd"))
    frame["target"] = target
    path = tmp_path / "data.csv"
    frame.to_csv(path, index=False)
    return frame, path


def _config(path, *, tuned=False, branched=False):
    """Give each branch a distinct split while sharing a learned scaler in the graph."""
    nodes = [
        _NodeIn(node_id="load", step_type="data_loader", params={"path": path.as_posix()}),
        _NodeIn(
            node_id="scale",
            step_type="StandardScaler",
            params={"columns": list("abcd")},
            inputs=["load"],
        ),
        _NodeIn(
            node_id="target",
            step_type="feature_target_split",
            params={"target_column": "target"},
            inputs=["scale"],
        ),
    ]
    for i in range(2 if branched else 1):
        nodes.append(
            _NodeIn(
                node_id=f"split{i}",
                step_type="TrainTestSplitter",
                params={"test_size": 0.25, "random_state": 17 + i},
                inputs=["target"],
            )
        )
        params = {"algorithm": "logistic_regression", "run_mode": "tuned" if tuned else "fixed"}
        if tuned:
            params["tuning_config"] = {
                "n_trials": 2,
                "strategy": "random",
                "cv_folds": 2,
                "search_space": {"C": [0.1, 1.0]},
            }
        nodes.append(
            _NodeIn(node_id=f"model{i}", step_type="training", params=params, inputs=[f"split{i}"])
        )
    return _PipelineIn(nodes=nodes)


def _execute(notebook):
    """Run every exported Python cell without mocking preprocessing, fit, or persistence."""
    namespace = {}
    for index, cell in enumerate(notebook["cells"]):
        if cell["cell_type"] == "code":
            exec(compile("".join(cell["source"]), f"export_cell_{index}", "exec"), namespace)
    return namespace


@pytest.mark.parametrize("mode", ["full", "compact"])
@pytest.mark.parametrize("tuned", [False, True])
@pytest.mark.parametrize("branched", [False, True])
def test_notebook_train_only_preprocessing_and_reload_predictions(
    mode, tuned, branched, export_data, tmp_path, monkeypatch
):
    """Saved notebooks must reproduce predictions from raw rows with branch-local fitting."""
    frame, path = export_data
    monkeypatch.chdir(tmp_path)
    builder = ne._build_full_notebook if mode == "full" else ne._build_compact_notebook
    namespace = _execute(
        builder(_config(path, tuned=tuned, branched=branched), "probe", "data.csv", path.as_posix())
    )
    for i in range(2 if branched else 1):
        suffix = f"_{chr(65 + i)}" if branched else ""
        artifact = Path(f"skyulf_pipeline{suffix}.pkl")
        assert artifact.is_file(), "Export must persist preprocessing and model together"
        loaded = SkyulfPipeline.load(str(artifact))
        pipeline = namespace[f"pipeline{suffix}"]
        train, _ = train_test_split(frame, test_size=0.25, random_state=17 + i)
        scaler = next(
            step
            for step in loaded.feature_engineer.fitted_steps
            if step["type"] == "StandardScaler"
        )
        np.testing.assert_allclose(
            scaler["artifact"]["mean"], train[list("abcd")].mean(), atol=1e-12
        )
        new_rows = frame.drop(columns="target").iloc[:5]
        predictions = loaded.predict(new_rows)
        assert np.asarray(predictions).shape == (5,)
        np.testing.assert_array_equal(predictions, pipeline.predict(new_rows))
        if mode == "full":
            np.testing.assert_array_equal(predictions, namespace[f"predict_new{suffix}"](new_rows))
        assert loaded.model_estimator is not None


@pytest.mark.parametrize("mode", ["full", "compact"])
def test_missing_split_uses_train_only_default(mode, export_data, tmp_path, monkeypatch):
    """A graph without a splitter still needs an honest held-out evaluation."""
    frame, path = export_data
    cfg = _config(path)
    cfg.nodes = [node for node in cfg.nodes if node.node_id != "split0"]
    cfg.nodes[-1].inputs = ["target"]
    monkeypatch.chdir(tmp_path)
    builder = ne._build_full_notebook if mode == "full" else ne._build_compact_notebook
    namespace = _execute(builder(cfg, "probe", "data.csv", path.as_posix()))
    train, _ = train_test_split(frame, test_size=0.2, random_state=42)
    scaler = next(
        step
        for step in namespace["pipeline"].feature_engineer.fitted_steps
        if step["type"] == "StandardScaler"
    )
    np.testing.assert_allclose(scaler["artifact"]["mean"], train[list("abcd")].mean(), atol=1e-12)


@pytest.mark.parametrize("mode", ["full", "compact"])
def test_branched_targets_with_quotes_execute(mode, export_data, tmp_path, monkeypatch):
    """Each branch must preserve an unusual target name through executable source."""
    frame, path = export_data
    target = 'label"\\true false null'
    frame.rename(columns={"target": target}).to_csv(path, index=False)
    cfg = _config(path, branched=True)
    cfg.nodes[2].params["target_column"] = target
    monkeypatch.chdir(tmp_path)
    builder = ne._build_full_notebook if mode == "full" else ne._build_compact_notebook
    namespace = _execute(builder(cfg, "probe", "data.csv", path.as_posix()))
    for letter in "AB":
        predictions = namespace[f"pipeline_{letter}"].predict(frame[list("abcd")].head(3))
        assert np.asarray(predictions).shape == (3,)


@pytest.mark.parametrize("mode", ["full", "compact"])
def test_preprocessing_only_export_has_no_fake_prediction(mode, export_data, tmp_path, monkeypatch):
    """Without a model, export must transform data without asking for a target or promising predictions."""
    frame, path = export_data
    cfg = _config(path)
    cfg.nodes = cfg.nodes[:2]
    monkeypatch.chdir(tmp_path)
    builder = ne._build_full_notebook if mode == "full" else ne._build_compact_notebook
    namespace = _execute(builder(cfg, "probe", "data.csv", path.as_posix()))
    assert "predict_new" not in namespace
    result = namespace["transform_new"](frame.head(3))
    assert result.shape == (3, 5)


@pytest.mark.parametrize("mode", ["full", "compact"])
def test_tuning_refits_scaler_inside_each_fold(mode, export_data, tmp_path, monkeypatch):
    """A train-only outer fit is insufficient unless tuning also refits inside CV folds."""
    from skyulf.preprocessing.scaling import standard

    _, path = export_data
    row_counts = []
    original = standard._fit_standard

    def record_fit(features, columns, config):
        """Observe real fits without replacing scaler behavior."""
        row_counts.append(len(features))
        return original(features, columns, config)

    monkeypatch.setattr(standard, "_fit_standard", record_fit)
    monkeypatch.chdir(tmp_path)
    builder = ne._build_full_notebook if mode == "full" else ne._build_compact_notebook
    _execute(builder(_config(path, tuned=True), "probe", "data.csv", path.as_posix()))
    assert row_counts.count(30) >= 4
    assert 60 in row_counts
    assert 80 not in row_counts


@pytest.mark.parametrize("mode", ["full", "compact"])
def test_parallel_training_keeps_both_model_paths(mode, export_data, tmp_path, monkeypatch):
    """Parallel execution inputs are independent models, not unsupported merged features."""
    frame, path = export_data
    cfg = _config(path, branched=True)
    cfg.nodes = [node for node in cfg.nodes if node.node_id != "model1"]
    model = next(node for node in cfg.nodes if node.node_id == "model0")
    model.inputs = ["split0", "split1"]
    model.params["execution_mode"] = "parallel"
    monkeypatch.chdir(tmp_path)
    builder = ne._build_full_notebook if mode == "full" else ne._build_compact_notebook
    namespace = _execute(builder(cfg, "probe", "data.csv", path.as_posix()))
    for letter in "AB":
        assert Path(f"skyulf_pipeline_{letter}.pkl").is_file()
        assert len(namespace[f"pipeline_{letter}"].predict(frame[list("abcd")].head(3))) == 3
