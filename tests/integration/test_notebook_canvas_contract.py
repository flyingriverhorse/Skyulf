"""Export must consume the model_type/hyperparameters payload sent by the real Canvas."""

import numpy as np
import pandas as pd
import pytest

from backend.ml_pipeline._internal._routers import notebook_export as ne
from backend.ml_pipeline._internal._routers._notebook_builders import _NodeIn, _PipelineIn


@pytest.mark.parametrize("mode", ["full", "compact"])
@pytest.mark.parametrize("tuned", [False, True])
def test_canvas_model_type_and_hyperparameters_survive_export(mode, tuned, tmp_path, monkeypatch):
    """Real Canvas model parameters must train the selected model with its chosen settings."""
    frame = pd.DataFrame({"x": np.arange(60), "target": [0, 1] * 30})
    path = tmp_path / "data.csv"
    frame.to_csv(path, index=False)
    params = {
        "model_type": "logistic_regression",
        "run_mode": "tuned" if tuned else "fixed",
        "hyperparameters": {"C": 0.37, "max_iter": 200},
        "target_column": "target",
        "cv_enabled": True,
        "cv_folds": 2,
        "cv_type": "k_fold",
        "cv_shuffle": True,
        "cv_random_state": 42,
        "cv_time_column": "",
    }
    if tuned:
        params["tuning_config"] = {
            "strategy": "grid",
            "search_space": {"C": [0.37]},
            "cv_folds": 2,
            "n_trials": 1,
        }
    cfg = _PipelineIn(
        nodes=[
            _NodeIn(node_id="load", step_type="data_loader", params={"path": path.as_posix()}),
            _NodeIn(
                node_id="split",
                step_type="TrainTestSplitter",
                params={"test_size": 0.2, "random_state": 42},
                inputs=["load"],
            ),
            _NodeIn(
                node_id="scale",
                step_type="StandardScaler",
                params={"columns": ["x"]},
                inputs=["split"],
            ),
            _NodeIn(node_id="model", step_type="training", params=params, inputs=["scale"]),
        ]
    )
    monkeypatch.chdir(tmp_path)
    builder = ne._build_full_notebook if mode == "full" else ne._build_compact_notebook
    notebook = builder(cfg, "probe", "data.csv")
    namespace = {}
    for cell in notebook["cells"]:
        if cell["cell_type"] == "code":
            exec(compile("".join(cell["source"]), "canvas_export", "exec"), namespace)
    pipeline = namespace["pipeline"]
    model, tuning = pipeline.model_estimator.model
    assert model.get_params()["C"] == 0.37
    assert pipeline.modeling_config["cv_folds"] == 2
    assert len(pipeline.predict(frame.drop(columns="target"))) == 60
    assert np.isfinite(tuning.best_score)
