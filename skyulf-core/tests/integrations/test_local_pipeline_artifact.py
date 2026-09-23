"""A fitted local pipeline keeps its execution semantics when transported."""

from __future__ import annotations

import json

import numpy as np
import pandas as pd
import polars as pl
import pytest
from sklearn.metrics import accuracy_score

from skyulf.data.dataset import SplitDataset
from skyulf.pipeline import SkyulfPipeline


def _fitted_pipeline(engine: str) -> tuple[SkyulfPipeline, pd.DataFrame]:
    """Fit categorical FE that the initial portable Spark bundle cannot encode."""
    train = pd.DataFrame(
        {
            "city": ["Riga", "Vilnius", "Riga", "Tallinn", "Vilnius", "Tallinn"],
            "amount": [1.0, 2.0, 3.0, 4.0, 5.0, 6.0],
            "target": [11.0, 24.0, 13.0, 36.0, 27.0, 38.0],
        }
    )
    pipeline = SkyulfPipeline(
        {
            "preprocessing": [
                {
                    "name": "encode_city",
                    "transformer": "OneHotEncoder",
                    "params": {
                        "columns": ["city"],
                        "drop_original": True,
                        "handle_unknown": "ignore",
                    },
                }
            ],
            "modeling": {"type": "linear_regression"},
        }
    )
    data = pl.from_pandas(train) if engine == "polars" else train
    if isinstance(data, pl.DataFrame):
        training, held_out = data.slice(0, 5), data.slice(5, 1)
    else:
        training, held_out = data.iloc[:5], data.iloc[5:]
    pipeline.fit(SplitDataset(train=training, test=held_out), target_column="target")
    query = pd.DataFrame(
        {"city": ["Vilnius", "Riga", "Tallinn"], "amount": [7.0, 8.0, 9.0]},
        index=[31, 9, 42],
    )
    return pipeline, query


@pytest.mark.parametrize("engine", ["pandas", "polars"])
def test_fit_engine_survives_existing_pipeline_round_trip(engine: str, tmp_path) -> None:
    """MLflow packaging must know which local engine produced the fitted state."""
    pipeline, _ = _fitted_pipeline(engine)
    path = tmp_path / "pipeline.pkl"
    pipeline.save(str(path))
    loaded = SkyulfPipeline.load(str(path))

    assert loaded.fitted_engine == engine


@pytest.mark.parametrize("engine", ["pandas", "polars"])
def test_local_artifact_preserves_fitted_categorical_predictions(engine: str, tmp_path) -> None:
    """A versioned local artifact must replay categorical FE without refitting."""
    from skyulf.inference.local_pipeline import (  # noqa: PLC0415 - red test imports new API
        load_local_pipeline,
        predict_local_pipeline,
        save_local_pipeline,
    )

    pipeline, query = _fitted_pipeline(engine)
    native_query = pl.from_pandas(query) if engine == "polars" else query
    expected = np.asarray(pipeline.predict(native_query))
    path = tmp_path / "local-artifact"
    save_local_pipeline(pipeline, path)
    manifest = json.loads((path / "manifest.json").read_text(encoding="utf-8"))
    restored = load_local_pipeline(path)
    result = predict_local_pipeline(query, restored)
    polars_result = predict_local_pipeline(pl.from_pandas(query), restored)

    assert manifest["format_version"] == 1
    assert manifest["fitted_engine"] == engine
    assert manifest["input_columns"] == ["city", "amount"]
    assert list(result.columns) == ["prediction"]
    assert result.index.tolist() == query.index.tolist()
    np.testing.assert_allclose(result["prediction"].to_numpy(), expected, rtol=0, atol=1e-10)
    assert polars_result.index.tolist() == list(range(len(query)))
    np.testing.assert_allclose(polars_result["prediction"].to_numpy(), expected, rtol=0, atol=1e-10)


def test_local_artifact_rejects_reordered_input(tmp_path) -> None:
    """Reordered raw columns must fail before fitted FE sees a changed schema."""
    from skyulf.inference.local_pipeline import (  # noqa: PLC0415 - red test imports new API
        load_local_pipeline,
        predict_local_pipeline,
        save_local_pipeline,
    )

    pipeline, query = _fitted_pipeline("pandas")
    path = tmp_path / "local-artifact"
    save_local_pipeline(pipeline, path)

    with pytest.raises(ValueError, match="column order"):
        predict_local_pipeline(query[["amount", "city"]], load_local_pipeline(path))


@pytest.mark.parametrize("engine", ["pandas", "polars"])
def test_local_artifact_preserves_class_probabilities_and_tuned_threshold(
    engine: str, tmp_path
) -> None:
    """Class labels, probability positions and explicit tuned decisions must survive saving."""
    from skyulf.inference.local_pipeline import (
        load_local_pipeline,
        predict_local_pipeline,
        save_local_pipeline,
    )

    train = pd.DataFrame(
        {"x": [-3.0, -2.0, -1.0, 1.0, 2.0, 3.0], "label": ["no", "no", "no", "yes", "yes", "yes"]}
    )
    native_train = pl.from_pandas(train) if engine == "polars" else train
    pipeline = SkyulfPipeline({"preprocessing": [], "modeling": {"type": "logistic_regression"}})
    pipeline.fit(SplitDataset(train=native_train, test=native_train.head(0)), target_column="label")
    query = pd.DataFrame({"x": [-2.5, -0.25, 0.25, 2.5]})
    native_query = pl.from_pandas(query) if engine == "polars" else query
    pipeline.optimize_thresholds(
        native_query, np.array(["no", "yes", "yes", "yes"]), accuracy_score, grid_points=5
    )
    expected = np.asarray(pipeline.predict(native_query, use_tuned_thresholds=True))
    path = tmp_path / "classified"
    save_local_pipeline(pipeline, path, use_tuned_thresholds=True)
    loaded = load_local_pipeline(path)
    actual = predict_local_pipeline(query, loaded)

    assert loaded.manifest.classes == ("no", "yes")
    assert loaded.manifest.use_tuned_thresholds is True
    assert actual.columns.tolist() == ["prediction", "probability_0", "probability_1"]
    np.testing.assert_array_equal(actual["prediction"].to_numpy(), expected)
    np.testing.assert_allclose(actual[["probability_0", "probability_1"]].sum(axis=1), 1.0)


def test_local_artifact_rejects_tampered_payload(tmp_path) -> None:
    """A damaged pipeline payload must be rejected before pickle deserialization."""
    from skyulf.inference.local_pipeline import load_local_pipeline, save_local_pipeline

    pipeline, _ = _fitted_pipeline("pandas")
    path = tmp_path / "local-artifact"
    save_local_pipeline(pipeline, path)
    payload = path / "pipeline.pkl"
    payload.write_bytes(payload.read_bytes() + b"changed")

    with pytest.raises(ValueError, match="checksum"):
        load_local_pipeline(path)


def test_legacy_pipeline_loads_but_requires_refit_for_local_artifact(tmp_path) -> None:
    """Older pickle files remain loadable without inventing a missing fit engine."""
    from skyulf.inference.local_pipeline import save_local_pipeline

    pipeline, _ = _fitted_pipeline("pandas")
    del pipeline._fitted_engine
    legacy_path = tmp_path / "legacy.pkl"
    pipeline.save(str(legacy_path))
    loaded = SkyulfPipeline.load(str(legacy_path))

    assert loaded.fitted_engine is None
    with pytest.raises(ValueError, match="refit"):
        save_local_pipeline(loaded, tmp_path / "new-local")


def test_local_artifact_rejects_serving_and_spark_scopes(tmp_path) -> None:
    """A whole-frame local package must not be treated as endpoint or Spark eligible."""
    from skyulf.inference.local_pipeline import (
        load_local_pipeline,
        require_local_pipeline_scope,
        save_local_pipeline,
    )

    pipeline, _ = _fitted_pipeline("pandas")
    path = tmp_path / "local-artifact"
    save_local_pipeline(pipeline, path)
    artifact = load_local_pipeline(path)

    assert require_local_pipeline_scope(artifact, "whole_frame_local") is None
    for scope in ("row_local_http", "spark_worker", "spark_native"):
        with pytest.raises(ValueError, match="not eligible"):
            require_local_pipeline_scope(artifact, scope)


@pytest.mark.parametrize("engine", ["pandas", "polars"])
def test_local_artifact_replays_binning_and_encoding(engine: str, tmp_path) -> None:
    """Saved bin edges and downstream category positions must both survive loading."""
    from skyulf.inference.local_pipeline import (
        load_local_pipeline,
        predict_local_pipeline,
        save_local_pipeline,
    )

    train = pd.DataFrame(
        {"x": [0.5, 1.5, 2.5, 3.5, 4.5, 5.5], "target": [1.0, 2.0, 3.0, 4.0, 5.0, 6.0]}
    )
    native = pl.from_pandas(train) if engine == "polars" else train
    pipeline = SkyulfPipeline(
        {
            "preprocessing": [
                {
                    "name": "bands",
                    "transformer": "CustomBinning",
                    "params": {
                        "columns": ["x"],
                        "bins": [0.0, 2.0, 4.0, 6.0],
                        "output_suffix": "_band",
                    },
                },
                {
                    "name": "encode_bands",
                    "transformer": "OneHotEncoder",
                    "params": {
                        "columns": ["x_band"],
                        "drop_original": True,
                        "handle_unknown": "ignore",
                    },
                },
            ],
            "modeling": {"type": "linear_regression"},
        }
    )
    if isinstance(native, pl.DataFrame):
        training, heldout = native.slice(0, 5), native.slice(5, 1)
    else:
        training, heldout = native.iloc[:5], native.iloc[5:]
    pipeline.fit(SplitDataset(train=training, test=heldout), target_column="target")
    query = pd.DataFrame({"x": [1.0, 3.0, 5.0]})
    expected = np.asarray(pipeline.predict(pl.from_pandas(query) if engine == "polars" else query))
    path = tmp_path / "binned"
    save_local_pipeline(pipeline, path)
    actual = predict_local_pipeline(query, load_local_pipeline(path))

    np.testing.assert_allclose(actual["prediction"], expected, rtol=0, atol=1e-10)
