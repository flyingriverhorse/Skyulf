"""A model bundle must preserve inference semantics and reject ambiguous input."""

import json
import pickle
from dataclasses import replace
from typing import Any, cast

import numpy as np
import pandas as pd
import polars as pl
import pytest
from sklearn.metrics import accuracy_score

from skyulf.core.execution import ExecutionOptions
from skyulf.core.schema import SchemaMismatchError
from skyulf.data.dataset import SplitDataset
from skyulf.inference.bundle import build_bundle, load_bundle, predict_local, save_bundle
from skyulf.pipeline import SkyulfPipeline
from skyulf.preprocessing import _feature_state


@pytest.fixture(params=["pandas", "polars"])
def fitted_regression_pipeline(request):
    """Different feature scales expose swapped columns and accidental double preprocessing."""
    rng = np.random.default_rng(17)
    frame = pd.DataFrame({"x": rng.normal(100, 20, 40), "z": rng.normal(5, 2, 40)})
    frame["target"] = 2 * frame.x - 3 * frame.z + 7
    pipeline = SkyulfPipeline(
        {
            "preprocessing": [
                {"name": "fill", "transformer": "SimpleImputer", "params": {"columns": ["x", "z"]}},
                {
                    "name": "scale",
                    "transformer": "StandardScaler",
                    "params": {"columns": ["x", "z"]},
                },
            ],
            "modeling": {"type": "linear_regression"},
        }
    )
    data = pl.from_pandas(frame) if request.param == "polars" else frame
    pipeline.fit(SplitDataset(train=data, test=data.head(0)), target_column="target")
    return pipeline


@pytest.fixture
def raw_frame():
    """Held-out values and a nondefault index expose re-fitting and row misalignment."""
    return pd.DataFrame({"x": [150.0, 80.0], "z": [9.0, 3.0]}, index=[17, 4])


@pytest.mark.parametrize("stage", ["raw", "features"])
def test_round_trip_preserves_predictions(fitted_regression_pipeline, raw_frame, tmp_path, stage):
    """Loading must preserve the same prediction function without applying FE twice."""
    pipeline = fitted_regression_pipeline
    bundle = build_bundle(pipeline, input_stage=stage, feature_order=("x", "z"))
    frame = raw_frame if stage == "raw" else pipeline.feature_engineer.transform(raw_frame)
    before = predict_local(frame, bundle)
    save_bundle(bundle, tmp_path / "model")
    restored = load_bundle(tmp_path / "model")
    after = predict_local(frame, restored)
    np.testing.assert_allclose(after.prediction, [280.0, 158.0], rtol=1e-10, atol=1e-12)
    pd.testing.assert_frame_equal(before, after)
    assert after.index.tolist() == [17, 4]
    assert restored.semantic_digest == bundle.semantic_digest
    assert restored.probability_columns == ()


def test_polars_apply_and_empty_input(fitted_regression_pipeline, raw_frame):
    """Local engine changes and empty batches must retain the declared output schema."""
    bundle = build_bundle(fitted_regression_pipeline, input_stage="raw", feature_order=("x", "z"))
    out = predict_local(pl.from_pandas(raw_frame), bundle)
    np.testing.assert_allclose(out.prediction, [280.0, 158.0])
    empty = predict_local(raw_frame.head(0), bundle)
    assert empty.empty and list(empty.columns) == ["prediction"]
    assert str(empty.prediction.dtype) == "float64"


@pytest.mark.parametrize("change", ["reorder", "missing", "extra", "dtype", "duplicate"])
def test_input_contract_fails_closed(fitted_regression_pipeline, raw_frame, change):
    """Names alone cannot protect NumPy model inputs from order or dtype mistakes."""
    bundle = build_bundle(fitted_regression_pipeline, input_stage="raw", feature_order=("x", "z"))
    frame = {
        "reorder": lambda: raw_frame[["z", "x"]],
        "missing": lambda: raw_frame[["x"]],
        "extra": lambda: raw_frame.assign(id=[1, 2]),
        "dtype": lambda: raw_frame.assign(x=raw_frame.x.astype(str)),
        "duplicate": lambda: pd.concat([raw_frame, raw_frame[["x"]]], axis=1),
    }[change]()
    with pytest.raises((SchemaMismatchError, ValueError)):
        predict_local(frame, bundle)


def test_feature_order_must_match_actual_training_order(fitted_regression_pipeline):
    """An explicit manifest cannot redefine the feature positions learned by the model."""
    with pytest.raises(ValueError, match="feature_order"):
        build_bundle(fitted_regression_pipeline, input_stage="features", feature_order=("z", "x"))


def test_model_budget_is_enforced_before_unpickling(
    fitted_regression_pipeline, tmp_path, monkeypatch
):
    """Oversized estimators must never reach deserialization even with valid checksums."""
    bundle = build_bundle(fitted_regression_pipeline, input_stage="raw", feature_order=("x", "z"))
    limit = len(bundle.model_payload) - 1
    with pytest.raises(ValueError, match="model_max_bytes"):
        build_bundle(
            fitted_regression_pipeline,
            input_stage="raw",
            feature_order=("x", "z"),
            options=ExecutionOptions("pandas", model_max_bytes=limit),
        )
    save_bundle(bundle, tmp_path / "model")

    def forbidden(*args, **kwargs):
        """A bounded load failure must happen before executing the pickle loader."""
        pytest.fail("Deserialized an oversized model")

    monkeypatch.setattr(pickle, "loads", forbidden)
    with pytest.raises(ValueError, match="model_max_bytes"):
        load_bundle(tmp_path / "model", options=ExecutionOptions("pandas", model_max_bytes=limit))


@pytest.mark.parametrize("part", ["manifest", "model", "features"])
def test_corruption_is_rejected_before_deserialization(
    fitted_regression_pipeline, tmp_path, monkeypatch, part
):
    """Corrupt metadata or either payload must not become an executable bundle."""
    bundle = build_bundle(fitted_regression_pipeline, input_stage="raw", feature_order=("x", "z"))
    path = tmp_path / "model"
    save_bundle(bundle, path)
    if part == "manifest":
        manifest = json.loads((path / "manifest.json").read_bytes())
        manifest["format_version"] = 999
        (path / "manifest.json").write_text(json.dumps(manifest), encoding="utf-8")
    else:
        target = path / ("model.pkl" if part == "model" else "features.json")
        target.write_bytes(target.read_bytes() + b"corrupt")

    def forbidden(*args, **kwargs):
        """Untrusted bytes must pass all structural and checksum gates first."""
        pytest.fail("Deserialized corrupt bundle")

    monkeypatch.setattr(pickle, "loads", forbidden)
    with pytest.raises(ValueError):
        load_bundle(path)


def test_bundle_freezes_source_and_preserves_legacy_pickle(
    fitted_regression_pipeline, raw_frame, tmp_path
):
    """A later training-object mutation cannot silently change an already built package."""
    pipeline = fitted_regression_pipeline
    pipeline.save(str(tmp_path / "legacy.pkl"))
    legacy = SkyulfPipeline.load(str(tmp_path / "legacy.pkl"))
    bundle = build_bundle(legacy, input_stage="raw", feature_order=("x", "z"))
    assert pipeline.model_estimator is not None
    pipeline.model_estimator.model.coef_[:] = 0
    np.testing.assert_allclose(predict_local(raw_frame, bundle).prediction, [280.0, 158.0])
    with pytest.raises(TypeError, match="standalone"):
        build_bundle(cast(Any, {"model": legacy}), input_stage="raw", feature_order=("x", "z"))


def test_manifest_and_payload_disagreement_rejected(fitted_regression_pipeline, raw_frame):
    """Frozen metadata cannot be paired with a different model or preprocessing payload."""
    bundle = build_bundle(fitted_regression_pipeline, input_stage="raw", feature_order=("x", "z"))
    with pytest.raises(ValueError, match="checksum"):
        predict_local(raw_frame, replace(bundle, model_payload=b"other"))


def test_class_order_and_opt_in_thresholds_round_trip(tmp_path):
    """Default model decisions and explicit pipeline thresholds must remain distinct."""
    frame = pd.DataFrame({"x": np.linspace(-3, 3, 30)})
    frame["label"] = np.where(frame.x > 0, "yes", "no")
    pipeline = SkyulfPipeline({"preprocessing": [], "modeling": {"type": "logistic_regression"}})
    pipeline.fit(SplitDataset(train=frame, test=frame.head(0)), target_column="label")
    query = pd.DataFrame({"x": [-0.1, 0.1, 2.0]})
    pipeline.optimize_thresholds(
        query, np.array(["no", "no", "yes"]), accuracy_score, grid_points=5
    )
    for enabled in (False, True):
        bundle = build_bundle(
            pipeline, input_stage="raw", feature_order=("x",), use_tuned_thresholds=enabled
        )
        path = tmp_path / str(enabled)
        save_bundle(bundle, path)
        restored = load_bundle(path)
        out = predict_local(query, restored)
        np.testing.assert_array_equal(
            out.prediction, pipeline.predict(query, use_tuned_thresholds=enabled)
        )
        assert pipeline.model_estimator is not None
        assert pipeline.model_estimator.model is not None
        np.testing.assert_allclose(
            out[list(restored.probability_columns)],
            pipeline.model_estimator.model.predict_proba(query.to_numpy()),
        )
        assert restored.classes == ("no", "yes") and restored.positive_label == "yes"
        assert tuple(out.columns) == ("prediction", *restored.probability_columns)


def test_boolean_input_schema_crosses_local_engines():
    """Polars Boolean and pandas bool represent the same supported primitive feature type."""
    data = pl.DataFrame({"flag": [True, False, True, False], "target": [3.0, 1.0, 3.0, 1.0]})
    pipeline = SkyulfPipeline({"preprocessing": [], "modeling": {"type": "linear_regression"}})
    pipeline.fit(SplitDataset(train=data, test=data.head(0)), target_column="target")
    bundle = build_bundle(pipeline, input_stage="features", feature_order=("flag",))
    out = predict_local(pd.DataFrame({"flag": [False, True]}), bundle)
    np.testing.assert_allclose(out.prediction, [1.0, 3.0])


def test_tuning_thresholds_remain_default_after_packaging(tmp_path):
    """Extracting the tuned estimator must not lose the tuner's active decision rule."""
    frame = pd.DataFrame({"x": np.linspace(-3, 3, 36)})
    frame["label"] = np.where(frame.x > 0, 2, 1)
    pipeline = SkyulfPipeline(
        {
            "preprocessing": [],
            "modeling": {
                "type": "hyperparameter_tuner",
                "base_model": {"type": "logistic_regression"},
                "strategy": "grid",
                "metric": "accuracy",
                "search_space": {"C": [1.0]},
                "cv_folds": 2,
                "n_jobs": 1,
                "tune_threshold": True,
            },
        }
    )
    pipeline.fit(
        SplitDataset(train=frame, test=frame.head(0), validation=frame.iloc[::2]),
        target_column="label",
    )
    query = pd.DataFrame({"x": [-0.2, 0.2, 2.0]})
    bundle = build_bundle(pipeline, input_stage="raw", feature_order=("x",))
    save_bundle(bundle, tmp_path / "tuned")
    restored = load_bundle(tmp_path / "tuned")
    assert restored.manifest.thresholds.source == "tuning"
    np.testing.assert_array_equal(
        predict_local(query, restored).prediction, pipeline.predict(query)
    )


def test_legacy_schema_missing_and_existing_destination_rejected(
    fitted_regression_pipeline, tmp_path
):
    """Existing files remain untouched and unnamed historical model inputs are never guessed."""
    pipeline = fitted_regression_pipeline
    bundle = build_bundle(pipeline, input_stage="raw", feature_order=("x", "z"))
    save_bundle(bundle, tmp_path / "model")
    original = (tmp_path / "model" / "model.pkl").read_bytes()
    with pytest.raises(FileExistsError):
        save_bundle(bundle, tmp_path / "model")
    assert (tmp_path / "model" / "model.pkl").read_bytes() == original
    del pipeline._inference_schemas
    with pytest.raises(ValueError, match="refit"):
        build_bundle(pipeline, input_stage="raw", feature_order=("x", "z"))


def test_manifest_corruption_and_runtime_mismatch_before_pickle(
    fitted_regression_pipeline, tmp_path, monkeypatch
):
    """Valid JSON with changed semantics or incompatible dependencies cannot reach pickle."""
    from skyulf.inference._manifest import semantic_digest

    bundle = build_bundle(fitted_regression_pipeline, input_stage="raw", feature_order=("x", "z"))
    path = tmp_path / "model"
    save_bundle(bundle, path)
    changed = bundle.manifest.model_copy(update={"input_stage": "features"})
    with pytest.raises(ValueError, match="digest"):
        predict_local(pd.DataFrame({"x": [0.0], "z": [0.0]}), replace(bundle, manifest=changed))
    requirements = tuple(
        (name, "0.0.0" if name == "scikit-learn" else value)
        for name, value in bundle.manifest.requirements
    )
    changed = bundle.manifest.model_copy(update={"requirements": requirements})
    changed = changed.model_copy(update={"semantic_digest": semantic_digest(changed)})
    (path / "manifest.json").write_text(changed.model_dump_json(), encoding="utf-8")

    def forbidden(*args, **kwargs):
        """Version checks must precede execution of even checksum-valid pickle bytes."""
        pytest.fail("Loaded incompatible model")

    monkeypatch.setattr(pickle, "loads", forbidden)
    with pytest.raises(ValueError, match="runtime mismatch"):
        load_bundle(path)


def test_metadata_and_fe_share_one_byte_budget(fitted_regression_pipeline):
    """Independent small components must not bypass a combined state budget."""
    bundle = build_bundle(fitted_regression_pipeline, input_stage="raw", feature_order=("x", "z"))
    with pytest.raises(ValueError, match="state_max_bytes"):
        build_bundle(
            fitted_regression_pipeline,
            input_stage="raw",
            feature_order=("x", "z"),
            options=ExecutionOptions("pandas", state_max_bytes=len(bundle.feature_state)),
        )


def test_bundle_identity_ignores_pickle_protocol(fitted_regression_pipeline, tmp_path):
    """Wire checksums may change without changing learned model or pipeline semantics."""
    from skyulf.inference._manifest import checksum

    bundle = build_bundle(fitted_regression_pipeline, input_stage="raw", feature_order=("x", "z"))
    payload = pickle.dumps(pickle.loads(bundle.model_payload), protocol=4)
    changed = replace(
        bundle,
        model_payload=payload,
        manifest=bundle.manifest.model_copy(update={"model_sha256": checksum(payload)}),
    )
    save_bundle(changed, tmp_path / "protocol4")
    loaded = load_bundle(tmp_path / "protocol4")
    assert loaded.semantic_digest == bundle.semantic_digest
    np.testing.assert_allclose(
        predict_local(pd.DataFrame({"x": [150.0], "z": [9.0]}), loaded).prediction, [280.0]
    )


def test_explicit_state_budget_reaches_export_and_raw_apply(
    fitted_regression_pipeline, raw_frame, monkeypatch
):
    """Every FE operation must use the bundle caller's budget rather than reverting to defaults."""
    monkeypatch.setattr(_feature_state, "DEFAULT_MAX_STATE_BYTES", 128)
    options = ExecutionOptions("pandas", state_max_bytes=64 * 1024)
    bundle = build_bundle(
        fitted_regression_pipeline, input_stage="raw", feature_order=("x", "z"), options=options
    )
    np.testing.assert_allclose(
        predict_local(raw_frame, bundle, options=options).prediction, [280.0, 158.0]
    )
    np.testing.assert_allclose(
        predict_local(pl.from_pandas(raw_frame), bundle, options=options).prediction, [280.0, 158.0]
    )


def test_large_protocol_five_array_serializes_with_real_byte_budget():
    """Protocol-five PickleBuffer chunks must work and be measured in bytes, not elements."""
    from skyulf.inference._model import serialize_model

    array = np.arange(100000, dtype=np.float64)
    payload = serialize_model(array, array.nbytes + 4096)
    np.testing.assert_array_equal(pickle.loads(payload), array)
    with pytest.raises(ValueError, match="model_max_bytes"):
        serialize_model(array, array.nbytes - 1)


def test_large_fitted_knn_bundle_round_trip(tmp_path):
    """Real fitted models with large NumPy buffers must survive public bundle persistence."""
    rng = np.random.default_rng(17)
    frame = pd.DataFrame(rng.normal(size=(512, 20)), columns=[f"x{i}" for i in range(20)])
    columns = tuple(frame.columns)
    frame["target"] = np.arange(len(frame), dtype=float)
    pipeline = SkyulfPipeline(
        {
            "preprocessing": [],
            "modeling": {
                "type": "k_neighbors_regressor",
                "params": {"n_neighbors": 1, "algorithm": "brute"},
            },
        }
    )
    pipeline.fit(SplitDataset(train=frame, test=frame.head(0)), target_column="target")
    bundle = build_bundle(pipeline, input_stage="features", feature_order=columns)
    save_bundle(bundle, tmp_path / "knn")
    output = predict_local(frame[list(columns)].head(3), load_bundle(tmp_path / "knn"))
    np.testing.assert_allclose(output.prediction, [0.0, 1.0, 2.0])
    assert len(bundle.model_payload) > 65536


def test_spark_input_is_rejected_without_driver_collection(
    spark, fitted_regression_pipeline, monkeypatch
):
    """The local prediction entry point must never silently collect a distributed input."""
    bundle = build_bundle(fitted_regression_pipeline, input_stage="raw", feature_order=("x", "z"))
    frame = spark.range(3)

    def forbidden(*args, **kwargs):
        """Rejection must happen before any driver materialization action."""
        pytest.fail("Collected Spark input")

    monkeypatch.setattr(type(frame), "collect", forbidden)
    monkeypatch.setattr(type(frame), "toPandas", forbidden)
    with pytest.raises(TypeError, match="Spark"):
        predict_local(frame, bundle)
