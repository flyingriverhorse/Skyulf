"""XGBoost must retain user target labels through fitting, tuning, and artifacts."""

import pickle

import numpy as np
import pandas as pd
import pytest
from sklearn.base import clone
from sklearn.datasets import make_classification
from sklearn.metrics import log_loss
from sklearn.model_selection import StratifiedKFold
from sklearn.preprocessing import LabelEncoder
from sklearn.utils.class_weight import compute_sample_weight

xgboost = pytest.importorskip("xgboost")

from skyulf.data.dataset import SplitDataset
from skyulf.modeling._tuning.engine import TuningCalculator
from skyulf.modeling._tuning.schemas import TuningConfig
from skyulf.modeling.classification import XGBClassifierApplier, XGBClassifierCalculator
from skyulf.modeling.cross_validation import perform_cross_validation
from skyulf.pipeline import SkyulfPipeline
from skyulf.pipeline.seal import artifact_digest

LABELS = [["c", "a", "b"], [0, 2], [0, 1]]
PARAMS = {"n_estimators": 6, "max_depth": 2, "n_jobs": 1, "random_state": 7}


def _data(labels):
    """Supply separable classes in an order different from the fitted class axis."""
    features, target = make_classification(
        n_samples=72,
        n_features=4,
        n_redundant=0,
        n_classes=len(labels),
        n_clusters_per_class=1,
        class_sep=2,
        random_state=7,
    )
    return pd.DataFrame(features, columns=list("abcd")), pd.Series(np.asarray(labels)[target])


@pytest.mark.parametrize("labels", LABELS)
@pytest.mark.parametrize("report_iterations", [False, True])
def test_xgb_calculator_preserves_class_axis_weights_and_native_margins(labels, report_iterations):
    """Encoding must preserve original-label weights, probability columns, and raw margins."""
    X, y = _data(labels)
    class_weight = {label: float(index + 1) for index, label in enumerate(labels)}
    sample_weight = np.linspace(0.5, 1.5, len(y))
    events = []
    calculator = XGBClassifierCalculator()
    model = calculator.fit(
        X,
        y,
        {"params": {**PARAMS, "class_weight": class_weight}},
        iteration_callback=(lambda *event, **kwargs: events.append(event))
        if report_iterations
        else None,
        sample_weight=sample_weight,
    )
    encoder = LabelEncoder().fit(y)
    oracle = xgboost.XGBClassifier(**{**calculator.default_params, **PARAMS}).fit(
        X.to_numpy(),
        encoder.transform(y),
        sample_weight=sample_weight * compute_sample_weight(class_weight, y),
    )

    np.testing.assert_array_equal(model.classes_, encoder.classes_)
    np.testing.assert_array_equal(model.predict(X), encoder.inverse_transform(oracle.predict(X)))
    np.testing.assert_allclose(model.predict_proba(X), oracle.predict_proba(X))
    np.testing.assert_allclose(
        model.predict(X, output_margin=True), oracle.predict(X, output_margin=True)
    )
    probabilities = XGBClassifierApplier().predict_proba(X, model)
    assert probabilities is not None
    assert probabilities.columns.tolist() == [str(label) for label in encoder.classes_]
    np.testing.assert_array_equal(
        XGBClassifierApplier().predict(X, model),
        encoder.classes_[np.argmax(probabilities.to_numpy(), axis=1)],
    )
    if report_iterations:
        assert len(events) == PARAMS["n_estimators"]
        assert model.callbacks is None
    restored = pickle.loads(pickle.dumps(model))
    np.testing.assert_array_equal(restored.classes_, encoder.classes_)
    np.testing.assert_array_equal(restored.predict(X), model.predict(X))


@pytest.mark.parametrize("labels", LABELS[:2])
def test_xgb_model_class_remains_cloneable_and_relearns_labels(labels):
    """Search clones and subsequent fits need fresh encoders without changing parameter names."""
    X, y = _data(labels)
    estimator = XGBClassifierCalculator().model_class(**PARAMS)
    cloned = clone(estimator).set_params(max_depth=3)
    cloned.fit(X, y, eval_set=[(X, y)], verbose=False)
    assert cloned.get_params()["max_depth"] == 3
    assert cloned.get_params()["n_estimators"] == PARAMS["n_estimators"]
    np.testing.assert_array_equal(cloned.classes_, np.unique(y))
    replacement = pd.Series(np.where(y == labels[0], "left", "right"))
    cloned.fit(X, replacement)
    assert set(cloned.predict(X)) == {"left", "right"}
    np.testing.assert_array_equal(cloned.classes_, ["left", "right"])


@pytest.mark.parametrize("labels", LABELS[:2])
@pytest.mark.parametrize("strategy", ["grid", "random", "halving_grid", "halving_random", "optuna"])
def test_xgb_tuning_scores_original_labels_in_every_strategy(labels, strategy):
    """CV scorers, native pruning callbacks, and final refit must share the label axis."""
    if strategy == "optuna":
        pytest.importorskip("optuna")
    X, y = _data(labels)
    calculator = XGBClassifierCalculator()
    calculator.default_params.update(PARAMS)
    # Halving permutes even full-resource fold rows; disable row sampling
    # so its model has the same independent fold oracle as the other strategies.
    calculator.default_params.update(subsample=1.0, colsample_bytree=1.0)
    model, result = TuningCalculator(calculator).fit(
        X,
        y,
        TuningConfig(
            strategy=strategy,
            metric="neg_log_loss",
            search_space={"max_depth": [2], "class_weight": ["balanced"]},
            n_trials=1,
            cv_folds=2,
            cv_type="stratified_k_fold",
            cv_shuffle=False,
            random_state=7,
            strategy_params={"min_resources": len(y)},
        ),
    )
    direct = calculator.fit(X, y, {"params": {"class_weight": "balanced"}})
    fold_losses = []
    for train, valid in StratifiedKFold(2, shuffle=False).split(X, y):
        fold_model = calculator.fit(
            X.iloc[train], y.iloc[train], {"params": {"class_weight": "balanced"}}
        )
        fold_losses.append(
            log_loss(y.iloc[valid], fold_model.predict_proba(X.iloc[valid]), labels=np.unique(y))
        )
    np.testing.assert_array_equal(model.classes_, np.unique(y))
    np.testing.assert_allclose(model.predict_proba(X), direct.predict_proba(X))
    assert set(model.predict(X)) == set(labels)
    assert np.isfinite(result.best_score)
    assert result.best_score < 0
    assert result.best_score == pytest.approx(-np.mean(fold_losses))
    assert result.best_params == {"max_depth": 2, "class_weight": "balanced"}


@pytest.mark.parametrize("labels", LABELS[:2])
@pytest.mark.parametrize("engine", ["pandas", "polars"])
def test_xgb_pipeline_retains_labels_after_save_load(tmp_path, labels, engine):
    """The public pipeline must train and reload without exposing native encoded targets."""
    X, y = _data(labels)
    frame = X.assign(target=y)
    pipeline = SkyulfPipeline(
        {"preprocessing": [], "modeling": {"type": "xgboost_classifier", "params": PARAMS}}
    )
    train, test = frame, frame.iloc[:12]
    if engine == "polars":
        pl = pytest.importorskip("polars")
        train, test, X = pl.from_pandas(train), pl.from_pandas(test), pl.from_pandas(X)
    pipeline.fit(SplitDataset(train=train, test=test), target_column="target")
    predictions = pipeline.predict(X)
    assert pipeline.model_estimator is not None
    model = pipeline.model_estimator.model
    assert model is not None
    probabilities = pipeline.model_estimator.applier.predict_proba(X, model)
    assert probabilities is not None
    fingerprint = pipeline.fingerprint()
    path = tmp_path / "xgb-labels.pkl"
    pipeline.save(str(path))
    restored = SkyulfPipeline.load(str(path))

    assert set(predictions) == set(labels)
    assert probabilities.columns.tolist() == [str(label) for label in np.unique(y)]
    np.testing.assert_array_equal(restored.predict(X), predictions)
    assert restored.model_estimator is not None
    restored_probabilities = restored.model_estimator.applier.predict_proba(
        X, restored.model_estimator.model
    )
    assert restored_probabilities is not None
    np.testing.assert_allclose(np.asarray(restored_probabilities), np.asarray(probabilities))
    assert restored.fingerprint() == fingerprint
    model.classes_[:] = model.classes_[::-1]
    assert pipeline.fingerprint() != fingerprint
    assert not np.array_equal(pipeline.predict(X), predictions)


@pytest.mark.parametrize("labels", [[0, 1, 2], ["a", "b", "c"]])
def test_xgb_unshuffled_cv_handles_a_class_absent_from_each_training_fold(labels):
    """A held-out middle class must leave a trainable {0, 2} fold with honest zero accuracy."""
    X = pd.DataFrame({"x": np.arange(36, dtype=float)})
    y = pd.Series(np.repeat(labels, 12))
    result = perform_cross_validation(
        XGBClassifierCalculator(),
        XGBClassifierApplier(),
        X,
        y,
        {"params": PARAMS},
        n_folds=3,
        shuffle=False,
    )

    assert len(result["folds"]) == 3
    assert [fold["metrics"]["accuracy"] for fold in result["folds"]] == [0.0, 0.0, 0.0]
    assert result["aggregated_metrics"]["accuracy"]["mean"] == 0.0


@pytest.mark.parametrize("engine", ["pandas", "polars"])
def test_xgb_local_mlflow_artifact_preserves_original_labels(tmp_path, monkeypatch, engine):
    """The real MLflow pyfunc package must reload the encoded estimator and original class axis."""
    mlflow = pytest.importorskip("mlflow")
    from skyulf.inference.local_pipeline import load_local_pipeline, save_local_pipeline
    from skyulf.integrations.mlflow.local_model import log_local_model
    from skyulf.integrations.mlflow.tracking import TrackingConfig, track_run

    monkeypatch.chdir(tmp_path)
    X, y = _data(LABELS[0])
    train = X.assign(target=y)
    if engine == "polars":
        pl = pytest.importorskip("polars")
        train = pl.from_pandas(train)
    pipeline = SkyulfPipeline(
        {"preprocessing": [], "modeling": {"type": "xgboost_classifier", "params": PARAMS}}
    )
    pipeline.fit(SplitDataset(train=train, test=train.head(0)), target_column="target")
    artifact_path = tmp_path / "local-artifact"
    save_local_pipeline(pipeline, artifact_path)
    tracking_uri = f"sqlite:///{(tmp_path / 'tracking.db').as_posix()}"
    previous_uri = mlflow.get_tracking_uri()
    try:
        config = TrackingConfig(enabled=True, tracking_uri=tracking_uri, experiment_name="xgb")
        with track_run(config, run_name="original-labels") as run:
            assert run.run_id is not None
            model_uri = log_local_model(
                artifact_path,
                run_id=run.run_id,
                artifact_path="model",
                tracking_uri=tracking_uri,
            )
        mlflow.set_tracking_uri(tracking_uri)
        result = mlflow.pyfunc.load_model(model_uri).predict(X)
    finally:
        mlflow.set_tracking_uri(previous_uri)

    artifact = load_local_pipeline(artifact_path)
    assert artifact.manifest.classes == ("a", "b", "c")
    np.testing.assert_array_equal(result["prediction"], pipeline.predict(X))
    assert pipeline.model_estimator is not None
    expected_probabilities = pipeline.model_estimator.applier.predict_proba(
        X, pipeline.model_estimator.model
    )
    assert expected_probabilities is not None
    np.testing.assert_allclose(
        np.asarray(result[["probability_0", "probability_1", "probability_2"]]),
        np.asarray(expected_probabilities),
    )


@pytest.mark.parametrize("labels", LABELS)
def test_xgb_native_booster_roundtrip_preserves_original_labels(tmp_path, labels):
    """Native booster files must retain the estimator's target values and probability axis."""
    X, y = _data(labels)
    calculator = XGBClassifierCalculator()
    model = calculator.fit(X, y, {"params": PARAMS})
    path = tmp_path / "original-labels.ubj"
    model.get_booster().save_model(path)
    restored = calculator.model_class(**PARAMS)
    restored.load_model(path)
    digest = artifact_digest(restored)

    np.testing.assert_array_equal(restored.classes_, model.classes_)
    assert restored.classes_.dtype == model.classes_.dtype
    np.testing.assert_array_equal(restored.predict(X), model.predict(X))
    np.testing.assert_array_equal(restored.predict_proba(X), model.predict_proba(X))
    assert artifact_digest(restored) == digest


@pytest.mark.parametrize("prefitted", [False, True])
def test_xgb_legacy_native_booster_load_uses_numeric_classes(tmp_path, prefitted):
    """Legacy native files without Skyulf metadata must clear any previous string mapping."""
    X, y = _data([0, 1])
    native = xgboost.XGBClassifier(**PARAMS).fit(X, y)
    path = tmp_path / "legacy-labels.ubj"
    native.get_booster().save_model(path)
    restored = XGBClassifierCalculator().model_class(**PARAMS)
    if prefitted:
        restored.fit(X, y.map({0: "no", 1: "yes"}))
    restored.load_model(path)

    np.testing.assert_array_equal(restored.classes_, native.classes_)
    np.testing.assert_array_equal(restored.predict(X), native.predict(X))
    np.testing.assert_array_equal(restored.predict_proba(X), native.predict_proba(X))


@pytest.mark.parametrize("route", ["calculator", "pipeline"])
@pytest.mark.parametrize("invalid", ["continuous", "nan", "none"])
def test_xgb_public_fit_rejects_nonclassification_targets(route, invalid):
    """Encoding must not disguise regression values or missing labels as valid classes."""
    X, _ = _data([0, 1])
    labels = {
        "continuous": np.linspace(0.1, 0.9, len(X)),
        "nan": np.tile([0.0, 1.0, np.nan], len(X) // 3),
        "none": np.tile(["a", "b", None], len(X) // 3),
    }
    y = pd.Series(labels[invalid])
    with pytest.raises((ValueError, TypeError)):
        if route == "calculator":
            XGBClassifierCalculator().fit(X, y, {"params": PARAMS})
        else:
            pipeline = SkyulfPipeline(
                {"preprocessing": [], "modeling": {"type": "xgboost_classifier", "params": PARAMS}}
            )
            frame = X.assign(target=y)
            pipeline.fit(SplitDataset(train=frame, test=frame.head(0)), target_column="target")


@pytest.mark.parametrize("engine", ["pandas", "polars"])
@pytest.mark.parametrize("labels", LABELS[:2])
def test_xgb_explicit_target_encoder_preserves_core_and_artifact_contracts(
    tmp_path, engine, labels
):
    """An explicit target encoder keeps Core codes and restores original labels for artifacts."""
    from skyulf.inference.local_pipeline import (
        load_local_pipeline,
        predict_local_pipeline,
        save_local_pipeline,
    )

    X, y = _data(labels)
    train = X.assign(target=y)
    if engine == "polars":
        pl = pytest.importorskip("polars")
        train, X = pl.from_pandas(train), pl.from_pandas(X)
    pipeline = SkyulfPipeline(
        {
            "preprocessing": [
                {
                    "name": "target_labels",
                    "transformer": "LabelEncoder",
                    "params": {"columns": ["target"]},
                }
            ],
            "modeling": {"type": "xgboost_classifier", "params": PARAMS},
        }
    )
    pipeline.fit(SplitDataset(train=train, test=train.head(0)), target_column="target")
    core_predictions = np.asarray(pipeline.predict(X))
    assert pipeline.model_estimator is not None
    model = pipeline.model_estimator.model
    assert model is not None
    core_probabilities = np.asarray(pipeline.model_estimator.applier.predict_proba(X, model))
    target_encoder = LabelEncoder().fit(y.astype(str))
    original_classes = target_encoder.classes_.astype(y.dtype)
    path = tmp_path / "encoded-target"
    save_local_pipeline(pipeline, path)
    artifact = load_local_pipeline(path)
    served = predict_local_pipeline(X, artifact)

    np.testing.assert_array_equal(model.classes_, np.arange(len(labels)))
    assert set(core_predictions) == set(range(len(labels)))
    np.testing.assert_array_equal(served["prediction"], original_classes[core_predictions])
    assert artifact.manifest.classes == tuple(original_classes)
    probability_columns = [f"probability_{index}" for index in range(len(labels))]
    np.testing.assert_allclose(served[probability_columns].to_numpy(), core_probabilities)
