"""Original target labels must survive encoding, decisions and persisted evaluation."""

import numpy as np
import pandas as pd
import polars as pl
import pytest
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import balanced_accuracy_score, log_loss, roc_auc_score
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from test_bundle_decision_thresholds import fixture_frame, recipe

from skyulf.data.dataset import SplitDataset
from skyulf.engines.pandas_engine import SkyulfPandasWrapper
from skyulf.engines.polars_engine import SkyulfPolarsWrapper
from skyulf.inference.local_evaluation import evaluate_local_holdout
from skyulf.inference.local_pipeline import load_local_pipeline, predict_local_pipeline
from skyulf.integrations.databricks.local_batch import fit_local_workflow
from skyulf.integrations.databricks.local_cv import LocalCVSpec, evaluate_training_cv
from skyulf.integrations.databricks.local_search import prepare_search_pipeline
from skyulf.preprocessing.encoding.label import LabelEncoderApplier, LabelEncoderCalculator
from skyulf.preprocessing.encoding.ordinal import OrdinalEncoderApplier, OrdinalEncoderCalculator


@pytest.mark.parametrize("engine", ["pandas", "polars"])
@pytest.mark.parametrize("classes", [2, 3])
@pytest.mark.parametrize("mode", ["off", "manual", "auto"])
@pytest.mark.parametrize("encoder", ["label_default", "label_named", "ordinal"])
def test_encoded_target_keeps_original_decisions_and_metrics(
    tmp_path, engine, classes, mode, encoder
):
    """Users must never substitute hidden integer codes for original threshold labels."""
    frame = fixture_frame(classes)
    labels = [f"class_{index}" for index in range(classes)]
    policy = {"mode": mode}
    if mode == "manual":
        policy.update(
            thresholds=[{"class": value, "value": 0.2 + i * 0.3} for i, value in enumerate(labels)]
        )
    config = recipe(policy)
    params: dict = {} if encoder == "label_default" else {"columns": ["target"]}
    if encoder == "ordinal":
        params["categories_order"] = ",".join(reversed(labels))
    config["preprocessing"].append(
        {
            "name": "encode",
            "transformer": "OrdinalEncoder" if encoder == "ordinal" else "LabelEncoder",
            "params": params,
        }
    )
    native = pl.from_pandas(frame) if engine == "polars" else frame
    train, holdout = (
        (native[:180], native[180:])
        if engine == "polars"
        else (native.iloc[:180], native.iloc[180:])
    )
    path = tmp_path / "model"
    fitted = fit_local_workflow(
        config,
        SplitDataset(train=train, test=holdout),
        target_column="target",
        artifact_path=path,
        max_rows=1000,
        max_bytes=10_000_000,
    )
    artifact = load_local_pipeline(path)
    features = frame.iloc[180:].drop(columns="target")
    prediction = predict_local_pipeline(features, artifact)
    assert set(artifact.manifest.classes) == set(labels)
    assert set(prediction.prediction).issubset(set(labels))
    np.testing.assert_array_equal(prediction, predict_local_pipeline(features, fitted))
    probabilities = prediction.filter(like="probability_").to_numpy()
    order = list(artifact.manifest.classes)
    expected_order = list(reversed(labels)) if encoder == "ordinal" else labels
    assert order == expected_order
    reference_train = frame.iloc[:180]
    if mode == "auto":
        reference_train, _ = train_test_split(
            reference_train, test_size=0.2, random_state=42, stratify=reference_train.target
        )
    reference_scaler = StandardScaler().fit(reference_train[list("abcd")])
    reference_model = LogisticRegression(max_iter=500).fit(
        reference_scaler.transform(reference_train[list("abcd")]),
        reference_train.target.map({label: code for code, label in enumerate(expected_order)}),
    )
    np.testing.assert_allclose(
        probabilities,
        reference_model.predict_proba(reference_scaler.transform(features)),
        atol=1e-10,
        rtol=0,
    )
    if mode == "manual":
        weights = {entry["class"]: entry["value"] for entry in policy["thresholds"]}
        expected = np.asarray(order)[
            np.argmax(probabilities / [weights[label] for label in order], axis=1)
        ]
        np.testing.assert_array_equal(prediction.prediction, expected)
    metrics = evaluate_local_holdout(artifact, holdout, target_column="target")
    actual = frame.iloc[180:].target
    assert metrics["heldout_balanced_accuracy"] == pytest.approx(
        balanced_accuracy_score(actual, prediction.prediction)
    )
    sorted_positions = np.argsort(order)
    sorted_probabilities = probabilities[:, sorted_positions]
    assert metrics["heldout_log_loss"] == pytest.approx(
        log_loss(actual, sorted_probabilities, labels=sorted(order))
    )
    if classes == 3:
        assert metrics["heldout_roc_auc_ovr"] == pytest.approx(
            roc_auc_score(actual, sorted_probabilities, labels=sorted(order), multi_class="ovr")
        )


def test_original_binary_positive_class_is_resolved_after_encoding(tmp_path):
    """A first original class remains positive even when ordinal codes reverse its order."""
    frame = fixture_frame()
    config = recipe({"mode": "manual", "positive_class": "class_0", "value": 0.7})
    config["preprocessing"].append(
        {
            "name": "encode",
            "transformer": "OrdinalEncoder",
            "params": {"columns": ["target"], "categories_order": "class_1,class_0"},
        }
    )
    artifact = fit_local_workflow(
        config,
        SplitDataset(train=frame.iloc[:180], test=frame.iloc[180:]),
        target_column="target",
        artifact_path=tmp_path / "model",
        max_rows=1000,
        max_bytes=10_000_000,
    )
    prediction = predict_local_pipeline(frame.iloc[180:].drop(columns="target"), artifact)
    position = artifact.manifest.classes.index("class_0")
    expected = np.where(prediction[f"probability_{position}"] >= 0.7, "class_0", "class_1")
    np.testing.assert_array_equal(prediction.prediction, expected)


@pytest.mark.parametrize("engine", ["pandas", "polars"])
@pytest.mark.parametrize("target_first", [False, True])
def test_ordinal_feature_and_target_orders_follow_column_names(engine, target_first):
    """Separating y must not shift the explicitly configured category orders."""
    frame = pd.DataFrame({"feature": ["a", "b"], "target": ["yes", "no"]})
    if engine == "polars":
        frame = pl.from_pandas(frame)
    columns = ["target", "feature"] if target_first else ["feature", "target"]
    order = "yes,no\nb,a" if target_first else "b,a\nyes,no"
    params = OrdinalEncoderCalculator().fit(
        frame, {"columns": columns, "target_column": "target", "categories_order": order}
    )
    result = OrdinalEncoderApplier().apply(frame, params)
    np.testing.assert_array_equal(np.asarray(result["feature"]), [1, 0])
    np.testing.assert_array_equal(np.asarray(result["target"]), [0, 1])


@pytest.mark.parametrize("engine", ["pandas", "polars"])
@pytest.mark.parametrize("encoder", ["label", "ordinal"])
def test_embedded_target_encoding_accepts_public_wrappers(engine, encoder):
    """Target restoration must follow the dispatcher's native/wrapped output contract."""
    frame = pd.DataFrame({"feature": [1, 2], "target": ["yes", "no"]})
    wrapped = (
        SkyulfPolarsWrapper(pl.from_pandas(frame))
        if engine == "polars"
        else SkyulfPandasWrapper(frame)
    )
    calculator, applier = (
        (LabelEncoderCalculator(), LabelEncoderApplier())
        if encoder == "label"
        else (OrdinalEncoderCalculator(), OrdinalEncoderApplier())
    )
    params = calculator.fit(wrapped, {"columns": ["target"], "target_column": "target"})
    result = applier.apply(wrapped, params)
    assert isinstance(result, SkyulfPolarsWrapper if engine == "polars" else pd.DataFrame)
    np.testing.assert_array_equal(np.asarray(result["target"]), [1, 0])


@pytest.mark.parametrize("engine", ["pandas", "polars"])
@pytest.mark.parametrize("kind", [bool, float, int, str])
def test_target_encoding_preserves_original_scalar_types(tmp_path, engine, kind):
    """Numerically equal encoder codes cannot erase original boolean or float labels."""
    frame = fixture_frame()
    frame["target"] = frame.target.map({"class_0": kind(0), "class_1": kind(1)})
    config = recipe({"mode": "auto"})
    config["preprocessing"].append({"name": "encode", "transformer": "LabelEncoder", "params": {}})
    native = pl.from_pandas(frame) if engine == "polars" else frame
    artifact = fit_local_workflow(
        config,
        SplitDataset(train=native, test=native.head(0)),
        target_column="target",
        artifact_path=tmp_path / "model",
        max_rows=1000,
        max_bytes=10_000_000,
    )
    assert all(type(label) is kind for label in artifact.manifest.classes)


@pytest.mark.parametrize("engine", ["pandas", "polars"])
@pytest.mark.parametrize("method", ["random_over", "smote", "random_under_sampling"])
@pytest.mark.parametrize("search", [False, True])
def test_encoded_sampling_cv_and_reload_keep_original_labels(tmp_path, engine, method, search):
    """Row-changing training cannot invalidate label identities in outer CV or saved scoring."""
    pytest.importorskip("imblearn")
    frame = fixture_frame(3)
    frame = frame[(frame.target == "class_0") | (frame.index % 4 == 0)].reset_index(drop=True)
    config = recipe({"mode": "auto", "metric": "f1_macro"})
    config["preprocessing"].extend(
        [
            {
                "name": "encode",
                "transformer": "OrdinalEncoder",
                "params": {"columns": ["target"], "categories_order": "class_2,class_1,class_0"},
            },
            {
                "name": "sample",
                "transformer": "Undersampling"
                if method == "random_under_sampling"
                else "Oversampling",
                "params": {"method": method, "random_state": 42, "k_neighbors": 1},
            },
            {"name": "encode_again", "transformer": "LabelEncoder", "params": {}},
        ]
    )
    cv = LocalCVSpec(enabled=True, folds=2, method="stratified_k_fold")
    if search:
        config["modeling"] = {
            "type": "hyperparameter_tuner",
            "base_model": config["modeling"],
            "strategy": "grid",
            "search_space": {"C": [0.5, 1.0]},
            "metric": "balanced_accuracy",
        }
    config = prepare_search_pipeline(config, cv, target_column="target", event_column=None)
    native = pl.from_pandas(frame) if engine == "polars" else frame
    report = evaluate_training_cv(native, config, cv, target_column="target")
    artifact = fit_local_workflow(
        config,
        SplitDataset(train=native, test=native.head(0)),
        target_column="target",
        artifact_path=tmp_path / "model",
        max_rows=1000,
        max_bytes=10_000_000,
    )
    prediction = predict_local_pipeline(frame.drop(columns="target"), artifact)
    assert set(artifact.manifest.classes) == {"class_0", "class_1", "class_2"}
    assert set(prediction.prediction).issubset(set(frame.target))
    assert report is not None and len(report["decision_threshold_folds"]) == 2
