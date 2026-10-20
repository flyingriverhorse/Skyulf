"""Exported scoring must preserve class probabilities and the recorded raw types."""

import hashlib
import importlib.util
import inspect
import json
import pickle
import sys
from datetime import date
from decimal import Decimal

import numpy as np
import pandas as pd
import pytest
from fastapi.testclient import TestClient
from sklearn.linear_model import LinearRegression, LogisticRegression
from sklearn.preprocessing import LabelEncoder
from sklearn.svm import LinearSVC

from backend.ml_pipeline.project_export import runtime, templates
from skyulf.modeling import apply_thresholds
from skyulf.preprocessing.pipeline import FeatureEngineer


def _write_runtime_project(directory, model, engineer, *, decoder=None, thresholds=None):
    """Write a real fitted payload using the exported runtime's on-disk contract."""
    directory.mkdir(parents=True, exist_ok=True)
    state = {
        "model": model,
        "feature_engineer": engineer,
        "feature_columns": ["x"],
        "target_encoder": decoder,
        "thresholds": thresholds,
    }
    payload = pickle.dumps(state)
    (directory / "model.pkl").write_bytes(payload)
    (directory / "schema.json").write_text(json.dumps([{"name": "x", "dtype": "Float64"}]))
    (directory / "manifest.json").write_text(
        json.dumps(
            {
                "job_id": "typed-model",
                "model_sha256": hashlib.sha256(payload).hexdigest(),
            }
        )
    )
    return runtime.Predictor(directory)


def _fit_project(directory, *, classifier=True, encoded=False, thresholds=False, no_proba=False):
    """Fit nonconstant predictions and a real imputer for probability/null assertions."""
    frame = pd.DataFrame({"x": [0.0, 1.0, np.nan, 3.0, 4.0, 5.0]})
    engineer = FeatureEngineer(
        [
            {"name": "fill", "transformer": "SimpleImputer", "params": {"columns": ["x"]}},
        ]
    )
    features, _ = engineer.fit_transform(frame)
    labels = ["stay", "stay", "stay", "leave", "leave", "leave"]
    decoder = LabelEncoder().fit(labels) if encoded else None
    target = decoder.transform(labels) if decoder is not None else labels
    if classifier:
        model = (LinearSVC() if no_proba else LogisticRegression()).fit(features, target)
    else:
        model = LinearRegression().fit(features, [1.0, 3.0, 5.0, 7.0, 9.0, 11.0])
    weights = (
        {str(label): value for label, value in zip(model.classes_, [0.8, 0.2], strict=True)}
        if thresholds
        else None
    )
    return _write_runtime_project(directory, model, engineer, decoder=decoder, thresholds=weights)


@pytest.mark.parametrize("encoded", [False, True])
@pytest.mark.parametrize("thresholds", [False, True])
def test_probabilities_follow_decoded_class_order_and_saved_thresholds(
    tmp_path, encoded, thresholds
):
    """Probabilities retain estimator order while labels use the saved decoder and thresholds."""
    predictor = _fit_project(tmp_path, encoded=encoded, thresholds=thresholds)
    rows = pd.DataFrame({"x": ["0.2", None, "4.2"]}, dtype=object)
    result = predictor.predict(rows)
    expected_frame = pd.DataFrame({"x": [0.2, 2.6, 4.2]})
    model = predictor.state["model"]
    expected_proba = model.predict_proba(expected_frame)
    classes = model.classes_
    decoder = predictor.state["target_encoder"]
    expected_labels = decoder.inverse_transform(classes) if decoder is not None else classes
    assert predictor.probability_labels() == list(expected_labels)
    assert list(result["probabilities"][0]) == list(expected_labels)
    np.testing.assert_allclose(
        np.array([list(row.values()) for row in result["probabilities"]]), expected_proba
    )
    if thresholds:
        weights = {label: predictor.state["thresholds"][str(label)] for label in classes}
        expected = apply_thresholds(expected_proba, weights, classes=classes)
    else:
        expected = model.predict(expected_frame)
    if decoder is not None:
        expected = decoder.inverse_transform(expected)
    assert result["predictions"] == expected.tolist()


@pytest.mark.parametrize("classifier,no_proba", [(False, False), (True, True)])
def test_models_without_probability_support_keep_prediction_contract(
    tmp_path, classifier, no_proba
):
    """Regressors and uncalibrated classifiers remain usable without fabricated probabilities."""
    predictor = _fit_project(tmp_path, classifier=classifier, no_proba=no_proba)
    result = predictor.predict(pd.DataFrame({"x": [0.1, 4.9]}))
    assert len(result["predictions"]) == 2
    assert predictor.probability_labels() is None
    assert "probabilities" not in result


def test_empty_classifier_retains_probability_schema(tmp_path):
    """Empty batch files still expose the fitted class columns without scoring dummy rows."""
    predictor = _fit_project(tmp_path, encoded=True)
    result = predictor.predict(pd.DataFrame(columns=["x"]))
    assert predictor.probability_labels() == ["leave", "stay"]
    assert result["probabilities"] == []
    assert result["predictions"] == []


def test_colliding_probability_keys_fail_explicitly(tmp_path):
    """Stringified class labels cannot silently overwrite another class's probability."""
    predictor = _fit_project(tmp_path)
    predictor.state["model"].classes_ = np.array([1, "1"], dtype=object)
    with pytest.raises(ValueError, match="unique"):
        predictor.predict(pd.DataFrame({"x": [1.0]}))


@pytest.mark.parametrize("dtype", ["int64", "Int64", "UInt64", "int64[pyarrow]"])
def test_integer_schema_preserves_large_values_and_null(dtype):
    """Schema conversion must not route exact integer text through a floating dtype."""
    frame = pd.DataFrame({"id": ["9007199254740993", None]}, dtype=object)
    actual = runtime.coerce_inputs(frame, [{"name": "id", "dtype": dtype}])
    assert actual["id"].iloc[0] == 9007199254740993
    assert pd.isna(actual["id"].iloc[1])
    assert frame["id"].iloc[0] == "9007199254740993"


@pytest.mark.parametrize(
    "dtype,values,expected",
    [
        ("UInt64", ["18446744073709551615"], 18446744073709551615),
        ("float64", ["1.5", None], 1.5),
        ("Float32", ["1.5", None], 1.5),
        ("boolean", ["false", None], False),
        ("Boolean", ["true", None], True),
        ("String", ["London", None], "London"),
        ("string[python]", ["London", None], "London"),
        ("Date", ["2024-01-02", None], date(2024, 1, 2)),
        ("datetime64[ns]", ["2024-01-02T03:04:05", None], pd.Timestamp("2024-01-02T03:04:05")),
        (
            "Datetime(time_unit='us', time_zone='UTC')",
            ["2024-01-02T03:04:05Z", None],
            pd.Timestamp("2024-01-02T03:04:05Z"),
        ),
        (
            "Decimal(precision=20, scale=3)",
            ["9007199254740993.125", None],
            Decimal("9007199254740993.125"),
        ),
        ("object", ["001", None], "001"),
        ("category", ["001", None], "001"),
        ("unknown", ["001", None], "001"),
    ],
)
def test_recorded_pandas_and_polars_types_coerce_without_guessing(dtype, values, expected):
    """The original dtype controls conversion while unconstrained object/category values survive."""
    actual = runtime.coerce_inputs(
        pd.DataFrame({"value": values}, dtype=object), [{"name": "value", "dtype": dtype}]
    )
    assert actual["value"].iloc[0] == expected
    if len(values) > 1:
        assert pd.isna(actual["value"].iloc[1])


@pytest.mark.parametrize(
    "dtype,value",
    [
        ("Int64", "1.5"),
        ("Int64", "9223372036854775808"),
        ("UInt64", "-1"),
        ("Int8", "128"),
        ("Int64", float(2**53 + 2)),
        ("Int64", float(2**53)),
        ("Float64", "nan"),
        ("Float64", "inf"),
        ("Float32", "1e100"),
        ("Float64", True),
        ("Boolean", "maybe"),
        ("Boolean", 2),
        ("String", {"nested": "value"}),
        ("Date", "yesterday"),
        ("Date", 42),
        ("datetime64[ns]", "invalid"),
        ("Decimal(precision=10, scale=2)", "NaN"),
    ],
)
def test_invalid_raw_values_name_the_column_and_recorded_type(dtype, value):
    """Invalid conversions must fail before estimator input and identify the raw column."""
    with pytest.raises(ValueError, match="value.*" + dtype.split("(")[0].split("[")[0]):
        runtime.coerce_inputs(
            pd.DataFrame({"value": [value]}, dtype=object), [{"name": "value", "dtype": dtype}]
        )


@pytest.mark.parametrize(
    "text",
    ["1234567890123456789012345678901.123456", "9999999999999999999999999999999999.123456"],
)
def test_decimal_scale_validation_never_rounds_high_precision_values(text):
    """Decimal context rounding must not hide a fractional digit beyond the recorded scale."""
    with pytest.raises(ValueError, match="precision or scale"):
        runtime.coerce_inputs(
            pd.DataFrame({"value": [text]}),
            [{"name": "value", "dtype": "Decimal(precision=38, scale=5)"}],
        )


def test_typed_request_reads_schema_only_and_preserves_aliases(tmp_path):
    """OpenAPI creation must not load pickle and must support reserved/raw column names."""
    schema = [
        {"name": "account id", "dtype": "Int64"},
        {"name": "_group", "dtype": "String"},
        {"name": "model_dump", "dtype": "Boolean"},
    ]
    (tmp_path / "schema.json").write_text(json.dumps(schema))
    request = runtime.build_request_model(tmp_path)
    parsed = request.model_validate(
        {
            "data": [
                {
                    "account id": "9007199254740993",
                    "_group": None,
                    "model_dump": "false",
                    "extra": 2,
                }
            ]
        }
    )
    row = parsed.model_dump(by_alias=True)["data"][0]
    assert row == {"account id": 9007199254740993, "_group": None, "model_dump": False, "extra": 2}
    properties = request.model_json_schema()["$defs"]["InputRow"]["properties"]
    assert set(properties) == {"account id", "_group", "model_dump"}
    assert properties["account id"]["anyOf"] == [{"type": "integer"}, {"type": "null"}]


def test_generated_api_exposes_typed_rows_and_validates_real_requests(tmp_path, monkeypatch):
    """The shipped API must advertise numeric inputs, coerce strings, preserve nulls and return 422."""
    _fit_project(tmp_path)
    (tmp_path / "scoring.py").write_text(inspect.getsource(runtime), encoding="utf-8")
    (tmp_path / "app.py").write_text(templates.API, encoding="utf-8")
    monkeypatch.syspath_prepend(str(tmp_path))
    monkeypatch.delitem(sys.modules, "scoring", raising=False)
    spec = importlib.util.spec_from_file_location("typed_export_app", tmp_path / "app.py")
    assert spec and spec.loader
    app_module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(app_module)
    with TestClient(app_module.app) as client:
        good = client.post("/predict", json={"data": [{"x": "1.25"}, {"x": None}]})
        bad = client.post("/predict", json={"data": [{"x": "not-a-number"}]})
        missing = client.post("/predict", json={"data": [{}]})
        document = client.get("/openapi.json").json()
    assert good.status_code == 200 and len(good.json()["probabilities"]) == 2
    assert bad.status_code == 422 and "x" in str(bad.json()["detail"])
    assert missing.status_code == 422
    row_schema = document["components"]["schemas"]["InputRow"]
    assert row_schema["required"] == ["x"]
    assert row_schema["properties"]["x"]["anyOf"] == [{"type": "number"}, {"type": "null"}]
