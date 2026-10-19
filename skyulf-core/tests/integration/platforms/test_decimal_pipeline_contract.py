"""Observed Spark Decimal conversions remain numeric through saved inference contracts."""

import json
from decimal import Decimal
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pandas as pd
import pytest
from sklearn.impute import SimpleImputer
from sklearn.linear_model import Ridge
from test_databricks_decimal_transport import _source, _spark

from skyulf.core.capabilities import UnsupportedExecutionError
from skyulf.inference.fitted_pipeline import predict_pipeline
from skyulf.inference.model_set import (
    ColumnSpec,
    ComponentReference,
    save_model_set,
)
from skyulf.integrations.databricks.jobs.serving_endpoint_job import (
    _smoke_inputs,
    sample_serving_rows,
)
from skyulf.integrations.databricks.serving import (
    PinnedEndpointSpec,
    build_serving_sql_function,
    prepare_pinned_endpoint,
)
from skyulf.integrations.databricks.serving.endpoints import validate_json_rows
from skyulf.integrations.databricks.training.fitting import candidate
from skyulf.integrations.mlflow.registration.registry import (
    download_registered_package,
    load_registered_pipeline,
    register_model,
    resolve_model,
)


def _decimal_rows(count=80):
    """Include missing numeric features so fitted imputation is exercised during prediction."""
    return [
        {
            "id": index,
            "amount": None if index in {2, 7} else Decimal(index) + Decimal("0.25"),
            "exposure": Decimal(index % 4) + Decimal("0.5"),
            "target": Decimal(2 * index) + Decimal("0.5"),
        }
        for index in range(count)
    ]


def _spec():
    """Keep keys outside the model contract and use a reproducible held-out split."""
    return candidate.TrainingSpec(
        table="main.data.decimal_inputs",
        version=3,
        record_key_columns=("id",),
        input_columns=("amount", "exposure"),
        target_column="target",
        max_rows=100,
        max_bytes=100000,
        admit_probed_custom_steps=True,
    )


def _recipe():
    """Use real numeric preprocessing and an estimator with an independent reference fit."""
    return {
        "preprocessing": [
            {
                "name": "fill",
                "transformer": "SimpleImputer",
                "params": {"columns": ["amount", "exposure"], "strategy": "mean"},
            }
        ],
        "modeling": {"type": "ridge_regression", "params": {"alpha": 0.1}},
    }


def _assert_registered_contract(client, uri, name, records, expected, *, engine, model_set=False):
    """Inspect actual MLflow bytes and derive serving/SQL plans from the registered version."""
    import mlflow

    endpoints = {"tracking_uri": uri, "registry_uri": uri}
    resolved = resolve_model(name, version="1", **endpoints)
    package = Path(download_registered_package(mlflow, client, name, "1", uri))
    info = mlflow.models.Model.load(package / "MLmodel")
    signature = {column.name: column.type.name for column in info.signature.inputs.inputs}
    assert signature["amount"] == signature["exposure"] == "double"
    spec = PinnedEndpointSpec("decimal-contract", name, "1", "", "", "", logging_mode="disabled")
    plan = None
    if engine == "pandas":
        plan = prepare_pinned_endpoint(spec, **endpoints)
        assert plan.record_key_columns == (("id",) if model_set else ())
        schema = dict(plan.input_schema)
        assert schema["amount"].lower() == schema["exposure"].lower() == "float64"
    else:
        with pytest.raises(UnsupportedExecutionError, match="pandas"):
            prepare_pinned_endpoint(spec, **endpoints)
    input_schema = tuple(
        (column, {"double": "float64", "long": "int64"}[dtype])
        for column, dtype in signature.items()
    )
    source_types = dict.fromkeys(records[0], "decimal(17,6)")
    if model_set:
        source_types["id"] = "bigint"
    rows = sample_serving_rows(
        _spark(_source(records, source_types)),
        "main.data.decimal_inputs",
        input_schema,
        rows=len(records),
        record_key_columns=("id",) if model_set else (),
    )
    encoded = pd.DataFrame(rows)
    if plan is not None:
        encoded, rows = _smoke_inputs(rows, plan)
    validate_json_rows(rows)
    restored = json.loads(json.dumps({"dataframe_records": rows}, allow_nan=False))
    for row in restored["dataframe_records"]:
        assert row["amount"] is None or type(row["amount"]) is float
        assert type(row["exposure"]) is float
    predictions = mlflow.pyfunc.load_model(str(package)).predict(encoded)
    assert isinstance(predictions, pd.DataFrame)
    output = "amount__prediction" if model_set else "prediction"
    np.testing.assert_allclose(predictions[output], expected, rtol=1e-10, atol=1e-10)
    if plan is not None:
        sql = build_serving_sql_function(plan, "main.api.decimal_predict").create_sql
        assert "`amount` DOUBLE" in sql and "`exposure` DOUBLE" in sql
        assert "COALESCE(`amount`, CAST('NaN' AS DOUBLE))" in sql
    return resolved


@pytest.mark.parametrize("engine", ["pandas", "polars"])
def test_decimal_candidate_and_model_set_keep_numeric_contract(tmp_path, monkeypatch, engine):
    """A real registered candidate and set must agree with numeric Ridge after save/load."""
    mlflow = pytest.importorskip("mlflow")
    from skyulf.integrations.mlflow.models.model_set import log_model_set

    monkeypatch.delenv("DATABRICKS_RUNTIME_VERSION", raising=False)
    types = {"id": "bigint", **dict.fromkeys(("amount", "exposure", "target"), "decimal(17,6)")}
    spark = _spark(_source(_decimal_rows(), types))
    spark.sql.return_value.first.return_value = {"format": "delta", "id": "decimal-source"}
    spec = _spec()
    frame = candidate.read_training_snapshot(spark, spec)
    assert [str(frame[column].dtype) for column in spec.input_columns] == ["float64"] * 2
    conversions = list(frame.attrs["source_column_conversions"])
    train, _, _ = candidate.split_labeled_snapshot(frame, spec, engine=engine)
    inputs = list(spec.input_columns)
    imputer = SimpleImputer(strategy="mean")
    reference = Ridge(alpha=0.1).fit(imputer.fit_transform(train[inputs]), train.target)
    uri = f"sqlite:///{(tmp_path / 'registry.db').as_posix()}"
    client = mlflow.MlflowClient(tracking_uri=uri, registry_uri=uri)
    client.create_experiment("decimal", artifact_location=(tmp_path / "runs").as_uri())
    name = f"main.ml.decimal_{engine}"
    result = candidate.train_candidate(
        spark,
        spec,
        _recipe(),
        model_name=name,
        tracking_uri=uri,
        registry_uri=uri,
        experiment_name="decimal",
        run_name=engine,
        artifact_path=tmp_path / "artifact",
        metric="heldout_rmse",
        min_improvement=0.0,
        engine=engine,
    )
    resolved = resolve_model(name, version="1", tracking_uri=uri, registry_uri=uri)
    artifact = load_registered_pipeline(resolved, tracking_uri=uri, registry_uri=uri)
    assert tuple(dtype.lower() for dtype in artifact.manifest.input_dtypes) == (
        "float64",
        "float64",
    )
    recorded = artifact.manifest.source_column_conversions
    assert recorded is not None
    assert [item.model_dump() for item in recorded] == conversions[:2]
    estimator = artifact.pipeline.model_estimator
    assert estimator is not None and isinstance(estimator._unwrap_tuned_model(), Ridge)
    for filename in ("training_data.json", "dataset_inputs.json"):
        path = client.download_artifacts(result.run_id, filename)
        assert json.loads(Path(path).read_text())["source_column_conversions"] == conversions
    records = [
        {"amount": Decimal("4.25"), "exposure": Decimal("2.5")},
        {"amount": None, "exposure": Decimal("1.5")},
    ]
    query = pd.DataFrame(records).astype("float64")
    expected = reference.predict(imputer.transform(query))
    np.testing.assert_allclose(predict_pipeline(query, artifact).prediction, expected, rtol=1e-10)
    _assert_registered_contract(client, uri, name, records, expected, engine=engine)
    set_path = tmp_path / "model-set"
    save_model_set(
        set_path,
        {
            "amount": (
                ComponentReference(
                    name=name, version="1", digest=artifact.manifest.pipeline_sha256
                ),
                tmp_path / "artifact",
            )
        },
        record_key_schema=(ColumnSpec(name="id", dtype="int64"),),
    )
    set_uri = log_model_set(set_path, run_id=result.run_id, artifact_path="set", tracking_uri=uri)
    set_name = f"main.ml.decimal_set_{engine}"
    register_model(set_uri, set_name, tracking_uri=uri, registry_uri=uri)
    set_records = [{"id": index, **row} for index, row in enumerate(records)]
    set_version = _assert_registered_contract(
        client, uri, set_name, set_records, expected, engine=engine, model_set=True
    )
    assert set_version.version == "1" and result.comparison.reason == "no_champion"


def test_native_spark_decimal_materialization_and_smoke(delta_spark):
    """An available native Spark lane must execute the actual Decimal projection and null transport."""
    source = delta_spark.createDataFrame(
        _decimal_rows(20),
        "id long, amount decimal(17,6), exposure decimal(17,6), target decimal(17,6)",
    )
    spec = _spec()
    frame = candidate._materialize_training_rows(source, spec, spec.source_columns)
    assert frame.amount.dtype == frame.exposure.dtype == np.dtype("float64")
    spark = SimpleNamespace(table=lambda name: source)
    records = sample_serving_rows(
        spark,
        spec.table,
        (("amount", "float64"), ("exposure", "float64")),
        rows=20,
    )
    validate_json_rows(records)
    assert len(records) == 20 and sum(row["amount"] is None for row in records) == 2
