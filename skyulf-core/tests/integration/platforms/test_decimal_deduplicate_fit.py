"""Exact duplicate identities become numeric model values only after membership is fixed."""

import json
from decimal import Decimal
from pathlib import Path

import pandas as pd
import polars as pl
import pytest

mlflow = pytest.importorskip("mlflow")

from tests.integration.platforms.test_databricks_decimal_transport import _source

from skyulf.integrations.databricks.training.fitting import candidate as fitting
from skyulf.integrations.databricks.training.shared import holdout_evidence
from skyulf.integrations.databricks.training.tuning.cv import CVSpec
from skyulf.integrations.mlflow.models.pipeline_model import pipeline_model_save_options
from skyulf.integrations.mlflow.runs.tracking import TrackingRun


@pytest.mark.parametrize("engine", ["pandas", "polars"])
def test_decimal_deduplicate_model_input_fits_and_replays_holdout(tmp_path, engine):
    """Late casts preserve every survivor and the numeric MLflow/holdout contract on both engines."""
    spec = fitting.TrainingSpec(
        table="main.data.amounts",
        version=0,
        record_key_columns=("id",),
        input_columns=("business_id", "amount"),
        target_column="target",
        max_rows=100,
        max_bytes=1_000_000,
        pre_split_steps=(
            {
                "name": "dedup",
                "transformer": "Deduplicate",
                "params": {"subset": ["business_id"]},
            },
        ),
    )
    identities = [Decimal(2**53 + index) for index in range(16)]
    source = _source(
        [
            {
                "id": str(index),
                "business_id": identity,
                "amount": Decimal(index),
                "target": float(index * 2),
            }
            for index, identity in enumerate(identities)
        ],
        {
            "id": "string",
            "business_id": "decimal(20,0)",
            "amount": "decimal(17,6)",
            "target": "double",
        },
    )
    frame = fitting._materialize_training_rows(source, spec, spec.source_columns)
    frame.attrs["source_table_id"] = "fixed-source"
    train, holdout, unavailable = fitting.split_labeled_snapshot(frame, spec, engine=engine)
    assert frame.business_id.tolist() == identities
    assert len(train) + len(holdout) == len(identities)
    assert holdout.attrs["pre_split_filter_counts"][0]["excluded_rows"] == 0

    uri = f"sqlite:///{(tmp_path / 'tracking.db').as_posix()}"
    client = mlflow.MlflowClient(tracking_uri=uri)
    experiment = client.create_experiment(
        "decimal-identity", artifact_location=(tmp_path / "runs").as_uri()
    )
    run_id = client.create_run(experiment).info.run_id
    assert isinstance(run_id, str)
    run = TrackingRun(client=client, run_id=run_id, enabled=True)
    replay = []
    for name, rows in (("source", frame), ("train", train), ("holdout", holdout)):
        receipt = holdout_evidence.save_frame_artifact(client, run_id, name, rows)
        path = Path(client.download_artifacts(run_id, f"lifecycle/data/{name}.parquet"))
        replay.append(holdout_evidence._read_parquet(path, receipt, spec))
    config = {"preprocessing": [], "modeling": {"type": "linear_regression"}}
    fitted = fitting.fit_candidate(
        None,
        spec,
        config,
        run=run,
        pipeline_config=config,
        artifact_path=tmp_path / "model",
        engine=engine,
        cv=CVSpec(),
        risk_category=None,
        prepared_data=(replay[0], replay[1], replay[2], unavailable),
    )
    options = pipeline_model_save_options(fitted.artifact, tmp_path / "model", tmp_path)
    mlflow.pyfunc.save_model(path=str(tmp_path / "pyfunc"), **options)
    loaded = mlflow.pyfunc.load_model(str(tmp_path / "pyfunc"))
    query = replay[2].loc[:, list(spec.input_columns)]
    predicted = loaded.predict(query)
    native_query = pl.from_pandas(query) if engine == "polars" else query
    assert predicted.prediction.tolist() == pytest.approx(
        fitted.artifact.pipeline.predict(native_query).tolist()
    )
    expected = fitting.evaluate_holdout(fitted.artifact, fitted.holdout, target_column="target")
    assert len(predicted) == len(holdout) and expected["heldout_rmse"] >= 0
    assert [dtype.lower() for dtype in fitted.artifact.manifest.input_dtypes] == [
        "float64",
        "float64",
    ]
    conversions = fitted.artifact.manifest.source_column_conversions
    assert conversions is not None
    assert {item.column for item in conversions} == {
        "amount",
        "business_id",
    }
    assert {item.column: item.source_dtype for item in conversions}[
        "business_id"
    ] == "decimal(20,0)"
    pd.testing.assert_frame_equal(replay[2], fitted.evidence_holdout)
    fitting.log_fitted_candidate(run, fitted, config, engine=engine, risk_category=None)
    logged = json.loads(Path(client.download_artifacts(run_id, "training_data.json")).read_text())
    rebound = fitting.with_source_conversions(config, logged, spec.input_columns)
    assert (
        rebound["source_column_conversions"]
        == dict(fitted.artifact.pipeline.config)["source_column_conversions"]
    )
    # Retrying the same saved holdout must reuse its actual serialized receipt.
    receipt = holdout_evidence._saved_or_new_frame(
        client, run_id, fitted.evidence_holdout, fitted.spec
    )
    assert receipt["rows"] == len(holdout)


def test_deferred_cast_does_not_change_model_input_lookup_identity():
    """An input also used as a native lookup key keeps its exact identity at every stage."""
    spec = fitting.TrainingSpec(
        table="main.data.amounts",
        version=0,
        record_key_columns=("id",),
        input_columns=("business_id", "amount"),
        target_column="target",
        max_rows=100,
        max_bytes=1_000_000,
        pre_split_steps=(
            {
                "name": "dedup",
                "transformer": "Deduplicate",
                "params": {"subset": ["business_id"]},
            },
        ),
        feature_lookup_json=json.dumps(
            {
                "label": "target",
                "lookups": [
                    {
                        "table_name": "main.features.amounts",
                        "lookup_key": ["business_id"],
                        "feature_names": ["amount"],
                    }
                ],
            }
        ),
    )
    assert fitting._deferred_model_columns(spec) == set()
    frame = pd.DataFrame({"business_id": [Decimal(2**53 + 1)]})
    frame.attrs["deferred_source_column_conversions"] = [
        {
            "column": "business_id",
            "source_dtype": "decimal(20,0)",
            "target_dtype": "float64",
        }
    ]
    with pytest.raises(ValueError, match="admitted model inputs"):
        fitting._normalize_model_partition(frame, spec)
