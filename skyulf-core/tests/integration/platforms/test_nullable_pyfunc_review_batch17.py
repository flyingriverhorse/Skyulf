"""Lossless nullable inputs cross MLflow validation before fitted preprocessing."""

import copy
import subprocess
import sys
from io import StringIO
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pandas as pd
import pytest

mlflow = pytest.importorskip("mlflow")
pl = pytest.importorskip("polars")

from skyulf.data.dataset import SplitDataset  # noqa: E402
from skyulf.inference.bundle import ColumnSpec  # noqa: E402
from skyulf.inference.local_pipeline import load_local_pipeline, save_local_pipeline  # noqa: E402
from skyulf.inference.local_scoring import score_local_pipeline  # noqa: E402
from skyulf.inference.model_set import (  # noqa: E402
    ComponentReference,
    load_model_set,
    save_model_set,
)
from skyulf.inference.model_set_scoring import predict_model_set  # noqa: E402
from skyulf.integrations.mlflow.models import local_model  # noqa: E402
from skyulf.integrations.mlflow.models.model_set import log_model_set  # noqa: E402
from skyulf.pipeline import SkyulfPipeline  # noqa: E402


def _prepare(frame, model):
    """Exercise only the documented public preparation entry point."""
    return local_model.prepare_pyfunc_input(frame, model)


def _artifact(path, engine="pandas", mode="echo", integer="Int64"):
    """Fit real preprocessing and preserve raw typed callback evidence."""
    values = (
        [2**53 + 1, None, 2**63 - 1, -(2**63)]
        if integer == "Int64"
        else [2**31 - 1, None, -(2**31), -5]
    )
    if mode != "echo":
        values = [25, 30, 35, 40]
    frame = pd.DataFrame(
        {
            "value": pd.Series(values, dtype=integer),
            "flag": pd.Series([True, None, False, True], dtype="boolean"),
            "x": [0.0, 1.0, 2.0, 3.0],
            "target": [0.0, 2.0, 4.0, 6.0],
        }
    )
    if mode == "echo":
        preprocessing = [
            {
                "name": "drop",
                "transformer": "DropMissingColumns",
                "params": {"columns": ["value", "flag"]},
            }
        ]
    else:
        preprocessing = [
            {
                "name": "flag",
                "transformer": "SimpleImputer",
                "params": {"columns": ["flag"], "strategy": "constant", "fill_value": False},
            }
        ]
        if mode == "impute":
            preprocessing.append(
                {
                    "name": "value",
                    "transformer": "SimpleImputer",
                    "params": {"columns": ["value"], "strategy": "mean"},
                }
            )
    source = (
        "import pandas as pd\n\ndef echo(frame, predictions, params):\n"
        f"    assert str(frame['value'].dtype) == '{integer}'\n"
        "    assert str(frame['flag'].dtype) == 'boolean'\n"
        "    return pd.DataFrame({'exact': frame['value'].astype('string').fillna('<NULL>'), "
        "'flag_raw': frame['flag'].astype('string').fillna('<NULL>')}, index=frame.index)\n"
    )
    config = {
        "preprocessing": preprocessing,
        "modeling": {"type": "linear_regression"},
        "project_python_source": source,
        "project_scoring": {
            "eligibility": [],
            "outputs": [
                {
                    "name": "echo",
                    "version": "1",
                    "function": "echo",
                    "params": {},
                    "columns": [
                        {"name": "exact", "dtype": "string"},
                        {"name": "flag_raw", "dtype": "string"},
                    ],
                }
            ],
        },
    }
    native = pl.from_pandas(frame) if engine == "polars" else frame
    pipeline = SkyulfPipeline(config)
    pipeline.fit(SplitDataset(train=native, test=native.head(0)), target_column="target")
    save_local_pipeline(pipeline, path)
    query = frame.drop(columns="target")
    query.index = pd.Index([8, 8, 2, -1], name="original")
    return load_local_pipeline(path), query


def _package(path, directory, logger=local_model.log_local_model):
    """Use explicit local tracking and a real pyfunc save/load boundary."""
    directory.mkdir(parents=True, exist_ok=True)
    uri = "sqlite:///" + (directory / "tracking.db").as_posix()
    client = mlflow.MlflowClient(tracking_uri=uri)
    experiment = client.create_experiment(
        "nullable", artifact_location=(directory / "runs").as_uri()
    )
    run = client.create_run(experiment)
    logged = logger(path, run_id=run.info.run_id, artifact_path="model", tracking_uri=uri)
    downloaded = mlflow.artifacts.download_artifacts(artifact_uri=logged, tracking_uri=uri)
    return mlflow.pyfunc.load_model(downloaded), downloaded


@pytest.fixture(
    scope="module",
    params=[("pandas", "Int64"), ("pandas", "Int32"), ("polars", "Int64"), ("polars", "Int32")],
)
def nullable_model(request, tmp_path_factory):
    """Amortize real packaging without sharing mutable caller input between tests."""
    directory = tmp_path_factory.mktemp("nullable")
    artifact, query = _artifact(directory / "artifact", request.param[0], integer=request.param[1])
    model, package = _package(directory / "artifact", directory / "tracking")
    return artifact, query, model, package


def test_nullable_round_trip_preserves_exact_values_and_input(nullable_model):
    """Canonical strings retain large integers, booleans, nulls and duplicate row labels."""
    artifact, query, model, _ = nullable_model
    original = query.copy(deep=True)
    wire = _prepare(query, model)
    expected = score_local_pipeline(query, artifact)
    result = model.predict(wire)
    pd.testing.assert_frame_equal(query, original)
    pd.testing.assert_frame_equal(result, expected)
    assert wire.index.equals(query.index)
    assert wire.columns.equals(query.columns)
    assert wire.value.tolist() == [str(value) if pd.notna(value) else None for value in query.value]
    assert wire.flag.tolist() == ["true", None, "false", "true"]
    assert model.metadata.signature.inputs.input_types()[:2] == [mlflow.types.DataType.string] * 2
    assert all(column.required for column in model.metadata.signature.inputs)


@pytest.mark.parametrize("engine", ["pandas", "polars"])
def test_original_nullable_types_reach_fitted_imputers(engine, tmp_path):
    """The saved mean and Boolean constant run after dtype restoration, without refitting."""
    artifact, query = _artifact(tmp_path / "artifact", engine, mode="impute")
    query.loc[:, "value"] = pd.array([25, None, 40, None], dtype="Int64")
    model, _ = _package(tmp_path / "artifact", tmp_path / "tracking")
    expected = score_local_pipeline(query, artifact)
    result = model.predict(_prepare(query, model))
    pd.testing.assert_frame_equal(result, expected)
    assert result.prediction.notna().all()
    assert result.exact.eq("<NULL>").tolist() == [False, True, False, True]
    assert result.flag_raw.eq("<NULL>").tolist() == [False, True, False, False]


def test_transport_does_not_add_missing_value_handling(tmp_path):
    """A model trained without a numeric imputer must still reject missing numeric values."""
    _, query = _artifact(tmp_path / "artifact", mode="no_handler")
    query.loc[:, "value"] = pd.array([25, None, 40, None], dtype="Int64")
    model, _ = _package(tmp_path / "artifact", tmp_path / "tracking")
    with pytest.raises(ValueError, match="NaN|missing"):
        model.predict(_prepare(query, model))


@pytest.mark.parametrize("bad", [1.0, float(2**53), True, "1", [1]])
def test_preparation_rejects_lossy_or_ambiguous_integer_inputs(nullable_model, bad):
    """Transport may not bless a float-rounded integer or reinterpret text as native input."""
    _, query, model, _ = nullable_model
    query = query.astype({"value": "object"})
    query.iat[0, 0] = bad
    with pytest.raises((ValueError, TypeError), match="value|integer"):
        _prepare(query, model)


@pytest.mark.parametrize(
    "bad", ["01", "+1", "-0", "1.0", "1e2", " 1", str(2**63), str(-(2**63) - 1), 1, True]
)
def test_pyfunc_rejects_noncanonical_integer_wire_values(nullable_model, bad):
    """Direct serving requests obey the same exact canonical decimal and range contract."""
    _, query, model, _ = nullable_model
    wire = _prepare(query, model)
    wire.iat[0, 0] = bad
    with pytest.raises(
        (ValueError, TypeError, mlflow.exceptions.MlflowException),
        match="value|integer|convert|type",
    ):
        model.predict(wire)


@pytest.mark.parametrize("bad", ["True", "1", "yes", 1, True])
def test_pyfunc_rejects_noncanonical_boolean_wire_values(nullable_model, bad):
    """Boolean text has one spelling, avoiding numeric truthiness and accidental coercion."""
    _, query, model, _ = nullable_model
    wire = _prepare(query, model)
    wire.iat[0, 1] = bad
    with pytest.raises(
        (ValueError, TypeError, mlflow.exceptions.MlflowException),
        match="flag|boolean|convert|type",
    ):
        model.predict(wire)


def test_metadata_cannot_change_artifact_transport(nullable_model):
    """Caller edits to public metadata cannot change which raw dtype reaches preprocessing."""
    _, query, model, _ = nullable_model
    original = copy.deepcopy(model.metadata.metadata)
    try:
        model.metadata.metadata["skyulf_input_transport"]["columns"]["value"] = (
            "Int32" if str(query.value.dtype) == "Int64" else "Int64"
        )
        with pytest.raises(ValueError, match="transport"):
            _prepare(query, model)
    finally:
        model.metadata.metadata.clear()
        model.metadata.metadata.update(original)
    assert _prepare(query, model).value.iloc[1] is None


def test_nullable_model_set_and_fresh_process(tmp_path):
    """A complete saved model set and a fresh interpreter retain the same transport contract."""
    artifact, query = _artifact(tmp_path / "component")
    path = tmp_path / "set"
    save_model_set(
        path,
        {
            "main": (
                ComponentReference(
                    name="one", version="1", digest=artifact.manifest.pipeline_sha256
                ),
                tmp_path / "component",
            )
        },
        record_key_schema=(ColumnSpec(name="id", dtype="int64"),),
        composition_source="",
        composition_config={"outputs": []},
    )
    query.insert(0, "id", np.array([10, 11, 12, 13], dtype="int64"))
    model, package = _package(path, tmp_path / "tracking", logger=log_model_set)
    expected = predict_model_set(query, load_model_set(path))
    wire = _prepare(query, model)
    pd.testing.assert_frame_equal(model.predict(wire), expected)
    query.to_pickle(tmp_path / "input.pkl")
    script = "import sys,pandas as pd,mlflow\nfrom skyulf.integrations.mlflow.models.local_model import prepare_pyfunc_input\nm=mlflow.pyfunc.load_model(sys.argv[1])\nx=pd.read_pickle(sys.argv[2])\nm.predict(prepare_pyfunc_input(x,m)).to_pickle(sys.argv[3])\n"
    result = subprocess.run(
        [
            sys.executable,
            "-c",
            script,
            package,
            str(tmp_path / "input.pkl"),
            str(tmp_path / "output.pkl"),
        ],
        capture_output=True,
        text=True,
        check=False,
    )
    assert result.returncode == 0, result.stderr
    pd.testing.assert_frame_equal(pd.read_pickle(tmp_path / "output.pkl"), expected)


@pytest.mark.parametrize("case", ["empty", "all_missing", "reordered_extra"])
def test_empty_missing_and_column_order_inputs(nullable_model, case):
    """Boundary batches retain their row count, exact schema and caller column order."""
    artifact, query, model, _ = nullable_model
    query = query.copy()
    if case == "empty":
        query = query.iloc[:0]
    elif case == "all_missing":
        query["value"] = pd.Series(pd.NA, index=query.index, dtype=query.value.dtype)
        query["flag"] = pd.Series(pd.NA, index=query.index, dtype="boolean")
    else:
        query = query[["x", "flag", "value"]]
        query["unused"] = 7
    original = query.copy(deep=True)
    wire = _prepare(query, model)
    expected_input = query[["value", "flag", "x"]]
    pd.testing.assert_frame_equal(
        model.predict(wire), score_local_pipeline(expected_input, artifact)
    )
    pd.testing.assert_frame_equal(query, original)
    assert wire.columns.equals(query.columns)
    assert wire.index.equals(query.index)


@pytest.mark.parametrize("bad", [0, 1, 1.0, "true", "False"])
def test_preparation_rejects_nonboolean_values(nullable_model, bad):
    """Native Boolean preparation rejects integer truthiness and previously encoded text."""
    _, query, model, _ = nullable_model
    query = query.astype({"flag": "object"})
    query.iat[0, 1] = bad
    with pytest.raises(TypeError, match="flag.*boolean"):
        _prepare(query, model)


@pytest.mark.parametrize("case", ["missing", "duplicate", "deleted_metadata", "detached_contract"])
def test_transport_contract_cannot_be_silently_reinterpreted(nullable_model, case):
    """Required columns and saved transport identity stay independent of caller mutations."""
    _, query, model, _ = nullable_model
    if case == "missing":
        with pytest.raises(ValueError, match="required columns"):
            _prepare(query.drop(columns="value"), model)
    elif case == "duplicate":
        with pytest.raises(ValueError, match="unique"):
            _prepare(pd.concat([query, query[["value"]]], axis=1), model)
    elif case == "deleted_metadata":
        original = model.metadata.metadata.pop("skyulf_input_transport")
        try:
            with pytest.raises(ValueError, match="transport"):
                _prepare(query, model)
        finally:
            model.metadata.metadata["skyulf_input_transport"] = original
    else:
        contract = model.unwrap_python_model().input_transport()
        contract["columns"].clear()
        assert _prepare(query, model).value.iloc[0] == str(query.value.iloc[0])


@pytest.mark.parametrize("dtype", ["int32", "int64", "bool", "Float32", "Float64"])
def test_unmapped_pandas_inputs_keep_native_signature(dtype, tmp_path):
    """Ordinary NumPy primitives and nullable floats preserve their existing native API."""
    values = [True, False, True, False] if dtype == "bool" else [1, 2, 3, 4]
    frame = pd.DataFrame(
        {
            "value": pd.Series(values, dtype=dtype),
            "x": [0.0, 1.0, 2.0, 3.0],
            "target": [0.0, 2.0, 4.0, 6.0],
        }
    )
    pipeline = SkyulfPipeline(
        {
            "preprocessing": [
                {
                    "name": "drop",
                    "transformer": "DropMissingColumns",
                    "params": {"columns": ["value"]},
                }
            ],
            "modeling": {"type": "linear_regression"},
        }
    )
    pipeline.fit(SplitDataset(train=frame, test=frame.head(0)), target_column="target")
    path = tmp_path / "artifact"
    save_local_pipeline(pipeline, path)
    query = frame.drop(columns="target")
    if dtype.startswith("Float"):
        query.loc[1, "value"] = pd.NA
    model, _ = _package(path, tmp_path / "tracking")
    prepared = _prepare(query, model)
    pd.testing.assert_frame_equal(prepared, query)
    pd.testing.assert_frame_equal(
        model.predict(prepared), score_local_pipeline(query, load_local_pipeline(path))
    )
    assert "skyulf_input_transport" not in model.metadata.metadata
    assert model.metadata.signature.inputs.input_types()[0] != mlflow.types.DataType.string


@pytest.mark.parametrize("dtype", ["Int8", "Int16", "UInt32", "UInt64"])
def test_unsupported_nullable_types_still_fail_at_logging(dtype, tmp_path):
    """Transport support may not silently widen unapproved primitive storage contracts."""
    frame = pd.DataFrame({"value": pd.Series([1, 2, 3], dtype=dtype), "target": [1.0, 2.0, 3.0]})
    pipeline = SkyulfPipeline({"preprocessing": [], "modeling": {"type": "linear_regression"}})
    pipeline.fit(SplitDataset(train=frame, test=frame.head(0)), target_column="target")
    path = tmp_path / "artifact"
    save_local_pipeline(pipeline, path)
    with pytest.raises(ValueError, match="cannot preserve.*dtype exactly"):
        _package(path, tmp_path / "tracking")


def test_legacy_nullable_package_keeps_native_replay(tmp_path):
    """Previously serialized adapters without a codec attribute still load and score."""
    artifact, query = _artifact(tmp_path / "artifact")
    query = query.iloc[[0, 2, 3]]
    model = local_model.SkyulfLocalPythonModel()
    del model._input_transport
    path = tmp_path / "legacy"
    inputs = mlflow.types.Schema(
        [
            mlflow.types.ColSpec("long", "value"),
            mlflow.types.ColSpec("boolean", "flag"),
            mlflow.types.ColSpec("double", "x"),
        ]
    )
    mlflow.pyfunc.save_model(
        path=path,
        python_model=model,
        artifacts={"local_pipeline": str(tmp_path / "artifact")},
        signature=mlflow.models.ModelSignature(inputs=inputs),
        pip_requirements=[],
    )
    loaded = mlflow.pyfunc.load_model(path)
    pd.testing.assert_frame_equal(
        loaded.predict(_prepare(query, loaded)), score_local_pipeline(query, artifact)
    )
    assert loaded.unwrap_python_model().input_transport() is None


@pytest.mark.parametrize("kind", ["unknown_codec", "wrong_dtype", "missing_column"])
def test_saved_codec_is_validated_against_artifact(nullable_model, kind):
    """A deserialized transport map cannot override the fitted artifact's input contract."""
    _, query, model, package = nullable_model
    spec = model.unwrap_python_model().input_transport()
    if kind == "unknown_codec":
        spec["codec"] = "unknown"
    elif kind == "wrong_dtype":
        spec["columns"]["value"] = "Int32" if str(query.value.dtype) == "Int64" else "Int64"
    else:
        spec["columns"].pop("flag")
    adapter = local_model.SkyulfLocalPythonModel(spec)
    artifacts = model.metadata.flavors["python_function"]["artifacts"]
    path = Path(package) / artifacts["local_pipeline"]["path"]
    with pytest.raises(ValueError, match="fitted artifact schema"):
        adapter.load_context(SimpleNamespace(artifacts={"local_pipeline": str(path)}))


@pytest.mark.parametrize("direction", ["low", "high"])
def test_native_integer_bounds_reject_before_encoding(nullable_model, direction):
    """Native object integers outside the declared bit width cannot enter string transport."""
    _, query, model, _ = nullable_model
    bits = 32 if str(query.value.dtype) == "Int32" else 64
    query = query.astype({"value": "object"})
    query.iat[0, 0] = -(2 ** (bits - 1)) - 1 if direction == "low" else 2 ** (bits - 1)
    with pytest.raises(ValueError, match="bounds"):
        _prepare(query, model)


@pytest.mark.parametrize("engine", ["pandas", "polars"])
@pytest.mark.parametrize("dtype", ["Float32", "Float64", "float64"])
def test_model_set_restores_mixed_nullable_float_storage(engine, dtype, tmp_path):
    """MLflow's erased float extension storage is restored alongside exact integer text."""
    frame = pd.DataFrame(
        {
            "value": pd.Series([2**53 + 1, None, 8, 9], dtype="Int64"),
            "f": pd.Series([1.5, None, 2.5, 3.5], dtype=dtype),
            "x": [0.0, 1.0, 2.0, 3.0],
            "target": [0.0, 2.0, 4.0, 6.0],
        }
    )
    native = pl.from_pandas(frame) if engine == "polars" else frame
    pipeline = SkyulfPipeline(
        {
            "preprocessing": [
                {
                    "name": "drop",
                    "transformer": "DropMissingColumns",
                    "params": {"columns": ["value", "f"]},
                }
            ],
            "modeling": {"type": "linear_regression"},
        }
    )
    pipeline.fit(SplitDataset(train=native, test=native.head(0)), target_column="target")
    component = tmp_path / "component"
    save_local_pipeline(pipeline, component)
    artifact = load_local_pipeline(component)
    path = tmp_path / "set"
    save_model_set(
        path,
        {
            "main": (
                ComponentReference(
                    name="one", version="1", digest=artifact.manifest.pipeline_sha256
                ),
                component,
            )
        },
        record_key_schema=(ColumnSpec(name="id", dtype="int64"),),
        composition_source="",
        composition_config={"outputs": []},
    )
    query = frame.drop(columns="target")
    query.insert(0, "id", np.array([10, 11, 12, 13], dtype="int64"))
    original = query.copy(deep=True)
    model, _ = _package(path, tmp_path / "tracking", logger=log_model_set)
    expected = predict_model_set(query, load_model_set(path))
    pd.testing.assert_frame_equal(model.predict(_prepare(query, model)), expected)
    pd.testing.assert_frame_equal(query, original)
    assert model.metadata.signature.inputs.input_types()[2] != mlflow.types.DataType.string


def test_nullable_pyfunc_split_json_round_trip(nullable_model):
    """Split JSON retains nulls and exact strings before real pyfunc schema validation."""
    artifact, query, model, _ = nullable_model
    # Split JSON records index values, but has no field for the index name.
    query = query.rename_axis(None)
    original = query.copy(deep=True)
    wire = _prepare(query, model)
    restored = pd.read_json(StringIO(wire.to_json(orient="split")), orient="split", dtype=False)
    expected = score_local_pipeline(query, artifact)
    pd.testing.assert_frame_equal(model.predict(restored), expected)
    pd.testing.assert_frame_equal(query, original)
    assert restored.index.tolist() == [8, 8, 2, -1]
    # pandas read_json represents JSON null as NaN here; the codec preserves
    # its missingness when restoring nullable extension storage before FE.
    assert restored.value.isna().tolist() == [False, True, False, False]
    assert restored.flag.isna().tolist() == [False, True, False, False]
    assert restored.value.dropna().tolist() == [str(value) for value in query.value.dropna()]
    assert restored.flag.dropna().tolist() == ["true", "false", "true"]
