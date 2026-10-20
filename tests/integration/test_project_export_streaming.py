"""Generated batch entry points must score bounded chunks without partial outputs."""

import importlib.util
import json
import sys

import numpy as np
import pandas as pd
import pytest

pa = pytest.importorskip("pyarrow")
pq = pytest.importorskip("pyarrow.parquet")

from test_project_export import fitted_bundle, unpack_project

from backend.ml_pipeline.project_export.templates import project_files


def load_batch_project(tmp_path, monkeypatch, classification=False):
    """Import the genuine exported scorer and batch command from their ZIP files."""
    predictor, _ = unpack_project(tmp_path, fitted_bundle(classification), mode="batch")
    scoring_spec = importlib.util.spec_from_file_location("scoring", tmp_path / "scoring.py")
    assert scoring_spec and scoring_spec.loader
    scoring = importlib.util.module_from_spec(scoring_spec)
    monkeypatch.setitem(sys.modules, "scoring", scoring)
    scoring_spec.loader.exec_module(scoring)
    batch_spec = importlib.util.spec_from_file_location("exported_batch", tmp_path / "batch.py")
    assert batch_spec and batch_spec.loader
    batch = importlib.util.module_from_spec(batch_spec)
    batch_spec.loader.exec_module(batch)
    return predictor, scoring, batch


def run_batch(monkeypatch, batch, source, destination, chunk_size=2):
    """Exercise the public CLI so chunk-size validation cannot be bypassed."""
    monkeypatch.setattr(
        sys,
        "argv",
        [
            "batch.py",
            "--input",
            str(source),
            "--output",
            str(destination),
            "--chunk-size",
            str(chunk_size),
        ],
    )
    batch.main()


def source_rows():
    """Include a null-only first chunk and lexical IDs to catch inference loss."""
    return pd.DataFrame(
        {
            "x": [None, None, 0.5, 4.5, 2.0],
            "z": [1.0] * 5,
            "account_id": ["001", "002", "003", "004", "005"],
            "optional": [None, None, 7.5, 8.5, 9.5],
            "large_id": pd.array([9007199254740993, None, 9007199254740995, 7, 8], dtype="Int64"),
        }
    )


@pytest.mark.parametrize("classification", [False, True])
@pytest.mark.parametrize("input_format", ["csv", "parquet"])
@pytest.mark.parametrize("output_format", ["csv", "parquet"])
def test_batch_scores_bounded_chunks_in_order(
    tmp_path, monkeypatch, classification, input_format, output_format
):
    """Large inputs retain row order, nulls, and probabilities across chunk boundaries."""
    predictor, scoring, batch = load_batch_project(tmp_path, monkeypatch, classification)
    frame = source_rows()
    source = tmp_path / f"input.{input_format}"
    destination = tmp_path / f"output.{output_format}"
    if input_format == "csv":
        frame.to_csv(source, index=False)
    else:
        table = pa.Table.from_pandas(frame, preserve_index=False).replace_schema_metadata(None)
        pq.write_table(table, source, row_group_size=3)
    expected = predictor.predict(frame)
    chunk_lengths = []
    real_predict = scoring.Predictor.predict

    def observe_prediction(instance, chunk):
        """Measure actual model-call size while retaining real preprocessing and inference."""
        chunk_lengths.append(len(chunk))
        return real_predict(instance, chunk)

    monkeypatch.setattr(scoring.Predictor, "predict", observe_prediction)
    read_limits = []
    real_read_csv = pd.read_csv
    real_iter_batches = pq.ParquetFile.iter_batches

    def read_csv_chunks(*args, **kwargs):
        """Record the public reader limit while exercising pandas' real streaming reader."""
        read_limits.append(kwargs.get("chunksize"))
        return real_read_csv(*args, **kwargs)

    def read_parquet_batches(instance, *args, **kwargs):
        """Record the Arrow reader limit without replacing its row-group handling."""
        read_limits.append(kwargs.get("batch_size"))
        return real_iter_batches(instance, *args, **kwargs)

    monkeypatch.setattr(pd, "read_csv", read_csv_chunks)
    monkeypatch.setattr(pq.ParquetFile, "iter_batches", read_parquet_batches)
    run_batch(monkeypatch, batch, source, destination)
    monkeypatch.setattr(pd, "read_csv", real_read_csv)
    actual = (
        pd.read_csv(destination, dtype={"account_id": "string", "large_id": "string"})
        if output_format == "csv"
        else pq.read_table(destination).to_pandas(types_mapper=pd.ArrowDtype)
    )
    assert chunk_lengths == [2, 2, 1]
    assert read_limits == [2]
    assert actual["account_id"].tolist() == ["001", "002", "003", "004", "005"]
    assert actual["large_id"].astype("string").fillna("<missing>").tolist() == [
        "9007199254740993",
        "<missing>",
        "9007199254740995",
        "7",
        "8",
    ]
    assert actual["optional"].isna().tolist() == [True, True, False, False, False]
    if classification:
        assert actual["prediction"].tolist() == expected["predictions"]
        probabilities = np.array([list(row.values()) for row in expected["probabilities"]])
        np.testing.assert_allclose(
            actual[["prediction_proba_no", "prediction_proba_yes"]].to_numpy(dtype=float),
            probabilities,
        )
    else:
        np.testing.assert_allclose(actual["prediction"], expected["predictions"])
        assert not any(name.startswith("prediction_proba_") for name in actual)
    if input_format == output_format == "parquet":
        assert pq.read_table(destination)["large_id"].to_pylist() == [
            9007199254740993,
            None,
            9007199254740995,
            7,
            8,
        ]
        assert pq.read_schema(destination).field("optional") == pq.read_schema(source).field(
            "optional"
        )


@pytest.mark.parametrize("input_format", ["csv", "parquet"])
@pytest.mark.parametrize("output_format", ["csv", "parquet"])
def test_failed_later_chunk_preserves_existing_output(
    tmp_path, monkeypatch, input_format, output_format
):
    """A bad later row cannot replace a prior result with a truncated file."""
    _, _, batch = load_batch_project(tmp_path, monkeypatch)
    source = tmp_path / f"input.{input_format}"
    destination = tmp_path / f"output.{output_format}"
    frame = pd.DataFrame({"x": ["1", "2", "bad", "4"], "z": ["1"] * 4})
    if input_format == "csv":
        frame.to_csv(source, index=False)
    else:
        frame.to_parquet(source, index=False)
    destination.write_bytes(b"previous complete output")
    with pytest.raises((ValueError, TypeError)):
        run_batch(monkeypatch, batch, source, destination)
    assert destination.read_bytes() == b"previous complete output"
    assert not list(tmp_path.glob(f".{destination.name}.*"))


@pytest.mark.parametrize("column", ["prediction", "prediction_proba_yes"])
def test_batch_rejects_output_column_collisions(tmp_path, monkeypatch, column):
    """Prediction output must never silently overwrite a caller's input column."""
    _, _, batch = load_batch_project(tmp_path, monkeypatch, classification=True)
    frame = source_rows()
    frame[column] = "original"
    source, destination = tmp_path / "input.csv", tmp_path / "output.parquet"
    frame.to_csv(source, index=False)
    with pytest.raises((ValueError, SystemExit), match="prediction"):
        run_batch(monkeypatch, batch, source, destination)
    assert not destination.exists()
    assert not list(tmp_path.glob(f".{destination.name}.*"))


@pytest.mark.parametrize("classification", [False, True])
@pytest.mark.parametrize("input_format", ["csv", "parquet"])
@pytest.mark.parametrize("output_format", ["csv", "parquet"])
def test_empty_batch_writes_valid_empty_output(
    tmp_path, monkeypatch, input_format, output_format, classification
):
    """Header-only input produces a readable empty result with the input schema."""
    _, _, batch = load_batch_project(tmp_path, monkeypatch, classification)
    frame = source_rows().iloc[:0]
    source = tmp_path / f"input.{input_format}"
    destination = tmp_path / f"output.{output_format}"
    if input_format == "csv":
        frame.to_csv(source, index=False)
    else:
        frame.to_parquet(source, index=False)
    run_batch(monkeypatch, batch, source, destination)
    actual = pd.read_csv(destination) if output_format == "csv" else pd.read_parquet(destination)
    assert actual.empty
    expected_columns = [*frame.columns, "prediction"]
    if classification:
        expected_columns.extend(["prediction_proba_no", "prediction_proba_yes"])
    assert actual.columns.tolist() == expected_columns


@pytest.mark.parametrize("chunk_size", [0, -1])
def test_batch_rejects_nonpositive_chunk_size(tmp_path, monkeypatch, capsys, chunk_size):
    """Invalid limits fail before reading data or creating any output file."""
    _, _, batch = load_batch_project(tmp_path, monkeypatch)
    destination = tmp_path / "output.csv"
    with pytest.raises(SystemExit):
        run_batch(monkeypatch, batch, tmp_path / "missing.csv", destination, chunk_size)
    assert "chunk-size must be positive" in capsys.readouterr().err
    assert not destination.exists()


def test_batch_rejects_same_input_and_output(tmp_path, monkeypatch, capsys):
    """A mistaken output path cannot overwrite the source dataset."""
    _, _, batch = load_batch_project(tmp_path, monkeypatch)
    source = tmp_path / "input.csv"
    source_rows().to_csv(source, index=False)
    original = source.read_bytes()
    with pytest.raises(SystemExit):
        run_batch(monkeypatch, batch, source, source)
    assert "Input and output paths must differ" in capsys.readouterr().err
    assert source.read_bytes() == original


def test_batch_write_failure_keeps_previous_output(tmp_path, monkeypatch):
    """A disk failure after a successful chunk leaves no partial or abandoned output."""
    _, _, batch = load_batch_project(tmp_path, monkeypatch)
    source, destination = tmp_path / "input.csv", tmp_path / "output.csv"
    source_rows().to_csv(source, index=False)
    destination.write_bytes(b"previous complete output")
    real_to_csv = pd.DataFrame.to_csv
    writes = 0

    def failing_write(frame, *args, **kwargs):
        """Fail after the first real write to exercise temporary-file cleanup."""
        nonlocal writes
        writes += 1
        if writes == 2:
            raise OSError("Disk full")
        return real_to_csv(frame, *args, **kwargs)

    monkeypatch.setattr(pd.DataFrame, "to_csv", failing_write)
    with pytest.raises(OSError, match="Disk full"):
        run_batch(monkeypatch, batch, source, destination)
    assert writes == 2
    assert destination.read_bytes() == b"previous complete output"
    assert not list(tmp_path.glob(f".{destination.name}.*"))


@pytest.mark.parametrize("empty", [False, True])
def test_smoke_request_keeps_exact_nullable_integer_values(tmp_path, monkeypatch, empty):
    """Smoke fixtures cannot round integers before schema coercion or reject empty records."""
    _, scoring, _ = load_batch_project(tmp_path, monkeypatch)
    spec = importlib.util.spec_from_file_location("exported_smoke", tmp_path / "smoke_test.py")
    assert spec and spec.loader
    smoke = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(smoke)
    rows = [] if empty else [{"x": 9007199254740993, "z": 1}, {"x": None, "z": 1}]
    request = tmp_path / "request.json"
    request.write_text(json.dumps({"data": rows}), encoding="utf-8")
    real_predict = scoring.Predictor.predict
    observed = []

    def observe_input(instance, frame):
        """Observe the actual input boundary before letting the fitted model run."""
        observed.append(frame)
        return real_predict(instance, frame)

    monkeypatch.setattr(scoring.Predictor, "predict", observe_input)
    monkeypatch.setattr(sys, "argv", ["smoke_test.py", "--request", str(request)])
    smoke.main()
    assert observed[0].columns.tolist() == ["x", "z"]
    if empty:
        assert observed[0].empty
    else:
        assert observed[0]["x"].tolist() == [9007199254740993, None]


@pytest.mark.parametrize("mode", ["api", "batch", "both"])
def test_docker_copies_private_project_for_runtime_user(mode):
    """ZIP members can be mode0600, so the image's runtime user must own copied artifacts."""
    dockerfile = project_files(mode, {"version": "0.9.5", "source": "editable"})["Dockerfile"]
    copy = next(
        line
        for line in dockerfile.splitlines()
        if line.startswith("COPY") and line.endswith(" . .")
    )
    assert "--chown=10001:10001" in copy
    assert "USER 10001" in dockerfile
