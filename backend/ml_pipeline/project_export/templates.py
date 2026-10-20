"""Small entry points and operating instructions for standalone model projects."""

import sys

API = '''"""Serve the exported fitted model; add deployment-specific authentication before exposing it."""

import os
from contextlib import asynccontextmanager

import pandas as pd
from fastapi import FastAPI, HTTPException
from scoring import Predictor, build_request_model


@asynccontextmanager
async def lifespan(app):
    """Load once at startup; fail startup if the artifact is invalid."""
    app.state.predictor = Predictor()
    yield


app = FastAPI(title="Kayra model API", lifespan=lifespan)


PredictionRequest = build_request_model()


@app.get("/health")
def health():
    """Confirm this process loaded the selected model."""
    return {"status": "ready", "model_version": app.state.predictor.manifest["job_id"]}


@app.get("/schema")
def schema():
    """Describe the raw columns expected by the saved preprocessing."""
    return app.state.predictor.schema


@app.post("/predict")
def predict(request: PredictionRequest):
    """Score without fitting or changing the saved model."""
    if len(request.data) > int(os.environ.get("MAX_PREDICT_ROWS", "10000")):
        raise HTTPException(413, "Request exceeds MAX_PREDICT_ROWS.")
    try:
        columns = [column["name"] for column in app.state.predictor.schema]
        rows = request.model_dump(by_alias=True)["data"]
        frame = pd.DataFrame(rows, dtype=object) if rows else pd.DataFrame(columns=columns)
        return app.state.predictor.predict(frame)
    except (ValueError, TypeError, KeyError) as exc:
        raise HTTPException(422, str(exc)) from exc
'''

BATCH = '''"""Score a CSV/Parquet file with the same runtime used by the HTTP API."""

import argparse
from contextlib import ExitStack, closing
from pathlib import Path
from tempfile import NamedTemporaryFile

import pandas as pd
import pyarrow as pa
import pyarrow.parquet as pq
from scoring import Predictor


def input_tables(source, chunk_size):
    """Read bounded batches without per-chunk inference changing raw field types."""
    if source.suffix.lower() == ".csv":
        with pd.read_csv(source, dtype="string", chunksize=chunk_size) as chunks:
            for frame in chunks:
                yield pa.Table.from_pandas(frame, preserve_index=False)
    else:
        with pq.ParquetFile(source) as parquet:
            if parquet.metadata.num_rows == 0:
                yield pa.Table.from_batches([], schema=parquet.schema_arrow)
            for batch in parquet.iter_batches(batch_size=chunk_size):
                yield pa.Table.from_batches([batch])


def score_table(predictor, table):
    """Append model outputs while preserving source columns, types and row order."""
    labels = predictor.probability_labels() or []
    excluding = predictor.has_exclusions()
    output_columns = ["prediction", *[f"prediction_proba_{label}" for label in labels]]
    if excluding:
        output_columns += ["scoring_status", "exclusion_reason"]
    collisions = set(output_columns).intersection(table.column_names)
    if collisions:
        raise ValueError(
            f"Input already contains prediction columns: {sorted(collisions)}; rename them first."
        )
    result = predictor.predict(table.to_pandas(types_mapper=pd.ArrowDtype))
    if len(result["predictions"]) != table.num_rows:
        raise ValueError("Model prediction changed row count.")
    classes = predictor.prediction_classes()
    # Fix the type up front: an all-excluded chunk would otherwise infer a null column.
    prediction_type = pa.float64() if classes is None else pa.array(classes).type
    table = table.append_column("prediction", pa.array(result["predictions"], prediction_type))
    probabilities = result.get("probabilities", [])
    if labels and len(probabilities) != table.num_rows:
        raise ValueError("Model probabilities changed row count.")
    for label in labels:
        values = [None if row is None else row[label] for row in probabilities]
        table = table.append_column(f"prediction_proba_{label}", pa.array(values, pa.float64()))
    if excluding:
        table = table.append_column("scoring_status", pa.array(result["scoring_status"], pa.string()))
        table = table.append_column(
            "exclusion_reason", pa.array(result["exclusion_reason"], pa.string())
        )
    return table


def write_chunks(source, temporary, output_format, predictor, chunk_size):
    """Close readers and writers even when a later chunk cannot be scored."""
    with closing(input_tables(source, chunk_size)) as tables, ExitStack() as stack:
        writer = None
        first = True
        for table in tables:
            table = score_table(predictor, table)
            if output_format == ".parquet":
                if writer is None:
                    writer = stack.enter_context(pq.ParquetWriter(temporary, table.schema))
                writer.write_table(table, row_group_size=chunk_size)
            else:
                table.to_pandas(types_mapper=pd.ArrowDtype).to_csv(
                    temporary, index=False, header=first, mode="w" if first else "a"
                )
            first = False


def score_file(source, destination, chunk_size):
    """Publish a complete result atomically, preserving an old output on any failure."""
    predictor = Predictor()
    with NamedTemporaryFile(
        dir=destination.parent, prefix=f".{destination.name}.", suffix=".tmp", delete=False
    ) as stream:
        temporary = Path(stream.name)
    try:
        write_chunks(source, temporary, destination.suffix.lower(), predictor, chunk_size)
        temporary.replace(destination)
    finally:
        temporary.unlink(missing_ok=True)


def main():
    """Validate paths and score bounded chunks beside the original input columns."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument(
        "--chunk-size", type=int, default=10000, help="Maximum rows scored at once (default: 10000)"
    )
    args = parser.parse_args()
    if args.chunk_size <= 0:
        parser.error("--chunk-size must be positive.")
    source, destination = Path(args.input), Path(args.output)
    if source.resolve() == destination.resolve() or (
        destination.exists() and source.samefile(destination)
    ):
        parser.error("Input and output paths must differ.")
    if source.suffix.lower() not in {".csv", ".parquet"}:
        parser.error("Input must be .csv or .parquet.")
    if destination.suffix.lower() not in {".csv", ".parquet"}:
        parser.error("Output must be .csv or .parquet.")
    score_file(source, destination, args.chunk_size)


if __name__ == "__main__":
    main()
'''

SMOKE = '''"""Verify the model loads, then optionally score a caller-provided request fixture."""

import argparse
import json

import pandas as pd
from scoring import Predictor


def main():
    """Avoid treating synthetic placeholders as validated business inputs."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--request", help="JSON file containing a data array of real input rows")
    args = parser.parse_args()
    predictor = Predictor()
    if args.request:
        with open(args.request, encoding="utf-8") as stream:
            rows = json.load(stream)["data"]
        columns = [column["name"] for column in predictor.schema]
        frame = pd.DataFrame(rows, dtype=object) if rows else pd.DataFrame(columns=columns)
        result = predictor.predict(frame)
        assert len(result["predictions"]) == len(rows)
        print(json.dumps(result, allow_nan=False))
    else:
        assert predictor.schema, "Missing input schema"
        print("Model loaded; prediction has not been tested. Use --request with your fixture.")


if __name__ == "__main__":
    main()
'''

README = """# Exported Kayra model

This project contains the selected **already fitted** model and preprocessing.
Exporting does not retrain or deploy it. See `manifest.json` for the job, version,
Python version, dependency pins and model checksum.

{installation_note}

## Install

Extract the ZIP and open a terminal in its root directory (where `scoring.py`
and `manifest.json` live). Use the Python major/minor version recorded in the
manifest, in a fresh environment:

```sh
uv venv
uv pip install -r requirements.txt
uv run --no-project python smoke_test.py
```

Pins describe the export environment. The matching wheel for Skyulf Core is included
in `wheels/`; requirements install that exact snapshot even for an unpublished or
editable build. Other pinned dependencies still require a package index or local
cache. Do not substitute a different version for a pickle model.

`smoke_test.py` without arguments checks loading only. Replace the **synthetic
placeholders** in `example_request.json` with meaningful values, then run:

```sh
uv run --no-project python smoke_test.py --request example_request.json
```

## Input and output

`schema.json` describes raw input columns, before preprocessing. Send all these
columns; their JSON order is irrelevant. Extra columns are ignored by scoring.
Inputs are validated and converted to the recorded training dtypes before fitted
preprocessing. Invalid values produce a clear error; null handling is determined
by the fitted preprocessing. Empty inputs with the required columns return no
predictions. Non-finite predictions or probabilities cause an error.

The response contains `predictions`, `model_version` (the job ID), and
`thresholds_applied` (null or the enabled per-class thresholds saved at export).
Classifiers that support probabilities also return `probabilities`, one dictionary
per row keyed by decoded class label. Regressors omit this field. Probabilities
remain the estimator's scores; saved thresholds affect the selected label only.
Target labels are decoded when the saved label encoder can map the predictions;
otherwise the estimator's values are retained, matching Kayra serving. Thresholds are frozen
in this project; later changes inside Kayra do not modify it.

If the pipeline contains a fitted outlier filter (IQR, Z-score, Elliptic Envelope,
Isolation Forest, Local Outlier Factor, Manual Bounds) or a Box-Cox Power
Transformer, rows they reject are kept in place and not scored: their prediction
and probabilities are null, `scoring_status` is `excluded` and `exclusion_reason`
names the step (`outlier:<step>` or `invalid:<step>`). Scored rows have
`scoring_status` `predicted`. These two fields appear only for such pipelines.

{commands}

## Scope and deployment

No database, scheduler, Databricks account, Kayra server or retraining process is
required. SQL ingestion, business eligibility/output rules, custom transformers,
model sets, neural sentence snapshots and history-dependent preprocessing are not
generated by this exporter.
Serve only built-in supervised models whose preprocessing works row by row:
row-preserving steps plus the built-in exclusion filters described above.
Train-only split, resampling, deduplication and missing-row filters are skipped at
prediction, matching Kayra serving.

The API has no built-in authentication; place it behind your own authenticated
gateway before exposing it. `MAX_PREDICT_ROWS` defaults to 10000 per request.
The Dockerfile provides a local packaging starting point; its build is not a
production deployment or an authentication setup. It runs as UID 10001. For batch
execution in Docker, mount an output directory writable by that UID and pass an
output path inside the mount (for example, `--output /output/predictions.parquet`);
the application directory is not a writable output location.

Only load trusted projects: pickle can execute code. The SHA256 check detects
accidental changes, not a malicious producer. Source datasets, connection settings,
job graphs and tuning results are not copied. **Fitted state can retain training
examples, support vectors, learned categories or statistics.** Treat the project
as a sensitive model artifact and review it before sharing.
"""


def project_files(mode: str, core_installation: dict) -> dict[str, str]:
    """Generate only the selected entry points and matching executable instructions."""
    files = {"smoke_test.py": SMOKE, ".dockerignore": ".venv\n__pycache__\n*.csv\n*.parquet\n"}
    commands = []
    if mode in {"api", "both"}:
        files["app.py"] = API
        commands.append(
            "## API\n\n```sh\nuv run --no-project uvicorn app:app --host 127.0.0.1 --port 8000\n"
            'curl -X POST http://127.0.0.1:8000/predict -H "Content-Type: application/json" '
            "--data-binary @example_request.json\n```\n\n"
            "`GET /health`, `GET /schema` and `/docs` expose readiness, input fields and OpenAPI."
        )
    if mode in {"batch", "both"}:
        files["batch.py"] = BATCH
        commands.append(
            "## Batch\n\n```sh\nuv run --no-project python batch.py --input input.parquet "
            "--output predictions.parquet --chunk-size 10000\n```\n\nCSV is supported too. "
            "Inputs are read and scored in bounded chunks, retaining row order and original "
            "columns. Output adds `prediction` and, when available, "
            "`prediction_proba_<class>` columns, plus `scoring_status` and "
            "`exclusion_reason` when the pipeline has exclusion filters. Existing "
            "output-column names are rejected.\n\n"
            "Parquet input retains its schema. CSV fields are read as text, preserving leading "
            "zeros and avoiding inconsistent inference between chunks; CSV-to-Parquet output "
            "therefore stores original columns as strings. Standard CSV missing-value markers "
            "remain nulls. The scorer separately converts required fields to the training "
            "dtypes. Output is replaced only after every chunk succeeds; a failure removes "
            "the temporary file and preserves any previous output."
        )
    files["Dockerfile"] = _dockerfile(mode)
    note = (
        f"**Core dependency:** skyulf-core {core_installation['version']} "
        f"(installation source: {core_installation['source']}). "
        "Its matching wheel is bundled in `wheels/`; no separate Core download or build is needed."
    )
    files["README.md"] = README.format(commands="\n\n".join(commands), installation_note=note)
    return files


def _dockerfile(mode: str) -> str:
    """Install bundled Core before copying the model and expose API readiness when present."""
    command = (
        '["uvicorn", "app:app", "--host", "0.0.0.0", "--port", "8000"]'
        if mode in {"api", "both"}
        else '["python", "batch.py", "--help"]'
    )
    health = (
        "EXPOSE 8000\n"
        "HEALTHCHECK --interval=30s --timeout=5s --start-period=30s --retries=3 "
        'CMD ["python", "-c", "import urllib.request; '
        "urllib.request.urlopen('http://127.0.0.1:8000/health', timeout=3).close()\"]\n"
        if mode in {"api", "both"}
        else ""
    )
    return (
        f"FROM python:{sys.version_info.major}.{sys.version_info.minor}-slim\n"
        "COPY --from=ghcr.io/astral-sh/uv:0.10.4 /uv /bin/uv\n"
        "RUN apt-get update && apt-get install -y --no-install-recommends libgomp1 "
        "&& rm -rf /var/lib/apt/lists/*\n"
        "WORKDIR /app\nCOPY requirements.txt .\nCOPY wheels ./wheels\n"
        "RUN uv pip install --system --no-cache -r requirements.txt\n"
        "COPY --chown=10001:10001 . .\nUSER 10001\n" + health + "CMD " + command + "\n"
    )
