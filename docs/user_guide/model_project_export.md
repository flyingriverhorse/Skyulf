# Export a trained model project

Kayra can download a completed model version as a standalone Python project.
The ZIP contains the fitted estimator and preprocessing, plus a FastAPI service,
a file-based batch command, or both. It does not retrain the model or change
the active deployment.

Use this when another team needs to run your trained model without operating
the Kayra application. For a notebook that **trains a model from the canvas
configuration**, use the canvas's existing notebook export instead.

New to project export? Start with the step-by-step
[project export quickstart](model_project_export_quickstart.md). This page is the
complete reference.

## Download from Kayra

1. Open **Model Registry** and select a model.
2. Locate the completed version you want to distribute.
3. Select **API + batch**, **API only**, or **Batch only**.
4. Click **Export project** and extract the downloaded ZIP.

The version's job ID determines which fitted artifact is exported. There is no
requirement to deploy that version first. An unsupported artifact produces a
visible error beside its export controls.

## What is included

| File | Purpose |
| --- | --- |
| `model.pkl` | Fitted estimator, preprocessing, target decoder and enabled saved decision thresholds. |
| `scoring.py` | Common prediction code used by both API and batch. |
| `app.py` | FastAPI service; included in API modes. |
| `batch.py` | CSV/Parquet input and output; included in batch modes. |
| `schema.json` | Ordered raw input column names and recorded dtypes. |
| `manifest.json` | Job/version identity, Python version, package pins and model SHA256. |
| `requirements.txt` | Exact installed versions of the runtime dependencies. |
| `wheels/skyulf_core-*.whl` | Portable snapshot of the Core code used by the exporter, including unpublished changes. |
| `example_request.json` | Synthetic placeholders, not a sample of training data. |
| `smoke_test.py` | Model loading check, with an optional caller-provided prediction fixture. |
| `Dockerfile`, `.dockerignore` | Starting point for container packaging. |
| `README.md` | Commands and limitations specific to the selected export mode. |

Source datasets, connection credentials, job graphs and tuning reports are not
copied. The fitted model itself can retain data: for example KNN training points,
support vectors or learned categories. Treat the export as a sensitive model
artifact. Never load an untrusted pickle; a matching checksum detects accidental
corruption but does not establish who produced the file.

## Install and check

Open a terminal in the extracted project directory. Use the Python major/minor
version in `manifest.json`.

```sh
uv venv
uv pip install -r requirements.txt
uv run --no-project python smoke_test.py
```

This last command verifies loading only. Edit `example_request.json` to contain
meaningful inputs for your model, then test prediction:

```sh
uv run --no-project python smoke_test.py --request example_request.json
```

Do not assume the placeholders are valid categories or realistic measurements.
They are provided to show the request structure.

The ZIP includes its matching Skyulf Core wheel. `requirements.txt` installs that
file directly, so an unpublished Core version does not need to exist on PyPI.
Keep the `wheels/` directory beside `requirements.txt`, including when building
the Docker image. Other dependencies still come from your configured package
index or local package cache.

The manifest's `core_installation` records the wheel filename, wheel SHA256 and
source SHA256. It also records editable/direct installation status without local
paths or package-source URLs. Each export checks the current Core source; changing
development code produces a different snapshot even if its version is unchanged.
Export does not contact PyPI to check release availability.

Changing package versions can make pickled estimators incompatible. The pins
describe the environment that exported the artifact; they are not a record of
an older training environment that is no longer available.

## Run the API

For **API only** or **API + batch**:

```sh
uv run --no-project uvicorn app:app --host 127.0.0.1 --port 8000
```

The model loads once during application startup. Check `GET /health`, inspect
the raw columns at `GET /schema`, or open `/docs` for interactive requests.
Swagger lists the model's actual input fields and their types. Every recorded
column is required; a null value is distinct from an omitted column.

For a model trained on `age` and `income`, a request could be:

```json
{
  "data": [
    {"age": 35, "income": 48000},
    {"age": 52, "income": 72000}
  ]
}
```

Use your model's own columns, then send the request:

```sh
curl -X POST http://127.0.0.1:8000/predict \
  -H "Content-Type: application/json" \
  --data-binary @example_request.json
```

An illustrative response:

```json
{
  "predictions": ["no", "yes"],
  "probabilities": [
    {"no": 0.85, "yes": 0.15},
    {"no": 0.1, "yes": 0.9}
  ],
  "model_version": "the-exported-training-job-id",
  "thresholds_applied": null
}
```

`model_version` identifies the immutable selected job. The numeric Registry
version is also recorded separately in the manifest. If tuned thresholds were
enabled when exported, `thresholds_applied` contains those per-class values.
Later threshold edits in Kayra do not change an already downloaded project.
When a saved target label encoder can decode predictions, original labels are
returned. Undecodable results retain the estimator's values, matching Kayra serving.

Classifiers with `predict_proba` also return per-class `probabilities` in input row
order. Keys identify the decoded class labels; numeric labels become JSON string
keys. Saved decision thresholds can change `predictions` but do not change the
probabilities. Regression and classifiers without `predict_proba`, such as
`LinearSVC`, omit this field; export does not refit or calibrate a model.

The API limits requests to 10,000 rows by default; configure `MAX_PREDICT_ROWS`
in its environment to change that. It has no built-in authentication. Use your
own authenticated gateway and deployment controls before exposing it to clients.

For local container packaging:

```sh
docker build -t kayra-model .
docker run --rm -p 127.0.0.1:8000:8000 kayra-model
```

The image binds to `0.0.0.0` inside the container; the command above publishes
only on the host's loopback interface. It includes the OpenMP runtime used by
native estimators; see the [LightGBM installation guide](https://lightgbm.readthedocs.io/en/stable/Installation-Guide.html).
Validate the image on your deployment platform. Core is installed from the
exported `wheels/` directory. The generated Dockerfile pins
the uv installer image and includes a health check for API modes.

## Run file-based batch prediction

For **Batch only** or **API + batch**:

```sh
uv run --no-project python batch.py \
  --input input.parquet \
  --output predictions.parquet \
  --chunk-size 10000
```

CSV is supported too. Output retains the original rows and columns and adds
`prediction`. Classifiers with probabilities also add `prediction_proba_<class>`
columns. The command rejects collisions with these output names and an output
path identical to the input.

The command reads CSV chunks or Parquet record batches, using at most the
configured number of input rows per prediction call. It preserves input order
and writes to a temporary sibling file; only a successful complete run replaces
the output. A later chunk failure leaves an existing output intact.

CSV inputs are read as text so IDs such as `00123` and large integers survive
reader inference. Standard CSV missing-value markers are interpreted as nulls.
Model input columns are converted according to `schema.json`
for prediction. Original CSV columns remain text when written to Parquet.
Parquet input preserves its original column types. Prefer Parquet when the output
must retain numeric, date or boolean types. This command runs in one process;
it does not submit a Spark job or watch for new files.

## Rows excluded by outlier filters

Pipelines can contain fitted filters that remove rows during training: the
outlier methods IQR, Z-Score, Elliptic Envelope, Isolation Forest, Local Outlier
Factor and Manual Bounds, and a Box-Cox `PowerTransformer` (which cannot transform
non-positive values). The exported project applies the same fitted filters to new
rows, but it never drops a requested row. A rejected row stays in its position
and is not scored, using the same rules as the Databricks saved scoring path:

| Field | Scored row | Excluded row |
| --- | --- | --- |
| `prediction` / `predictions` | model output | `null` |
| `probabilities` / `prediction_proba_<class>` | class scores | `null` |
| `scoring_status` | `predicted` | `excluded` |
| `exclusion_reason` | `null` | `outlier:<step>` or `invalid:<step>` |

`scoring_status` and `exclusion_reason` are added to API responses and batch
outputs only when the exported pipeline contains such a filter. Projects without
one keep the response shape described above. Count or route excluded rows in
your own system; the project does not send them anywhere.

Winsorize and Clip to Fixed Bounds keep every row and simply cap values, so they
never exclude rows.

## Input and preprocessing contract

- Send **raw input columns**, before encoding, imputation or scaling. The saved
  preprocessing performs those operations.
- All columns in `schema.json` are required. Their request order can differ;
  the model's final feature order is enforced after transformation.
- Extra columns are ignored for prediction; batch output retains them.
- Recorded dtypes control conversion of numeric, boolean and date/time inputs.
  Invalid values produce an API 422 response or a failed batch command; integer
  conversion rejects fractions and overflow instead of silently rounding.
- For large integer columns, send JSON integers or decimal strings, never
  rounded floating-point values. An `object` dtype has no recorded scalar type;
  send the same value types used during training.
- Nulls remain missing values. Whether prediction accepts them depends on the
  fitted preprocessing and estimator.
- Input row order and count are preserved. Rows rejected by outlier filters stay
  in place with a null prediction (see above). Non-finite predictions are rejected.
- Train-only split, resampling, deduplication and missing-row filters are skipped
  during unkeyed prediction, matching Kayra's existing deployment path.

The exporter supports built-in supervised classification/regression bundles whose
preprocessing works row by row: steps that keep every row, plus the built-in
exclusion filters listed above. H3 steps include their optional `h3` package pin; embedded
`sentence_embedder` snapshots require separate neural dependency packaging and are
not exported. Context/history-dependent steps, arbitrary custom classes,
legacy transformer-list artifacts, model sets and merged graphs without a proven
raw schema are rejected. No SQL connector, scheduler, business eligibility rule
or monitoring service is generated.

Every retained step must declare that it works on each row independently. Steps
that need other rows at prediction time (lag and rolling windows, other
history-dependent features), custom code nodes (`ColumnFunction`,
`FittedFunction`, `RowFilterFunction`) and `sentence_embedder` are refused before
download. Use notebook export for those pipelines.

New linear training pipelines save their raw input schema. Older artifacts
without that metadata cannot safely infer inputs from their engineered features:
retrain in the updated Kayra application or use notebook export.

## Calling the export endpoint directly

Within the same Kayra API access configuration:

```text
POST /api/deployment/export/{job_id}?mode=both
```

`mode` is `api`, `batch`, or `both`. Success returns `application/zip` with an
attachment filename. Missing jobs return 404; incomplete jobs and unsupported
artifacts return 400; invalid modes return 422. Exporting does not create a
deployment or modify training state.
