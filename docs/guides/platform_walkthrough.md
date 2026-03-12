# Platform Walkthrough

This guide walks through the full Skyulf web platform — from uploading a CSV to deploying a trained model. No screenshots yet, but every step maps to a specific page and API endpoint.

> **Prerequisite:** The platform is running (`python run_skyulf.py` or `docker-compose up`). Open [http://127.0.0.1:8000](http://127.0.0.1:8000) in your browser.

---

## Architecture overview

```
Browser (React SPA)
    |
    v
FastAPI Backend (REST API)
    |
    +-- Celery Workers (async jobs: pipeline execution, EDA)
    +-- Redis (message broker + result backend)
    +-- SQLite / PostgreSQL (metadata, job history)
    +-- File Storage (uploads/, exports/)
    |
    v
skyulf-core (ML engine)
```

---

## Step 1: Upload data

**Page:** `/data` (Data Sources)

1. Navigate to the **Data Sources** page.
2. Click **Upload** and select a CSV file.
3. The backend ingests the file asynchronously (`POST /api/ingestion/upload`).
4. Once ingested, the dataset appears in the list with row count, column count, and status.

**API equivalent:**

```bash
curl -X POST http://127.0.0.1:8000/api/ingestion/upload \
  -F "file=@my_dataset.csv"
```

You can preview sample rows via the **sample** button or API:

```bash
curl http://127.0.0.1:8000/data/api/sources/{source_id}/sample
```

---

## Step 2: Explore data (EDA)

**Page:** `/eda` (Exploratory Data Analysis)

1. Select a dataset from the dropdown.
2. Click **Analyze** to trigger a full automated EDA (`POST /api/eda/{dataset_id}/analyze`).
3. The EDA runs as a background Celery task — poll status until complete.
4. View the report: data quality summary, column statistics, distributions, correlations, outlier detection, and smart alerts.

**What the EDA covers:**

- Missing value analysis per column
- Numeric distributions (mean, std, skewness, kurtosis)
- Cardinality analysis for categorical columns
- Correlation matrix (Pearson, Spearman)
- Outlier detection (IQR, ZScore)
- Target variable analysis (classification balance / regression distribution)
- PCA loadings and explained variance
- Smart alerts (high cardinality, constant columns, high correlation pairs)

---

## Step 3: Build a pipeline (ML Canvas)

**Page:** `/canvas` (ML Canvas)

This is the core of Skyulf. The canvas is a React Flow-based visual editor where you build ML pipelines by connecting nodes.

### Available node categories:

| Category | Examples |
|---|---|
| **Data** | Dataset selector (connects to uploaded data) |
| **Splitting** | TrainTestSplitter |
| **Cleaning** | TextCleaning, ValueReplacement, Deduplicate |
| **Imputation** | SimpleImputer, KNNImputer, IterativeImputer |
| **Encoding** | OneHotEncoder, OrdinalEncoder, LabelEncoder, TargetEncoder |
| **Scaling** | StandardScaler, MinMaxScaler, RobustScaler |
| **Outliers** | IQR, ZScore, Winsorize, EllipticEnvelope |
| **Feature Engineering** | PolynomialFeatures, FeatureGeneration, FeatureSelection |
| **Resampling** | SMOTE, ADASYN, RandomUndersampling |
| **Modeling** | All 20 classifiers and regressors |
| **Tuning** | Hyperparameter tuner (grid, random, Optuna, halving) |

### Building a pipeline:

1. **Add a data node** and select your uploaded dataset.
2. **Add preprocessing nodes** from the sidebar (drag onto the canvas).
3. **Connect nodes** by dragging edges from output ports to input ports.
4. **Configure each node** by clicking it and setting parameters in the side panel.
5. **Add a modeling node** at the end (e.g., `random_forest_classifier`).
6. Optionally, add an **Advanced Tuning node** and configure the search strategy.

### Recommended pipeline order:

```
Dataset → TrainTestSplitter → Imputer → Encoder → Scaler → Model
```

> **Tip:** Always place `TrainTestSplitter` early to prevent data leakage. Skyulf's Calculator/Applier pattern ensures preprocessing statistics are learned only from the training split.

---

## Step 4: Execute the pipeline

**From the canvas:** Click the **Run** / **Execute** button.

The frontend converts your visual graph into a pipeline config and sends it to the backend:

```
POST /api/pipeline/run
```

The backend:

1. Validates the config.
2. Queues the job in Celery.
3. Returns a `job_id`.

**Monitoring progress:**

- The **Jobs** page (`/jobs`) shows all running and completed jobs.
- Poll job status: `GET /api/pipeline/jobs/{job_id}`
- For tuning jobs, real-time progress updates show trial-by-trial results.

---

## Step 5: Review results

**Page:** `/jobs` (Jobs)

Once a job completes, view:

- **Preprocessing metrics:** Per-step artifacts (what was learned, e.g., imputed means, encoder categories).
- **Modeling metrics:** accuracy, F1, precision, recall, ROC-AUC (classification) or MSE, RMSE, R2, MAE (regression).
- **Tuning results:** Best parameters, trial history, convergence.

**API equivalent:**

```bash
curl http://127.0.0.1:8000/api/pipeline/jobs/{job_id}/evaluation
```

---

## Step 6: Deploy the model

**Page:** `/deployments` (Deployments)

1. From a completed job, click **Deploy**.
2. The backend registers the model artifact and activates it (`POST /api/deployment/deploy/{job_id}`).
3. Only one model can be active at a time. Deploying a new model deactivates the previous one.

**Check active deployment:**

```bash
curl http://127.0.0.1:8000/api/deployment/active
```

---

## Step 7: Run predictions (inference)

**Page:** `/deployments` (Deployments) — Inference testing panel

Send new data to the active model:

```bash
curl -X POST http://127.0.0.1:8000/api/deployment/predict \
  -H "Content-Type: application/json" \
  -d '{"data": [{"age": 25, "income": 60000, "city": "London"}]}'
```

The deployed model applies the full preprocessing pipeline (imputation, encoding, scaling) and returns predictions.

---

## Step 8: Monitor for drift

**Page:** `/drift` (Data Drift)

After your model is in production, monitor whether incoming data has shifted:

1. Navigate to the **Drift** page.
2. Select the reference dataset (training data) and current dataset (new data).
3. Click **Calculate Drift** (`POST /api/monitoring/drift/calculate`).
4. Review per-column drift metrics (Wasserstein distance, KS test, PSI, KL divergence).

If significant drift is detected, consider retraining the model.

---

## API quick reference

| Action | Method | Endpoint |
|---|---|---|
| Upload data | POST | `/api/ingestion/upload` |
| List datasets | GET | `/data/api/sources` |
| Preview data | GET | `/data/api/sources/{id}/sample` |
| Run EDA | POST | `/api/eda/{dataset_id}/analyze` |
| Execute pipeline | POST | `/api/pipeline/run` |
| Job status | GET | `/api/pipeline/jobs/{job_id}` |
| Evaluation metrics | GET | `/api/pipeline/jobs/{job_id}/evaluation` |
| Deploy model | POST | `/api/deployment/deploy/{job_id}` |
| Active deployment | GET | `/api/deployment/active` |
| Predict | POST | `/api/deployment/predict` |
| Calculate drift | POST | `/api/monitoring/drift/calculate` |

---

## What's next?

- [Getting Started](getting_started.md) — Quickest path to a working pipeline (Python library).
- [Configuration](../user_guide/configuration.md) — All models and config keys.
- [FAQ & Comparison](faq.md) — How Skyulf compares to MLflow, Kubeflow, etc.
- [Troubleshooting](../user_guide/troubleshooting.md) — Common issues and fixes.
