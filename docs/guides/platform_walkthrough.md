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

### Multi-path pipelines *(v0.3.0+)*

You can build pipelines with **multiple branches**:

- **Merge branches** — Route data through different preprocessing paths (e.g., Scaling + Encoding), then connect both into a single training node. The node displays a **⊕ Merge** badge showing how many inputs are being combined.
- **Parallel experiments** — Connect the dataset to multiple separate training nodes (e.g., RandomForest and XGBoost). Each runs as an independent experiment.
- **Copy-paste nodes** — Select nodes and press **Ctrl+C / Ctrl+V** to duplicate them with their internal edges.

> **Note:** Model-to-model connections are blocked. See the [Multi-Path Pipelines guide](multi_path_pipelines.md) for details.

---

## Step 4: Execute the pipeline

**From the canvas:** Click the **Run Preview** button to preview the pipeline, or use **Train** on an individual training node.

### Single training node

The frontend converts your visual graph into a pipeline config and sends it to the backend:

```
POST /api/pipeline/run
```

The backend validates the config, queues the job in Celery, and returns a `job_id`.

### Multiple training nodes *(v0.4.0+)*

When your canvas has 2+ training nodes on separate branches:

- **Individual Train buttons** — Each node's Train button runs only that branch (using `target_node_id` filtering).
- **Run All Experiments** — A 🚀 **Run All Experiments** button appears in the toolbar if two separate branches connected to two separate training nodes. Clicking it queues all branches at once, returning `job_ids` for each.

### Merge/Parallel toggle *(v0.4.0+)*

Training nodes with 2+ incoming connections show a **Merge / Parallel** toggle:

- **Merge**: Combines upstream data before training.
- **Parallel**: Each incoming branch becomes a separate experiment job.

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

## Step 5b: Export to Jupyter Notebook

**From the canvas:** Click the **Export Notebook** button (toolbar, top-right area) to download a ready-to-run `.ipynb` file for any completed or configured pipeline.

Two export modes are available:

| Mode | When to use |
|---|---|
| **Full** | One section per preprocessing node + one training section per branch. Best for exploration and debugging — each preprocessing step is its own cell so you can tweak parameters and inspect intermediate state. |
| **Compact** | All preprocessing steps collapsed into a single `SkyulfPipeline.fit()` call per branch. Best for sharing clean, readable notebooks or running in CI. |

### What the exported notebook contains

1. **Imports & helper cell** — injects `_summarize_metrics(m)` which flattens Skyulf's nested metric return shapes into a tidy `(split × metric)` DataFrame.
2. **Data loading** — loads the same dataset you used on the canvas via `pd.read_csv` / Parquet.
3. **Preprocessing cells** — one cell per node (Full mode) or a single pipeline config block (Compact mode).
4. **Training cell** — wired up to run the same algorithm the canvas node was configured for:
   - **Basic Training nodes** call `estimator.fit_predict()` with `log_callback=print`.
   - **Advanced Tuning nodes** automatically use `TuningCalculator`/`TuningApplier` wrappers so the Optuna / grid / random search actually runs. Each trial prints its progress in the notebook output:
     ```
     Trial 1/10 — score=0.9221
     Trial 2/10 — score=0.9306
     ...
     Tuning Completed (optuna). Best Score: 0.9497
     Best Params: {'n_estimators': 500, ...}
     Refitting best model with params: {...}
     ```
5. **Metrics cell** — renders a coloured `train vs test` table (green = high, red = low) via pandas Styler `background_gradient`.
6. **Multi-branch comparison** (if 2+ training nodes) — a `(branch × split)` metrics table after all branches have run, so you can compare experiments side-by-side.
7. **Inference cell** — shows how to reload the saved `.pkl` artifact and call `estimator.predict(new_df)`.

### Multi-branch pipelines

If your canvas has multiple training nodes (e.g., five parallel XGBoost variants), the exporter detects each branch independently — including branches produced by **parallel** trainer nodes (one branch per incoming data path). Every branch gets its own labelled section (`## Branch A`, `## Branch B`, …) so metrics are never mixed.

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
