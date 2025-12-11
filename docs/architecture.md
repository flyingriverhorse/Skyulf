# Architecture

Skyulf is built on a modular architecture designed for scalability and flexibility.

## High-Level Overview

The system consists of three main components:

1.  **Frontend (Feature Canvas)**: A React + Vite Single Page Application (SPA) that provides the visual interface for data cleaning and feature engineering.
2.  **Backend (API)**: A FastAPI application that handles data ingestion, pipeline execution, and model management.
3.  **Async Worker**: A Celery worker (backed by Redis) that executes long-running tasks like model training and hyperparameter tuning.

```
┌─────────────────────────────────────────────────────────────────┐
│                         Frontend (React)                        │
│                    Visual Feature Canvas UI                     │
└─────────────────────────────────────────────────────────────────┘
                                │
                                ▼
┌─────────────────────────────────────────────────────────────────┐
│                     FastAPI Backend                             │
│  ┌─────────────┐  ┌─────────────┐  ┌─────────────────────────┐  │
│  │   Data      │  │  ML Pipeline │  │    Model Registry      │  │
│  │  Ingestion  │  │   Engine     │  │    & Deployment        │  │
│  │  (Polars)   │  │  (Pandas)    │  │                        │  │
│  └─────────────┘  └─────────────┘  └─────────────────────────┘  │
└─────────────────────────────────────────────────────────────────┘
                                │
                    ┌───────────┴───────────┐
                    ▼                       ▼
           ┌──────────────┐        ┌──────────────┐
           │   SQLite/    │        │    Redis     │
           │  PostgreSQL  │        │   + Celery   │
           └──────────────┘        └──────────────┘
```

## Data Flow: Polars vs Pandas

Skyulf uses two data libraries for optimal performance:

| Stage | Library | Why |
|-------|---------|-----|
| **Data Ingestion** | Polars | Fast I/O, lazy evaluation, low memory footprint |
| **ML Pipeline** | Pandas | Scikit-learn compatibility, rich ML ecosystem |

Data automatically converts from Polars → Pandas when moving from ingestion to preprocessing.

## Core Components

### 1. Data Ingestion (`core.data_ingestion`)
Handles loading datasets from various sources using Polars:
*   **Connectors**: `LocalFileConnector` (CSV, Excel, Parquet, JSON), `DatabaseConnector` (SQL), `ApiConnector` (REST)
*   **Profiler**: `DataProfiler` computes statistics for schema discovery
*   **Service**: `DataIngestionService` orchestrates uploads and background ingestion

### 2. ML Pipeline Data (`core.ml_pipeline.data`)
Provides Pandas-based utilities for the ML pipeline:
*   **DataLoader**: Reads CSV/Parquet into Pandas DataFrames
*   **SplitDataset**: Container for train/test/validation splits

### 3. Feature Engineering (`core.ml_pipeline.preprocessing`)
Implements the "Calculator/Applier" pattern:
*   **Calculator**: Computes statistics (e.g., mean, std, vocabulary) from training data
*   **Applier**: Applies computed statistics to new data (inference) in a stateless manner

### 4. Execution Engine (`core.ml_pipeline.execution`)
Orchestrates pipeline execution:
*   **PipelineEngine**: Runs the DAG of nodes (data_loader → feature_engineering → model_training)
*   **JobManager**: Tracks job status, progress, and cancellation

### 5. Model Registry (`core.ml_pipeline.model_registry`)
Manages model versions and lineage:
*   Tracks which job produced which model
*   Handles version incrementing per dataset/model type

### 6. Artifact Store (`core.ml_pipeline.artifacts`)
Abstracts storage of model binaries and fitted transformers:
*   `LocalArtifactStore`: Filesystem storage (joblib serialization)
*   Extensible to S3/Azure Blob for production deployments

### 7. Deployment (`core.ml_pipeline.deployment`)
Handles model serving:
*   `DeploymentService`: Deploys models and handles predictions
*   `APPLIER_MAP`: Registry of all transformer appliers for pipeline reconstruction

### 8. Recommendations (`core.ml_pipeline.recommendations`)
AI-powered preprocessing suggestions:
*   `AdvisorEngine`: Analyzes data profiles and suggests transformations
*   Plugins: CleaningAdvisor, ImputationAdvisor, ScalingAdvisor, etc.

## Database Schema

Skyulf uses SQLAlchemy (async) with Alembic for migrations. Core entities:

| Entity | Description |
|--------|-------------|
| `DataSource` | Uploaded/connected data with metadata and profiling results |
| `TrainingJob` | Training run with parameters, metrics, and artifact URI |
| `HyperparameterTuningJob` | Tuning job with search space and best results |
| `Deployment` | Active/historical model deployments |
| `User` | User accounts (optional authentication) |

## File Structure

```
core/
├── data_ingestion/          # Polars-based data loading
│   ├── connectors/          # File, SQL, API connectors
│   ├── engine/              # Profiler
│   └── service.py           # Orchestration
│
├── ml_pipeline/             # Pandas-based ML pipeline
│   ├── data/                # DataLoader, SplitDataset
│   ├── preprocessing/       # All transformers (Calculator + Applier)
│   ├── modeling/            # Classification, Regression, Tuning
│   ├── execution/           # PipelineEngine, JobManager
│   ├── deployment/          # DeploymentService
│   ├── artifacts/           # ArtifactStore
│   ├── model_registry/      # Version management
│   └── recommendations/     # AI suggestions
│
└── database/                # SQLAlchemy models
```
