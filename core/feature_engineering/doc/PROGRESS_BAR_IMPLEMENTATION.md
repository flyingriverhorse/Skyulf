# Progress Bar Implementation Architecture

## Overview
The progress bar functionality provides real-time feedback for long-running operations like Model Training and Hyperparameter Tuning. It involves coordination between the Celery worker, the Database, the FastAPI backend, and the React frontend.

## Architecture Components

### 1. Log Capturing (Backend)
The core mechanism relies on intercepting logs and stdout/stderr from the underlying libraries (Scikit-learn, Optuna).

*   **`StdoutProgressCapture`**: A custom `logging.Handler` located in `core/feature_engineering/modeling/hyperparameter_tuning/tasks/execution.py`.
    *   **Optuna**: Attaches to the `optuna` logger. Since Optuna defaults to `WARNING`, we explicitly set `optuna.logging.set_verbosity(optuna.logging.INFO)` during execution to ensure "Trial finished" messages are emitted.
    *   **Scikit-learn**: Uses a `StreamCapture` proxy to intercept `sys.stdout` and `sys.stderr` for verbose output (e.g., `[CV] END ...`).
    *   **Deduplication**: Implements logic to prevent double-counting logs that might appear in both the logger stream and stdout.

### 2. Database Updates (Backend)
*   **`update_tuning_job_progress_sync`**: A synchronous function in `core/feature_engineering/modeling/hyperparameter_tuning/jobs/status.py`.
    *   It is called by the `StdoutProgressCapture` callback.
    *   It updates the `progress` (0-100 integer) and `current_step` (string) columns in the `hyperparameter_tuning_jobs` table.
    *   It uses a synchronous SQLAlchemy session since it runs within the Celery worker process (which may not be fully async-compatible in all contexts).

### 3. API Exposure (Backend)
*   **Schemas**: The Pydantic models in `core/feature_engineering/schemas.py` were updated to include `progress` and `current_step`.
    *   `HyperparameterTuningJobSummary`: Used for the list endpoint.
    *   `TrainingJobSummary`: Used for the list endpoint.
    *   **Crucial**: Without these fields in the *Summary* schemas, the list endpoint (used by the frontend polling) filters out the progress data.

### 4. Frontend Polling (React)
*   **Polling**: The frontend (`HyperparameterTuningSection.tsx`) uses `react-query` to poll the job list endpoint every 5 seconds when a job is active (`queued` or `running`).
*   **Visualization**:
    *   Renders a progress bar using the `progress` value.
    *   Displays the `current_step` text (e.g., "Trial 5/10").
    *   **Styling**: The progress bar uses a gradient background (`linear-gradient(90deg, #3b82f6 0%, #8b5cf6 100%)`) to match the application's primary button styling.

## Troubleshooting

If the progress bar is not moving:

1.  **Check Celery Logs**: Ensure the worker is running and not crashing.
2.  **Check Verbosity**: Verify that `optuna.logging.set_verbosity(optuna.logging.INFO)` is being called.
3.  **Check Database**: Query the `hyperparameter_tuning_jobs` table to see if `progress` is updating.
4.  **Check API Response**: Use the browser network tab to verify the `/hyperparameter-tuning-jobs` response contains the `progress` field.
