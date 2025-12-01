Version 0.0.6 updates

# Phase 1: Stability & UX Improvements

This document tracks the progress of stability and user experience improvements for the Model Training module.

## 1. Stability & Reliability

- [x] **Cancellation of train and tune jobs**
    - Add "Cancel" button in UI for running jobs both in Model Training and Hyperparameter Tuning sections.(Not tested on Windows/Solo pool due to Celery limitations and linux as well due to windows environment).
    - Update Celery task to check for revocation signals between folds/steps.
- [ ] **"Pre-flight" Validation**
    - Implement synchronous checks before async task submission (Target existence, NaN checks, Dataset size vs CV folds).
- [ ] **Error Visibility & Categorization**
    - Backend: Categorize errors (`DataError`, `ConfigurationError`, `SystemError`).
    - Frontend: Add "View Logs/Error" modal for detailed stack traces.
- [x] **Routes.py**
    - Routes.py modularization and cleanup for better maintainability. Separated under Execution and api folders.
- [x] **Background Execution Architecture**
    - Refactored "Full Execution" (Preview Reset) to run asynchronously via Celery.
    - Eliminated main-thread blocking during heavy data loading.
    - Fixed Celery worker database initialization (`init_db`) for standalone process stability.
- [x] **Recommendations**
    - Recommendations no longer suggest incompatible transformations for the target column based on its role (feature vs. target) and data type. Only Label encoding will suggest for the target column encoding.

## 2. User Experience (UX)

- [x] **Frontend Code Structure (Refactoring)**
    - Split `ModelTrainingSection.tsx` (~1650 lines) into smaller components:
        - `TrainingJobHistory` (List/Table)
        - `HyperparameterControls` (Advanced inputs)
        - `modelingUtils.ts` (Shared utilities)
- [x] **Model Comparison Table**
    - Replace job list `<ul>` with a sortable table.
    - Columns: Version, Model Type, Status, Accuracy, F1, Duration, Created At.
    - Highlight best scores.
    - Added "Target" column for better context.
    - Implemented smart column filtering to hide redundant weighted metrics when unweighted ones are available.
- [x] **Backend Metric Calculation**
    - Updated backend to calculate unweighted metrics (F1, Precision, Recall) for binary classification.
    - Refactored `core/feature_engineering/modeling/shared/common.py` to remove legacy dependencies and fix circular imports.
- [x] **Model Registry UI Improvements**
    - Unified styling with Model Comparison Table (status badges, monospace fonts, spacing).
    - Added "Target" column.
    - Improved readability with better table layout and horizontal scrolling.
    - Using now lucide-react icons for better visual consistency.
- [x] **Real-time Progress Tracking**
    - Backend: Update job status with `progress` (0-100) and `current_step`.
    - Frontend: Display progress bar instead of static "Running" badge.
    - Fixed Optuna log capturing by forcing INFO verbosity.
    - Updated API schemas (`HyperparameterTuningJobSummary`, `TrainingJobSummary`) to expose progress fields.
