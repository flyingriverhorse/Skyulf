# Version Updates

*   **v0.1.4 :** "The Polynomial, Security & API Update" — Major release combining frontend refactoring, security hardening, backend logic fixes, and critical API routing improvements.
*   **v0.1.3 :** "The Code Quality & Stability Update" — Extensive static analysis fixes, safer type handling, and backend runtime stability improvements.
*   **v0.1.2 :** "The Tuning & Versioning Consistency Update" — Unified versioning, robust tuning evaluation, and PyPI release.
*   **v0.1.1 :** "The Observability & Stability Update" — Full test suite pass, live tuning logs, and VS Code fixes.
*   **v0.1.0 :** "The Foundation & Deployment Update" — Added Deployments, Polars integration, and Optional Celery.

------------------------------------------------------------

## v0.1.4
**"The Polynomial, Security & API Update"**

This comprehensive release addresses critical connectivity issues, refines the feature engineering experience, and hardens the application's security and stability.

### 🎨 Frontend Refactoring & Security
- **Standalone Polynomial Node:** Extracted `PolynomialFeatures` from the generic "Feature Generation" node into its own dedicated node. This improves discoverability and simplifies the configuration UX.
- **Secure ID Generation:** Replaced insecure `Math.random()` calls with cryptographically secure `uuid` (v4) for generating Node and Pipeline IDs, satisfying strict security compliance rules.
- **Code Hygiene:** Cleaned up console logs and improved TypeScript type safety across the frontend codebase.

### 🧠 Backend Logic & Stability
- **Polynomial Logic Fix:** Resolved a critical bug where `PolynomialFeatures` would duplicate input columns even when `include_input_features` was disabled.
- **Pandas Output Compatibility:** Fixed a crash that occurred when Scikit-Learn was configured to output Pandas DataFrames (`transform_output="pandas"`), ensuring the backend handles both Numpy and Pandas formats seamlessly.
- **Null Value Verification:** Verified and hardened the handling of null values in polynomial generation, confirming that nulls are only propagated from existing NaNs in the input data.

### 🔌 API & Routing
- **Router Restructuring:** Refactored `jobs.py` and correctly mounted `ml_pipeline_router` and `model_registry_router` under the `/api/ml` prefix.
- **404 Resolution:** Fixed issues where the Model Registry and Job endpoints were unreachable due to incorrect router prefixes.

### 📜 Observability
- **Live Logs Restored:** Fixed a mapping issue in `TrainingManager` and `TuningManager` where the `logs` field was not being propagated from the database to the API response, restoring live log visibility in the frontend.
- **Cleaner Tuning Logs:** Suppressed harmless but noisy `UserWarning: Failed to report cross validation scores for TerminatorCallback` from Optuna, ensuring cleaner output during hyperparameter tuning.

### 🛠 Stability
- **Dependency Fix:** Added `aiofiles` to `requirements-fastapi.txt` to resolve `ImportError` failures during testing.
- **Test Verification:** Verified system stability with passing tests for execution and core pipeline logic.

---

## v0.1.3
**"The Code Quality & Stability Update"**

This release focuses on hardening the codebase against runtime errors and improving maintainability through extensive static analysis fixes (Codacy).

### 🛡️ Backend Stability
- **Robust Artifact Handling:** Fixed a critical runtime error where Hyperparameter Tuning jobs saved artifacts as tuples `(model, metadata)`, causing prediction failures. The system now correctly unwraps these artifacts.
- **Human-Readable Classification Outputs:** Deployment inference now automatically decodes label-encoded classification predictions back to their original text labels when a target `LabelEncoder` is present in the bundled artifacts.
- **Evaluation Label Decoding:** Job evaluation responses now return decoded `y_true` / `y_pred` (when available) and include `y_proba.labels` alongside `y_proba.classes` to enable readable class names in the UI.
- **Deterministic Artifact Bundling:** Fixed a subtle bundling bug where the wrong feature-engineering pipeline artifact could be attached by directory scanning.
- **Composite Pipeline Bundling (Multi-Step Preprocessing):** Fixed an issue where adding an additional preprocessing node (e.g., scaling after encoding) could cause the final bundled `feature_engineer` to include only the *last* step (dropping the `LabelEncoder`). Bundling now composes a single ordered pipeline from all upstream `exec_*_pipeline` artifacts so deployments and Experiments always retain the full preprocessing chain and target label decoding.
- **Database Safety:**
    - Fixed a variable shadowing issue in `crud.py` where the built-in `filter` was being overridden.
    - Corrected SQL statement execution logic for MySQL/Snowflake paths.
    - Cleaned up unused imports and variables across the backend.

### 🧠 Engine Reliability
- **Reduced Duplication:** Consolidated repeated logic for resolving feature-engineering artifact keys into a single helper.
- **Safer Logging:** Fixed a feature-engineering logging bug that could cause runtime failures (malformed f-string) and improved shape-safe logging.

### 🧹 Frontend Code Quality (Codacy)
- **Type Safety:**
    - Replaced dangerous non-null assertions (`!`) with proper runtime checks to prevent crashes when data is missing.
    - Removed redundant conditional checks (e.g., checking if a required array exists) to clean up "dead code".
- **Best Practices:**
    - Enforced explicit `void` return types for Promise-returning functions to prevent "floating promises".
    - Standardized arrow function syntax in event handlers for better readability and safety.
    - Replaced the `delete` operator with `Reflect.deleteProperty` for safer object manipulation.

### 📈 Experiments & Evaluation UX
- **ROC Curve Availability:** Fixed ROC rendering when class IDs were numeric but `<select>` values were strings (type normalization in ROC computation).
- **Readable ROC Class Selector:** The ROC “Target Class” selector now displays class names when provided by the backend (falls back to encoded IDs when names are unavailable).
- **Cleaner Controls Layout:** The ROC mapping display is placed on its own full-width row to avoid packing controls into a single line.
- **Confusion Matrix Clarity:** Confusion matrix axis labels remain encoded (IDs) to avoid layout issues with many/long class names, while still maintaining correct alignment.

---

## v0.1.2 
**"The Tuning (Advanced Training) & Versioning Consistency Update"**

This release resolves critical discrepancies between Training and Tuning workflows, ensuring consistent versioning and full metric visibility.

### 🎯 Tuning & Evaluation Overhaul
- **Automatic Refit:** `TunerCalculator` now automatically refits the best model on the full dataset after tuning completes.
- **Unified Execution:** The Backend Engine now uses `StatefulEstimator` for tuning jobs, ensuring the exact same execution path as standard training.
- **Robust Evaluation:**
    - Fixed evaluation logic to correctly unpack tuning results `(model, result)`.
    - Resolved issues with `(X, y)` vs `(train, test)` tuple handling that caused evaluation failures.
    - **Full Metrics:** Tuning jobs now report full evaluation metrics (Accuracy, F1, Precision, Recall) in the UI, not just the optimization score.

### 🔢 Unified Versioning
- **Sequential Versioning:** Implemented a unified version counter across `TrainingJob` and `HyperparameterTuningJob` (Advanced Training).
- **Consistency:** New jobs (whether Training or Tuning) now increment from the single highest version number (e.g., v1 -> v2 -> v3), eliminating disjoint version sequences (v10 vs v40).

### 📦 Distribution
- **PyPI Release:** `skyulf-core` version `0.1.2` has been successfully built and published to PyPI.

---

## v0.1.1 
**"The Observability & Stability Update"**

This release focuses on developer experience, system stability, and deep observability into the training process.

### 🔍 Observability & Live Logging
- **Granular Tuning Logs:** Replaced black-box Scikit-Learn search engines with a custom execution loop.
    - **Real-time Updates:** Now shows "Evaluating Candidate X/Y" and "CV Fold N/M Score" in the live dashboard.
    - **Optuna Integration:** Added explicit logging for Optuna trials with mean CV scores.
- **Transparent Configuration:**
    - Fixed the confusing "Fit config params: {}" log.
    - The system now logs the **exact final parameters** (including defaults) used to initialize every model (e.g., `Initializing RandomForestClassifier with params: {'n_estimators': 100, ...}`).
- **Cross-Validation Streaming:** CV progress is now streamed fold-by-fold to the frontend logs.

### ✅ Quality Assurance & Testing
- **100% Test Pass Rate:** Achieved **108/108** passing tests across all modules:
    - **Core:** Pipeline engine, stateful transformers, and artifact management.
    - **Modeling:** Classification, Regression, and Hyperparameter Tuning.
    - **Frontend Nodes:** Dynamic verification of all nodes in the `NodeRegistry`.
    - **API & Deployment:** Full integration testing of recommendation engine and inference endpoints.

### 🏗️ Skyulf-core & Architecture Refactoring
- **skyulf-core Consolidation:** All core ML logic (preprocessing, modeling, tuning) is now centralized in `skyulf-core/`. This ensures that the `core/ml_pipeline` folder focuses purely on orchestration and API handling.
- **Tuning Logic Migration:** Moved hyperparameter tuning logic from `ml_pipeline` into `skyulf-core/skyulf/modeling/tuning/`, creating a unified and reusable tuning engine for both the web platform and standalone scripts.
- **Clean Separation of Concerns:**
    - `skyulf-core/`: Contains the "Brain" (Calculators, Appliers, Estimators).
    - `core/ml_pipeline/`: Contains the "Nervous System" (Engine, Registry, API).

### 🛠 Developer Experience
- **Architecture Cleanup:** Clarified the separation between `NodeRegistry` (static node definitions) and `AdvisorEngine` (dynamic recommendations).

---

## v0.1.0 
**"The Foundation & Deployment Update"**

This release marks a major milestone, solidifying the core architecture and completing the end-to-end MLOps loop from ingestion to deployment.

### 🚀 Core Architecture & Performance
- **Polars Integration:** Replaced Pandas with **Polars** for high-performance data ingestion and parsing.
- **Calculator/Applier Pattern:** Refactored all transformers (Scaling, Encoding, Imputation) into a robust split architecture:
    - `Calculator`: Computes statistics (mean, std, vocabulary) during training.
    - `Applier`: Applies these statistics statelessly during inference, ensuring 100% reproducibility.
- **Flexible Background Jobs:**
    - **Hybrid Async Engine:** Configurable switch (`USE_CELERY`) to run jobs via robust Redis/Celery queues or lightweight background threads.
    - **Job Cancellation:** Ability to gracefully cancel running ingestion or training jobs.

### 🎨 Frontend & Experience
- **Full React Migration:** Complete transition to a modern **React + Vite** SPA for a snappy, interactive experience.
- **New Dashboard:** Centralized view of recent activity, system status, and quick actions.
- **Detailed Data Preview:** Rich statistical summaries (distributions, missing values, unique counts) for uploaded datasets.
- **UI Consistency:** Standardized layout and full UUID display across Experiments, Jobs, and Deployments pages.

### 🧠 Training & AutoML
- **Dual Training Modes:**
    - **Standard Training:** Direct model training with fixed parameters.
    - **Advanced Training:** Automated Hyperparameter Tuning (Grid/Random/Halving) followed by auto-retraining the best model on the full dataset.
- **Experiment Tracking:**
    - **Side-by-Side Comparison:** Compare metrics (Accuracy, F1, RMSE) across multiple runs.
    - **Visualizations:** Confusion Matrices, ROC Curves, and Feature Importance charts.

### 📦 MLOps & Deployment
- **Model Registry:** Version-controlled storage for trained models with lineage tracking.
- **Deployments Page:**
    - **One-Click Deployment:** Instantly serve registered models via a local inference API.
    - **Inference Interface:** Built-in "Test Prediction" tool to send JSON payloads and verify model outputs immediately.
- **Job Management:** Unified "Jobs Drawer" to view history, logs, and status for all system activities.

### 📚 Documentation
- Updated `ROADMAP.md` to reflect current status and future "App Hub" vision.
- Updated `QUICKSTART.md` and `README.md` with Docker Compose details and optional Celery instructions.
- Updated `index.html` to accurately showcase the "Train, Track & Deploy" workflow.

---
