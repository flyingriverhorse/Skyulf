# Version Updates

*   **v0.1.1 (2025-12-xx):** "The Observability & Stability Update" — Full test suite pass, live tuning logs, and VS Code fixes.
*   **v0.1.0 (2025-12-10):** "The Foundation & Deployment Update" — Added Deployments, Polars integration, and Optional Celery.

------------------------------------------------------------

## v0.1.1 (2025-12-xx)
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

### 🏗️ SDK & Architecture Refactoring
- **SDK Consolidation:** All core ML logic (preprocessing, modeling, tuning) is now centralized in `skyulf-core/`. This ensures that the `core/ml_pipeline` folder focuses purely on orchestration and API handling.
- **Tuning Logic Migration:** Moved hyperparameter tuning logic from `ml_pipeline` into `skyulf-core/skyulf/modeling/tuning/`, creating a unified and reusable tuning engine for both the web platform and standalone scripts.
- **Clean Separation of Concerns:**
    - `skyulf-core/`: Contains the "Brain" (Calculators, Appliers, Estimators).
    - `core/ml_pipeline/`: Contains the "Nervous System" (Engine, Registry, API).

### 🛠 Developer Experience
- **VS Code Integration:** Added `.vscode/settings.json` to resolve Pylance import errors for `skyulf-core/`.
- **Architecture Cleanup:** Clarified the separation between `NodeRegistry` (static node definitions) and `AdvisorEngine` (dynamic recommendations).

---

## v0.1.0 (2025-12-10)
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
