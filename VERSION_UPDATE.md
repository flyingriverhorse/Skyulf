# Version Updates

*   **v0.1.7 :** "The Advanced EDA & Profiling Update" — Professional-grade automated data analysis with smart alerts, rich visualizations, and Polars-powered profiling.
*   **v0.1.6 :** "The Backend Abstraction & Modern Frameworks Update" — Polars migration, leakage proof, and performance optimization.
*   **v0.1.5 :** "The Registry & Catalog Architecture Update" — Decoupled data loading with Smart Catalog, full S3 integration, and dynamic node registry architecture.
*   **v0.1.4 :** "The Polynomial, Security & API Update" — Major release combining frontend refactoring, security hardening, backend logic fixes, and critical API routing improvements.
*   **v0.1.3 :** "The Code Quality & Stability Update" — Extensive static analysis fixes, safer type handling, and backend runtime stability improvements.
*   **v0.1.2 :** "The Tuning & Versioning Consistency Update" — Unified versioning, robust tuning evaluation, and PyPI release.
*   **v0.1.1 :** "The Observability & Stability Update" — Full test suite pass, live tuning logs, and VS Code fixes.
*   **v0.1.0 :** "The Foundation & Deployment Update" — Added Deployments, Polars integration, and Optional Celery.

------------------------------------------------------------
## v0.1.7
**"The Advanced EDA & Profiling Update"**
This release introduces a professional-grade Automated EDA module, enabling deep statistical analysis of datasets directly within the platform.

### Bug Fixes
- **Backend Schema:** Fixed `ColumnProfile` schema to include `normality_test` field, resolving runtime errors during analysis.
- **Type Safety:** Added `NormalityTestResult` Pydantic model for better validation.- **Dependencies:** Added `statsmodels`, `scipy`, and `patsy` to `requirements-fastapi.txt` to ensure statistical tests run correctly.
## Profiling & EDA
- **Statistical Tests:** Implemented automated Normality Tests (Shapiro-Wilk/KS) for numeric columns and Stationarity Tests (ADF) for time series.
- **Visual Indicators:** Added "Normal Dist" badges to variable cards and detailed p-value tooltips to distribution charts.
- **Detailed Analysis:** Exposed full statistical test results (Test Name, p-value, Conclusion) in the Variable Details modal.
- **3D Visualization:** Added interactive 3D Scatter Plots for both Bivariate Analysis and PCA using `react-plotly.js`.
- **PCA Upgrade:** Updated PCA calculation to compute 3 components, enabling 3D visualization of data structure.
- **Timeseries Analysis:** Added automated detection and visualization of datetime columns with interactive bars.
- **Geospatial Analysis:** Added automated detection and visualization of Latitude/Longitude columns with interactive scatter maps.
- **Multivariate Analysis (PCA):** Implemented robust PCA visualization with dynamic coloring by target, legends, and axis labels.
- **Advanced Profiler:** Implemented `SkyulfProfiler` in `skyulf-core` using Polars for high-performance automated EDA.
- **Smart Alerts:** Added automated detection for High Correlation, High Cardinality, Constant Columns, and Class Imbalance.
- **Rich Stats:** Added support for histograms, correlation matrices, and scatter plot sampling in the profile output.
- **Bivariate Analysis:** Added interactive Scatter Plot tab for visualizing relationships between any two numeric variables.
- **Outlier Detection:** Implemented Isolation Forest based outlier detection with a dedicated analysis tab.
- **Performance:** Switched to Canvas-based rendering (Chart.js) for scatter plots to support larger datasets (5000+ points).
- **UX Improvement:** Increased grid line visibility in 2D scatter plots for better readability.
- **Reporting:** Added "Print / Save as PDF" button to the EDA dashboard for easy report sharing.
- **Statistical Tests:** Added automated Normality Tests (Shapiro-Wilk/KS) for numeric columns and Stationarity Tests (ADF) for time series.

### 🎯 Target Analysis
- **Supervised EDA:** Added `Target Analysis` tab to the EDA module. Users can now select a target column to analyze correlations and detect leakage.
- **Categorical Targets:** Added support for categorical targets using Eta-squared (Correlation Ratio) to measure association with numeric features.
- **History:** Added analysis history sidebar to view and load past reports.
- **Leakage Detection:** Implemented automated leakage detection (alerting on >95% correlation with target).
- **Visualization:** Added interactive Bar Charts for top correlated features in the frontend.

### 🐛 Bug Fixes
- **EDA Downloads:** Fixed 3D Scatter and PCA plot downloads by implementing `Plotly.toImage` for WebGL contexts.
- **Correlation Matrix:** Fixed label truncation in downloaded correlation heatmaps by implementing dynamic text measurement and improved layout logic.
- **Download Layout:** Improved "Download All" feature to use a 2-column grid layout for better readability.
- **Dark Mode:** Added dark mode support for chart downloads (Autocorrelation, Interactions, etc.) to respect the active theme.
- **EDA Visualization:** Fixed correlation heatmap visibility for low values and handled nulls gracefully.
- **Outlier Explanation:** Added "Why is this an outlier?" column to the Outlier Analysis table, showing features with the highest deviation from the median.
- **EDA Histograms:** Fixed an issue where histograms were empty due to incorrect Polars aggregation aliases.
- **EDA Updates:** Fixed the "Update" button behavior to correctly poll for pending reports.

------------------------------------------------------------
## v0.1.6
**"The Backend Abstraction & Modern Frameworks Update"**
This release lays the groundwork for backend abstraction, enabling future support for multiple data processing frameworks like Polars and other extensions.


### �🚀 Core Engine & Performance (Polars Migration)
- **Hybrid Engine:** Introduced `SkyulfDataFrame` Protocol and `EngineRegistry` to support both Pandas and Polars backends.
- **High Performance:** Migrated key preprocessing nodes (Scalers, Imputers, Encoders, Feature Selection, etc.) to use native Polars expressions, achieving up to **4x speedup** on large datasets.
- **Sklearn Bridge:** Implemented `SklearnBridge` to seamlessly convert Polars/Arrow data to Numpy for Scikit-Learn compatibility without unnecessary copies.

### 🐛 Bug Fixes & Stability
- **Target Encoding:** Fixed a critical bug where `TargetEncoder` would skip fitting if `y` was not explicitly passed.
- **Robustness:** Fixed `LabelEncoder` and `PolynomialFeatures` to robustly handle Polars inputs and missing target columns.
- **Type Safety:** Standardized all preprocessing nodes with strict type hints and added a `compliance_suite` to verify Pandas/Polars parity.

### 📚 Documentation
- **Leakage Proof:** Added `examples/05_leakage_proof_titanic.ipynb` proving that Skyulf's architecture prevents data leakage.
- **Leakage Proof Refinement:** Updated `examples/05_leakage_proof_titanic.ipynb` and `docs/examples/leakage_proof.md` with rigorous statistical checks (tolerant math, exact scaling verification) and a "Poisoned Test" experiment to empirically prove leakage resistance.
- **Performance:** Added benchmarks demonstrating the speed advantages of the new Polars engine.

------------------------------------------------------------

## v0.1.5
**"The Registry & Catalog Architecture Update"**

This release finalizes the migration to a fully decoupled, registry-based architecture and introduces the Data Catalog pattern.

### 🔧 Backend Architecture
- **Data Catalog & S3:** Decoupled data loading via `DataCatalog`. Added `S3Catalog` and `S3Connector` for full S3 integration (ingestion, pipeline execution, artifact storage) with secure credential handling and local caching.
- **Hybrid Artifact Storage:** Implemented smart artifact routing. S3 data sources automatically save models to S3. Local data sources save locally by default.
- **Storage Control:** Added `UPLOAD_TO_S3_FOR_LOCAL_FILES` to upload local training to S3, and `SAVE_S3_ARTIFACTS_LOCALLY` to force local storage even for S3 data sources.
- **Node Registry:** All preprocessing nodes now self-register. Removed monolithic pipeline factory in favor of dynamic instantiation.
- **Artifacts:** Implemented `S3ArtifactStore` for saving models to S3. Added API to browse job artifacts.

### 🧹 Refactoring
- **Artifact Factory:** Centralized artifact storage logic into `ArtifactFactory`, eliminating code duplication across API, Tasks, and Deployment services.
- **Job Service:** Created `JobService` to unify job retrieval logic, removing the repetitive "Try TrainingJob then TuningJob" pattern.
- **Evaluation Service:** Extracted evaluation logic from `api.py` into `EvaluationService`, improving separation of concerns.
- **Logging:** Moved logging configuration to `backend/utils/logging_utils.py` to clean up `config.py`.
- **Naming:** Renamed `registry.py` to `node_definitions.py` to avoid confusion with `model_registry`.

### 🐛 Bug Fixes
- **S3 Artifacts:** Fixed `500 Internal Server Error` in artifact listing by correcting `AWS_REGION` setting and `s3fs` region configuration.
- **S3 Inference:** Fixed `400 Bad Request` in prediction endpoint by adding S3 URI support to `DeploymentService`.
- **S3 Evaluation:** Fixed `404 Not Found` in evaluation endpoint by enabling S3 artifact loading.
- **S3 Reliability:** Fixed multiple S3 issues including credential mapping for Polars/fsspec, path resolution, and file type detection.
- **Inference:** Fixed schema detection for S3-trained models and added automatic target column dropping to prevent shape mismatches.
- **Data Loading:** Resolved `FileNotFoundError` in pipeline execution and fixed "File out of specification" errors for Parquet/CSV.
- **Stability:** Achieved 100% test pass rate (113 tests). Fixed circular dependencies and logic inversions in model initialization.

### 🎨 Frontend
- **S3 UI:** Updated "Add Data Source" to support S3 paths with secure credentials.
- **Artifact Browser:** Added UI to view files associated with model versions, now displaying storage type (S3 vs Local) and full URI.
- **UX Improvements:** Improved "Data Leakage" warnings, modal responsiveness, and fixed dataset node duplication bugs.

------------------------------------------------------------

## v0.1.4
**"The Polynomial, Security & API Update"**

This comprehensive release addresses critical connectivity issues, refines the feature engineering experience, and hardens the application's security and stability.

### 🎨 Frontend Refactoring & Security
- **Standalone Polynomial Node:** Extracted `PolynomialFeatures` from the generic "Feature Generation" node into its own dedicated node. This improves discoverability and simplifies the configuration UX.
- **Secure ID Generation:** Replaced insecure `Math.random()` calls with cryptographically secure `uuid` (v4) for generating Node and Pipeline IDs, satisfying strict security compliance rules.
- **Code Hygiene:** Cleaned up console logs and improved TypeScript type safety across the frontend codebase.
- **Data Leakage Prevention:** Added intelligent warnings in the Frontend Graph Store to alert users when connecting "X/Y Split" nodes directly to models without a prior "Train/Test Split".

### 🔧 Backend & Core
- **Dependency Management:** Pinned `scikit-learn` to `<1.6.0` to resolve critical incompatibility with `imbalanced-learn` (missing `_is_pandas_df`).
- **Type Safety:** Resolved 82 `mypy` errors across `backend` and `skyulf-core` by adding `pandas-stubs` and fixing SQLAlchemy/Pydantic type mismatches.
- **Linting:** Achieved zero `flake8` errors by fixing indentation and whitespace issues in `engine.py` and `tuner.py`.
- **Logging:** Added `log_callback` support to `Tuner` and `BaseModelCalculator` to ensure warnings and tuning results are visible in the frontend.
- **CI Stability:** Added Redis service in CI and hardened teardown to avoid pytest hangs from lingering Redis/Celery background threads.
- **Pipeline Robustness:** Made `data_loader` accept `dataset_id` as a fallback for `path` in configs used by tests.
- **Pandas Compatibility:** Fixed `ValueReplacement` to handle dict-like replacements correctly on Pandas 2.x.

### 📚 Documentation
- **Quality Protocols:** Added `.github/instructions/quality_checks.instructions.md` defining mandatory Linting, Type Checking, and Versioning protocols.

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
