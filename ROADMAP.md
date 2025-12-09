# Skyulf Roadmap

Skyulf is a passion project born from a simple idea: **Machine Learning Operations (MLOps) shouldn't be this hard.** I wanted to build a tool that feels like a "Hub"—a central place where you can bring your data, clean it, engineer features, and train models without fighting with glue code or complex infrastructure.

I've built a lot of the core foundation, but there is still a long way to go. This roadmap is my commitment to the project and a guide for where we are going. It's not just a to-do list; it's a vision for a community-driven tool that puts simplicity first.

---

## 1. Current Status: The Foundation

I have spent a lot of time building a robust, modular architecture. Here is what is **100% ready** and working in the project today:

### Core Engine (Backend)
*   **Solid Architecture:** Built on **FastAPI**, designed with a modular "core" structure (`core/feature_engineering`, `core/data_ingestion`, etc.).
*   **Asynchronous Processing:** Heavy lifting is handled by **Celery** and **Redis**, so the UI never freezes.
*   **Data Ingestion:** A dedicated module (`core/data_ingestion`) that handles file uploads, parsing, and initial validation.
*   **Database:** Flexible support starting with **SQLite** for zero-config local runs, scalable to **PostgreSQL**.

### The Feature Canvas (Frontend & Logic)
This is the heart of Skyulf. I've built a visual, node-based editor for feature engineering.
*   **React + Vite:** The `frontend/feature-canvas` is a modern, fast Single Page Application.
*   **Extensive Node Library:** We currently have a massive catalog of nodes defined in `node_catalog.json`, backed by real Python logic in `core/feature_engineering/preprocessing`:
    *   **Cleaning:** `drop_and_missing`, `cleaning` (imputation, outlier removal).
    *   **Transformation:** `bucketing`, `casting`, `resampling` and more.
    *   **Encoding:** Full support for categorical encoding.
    *   **Feature Generation:** Creating new features from existing data.
*   **Smart Recommendations:** The backend can suggest actions (like which columns to drop) based on data analysis.

---

## 2. The Roadmap

Since I am building this solo, I am prioritizing **stability and user experience** before adding too many new "shiny" features. I want Skyulf to be reliable enough that you trust it with your data, and simple enough that you don't need a full-time MLOps team to use it.

### Phase 1: Polish & Stability (Immediate Focus)
*Goal: Make the current experience smooth, predictable, and enjoyable.*

#### UI/UX Consistency & Architecture
*   **Extract Core Library (`skyulf-core`):** Separate the pure ML logic (transformers, pipeline utils) from the web application. This allows deployed models to run with a lightweight `pip install skyulf-core` instead of the full platform.
*   **Architecture Strategy:** Use **React + Vite** exclusively for the high-interactivity Feature Canvas and Dashboard. FastAPI serves only JSON APIs.
*   **Unified Theming:** Ensure consistent styling across the app.
*   Unify layout, typography, and theming between the Canvas and the rest of the app.
*   Make node interactions feel "buttery" (dragging, zooming, connecting edges).

#### Robustness
*   **Type Checking:** Strengthen type hints across `core/feature_engineering`, `core/data_ingestion`, and training code.
*   **Testing:** Grow test coverage for feature engineering pipelines and training jobs (unit + a few end-to-end tests).
*   **Resilience:** Handle large files, weird encodings, and missing data patterns without crashes.

#### Documentation & Onboarding
*   Start writing the "Skyulf Book", from "Hello CSV" to advanced workflows.
*   Add contextual help/tooltips for nodes, inputs, and outputs.
*   **Project Templates:** "Start with a Churn Prediction Template" or "Forecasting Template" to help new users get started instantly.

---

### Phase 2: Deepening Data Science (Mid-Term)
*Goal: Help users trust their models and understand their data.*

#### Core Engine Optimization
*   **"Rust Core" Execution Engine:** Move the pipeline orchestration loop from Python to Rust to eliminate overhead and enable parallel branch execution.
*   **High-Performance Ingestion:** Replace Pandas with **Polars** and Rust CSV parsers for parallel, chunked reading of large datasets.
*   **Hot Path Optimization:** Rewrite CPU-intensive row-level operations (Hashing, Regex cleaning) in Rust using **PyO3** for 50x-100x speedups.

#### Advanced EDA & Validation
*   **Data Quality (Great Expectations):** Integrate **Great Expectations** to automatically validate data (e.g., "Age must be > 0") and generate beautiful quality reports.
*   **Ethics & Fairness:** Detect bias in datasets (e.g., "Is this model unfair to a specific demographic?") and calculate fairness metrics.
*   **Synthetic Data:** Generate synthetic rows to augment small datasets or create anonymized replicas for safe sharing.
*   **Advanced Validation & Simulation:**
    *   **Repeated Stratified K-Fold (Monte Carlo CV):** Run cross-validation multiple times with different random splits to ensure stability.
    *   **Bootstrapping:** Estimate confidence intervals for model performance metrics.
    *   **Sensitivity Analysis:** Understand how changes in specific features impact predictions.
    *   **Adversarial Validation:** Check if training and test data come from the same distribution.
    *   **Feature Stability Analysis:** Run feature selection on random subsets to identify robust features vs noise.
    *   **Permutation Importance (with CI):** Shuffle features multiple times to measure importance stability with confidence intervals.
    *   **Monte Carlo Dropout:** For neural networks, estimate prediction uncertainty by keeping dropout active during inference.
*   **Data drift checks** between training data and new incoming data.
*   **Explainability & Interpretation:**
    *   **Global Feature Importance:** Visual rankings of top features (Gain, Split, Permutation).
    *   **SHAP/LIME:** Explain *why* a model made a specific prediction for a single row.
    *   **Partial Dependence Plots (PDP):** Visualize how a feature affects predictions.
*   Richer visualizations directly in the Canvas (distributions, correlations, target leakage warnings).
*   **Lightweight Data Labeling:** A simple interface to label raw text or images for classification tasks.

#### Advanced Modeling & Algorithms
*   **Segmentation & Clustering:** Add **K-Means**, **DBSCAN**, and **Hierarchical Clustering** nodes to group users/data (e.g., "Customer Segmentation").
*   **Advanced Trees:** Add **LightGBM**, **XGBoost** and **CatBoost** (often faster/better than standard Random Forest).
*   **Anomaly Detection:** Add **Isolation Forest** nodes to automatically flag weird data points.
*   **Time Series Basics:** Add "Lag Features" and "Rolling Window" nodes for basic forecasting tasks.
*   **Deep Learning:** Add support for basic **Neural Networks (MLP)** for tabular data (PyTorch/TensorFlow).
*   **NLP Support:** Nodes for text processing (**Tokenization**, **TF-IDF**, **Embeddings**) and text classification models.

#### Notifications & Reporting
*   **Smart Alerts:** Get notified via Email/Slack/Discord when a long training job finishes or fails.
*   **Model Cards:** Auto-generate PDF/HTML reports summarizing your model's performance for stakeholders.

#### Data Management & Connectivity
*   **Cloud Connectors:** Direct import from AWS S3, Azure Blob, and Google Sheets.
*   **Public Data Hub:** Direct integration with Kaggle and Hugging Face to download datasets via URL.
*   **Simple Dataset Versioning:** Track changes to datasets so you can roll back to previous versions.

#### Performance & Hardware
*   **GPU Acceleration:** Optional support for CUDA-enabled training for Deep Learning nodes.
*   Optimize caches and intermediate storage so even bigger datasets stay responsive.

---

### Phase 3: The "App Hub" Vision (Long-Term)
*Goal: Turn Skyulf into a hub where people build, share, and deploy ML systems.*

#### Production Serving & Real-time - High-Performance Architecture (Rust Integration): Scale to massive datasets and production-grade latency.
*   **Rust Model Serving:** Build a low-latency inference API using **Axum** and **ONNX Runtime** to serve models with <10ms latency.
*   **Real-time Sidecar:** Implement a lightweight Rust sidecar service to handle thousands of concurrent WebSocket connections for live progress updates.

#### Deployment & Export (Early Access)
*   **Export Standalone API (ZIP):** Generate a lightweight, self-contained ZIP file containing the trained model, preprocessing pipeline, and a ready-to-run FastAPI/Flask app. Users can unzip and run `python main.py` to serve their model anywhere.
*   **Notebook Export:** "Export to Jupyter Notebook" button that generates a clean, runnable notebook with all your pipeline steps, so you can tweak the code manually

#### Plugin System
*   Let users drop their own node definitions (Python + JSON spec) into a folder and see them in the Canvas.
*   Curate a small "community gallery" of reusable pipelines and nodes.

#### Automation
*   **Scheduled Retraining:** Set up cron-like schedules (e.g., "Retrain every Sunday") to keep models fresh.

#### GenAI & LangChain
*   **Visual LLM Builder:** Expose LangChain primitives (Prompts, Chains, Agents) as nodes. Users can visually drag-and-drop to build complex LLM applications (e.g., a customer support bot) without writing code.
*   **Unstructured Data Extraction:** "Chat with your documents" to extract structured datasets. For example, upload 50 PDF invoices and ask the LLM to "Extract Invoice Number and Total Amount into a table."
*   **RAG Workflows:** Build nodes that let users chat with their documents.

#### Deployment & Export
*   **Python SDK:** A developer-friendly Python library (`import skyulf`) to interact with the platform programmatically, trigger pipelines, or fetch predictions.
*   **One-Click App (Streamlit/Gradio):** Automatically generate a simple web app from trained model so you can demo it to stakeholders instantly.
*   **Docker Export:** Generate a Dockerfile and build script to containerize the standalone API for cloud deployment (AWS/Azure/GCP).
*   **ONNX export:** For supported models, export to ONNX so they can be served in non-Python environments.
*   **Data Export Nodes (ETL Mode):** Ability to save processed data to CSV, Parquet, or SQL, allowing Skyulf to be used for pure data transformation pipelines without model training.
*   **MLflow Integration:** Deep integration with MLflow for experiment tracking (charts, run comparison) and model registry.
    *   *Strategy:* Use Skyulf's simple UI for day-to-day work, but back it with MLflow so power users can access the full professional dashboard.

#### Security & Privacy
*   **Data masking & PII protection:** Add configurable masking/anonymization during ingestion so sensitive columns are never stored or shown in plain form.

#### Collaboration & Enterprise
*   **Team Spaces:** Multi-user support with simple roles (Admin, Editor, Viewer) so teams can collaborate on pipelines.
*   **Pipeline Version Control:** Visual history of your pipeline changes ("Undo", "Restore", "Diff") so you can experiment safely.

---

### Phase 4: Expansion & Ecosystem (Future Vision)
*Goal: Scale from a tool to a platform powering entire industries.*

#### Model Serving & Operations
*   **API Management:** Host models directly as REST APIs with a dashboard for "Requests per Second" and latency.
*   **Live Monitoring:** Track model performance in production (drift, errors).
*   **A/B Testing:** Split traffic between models to compare performance.

#### Domain-Specific Toolkits
*   **Finance:** RSI, Moving Averages, Bollinger Bands.
*   **Marketing:** RFM Analysis, Customer LTV.
*   **Computer Vision:** Image resizing, augmentation, basic object detection.
*   **Audio & Speech:** Integrate **Whisper** for speech-to-text and nodes for audio feature extraction (Spectrograms, MFCCs).

#### Code-First Flexibility
*   **Python Script Node:** An "Escape Hatch" node with an embedded code editor. Users can write a custom Python function to handle edge cases that standard nodes can't cover.

#### Developer Experience
*   **Visual Debugger:** "Step-Through" mode to run pipelines one node at a time and inspect data at every step.
*   **Error Replay:** Resume failed pipelines from the specific node that crashed after fixing the bug.

#### Community & Learning
*   **Skyulf Academy:** Interactive "Challenges" (e.g., "Fix the missing values") with automated checking and badges.
*   **Blueprints:** Pre-built, explained pipelines for common tasks (Churn, Forecasting) to teach best practices.

#### Next-Gen Collaboration
*   **Real-Time Multiplayer:** "Figma for ML" — see colleagues' cursors and edit pipelines simultaneously in real-time.
*   **In-Editor Chat:** Discuss specific nodes or data points directly inside the Canvas.

---
