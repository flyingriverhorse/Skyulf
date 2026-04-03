# FAQ & Comparison

Frequently asked questions and how Skyulf compares to other ML platforms.

---

## General

### What is Skyulf?

Skyulf is a **self-hosted, privacy-first MLOps platform** that combines:

- A **Python library** (`skyulf-core`) for reproducible ML pipelines.
- A **FastAPI backend** for data management, pipeline execution, and model serving.
- A **React-based visual ML Canvas** for building pipelines without writing code.

### Who is Skyulf for?

- **Data Scientists** who want a visual pipeline builder with proper leakage prevention.
- **ML Engineers** who need a self-hosted alternative to cloud ML platforms.
- **Teams** who need reproducible, auditable ML workflows without vendor lock-in.
- **Students/Researchers** who want to learn ML engineering best practices.

### Is Skyulf free?

Yes. Skyulf is open-source. See the [LICENSE](https://github.com/flyingriverhorse/Skyulf/blob/master/LICENSE) for details.

### Can I use skyulf-core without the web platform?

Absolutely. `skyulf-core` is a standalone PyPI package. Install it with `pip install skyulf-core` and use it like any Python library. The web platform is optional.

---

## How Skyulf differs from other tools

### Skyulf vs. MLflow

| Aspect | Skyulf | MLflow |
|---|---|---|
| **Focus** | End-to-end pipeline (preprocessing + training + deploy) | Experiment tracking and model registry |
| **Pipeline building** | Visual canvas + config-driven | Code-only (no visual builder) |
| **Preprocessing** | 30+ built-in nodes (imputation, encoding, scaling, outliers, feature selection, resampling) | None — you bring your own preprocessing |
| **Leakage prevention** | Calculator/Applier pattern enforces train-only statistics | Not addressed |
| **Deployment** | Built into the platform | Separate deployment step |
| **Self-hosted** | Yes | Yes |

**Summary:** MLflow is great for experiment tracking. Skyulf covers the full pipeline from raw data to deployed model, including preprocessing.

### Skyulf vs. Kubeflow / ZenML

| Aspect | Skyulf | Kubeflow / ZenML |
|---|---|---|
| **Infrastructure** | Single machine, Docker optional | Kubernetes required (Kubeflow) or multi-runtime (ZenML) |
| **Setup complexity** | `pip install` or `docker-compose up` | Significant infrastructure setup |
| **Visual builder** | Drag-and-drop React Flow canvas | DAG visualizations (read-only) |
| **Target audience** | Small-to-medium teams, individuals | Enterprise orchestration at scale |
| **Preprocessing** | Built-in node library | BYO preprocessing code |

**Summary:** Kubeflow/ZenML excel at large-scale orchestration. Skyulf is simpler to set up and includes preprocessing out of the box.

### Skyulf vs. scikit-learn Pipelines

| Aspect | Skyulf | scikit-learn Pipeline |
|---|---|---|
| **Config format** | JSON-compatible dicts (serializable, storable) | Python objects (code-defined) |
| **State management** | Explicit `params` dict (inspectable, portable) | Hidden in `self.` attributes |
| **Leakage safety** | Enforced by architecture (Calculator learns on train only) | Manual responsibility (`fit` on train, `transform` on test) |
| **Visual builder** | Yes (ML Canvas) | No |
| **Model variety** | 20 models + hyperparameter tuning | Full scikit-learn ecosystem |
| **EDA/Profiling** | Built-in analyzer + visualizer | None |

**Summary:** scikit-learn is the gold standard for ML in Python. Skyulf wraps scikit-learn models and adds config-driven pipelines, leakage prevention, and a visual interface.

### Skyulf vs. AutoML (Auto-sklearn, FLAML, H2O)

| Aspect | Skyulf | AutoML tools |
|---|---|---|
| **Approach** | Manual or semi-automated pipeline building | Fully automated model selection |
| **Control** | Full control over every preprocessing and modeling step | Black-box optimization |
| **Tuning** | Configurable (grid, random, Optuna, halving) | Automatic (built-in) |
| **Transparency** | Every step inspectable, every parameter visible | Results-focused, less transparent |
| **Use case** | When you need to understand and control your pipeline | When you want fastest time-to-result |

**Summary:** AutoML tools optimize for speed. Skyulf optimizes for transparency and control.

---

## Technical FAQ

### What Python version is required?

Python 3.9 or higher. We recommend 3.10 or 3.11 or 3.12.

### Does Skyulf support GPU training?

Not directly. Models use scikit-learn (CPU) and XGBoost (which supports GPU if configured). There is no built-in PyTorch/TensorFlow integration.

### Can I add my own preprocessing nodes?

Yes. Implement a Calculator and Applier, decorate with `@node_meta` and `@NodeRegistry.register`. See [Extending Skyulf-Core](../user_guide/extending_custom_nodes.md).

### Can I add my own models?

Yes. Implement a `BaseModelCalculator` and `BaseModelApplier`, register with `@NodeRegistry.register`, and use the model key in your config. See [Extending Skyulf-Core](../user_guide/extending_custom_nodes.md).

### Does Skyulf handle feature engineering?

Yes. The preprocessing system includes 30+ nodes: imputation (Simple, KNN, Iterative), encoding (OneHot, Ordinal, Label, Target, Hash), scaling (Standard, MinMax, Robust, MaxAbs), outlier detection (IQR, ZScore, Winsorize, EllipticEnvelope), feature generation (Polynomial, Math), feature selection (Variance, Correlation, Univariate, Model-based), and more.

### What data formats are supported?

- **skyulf-core library:** Pandas DataFrames and Polars DataFrames (auto-detected).
- **Web platform:** CSV upload via the data ingestion API. Database sources (PostgreSQL, etc.) via the ingestion endpoint.

### How does the hybrid Polars/Pandas engine work?

Skyulf auto-detects whether your data is Polars or Pandas. Simple operations (scaling, imputation) run natively in Polars for speed. Complex operations (feature selection, some sklearn-backed nodes) temporarily bridge to Pandas/NumPy via Apache Arrow (near zero-copy). See [Engine Mechanics](engine_mechanics.md).

### Is there an API for programmatic access?

Yes. The backend exposes a REST API with endpoints for data upload, pipeline execution, model deployment, and inference. See [Platform Walkthrough](platform_walkthrough.md).

### How do I run multiple experiments in parallel? *(v0.4.0+)*

Connect 2+ training nodes to your dataset (each with its own preprocessing path). A **Run All Experiments** button appears in the toolbar — clicking it queues all branches at once and returns separate `job_ids` for each. You can also click **Train** on an individual node to run just that branch.

### What's the difference between Merge and Parallel?

- **Merge:** Combines data from multiple upstream branches into a single DataFrame before training. Use when you have parallel preprocessing paths feeding one model.
- **Parallel:** Each incoming branch becomes a separate experiment job. Use when you want independent experiments.

Training nodes with 2+ inputs show a toggle to switch between modes. See the [Multi-Path Pipelines guide](multi_path_pipelines.md).

### How do I copy-paste nodes on the canvas? *(v0.4.0+)*

Select one or more nodes, press **Ctrl+C** (Cmd+C on Mac) to copy, then **Ctrl+V** (Cmd+V on Mac) to paste. Nodes are pasted with a position offset. Internal edges between selected nodes are preserved.
