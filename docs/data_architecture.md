# Skyulf Architecture & Data Flow

## 1. The Dual-Engine Strategy: Polars & Pandas

Skyulf uses a hybrid approach to data processing to balance **performance** (Ingestion) with **compatibility** (Machine Learning).

### A. Ingestion & Preview (Polars)
*   **Engine:** [Polars](https://pola.rs/)
*   **Why:** Polars is significantly faster than Pandas for reading large files (CSV, Parquet) and performing initial scans. It uses lazy evaluation and multi-threading.
*   **Where:** 
    *   `backend/data_ingestion/`: Reading uploaded files.
    *   `backend/services/data_service.py`: Generating data previews and samples for the UI.
*   **Format:** Data is kept in Polars DataFrames or converted to Python dictionaries (`to_dicts()`) for JSON API responses.

### B. Machine Learning Core (Pandas/Numpy)
*   **Engine:** [Pandas](https://pandas.pydata.org/) & [Numpy](https://numpy.org/)
*   **Why:** The vast majority of the Python ML ecosystem (Scikit-Learn, XGBoost, LightGBM) is built around Numpy arrays and Pandas DataFrames.
*   **Where:** 
    *   `skyulf-core/`: The actual ML pipeline execution.
    *   `backend/ml_pipeline/execution/`: The orchestration layer that runs the core library.
*   **Format:** Data is converted to Pandas DataFrames before entering the `SkyulfPipeline`.

### C. The Bridge: Apache Arrow
*   **Technology:** [Apache Arrow](https://arrow.apache.org/)
*   **Role:** Arrow is the in-memory columnar format that both Polars and Pandas (2.0+) support.
*   **Benefit:** It allows for **zero-copy** (or near zero-copy) conversion between Polars and Pandas. When we load data with Polars and then hand it to Scikit-Learn, we aren't serializing/deserializing text; we are just passing memory pointers. This makes the "switch" extremely efficient.

---

## 2. Future Architecture: The AI Hub

Skyulf is evolving from a Tabular ML tool into a multi-modal AI Hub.

### The "Node" Abstraction
The core architecture (Graph -> Nodes -> Artifacts) is agnostic to the data type.
*   **Today:** Nodes process `pd.DataFrame`.
*   **Tomorrow:** Nodes will process `ImageBatch`, `TextCorpus`, or `HuggingFaceDataset`.

### Planned Engines
We will introduce specialized engines alongside `PandasEngine` and `PolarsEngine`:
1.  **TorchEngine:** For Deep Learning workflows (PyTorch).
2.  **HuggingFaceEngine:** For NLP pipelines (Tokenizers, Transformers).
3.  **LlamaIndexEngine:** For RAG and LLM orchestration.

### Artifact Store Evolution
The `ArtifactStore` will evolve to handle:
*   **Large Blobs:** Storing images/audio directly or via S3 references.
*   **Model Weights:** Managing `.pt`, `.onnx`, and `.gguf` files efficiently.
*   **Vector Indices:** Storing FAISS/ChromaDB indices for RAG.

See `ROADMAP.md` for the detailed timeline.
