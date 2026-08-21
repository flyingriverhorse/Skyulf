# Performance & Scalability

Skyulf is designed to handle production-scale workloads efficiently. By leveraging a **Hybrid Engine** architecture, it automatically selects the best tool for the job: **Polars** for high-performance data transformation and **Pandas/Scikit-Learn** for compatibility with the vast ML ecosystem.

## Benchmarks

We regularly benchmark Skyulf to ensure it meets performance standards. Below are the results from our latest internal benchmarks comparing the Pandas-only path vs. the Polars-optimized path.

### Scenario: Large Scale Transformation
*   **Dataset:** 2,000,000 rows, 20 columns (10 numeric, 10 categorical).
*   **Pipeline:**
    1.  Imputation (Mean)
    2.  Standard Scaling (10 columns)
    3.  One-Hot Encoding (5 columns)
    4.  Hash Encoding (5 columns)
*   **Hardware:** Standard Dev Environment

| Engine | Execution Time | Speedup |
| :--- | :--- | :--- |
| **Pandas** | 11.91s | 1.0x |
| **Polars** | **3.04s** | **3.91x** 🚀 |

### Scenario: Round-Trip Removal (Category B nodes)

Previously, several Polars-engine nodes converted the **whole frame** to pandas
and back inside `apply` (`to_pandas()` + `from_pandas()`). That round-trip cost
time, memory, and dtype fidelity (nullable `Int64` upcast to `Float64`). These
nodes now operate on Polars natively — the script
`skyulf-core/benchmarks/bench_roundtrip_removal.py` measures the removed
overhead against a reconstruction of the old path:

*   **Dataset:** 500,000 rows × 21 columns (200,000 for the fit-heavy nodes).
*   **Measurement:** median of 3 runs, same fitted artifacts.

| Node | Old (round-trip) | New (native) | Speedup |
| :--- | :--- | :--- | :--- |
| **TrainTestSplitter** | 0.058s | 0.021s | **2.79x** |
| **EllipticEnvelope** | 0.020s | 0.006s | **3.47x** |
| **CountVectorizer** | 0.685s | 0.605s | 1.13x |

The vectorizer gain is smaller because its runtime is dominated by sklearn's
`transform` and text joining, not by the frame conversion — but the conversion
of *unrelated* columns is gone, and every Polars input now keeps its exact
dtypes end to end.

### Scenario: Engine Comparison Across Nodes & Models

`skyulf-core/benchmarks/bench_engine_comparison.py` runs the **same node —
fit + apply — on the same dataset twice**: once as a pandas frame, once as a
Polars frame. The numbers therefore include the legitimate pandas/sklearn
boundaries Polars still has to cross, i.e. they measure what the
`SKYULF_ENGINE` choice actually costs.

*   **Dataset:** 200,000 rows × 21 columns (12 floats incl. 4 with 5% missing,
    4 nullable ints, 3 categorical, 1 text, 1 binary target).
*   **Configs:** every node runs with realistic, fully-specified settings —
    8–12 column sets, n-gram (1,2) + `max_features` text options, stratified
    train/test/validation split, quantile binning with missing labels,
    `drop_first`/unknown-handling encoders, 50-tree models with subsampling.
*   **Measurement:** median of 3 runs per node (model fits: 1 run).
*   **Correctness:** before any timing, the script runs a **parity check** —
    each node is fit+applied on both engines and the outputs are compared
    value-for-value (numerics at 1e-9 tolerance, strings exactly). A row only
    appears in the table if both engines produced identical results.

| Node | pandas | polars | Speedup |
| :--- | :--- | :--- | :--- |
| **SimpleImputer** | 0.171s | **0.002s** | **105.8x** |
| **HashEncoder** | 0.046s | **0.009s** | **4.90x** |
| **MinMaxScaler** | 0.023s | **0.005s** | **4.57x** |
| **StandardScaler** | 0.051s | **0.029s** | **1.78x** |
| **TrainTestSplitter** | 0.118s | **0.069s** | **1.71x** |
| **LabelEncoder** | 0.046s | **0.028s** | **1.65x** |
| **Winsorize** | 0.053s | **0.037s** | **1.42x** |
| **ZScore** | 0.035s | **0.025s** | **1.40x** |
| **RobustScaler** | 0.076s | **0.055s** | **1.37x** |
| **EllipticEnvelope** | 0.182s | **0.144s** | **1.27x** |
| **OrdinalEncoder** | 0.106s | **0.088s** | **1.20x** |
| **Tokenizer** | 1.033s | **0.909s** | **1.14x** |
| **LogisticRegression** | 0.104s | **0.102s** | **1.01x** |
| **RandomForest (n=50)** | 1.715s | **1.695s** | **1.01x** |
| **GradientBoosting (n=50)** | 15.343s | **15.342s** | **1.00x** |
| **TfidfVectorizer** | 2.350s | 2.406s | 0.98x |
| **OneHotEncoder** | 0.256s | 0.264s | 0.97x |
| **CountVectorizer** | 2.233s | 2.292s | 0.97x |
| **IQR** | 0.054s | 0.056s | 0.96x |
| **XGBoost (n=50)** | 0.270s | 0.283s | 0.95x |
| **PowerTransformer** | 1.542s | 1.666s | 0.93x |
| **GeneralBinning** | 0.002s | 0.003s | 0.50x |

**Read:** data-munging nodes (imputation, scaling, splitting, hash encoding)
win big on Polars — up to 106x. Nodes whose runtime is dominated by
scikit-learn / XGBoost compute (PowerTransformer, text vectorizers, all model
fits) land at ~1.0x, because their work happens on the other side of the
unavoidable sklearn boundary and the frame conversion is a rounding error.
Polars never loses meaningfully: everything at or below 1.0x is noise-level
(GeneralBinning's whole fit+apply takes ~2ms — below the measurement floor).
Sub-100ms rows vary a few tenths of a speedup step between runs on dev
hardware; the ordering is stable.

That is exactly the hybrid design's bet: stay Polars end to end for data
movement, convert only at the sklearn boundary, and pay no penalty for doing so.

### Why is Polars Faster?

1.  **Parallelization:** Polars executes operations in parallel across available CPU cores, whereas Pandas is largely single-threaded.
2.  **Memory Efficiency:** Polars uses Arrow memory format and optimizes memory usage, reducing overhead during large transformations.
3.  **Lazy Evaluation:** (Future Roadmap) While Skyulf currently uses Polars in eager mode for compatibility, the underlying engine allows for query optimization.

## Optimization Tips

To get the most out of Skyulf's performance:

1.  **Use Polars for Ingestion:** When loading data in your backend or scripts, prefer `pl.read_parquet()` or `pl.read_csv()`. Skyulf will detect the Polars DataFrame and stay in the fast lane.
2.  **Batch Processing:** For massive datasets (larger than RAM), consider splitting your data into batches. Skyulf's `Applier` is stateless and thread-safe, making it ideal for parallel batch processing.
3.  **Avoid "Slow" Nodes:** Some Scikit-Learn transformers (like `IterativeImputer` or complex kernel approximations) are inherently computationally expensive and may bottleneck the pipeline regardless of the dataframe engine.
