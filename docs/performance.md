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
*   **Measurement:** median of 3 runs per node (model fits: 1 run).
*   **Correctness:** before any timing, the script runs a **parity check** —
    each node is fit+applied on both engines and the outputs are compared
    value-for-value (numerics at 1e-9 tolerance, strings exactly). A row only
    appears in the table if both engines produced identical results.

| Node | pandas | polars | Speedup |
| :--- | :--- | :--- | :--- |
| **SimpleImputer** | 0.166s | **0.003s** | **58.1x** |
| **HashEncoder** | 0.040s | **0.010s** | **4.18x** |
| **MinMaxScaler** | 0.025s | **0.007s** | **3.45x** |
| **TrainTestSplitter** | 0.040s | **0.018s** | **2.17x** |
| **GeneralBinning** | 0.016s | **0.009s** | **1.80x** |
| **LabelEncoder** | 0.041s | **0.027s** | **1.52x** |
| **Winsorize** | 0.029s | **0.019s** | **1.51x** |
| **ZScore** | 0.022s | **0.015s** | **1.48x** |
| **RobustScaler** | 0.074s | **0.056s** | **1.33x** |
| **StandardScaler** | 0.048s | **0.037s** | **1.30x** |
| **EllipticEnvelope** | 0.090s | **0.071s** | **1.26x** |
| **IQR** | 0.029s | **0.026s** | **1.14x** |
| **OrdinalEncoder** | 0.104s | **0.093s** | **1.12x** |
| **Tokenizer** | 0.572s | **0.509s** | **1.12x** |
| **LogisticRegression** | 0.104s | **0.095s** | **1.10x** |
| **PowerTransformer** | 1.437s | **1.381s** | **1.04x** |
| **CountVectorizer** | 1.173s | **1.132s** | **1.04x** |
| **GradientBoosting (n=30)** | 10.439s | **10.285s** | **1.02x** |
| **TfidfVectorizer** | 1.141s | 1.163s | 0.98x |
| **XGBoost (n=30)** | 0.141s | 0.143s | 0.98x |
| **RandomForest (n=20)** | 0.749s | 0.769s | 0.97x |
| **OneHotEncoder** | 0.225s | 0.233s | 0.96x |

**Read:** data-munging nodes (imputation, scaling, binning, splitting, hash
encoding) win big on Polars — up to 58x. Nodes whose runtime is dominated by
scikit-learn / XGBoost compute (PowerTransformer, text vectorizers, all model
fits) land at ~1.0x, because their work happens on the other side of the
unavoidable sklearn boundary and the frame conversion is a rounding error.
Polars never loses meaningfully: the worst case is noise-level. Sub-100ms rows
vary a few tenths of a speedup step between runs on dev hardware; the ordering
is stable.

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
