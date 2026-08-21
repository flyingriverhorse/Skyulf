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

### Why is Polars Faster?

1.  **Parallelization:** Polars executes operations in parallel across available CPU cores, whereas Pandas is largely single-threaded.
2.  **Memory Efficiency:** Polars uses Arrow memory format and optimizes memory usage, reducing overhead during large transformations.
3.  **Lazy Evaluation:** (Future Roadmap) While Skyulf currently uses Polars in eager mode for compatibility, the underlying engine allows for query optimization.

## Optimization Tips

To get the most out of Skyulf's performance:

1.  **Use Polars for Ingestion:** When loading data in your backend or scripts, prefer `pl.read_parquet()` or `pl.read_csv()`. Skyulf will detect the Polars DataFrame and stay in the fast lane.
2.  **Batch Processing:** For massive datasets (larger than RAM), consider splitting your data into batches. Skyulf's `Applier` is stateless and thread-safe, making it ideal for parallel batch processing.
3.  **Avoid "Slow" Nodes:** Some Scikit-Learn transformers (like `IterativeImputer` or complex kernel approximations) are inherently computationally expensive and may bottleneck the pipeline regardless of the dataframe engine.
