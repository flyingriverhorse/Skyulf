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

### Why is Polars Faster?

1.  **Parallelization:** Polars executes operations in parallel across available CPU cores, whereas Pandas is largely single-threaded.
2.  **Memory Efficiency:** Polars uses Arrow memory format and optimizes memory usage, reducing overhead during large transformations.
3.  **Lazy Evaluation:** (Future Roadmap) While Skyulf currently uses Polars in eager mode for compatibility, the underlying engine allows for query optimization.

## Optimization Tips

To get the most out of Skyulf's performance:

1.  **Use Polars for Ingestion:** When loading data in your backend or scripts, prefer `pl.read_parquet()` or `pl.read_csv()`. Skyulf will detect the Polars DataFrame and stay in the fast lane.
2.  **Batch Processing:** For massive datasets (larger than RAM), consider splitting your data into batches. Skyulf's `Applier` is stateless and thread-safe, making it ideal for parallel batch processing.
3.  **Avoid "Slow" Nodes:** Some Scikit-Learn transformers (like `IterativeImputer` or complex kernel approximations) are inherently computationally expensive and may bottleneck the pipeline regardless of the dataframe engine.
