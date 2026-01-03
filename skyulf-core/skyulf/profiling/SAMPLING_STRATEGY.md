# EDA Sampling Strategy

To ensure the EDA module remains responsive even with large datasets (millions of rows), we apply strategic sampling for computationally expensive operations.

| Analysis Module | Sample Size / Strategy | Reason |
| :--- | :--- | :--- |
| **Rule Discovery (Decision Tree)** | **Top 100,000 rows** | Training a decision tree is expensive. 100k rows are sufficient to extract global rules and feature importance. |
| **Outlier Detection (Isolation Forest)** | **Top 50,000 rows** | Isolation Forest complexity is high. 50k rows provide a robust baseline for anomaly detection. |
| **PCA (3D Visualization)** | **Random Sample of 5,000** | Plotting more than 5k points in a 3D browser chart causes lag. We use `sample(n=5000)` to maintain density without freezing the UI. |
| **Causal Discovery (PC Algorithm)** | **Top 5,000 rows** | The PC algorithm has exponential complexity with the number of features. 5k rows are enough to detect conditional independence. |
| **Normality Tests (Shapiro-Wilk)** | **Top 5,000 rows** | The Shapiro-Wilk test is computationally intensive and often too sensitive on large N. Scipy implementation also has a limit of 5000. |
| **Geospatial Analysis** | **Random Sample of 5,000** | Rendering thousands of markers on a Leaflet map slows down the browser. |
| **Time Series Trend** | **Resampled to ~100 points** | If rows > 1000, we aggregate (mean/count) by time interval (1h, 1d, 1w) to show a clean trend line instead of raw noise. |
| **Date Format Detection** | **Top 50 rows** | We only need a small sample to infer the datetime format string. |
| **General Statistics (Mean, Min, Max)** | **Full Dataset (Lazy)** | Basic stats are calculated on the **full dataset** using Polars' efficient lazy evaluation. |

## Note on Sampling Methods
*   **`head(N)`**: Takes the first N rows. This is faster and preserves temporal order, which is crucial for time-series related checks or when data is sorted.
*   **`sample(N)`**: Takes a random set of N rows. This is used for distribution visualizations (PCA, Maps) to ensure the sample is representative of the whole dataset and avoids bias from sorted data.
