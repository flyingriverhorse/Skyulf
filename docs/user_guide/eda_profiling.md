# Automated EDA & Profiling

Skyulf provides a powerful automated Exploratory Data Analysis (EDA) engine powered by **Polars**, designed to handle large datasets efficiently. The `EDAAnalyzer` generates comprehensive statistical profiles, detects data quality issues, and provides actionable insights.

## Quick Start

### The Easy Way (Automated Visualization)

Skyulf provides a `EDAVisualizer` helper that automatically generates a rich terminal dashboard and interactive plots.

> **Note:** This requires the visualization extras: `pip install skyulf-core[viz]`

```python
import polars as pl
from skyulf.profiling.analyzer import EDAAnalyzer
from skyulf.profiling.visualizer import EDAVisualizer

# 1. Load Data
df = pl.read_csv("your_dataset.csv")

# 2. Analyze
analyzer = EDAAnalyzer(df)
profile = analyzer.analyze(
    target_col="target",
    task_type="Classification", # Optional: Force "Classification" or "Regression"
    date_col="timestamp",       # Optional: Manually specify if auto-detection fails
    lat_col="latitude",         # Optional
    lon_col="longitude"         # Optional
)

# 3. Visualize
viz = EDAVisualizer(profile, df)
viz.summary()  # Prints Data Quality, Stats, Outliers, Target Analysis, Alerts to terminal
viz.plot()     # Opens Matplotlib plots (Distributions, Correlations, PCA, etc.)
```

### The Manual Way (Accessing Raw Stats)

If you want to build custom reports or integrate into a pipeline, you can access the raw `DatasetProfile` object.

```python
import polars as pl
from skyulf.profiling.analyzer import EDAAnalyzer

# 1. Load your data into a Polars DataFrame
df = pl.read_csv("your_dataset.csv")

# 2. Initialize the Analyzer
analyzer = EDAAnalyzer(df)

# 3. Run the analysis
# You can optionally specify a target column for supervised analysis
profile = analyzer.analyze(target_col="target_variable")

# 4. Access the results
print(f"Rows: {profile.row_count}")
print(f"Missing Cells: {profile.missing_cells_percentage}%")

# Access column-specific stats
age_stats = profile.columns["age"].stats
print(f"Mean Age: {age_stats.mean}")

# Check for data quality alerts
for alert in profile.alerts:
    print(f"[{alert.severity}] {alert.message}")
```

## Features

### 1. Comprehensive Profiling
The analyzer automatically detects column types (Numeric, Categorical, Date, Text, Boolean) and computes relevant statistics:
*   **Numeric:** Mean, Median, Std Dev, Quantiles, Skewness, Kurtosis, Histogram.
    *   *Normality Tests:* For performance, normality tests are performed on a sample of up to 5,000 rows.
        *   **Shapiro-Wilk:** Used for N < 5,000.
        *   **Kolmogorov-Smirnov:** Used for N >= 5,000 (on a 5k sample).
*   **Categorical:** Unique count, Mode, Frequency distribution.
*   **Date:** Min/Max date, Range, Year/Month distribution.
*   **Text:** Avg length, Common words, **Sentiment Analysis** (Positive/Neutral/Negative).

### 2. Smart Alerts
Skyulf automatically flags potential data quality issues:
*   **High Null Rate:** Columns with >50% missing values.
*   **Constant Columns:** Columns with only 1 unique value.
*   **High Cardinality:** Categorical columns with too many unique values.
*   **High Correlation:** Pairs of features with correlation > 0.95.
*   **Multicollinearity (VIF):** Detects features with Variance Inflation Factor > 5.0.
*   **Class Imbalance:** Target variables with skewed class distributions.
*   **Data Leakage:** Features with 1.0 correlation to the target.

### 3. Advanced Analysis
*   **Outlier Detection:** Uses Isolation Forest to identify anomalous rows.
*   **PCA:** Computes Principal Components to visualize high-dimensional data structure.
*   **Causal Discovery:** Uses the PC Algorithm to infer causal relationships between variables (DAG).
*   **Decision Tree (Rule Discovery):** Uses a surrogate Decision Tree model to extract human-readable rules.
    *   **Classification:** "If Age > 50 -> High Risk (Confidence: 85%)"
    *   **Regression:** "If Age > 50 -> Value = 120.5 (Samples: 100)"
    *   **High Cardinality:** Automatically groups rare classes into "Other" for readable trees (e.g. Top 10 Zip Codes).
*   **Target Analysis:** Analyzes relationships between features and the target variable (Correlations, ANOVA, Interactions).
*   **Geospatial:** Automatically detects Lat/Lon columns and computes bounding boxes.
*   **Time Series:** Detects seasonality and trends in datetime columns.

### 4. Task Type Control
By default, Skyulf automatically detects if your target is **Classification** (Categorical) or **Regression** (Numeric). However, you can override this behavior:

*   **Force Classification:** Useful for ID columns or numeric codes (e.g., Zip Code, Status Code) that should be treated as categories.
*   **Force Regression:** Useful for ordinal categories (e.g., Rating 1-5) that you want to treat as continuous.

```python
# Force Classification on a numeric ID column
profile = analyzer.analyze(target_col="zip_code", task_type="Classification")
```

## Visualization Support (`skyulf-core[viz]`)

The core library is designed to be lightweight. To use the `EDAVisualizer` and generate plots, you must install the optional visualization dependencies:

```bash
pip install skyulf-core[viz]
```

This installs:
*   `matplotlib`: For generating plots.
*   `rich`: For beautiful terminal dashboards.

## Filtering & Exclusion

You can refine your analysis by filtering rows or excluding columns without modifying your original dataframe.

```python
# Run analysis on a subset of data
profile = analyzer.analyze(
    target_col="price",
    # Filter rows where 'region' is 'US' and 'age' > 18
    filters=[
        {"column": "region", "operator": "==", "value": "US"},
        {"column": "age", "operator": ">", "value": 18}
    ],
    # Exclude 'id' and 'timestamp' columns from the report
    exclude_cols=["id", "timestamp"]
)
```

## Advanced Analysis Modules

The `EDAAnalyzer` automatically runs several advanced analysis modules if the data supports them. Here is how to use and interpret each one.

### 1. Outlier Detection
Uses **Isolation Forest** to identify anomalous rows in your dataset. This is useful for cleaning data or detecting fraud/errors.

```python
if profile.outliers:
    print(f"Total Outliers: {profile.outliers.total_outliers}")
    print(f"Outlier Percentage: {profile.outliers.outlier_percentage}%")
    
    # Inspect the top most anomalous rows
    for outlier in profile.outliers.top_outliers[:5]:
        print(f"Row Index: {outlier.index}, Score: {outlier.score}")
        
        # See why it was flagged (feature contribution)
        if outlier.explanation:
            print("  Reason:", outlier.explanation)
```

### 2. Multivariate Analysis (PCA)
Principal Component Analysis (PCA) projects high-dimensional data into 2D or 3D space to visualize clusters and structure.

```python
if profile.pca_data:
    # Access the projected points (x, y, z)
    for point in profile.pca_data[:5]:
        print(f"x: {point.x}, y: {point.y}, Label: {point.label}")
```

### 3. Causal Discovery
Skyulf integrates `causal-learn` to perform causal discovery using the PC Algorithm. This helps distinguish between correlation and causation by building a Directed Acyclic Graph (DAG).

> **Note:** To ensure performance on wide datasets, the causal graph is built using the **Target** and the **top 14 features** most correlated with it (instead of just high variance).

```python
if profile.causal_graph:
    print(f"Graph has {len(profile.causal_graph.nodes)} nodes and {len(profile.causal_graph.edges)} edges.")
    
    for edge in profile.causal_graph.edges:
        # Types: "directed" (->), "undirected" (--), "bidirected" (<->)
        print(f"{edge.source} {edge.type} {edge.target}")
```

### 4. Geospatial Analysis
Automatically detects Latitude/Longitude columns and computes bounding boxes and centroids.

```python
if profile.geospatial:
    print(f"Bounds: {profile.geospatial.min_lat}, {profile.geospatial.min_lon} to {profile.geospatial.max_lat}, {profile.geospatial.max_lon}")
    print(f"Centroid: {profile.geospatial.centroid_lat}, {profile.geospatial.centroid_lon}")
```

### 5. Time Series Analysis
Detects datetime columns and analyzes trends and seasonality. It automatically adjusts the resampling interval (e.g., 1s, 1h, 1d) based on the data duration to provide meaningful trend lines.

```python
if profile.timeseries:
    print(f"Time Column: {profile.timeseries.date_col}")
    
    # Access trend data
    for point in profile.timeseries.trend[:5]:
        print(f"Date: {point.date}, Value: {point.values}")
```

### 6. Smart Date Detection
Skyulf uses a robust "Tournament Strategy" to correctly identify date formats in ambiguous columns (e.g., distinguishing between `MM/DD/YYYY` and `DD/MM/YYYY`).

**How it works:**
1.  **Heuristic Check:** First, it checks column names for keywords like `date`, `time`, `created_at` to avoid expensive parsing on non-date columns.
2.  **Candidate Testing:** It attempts to parse a sample of the data using multiple formats:
    *   ISO Standard (`YYYY-MM-DD`)
    *   Common US Format (`MM/DD/YYYY`)
    *   Common EU Format (`DD/MM/YYYY`)
3.  **Entropy Selection:** For ambiguous cases (e.g., `01/02/2023` could be Jan 2nd or Feb 1st), it calculates the **Month Entropy** (number of unique months) for each candidate format.
    *   It selects the format that yields the highest number of unique months, assuming that real-world data typically spans more than just January.

## Examples

We provide complete, runnable scripts in the `docs/examples/scripts/` directory. These examples demonstrate how to use Skyulf as a library with rich terminal output and visualizations.

### Prerequisites for Examples
To run these examples, you'll need a few extra libraries for visualization and pretty printing:
```bash
pip install matplotlib rich scikit-learn
```

### 1. Time Series Analysis
**Script:** [`docs/examples/scripts/eda_timeseries.py`](../examples/scripts/eda_timeseries.py)

This example generates synthetic sales data with a trend and weekly seasonality. It demonstrates:
- Automatic date column detection.
- Extracting and plotting the **Trend** component.
- Identifying seasonality patterns (e.g., day of week).

**Output:**
- A terminal table showing trend values.
- A plot window showing the trend.

### 2. Geospatial Analysis
**Script:** [`docs/examples/scripts/eda_geospatial.py`](../examples/scripts/eda_geospatial.py)

This example simulates store locations around New York City. It demonstrates:
- Automatic detection of Latitude/Longitude columns.
- Computing the **Centroid** and **Bounding Box**.
- Visualizing the store locations and centroid on a map.

**Output:**
- A terminal summary of spatial stats.
- A scatter plot window.

### 3. Comprehensive Workflow (Iris Dataset)
**Script:** [`docs/examples/scripts/eda_comprehensive.py`](../examples/scripts/eda_comprehensive.py)

A full-featured example using the classic Iris dataset. It covers the entire Skyulf EDA pipeline:
- **Data Quality:** Missing values and duplicates.
- **Numeric Stats:** Skewness, Kurtosis, and Normality Tests (Shapiro-Wilk/D'Agostino).
- **Outlier Detection:** Finding anomalies with Isolation Forest.
- **Distributions:** Histograms for numeric features.
- **Correlations:** Heatmap of feature relationships.
- **Target Analysis:** Boxplots showing feature distribution by target class.
- **Scatter Matrix:** Pairwise scatter plots of features.
- **PCA:** Projecting data into 2D space for visualization.
- **Causal Discovery:** Inferring causal relationships between variables.
- **Smart Alerts:** Automatic warnings about data issues.

**Output:**
- A detailed "Dashboard" in the terminal.
- Plot windows for Distributions, Correlations, Target Analysis, Scatter Matrix, and PCA.

### 4. Automated Analysis (The Easy Way)
**Script:** [`docs/examples/scripts/eda_automated.py`](../examples/scripts/eda_automated.py)

This example demonstrates the simplified workflow using the `EDAVisualizer` class. Instead of writing manual plotting code, you can generate a full report in just 3 lines.

**Output:**
- A rich terminal dashboard.
- All standard plots (Distributions, Correlations, PCA, etc.).

### 5. Time Series & Geospatial (Manual Config)
**Script:** [`docs/examples/scripts/eda_timeseries_geo.py`](../examples/scripts/eda_timeseries_geo.py)

This example shows how to handle cases where auto-detection might fail or when you want to be explicit. It demonstrates:
- Manually specifying `date_col`, `lat_col`, and `lon_col` in the `analyze()` method.
- Visualizing both Time Series trends and Geospatial maps in the same report.

**Output:**
- A terminal summary including Time Series and Geospatial sections.
- Plots for Time Series Trend and Geospatial Distribution.

## Performance

The profiler is built on **Polars**, making it significantly faster than Pandas-based alternatives like `pandas-profiling` or `ydata-profiling`, especially for datasets with millions of rows.

## Data Drift & Monitoring

Skyulf includes **Data Drift Detection** module to monitor how your data changes over time, ensuring model reliability in production.

### Concept
Data Drift occurs when the statistical properties of the input data (production) change compared to the data used to train the model (reference). This can lead to model degradation.

### Metrics
Skyulf calculates drift using four key statistical metrics for every numerical column:

1.  **Wasserstein Distance (Earth Mover's Distance):** Measures the "work" needed to transform one distribution into the other. Good for detecting shifts in shape and location.
2.  **KS Test (Kolmogorov-Smirnov):** A non-parametric test that compares cumulative distribution functions. The p-value indicates the probability that the two samples come from the same distribution.
3.  **PSI (Population Stability Index):** A widely used industry standard for measuring population shifts.
    *   `PSI < 0.1`: No significant drift.
    *   `0.1 <= PSI < 0.25`: Moderate drift.
    *   `PSI >= 0.25`: Significant drift (Action required).
4.  **KL Divergence (Kullback-Leibler):** Measures how one probability distribution diverges from a second, expected probability distribution.

### Schema Drift
The system also monitors for structural changes:
*   **Missing Columns:** Critical alerts for features present in training but missing in production.
*   **New Columns:** Alerts for unexpected features appearing in production data.

### Using the Data Drift UI
1.  Navigate to the **Data Drift** page in the ML Canvas.
2.  **Step 1:** Select a **Reference Job** (Training Job) from the list. This loads the statistical profile of the data used during training.
3.  **Step 2:** Upload your **Current Data** (Production Data) as a CSV or Parquet file.
4.  **Analyze:** Click "Run Analysis".
5.  **Review Report:**
    *   Check the **Drift Score** and **Schema Alerts** at the top.
    *   Expand any column row to see detailed metrics and **Interactive Histograms** comparing the two distributions side-by-side.

### Using the Library (Python API)
**Script:** [`docs/examples/scripts/data_drift_check.py`](../examples/scripts/data_drift_check.py)

You can also use the drift detection engine programmatically within your own scripts or pipelines using `skyulf-core`.

```python
import polars as pl
from skyulf.profiling.drift import DriftCalculator

# 1. Load Data (Reference vs Current)
# Reference: The data your model was trained on
# Current: The new data from production
ref_df = pl.read_csv("training_data.csv")
curr_df = pl.read_csv("production_data.csv")

# 2. Initialize Calculator
calculator = DriftCalculator(ref_df, curr_df)

# 3. Calculate Drift
# You can optionally override default thresholds
report = calculator.calculate_drift(thresholds={"psi": 0.2, "wasserstein": 0.1})

# 4. Inspect Results
print(f"Drifted Columns: {report.drifted_columns_count}")

# Check for Schema Drift
if report.missing_columns:
    print(f"⚠️ Missing Columns: {report.missing_columns}")
if report.new_columns:
    print(f"ℹ️ New Columns: {report.new_columns}")

# Check for Statistical Drift
for col_name, drift_info in report.column_drifts.items():
    if drift_info.drift_detected:
        print(f"\n🚨 Drift detected in '{col_name}':")
        for metric in drift_info.metrics:
            status = "FAIL" if metric.has_drift else "PASS"
            print(f"  - {metric.metric}: {metric.value:.4f} [{status}]")
        
        if drift_info.suggestions:
            print(f"  💡 Suggestion: {drift_info.suggestions[0]}")
```

