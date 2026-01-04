# Skyulf Profiling vs. YData Profiling vs. Sweetviz

Choosing the right EDA (Exploratory Data Analysis) tool matters. This guide offers a practical, honest comparison between Skyulf's profiling engine and two popular alternatives: YData Profiling (formerly pandas-profiling) and Sweetviz.

## TL;DR

| Feature | Skyulf | YData Profiling | Sweetviz |
|---------|--------|-----------------|----------|
| **Backend** | Polars (Rust) | Pandas (optional Spark support) | Pandas |
| **Outputs** | JSON profile object; optional terminal/plots | HTML report; JSON export; notebook widgets | HTML report; notebook embedding |
| **Scales to larger data** | Often better (Polars-based) | Can scale via Spark; Pandas mode can be heavy | Can be heavy (Pandas-based) |
| **Target-Aware Analysis** | Yes | Yes | Yes |
| **Dataset comparison report** | No | Yes | Yes |
| **Time series analysis** | Yes | Yes | Not advertised |
| **Causal discovery (PC algorithm)** | Yes | Not advertised | Not advertised |
| **Rule extraction (surrogate decision tree)** | Yes | Not advertised | Not advertised |
| **Model-based outlier detection (Isolation Forest)** | Yes | Not advertised | Not advertised |
| **PCA projection** | Yes | Not advertised | Not advertised |
| **Geospatial analysis (lat/lon detection)** | Yes | Not advertised | Not advertised |
| **Subset profiling via filters** | Yes | No | No |
| **Normality / stationarity tests** | Yes | No | No |
| **ANOVA p-values (target interactions)** | Yes (optional SciPy) | Not advertised | Not advertised |
| **Feature importance (from surrogate tree)** | Yes (scikit-learn) | Not advertised | Not advertised |
| **Recommendations (drop/impute/encode hints)** | Yes | Not advertised | Not advertised |
| **PII heuristics (email/phone)** | Yes | Not advertised | Not advertised |
| **Leakage warnings (high corr to target)** | Yes | Not advertised | Not advertised |
| **Correlation Matrix** | Yes | Yes | Yes |
| **Missing Value Analysis** | Yes | Yes | Yes |
| **Duplicate Detection** | Yes | Yes | Yes |

---

## The Honest Truth

### Where Skyulf Excels

**1. Performance on Large Datasets**

Skyulf is built on Polars, a DataFrame library written in Rust. This makes a real difference when you're working with datasets over 100K rows. YData Profiling and Sweetviz both rely on Pandas, which can become painfully slow and memory-hungry on larger datasets.

If you regularly work with datasets that push RAM limits, Polars-based workflows tend to be more resilient than Pandas-based ones. (Exact timing depends heavily on data types, cardinality, and what options you enable.)

**2. ML-Focused Analysis**

Skyulf was designed with machine learning workflows in mind, not just descriptive statistics. This means:

- **Causal Discovery:** Using the PC algorithm from `causal-learn`, Skyulf can infer potential causal relationships between variables. This helps you understand not just correlations, but which features might actually drive your target. Neither YData nor Sweetviz offers this.

- **Rule Extraction:** Skyulf trains a surrogate Decision Tree on your data and extracts human-readable rules like "If Age > 50 AND Income < 30k → High Risk". This is invaluable for fraud detection, churn analysis, or any use case where you need to explain segments to stakeholders.

- **Feature Importance (from the surrogate tree):** Alongside rules, Skyulf exposes feature importances from the surrogate Decision Tree. This is not a replacement for model explainability, but it’s a fast way to see which columns dominate the tree’s decisions.

- **Outlier Detection:** Built-in Isolation Forest identifies anomalous rows and explains *why* they're outliers (which features deviate most from the median). YData shows distribution plots but doesn't flag specific outlier rows.

- **PCA Projection:** Skyulf computes 2D/3D PCA projections colored by target class, helping you visually assess class separability before training a model.

- **Target Interactions with ANOVA (p-values):** For categorical targets, Skyulf can compute ANOVA p-values for numeric features (when SciPy is available) and rank associations accordingly. This helps you quickly find features that differ meaningfully across target classes.

**3. Specialized Domain Analysis**

- **Geospatial:** If your data contains latitude/longitude columns, Skyulf automatically detects them and provides bounding box statistics plus sample points for map visualization.

- **Time Series:** Skyulf detects datetime columns and analyzes trends, seasonality (day-of-week, month-of-year patterns), and stationarity. This context is critical before building forecasting models.

**4. API-First Design**

Skyulf returns structured JSON (Pydantic models) rather than HTML. This makes it easy to:
- Integrate profiling into automated pipelines
- Build custom dashboards
- Store profiles in databases for tracking data drift over time
- Apply dynamic filters and re-analyze subsets

If you want an HTML artifact you can email around, Skyulf is not trying to replace YData/Sweetviz today. Skyulf focuses on programmatic profiling that you can embed into products.

---

## When to Use What

| Scenario | Recommended Tool |
|----------|------------------|
| Large dataset (500K+ rows) | **Skyulf** |
| Need causal inference or rule extraction | **Skyulf** |
| Building ML pipelines (need JSON output) | **Skyulf** |
| Geospatial or time series data | **Skyulf** |
| Sharing HTML reports with business users | **YData Profiling** |
| Quick one-off HTML EDA on small datasets | **Sweetviz** or **YData Profiling** |
| Comparing train/test splits visually | **Sweetviz** |
| Spark/distributed environment | **YData Profiling** |

---

## Quick Start with Skyulf

```python
import polars as pl
from skyulf.profiling.analyzer import EDAAnalyzer
from skyulf.profiling.visualizer import EDAVisualizer

# 1. Load Data
df = pl.read_csv("your_dataset.csv")

# 2. Run Analysis
analyzer = EDAAnalyzer(df)
profile = analyzer.analyze(target_col="target")

# 3. Visualize Results (The Easy Way)
# This single class handles all the rich terminal output and matplotlib plots
viz = EDAVisualizer(profile, df)

# Print the dashboard
viz.summary()

# Show the plots
viz.plot()
```

---

## Conclusion

There's no universally "best" profiling tool. Choose based on your needs:

- If you want a polished HTML report: **YData Profiling** or **Sweetviz**.
- If you want ML-oriented signals (rules, outliers, causal hypotheses) and an API-first profile object: **Skyulf**.

---

## Related Resources

- [Skyulf EDA Documentation](../user_guide/eda_profiling.md)
- [YData Profiling Documentation](https://docs.profiling.ydata.ai/)
- [Sweetviz GitHub](https://github.com/fbdesignpro/sweetviz)
