# Skyulf Profiling vs. YData Profiling vs. Sweetviz

Choosing the right EDA (Exploratory Data Analysis) tool matters. This guide
offers a practical comparison between Skyulf's profiling engine and two
popular alternatives: YData Profiling (formerly pandas-profiling) and
Sweetviz. The external-tool claims were checked against their official
documentation on 2026-09-07; availability and performance can change by
version, configuration, and dataset.

## TL;DR

| Feature | Skyulf | YData Profiling | Sweetviz |
|---------|--------|-----------------|----------|
| **Backend** | Polars (Rust) | Pandas; limited Spark profiling mode | Pandas |
| **Outputs** | Structured profile object; optional terminal/plots | HTML report; JSON export; notebook widgets/iframe | Self-contained HTML report; notebook output |
| **Large-data options** | Polars supports parallel and streaming workflows | Minimal mode; limited Spark feature set | Pandas-based; no distributed mode documented |
| **Target-Aware Analysis** | Yes | Yes | Yes |
| **Dataset comparison report** | No | Yes (not currently for Spark) | Yes |
| **Time series analysis** | Yes | Yes | Not advertised |
| **Causal discovery (PC algorithm)** | Yes | Not advertised | Not advertised |
| **Rule extraction (surrogate decision tree)** | Yes | Not advertised | Not advertised |
| **Model-based outlier detection (Isolation Forest)** | Yes | Not advertised | Not advertised |
| **PCA projection** | Yes | Not advertised | Not advertised |
| **Geospatial analysis (lat/lon detection)** | Yes | Not advertised | Not advertised |
| **Subset profiling via filters** | Yes | No | No |
| **Normality / stationarity diagnostics** | Yes | Yes (including time-series stationarity) | Not advertised |
| **ANOVA p-values (target interactions)** | Yes (optional SciPy) | Not advertised | Not advertised |
| **Feature importance (from surrogate tree)** | Yes (scikit-learn) | Not advertised | Not advertised |
| **Recommendations (drop/impute/encode hints)** | Yes | Not advertised | Not advertised |
| **PII heuristics (email/phone)** | Yes (advisory) | Not advertised | Not advertised |
| **Leakage warnings (high corr to target)** | Yes | Not advertised | Not advertised |
| **Correlation Matrix** | Yes | Yes | Yes |
| **Missing Value Analysis** | Yes | Yes | Yes |
| **Duplicate Detection** | Yes | Yes | Yes |

---

## PII alerts

Skyulf includes a narrow, advisory PII heuristic in its profiler. It checks
Text and Categorical columns for values that look like email addresses or
phone numbers and adds a `PII` alert to the profile when it finds a possible
match. It does not modify, mask, delete, block, or classify the data.

This is not a complete sensitive-data scanner: it does not identify every
kind of personal data, such as names, addresses, national identifiers,
payment-card numbers, or health information. Unusual formatted values can
still produce false positives, so findings should be reviewed before taking
action.

For the direct profile accessors, example output, and current limitations, see
the [PII Detection guide](../user_guide/pii_detection.md).

---

## The Honest Truth

### Where Skyulf Excels

**1. Performance on Large Datasets**

Skyulf is built on Polars, whose core is written in Rust and supports
parallel and streaming workflows. That can make Skyulf a good fit for larger
Polars-native workloads, but this page does not claim a universal speed or
memory win: actual performance depends on the operation, data shape, hardware,
and profiling options. YData also offers a limited Spark profiling mode and a
minimal configuration for larger datasets, while Sweetviz remains
Pandas-based.

See the [Polars user guide](https://docs.pola.rs/) and YData's
[big-data documentation](https://docs.profiling.ydata.ai/latest/features/big_data/)
for the implementation details behind those statements.

**2. ML-Focused Analysis**

Skyulf was designed with machine learning workflows in mind, not just
descriptive statistics. Its profile can include:

- **Causal Discovery:** Using the PC algorithm from `causal-learn`, Skyulf can
  infer potential causal relationships between variables. These are hypotheses,
  not proof of causation. Causal discovery is not listed among the official
  feature descriptions for YData Profiling or Sweetviz that this comparison
  checked.

- **Rule Extraction:** Skyulf trains a surrogate Decision Tree on your data
  and extracts human-readable rules such as "If Age > 50 AND Income < 30k →
  High Risk". This is useful for exploratory explanation, but it is not a
  substitute for a model-specific explanation or causal analysis.

- **Feature Importance (from the surrogate tree):** Alongside rules, Skyulf exposes feature importances from the surrogate Decision Tree. This is not a replacement for model explainability, but it's a fast way to see which columns dominate the tree's decisions.

- **Outlier Detection:** Built-in Isolation Forest identifies anomalous rows
  and reports which features deviate most from the median. This row-level
  model-based output is distinct from the distribution summaries in the
  compared tools.

- **PCA Projection:** Skyulf computes 2D/3D PCA projections colored by target class, helping you visually assess class separability before training a model.

- **Target Interactions with ANOVA (p-values):** For categorical targets, Skyulf can compute ANOVA p-values for numeric features (when SciPy is available) and rank associations accordingly. This helps you quickly find features that differ meaningfully across target classes.

**3. Specialized Domain Analysis**

- **Geospatial:** If your data contains latitude/longitude columns, Skyulf automatically detects them and provides bounding box statistics plus sample points for map visualization.

- **Time Series:** Skyulf detects datetime columns and analyzes trends, seasonality (day-of-week, month-of-year patterns), and stationarity. This context is critical before building forecasting models.

**4. Structured, API-first output**

Skyulf returns a structured profile object (serializable to JSON) and keeps
visualization optional. This makes it easy to:
- Integrate profiling into automated pipelines
- Build custom dashboards
- Store profiles in databases for tracking data drift over time
- Apply dynamic filters and re-analyze subsets

If you want a polished, self-contained HTML artifact, YData Profiling or
Sweetviz is a better fit today. Skyulf focuses on programmatic profiling that
can be embedded into an ML workflow.

---

## When to Use What

| Scenario | Recommended Tool |
|----------|------------------|
| Larger Polars-native workload | **Skyulf**, subject to benchmarking |
| Larger dataset with distributed Spark profiling | **YData Profiling** |
| Need causal inference or rule extraction | **Skyulf** |
| Building a Skyulf ML pipeline with structured profile output | **Skyulf** |
| Geospatial or time series data | **Skyulf** |
| Sharing HTML reports with business users | **YData Profiling** |
| Quick one-off HTML EDA on small datasets | **Sweetviz** or **YData Profiling** |
| Comparing train/test splits visually | **Sweetviz** or **YData Profiling** |
| Spark environment (within YData's supported feature set) | **YData Profiling** |

---

## Quick Start with Skyulf

```python
import polars as pl
from skyulf import EDAAnalyzer, EDAVisualizer

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
- [PII Detection](../user_guide/pii_detection.md)
- [Polars User Guide](https://docs.pola.rs/)
- [YData Profiling documentation](https://docs.profiling.ydata.ai/latest/)
- [YData big-data support](https://docs.profiling.ydata.ai/latest/features/big_data/)
- [Sweetviz official documentation](https://github.com/fbdesignpro/sweetviz)
