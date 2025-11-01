# Lightweight EDA API Documentation

## Overview

The `skyulf_mlflow.eda` package provides quick, dependency-light helpers for assessing dataset quality, understanding text columns, and inferring the likely business domain of a table. These utilities are designed for exploratory workflows, notebook experiments, and lightweight services where you need actionable insights without running a full analytical pipeline.

**Key Components**
- **`generate_quality_report`** (`quality.py`): Profile numerical and categorical columns, missing values, and dataset health indicators.
- **`QualityReport`**: Data class encapsulating quality summary information and recommended next steps.
- **`get_text_insights`** (`text.py`): Analyze object columns for NLP readiness, length distribution, tokenization hints, and enrichment suggestions.
- **`DomainAnalyzer` / `infer_domain`** (`domain.py`): Score column patterns, optional ML features, and produce tailored recommendations for the detected domain (see also [Domain Analysis](domain_analysis.md)).

---

## 📊 Quality Reporting

```python
from skyulf_mlflow_library.eda import generate_quality_report

report = generate_quality_report(df)
print(report.summary)
print(report.recommendations[0])
```

### Highlights
- Detects missing values, dtype mismatches, duplicate rows, and outliers
- Surfaces column-level stats (min, max, mean, missing counts) for quick audits
- Generates human-readable recommendations for cleansing and validation steps

`QualityReport` fields include:
- `summary`: Natural-language summary of dataset health
- `column_diagnostics`: Per-column profile (dtype, unique count, missing percentage)
- `recommendations`: List of actionable suggestions

---

## ✍️ Text Insights

```python
from skyulf_mlflow_library.eda import get_text_insights

text_details = get_text_insights(df[["feedback", "comments"]])
for column, info in text_details.items():
    print(column, info["avg_length"], info["suggested_use_cases"])
```

### What You Get
- Average/median text lengths and character distributions
- Punctuation vs alphanumeric ratios to spot noisy inputs
- Suggested downstream workflows (sentiment analysis, topic modeling, entity extraction)
- Language detection heuristics for multi-lingual datasets

Use these signals to triage which columns merit NLP preprocessing or annotation work.

---

## 🧠 Domain Inference

The domain analyzer combines rule-based heuristics with an optional gradient-boosted classifier to guess the dataset's business vertical.

```python
from skyulf_mlflow_library.eda import DomainAnalyzer

analyzer = DomainAnalyzer(enable_ml_classifier=False)
result = analyzer.analyze_dataset_domain(df.columns, df)
print(result.primary_domain)
print(result.recommendations[:3])
```

See the dedicated [Domain Analysis API](domain_analysis.md) for complete reference, classifier configuration, and interpretation guidance.

---

## 🧪 Example Usage

Run `examples/12_eda_overview.py` for an end-to-end demonstration of all three utilities (quality, text, domain). For domain-specific scenarios, inspect `examples/11_domain_analyzer.py`.

---

## Best Practices

1. **Start small**: Sample 1–5k rows before generating reports to keep runtimes low.
2. **Treat text separately**: Pass only object columns to `get_text_insights` for faster analysis.
3. **Disable ML when packaging**: Set `enable_ml_classifier=False` by default to avoid extra dependencies unless you ship the classifier artifact.
4. **Log metadata**: Store `QualityReport.to_dict()` and `DomainInferenceResult.to_dict()` outputs for monitoring data drift over time.

The EDA helpers ship with Skyulf-MLFlow starting in **v0.1.1** and require pandas ≥1.5 and numpy ≥1.21.
