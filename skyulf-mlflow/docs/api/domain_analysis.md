# Domain Analysis API Documentation

## Overview

The domain analysis module helps identify the most likely business vertical for a dataset using a mix of column-name heuristics, value profiling, and an optional machine learning classifier. It returns actionable recommendations tailored to the inferred domain so that data teams can accelerate their exploratory analysis.

**Available Components:**
- **`DomainAnalyzer`**: Main class that inspects a pandas `DataFrame` and produces a `DomainInferenceResult`
- **`DomainInferenceResult`**: Structured response containing the primary domain, secondary candidates, confidence scores, detected patterns, and recommendations
- **`infer_domain()`**: Convenience helper for one-line inference when you already have a `DataFrame`

---

## 🚀 Quick Start

```python
import pandas as pd
from skyulf_mlflow_library.eda import DomainAnalyzer

# Sample retail dataset
sales = pd.DataFrame(
    {
        "order_id": ["A-100", "A-101", "A-102", "A-103"],
        "customer_id": ["C001", "C002", "C001", "C003"],
        "city": ["New York", "Los Angeles", "New York", "Chicago"],
        "order_date": pd.date_range("2024-01-01", periods=4, freq="D"),
        "revenue": [420.0, 560.5, 315.0, 680.0],
        "channel": ["Web", "Web", "Store", "Partner"],
    }
)

analyzer = DomainAnalyzer(enable_ml_classifier=False)
result = analyzer.analyze_dataset_domain(sales.columns, sales)

print(result.primary_domain)            # "retail"
print(result.primary_confidence)        # 0.62 (example value)
print(result.recommendations[:2])       # First two action items for analysts
```

`DomainAnalyzer` automatically handles:
- Column pattern detection (price, customer, marketing, location, environmental signals)
- Text vs numeric profiling, mixed feature detection, outlier flags
- Object detection and computer-vision heuristics (bounding boxes, masks, image URLs)
- Time-series hints from datetime columns or parseable strings
- Optional ML classifier if `domain_analyzer_xgb.pkl` is available (see below)

---

## 🧠 DomainAnalyzer Class

```python
from skyulf_mlflow_library.eda import DomainAnalyzer

DomainAnalyzer(
    *,
    enable_ml_classifier: bool = True,
    ml_model_path: Optional[Union[str, Path]] = None,
    additional_model_paths: Optional[Iterable[Union[str, Path]]] = None,
    ml_min_confidence: float = 0.7,
)
```

### Parameters
- `enable_ml_classifier`: Toggle the optional gradient boosted classifier for additional accuracy
- `ml_model_path`: Explicit path to a serialized artifact (joblib or pickle) containing `{"model", "label_encoder", "feature_names"}`
- `additional_model_paths`: Iterable of extra paths to scan when searching for the classifier artifact
- `ml_min_confidence`: Minimum probability required before the ML prediction can override the heuristic result

When `enable_ml_classifier` is `True`, the analyzer searches these locations by default:

1. `skyulf_mlflow/eda/models/domain_analyzer_xgb.pkl`
2. `skyulf_mlflow/eda/model_ml/domain_analyzer_xgb.pkl` (legacy docs artifact)

If no artifact is found, the analyzer gracefully falls back to purely rule-based inference.

### Public Method

```python
DomainAnalyzer.analyze_dataset_domain(
    columns: Optional[Sequence[str]],
    sample_data: Union[pd.DataFrame, Sequence[Dict[str, Any]], Sequence[Sequence[Any]]],
    *,
    metadata: Optional[Dict[str, Any]] = None,
) -> DomainInferenceResult
```

- Accepts either a `DataFrame` or any sequence of records (dicts/rows) plus column names
- Normalises the sample into a `DataFrame` under the hood
- Computes aggregate scores for built-in domains (finance, retail, healthcare, fraud, marketing, computer vision, environmental, geospatial, etc.)
- Returns rich metadata about detected patterns (time series, text columns, object detection hints, image data, geospatial columns, outliers, missing values)

---

## 📦 DomainInferenceResult

The result object surfaces everything you need for downstream reporting:

```python
from skyulf_mlflow_library.eda import DomainInferenceResult

result.primary_domain            # str
result.primary_confidence        # float in [0, 0.95]
result.secondary_domains         # List[{"domain": str, "confidence": float}]
result.domain_scores             # Dict[str, float]
result.recommendations           # Up to 6 targeted suggestions
result.patterns                  # {"column_patterns": {...}, "data_patterns": {...}}
result.ml_prediction             # Optional ML metadata when classifier is enabled
result.to_dict()                 # Serialise for JSON APIs or logging
```

**Common values in `patterns`:**
- `column_patterns`: Counts of column name matches (price, marketing, environmental, geospatial, object_detection)
- `data_patterns`: Flags like `has_time_series`, `has_text_columns`, `has_image_data`, `has_object_detection_annotations`, `has_environmental_signals`

---

## ⚡ infer_domain Helper

For the quickest path from `DataFrame` to insight:

```python
from skyulf_mlflow_library.eda import infer_domain

summary = infer_domain(sales)
print(summary.primary_domain)
print(summary.recommendations)
```

Internally this constructs a `DomainAnalyzer` with `enable_ml_classifier=False` for deterministic, dependency-light usage. Pass your own analyzer via the `analyzer=` keyword to reuse a classifier-enabled instance.

---

## 🔍 Example Output Snapshot

```python
>>> from pprint import pprint
>>> pprint(result.to_dict())
{
    'primary_domain': 'retail',
    'primary_confidence': 0.62,
    'secondary_domains': [
        {'domain': 'marketing', 'confidence': 0.41},
        {'domain': 'time_series', 'confidence': 0.18},
    ],
    'domain_scores': {
        'retail': 6.2,
        'marketing': 4.1,
        'time_series': 1.8,
        'computer_vision': 0.0,
        'environmental': 0.0,
        ...
    },
    'recommendations': [
        'Analyze customer purchase patterns',
        'Examine seasonal trends in sales',
        'Check product performance metrics',
        'Identify customer segmentation opportunities',
        'Address missing values before analysis',
    ],
    'patterns': {
        'column_patterns': {
            'price': 1.0,
            'marketing': 2.0,
            'location': 1.5,
        },
        'data_patterns': {
            'has_time_series': True,
            'has_text_columns': True,
            'has_missing_values': False,
            'has_outliers': False,
        },
    },
}
```

---

## 🧪 Testing & Examples

- Run `examples/11_domain_analyzer.py` for a CLI demonstration that prints domain scores, patterns, and next-step recommendations
- Add the inference result to automated pipelines by serialising `result.to_dict()`
- For regression tests, assert against `primary_domain` and key flags rather than exact scores (which may change as heuristics evolve)

---

## ✅ Best Practices

1. **Start heuristic-only**: Keep `enable_ml_classifier=False` for reproducible CI builds and lightweight environments
2. **Provide clean column names**: Explicit business terms ("revenue", "policy_id", "bbox") significantly improve scoring accuracy
3. **Sample carefully**: Use representative rows; the analyzer defaults to inspecting up to 5k rows when the classifier is enabled
4. **Leverage metadata**: Pass `metadata={"dataset_name": "..."}` to embed context in downstream logs or dashboards
5. **Monitor model drift**: When the optional classifier is enabled, retrain or review the artifact whenever your data schema changes

---

## 🔄 Release Notes

- Available since **v0.1.1**
- Public imports: `from skyulf_mlflow_library.eda import DomainAnalyzer, DomainInferenceResult, infer_domain`
- Works seamlessly with pandas 1.5+ and numpy 1.21+

For questions or feedback, open an issue on the [Skyulf MLflow repository](https://github.com/skyulf/skyulf-mlflow/issues).
