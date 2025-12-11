# Recommendation Engine

The Recommendation Engine analyzes your dataset's statistical profile and suggests appropriate preprocessing steps. This helps automate the feature engineering process by identifying common data issues and proposing standard solutions.

## Overview

The engine works by:
1.  Taking an `AnalysisProfile` (generated from your dataset).
2.  Running it through a series of "Advisors" (plugins).
3.  Aggregating and ranking the recommendations based on confidence.

## Usage Example

```python
from core.ml_pipeline.recommendations.engine import AdvisorEngine
from core.ml_pipeline.recommendations.schemas import AnalysisProfile, ColumnProfile, ColumnType

# 1. Create or Load a Profile
# In a real scenario, this comes from the DatasetProfile node output
profile = AnalysisProfile(
    row_count=1000,
    column_count=2,
    duplicate_row_count=50,
    columns={
        "age": ColumnProfile(
            name="age",
            dtype="float",
            column_type=ColumnType.NUMERIC,
            missing_count=100,
            missing_ratio=0.1,
            unique_count=50
        ),
        "city": ColumnProfile(
            name="city",
            dtype="object",
            column_type=ColumnType.CATEGORICAL,
            missing_count=0,
            missing_ratio=0.0,
            unique_count=5
        )
    }
)

# 2. Initialize Engine
engine = AdvisorEngine()

# 3. Get Recommendations
recommendations = engine.analyze(profile)

# 4. Inspect Recommendations
for rec in recommendations:
    print(f"[{rec.confidence:.2f}] {rec.rule_id}: {rec.description}")
    print(f"  -> Suggested Node: {rec.suggested_node_type}")
    print(f"  -> Params: {rec.suggested_params}")
```

## Available Advisors

The engine includes several built-in advisors:

*   **CleaningAdvisor**: Detects high missing rates, constant columns, and duplicate rows.
*   **ImputationAdvisor**: Suggests imputation strategies (mean/median/mode) based on data type and missingness.
*   **ScalingAdvisor**: Recommends scaling (StandardScaler/MinMaxScaler) for numeric features.
*   **EncodingAdvisor**: Suggests OneHot or Ordinal encoding for categorical features based on cardinality.
*   **OutlierAdvisor**: Identifies potential outliers and suggests removal or clipping.
*   **TransformationAdvisor**: Detects skewed distributions and suggests power transformations (e.g., Yeo-Johnson).
*   **ResamplingAdvisor**: Checks for class imbalance in classification targets.
*   **FeatureGenerationAdvisor**: Suggests interaction features for numeric columns.
*   **BucketingAdvisor**: Recommends binning for high-cardinality numeric features.

## Extending the Engine

You can create custom advisors by inheriting from `BaseAdvisorPlugin` and registering them with the engine.

```python
from core.ml_pipeline.recommendations.plugins.base import BaseAdvisorPlugin
from core.ml_pipeline.recommendations.schemas import Recommendation

class MyCustomAdvisor(BaseAdvisorPlugin):
    def analyze(self, profile):
        # Custom logic here
        return []

engine.register_plugin(MyCustomAdvisor())
```
