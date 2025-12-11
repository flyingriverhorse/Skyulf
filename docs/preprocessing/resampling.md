# Resampling

The `resampling` module handles class imbalance using oversampling and undersampling techniques from `imbalanced-learn`. These transformers are typically used within a `FeatureEngineer` pipeline.

> **Note:** Resampling requires all features to be **numeric**. Use encoding transformers (OneHotEncoder, OrdinalEncoder) before resampling.

## Usage Example

```python
from core.ml_pipeline.preprocessing.pipeline import FeatureEngineer
import pandas as pd

# Sample Data (Imbalanced)
df = pd.DataFrame({
    'feature': range(20),
    'target': [0]*14 + [1]*6 # 14 zeros, 6 ones
})

# Define Steps
steps = [
    {
        "name": "split_xy",
        "transformer": "feature_target_split",
        "params": {"target_column": "target"}
    },
    {
        "name": "oversample",
        "transformer": "Oversampling",
        "params": {
            "method": "smote",
            "sampling_strategy": "auto",
            "k_neighbors": 5
        }
    }
]

# Execute
engineer = FeatureEngineer(steps)
resampled_data, metrics = engineer.fit_transform(df)
```

## Available Transformers

### Oversampling
Oversamples the minority class using SMOTE variants.

**Parameters:**
- `method` (str): Oversampling method. One of:
  - `'smote'` (default) - Standard SMOTE
  - `'adasyn'` - Adaptive Synthetic Sampling
  - `'borderline_smote'` - Borderline-SMOTE
  - `'svm_smote'` - SVM-SMOTE
  - `'kmeans_smote'` - KMeans-SMOTE
  - `'smote_tomek'` - SMOTE + Tomek Links cleaning
- `sampling_strategy` (str|float|dict): Sampling strategy. Default `'auto'`.
- `random_state` (int): Random seed. Default `42`.
- `k_neighbors` (int): Number of nearest neighbors for SMOTE. Default `5`.

**Python Config:**
```python
{
    "name": "smote_oversample",
    "transformer": "Oversampling",
    "params": {
        "method": "borderline_smote",
        "sampling_strategy": "minority",
        "k_neighbors": 5,
        "random_state": 42
    }
}
```

### Undersampling
Undersamples the majority class.

**Parameters:**
- `method` (str): Undersampling method. One of:
  - `'random'` (default) - Random undersampling
  - `'nearmiss'` - NearMiss algorithm
  - `'tomek'` - Tomek Links removal
  - `'enn'` - Edited Nearest Neighbours
- `sampling_strategy` (str|float|dict): Sampling strategy. Default `'auto'`.
- `random_state` (int): Random seed. Default `42`.
