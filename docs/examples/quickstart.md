# Quickstart Guide

This guide demonstrates how to create a simple end-to-end pipeline using Skyulf Core.

## 1. Setup

First, import the necessary modules.

```python
import numpy as np
import pandas as pd
from skyulf import SkyulfPipeline
```

## 2. Create Dummy Data

We'll create a synthetic dataset with some missing values and categorical features.

```python
def create_dummy_data(n: int = 200) -> pd.DataFrame:
    np.random.seed(42)
    df = pd.DataFrame({
        'age': np.random.randint(18, 80, n),
        'income': np.random.normal(50000, 15000, n),
        'city': np.random.choice(['New York', 'London', 'Paris'], n),
        'is_customer': np.random.choice([0, 1], n),
    })
    # Introduce missing values
    df.loc[0:10, 'income'] = np.nan
    return df

data = create_dummy_data()
print(data.head())
```

**Output:**
```text
   age        income      city  is_customer
0   56           NaN     Paris            0
1   69           NaN  New York            1
2   46           NaN     Paris            0
3   32           NaN     Paris            0
4   60           NaN  New York            0
```

## 3. Define Pipeline Configuration

Skyulf pipelines are defined using a JSON-compatible dictionary. This makes them easy to serialize and store.

```python
config = {
    'preprocessing': [
        # 1. Split Data into Train/Test
        {
            'name': 'split_data',
            'transformer': 'TrainTestSplitter',
            'params': {
                'test_size': 0.2,
                'target_column': 'is_customer',
            },
        },
        # 2. Impute Missing Income
        {
            'name': 'impute_income',
            'transformer': 'SimpleImputer',
            'params': {
                'columns': ['income'],
                'strategy': 'mean',
            },
        },
        # 3. Encode City (Categorical)
        {
            'name': 'encode_city',
            'transformer': 'OneHotEncoder',
            'params': {'columns': ['city']},
        },
        # 4. Scale Numeric Features
        {
            'name': 'scale_features',
            'transformer': 'StandardScaler',
            'params': {'columns': ['age', 'income']},
        },
    ],
    'modeling': {
        'type': 'random_forest_classifier',
        'params': {'n_estimators': 50, 'max_depth': 5},
    },
}
```

## 4. Run Pipeline

Initialize and fit the pipeline.

```python
pipeline = SkyulfPipeline(config)
metrics = pipeline.fit(data, target_column='is_customer')
print(metrics)
```

**Output:**
```text
{
    'preprocessing': {...},
    'modeling': {
        'accuracy': 0.85,
        'f1_score': 0.82,
        ...
    }
}
```

## 5. Save and Load

Pipelines can be saved to disk and reloaded for inference.

```python
import os

artifact_path = 'my_model.pkl'
pipeline.save(artifact_path)

# Load back
loaded = SkyulfPipeline.load(artifact_path)

# Predict on new data
new_data = pd.DataFrame({
    'age': [25, 40],
    'income': [60000, np.nan],
    'city': ['London', 'Paris'],
})

predictions = loaded.predict(new_data)
print(predictions)

# Cleanup
if os.path.exists(artifact_path):
    os.remove(artifact_path)
```

**Output:**
```text
0    0
1    1
dtype: int64
```
