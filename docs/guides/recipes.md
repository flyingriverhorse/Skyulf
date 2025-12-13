# Common Recipes

Quick reference for common ML preprocessing and modeling tasks in Skyulf.

---

## Data Cleaning

### Handle Missing Values

```python
steps = [
    # Fill numeric columns with mean
    {
        "name": "impute_numeric",
        "transformer": "SimpleImputer",
        "params": {"strategy": "mean", "columns": ["age", "income"]}
    },
    # Fill categorical columns with mode
    {
        "name": "impute_categorical",
        "transformer": "SimpleImputer",
        "params": {"strategy": "most_frequent", "columns": ["city", "gender"]}
    }
]
```

### Advanced Imputation (KNN)

```python
steps = [
    {
        "name": "knn_impute",
        "transformer": "KNNImputer",
        "params": {
            "n_neighbors": 5,
            "columns": ["feature1", "feature2"]
        }
    }
]
```

### Drop Rows/Columns with Missing Values

```python
steps = [
    # Drop columns with >50% missing
    {
        "name": "drop_cols",
        "transformer": "DropMissingColumns",
        "params": {"threshold": 0.5}
    },
    # Drop rows with any missing values
    {
        "name": "drop_rows",
        "transformer": "DropMissingRows",
        "params": {"how": "any"}  # or "all"
    }
]
```

### Remove Duplicates

```python
steps = [
    {
        "name": "dedup",
        "transformer": "Deduplicate",
        "params": {"subset": ["id", "timestamp"]}  # Optional: specific columns
    }
]
```

---

## Encoding Categorical Variables

### One-Hot Encoding

```python
steps = [
    {
        "name": "onehot",
        "transformer": "OneHotEncoder",
        "params": {
            "columns": ["category", "region"],
            "handle_unknown": "ignore",
            "drop_first": True  # Avoid multicollinearity
        }
    }
]
```

### Label Encoding (Ordinal)

```python
steps = [
    {
        "name": "ordinal",
        "transformer": "OrdinalEncoder",
        "params": {"columns": ["size"]}  # small=0, medium=1, large=2
    }
]
```

### Target Encoding

```python
steps = [
    {
        "name": "target_encode",
        "transformer": "TargetEncoder",
        "params": {
            "columns": ["high_cardinality_col"],
            "smoothing": 10.0
        }
    }
]
```

---

## Scaling Numeric Features

### Standard Scaling (Z-score)

```python
steps = [
    {
        "name": "standardize",
        "transformer": "StandardScaler",
        "params": {"columns": ["age", "income", "score"]}
    }
]
```

### Min-Max Scaling (0-1 range)

```python
steps = [
    {
        "name": "minmax",
        "transformer": "MinMaxScaler",
        "params": {
            "columns": ["feature1", "feature2"],
            "feature_range": (0, 1)
        }
    }
]
```

### Robust Scaling (Outlier-resistant)

```python
steps = [
    {
        "name": "robust",
        "transformer": "RobustScaler",
        "params": {"columns": ["feature_with_outliers"]}
    }
]
```

---

## Handling Outliers

### IQR Method

```python
steps = [
    {
        "name": "clip_outliers",
        "transformer": "IQR",
        "params": {
            "multiplier": 1.5,  # IQR multiplier for bounds
            "columns": ["price", "quantity"]
        }
    }
]
```

### Z-Score Method

```python
steps = [
    {
        "name": "zscore_filter",
        "transformer": "ZScore",
        "params": {
            "threshold": 3.0,
            "columns": ["outlier_prone_feature"]
        }
    }
]
```

### Manual Bounds

```python
steps = [
    {
        "name": "manual_clip",
        "transformer": "ManualBounds",
        "params": {
            "bounds": {
                "age": {"lower": 0, "upper": 120},
                "temperature": {"lower": -50, "upper": 60}
            }
        }
    }
]
```

---

## Feature Engineering

### Create Polynomial Features

```python
steps = [
    {
        "name": "poly",
        "transformer": "PolynomialFeatures",
        "params": {
            "degree": 2,
            "include_bias": False,
            "columns": ["x1", "x2"]
        }
    }
]
```

### Custom Feature Math

```python
steps = [
    {
        "name": "ratio_features",
        "transformer": "FeatureGeneration",
        "params": {
            "operations": [
                {
                    "type": "ratio",
                    "numerator": "price",
                    "denominator": "sqft",
                    "output_column": "price_per_sqft"
                },
                {
                    "type": "ratio",
                    "numerator": "age",
                    "denominator": "income",
                    "output_column": "age_income_ratio"
                }
            ]
        }
    }
]
```

### Binning/Bucketing

```python
steps = [
    {
        "name": "age_buckets",
        "transformer": "GeneralBinning",
        "params": {
            "columns": ["age"],
            "strategy": "quantile",  # or "uniform", "kmeans"
            "n_bins": 5
        }
    }
]
```

---

## Feature Selection

### Remove Low Variance Features

```python
steps = [
    {
        "name": "variance_filter",
        "transformer": "VarianceThreshold",
        "params": {"threshold": 0.01}
    }
]
```

### Remove Highly Correlated Features

```python
steps = [
    {
        "name": "correlation_filter",
        "transformer": "CorrelationThreshold",
        "params": {"threshold": 0.95}
    }
]
```

### Select K Best Features

```python
steps = [
    {
        "name": "select_best",
        "transformer": "UnivariateSelection",
        "params": {
            "k": 10,
            "score_func": "f_classif"  # or "f_regression", "chi2"
        }
    }
]
```

---

## Power Transformations

### Yeo-Johnson Transform (Handles negatives)

```python
steps = [
    {
        "name": "power_transform",
        "transformer": "PowerTransformer",
        "params": {
            "method": "yeo-johnson",
            "columns": ["skewed_feature"]
        }
    }
]
```

### Log Transform

```python
steps = [
    {
        "name": "log_transform",
        "transformer": "SimpleTransformation",
        "params": {
            "operation": "log1p",  # log(1 + x)
            "columns": ["highly_skewed"]
        }
    }
]
```

---

## Model Training

### Classification with Random Forest

```python
from skyulf.modeling.classification import RandomForestClassifierCalculator

config = {
    "n_estimators": 100,
    "max_depth": 10,
    "min_samples_split": 5,
    "random_state": 42
}
```

### Regression with Ridge

```python
from skyulf.modeling.regression import RidgeRegressionCalculator

config = {
    "alpha": 1.0,
    "random_state": 42
}
```

---

## Hyperparameter Tuning

### Grid Search

```python
tuning_config = {
    "strategy": "grid",
    "metric": "accuracy",
    "cv_folds": 5,
    "search_space": {
        "n_estimators": [50, 100, 200],
        "max_depth": [5, 10, 20]
    }
}
```

### Random Search

```python
tuning_config = {
    "strategy": "random",
    "metric": "f1",
    "cv_folds": 3,
    "n_trials": 20,
    "search_space": {
        "n_estimators": [50, 100, 200, 500],
        "max_depth": [5, 10, 15, 20, None],
        "min_samples_split": [2, 5, 10]
    }
}
```

### Optuna (Bayesian Optimization)

```python
tuning_config = {
    "strategy": "optuna",
    "metric": "roc_auc",
    "cv_folds": 5,
    "n_trials": 50,
    "search_space": {
        "n_estimators": {"type": "int", "low": 50, "high": 500},
        "max_depth": {"type": "int", "low": 3, "high": 20},
        "learning_rate": {"type": "float", "low": 0.01, "high": 0.3, "log": True}
    }
}
```

---

## Complete Pipeline Example

```python
from skyulf.preprocessing.pipeline import FeatureEngineer

# Full preprocessing pipeline
steps = [
    # 1. Clean
    {"name": "dedup", "transformer": "Deduplicate", "params": {}},
    {"name": "drop_missing", "transformer": "DropMissingColumns", "params": {"threshold": 0.7}},
    
    # 2. Impute
    {"name": "impute_num", "transformer": "SimpleImputer", "params": {"strategy": "median"}},
    
    # 3. Handle Outliers
    {"name": "clip", "transformer": "IQR", "params": {"multiplier": 1.5}},
    
    # 4. Encode
    {"name": "encode", "transformer": "OneHotEncoder", "params": {"columns": ["category"]}},
    
    # 5. Scale
    {"name": "scale", "transformer": "StandardScaler", "params": {}},
    
    # 6. Feature Selection
    {"name": "select", "transformer": "VarianceThreshold", "params": {"threshold": 0.01}}
]

engineer = FeatureEngineer(steps)
processed_df, metrics = engineer.fit_transform(train_df)
```
