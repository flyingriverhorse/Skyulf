# Preprocessing API Documentation

## Overview

The preprocessing module provides tools for data cleaning, imputation, scaling, and sampling. These operations prepare your raw data for machine learning by handling missing values, normalizing features, and balancing datasets.

---

## 📦 Module: `skyulf_mlflow_library.preprocessing`

### Submodules

- **cleaning**: Remove missing values, duplicates, and outliers
- **imputation**: Fill missing values with various strategies
- **scaling**: Normalize and standardize features
- **sampling**: Handle imbalanced datasets

---

## 🧹 Cleaning Functions

### `drop_missing_rows()`

Remove rows containing missing values.

**Signature:**
```python
def drop_missing_rows(
    df: pd.DataFrame,
    columns: Optional[List[str]] = None,
    threshold: Optional[float] = None
) -> pd.DataFrame
```

**Parameters:**
- `df` (DataFrame): Input data
- `columns` (list, optional): Specific columns to check. If None, checks all columns
- `threshold` (float, optional): Minimum fraction of non-missing values required (0.0-1.0)

**Returns:**
- DataFrame with rows containing missing values removed

**Example:**
```python
from skyulf_mlflow_library.preprocessing import drop_missing_rows
import pandas as pd
import numpy as np

# Create sample data with missing values
df = pd.DataFrame({
    'age': [25, np.nan, 30, 35],
    'income': [50000, 60000, np.nan, 70000],
    'score': [85, 90, 88, np.nan]
})

# Drop any row with missing values
clean_df = drop_missing_rows(df)
print(clean_df)
# Output: Only the first row remains

# Drop rows with missing values in specific columns
clean_df = drop_missing_rows(df, columns=['age', 'income'])
print(clean_df)
# Output: Rows 0, 1, 3 remain (score can be missing)
```

**When to Use:**
- Small datasets where losing rows is acceptable
- When missing values indicate incomplete records
- As a quick cleaning step for exploratory analysis

---

### `drop_missing_columns()`

Remove columns with excessive missing values.

**Signature:**
```python
def drop_missing_columns(
    df: pd.DataFrame,
    threshold: float = 0.5
) -> pd.DataFrame
```

**Parameters:**
- `df` (DataFrame): Input data
- `threshold` (float, default=0.5): Maximum fraction of missing values allowed (0.0-1.0)

**Returns:**
- DataFrame with high-missing columns removed

**Example:**
```python
from skyulf_mlflow_library.preprocessing import drop_missing_columns

df = pd.DataFrame({
    'a': [1, 2, 3, 4],
    'b': [np.nan, np.nan, np.nan, 4],  # 75% missing
    'c': [1, np.nan, 3, 4]  # 25% missing
})

# Drop columns with >50% missing
clean_df = drop_missing_columns(df, threshold=0.5)
print(clean_df.columns)
# Output: ['a', 'c'] (column 'b' was dropped)
```

**When to Use:**
- Early data exploration to identify useless features
- When a feature has too few valid values to be useful
- Before imputation to focus on salvageable columns

---

### `remove_duplicates()`

Remove duplicate rows from DataFrame.

**Signature:**
```python
def remove_duplicates(
    df: pd.DataFrame,
    columns: Optional[List[str]] = None,
    keep: str = 'first'
) -> pd.DataFrame
```

**Parameters:**
- `df` (DataFrame): Input data
- `columns` (list, optional): Columns to consider for duplicate detection
- `keep` (str, default='first'): Which duplicate to keep ('first', 'last', or False)

**Returns:**
- DataFrame with duplicates removed

**Example:**
```python
from skyulf_mlflow_library.preprocessing import remove_duplicates

df = pd.DataFrame({
    'id': [1, 2, 2, 3],
    'name': ['Alice', 'Bob', 'Bob', 'Charlie'],
    'value': [100, 200, 200, 300]
})

# Remove exact duplicate rows
clean_df = remove_duplicates(df)
print(clean_df)

# Remove duplicates based on specific columns
clean_df = remove_duplicates(df, columns=['id', 'name'], keep='last')
print(clean_df)
```

**When to Use:**
- Data quality checks
- After merging datasets
- When records are accidentally repeated

---

### `remove_outliers()`

Remove outliers using IQR or Z-score methods.

**Signature:**
```python
def remove_outliers(
    df: pd.DataFrame,
    columns: Optional[List[str]] = None,
    method: str = 'iqr',
    threshold: float = 1.5
) -> pd.DataFrame
```

**Parameters:**
- `df` (DataFrame): Input data
- `columns` (list, optional): Columns to check for outliers
- `method` (str, default='iqr'): Method to use ('iqr' or 'zscore')
- `threshold` (float): IQR multiplier (1.5) or Z-score threshold (3.0)

**Returns:**
- DataFrame with outlier rows removed

**Example:**
```python
from skyulf_mlflow_library.preprocessing import remove_outliers

df = pd.DataFrame({
    'price': [100, 110, 105, 1000, 115, 120],  # 1000 is outlier
    'quantity': [10, 12, 11, 15, 13, 14]
})

# Remove outliers using IQR method
clean_df = remove_outliers(df, columns=['price'], method='iqr', threshold=1.5)
print(clean_df)

# Remove outliers using Z-score method
clean_df = remove_outliers(df, columns=['price'], method='zscore', threshold=3.0)
print(clean_df)
```

**When to Use:**
- Detecting data entry errors
- Removing extreme values that could skew models
- Data quality validation

---

## 🔢 Imputation Classes

### `SimpleImputer`

Fill missing values with statistical measures.

**Signature:**
```python
class SimpleImputer(BaseImputer):
    def __init__(
        self,
        strategy: str = 'mean',
        fill_value: Optional[Any] = None,
        columns: Optional[List[str]] = None
    )
```

**Parameters:**
- `strategy` (str, default='mean'): Imputation strategy
  - `'mean'`: Replace with column mean (numeric only)
  - `'median'`: Replace with column median (numeric only)
  - `'most_frequent'`: Replace with mode (works for all types)
  - `'constant'`: Replace with fixed value
- `fill_value` (any, optional): Value to use when strategy='constant'
- `columns` (list, optional): Specific columns to impute

**Methods:**
- `fit(X, y=None)`: Calculate imputation values from training data
- `transform(X)`: Apply imputation to data
- `fit_transform(X, y=None)`: Fit and transform in one step

**Example:**
```python
from skyulf_mlflow_library.preprocessing import SimpleImputer
import pandas as pd
import numpy as np

# Create data with missing values
df = pd.DataFrame({
    'age': [25, np.nan, 30, 35, np.nan],
    'income': [50000, 60000, np.nan, 70000, 55000],
    'category': ['A', 'B', np.nan, 'A', 'B']
})

# Mean imputation for numeric columns
imputer = SimpleImputer(strategy='mean')
df_imputed = imputer.fit_transform(df[['age', 'income']])
print(df_imputed)

# Most frequent imputation for categorical
imputer = SimpleImputer(strategy='most_frequent')
df['category'] = imputer.fit_transform(df[['category']])

# Constant imputation
imputer = SimpleImputer(strategy='constant', fill_value=0)
df_zero = imputer.fit_transform(df[['age']])
```

**When to Use:**
- **Mean/Median**: Numeric features with random missing patterns
- **Most Frequent**: Categorical features
- **Constant**: When missing has specific meaning (e.g., 0 for counts)

---

### `KNNImputer`

Impute missing values using K-Nearest Neighbors.

**Signature:**
```python
class KNNImputer(BaseImputer):
    def __init__(
        self,
        n_neighbors: int = 5,
        weights: str = 'uniform',
        metric: str = 'nan_euclidean',
        columns: Optional[List[str]] = None
    )
```

**Parameters:**
- `n_neighbors` (int, default=5): Number of neighbors to use
- `weights` (str, default='uniform'): Weight function ('uniform' or 'distance')
- `metric` (str, default='nan_euclidean'): Distance metric
- `columns` (list, optional): Columns to impute

**Example:**
```python
from skyulf_mlflow_library.preprocessing import KNNImputer

# KNN imputation preserves relationships between features
imputer = KNNImputer(n_neighbors=3, weights='distance')
df_imputed = imputer.fit_transform(df)

# Similar samples influence the imputed value
# Better for structured patterns in missing data
```

**When to Use:**
- Missing values depend on other features
- Dataset has clear patterns/clusters
- More accurate than simple statistics

**Advantages:**
- Preserves feature relationships
- Better for non-random missing patterns
- No assumption about distribution

**Disadvantages:**
- Slower than SimpleImputer
- Requires sufficient complete samples
- Sensitive to feature scaling

---

## 📏 Scaling Classes

### `StandardScaler`

Standardize features by removing mean and scaling to unit variance.

**Formula:** `z = (x - μ) / σ`

**Signature:**
```python
class StandardScaler(BaseScaler):
    def __init__(
        self,
        with_mean: bool = True,
        with_std: bool = True,
        columns: Optional[List[str]] = None
    )
```

**Parameters:**
- `with_mean` (bool, default=True): Center data to mean 0
- `with_std` (bool, default=True): Scale to standard deviation 1
- `columns` (list, optional): Columns to scale

**Example:**
```python
from skyulf_mlflow_library.preprocessing import StandardScaler

df = pd.DataFrame({
    'age': [25, 30, 35, 40],
    'income': [50000, 60000, 70000, 80000]
})

scaler = StandardScaler()
df_scaled = scaler.fit_transform(df)

print(df_scaled.mean())  # ~0
print(df_scaled.std())   # ~1
```

**When to Use:**
- Features have different units/scales
- Algorithms sensitive to scale (SVM, KNN, Neural Networks)
- When features are normally distributed
- Default choice for most ML tasks

**Properties:**
- Output range: Typically [-3, 3] for normal distributions
- Preserves outlier information
- Assumes normal distribution

---

### `MinMaxScaler`

Scale features to a fixed range [0, 1] or [min, max].

**Formula:** `x_scaled = (x - x_min) / (x_max - x_min)`

**Signature:**
```python
class MinMaxScaler(BaseScaler):
    def __init__(
        self,
        feature_range: Tuple[float, float] = (0, 1),
        columns: Optional[List[str]] = None
    )
```

**Parameters:**
- `feature_range` (tuple, default=(0,1)): Desired output range
- `columns` (list, optional): Columns to scale

**Example:**
```python
from skyulf_mlflow_library.preprocessing import MinMaxScaler

# Scale to [0, 1]
scaler = MinMaxScaler()
df_scaled = scaler.fit_transform(df)

# Scale to custom range [-1, 1]
scaler = MinMaxScaler(feature_range=(-1, 1))
df_scaled = scaler.fit_transform(df)
```

**When to Use:**
- Need bounded values (e.g., neural network inputs)
- Features already in specific range
- Image processing (pixel values 0-255 → 0-1)
- When distribution is not Gaussian

**Properties:**
- Output range: Specified by feature_range
- Sensitive to outliers
- Preserves zero values when range starts at 0

---

### `RobustScaler`

Scale features using median and IQR (robust to outliers).

**Formula:** `x_scaled = (x - median) / IQR`

**Signature:**
```python
class RobustScaler(BaseScaler):
    def __init__(
        self,
        with_centering: bool = True,
        with_scaling: bool = True,
        quantile_range: Tuple[float, float] = (25.0, 75.0),
        columns: Optional[List[str]] = None
    )
```

**Parameters:**
- `with_centering` (bool, default=True): Center to median
- `with_scaling` (bool, default=True): Scale by IQR
- `quantile_range` (tuple, default=(25,75)): Quantiles for IQR
- `columns` (list, optional): Columns to scale

**Example:**
```python
from skyulf_mlflow_library.preprocessing import RobustScaler

# Data with outliers
df = pd.DataFrame({
    'price': [100, 110, 105, 1000, 115, 120]  # 1000 is outlier
})

scaler = RobustScaler()
df_scaled = scaler.fit_transform(df)
# Outlier won't dominate the scaling
```

**When to Use:**
- Data contains outliers
- Need robust statistics
- After outlier detection shows issues

**Advantages:**
- Outlier-resistant
- Better than StandardScaler for skewed data
- Preserves shape of distribution

---

### `MaxAbsScaler`

Scale by maximum absolute value to range [-1, 1].

**Formula:** `x_scaled = x / max(|x|)`

**Signature:**
```python
class MaxAbsScaler(BaseScaler):
    def __init__(
        self,
        columns: Optional[List[str]] = None
    )
```

**Example:**
```python
from skyulf_mlflow_library.preprocessing import MaxAbsScaler

scaler = MaxAbsScaler()
df_scaled = scaler.fit_transform(df)
# All values in [-1, 1]
```

**When to Use:**
- Sparse data (preserves zeros)
- Features already centered around zero
- When you need [-1, 1] range

**Properties:**
- Preserves sparsity
- Doesn't shift/center data
- Good for sparse matrices

---

## ⚖️ Sampling Methods

### `SMOTE`

Synthetic Minority Over-sampling Technique for imbalanced classification.

**Signature:**
```python
class SMOTE:
    def __init__(
        self,
        sampling_strategy: Union[str, float, dict] = 'auto',
        k_neighbors: int = 5,
        random_state: Optional[int] = None
    )
```

**Parameters:**
- `sampling_strategy`: How to resample
  - `'auto'`: Resample all classes to match majority
  - `float`: Ratio of minority to majority after resampling
  - `dict`: Specific count for each class
- `k_neighbors` (int, default=5): Number of nearest neighbors for synthesis
- `random_state` (int, optional): Random seed

**Methods:**
- `fit_resample(X, y)`: Generate synthetic samples

**Example:**
```python
from skyulf_mlflow_library.preprocessing import SMOTE
import pandas as pd

# Imbalanced dataset
X = pd.DataFrame({
    'feature1': [1, 2, 3, 4, 5, 6, 7, 8],
    'feature2': [10, 20, 30, 40, 50, 60, 70, 80]
})
y = [0, 0, 0, 0, 0, 0, 1, 1]  # 6:2 imbalance

print(f"Original: {pd.Series(y).value_counts()}")
# 0: 6, 1: 2

# Apply SMOTE
smote = SMOTE(sampling_strategy='auto', random_state=42)
X_resampled, y_resampled = smote.fit_resample(X, y)

print(f"After SMOTE: {pd.Series(y_resampled).value_counts()}")
# 0: 6, 1: 6 (synthetic samples created)
```

**When to Use:**
- Binary or multiclass classification
- Minority class has too few samples
- Better than simple duplication
- When you need to augment training data

**How it Works:**
1. For each minority sample
2. Find k nearest neighbors from same class
3. Create synthetic samples between sample and neighbors
4. Add noise for diversity

**Advantages:**
- More effective than random over-sampling
- Reduces overfitting risk
- Works well for most classifiers

**Disadvantages:**
- Can create noise near class boundaries
- Requires enough minority samples (>k_neighbors)
- Doesn't work for regression

---

### `RandomOverSampler`

Randomly duplicate minority class samples.

**Signature:**
```python
class RandomOverSampler:
    def __init__(
        self,
        sampling_strategy: Union[str, float, dict] = 'auto',
        random_state: Optional[int] = None
    )
```

**Example:**
```python
from skyulf_mlflow_library.preprocessing import RandomOverSampler

oversampler = RandomOverSampler(sampling_strategy='auto', random_state=42)
X_resampled, y_resampled = oversampler.fit_resample(X, y)
# Minority samples are duplicated
```

**When to Use:**
- Quick baseline for imbalanced data
- When SMOTE is too slow
- Small datasets where duplication is acceptable

**Advantages:**
- Simple and fast
- No data assumptions
- Works with any classifier

**Disadvantages:**
- No new information added
- Can lead to overfitting
- Exact duplicates in training set

---

### `RandomUnderSampler`

Randomly remove majority class samples.

**Signature:**
```python
class RandomUnderSampler:
    def __init__(
        self,
        sampling_strategy: Union[str, float, dict] = 'auto',
        random_state: Optional[int] = None,
        replacement: bool = False
    )
```

**Example:**
```python
from skyulf_mlflow_library.preprocessing import RandomUnderSampler

undersampler = RandomUnderSampler(sampling_strategy='auto', random_state=42)
X_resampled, y_resampled = undersampler.fit_resample(X, y)
# Majority samples are removed
```

**When to Use:**
- Large datasets where losing data is acceptable
- When training is too slow with full dataset
- Extreme imbalance (1:100 or worse)

**Advantages:**
- Fast training
- Reduces computational cost
- Simple to implement

**Disadvantages:**
- Loses information
- Can underfit
- May remove important samples

---

## 🔄 Complete Preprocessing Pipeline

```python
from skyulf_mlflow_library.preprocessing import (
    SimpleImputer,
    StandardScaler,
    SMOTE,
    remove_duplicates
)
from skyulf_mlflow_library.utils import train_test_split

# 1. Load and clean data
df = pd.read_csv('data.csv')
df = remove_duplicates(df)

# 2. Split features and target
X = df.drop('target', axis=1)
y = df['target']

# 3. Train-test split
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

# 4. Impute missing values
imputer = SimpleImputer(strategy='mean')
X_train = imputer.fit_transform(X_train)
X_test = imputer.transform(X_test)

# 5. Scale features
scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)

# 6. Handle imbalanced data (training only)
smote = SMOTE(random_state=42)
X_train, y_train = smote.fit_resample(X_train, y_train)

# Ready for model training!
```

---

## 🎯 Best Practices

### Imputation
1. **Fit on training data only**: Prevent data leakage
2. **Choose strategy based on data type**: Mean for numeric, most_frequent for categorical
3. **Consider KNN for structured missing patterns**

### Scaling
1. **Scale after splitting**: Fit scaler on training data
2. **Match algorithm needs**: 
   - Tree-based models: No scaling needed
   - Distance-based (KNN, SVM): StandardScaler
   - Neural networks: MinMaxScaler or StandardScaler
3. **Use RobustScaler for outliers**

### Sampling
1. **Apply to training data only**: Never resample test data
2. **After splitting, before training**
3. **Try SMOTE before random over-sampling**
4. **Consider class weights as alternative**

### Order Matters
```
1. Remove duplicates (once, at start)
2. Train-test split
3. Imputation (fit on train)
4. Scaling (fit on train)
5. Sampling (train only)
6. Model training
```

---

## ⚠️ Common Pitfalls

**❌ Fitting on full dataset**
```python
# WRONG
scaler.fit(df)  # Includes test data!
X_train_scaled = scaler.transform(X_train)
```

**✅ Fitting on training data only**
```python
# CORRECT
scaler.fit(X_train)
X_train_scaled = scaler.transform(X_train)
X_test_scaled = scaler.transform(X_test)
```

**❌ Resampling test data**
```python
# WRONG
smote.fit_resample(X_test, y_test)  # Never resample test!
```

**✅ Resampling training data only**
```python
# CORRECT
X_train, y_train = smote.fit_resample(X_train, y_train)
# Test data remains unchanged
```

---

## 📚 See Also

- [Feature Engineering Guide](feature_engineering.md)
- [Model Training Guide](model_training.md)
- [Complete Pipeline Example](../guides/complete_pipeline.md)
