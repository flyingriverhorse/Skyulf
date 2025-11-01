# Feature Engineering API Documentation

## Overview

The feature engineering module provides tools for encoding categorical variables, transforming features, selecting important features, and creating new derived features. These operations enhance model performance by extracting meaningful patterns from raw data.

---

## 📦 Module: `skyulf_mlflow_library.features`

### Submodules

- **encoding**: Convert categorical variables to numeric
- **transform**: Create new features through mathematical operations
- **selection**: Identify and select important features
- **scaling**: Normalize feature distributions
- **imputation**: Handle missing values in features

---

## 🏷️ Encoding Methods

### `OneHotEncoder`

Convert categorical variables into binary columns (dummy variables).

**Signature:**
```python
class OneHotEncoder(BaseEncoder):
    def __init__(
        self,
        columns: Optional[List[str]] = None,
        drop: Optional[str] = None,
        handle_unknown: str = 'ignore',
        sparse: bool = False
    )
```

**Parameters:**
- `columns` (list, optional): Specific columns to encode
- `drop` (str, optional): Drop one column per category ('first', 'if_binary', or None)
- `handle_unknown` (str, default='ignore'): How to handle unknown categories
- `sparse` (bool, default=False): Return sparse matrix

**Example:**
```python
from skyulf_mlflow_library.features.encoding import OneHotEncoder
import pandas as pd

df = pd.DataFrame({
    'color': ['red', 'blue', 'green', 'red', 'blue'],
    'size': ['S', 'M', 'L', 'M', 'S']
})

encoder = OneHotEncoder()
df_encoded = encoder.fit_transform(df)

print(df_encoded.columns)
# color_red, color_blue, color_green, size_S, size_M, size_L
```

**When to Use:**
- **Nominal categorical variables** (no inherent order): colors, countries, product types
- **Tree-based models**: Works well with decision trees, random forests
- **Small cardinality**: <10-20 unique categories per feature

**Advantages:**
- No ordinal assumption
- Works with all algorithms
- Easy to interpret

**Disadvantages:**
- High dimensionality with many categories
- Sparse features
- Memory intensive for large datasets

---

### `LabelEncoder`

Encode categorical variables as integers (0, 1, 2, ...).

**Signature:**
```python
class LabelEncoder(BaseEncoder):
    def __init__(
        self,
        columns: Optional[List[str]] = None
    )
```

**Example:**
```python
from skyulf_mlflow_library.features.encoding import LabelEncoder

df = pd.DataFrame({
    'grade': ['A', 'B', 'C', 'A', 'B', 'C', 'A']
})

encoder = LabelEncoder()
df_encoded = encoder.fit_transform(df)

print(df_encoded['grade'])
# 0, 1, 2, 0, 1, 2, 0 (A=0, B=1, C=2)
```

**When to Use:**
- **Ordinal categories** with natural order: grades (A, B, C), sizes (S, M, L, XL)
- **Target encoding** for classification labels
- **Tree-based models**: Can work without implying order

**⚠️ Warning:**
- Creates artificial ordering for nominal variables
- Linear models may interpret as continuous
- Use OneHotEncoder for non-ordinal categories

---

### `OrdinalEncoder`

Encode ordinal categories with specified order.

**Signature:**
```python
class OrdinalEncoder(BaseEncoder):
    def __init__(
        self,
        columns: Optional[List[str]] = None,
        categories: Optional[Dict[str, List]] = None,
        handle_unknown: str = 'error'
    )
```

**Parameters:**
- `columns` (list, optional): Columns to encode
- `categories` (dict, optional): Explicit ordering for each column
- `handle_unknown` (str, default='error'): How to handle unknown values

**Example:**
```python
from skyulf_mlflow_library.features.encoding import OrdinalEncoder

df = pd.DataFrame({
    'education': ['High School', 'Bachelor', 'Master', 'PhD', 'Bachelor'],
    'satisfaction': ['Low', 'Medium', 'High', 'Medium', 'Low']
})

# Define explicit ordering
categories = {
    'education': ['High School', 'Bachelor', 'Master', 'PhD'],
    'satisfaction': ['Low', 'Medium', 'High']
}

encoder = OrdinalEncoder(categories=categories)
df_encoded = encoder.fit_transform(df)

print(df_encoded)
# education: 0, 1, 2, 3, 1
# satisfaction: 0, 1, 2, 1, 0
```

**When to Use:**
- **Clear ordering**: education levels, income brackets, ratings
- **Meaningful distances**: Low < Medium < High
- **Linear relationships**: Higher education → higher salary

**Advantages:**
- Preserves ordinal information
- Single column (no dimensionality explosion)
- Works well with linear models

**Best Practices:**
- Always specify `categories` parameter
- Ensure order makes semantic sense
- Use consistent ordering across datasets

---

### `TargetEncoder`

Encode categories using target statistics (supervised encoding).

**Signature:**
```python
class TargetEncoder(BaseEncoder):
    def __init__(
        self,
        columns: Optional[List[str]] = None,
        smoothing: float = 1.0,
        min_samples_leaf: int = 1
    )
```

**Parameters:**
- `columns` (list, optional): Columns to encode
- `smoothing` (float, default=1.0): Regularization strength
- `min_samples_leaf` (int, default=1): Minimum samples for stable estimates

**Example:**
```python
from skyulf_mlflow_library.features.encoding import TargetEncoder

df = pd.DataFrame({
    'city': ['NYC', 'LA', 'NYC', 'SF', 'LA', 'NYC', 'SF'],
    'price': [500, 300, 550, 400, 350, 520, 420]
})

X = df[['city']]
y = df['price']

# Encode city with mean price
encoder = TargetEncoder(smoothing=1.0)
X_encoded = encoder.fit_transform(X, y)

print(X_encoded)
# NYC → ~523 (mean of 500, 550, 520)
# LA → ~325 (mean of 300, 350)
# SF → ~410 (mean of 400, 420)
```

**How It Works:**
1. Calculate mean target value for each category
2. Apply smoothing to prevent overfitting:
   ```
   encoded_value = (n * category_mean + smoothing * global_mean) / (n + smoothing)
   ```
3. Replace category with encoded value

**When to Use:**
- **High cardinality**: Many unique categories (cities, IDs, brands)
- **Strong predictive power**: Category correlates with target
- **Regression or classification targets**

**Advantages:**
- Handles high cardinality efficiently
- Single numeric column
- Captures target relationship
- Better than one-hot for 100+ categories

**Disadvantages:**
- Risk of overfitting (use cross-validation encoding)
- Requires target variable
- Can leak information if not careful

**Best Practices:**
```python
from sklearn.model_selection import KFold

# Cross-validated target encoding
kf = KFold(n_splits=5, shuffle=True, random_state=42)
encoded_train = []

for train_idx, val_idx in kf.split(X_train):
    encoder = TargetEncoder(smoothing=1.0)
    encoder.fit(X_train.iloc[train_idx], y_train.iloc[train_idx])
    encoded_val = encoder.transform(X_train.iloc[val_idx])
    encoded_train.append(encoded_val)

X_train_encoded = pd.concat(encoded_train)
```

---

### `HashEncoder`

Encode high-cardinality features using feature hashing.

**Signature:**
```python
class HashEncoder(BaseEncoder):
    def __init__(
        self,
        columns: Optional[List[str]] = None,
        n_components: int = 8
    )
```

**Parameters:**
- `columns` (list, optional): Columns to encode
- `n_components` (int, default=8): Number of hash buckets

**Example:**
```python
from skyulf_mlflow_library.features.encoding import HashEncoder

df = pd.DataFrame({
    'product_id': ['PROD_001', 'PROD_002', 'PROD_003', 'PROD_001', 'PROD_004']
})

encoder = HashEncoder(n_components=8)
df_encoded = encoder.fit_transform(df)

print(df_encoded.shape)
# (5, 8) - 8 hash buckets instead of 4 one-hot columns
```

**When to Use:**
- **Very high cardinality**: 1000+ unique values (user IDs, product codes)
- **Memory constraints**: Can't afford one-hot explosion
- **Streaming data**: New categories appear frequently

**How It Works:**
1. Apply hash function to category
2. Map to one of `n_components` buckets
3. Create binary/count features

**Advantages:**
- Fixed dimensionality regardless of cardinality
- No fitting required (stateless)
- Memory efficient
- Handles unseen categories automatically

**Disadvantages:**
- Hash collisions (different categories → same bucket)
- Not interpretable
- Information loss

**Choosing n_components:**
```python
# Rule of thumb: n_components = sqrt(n_unique_categories)
n_unique = df['category'].nunique()
n_components = int(n_unique ** 0.5)

encoder = HashEncoder(n_components=n_components)
```

---

## 🔧 Feature Transformation

### `FeatureMath`

Apply mathematical operations to create new features.

**Signature:**
```python
class FeatureMath(BaseTransformer):
    def __init__(
        self,
        operations: Optional[List[Dict]] = None
    )
```

**Parameters:**
- `operations` (list of dict): Operations to perform
  - `operation`: 'add', 'subtract', 'multiply', 'divide', 'ratio', 'log', 'sqrt', 'square', 'abs'
  - `columns`: Input columns
  - `output_name`: Name for new feature

**Example:**
```python
from skyulf_mlflow_library.features.transform import FeatureMath

df = pd.DataFrame({
    'length': [10, 20, 30],
    'width': [5, 10, 15],
    'height': [2, 4, 6]
})

# Define operations
operations = [
    {
        'operation': 'multiply',
        'columns': ['length', 'width'],
        'output_name': 'area'
    },
    {
        'operation': 'multiply',
        'columns': ['area', 'height'],  # Can reference new features
        'output_name': 'volume'
    },
    {
        'operation': 'ratio',
        'columns': ['length', 'width'],
        'output_name': 'aspect_ratio'
    }
]

transformer = FeatureMath(operations=operations)
df_transformed = transformer.fit_transform(df)

print(df_transformed)
# Original columns + area, volume, aspect_ratio
```

**Available Operations:**

**Binary Operations** (require 2 columns):
- `add`: `col1 + col2`
- `subtract`: `col1 - col2`
- `multiply`: `col1 * col2`
- `divide`: `col1 / col2`
- `ratio`: `col1 / col2` (alias for divide)

**Unary Operations** (require 1 column):
- `log`: `log(col)` (natural logarithm)
- `log10`: `log10(col)` (base-10 logarithm)
- `sqrt`: `√col`
- `square`: `col²`
- `abs`: `|col|`
- `inverse`: `1/col`

**When to Use:**
- **Domain knowledge**: Physics formulas, business metrics
- **Feature interactions**: Multiplicative effects
- **Non-linear relationships**: Log transforms for exponential growth

**Real-World Examples:**
```python
# E-commerce
operations = [
    {'operation': 'multiply', 'columns': ['quantity', 'price'], 'output_name': 'revenue'},
    {'operation': 'ratio', 'columns': ['revenue', 'cost'], 'output_name': 'profit_margin'}
]

# Real estate
operations = [
    {'operation': 'multiply', 'columns': ['length', 'width'], 'output_name': 'floor_area'},
    {'operation': 'ratio', 'columns': ['price', 'floor_area'], 'output_name': 'price_per_sqft'}
]

# Finance
operations = [
    {'operation': 'log', 'columns': ['income'], 'output_name': 'log_income'},
    {'operation': 'ratio', 'columns': ['debt', 'income'], 'output_name': 'debt_to_income'}
]
```

---

### `PolynomialFeatures`

Automatically generate polynomial and interaction features.

**Signature:**
```python
class PolynomialFeatures(BaseTransformer):
    def __init__(
        self,
        degree: int = 2,
        interaction_only: bool = False,
        include_bias: bool = False,
        columns: Optional[List[str]] = None
    )
```

**Parameters:**
- `degree` (int, default=2): Maximum polynomial degree
- `interaction_only` (bool, default=False): Only interactions, no powers
- `include_bias` (bool, default=False): Include bias column (all 1s)
- `columns` (list, optional): Columns to use

**Example:**
```python
from skyulf_mlflow_library.features.transform.polynomial import PolynomialFeatures

df = pd.DataFrame({
    'x1': [1, 2, 3],
    'x2': [4, 5, 6]
})

# Full polynomial (degree=2)
poly = PolynomialFeatures(degree=2, include_bias=False)
df_poly = poly.fit_transform(df)

print(df_poly.columns)
# x1, x2, x1², x1*x2, x2²

# Interaction only (no powers)
poly = PolynomialFeatures(degree=2, interaction_only=True)
df_poly = poly.fit_transform(df)

print(df_poly.columns)
# x1, x2, x1*x2 (no x1², x2²)
```

**Generated Features:**

**Degree 2:**
- Original: `x1, x2`
- Powers: `x1², x2²`
- Interactions: `x1*x2`

**Degree 3:**
- Degree 2 features +
- Cubic: `x1³, x2³`
- Interactions: `x1²*x2, x1*x2², x1*x2*x3`

**When to Use:**
- **Non-linear relationships**: Curves, parabolas
- **Interaction effects**: Combined influence of features
- **Linear models**: Add complexity to simple models
- **Small feature sets**: 2-10 features

**⚠️ Warning:**
- **Feature explosion**: n features → n²/2 interactions
  - 10 features → 55 features (degree=2)
  - 20 features → 210 features (degree=2)
- **Multicollinearity**: Correlated features
- **Overfitting risk**: Too many features

**Best Practices:**
```python
# 1. Start with degree=2
poly = PolynomialFeatures(degree=2, interaction_only=True)

# 2. Use feature selection after
from sklearn.feature_selection import SelectKBest, f_regression
selector = SelectKBest(f_regression, k=20)
X_selected = selector.fit_transform(X_poly, y)

# 3. Use regularization (Ridge, Lasso)
from sklearn.linear_model import Ridge
model = Ridge(alpha=1.0)
model.fit(X_poly, y)

# 4. Or limit to specific columns
poly = PolynomialFeatures(degree=2, columns=['x1', 'x2', 'x3'])
```

---

### `InteractionFeatures`

Create specific pairwise interactions between features.

**Signature:**
```python
class InteractionFeatures(BaseTransformer):
    def __init__(
        self,
        interactions: List[Tuple[str, str]],
        operation: str = 'multiply'
    )
```

**Parameters:**
- `interactions` (list of tuples): Feature pairs to interact
- `operation` (str, default='multiply'): Operation ('multiply', 'add', 'subtract', 'divide')

**Example:**
```python
from skyulf_mlflow_library.features.transform.polynomial import InteractionFeatures

df = pd.DataFrame({
    'price': [100, 200, 300],
    'quantity': [10, 5, 20],
    'discount': [0.1, 0.2, 0.15]
})

# Create specific interactions
interactions = [
    ('price', 'quantity'),      # revenue
    ('price', 'discount'),      # discount_amount
]

transformer = InteractionFeatures(interactions=interactions, operation='multiply')
df_transformed = transformer.fit_transform(df)

print(df_transformed.columns)
# price, quantity, discount, price*quantity, price*discount
```

**When to Use:**
- **Domain knowledge**: Known important interactions
- **Avoid feature explosion**: Select specific pairs
- **Interpretability**: Meaningful combinations

**vs PolynomialFeatures:**
- `InteractionFeatures`: Manual control, specific pairs
- `PolynomialFeatures`: Automatic, all combinations

---

### `SmartBinning`

Discretize continuous features into bins.

**Signature:**
```python
class SmartBinning(BaseTransformer):
    def __init__(
        self,
        columns: Optional[List[str]] = None,
        n_bins: int = 5,
        strategy: str = 'quantile',
        encode: str = 'ordinal'
    )
```

**Parameters:**
- `columns` (list, optional): Columns to bin
- `n_bins` (int, default=5): Number of bins
- `strategy` (str, default='quantile'): Binning strategy
  - `'uniform'`: Equal-width bins
  - `'quantile'`: Equal-frequency bins
  - `'kmeans'`: K-means clustering
- `encode` (str, default='ordinal'): Output encoding ('ordinal' or 'onehot')

**Example:**
```python
from skyulf_mlflow_library.features.transform import SmartBinning

df = pd.DataFrame({
    'age': [18, 25, 35, 45, 55, 65, 75, 85],
    'income': [20000, 35000, 50000, 75000, 90000, 110000, 85000, 60000]
})

# Quantile binning (equal samples per bin)
binner = SmartBinning(n_bins=4, strategy='quantile', encode='ordinal')
df_binned = binner.fit_transform(df)

print(df_binned)
# age: 0, 0, 1, 2, 2, 3, 3, 3
# Each bin has ~2 samples
```

**Strategies:**

**Uniform (Equal Width):**
```python
binner = SmartBinning(strategy='uniform', n_bins=5)
# [0-20], [20-40], [40-60], [60-80], [80-100]
```
- Use when: Data is uniformly distributed
- Advantage: Consistent bin widths
- Disadvantage: Unequal sample counts

**Quantile (Equal Frequency):**
```python
binner = SmartBinning(strategy='quantile', n_bins=5)
# Each bin has 20% of data
```
- Use when: Skewed distributions
- Advantage: Balanced bins
- Disadvantage: Variable bin widths

**K-Means:**
```python
binner = SmartBinning(strategy='kmeans', n_bins=5)
# Bins based on clustering
```
- Use when: Natural groupings exist
- Advantage: Data-driven boundaries
- Disadvantage: More complex

**When to Use Binning:**
- **Capture non-linearity**: Linear models with non-linear features
- **Reduce noise**: Smooth out measurement errors
- **Interpretability**: "age groups" vs exact age
- **Outlier handling**: Cap extreme values

**Example Use Cases:**
```python
# Age groups
age_binner = SmartBinning(
    columns=['age'],
    n_bins=5,
    strategy='uniform',
    encode='ordinal'
)
# 0-20, 21-40, 41-60, 61-80, 81+

# Credit score ranges
score_binner = SmartBinning(
    columns=['credit_score'],
    n_bins=5,
    strategy='quantile',  # Equal distribution
    encode='onehot'
)
# Poor, Fair, Good, Very Good, Excellent
```

---

## 🎯 Feature Selection

### `FeatureSelector`

Unified interface that wraps multiple scikit-learn selection strategies while providing
rich metadata (selected columns, support mask, scores, notes) and optional column dropping.

**Signature:**
```python
class FeatureSelector(BaseTransformer):
    def __init__(
        self,
        columns: Optional[List[str]] = None,
        target_column: Optional[str] = None,
        method: str = "select_k_best",
        score_func: Optional[str] = None,
        k: Optional[int] = None,
        percentile: Optional[float] = None,
        alpha: Optional[float] = None,
        threshold: Optional[float] = None,
        estimator: Optional[str] = None,
        drop_unselected: bool = True,
        auto_detect: bool = True,
        **kwargs,
    )
```

**Supported methods:**

| Method name           | Description                                   | Notes |
|-----------------------|-----------------------------------------------|-------|
| `select_k_best`       | Top-K by score function                       | `score_func` = `f_classif`, `f_regression`, etc. |
| `select_percentile`   | Keep top percentile                           | |
| `generic_univariate_select` | Flexible mode (`k_best`, `percentile`, `fpr`, `fdr`, `fwe`) | |
| `select_fpr` / `select_fdr` / `select_fwe` | Statistical thresholding models | |
| `select_from_model`   | Model-based importance (Lasso, RF, etc.)      | `estimator` = `auto`, `logistic_regression`, `linear_regression`, `random_forest` |
| `variance_threshold`  | Drop near-constant features                   | `threshold` controls minimum variance |
| `rfe`                 | Recursive Feature Elimination                 | Requires estimator |

**Example (classification):**
```python
import pandas as pd
from skyulf_mlflow_library.features.selection import FeatureSelector

df = pd.DataFrame({...})  # features + 'target' column

selector = FeatureSelector(
    columns=["feat_1", "feat_2", "feat_3", "feat_4"],
    target_column="target",
    method="select_k_best",
    score_func="f_classif",
    k=2,
    drop_unselected=True,
    auto_detect=False,
)

df_selected = selector.fit_transform(df, df["target"])

print("Selected columns:", selector.get_selected_columns())
print("Dropped columns:", selector.get_dropped_columns())
print("Metadata summary:", selector.metadata["summary"])
```

**Metadata highlights:**
- `selected_columns` / `dropped_columns`
- `support_mask` (boolean list aligned with candidate columns)
- `feature_summaries` (score, p-value, rank, importance, notes)
- `method_label` and `score_func` for audit logging

**When to Use:**
- After polynomial/interaction features to keep the strongest signals
- Ahead of resource-intensive models to cut dimensionality
- To compare statistical vs model-based selection quickly

**Best Practices:**
- Provide `columns` explicitly when your frame contains targets or identifiers
- Enable `drop_unselected=True` to immediately prune columns downstream
- Use `auto_detect=True` to automatically pick numeric predictors (excluding the target)
- Combine with splitter-aware pipelines to re-use fitted selectors on validation/test splits

---

## 🔄 Complete Feature Engineering Pipeline

```python
from skyulf_mlflow_library.features.encoding import OneHotEncoder, TargetEncoder
from skyulf_mlflow_library.features.transform import FeatureMath, SmartBinning
from skyulf_mlflow_library.features.transform.polynomial import PolynomialFeatures
from skyulf_mlflow_library.features.selection import FeatureSelector
from skyulf_mlflow_library.preprocessing import StandardScaler

# 1. Encode categorical variables
# Low cardinality: One-hot
onehot = OneHotEncoder(columns=['color', 'size'])
df = onehot.fit_transform(df)

# High cardinality: Target encoding
target_enc = TargetEncoder(columns=['city', 'product_id'])
df = target_enc.fit_transform(df[['city', 'product_id']], y)

# 2. Create domain-specific features
operations = [
    {'operation': 'multiply', 'columns': ['price', 'quantity'], 'output_name': 'revenue'},
    {'operation': 'ratio', 'columns': ['revenue', 'cost'], 'output_name': 'margin'}
]
math_transformer = FeatureMath(operations=operations)
df = math_transformer.fit_transform(df)

# 3. Bin continuous features
binner = SmartBinning(columns=['age', 'income'], n_bins=5, strategy='quantile')
df = binner.fit_transform(df)

# 4. Create polynomial features (selected columns only)
poly = PolynomialFeatures(degree=2, interaction_only=True, columns=['feature1', 'feature2'])
df = poly.fit_transform(df)

# 5. Scale features (excluding the target)
feature_columns = [col for col in df.columns if col != 'target']
scaler = StandardScaler(columns=feature_columns)
df_scaled = scaler.fit_transform(df)

# 6. Select best features
selector = FeatureSelector(
    columns=feature_columns,
    target_column='target',
    method='select_k_best',
    score_func='f_regression',
    k=50,
    drop_unselected=True,
    auto_detect=False,
)
df_final = selector.fit_transform(df_scaled, y)

# Ready for modeling!
```

---

## 🎯 Best Practices

### Encoding
1. **Match cardinality to method:**
   - Low (<10): OneHotEncoder
   - Medium (10-100): OrdinalEncoder or TargetEncoder
   - High (>100): HashEncoder or TargetEncoder
   
2. **Preserve meaning:**
   - Ordinal: Use OrdinalEncoder with explicit order
   - Nominal: Use OneHotEncoder or HashEncoder

3. **Prevent leakage:**
   - Fit encoders on training data only
   - Use cross-validation for TargetEncoder

### Transformation
1. **Domain knowledge first:** Manual features before automatic
2. **Start simple:** FeatureMath before PolynomialFeatures
3. **Control complexity:** Use interaction_only=True for polynomials
4. **Select features:** Don't keep all generated features

### Workflow
```
1. Encode categorical → numeric
2. Create domain features (FeatureMath)
3. Bin if needed (SmartBinning)
4. Generate interactions (PolynomialFeatures)
5. Scale features
6. Select features
7. Train model
```

---

## 📚 See Also

- [Preprocessing Guide](preprocessing.md)
- [Model Training Guide](model_training.md)
- [Encoding Tutorial](../tutorials/encoding.md)
- [Feature Selection Guide](../guides/feature_selection.md)
