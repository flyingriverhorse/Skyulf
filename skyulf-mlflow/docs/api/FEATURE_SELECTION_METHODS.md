# Feature Selection Methods Reference

This document provides a comprehensive guide to the feature selection methods available in `skyulf_mlflow_library`.

## Overview

The `FeatureSelector` class supports **9 different selection methods**, each suited for different problem types and use cases. Feature selection helps reduce dimensionality, improve model performance, and prevent overfitting by identifying the most relevant features.

---

## Quick Reference Table

| Method | Classification | Regression | Description | Use Case |
|--------|---------------|------------|-------------|----------|
| `selectkbest` | ✅ | ✅ | Statistical tests (chi2, f_classif, f_regression, mutual_info) | Fast, univariate feature ranking |
| `rfe` | ✅ | ✅ | Recursive Feature Elimination with any estimator | Iterative backward feature removal |
| `from_model` | ✅ | ✅ | Use feature importances from tree-based models | Extract importance from trained models |
| `variance_threshold` | ✅ | ✅ | Remove low-variance features | Filter out constant/quasi-constant features |
| `mutual_info` | ✅ | ✅ | Mutual information criterion | Capture non-linear relationships |
| `chi2` | ✅ | ❌ | Chi-squared test for categorical targets | Classification with non-negative features |
| `f_classif` | ✅ | ❌ | ANOVA F-value for classification | Linear classification problems |
| `f_regression` | ❌ | ✅ | F-value for regression | Linear regression problems |
| `sequential` | ✅ | ✅ | Forward/backward sequential selection | Greedy search with custom scorer |

---

## Detailed Method Descriptions

### 1. SelectKBest (`selectkbest`)

**Supported Tasks**: Classification, Regression

**How it works**: Uses univariate statistical tests to select the k best features based on a scoring function.

**Parameters**:
- `k`: Number of top features to select (default: 10)
- `score_func`: Statistical test to use
  - `'chi2'`: Chi-squared test (classification, non-negative features)
  - `'f_classif'`: ANOVA F-value (classification)
  - `'f_regression'`: F-value (regression)
  - `'mutual_info_classif'`: Mutual information (classification)
  - `'mutual_info_regression'`: Mutual information (regression)

**Example**:
```python
from skyulf_mlflow_library.features.selection import FeatureSelector

selector = FeatureSelector(method='selectkbest', k=10, score_func='f_classif')
X_selected = selector.fit_transform(X, y)
selected_features = selector.get_selected_features()
```

**Best for**:
- Quick feature ranking
- High-dimensional datasets
- When you want to try multiple scoring functions

---

### 2. Recursive Feature Elimination (`rfe`)

**Supported Tasks**: Classification, Regression

**How it works**: Recursively removes features and builds a model on the remaining attributes until the desired number of features is reached.

**Parameters**:
- `n_features_to_select`: Number of features to keep (default: 10)
- `step`: Number of features to remove at each iteration (default: 1)
- `estimator`: Model to use for ranking (default: `RandomForestClassifier` or `RandomForestRegressor`)

**Example**:
```python
from sklearn.ensemble import RandomForestClassifier

selector = FeatureSelector(
    method='rfe',
    n_features_to_select=15,
    step=2,
    estimator=RandomForestClassifier(n_estimators=100, random_state=42)
)
X_selected = selector.fit_transform(X, y)
```

**Best for**:
- Finding optimal feature subset
- When computational resources allow
- Non-linear relationships

---

### 3. Model-Based Selection (`from_model`)

**Supported Tasks**: Classification, Regression

**How it works**: Uses feature importances from a trained model (e.g., Random Forest, XGBoost) to select features.

**Parameters**:
- `estimator`: Pre-trained model or model class
- `threshold`: Importance threshold ('mean', 'median', or numeric value)
- `prefit`: Whether the estimator is already fitted (default: False)

**Example**:
```python
from sklearn.ensemble import RandomForestClassifier

# Option 1: Fit during selection
selector = FeatureSelector(
    method='from_model',
    estimator=RandomForestClassifier(n_estimators=100, random_state=42),
    threshold='median'
)
X_selected = selector.fit_transform(X, y)

# Option 2: Use pre-trained model
rf = RandomForestClassifier(n_estimators=100, random_state=42)
rf.fit(X, y)
selector = FeatureSelector(method='from_model', estimator=rf, prefit=True)
X_selected = selector.transform(X)
```

**Best for**:
- Tree-based models (Random Forest, XGBoost, LightGBM)
- Leveraging existing model training
- Interpretable feature importance

---

### 4. Variance Threshold (`variance_threshold`)

**Supported Tasks**: Classification, Regression

**How it works**: Removes features with variance below a specified threshold. Features with zero or near-zero variance are uninformative.

**Parameters**:
- `threshold`: Minimum variance threshold (default: 0.0)

**Example**:
```python
# Remove features with less than 1% variance
selector = FeatureSelector(method='variance_threshold', threshold=0.01)
X_selected = selector.fit_transform(X, y)

# Remove constant features
selector = FeatureSelector(method='variance_threshold', threshold=0.0)
X_selected = selector.fit_transform(X, y)
```

**Best for**:
- Initial preprocessing step
- Removing constant or quasi-constant features
- High-dimensional sparse data

---

### 5. Mutual Information (`mutual_info`)

**Supported Tasks**: Classification, Regression

**How it works**: Measures mutual information between each feature and the target, capturing both linear and non-linear relationships.

**Parameters**:
- `k`: Number of features to select (default: 10)
- `discrete_features`: Boolean array indicating which features are discrete

**Example**:
```python
selector = FeatureSelector(method='mutual_info', k=20)
X_selected = selector.fit_transform(X, y)
```

**Best for**:
- Non-linear relationships
- Mixed feature types (continuous + categorical)
- Capturing complex dependencies

---

### 6. Chi-Squared Test (`chi2`)

**Supported Tasks**: Classification only

**How it works**: Computes chi-squared statistics between each non-negative feature and the target class.

**Parameters**:
- `k`: Number of features to select (default: 10)

**Requirements**:
- Features must be non-negative (e.g., counts, frequencies)
- Target must be categorical (classification)

**Example**:
```python
# For text classification with count/TF-IDF features
selector = FeatureSelector(method='chi2', k=100)
X_selected = selector.fit_transform(X_counts, y)
```

**Best for**:
- Text classification (with count/TF-IDF features)
- Count data
- Categorical features encoded as integers

---

### 7. F-Classification (`f_classif`)

**Supported Tasks**: Classification only

**How it works**: Computes ANOVA F-value between each feature and the target, measuring linear dependency.

**Parameters**:
- `k`: Number of features to select (default: 10)

**Example**:
```python
selector = FeatureSelector(method='f_classif', k=15)
X_selected = selector.fit_transform(X, y)
```

**Best for**:
- Linear classification problems
- Normally distributed features
- Fast univariate tests

---

### 8. F-Regression (`f_regression`)

**Supported Tasks**: Regression only

**How it works**: Computes F-value for regression problems, measuring linear dependency between features and continuous target.

**Parameters**:
- `k`: Number of features to select (default: 10)

**Example**:
```python
selector = FeatureSelector(method='f_regression', k=20)
X_selected = selector.fit_transform(X, y)
```

**Best for**:
- Linear regression problems
- Continuous target variables
- Fast feature ranking

---

### 9. Sequential Feature Selection (`sequential`)

**Supported Tasks**: Classification, Regression

**How it works**: Greedy search that adds (forward) or removes (backward) features one at a time based on cross-validated model performance.

**Parameters**:
- `n_features_to_select`: Number of features to select
- `direction`: `'forward'` or `'backward'`
- `scoring`: Scoring metric (e.g., 'accuracy', 'f1', 'r2')
- `cv`: Number of cross-validation folds
- `estimator`: Model to use for evaluation

**Example**:
```python
from sklearn.linear_model import LogisticRegression

selector = FeatureSelector(
    method='sequential',
    n_features_to_select=10,
    direction='forward',
    scoring='accuracy',
    cv=5,
    estimator=LogisticRegression(random_state=42)
)
X_selected = selector.fit_transform(X, y)
```

**Best for**:
- Finding optimal feature combinations
- When you have computational budget
- Small to medium datasets

---

## Column Behavior: Important Note

**Key Concept**: When you specify `columns` in the `FeatureSelector`, **only those specified columns will be considered for selection**. All other columns will **pass through unchanged**.

### Example:

```python
# Original DataFrame
df = pd.DataFrame({
    'age': [25, 30, 35, 40],
    'income': [50000, 60000, 70000, 80000],
    'education': [12, 16, 18, 20],
    'id': [1, 2, 3, 4]  # Should not be selected
})

# Only consider numeric features for selection
selector = FeatureSelector(
    method='selectkbest',
    k=2,
    columns=['age', 'income', 'education']  # 'id' is NOT included
)
X_selected = selector.fit_transform(df, y)

# Result: 'id' column is preserved, and 2 best features from ['age', 'income', 'education'] are selected
```

**Result**:
- The selector will choose the 2 best features from `['age', 'income', 'education']`
- The `'id'` column will remain in the output DataFrame unchanged
- This is useful for preserving identifier columns, metadata, or pre-selected important features

---

## Usage Patterns

### Pattern 1: Quick Feature Screening

```python
# Fast univariate test to get top 20 features
selector = FeatureSelector(method='selectkbest', k=20, score_func='mutual_info_classif')
X_selected = selector.fit_transform(X, y)
```

### Pattern 2: Iterative Refinement

```python
# Step 1: Remove constant features
selector1 = FeatureSelector(method='variance_threshold', threshold=0.01)
X1 = selector1.fit_transform(X, y)

# Step 2: Use RFE for final selection
selector2 = FeatureSelector(method='rfe', n_features_to_select=15)
X_final = selector2.fit_transform(X1, y)
```

### Pattern 3: Compare Multiple Methods

```python
methods = ['selectkbest', 'mutual_info', 'rfe', 'from_model']
results = {}

for method in methods:
    selector = FeatureSelector(method=method, k=10)
    X_selected = selector.fit_transform(X, y)
    results[method] = selector.get_selected_features()
    
# Find common features across methods
common_features = set.intersection(*[set(features) for features in results.values()])
```

### Pattern 4: Pipeline Integration

```python
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestClassifier

pipeline = Pipeline([
    ('scaler', StandardScaler()),
    ('selector', FeatureSelector(method='rfe', n_features_to_select=15)),
    ('classifier', RandomForestClassifier(random_state=42))
])

pipeline.fit(X, y)
y_pred = pipeline.predict(X_test)
```

---

## Performance Considerations

| Method | Speed | Memory | Best Dataset Size |
|--------|-------|--------|-------------------|
| `variance_threshold` | ⚡⚡⚡ | Low | Any |
| `chi2` | ⚡⚡⚡ | Low | Large |
| `f_classif` / `f_regression` | ⚡⚡⚡ | Low | Large |
| `selectkbest` | ⚡⚡ | Low | Large |
| `mutual_info` | ⚡⚡ | Medium | Medium to Large |
| `from_model` | ⚡ | Medium | Medium |
| `rfe` | ⚡ | High | Small to Medium |
| `sequential` | 🐌 | High | Small |

---

## Common Pitfalls and Tips

### ❌ Don't:
- Use `chi2` with negative features (will error)
- Use classification methods (`f_classif`, `chi2`) for regression
- Select more features than samples (overfitting risk)
- Forget to fit the selector before transforming

### ✅ Do:
- Start with fast methods (`variance_threshold`, `selectkbest`)
- Use domain knowledge to exclude irrelevant columns
- Try multiple methods and compare results
- Use cross-validation to validate selected features
- Document which features were selected and why

---

## API Reference

### FeatureSelector Class

```python
from skyulf_mlflow_library.features.selection import FeatureSelector

selector = FeatureSelector(
    method='selectkbest',  # Selection method (required)
    columns=None,          # Columns to consider (default: all numeric)
    k=10,                  # Number of features (method-dependent)
    **kwargs               # Method-specific parameters
)

# Fit and transform
X_selected = selector.fit_transform(X, y)

# Get selected feature names
selected_features = selector.get_selected_features()

# Get feature scores (if available)
scores = selector.get_feature_scores()
```

---

## Further Reading

- [scikit-learn Feature Selection Guide](https://scikit-learn.org/stable/modules/feature_selection.html)
- [Feature Engineering and Selection Book](http://www.feat.engineering/)
- [An Introduction to Variable and Feature Selection (Guyon & Elisseeff)](http://www.jmlr.org/papers/v3/guyon03a.html)

---

**Need help?** Check the [main README](../README.md) or [open an issue](https://github.com/flyingriverhorse/skyulf_mlflow_library/issues).
