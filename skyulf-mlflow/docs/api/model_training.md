# Model Training API Documentation

## Overview

The modeling module provides unified interfaces for classification and regression algorithms with built-in hyperparameter tuning, cross-validation, and model evaluation. All models are scikit-learn compatible with consistent fit/predict interfaces.

---

## 📦 Module: `skyulf_mlflow_library.modeling`

### Components

- **classifiers**: Classification algorithms
- **regressors**: Regression algorithms
- **metrics**: Model evaluation metrics
- **registry**: Model versioning and storage

---

## 🎯 Classification Models

All classifiers inherit from `BaseClassifier` and provide:
- `fit(X, y)`: Train the model
- `predict(X)`: Predict class labels
- `predict_proba(X)`: Predict class probabilities
- `get_param_grid()`: Get default hyperparameter grid

---

### `RandomForestClassifier`

Ensemble of decision trees with bootstrap aggregating.

**Signature:**
```python
class RandomForestClassifier(BaseClassifier):
    def __init__(
        self,
        n_estimators: int = 100,
        max_depth: Optional[int] = None,
        min_samples_split: int = 2,
        min_samples_leaf: int = 1,
        max_features: str = 'sqrt',
        random_state: Optional[int] = None,
        n_jobs: int = -1
    )
```

**Parameters:**
- `n_estimators` (int, default=100): Number of trees
- `max_depth` (int, optional): Maximum tree depth
- `min_samples_split` (int, default=2): Min samples to split node
- `min_samples_leaf` (int, default=1): Min samples in leaf
- `max_features` (str, default='sqrt'): Features per split
- `random_state` (int, optional): Random seed
- `n_jobs` (int, default=-1): Parallel jobs

**Example:**
```python
from skyulf_mlflow_library.modeling import RandomForestClassifier
from sklearn.datasets import make_classification

# Generate data
X, y = make_classification(n_samples=1000, n_features=20, random_state=42)

# Create and train model
model = RandomForestClassifier(
    n_estimators=100,
    max_depth=20,
    random_state=42
)
model.fit(X, y)

# Predictions
y_pred = model.predict(X)
y_proba = model.predict_proba(X)

print(f"Accuracy: {(y == y_pred).mean():.4f}")
```

**When to Use:**
- **General purpose**: Works well on most problems
- **Non-linear relationships**: Captures complex patterns
- **Mixed feature types**: Handles numeric and categorical
- **Feature importance**: Built-in feature ranking

**Advantages:**
- Robust to overfitting
- Handles missing values
- No feature scaling needed
- Parallel training
- Low hyperparameter tuning needed

**Disadvantages:**
- Can be slow on large datasets
- Memory intensive
- Black box (less interpretable)

**Hyperparameter Tuning:**
```python
# Get default parameter grid
param_grid = model.get_param_grid()
print(param_grid)
# {
#     'n_estimators': [50, 100, 200, 300],
#     'max_depth': [None, 10, 20, 30],
#     'min_samples_split': [2, 5, 10],
#     'min_samples_leaf': [1, 2, 4],
#     'max_features': ['sqrt', 'log2']
# }
```

**Best Practices:**
```python
# 1. Start with defaults
model = RandomForestClassifier(random_state=42)

# 2. Increase trees for better performance
model = RandomForestClassifier(n_estimators=500, random_state=42)

# 3. Limit depth to prevent overfitting
model = RandomForestClassifier(n_estimators=200, max_depth=15, random_state=42)

# 4. Use feature importance
model.fit(X, y)
importances = model.model.feature_importances_
```

---

### `GradientBoostingClassifier`

Sequential ensemble that corrects previous tree errors.

**Signature:**
```python
class GradientBoostingClassifier(BaseClassifier):
    def __init__(
        self,
        n_estimators: int = 100,
        learning_rate: float = 0.1,
        max_depth: int = 3,
        min_samples_split: int = 2,
        min_samples_leaf: int = 1,
        subsample: float = 1.0,
        random_state: Optional[int] = None
    )
```

**Parameters:**
- `n_estimators` (int, default=100): Number of boosting stages
- `learning_rate` (float, default=0.1): Shrinkage rate
- `max_depth` (int, default=3): Maximum tree depth
- `subsample` (float, default=1.0): Fraction of samples per tree

**Example:**
```python
from skyulf_mlflow_library.modeling import GradientBoostingClassifier

model = GradientBoostingClassifier(
    n_estimators=200,
    learning_rate=0.05,
    max_depth=5,
    random_state=42
)
model.fit(X_train, y_train)

y_pred = model.predict(X_test)
```

**When to Use:**
- **High accuracy needs**: Often wins competitions
- **Structured/tabular data**: Best for non-image data
- **Feature interactions**: Captures complex relationships

**Advantages:**
- Often highest accuracy
- Handles missing values
- Built-in regularization
- Feature importance

**Disadvantages:**
- Slower training (sequential)
- More hyperparameters to tune
- Can overfit easily
- Requires careful tuning

**Key Hyperparameters:**
```python
# Learning rate vs n_estimators trade-off
# Lower learning rate → More estimators needed
# Higher learning rate → Faster but less accurate

# Conservative (high accuracy)
model = GradientBoostingClassifier(
    n_estimators=500,
    learning_rate=0.01,
    max_depth=5
)

# Aggressive (faster)
model = GradientBoostingClassifier(
    n_estimators=100,
    learning_rate=0.1,
    max_depth=3
)
```

---

### `LogisticRegression`

Linear model for binary and multiclass classification.

**Signature:**
```python
class LogisticRegression(BaseClassifier):
    def __init__(
        self,
        penalty: str = 'l2',
        C: float = 1.0,
        solver: str = 'lbfgs',
        max_iter: int = 100,
        random_state: Optional[int] = None
    )
```

**Parameters:**
- `penalty` (str, default='l2'): Regularization type ('l1', 'l2', 'elasticnet', None)
- `C` (float, default=1.0): Inverse regularization strength (smaller = stronger)
- `solver` (str, default='lbfgs'): Optimization algorithm
- `max_iter` (int, default=100): Maximum iterations

**Example:**
```python
from skyulf_mlflow_library.modeling import LogisticRegression
from skyulf_mlflow_library.preprocessing import StandardScaler

# Scale features first!
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

# Train model
model = LogisticRegression(C=1.0, penalty='l2', random_state=42)
model.fit(X_train_scaled, y_train)

# Get coefficients
print(f"Coefficients: {model.model.coef_}")
print(f"Intercept: {model.model.intercept_}")
```

**When to Use:**
- **Linear relationships**: Features have linear effect on log-odds
- **Need interpretability**: Coefficient = feature importance
- **Baseline model**: Fast to train and evaluate
- **Large datasets**: Efficient on millions of samples

**Advantages:**
- Fast training and prediction
- Probabilistic output
- Easy to interpret
- Low memory usage
- Works with sparse data

**Disadvantages:**
- Assumes linear boundaries
- Requires feature scaling
- Can't capture complex patterns
- Sensitive to outliers

**Regularization:**
```python
# L2 (Ridge): Shrinks all coefficients
model = LogisticRegression(penalty='l2', C=1.0)

# L1 (Lasso): Feature selection (some coefficients = 0)
model = LogisticRegression(penalty='l1', solver='saga', C=1.0)

# No regularization
model = LogisticRegression(penalty=None)

# Stronger regularization (smaller C)
model = LogisticRegression(C=0.01)  # More regularization
model = LogisticRegression(C=100)   # Less regularization
```

---

### `SupportVectorClassifier`

Maximum margin classifier with kernel trick.

**Signature:**
```python
class SupportVectorClassifier(BaseClassifier):
    def __init__(
        self,
        C: float = 1.0,
        kernel: str = 'rbf',
        gamma: str = 'scale',
        random_state: Optional[int] = None
    )
```

**Parameters:**
- `C` (float, default=1.0): Regularization parameter
- `kernel` (str, default='rbf'): Kernel type ('linear', 'poly', 'rbf', 'sigmoid')
- `gamma` (str, default='scale'): Kernel coefficient

**Example:**
```python
from skyulf_mlflow_library.modeling import SupportVectorClassifier

# RBF kernel (default)
model = SupportVectorClassifier(C=1.0, kernel='rbf', random_state=42)
model.fit(X_train_scaled, y_train)

# Linear kernel (faster)
model = SupportVectorClassifier(C=1.0, kernel='linear', random_state=42)
model.fit(X_train_scaled, y_train)
```

**When to Use:**
- **Small to medium datasets**: <10,000 samples
- **High-dimensional data**: More features than samples
- **Clear margin separation**: Well-separated classes
- **Non-linear boundaries**: With RBF kernel

**Kernel Selection:**
```python
# Linear: For linearly separable data
model = SupportVectorClassifier(kernel='linear')

# RBF: General purpose, most common
model = SupportVectorClassifier(kernel='rbf', gamma='scale')

# Polynomial: For polynomial relationships
model = SupportVectorClassifier(kernel='poly', degree=3)
```

**⚠️ Important:**
- **Must scale features**: Very sensitive to scale
- **Slow on large datasets**: O(n²) to O(n³) complexity
- **Memory intensive**: Stores support vectors

---

### `DecisionTreeClassifier`

Single tree for interpretable decisions.

**Signature:**
```python
class DecisionTreeClassifier(BaseClassifier):
    def __init__(
        self,
        max_depth: Optional[int] = None,
        min_samples_split: int = 2,
        min_samples_leaf: int = 1,
        random_state: Optional[int] = None
    )
```

**Example:**
```python
from skyulf_mlflow_library.modeling import DecisionTreeClassifier

model = DecisionTreeClassifier(max_depth=5, random_state=42)
model.fit(X_train, y_train)

# Visualize tree
from sklearn.tree import plot_tree
import matplotlib.pyplot as plt

plt.figure(figsize=(20,10))
plot_tree(model.model, filled=True, feature_names=feature_names)
plt.show()
```

**When to Use:**
- **Need interpretability**: Easy to visualize and explain
- **Mixed feature types**: Handles numeric and categorical
- **Fast predictions**: O(log n) complexity
- **Baseline model**: Before trying ensembles

**Advantages:**
- Highly interpretable
- No feature scaling needed
- Handles missing values
- Captures non-linearity

**Disadvantages:**
- Prone to overfitting
- Unstable (small data changes = different tree)
- Lower accuracy than ensembles

**Prevent Overfitting:**
```python
# Limit depth
model = DecisionTreeClassifier(max_depth=5)

# Increase minimum samples
model = DecisionTreeClassifier(min_samples_split=20, min_samples_leaf=10)

# Use ensemble instead (Random Forest)
from skyulf_mlflow_library.modeling import RandomForestClassifier
model = RandomForestClassifier(n_estimators=100)
```

---

### `KNeighborsClassifier`

Classify based on nearest neighbors.

**Signature:**
```python
class KNeighborsClassifier(BaseClassifier):
    def __init__(
        self,
        n_neighbors: int = 5,
        weights: str = 'uniform',
        metric: str = 'minkowski',
        p: int = 2,
        n_jobs: int = -1
    )
```

**Parameters:**
- `n_neighbors` (int, default=5): Number of neighbors
- `weights` (str, default='uniform'): Weight function ('uniform' or 'distance')
- `metric` (str, default='minkowski'): Distance metric
- `p` (int, default=2): Power for Minkowski (2 = Euclidean)

**Example:**
```python
from skyulf_mlflow_library.modeling import KNeighborsClassifier

model = KNeighborsClassifier(n_neighbors=5, weights='distance', n_jobs=-1)
model.fit(X_train_scaled, y_train)

y_pred = model.predict(X_test_scaled)
```

**When to Use:**
- **Small datasets**: Fast with <10,000 samples
- **Low dimensions**: <20 features
- **Non-parametric**: No assumptions about data distribution
- **Anomaly detection**: Outliers have few neighbors

**Choosing k:**
```python
# Small k: More sensitive to noise
model = KNeighborsClassifier(n_neighbors=3)

# Large k: Smoother boundaries, more robust
model = KNeighborsClassifier(n_neighbors=15)

# Rule of thumb: k = sqrt(n_samples)
import math
k = int(math.sqrt(len(X_train)))
model = KNeighborsClassifier(n_neighbors=k)
```

**⚠️ Important:**
- **Must scale features**: Distance-based
- **Slow predictions**: Searches all training samples
- **Curse of dimensionality**: Poor with many features

---

## 📈 Regression Models

All regressors inherit from `BaseRegressor` and provide:
- `fit(X, y)`: Train the model
- `predict(X)`: Predict continuous values
- `score(X, y)`: Calculate R² score
- `get_param_grid()`: Get default hyperparameter grid

---

### `RandomForestRegressor`

Ensemble of regression trees.

**Example:**
```python
from skyulf_mlflow_library.modeling import RandomForestRegressor

model = RandomForestRegressor(
    n_estimators=100,
    max_depth=20,
    min_samples_split=5,
    random_state=42
)
model.fit(X_train, y_train)

y_pred = model.predict(X_test)
```

**When to Use:**
- Same as classification: general purpose, robust, non-linear

---

### `GradientBoostingRegressor`

Sequential boosting for regression.

**Example:**
```python
from skyulf_mlflow_library.modeling import GradientBoostingRegressor

model = GradientBoostingRegressor(
    n_estimators=200,
    learning_rate=0.05,
    max_depth=5,
    random_state=42
)
model.fit(X_train, y_train)

y_pred = model.predict(X_test)
```

---

### `LinearRegression`

Ordinary least squares regression.

**Signature:**
```python
class LinearRegression(BaseRegressor):
    def __init__(
        self,
        fit_intercept: bool = True,
        normalize: bool = False
    )
```

**Example:**
```python
from skyulf_mlflow_library.modeling import LinearRegression

model = LinearRegression()
model.fit(X_train, y_train)

# Coefficients
print(f"Coefficients: {model.model.coef_}")
print(f"Intercept: {model.model.intercept_}")

y_pred = model.predict(X_test)
```

**When to Use:**
- **Linear relationships**: Y is linear function of X
- **Baseline model**: Start simple
- **Interpretability**: Coefficient = feature impact
- **Fast training**: Analytical solution

---

### `Ridge`

Linear regression with L2 regularization.

**Signature:**
```python
class Ridge(BaseRegressor):
    def __init__(
        self,
        alpha: float = 1.0,
        fit_intercept: bool = True
    )
```

**Example:**
```python
from skyulf_mlflow_library.modeling import Ridge

model = Ridge(alpha=1.0)
model.fit(X_train, y_train)
```

**When to Use:**
- **Multicollinearity**: Correlated features
- **Regularization needed**: Prevent overfitting
- **Many features**: High-dimensional data

**Alpha Selection:**
```python
# No regularization (similar to LinearRegression)
model = Ridge(alpha=0.0)

# Moderate regularization
model = Ridge(alpha=1.0)

# Strong regularization
model = Ridge(alpha=10.0)
```

---

### `Lasso`

Linear regression with L1 regularization (feature selection).

**Signature:**
```python
class Lasso(BaseRegressor):
    def __init__(
        self,
        alpha: float = 1.0,
        fit_intercept: bool = True,
        max_iter: int = 1000
    )
```

**Example:**
```python
from skyulf_mlflow_library.modeling import Lasso

model = Lasso(alpha=0.1)
model.fit(X_train, y_train)

# Check which features were selected
selected = model.model.coef_ != 0
print(f"Selected features: {sum(selected)}/{len(selected)}")
```

**When to Use:**
- **Feature selection**: Automatically zeros out coefficients
- **Sparse models**: Need few important features
- **High-dimensional**: More features than samples

---

### `ElasticNet`

Combines L1 and L2 regularization.

**Signature:**
```python
class ElasticNet(BaseRegressor):
    def __init__(
        self,
        alpha: float = 1.0,
        l1_ratio: float = 0.5,
        fit_intercept: bool = True,
        max_iter: int = 1000
    )
```

**Parameters:**
- `alpha` (float): Regularization strength
- `l1_ratio` (float): L1 vs L2 mix (1.0 = pure Lasso, 0.0 = pure Ridge)

**Example:**
```python
from skyulf_mlflow_library.modeling import ElasticNet

# 50% L1, 50% L2
model = ElasticNet(alpha=1.0, l1_ratio=0.5)
model.fit(X_train, y_train)
```

**When to Use:**
- **Best of both**: Feature selection + handles collinearity
- **Many correlated features**: Better than pure Lasso

---

## 🔧 Hyperparameter Tuning

### Manual Tuning

**Grid Search** (exhaustive):
```python
from skyulf_mlflow_library.modeling.classifiers import tune_hyperparameters

model = RandomForestClassifier(random_state=42)

# Define parameter grid
param_grid = {
    'n_estimators': [50, 100, 200],
    'max_depth': [10, 20, 30],
    'min_samples_split': [2, 5, 10]
}

# Tune (Grid Search)
best_model, best_params = tune_hyperparameters(
    model,
    X_train,
    y_train,
    param_grid=param_grid,
    cv=5,
    scoring='f1',
    n_jobs=-1
)

print(f"Best parameters: {best_params}")
print(f"Best CV score: {best_model.model.best_score_}")
```

**Randomized Search** (faster):
```python
# Use n_iter for random search
best_model, best_params = tune_hyperparameters(
    model,
    X_train,
    y_train,
    param_grid=param_grid,
    cv=5,
    scoring='f1',
    n_iter=20,  # Try 20 random combinations
    random_state=42,
    n_jobs=-1
)
```

### Using Default Grids

```python
# Get model's default parameter grid
model = RandomForestClassifier(random_state=42)
param_grid = model.get_param_grid()

# Tune with defaults
best_model, best_params = tune_hyperparameters(
    model,
    X_train,
    y_train,
    cv=5,
    scoring='accuracy'
)
```

---

## 📊 Model Evaluation

### Classification Metrics

```python
from skyulf_mlflow_library.modeling import MetricsCalculator

# Create calculator
calc = MetricsCalculator(problem_type='classification')

# Calculate metrics
metrics = calc.calculate(
    y_true=y_test,
    y_pred=y_pred,
    y_prob=y_pred_proba
)

print(metrics)
# {
#     'accuracy': 0.85,
#     'precision': 0.82,
#     'recall': 0.88,
#     'f1': 0.85,
#     'roc_auc': 0.91
# }
```

### Regression Metrics

```python
calc = MetricsCalculator(problem_type='regression')

metrics = calc.calculate(
    y_true=y_test,
    y_pred=y_pred
)

print(metrics)
# {
#     'mse': 10.5,
#     'rmse': 3.24,
#     'mae': 2.15,
#     'r2': 0.87
# }
```

---

## 🎯 Model Selection Guide

### Classification

**High Accuracy (Kaggle, Production):**
1. GradientBoostingClassifier
2. RandomForestClassifier
3. XGBoost/LightGBM (external)

**Interpretability:**
1. LogisticRegression
2. DecisionTreeClassifier

**Large Dataset (>100k samples):**
1. LogisticRegression
2. LinearSVC

**Small Dataset (<1k samples):**
1. RandomForestClassifier
2. KNeighborsClassifier

### Regression

**High Accuracy:**
1. GradientBoostingRegressor
2. RandomForestRegressor

**Interpretability:**
1. LinearRegression
2. Ridge/Lasso

**Feature Selection:**
1. Lasso
2. ElasticNet

---

## 📚 Complete Training Pipeline

```python
from skyulf_mlflow_library.preprocessing import StandardScaler, SimpleImputer, SMOTE
from skyulf_mlflow_library.features.encoding import OneHotEncoder
from skyulf_mlflow_library.modeling import RandomForestClassifier, MetricsCalculator
from skyulf_mlflow_library.utils import train_test_split

# 1. Load data
X = df.drop('target', axis=1)
y = df['target']

# 2. Split data
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, stratify=y, random_state=42
)

# 3. Encode categorical
encoder = OneHotEncoder()
X_train = encoder.fit_transform(X_train)
X_test = encoder.transform(X_test)

# 4. Impute missing
imputer = SimpleImputer(strategy='mean')
X_train = imputer.fit_transform(X_train)
X_test = imputer.transform(X_test)

# 5. Scale features
scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)

# 6. Handle imbalance
smote = SMOTE(random_state=42)
X_train, y_train = smote.fit_resample(X_train, y_train)

# 7. Train model
model = RandomForestClassifier(n_estimators=100, random_state=42)
model.fit(X_train, y_train)

# 8. Predict
y_pred = model.predict(X_test)
y_pred_proba = model.predict_proba(X_test)

# 9. Evaluate
calc = MetricsCalculator(problem_type='classification')
metrics = calc.calculate(y_test, y_pred, y_pred_proba)

print(f"F1 Score: {metrics['f1']:.4f}")
print(f"ROC-AUC: {metrics['roc_auc']:.4f}")
```

---

## 📚 See Also

- [Preprocessing Guide](preprocessing.md)
- [Feature Engineering Guide](feature_engineering.md)
- [Model Registry Guide](model_registry.md)
- [Hyperparameter Tuning Tutorial](../tutorials/hyperparameter_tuning.md)
