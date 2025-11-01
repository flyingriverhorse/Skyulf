"""
Example: Model Training with Skyulf MLflow

This example demonstrates:
1. Loading data
2. Preprocessing and feature engineering
3. Training classification models
4. Training regression models
5. Hyperparameter tuning
6. Model comparison
"""

import pandas as pd
import numpy as np
from sklearn.datasets import make_classification, make_regression

# Classification models
from skyulf_mlflow_library.modeling import (
    RandomForestClassifier,
    GradientBoostingClassifier,
    LogisticRegression,
)

# Regression models
from skyulf_mlflow_library.modeling import (
    RandomForestRegressor,
    LinearRegression,
    Ridge,
)

from skyulf_mlflow_library.modeling import MetricsCalculator
from skyulf_mlflow_library.modeling.classifiers import tune_hyperparameters as tune_classifier
from skyulf_mlflow_library.modeling.regressors import tune_hyperparameters as tune_regressor

# Preprocessing
from skyulf_mlflow_library.preprocessing import (
    SimpleImputer,
    StandardScaler,
)

from skyulf_mlflow_library.utils import train_test_split

print("=" * 80)
print("Skyulf MLflow - Model Training Example")
print("=" * 80)

# ============================================================================
# Example 1: Classification with Multiple Models
# ============================================================================
print("\n" + "=" * 80)
print("Example 1: Classification Model Comparison")
print("=" * 80)

# Generate synthetic classification data
X_class, y_class = make_classification(
    n_samples=1000,
    n_features=20,
    n_informative=15,
    n_redundant=5,
    n_classes=2,
    random_state=42
)

# Convert to DataFrame
X_class_df = pd.DataFrame(
    X_class,
    columns=[f'feature_{i}' for i in range(X_class.shape[1])]
)

print(f"\nClassification Dataset: {X_class_df.shape[0]} samples, {X_class_df.shape[1]} features")
print(f"Class distribution: {np.bincount(y_class)}")

# Split data
X_train, X_test, y_train, y_test = train_test_split(
    X_class_df, y_class, test_size=0.2, random_state=42
)

# Preprocess
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

# Train multiple classification models
models = {
    'Random Forest': RandomForestClassifier(n_estimators=100, random_state=42),
    'Gradient Boosting': GradientBoostingClassifier(n_estimators=100, random_state=42),
    'Logistic Regression': LogisticRegression(random_state=42),
}

results = {}
for name, model in models.items():
    print(f"\nTraining {name}...")
    
    # Train model
    model.fit(X_train_scaled, y_train)
    
    # Make predictions
    y_pred = model.predict(X_test_scaled)
    y_pred_proba = model.predict_proba(X_test_scaled)
    
    # Calculate metrics
    metrics_calc = MetricsCalculator(problem_type='classification')
    metrics = metrics_calc.calculate(y_test, y_pred, y_pred_proba)
    
    results[name] = metrics
    
    print(f"  Accuracy: {metrics['accuracy']:.4f}")
    print(f"  Precision: {metrics['precision']:.4f}")
    print(f"  Recall: {metrics['recall']:.4f}")
    print(f"  F1-Score: {metrics['f1']:.4f}")

# Find best model
best_model = max(results.items(), key=lambda x: x[1]['f1'])
print(f"\n✓ Best Classification Model: {best_model[0]} (F1={best_model[1]['f1']:.4f})")

# ============================================================================
# Example 2: Regression with Multiple Models
# ============================================================================
print("\n" + "=" * 80)
print("Example 2: Regression Model Comparison")
print("=" * 80)

# Generate synthetic regression data
X_reg, y_reg = make_regression(
    n_samples=1000,
    n_features=20,
    n_informative=15,
    noise=10.0,
    random_state=42
)

# Convert to DataFrame
X_reg_df = pd.DataFrame(
    X_reg,
    columns=[f'feature_{i}' for i in range(X_reg.shape[1])]
)

print(f"\nRegression Dataset: {X_reg_df.shape[0]} samples, {X_reg_df.shape[1]} features")
print(f"Target range: [{y_reg.min():.2f}, {y_reg.max():.2f}]")

# Split data
X_train, X_test, y_train, y_test = train_test_split(
    X_reg_df, y_reg, test_size=0.2, random_state=42
)

# Preprocess
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

# Train multiple regression models
reg_models = {
    'Random Forest': RandomForestRegressor(n_estimators=100, random_state=42),
    'Linear Regression': LinearRegression(),
    'Ridge Regression': Ridge(alpha=1.0),
}

reg_results = {}
for name, model in reg_models.items():
    print(f"\nTraining {name}...")
    
    # Train model
    model.fit(X_train_scaled, y_train)
    
    # Make predictions
    y_pred = model.predict(X_test_scaled)
    
    # Calculate metrics
    metrics_calc = MetricsCalculator(problem_type='regression')
    metrics = metrics_calc.calculate(y_test, y_pred)
    
    reg_results[name] = metrics
    
    print(f"  MSE: {metrics['mse']:.4f}")
    print(f"  RMSE: {metrics['rmse']:.4f}")
    print(f"  MAE: {metrics['mae']:.4f}")
    print(f"  R²: {metrics['r2']:.4f}")

# Find best model
best_reg_model = max(reg_results.items(), key=lambda x: x[1]['r2'])
print(f"\n✓ Best Regression Model: {best_reg_model[0]} (R²={best_reg_model[1]['r2']:.4f})")

# ============================================================================
# Example 3: Hyperparameter Tuning
# ============================================================================
print("\n" + "=" * 80)
print("Example 3: Hyperparameter Tuning")
print("=" * 80)

# Classification hyperparameter tuning
print("\n--- Classification Hyperparameter Tuning ---")

# Re-split classification data for tuning
X_train_clf, X_test_clf, y_train_clf, y_test_clf = train_test_split(
    X_class_df, y_class, test_size=0.2, random_state=42
)
scaler_clf = StandardScaler()
X_train_clf_scaled = scaler_clf.fit_transform(X_train_clf)
X_test_clf_scaled = scaler_clf.transform(X_test_clf)

rf_classifier = RandomForestClassifier(random_state=42)

# Get default parameter grid
param_grid = rf_classifier.get_param_grid()
print(f"Parameter grid: {param_grid}")

# Tune hyperparameters (using smaller grid for faster execution)
print("\nTuning Random Forest Classifier (this may take a moment)...")
small_param_grid = {
    'n_estimators': [50, 100],
    'max_depth': [10, 20],
    'min_samples_split': [2, 5]
}
best_rf_clf, best_params = tune_classifier(
    rf_classifier,
    X_train_clf_scaled,
    y_train_clf,
    param_grid=small_param_grid,
    cv=3,
    scoring='f1',
    n_jobs=1
)

# Evaluate tuned model
y_pred_tuned = best_rf_clf.predict(X_test_clf_scaled)
y_pred_proba_tuned = best_rf_clf.predict_proba(X_test_clf_scaled)

metrics_calc = MetricsCalculator(problem_type='classification')
tuned_metrics = metrics_calc.calculate(y_test_clf, y_pred_tuned, y_pred_proba_tuned)

print(f"✓ Tuned Model F1-Score: {tuned_metrics['f1']:.4f}")
print(f"  Best parameters: {best_params}")

# Regression hyperparameter tuning
print("\n--- Regression Hyperparameter Tuning ---")

# Re-split regression data for tuning
X_train_reg, X_test_reg, y_train_reg, y_test_reg = train_test_split(
    X_reg_df, y_reg, test_size=0.2, random_state=42
)
scaler_reg = StandardScaler()
X_train_reg_scaled = scaler_reg.fit_transform(X_train_reg)
X_test_reg_scaled = scaler_reg.transform(X_test_reg)

rf_regressor = RandomForestRegressor(random_state=42)

# Tune hyperparameters (using smaller grid for faster execution)
print("\nTuning Random Forest Regressor (this may take a moment)...")
small_param_grid_reg = {
    'n_estimators': [50, 100],
    'max_depth': [10, 20],
    'min_samples_split': [2, 5]
}
best_rf_reg, best_params_reg = tune_regressor(
    rf_regressor,
    X_train_reg_scaled,
    y_train_reg,
    param_grid=small_param_grid_reg,
    cv=3,
    scoring='r2',
    n_jobs=1
)

# Evaluate tuned model
y_pred_reg_tuned = best_rf_reg.predict(X_test_reg_scaled)

metrics_calc_reg = MetricsCalculator(problem_type='regression')
tuned_reg_metrics = metrics_calc_reg.calculate(y_test_reg, y_pred_reg_tuned)

print(f"✓ Tuned Model R²: {tuned_reg_metrics['r2']:.4f}")
print(f"  Best parameters: {best_params_reg}")

# ============================================================================
# Example 4: Full Pipeline with Feature Engineering
# ============================================================================
print("\n" + "=" * 80)
print("Example 4: Complete ML Pipeline")
print("=" * 80)

from skyulf_mlflow_library.features.transform.polynomial import PolynomialFeatures

# Add missing values to demonstrate handling
X_with_missing = X_class_df.copy()
np.random.seed(42)
mask = np.random.random(X_with_missing.shape) < 0.1
X_with_missing[mask] = np.nan

print(f"\nOriginal features: {X_with_missing.shape[1]}")
print(f"Missing values: {X_with_missing.isnull().sum().sum()}")

# Split
X_train, X_test, y_train, y_test = train_test_split(
    X_with_missing, y_class, test_size=0.2, random_state=42
)

# Handle missing values
print("\n1. Handling missing values...")
missing_handler = SimpleImputer(strategy='mean')
X_train_clean = missing_handler.fit_transform(X_train)
X_test_clean = missing_handler.transform(X_test)
print(f"   ✓ Missing values filled")

# Feature engineering
print("\n2. Creating polynomial features...")
poly = PolynomialFeatures(degree=2, include_bias=False, interaction_only=True)
X_train_poly = poly.fit_transform(X_train_clean)
X_test_poly = poly.transform(X_test_clean)
print(f"   ✓ Features: {X_train_clean.shape[1]} → {X_train_poly.shape[1]}")

# Scale
print("\n3. Scaling features...")
scaler = StandardScaler()
X_train_final = scaler.fit_transform(X_train_poly)
X_test_final = scaler.transform(X_test_poly)
print(f"   ✓ Features scaled")

# Train model
print("\n4. Training Random Forest...")
model = RandomForestClassifier(n_estimators=100, max_depth=10, random_state=42)
model.fit(X_train_final, y_train)
print(f"   ✓ Model trained")

# Evaluate
print("\n5. Evaluating model...")
y_pred = model.predict(X_test_final)
y_pred_proba = model.predict_proba(X_test_final)

metrics_calc = MetricsCalculator(problem_type='classification')
final_metrics = metrics_calc.calculate(y_test, y_pred, y_pred_proba)

print(f"   ✓ Accuracy: {final_metrics['accuracy']:.4f}")
print(f"   ✓ F1-Score: {final_metrics['f1']:.4f}")
print(f"   ✓ ROC-AUC: {final_metrics['roc_auc']:.4f}")

# ============================================================================
# Summary
# ============================================================================
print("\n" + "=" * 80)
print("Summary")
print("=" * 80)

print("\n✓ Successfully demonstrated:")
print("  1. Training multiple classification models")
print("  2. Training multiple regression models")
print("  3. Hyperparameter tuning with cross-validation")
print("  4. Complete ML pipeline with preprocessing and feature engineering")

print("\n✓ Available models:")
print("  Classification:")
print("    - RandomForestClassifier")
print("    - GradientBoostingClassifier")
print("    - LogisticRegression")
print("    - SupportVectorClassifier")
print("    - DecisionTreeClassifier")
print("    - KNeighborsClassifier")
print("\n  Regression:")
print("    - RandomForestRegressor")
print("    - GradientBoostingRegressor")
print("    - LinearRegression")
print("    - Ridge")
print("    - Lasso")
print("    - ElasticNet")
print("    - DecisionTreeRegressor")
print("    - KNeighborsRegressor")

print("\n" + "=" * 80)
print("Example completed successfully!")
print("=" * 80)
