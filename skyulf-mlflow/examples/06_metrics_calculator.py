"""
Example: Metrics Calculator

This example demonstrates the MetricsCalculator for evaluating machine learning models.

The MetricsCalculator supports:
- Classification metrics: accuracy, precision, recall, F1, ROC AUC, PR AUC
- Regression metrics: MAE, MSE, RMSE, R², MAPE
- Multi-class classification support
- Confusion matrix and classification reports
- Batch metric calculation

Author: Skyulf MLflow Library
"""

import numpy as np
import pandas as pd
from sklearn.datasets import make_classification, make_regression
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor
from sklearn.linear_model import LogisticRegression, LinearRegression
from skyulf_mlflow_library.modeling import MetricsCalculator, calculate_metrics

# Set random seed for reproducibility
np.random.seed(42)

print("=" * 80)
print("SKYULF MLFLOW - METRICS CALCULATOR EXAMPLES")
print("=" * 80)
print()

# ============================================================================
# Example 1: Binary Classification Metrics
# ============================================================================
print("\n" + "=" * 80)
print("Example 1: Binary Classification Metrics")
print("=" * 80)

# Generate binary classification dataset
X_bin, y_bin = make_classification(
    n_samples=1000,
    n_features=20,
    n_informative=15,
    n_redundant=5,
    n_classes=2,
    random_state=42
)

# Split data
X_train, X_test, y_train, y_test = train_test_split(
    X_bin, y_bin, test_size=0.3, random_state=42
)

# Train model
model_bin = RandomForestClassifier(n_estimators=100, random_state=42)
model_bin.fit(X_train, y_train)

# Make predictions
y_pred = model_bin.predict(X_test)
y_prob = model_bin.predict_proba(X_test)

print(f"\nDataset size: {len(X_test)} samples")
print(f"Class distribution: {np.bincount(y_test)}")

# Calculate metrics using MetricsCalculator
calculator = MetricsCalculator('classification')
metrics = calculator.calculate(y_test, y_pred, y_prob)

print("\nBinary Classification Metrics:")
print("-" * 40)
for metric_name, value in sorted(metrics.items()):
    print(f"{metric_name:20s}: {value:.4f}")

# Get confusion matrix
cm = calculator.get_confusion_matrix(y_test, y_pred)
print("\nConfusion Matrix:")
print(cm)

# Get classification report
report = calculator.get_classification_report(
    y_test, y_pred, target_names=['Class 0', 'Class 1']
)
print("\nClassification Report:")
print(report)

# ============================================================================
# Example 2: Multi-class Classification Metrics
# ============================================================================
print("\n" + "=" * 80)
print("Example 2: Multi-class Classification Metrics")
print("=" * 80)

# Generate multi-class classification dataset
X_multi, y_multi = make_classification(
    n_samples=1000,
    n_features=20,
    n_informative=15,
    n_redundant=5,
    n_classes=4,
    n_clusters_per_class=1,
    random_state=42
)

# Split data
X_train, X_test, y_train, y_test = train_test_split(
    X_multi, y_multi, test_size=0.3, random_state=42
)

# Train model
model_multi = RandomForestClassifier(n_estimators=100, random_state=42)
model_multi.fit(X_train, y_train)

# Make predictions
y_pred = model_multi.predict(X_test)
y_prob = model_multi.predict_proba(X_test)

print(f"\nDataset size: {len(X_test)} samples")
print(f"Number of classes: {len(np.unique(y_test))}")
print(f"Class distribution: {np.bincount(y_test)}")

# Calculate metrics
metrics = calculator.calculate(y_test, y_pred, y_prob, average='weighted')

print("\nMulti-class Classification Metrics (Weighted Average):")
print("-" * 40)
for metric_name, value in sorted(metrics.items()):
    print(f"{metric_name:20s}: {value:.4f}")

# Confusion matrix
cm = calculator.get_confusion_matrix(y_test, y_pred, normalize='true')
print("\nNormalized Confusion Matrix (by true labels):")
print(np.round(cm, 3))

# ============================================================================
# Example 3: Regression Metrics
# ============================================================================
print("\n" + "=" * 80)
print("Example 3: Regression Metrics")
print("=" * 80)

# Generate regression dataset
X_reg, y_reg = make_regression(
    n_samples=1000,
    n_features=20,
    n_informative=15,
    noise=10.0,
    random_state=42
)

# Split data
X_train, X_test, y_train, y_test = train_test_split(
    X_reg, y_reg, test_size=0.3, random_state=42
)

# Train model
model_reg = RandomForestRegressor(n_estimators=100, random_state=42)
model_reg.fit(X_train, y_train)

# Make predictions
y_pred = model_reg.predict(X_test)

print(f"\nDataset size: {len(X_test)} samples")
print(f"Target range: [{y_test.min():.2f}, {y_test.max():.2f}]")
print(f"Target mean: {y_test.mean():.2f}, std: {y_test.std():.2f}")

# Calculate metrics
calculator_reg = MetricsCalculator('regression')
metrics = calculator_reg.calculate(y_test, y_pred)

print("\nRegression Metrics:")
print("-" * 40)
for metric_name, value in sorted(metrics.items()):
    if value is not None:
        print(f"{metric_name:20s}: {value:.4f}")
    else:
        print(f"{metric_name:20s}: N/A")

# ============================================================================
# Example 4: Comparing Multiple Models
# ============================================================================
print("\n" + "=" * 80)
print("Example 4: Comparing Multiple Models")
print("=" * 80)

# Use binary classification dataset
X_train, X_test, y_train, y_test = train_test_split(
    X_bin, y_bin, test_size=0.3, random_state=42
)

# Train multiple models
models = {
    'Random Forest': RandomForestClassifier(n_estimators=50, random_state=42),
    'Logistic Regression': LogisticRegression(max_iter=1000, random_state=42),
}

calculator = MetricsCalculator('classification')
results = []

for model_name, model in models.items():
    # Train
    model.fit(X_train, y_train)
    
    # Predict
    y_pred = model.predict(X_test)
    y_prob = model.predict_proba(X_test) if hasattr(model, 'predict_proba') else None
    
    # Calculate metrics
    metrics = calculator.calculate(y_test, y_pred, y_prob)
    metrics['model'] = model_name
    results.append(metrics)

# Create comparison DataFrame
df_results = pd.DataFrame(results)
df_results = df_results.set_index('model')

print("\nModel Comparison:")
print("-" * 80)
print(df_results.round(4))

# Find best model for each metric
print("\nBest Model per Metric:")
print("-" * 40)
for col in df_results.columns:
    best_model = df_results[col].idxmax()
    best_value = df_results[col].max()
    print(f"{col:20s}: {best_model:20s} ({best_value:.4f})")

# ============================================================================
# Example 5: Batch Metric Calculation with Convenience Function
# ============================================================================
print("\n" + "=" * 80)
print("Example 5: Batch Metric Calculation")
print("=" * 80)

# Create synthetic datasets for different tasks
datasets = {
    'Binary Classification': {
        'problem_type': 'classification',
        'y_true': np.random.randint(0, 2, 100),
        'y_pred': np.random.randint(0, 2, 100),
    },
    'Multi-class Classification': {
        'problem_type': 'classification',
        'y_true': np.random.randint(0, 5, 100),
        'y_pred': np.random.randint(0, 5, 100),
    },
    'Regression': {
        'problem_type': 'regression',
        'y_true': np.random.randn(100) * 10 + 50,
        'y_pred': np.random.randn(100) * 10 + 50,
    },
}

print("\nCalculating metrics for multiple datasets:")
print("-" * 80)

for dataset_name, dataset_info in datasets.items():
    print(f"\n{dataset_name}:")
    
    # Use convenience function
    metrics = calculate_metrics(
        y_true=dataset_info['y_true'],
        y_pred=dataset_info['y_pred'],
        problem_type=dataset_info['problem_type']
    )
    
    for metric_name, value in sorted(metrics.items())[:5]:  # Show first 5 metrics
        if value is not None:
            print(f"  {metric_name:20s}: {value:.4f}")
        else:
            print(f"  {metric_name:20s}: N/A")

# ============================================================================
# Example 6: Real-World Use Case - Model Evaluation Pipeline
# ============================================================================
print("\n" + "=" * 80)
print("Example 6: Real-World Use Case - Model Evaluation Pipeline")
print("=" * 80)

# Simulate a model evaluation pipeline
def evaluate_model_pipeline(X, y, model, problem_type):
    """Complete model evaluation pipeline."""
    # Split data
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42
    )
    
    # Train model
    model.fit(X_train, y_train)
    
    # Make predictions
    y_pred = model.predict(X_test)
    y_prob = None
    if problem_type == 'classification' and hasattr(model, 'predict_proba'):
        y_prob = model.predict_proba(X_test)
    
    # Calculate metrics
    calculator = MetricsCalculator(problem_type)
    train_metrics = calculator.calculate(
        y_train, 
        model.predict(X_train),
        model.predict_proba(X_train) if y_prob is not None else None
    )
    test_metrics = calculator.calculate(y_test, y_pred, y_prob)
    
    return {
        'train_metrics': train_metrics,
        'test_metrics': test_metrics,
        'train_size': len(y_train),
        'test_size': len(y_test),
    }

# Classification pipeline
print("\nClassification Model Evaluation:")
print("-" * 40)
clf_results = evaluate_model_pipeline(
    X_bin, y_bin,
    RandomForestClassifier(n_estimators=100, random_state=42),
    'classification'
)

print(f"Train size: {clf_results['train_size']}, Test size: {clf_results['test_size']}")
print("\nTrain Metrics:")
for metric, value in list(clf_results['train_metrics'].items())[:4]:
    print(f"  {metric:20s}: {value:.4f}")

print("\nTest Metrics:")
for metric, value in list(clf_results['test_metrics'].items())[:4]:
    print(f"  {metric:20s}: {value:.4f}")

# Check for overfitting
train_acc = clf_results['train_metrics']['accuracy']
test_acc = clf_results['test_metrics']['accuracy']
overfit_gap = train_acc - test_acc
print(f"\nOverfitting Analysis:")
print(f"  Train Accuracy: {train_acc:.4f}")
print(f"  Test Accuracy:  {test_acc:.4f}")
print(f"  Gap:            {overfit_gap:.4f}")
if overfit_gap > 0.1:
    print("  Status: Model may be overfitting")
else:
    print("  Status: Model is generalizing well")

# Regression pipeline
print("\n\nRegression Model Evaluation:")
print("-" * 40)
reg_results = evaluate_model_pipeline(
    X_reg, y_reg,
    RandomForestRegressor(n_estimators=100, random_state=42),
    'regression'
)

print(f"Train size: {reg_results['train_size']}, Test size: {reg_results['test_size']}")
print("\nTrain Metrics:")
for metric, value in reg_results['train_metrics'].items():
    if value is not None:
        print(f"  {metric:20s}: {value:.4f}")

print("\nTest Metrics:")
for metric, value in reg_results['test_metrics'].items():
    if value is not None:
        print(f"  {metric:20s}: {value:.4f}")

print("\n" + "=" * 80)
print("EXAMPLES COMPLETED SUCCESSFULLY!")
print("=" * 80)
print("\nKey Takeaways:")
print("- MetricsCalculator provides unified interface for classification and regression")
print("- Supports binary and multi-class classification with probability metrics")
print("- Includes confusion matrix and classification reports")
print("- Convenience function calculate_metrics() for quick evaluations")
print("- Perfect for model comparison and evaluation pipelines")
print("=" * 80)
