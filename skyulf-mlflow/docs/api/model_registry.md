# Model Registry API Documentation

## Overview

The Model Registry provides version control, metadata tracking, and deployment management for machine learning models. Store models with their hyperparameters, metrics, training data, and tags in a SQLite database for easy retrieval and comparison.

---

## 📦 Module: `skyulf_mlflow_library.modeling.registry`

### Key Class

- **ModelRegistry**: Central registry for model management

---

## 🗄️ ModelRegistry

Manage model versions, metadata, and deployment lifecycle.

**Signature:**
```python
class ModelRegistry:
    def __init__(
        self,
        db_path: str = 'model_registry.db',
        models_dir: str = 'models/'
    )
```

**Parameters:**
- `db_path` (str, default='model_registry.db'): Path to SQLite database
- `models_dir` (str, default='models/'): Directory to store model files

**Database Schema:**
```sql
CREATE TABLE models (
    id INTEGER PRIMARY KEY,
    name TEXT NOT NULL,
    version TEXT NOT NULL,
    model_type TEXT,
    created_at TIMESTAMP,
    metrics JSON,
    hyperparameters JSON,
    tags JSON,
    file_path TEXT,
    description TEXT,
    status TEXT DEFAULT 'development'
)
```

---

## 📝 Register Models

### Basic Registration

**Example:**
```python
from skyulf_mlflow_library.modeling import ModelRegistry, RandomForestClassifier

# Initialize registry
registry = ModelRegistry(
    db_path='models/registry.db',
    models_dir='models/artifacts/'
)

# Train model
model = RandomForestClassifier(n_estimators=100, random_state=42)
model.fit(X_train, y_train)

# Calculate metrics
y_pred = model.predict(X_test)
from sklearn.metrics import accuracy_score, f1_score
metrics = {
    'accuracy': accuracy_score(y_test, y_pred),
    'f1_score': f1_score(y_test, y_pred, average='weighted')
}

# Register model
model_id = registry.register_model(
    name='customer_churn_classifier',
    model=model,
    version='1.0.0',
    metrics=metrics,
    hyperparameters=model.model.get_params(),
    description='Random Forest classifier for customer churn prediction',
    tags=['production', 'v1', 'random_forest']
)

print(f"✓ Model registered with ID: {model_id}")
```

---

### Register with Full Metadata

**Example:**
```python
# Complete registration with all metadata
model_id = registry.register_model(
    name='fraud_detector',
    model=model,
    version='2.1.0',
    model_type='GradientBoostingClassifier',
    metrics={
        'accuracy': 0.95,
        'precision': 0.93,
        'recall': 0.94,
        'f1_score': 0.935,
        'roc_auc': 0.97,
        'training_samples': 10000,
        'test_samples': 2500
    },
    hyperparameters={
        'n_estimators': 200,
        'learning_rate': 0.05,
        'max_depth': 5,
        'min_samples_split': 20,
        'subsample': 0.8
    },
    tags=[
        'production',
        'fraud-detection',
        'gradient-boosting',
        'model-v2'
    ],
    description='''
        Gradient Boosting Classifier v2.1.0
        - Improved feature engineering
        - SMOTE oversampling applied
        - Tuned hyperparameters via GridSearchCV
        - Production-ready
    ''',
    status='production'
)
```

---

## 🔍 Search and Retrieve Models

### Get Model by ID

**Example:**
```python
# Load model by ID
model_info = registry.get_model(model_id=5)

print(f"Name: {model_info['name']}")
print(f"Version: {model_info['version']}")
print(f"Metrics: {model_info['metrics']}")
print(f"Status: {model_info['status']}")

# Load the actual model object
model = model_info['model']
predictions = model.predict(X_new)
```

---

### Get Latest Model Version

**Example:**
```python
# Get latest version of a model
latest = registry.get_latest_version('customer_churn_classifier')

print(f"Latest version: {latest['version']}")
print(f"Metrics: {latest['metrics']}")

# Use the model
model = latest['model']
predictions = model.predict(X_new)
```

---

### Search by Name

**Example:**
```python
# Get all versions of a model
versions = registry.get_model_by_name('fraud_detector')

print(f"Found {len(versions)} versions")
for v in versions:
    print(f"Version {v['version']}: Accuracy = {v['metrics']['accuracy']:.4f}")

# Compare versions
best_version = max(versions, key=lambda x: x['metrics']['accuracy'])
print(f"Best version: {best_version['version']}")
```

---

### Search by Tags

**Example:**
```python
# Find all production models
production_models = registry.search_by_tag('production')

for model_info in production_models:
    print(f"{model_info['name']} v{model_info['version']}")
    print(f"  Metrics: {model_info['metrics']}")
    print(f"  Status: {model_info['status']}")

# Find by multiple tags
fraud_models = registry.search_by_tag('fraud-detection')
gb_models = registry.search_by_tag('gradient-boosting')

# Intersection: models with both tags
fraud_gb = [m for m in fraud_models if m in gb_models]
```

---

### List All Models

**Example:**
```python
# Get all registered models
all_models = registry.list_models()

print(f"Total models: {len(all_models)}")

# Group by name
from collections import defaultdict
models_by_name = defaultdict(list)
for model in all_models:
    models_by_name[model['name']].append(model)

for name, versions in models_by_name.items():
    print(f"\n{name}: {len(versions)} versions")
    for v in versions:
        print(f"  - v{v['version']}: {v['status']}")
```

---

## 🔄 Update Models

### Update Model Status

**Example:**
```python
# Promote model to production
registry.update_model_status(
    model_id=5,
    status='production'
)

# Deprecate old model
registry.update_model_status(
    model_id=3,
    status='deprecated'
)

# Archive retired model
registry.update_model_status(
    model_id=1,
    status='archived'
)
```

**Status Lifecycle:**
```
development → staging → production → deprecated → archived
```

---

### Update Metadata

**Example:**
```python
# Add tags
current_tags = registry.get_model(5)['tags']
new_tags = current_tags + ['champion-model', 'high-accuracy']

registry.update_model_metadata(
    model_id=5,
    tags=new_tags
)

# Update description
registry.update_model_metadata(
    model_id=5,
    description='Updated with new training data (Q4 2024)'
)

# Update metrics after re-evaluation
new_metrics = {
    'accuracy': 0.96,
    'f1_score': 0.94,
    'evaluated_on': '2024-11-01'
}
registry.update_model_metadata(
    model_id=5,
    metrics=new_metrics
)
```

---

## 📊 Compare Models

### Compare Metrics

**Example:**
```python
# Compare all versions of a model
def compare_model_versions(registry, model_name):
    versions = registry.get_model_by_name(model_name)
    
    print(f"\n{'Version':<10} {'Accuracy':<10} {'F1-Score':<10} {'Status':<15}")
    print("-" * 50)
    
    for v in sorted(versions, key=lambda x: x['version']):
        metrics = v['metrics']
        print(f"{v['version']:<10} "
              f"{metrics.get('accuracy', 0):<10.4f} "
              f"{metrics.get('f1_score', 0):<10.4f} "
              f"{v['status']:<15}")

compare_model_versions(registry, 'customer_churn_classifier')

# Output:
# Version    Accuracy   F1-Score   Status         
# --------------------------------------------------
# 1.0.0      0.8500     0.8300     deprecated     
# 1.1.0      0.8700     0.8500     production     
# 2.0.0      0.9100     0.8900     staging        
```

---

### Find Best Model

**Example:**
```python
# Find best model across all versions
def find_best_model(registry, model_name, metric='accuracy'):
    versions = registry.get_model_by_name(model_name)
    
    if not versions:
        return None
    
    best = max(versions, key=lambda x: x['metrics'].get(metric, 0))
    
    return {
        'version': best['version'],
        'metric_value': best['metrics'].get(metric),
        'model_id': best['id'],
        'status': best['status']
    }

best = find_best_model(registry, 'fraud_detector', metric='f1_score')
print(f"Best model: v{best['version']} (F1={best['metric_value']:.4f})")
```

---

## 🚀 Deployment Workflow

### Development to Production

**Example:**
```python
# 1. Development: Train and register model
model = train_model(X_train, y_train)
metrics = evaluate_model(model, X_test, y_test)

model_id = registry.register_model(
    name='sales_forecaster',
    model=model,
    version='1.0.0',
    metrics=metrics,
    status='development',
    tags=['development', 'experimental']
)

# 2. Staging: Promote to staging for validation
if metrics['r2'] > 0.85:
    registry.update_model_status(model_id, 'staging')
    print("✓ Promoted to staging")

# 3. Validation: Test on hold-out data
model_info = registry.get_model(model_id)
model = model_info['model']
validation_metrics = evaluate_model(model, X_validation, y_validation)

# 4. Production: Deploy if validation passes
if validation_metrics['r2'] > 0.80:
    # Deprecate old production model
    old_production = registry.search_by_tag('production-active')
    for old_model in old_production:
        if old_model['name'] == 'sales_forecaster':
            registry.update_model_status(old_model['id'], 'deprecated')
    
    # Promote new model
    registry.update_model_status(model_id, 'production')
    registry.update_model_metadata(
        model_id,
        tags=model_info['tags'] + ['production-active']
    )
    print("✓ Deployed to production")
```

---

### A/B Testing

**Example:**
```python
# Deploy two models for A/B testing
model_a_id = registry.register_model(
    name='recommender_v1',
    model=model_a,
    version='1.0.0',
    metrics=metrics_a,
    status='production',
    tags=['production', 'variant-a', 'ab-test']
)

model_b_id = registry.register_model(
    name='recommender_v1',
    model=model_b,
    version='1.1.0',
    metrics=metrics_b,
    status='production',
    tags=['production', 'variant-b', 'ab-test']
)

# Load models for serving
import random

def get_model_for_user(user_id):
    # Assign 50% to each variant
    if hash(user_id) % 2 == 0:
        model_info = registry.get_model(model_a_id)
        variant = 'A'
    else:
        model_info = registry.get_model(model_b_id)
        variant = 'B'
    
    return model_info['model'], variant

# Usage
model, variant = get_model_for_user('user_12345')
prediction = model.predict(features)
log_prediction(prediction, variant)  # Track which variant was used
```

---

## 🗑️ Delete Models

**Example:**
```python
# Delete specific version
registry.delete_model(model_id=3)

# Delete all versions of a model
versions = registry.get_model_by_name('old_model')
for v in versions:
    registry.delete_model(v['id'])

# Clean up archived models
archived = [m for m in registry.list_models() if m['status'] == 'archived']
for model in archived:
    registry.delete_model(model['id'])
```

---

## 📋 Model Information Report

**Example:**
```python
def print_model_report(registry, model_id):
    """Generate detailed model report"""
    info = registry.get_model(model_id)
    
    print("=" * 60)
    print(f"MODEL REPORT")
    print("=" * 60)
    print(f"Name:        {info['name']}")
    print(f"Version:     {info['version']}")
    print(f"Type:        {info['model_type']}")
    print(f"Status:      {info['status']}")
    print(f"Created:     {info['created_at']}")
    print(f"\nDescription:")
    print(f"  {info['description']}")
    
    print(f"\nMetrics:")
    for metric, value in info['metrics'].items():
        print(f"  {metric:<20} {value}")
    
    print(f"\nHyperparameters:")
    for param, value in info['hyperparameters'].items():
        print(f"  {param:<20} {value}")
    
    print(f"\nTags:")
    print(f"  {', '.join(info['tags'])}")
    
    print("=" * 60)

# Generate report
print_model_report(registry, model_id=5)
```

---

## 🔄 Complete Registry Workflow

```python
from skyulf_mlflow_library.modeling import (
    ModelRegistry,
    RandomForestClassifier,
    MetricsCalculator
)
from skyulf_mlflow_library.utils import train_test_split

# 1. Initialize registry
registry = ModelRegistry(
    db_path='production/registry.db',
    models_dir='production/models/'
)

# 2. Train model
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

model = RandomForestClassifier(n_estimators=200, random_state=42)
model.fit(X_train, y_train)

# 3. Evaluate
y_pred = model.predict(X_test)
y_proba = model.predict_proba(X_test)

calc = MetricsCalculator(problem_type='classification')
metrics = calc.calculate(y_test, y_pred, y_proba)

# 4. Register
model_id = registry.register_model(
    name='churn_predictor',
    model=model,
    version='1.0.0',
    metrics=metrics,
    hyperparameters=model.model.get_params(),
    tags=['random-forest', 'churn', 'v1'],
    status='development'
)

# 5. Promote through stages
print("Testing in staging...")
registry.update_model_status(model_id, 'staging')

# Validate
validation_score = validate_on_holdout(model, X_validation, y_validation)
if validation_score > 0.85:
    print("✓ Validation passed")
    registry.update_model_status(model_id, 'production')
    print("✓ Deployed to production")

# 6. Later: Load for inference
print("\nLoading model for inference...")
latest = registry.get_latest_version('churn_predictor')
production_model = latest['model']

# Make predictions
new_predictions = production_model.predict(new_data)
```

---

## 🎯 Best Practices

### Versioning Strategy

**Semantic Versioning:**
```
MAJOR.MINOR.PATCH

MAJOR: Incompatible API changes, different algorithm
MINOR: New features, improved accuracy
PATCH: Bug fixes, minor improvements

Examples:
1.0.0 → Initial model
1.1.0 → Added new features
1.1.1 → Bug fix in preprocessing
2.0.0 → Changed from RF to GB (breaking change)
```

**Implementation:**
```python
def get_next_version(current_version, change_type='patch'):
    """
    Calculate next version number
    
    change_type: 'major', 'minor', or 'patch'
    """
    major, minor, patch = map(int, current_version.split('.'))
    
    if change_type == 'major':
        return f"{major + 1}.0.0"
    elif change_type == 'minor':
        return f"{major}.{minor + 1}.0"
    else:  # patch
        return f"{major}.{minor}.{patch + 1}"

# Usage
current = '1.2.3'
next_version = get_next_version(current, 'minor')  # '1.3.0'
```

---

### Tagging Strategy

**Recommended Tags:**
```python
# Environment
tags = ['development', 'staging', 'production']

# Algorithm
tags = ['random-forest', 'gradient-boosting', 'neural-network']

# Use case
tags = ['fraud-detection', 'churn-prediction', 'recommendation']

# Performance
tags = ['high-accuracy', 'fast-inference', 'low-memory']

# Experiment
tags = ['experiment-001', 'ablation-study', 'hyperparameter-tuning']

# Status
tags = ['champion-model', 'challenger-model', 'baseline']
```

---

### Monitoring and Maintenance

**Regular Checks:**
```python
# Check for outdated models
def check_outdated_models(registry, days=90):
    """Find models older than X days"""
    from datetime import datetime, timedelta
    
    cutoff = datetime.now() - timedelta(days=days)
    all_models = registry.list_models()
    
    outdated = [
        m for m in all_models
        if datetime.fromisoformat(m['created_at']) < cutoff
        and m['status'] == 'production'
    ]
    
    return outdated

# Review and retrain if needed
outdated = check_outdated_models(registry, days=90)
for model in outdated:
    print(f"⚠️  {model['name']} v{model['version']} is {model['created_at']}")
```

---

## 📚 See Also

- [Model Training Guide](model_training.md)
- [Deployment Tutorial](../tutorials/deployment.md)
- [MLOps Best Practices](../guides/mlops.md)
