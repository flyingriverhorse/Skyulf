"""
Example: Complete ML Pipeline

This example demonstrates an end-to-end machine learning pipeline using
Skyulf MLflow components:

1. Data Ingestion - Load data from files
2. Preprocessing - Handle missing values, scale features
3. Feature Engineering - Create new features, encode categoricals
4. Model Training - Train and evaluate models
5. Model Registry - Save and version models
6. Pipeline - Chain all steps together

Author: Skyulf MLflow Library
"""

import numpy as np
import pandas as pd
from sklearn.datasets import make_classification
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier

# Skyulf MLflow imports
from skyulf_mlflow_library.data_ingestion import save_data, load_data
from skyulf_mlflow_library.preprocessing import SimpleImputer, StandardScaler, drop_missing
from skyulf_mlflow_library.features import OneHotEncoder, FeatureMath
from skyulf_mlflow_library.modeling import MetricsCalculator, ModelRegistry
from skyulf_mlflow_library.pipeline import make_pipeline

# Set random seed
np.random.seed(42)

print("=" * 80)
print("SKYULF MLFLOW - COMPLETE ML PIPELINE")
print("=" * 80)
print()

# ============================================================================
# Step 1: Data Ingestion
# ============================================================================
print("\n" + "=" * 80)
print("Step 1: Data Ingestion")
print("=" * 80)

# Generate synthetic dataset
X, y = make_classification(
    n_samples=1000,
    n_features=10,
    n_informative=7,
    n_redundant=2,
    n_classes=2,
    random_state=42
)

# Create DataFrame with realistic column names
feature_names = [f'feature_{i}' for i in range(10)]
df = pd.DataFrame(X, columns=feature_names)
df['target'] = y

# Add some categorical features
df['category_A'] = np.random.choice(['Type1', 'Type2', 'Type3'], size=len(df))
df['category_B'] = np.random.choice(['Low', 'Medium', 'High'], size=len(df))

# Introduce some missing values (realistic scenario)
missing_cols = ['feature_0', 'feature_3', 'feature_7']
for col in missing_cols:
    missing_idx = np.random.choice(df.index, size=int(0.05 * len(df)), replace=False)
    df.loc[missing_idx, col] = np.nan

print(f"\nGenerated dataset shape: {df.shape}")
print(f"Missing values:\n{df.isnull().sum()[df.isnull().sum() > 0]}")
print(f"\nFirst few rows:")
print(df.head())

# Save to CSV
save_data(df, 'temp/sample_data.csv', index=False)
print("\n✓ Data saved to 'temp/sample_data.csv'")

# Load back (demonstrating data ingestion)
df_loaded = load_data('temp/sample_data.csv')
print(f"✓ Data loaded from file, shape: {df_loaded.shape}")

# ============================================================================
# Step 2: Data Splitting
# ============================================================================
print("\n" + "=" * 80)
print("Step 2: Data Splitting")
print("=" * 80)

# Separate features and target
X = df_loaded.drop('target', axis=1)
y = df_loaded['target']

# Split data
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)

print(f"\nTrain set: {X_train.shape}")
print(f"Test set: {X_test.shape}")
print(f"Class distribution in train: {np.bincount(y_train)}")
print(f"Class distribution in test: {np.bincount(y_test)}")

# ============================================================================
# Step 3: Build Preprocessing Pipeline
# ============================================================================
print("\n" + "=" * 80)
print("Step 3: Build Preprocessing Pipeline")
print("=" * 80)

# Define preprocessing steps
preprocessing_pipeline = make_pipeline(
    # Step 1: Impute missing values
    SimpleImputer(strategy='mean', columns=missing_cols),
    
    # Step 2: Encode categorical features
    OneHotEncoder(columns=['category_A', 'category_B'], drop='first'),
    
    # Step 3: Scale numeric features
    StandardScaler(columns=[col for col in feature_names if col not in missing_cols] + missing_cols),
)

print("\nPreprocessing Pipeline:")
for i, (name, step) in enumerate(preprocessing_pipeline.steps, 1):
    print(f"  {i}. {name}: {step.__class__.__name__}")

# Fit and transform
X_train_processed = preprocessing_pipeline.fit_transform(X_train)
X_test_processed = preprocessing_pipeline.transform(X_test)

print(f"\n✓ Preprocessing complete")
print(f"  Train shape after preprocessing: {X_train_processed.shape}")
print(f"  Test shape after preprocessing: {X_test_processed.shape}")
print(f"  Original features: {X_train.shape[1]}, After preprocessing: {X_train_processed.shape[1]}")

# ============================================================================
# Step 4: Feature Engineering
# ============================================================================
print("\n" + "=" * 80)
print("Step 4: Feature Engineering")
print("=" * 80)

# Create interaction features using FeatureMath
feature_engineer = FeatureMath(operations=[
    {
        'output': 'feature_sum',
        'type': 'stat',
        'columns': ['feature_0', 'feature_1', 'feature_2'],
        'method': 'sum'
    },
    {
        'output': 'feature_ratio',
        'type': 'ratio',
        'numerator': 'feature_0',
        'denominator': 'feature_1'
    },
])

# Apply feature engineering
X_train_eng = feature_engineer.fit_transform(X_train_processed)
X_test_eng = feature_engineer.transform(X_test_processed)

print(f"\n✓ Feature engineering complete")
print(f"  Features after engineering: {X_train_eng.shape[1]}")
print(f"  New features created: {X_train_eng.shape[1] - X_train_processed.shape[1]}")

# ============================================================================
# Step 5: Model Training
# ============================================================================
print("\n" + "=" * 80)
print("Step 5: Model Training")
print("=" * 80)

# Train Random Forest model
model = RandomForestClassifier(
    n_estimators=100,
    max_depth=10,
    random_state=42
)

print("\nTraining Random Forest Classifier...")
model.fit(X_train_eng, y_train)
print("✓ Model trained successfully")

# Make predictions
y_train_pred = model.predict(X_train_eng)
y_test_pred = model.predict(X_test_eng)
y_test_proba = model.predict_proba(X_test_eng)

# ============================================================================
# Step 6: Model Evaluation
# ============================================================================
print("\n" + "=" * 80)
print("Step 6: Model Evaluation")
print("=" * 80)

# Calculate metrics
calculator = MetricsCalculator('classification')

train_metrics = calculator.calculate(y_train, y_train_pred)
test_metrics = calculator.calculate(y_test, y_test_pred, y_test_proba)

print("\nTraining Metrics:")
print("-" * 40)
for metric, value in sorted(train_metrics.items())[:5]:
    print(f"  {metric:20s}: {value:.4f}")

print("\nTest Metrics:")
print("-" * 40)
for metric, value in sorted(test_metrics.items())[:5]:
    print(f"  {metric:20s}: {value:.4f}")

# Check for overfitting
train_acc = train_metrics['accuracy']
test_acc = test_metrics['accuracy']
overfit_gap = train_acc - test_acc

print(f"\nGeneralization Analysis:")
print(f"  Train Accuracy:  {train_acc:.4f}")
print(f"  Test Accuracy:   {test_acc:.4f}")
print(f"  Accuracy Gap:    {overfit_gap:.4f}")

if overfit_gap < 0.05:
    print(f"  Status: ✓ Excellent generalization")
elif overfit_gap < 0.10:
    print(f"  Status: ✓ Good generalization")
else:
    print(f"  Status: ⚠ Potential overfitting")

# Confusion matrix
cm = calculator.get_confusion_matrix(y_test, y_test_pred)
print(f"\nConfusion Matrix:")
print(cm)

# ============================================================================
# Step 7: Model Registry
# ============================================================================
print("\n" + "=" * 80)
print("Step 7: Model Registry")
print("=" * 80)

# Initialize model registry
registry = ModelRegistry('./temp/model_registry')

# Save model with metadata
model_id = registry.save_model(
    model=model,
    name='fraud_detector',
    problem_type='classification',
    description='Random Forest classifier for fraud detection',
    metrics=test_metrics,
    tags=['production', 'random_forest', 'v1'],
    version=1
)

print(f"\n✓ Model saved to registry")
print(f"  Model ID: {model_id}")
print(f"  Model Name: fraud_detector")
print(f"  Version: 1")

# List all models in registry
models_df = registry.list_models()
print(f"\nModels in registry:")
print(models_df[['name', 'version', 'problem_type', 'created_at']])

# Get model info
model_info = registry.get_model_info('fraud_detector', version=1)
print(f"\nModel Information:")
print(f"  Name: {model_info['name']}")
print(f"  Version: {model_info['version']}")
print(f"  Problem Type: {model_info['problem_type']}")
print(f"  Description: {model_info['description']}")
print(f"  Tags: {model_info['tags']}")
print(f"  Test Accuracy: {model_info['metrics']['accuracy']:.4f}")

# ============================================================================
# Step 8: Model Loading and Inference
# ============================================================================
print("\n" + "=" * 80)
print("Step 8: Model Loading and Inference")
print("=" * 80)

# Load model from registry
loaded_model = registry.load_model('fraud_detector', version=1)
print("\n✓ Model loaded from registry")

# Make predictions with loaded model
test_predictions = loaded_model.predict(X_test_eng[:5])
test_probas = loaded_model.predict_proba(X_test_eng[:5])

print(f"\nSample Predictions (first 5 test samples):")
print("-" * 60)
print(f"{'True Label':<15} {'Predicted':<15} {'Confidence':<15}")
print("-" * 60)
for i in range(5):
    true_label = y_test.iloc[i]
    pred_label = test_predictions[i]
    confidence = test_probas[i].max()
    print(f"{true_label:<15} {pred_label:<15} {confidence:<15.4f}")

# ============================================================================
# Step 9: Complete Pipeline Summary
# ============================================================================
print("\n" + "=" * 80)
print("Step 9: Complete Pipeline Summary")
print("=" * 80)

print("""
Pipeline Steps:
1. ✓ Data Ingestion    - Loaded 1000 samples with 13 features
2. ✓ Data Splitting    - 80/20 train-test split
3. ✓ Preprocessing     - Imputation, encoding, scaling
4. ✓ Feature Eng.      - Created 2 new features
5. ✓ Model Training    - Random Forest with 100 trees
6. ✓ Evaluation        - Calculated 7+ metrics
7. ✓ Model Registry    - Saved with versioning
8. ✓ Inference         - Loaded and tested model

Key Achievements:
- Test Accuracy: {:.2%}
- ROC AUC: {:.4f}
- Model Size: Stored in registry with metadata
- Reproducibility: All steps tracked and versioned
""".format(test_acc, test_metrics.get('roc_auc', 0.0)))

print("=" * 80)
print("PIPELINE COMPLETED SUCCESSFULLY!")
print("=" * 80)

print("\nNext Steps:")
print("1. Fine-tune hyperparameters for better performance")
print("2. Add cross-validation for robust evaluation")
print("3. Deploy model using registry for production")
print("4. Monitor model performance over time")
print("5. Retrain when performance degrades")

print("\nSkyulf MLflow Features Used:")
print("  • Data Ingestion (load_data, save_data)")
print("  • Preprocessing (SimpleImputer, StandardScaler, OneHotEncoder)")
print("  • Feature Engineering (FeatureMath)")
print("  • Metrics Calculation (MetricsCalculator)")
print("  • Model Registry (ModelRegistry)")
print("  • Pipeline (make_pipeline)")

print("\n" + "=" * 80)
