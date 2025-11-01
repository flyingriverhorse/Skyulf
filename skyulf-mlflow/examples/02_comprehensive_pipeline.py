"""Comprehensive example showcasing all library features with proper train/test split."""

import pandas as pd
import numpy as np

print("="*80)
print("Skyulf-MLFlow Library - Comprehensive Example")
print("="*80)
print()

# Create larger sample data with various issues
np.random.seed(42)
n_samples = 100
df = pd.DataFrame({
    'age': np.concatenate([np.random.randint(22, 70, 95), [120, 18, np.nan, np.nan, 125]]),  # Has missing and outliers
    'salary': np.concatenate([np.random.randint(40000, 120000, 96), [np.nan, np.nan, 35000, 150000]]),
    'city': np.random.choice(['NYC', 'LA', 'Chicago', 'Boston', 'Seattle'], n_samples),
    'education': np.random.choice(['BS', 'MS', 'PhD'], n_samples),
    'department': np.random.choice(['Sales', 'Tech', 'Marketing', 'Finance'], n_samples),
    'target': np.random.choice([0, 1], n_samples)
})

# Add some duplicate rows
df = pd.concat([df, df.iloc[[0, 1]]], ignore_index=True)

print("Original Data (with missing values, duplicates, and outliers):")
print(df.head(10))
print(f"\nShape: {df.shape}")
print(f"Missing values:\n{df.isna().sum()}")
print(f"Duplicates: {df.duplicated().sum()}")
print("\n" + "="*80 + "\n")

# ============================================================================
# Step 1: Data Cleaning (BEFORE SPLIT)
# ============================================================================
print("STEP 1: Data Cleaning (Before Split)")
print("-" * 40)

from skyulf_mlflow_library.preprocessing import (
    remove_duplicates,
    remove_outliers,
)

# Remove duplicates first
df_clean = remove_duplicates(df, keep='first')
print(f"After removing duplicates: {df_clean.shape}")

# Remove outliers from age column
df_clean = remove_outliers(df_clean, columns=['age'], method='iqr', threshold=1.5)
print(f"After removing outliers: {df_clean.shape}")
print("\n" + "="*80 + "\n")

# ============================================================================
# CRITICAL: Split data BEFORE any fitting operations
# ============================================================================
print("SPLIT DATA (80/20 train/test)")
print("-" * 40)

train_df = df_clean.sample(frac=0.8, random_state=42)
test_df = df_clean.drop(train_df.index)

print(f"Training set: {train_df.shape}")
print(f"Test set: {test_df.shape}")
print(f"\nTrain missing values:\n{train_df[['age', 'salary']].isna().sum()}")
print(f"\nTest missing values:\n{test_df[['age', 'salary']].isna().sum()}")
print("\n" + "="*80 + "\n")

# ============================================================================
# Step 2: Imputation - Handle Missing Values
# ============================================================================
print("STEP 2: Missing Value Imputation (fit on train, transform on both)")
print("-" * 40)

from skyulf_mlflow_library.preprocessing import SimpleImputer

# Fit imputer on training data ONLY
imputer = SimpleImputer(columns=['age', 'salary'], strategy='mean')
train_imputed = imputer.fit_transform(train_df.copy())
test_imputed = imputer.transform(test_df.copy())

print("After imputation:")
print(f"Train missing values: {train_imputed[['age', 'salary']].isna().sum().sum()}")
print(f"Test missing values: {test_imputed[['age', 'salary']].isna().sum().sum()}")
print(f"\nImputation statistics (learned from training data):")
print(f"{imputer.metadata['statistics']}")
print("\n" + "="*80 + "\n")

# ============================================================================
# Step 3: Encoding - Transform Categorical Variables
# ============================================================================
print("STEP 3: Categorical Encoding (fit on train, transform on both)")
print("-" * 40)

from skyulf_mlflow_library.features.encoding import OneHotEncoder, LabelEncoder

# One-hot encode city and education - fit on train only
onehot_encoder = OneHotEncoder(columns=['city', 'education'])
train_encoded = onehot_encoder.fit_transform(train_imputed)
test_encoded = onehot_encoder.transform(test_imputed)

print("After one-hot encoding:")
print(f"New columns: {onehot_encoder.get_feature_names_out()}")
print(f"Train shape: {train_encoded.shape}, Test shape: {test_encoded.shape}")
print()

# Label encode department - fit on train only
label_encoder = LabelEncoder(columns=['department'])
train_encoded = label_encoder.fit_transform(train_encoded)
test_encoded = label_encoder.transform(test_encoded)

print("After label encoding:")
print(f"Department classes (from train): {label_encoder.get_classes('department')}")
print(f"Train shape: {train_encoded.shape}, Test shape: {test_encoded.shape}")
print("\n" + "="*80 + "\n")

# ============================================================================
# Step 4: Scaling - Normalize Numeric Features
# ============================================================================
print("STEP 4: Feature Scaling (fit on train, transform on both)")
print("-" * 40)

from skyulf_mlflow_library.preprocessing import MinMaxScaler, StandardScaler

# Standard scaling for age - fit on train only
standard_scaler = StandardScaler(columns=['age'])
train_scaled = standard_scaler.fit_transform(train_encoded)
test_scaled = standard_scaler.transform(test_encoded)

print("After standard scaling (age):")
print(f"Mean (from train): {standard_scaler.metadata['mean']}")
print(f"Std (from train): {[np.sqrt(v) for v in standard_scaler.metadata['var']]}")
print(f"Train sample:\n{train_scaled['age'].head(3)}")
print(f"Test sample:\n{test_scaled['age'].head(3)}")
print()

# MinMax scaling for salary - fit on train only
minmax_scaler = MinMaxScaler(columns=['salary'], feature_range=(0, 1))
train_scaled = minmax_scaler.fit_transform(train_scaled)
test_scaled = minmax_scaler.transform(test_scaled)

print("After MinMax scaling (salary):")
print(f"Range: {minmax_scaler.metadata['feature_range']}")
print(f"Min (from train): {minmax_scaler.metadata['data_min']}")
print(f"Max (from train): {minmax_scaler.metadata['data_max']}")
print(f"Train sample:\n{train_scaled['salary'].head(3)}")
print(f"Test sample:\n{test_scaled['salary'].head(3)}")
print("\n" + "="*80 + "\n")

# ============================================================================
# Final Result
# ============================================================================
print("FINAL RESULT")
print("-" * 40)
print(f"Training set: {train_scaled.shape}")
print(f"Test set: {test_scaled.shape}")
print(f"\nFinal columns: {train_scaled.columns.tolist()[:10]}... ({len(train_scaled.columns)} total)")
print(f"\nFirst 3 rows of training data:")
print(train_scaled.head(3))
print(f"\nFirst 3 rows of test data:")
print(test_scaled.head(3))
print("\n" + "="*80 + "\n")

# ============================================================================
# Summary
# ============================================================================
print("PROCESSING SUMMARY")
print("-" * 40)
print(f"✅ Original data: {df.shape}")
print(f"✅ After cleaning (dedup + outliers): {df_clean.shape}")
print(f"✅ Train/test split: {train_df.shape[0]} / {test_df.shape[0]}")
print(f"✅ Missing values imputed: {imputer.metadata['n_features']} columns")
print(f"✅ One-hot encoded: {len(onehot_encoder.metadata['feature_names_in'])} columns → {len(onehot_encoder.get_feature_names_out())} features")
print(f"✅ Label encoded: {len(label_encoder.metadata['feature_names_in'])} column")
print(f"✅ Scaled features: 2 columns (StandardScaler + MinMaxScaler)")
print(f"✅ Final train/test: {train_scaled.shape} / {test_scaled.shape}")
print("\n🎯 KEY PRINCIPLE: Fit all transformers on training data,")
print("   then apply same transformations to test data!")
print("\n" + "="*80 + "\n")

print("SUCCESS! All transformations completed. ✨")
print("\nThe library provides:")
print("  • Clean, scikit-learn compatible API")
print("  • Automatic metadata tracking")
print("  • Comprehensive error handling")
print("  • Type-safe operations")
print("  • Production-ready code")
print("\nReady for your ML workflows! 🚀")
