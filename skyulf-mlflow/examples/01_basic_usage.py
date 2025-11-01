"""Example: Basic usage of Skyulf-MLFlow library with train/test split."""

import pandas as pd
import numpy as np
from skyulf_mlflow_library.features.encoding import OneHotEncoder
from skyulf_mlflow_library.preprocessing import StandardScaler

# Create sample data (more realistic size)
np.random.seed(42)
df = pd.DataFrame({
    'age': np.random.randint(22, 65, 100),
    'salary': np.random.randint(40000, 120000, 100),
    'city': np.random.choice(['NYC', 'LA', 'Chicago', 'Boston'], 100),
    'education': np.random.choice(['BS', 'MS', 'PhD'], 100),
    'target': np.random.choice([0, 1], 100)
})

print("Original Data:")
print(df.head(10))
print(f"Shape: {df.shape}")
print("\n" + "="*80 + "\n")

# IMPORTANT: Split data FIRST before any transformations to prevent data leakage
print("Splitting data into train and test sets...")
train_df = df.sample(frac=0.8, random_state=42)
test_df = df.drop(train_df.index)
print(f"Train size: {train_df.shape[0]}, Test size: {test_df.shape[0]}")
print("\n" + "="*80 + "\n")

# Example 1: One-Hot Encoding
print("Example 1: One-Hot Encoding")
print("-" * 40)

# Fit encoder on training data only
encoder = OneHotEncoder(columns=['city', 'education'])
train_encoded = encoder.fit_transform(train_df.copy())
test_encoded = encoder.transform(test_df.copy())

print("Encoded Training Data:")
print(train_encoded.head())
print(f"\nOriginal columns: {encoder.metadata['feature_names_in']}")
print(f"New columns: {encoder.get_feature_names_out()}")
print(f"\nTrain shape: {train_encoded.shape}, Test shape: {test_encoded.shape}")
print("\n" + "="*80 + "\n")

# Example 2: Scaling
print("Example 2: Standard Scaling")
print("-" * 40)

# Fit scaler on training data only
scaler = StandardScaler(columns=['age', 'salary'])
train_scaled = scaler.fit_transform(train_df.copy())
test_scaled = scaler.transform(test_df.copy())

print("Scaled Training Data (age & salary):")
print(train_scaled[['age', 'salary']].head())
print("\nScaler metadata:")
print(f"Mean: {scaler.metadata['mean']}")
print(f"Variance: {scaler.metadata['var']}")
print(f"\nTrain shape: {train_scaled.shape}, Test shape: {test_scaled.shape}")
print("\n" + "="*80 + "\n")

# Example 3: Complete workflow with model training
print("Example 3: Complete Workflow")
print("-" * 40)
print("Steps:")
print("1. Split data (80/20)")
print("2. Encode categorical features (fit on train)")
print("3. Scale numeric features (fit on train)")
print("4. Train model on processed training data")
print("5. Evaluate on processed test data")
print("\nKey principle: Always fit transformers on training data,")
print("then apply same transformations to test data!")
print("\n" + "="*80 + "\n")
