"""
Example: Smart Binning Transformer with Train/Test Split

This example demonstrates the SmartBinning transformer for discretizing
continuous features into bins using various strategies.

The SmartBinning transformer supports:
- Equal-width binning: bins with equal ranges
- Equal-frequency binning: bins with equal number of samples
- K-Bins discretization: kmeans clustering-based binning
- Custom binning: user-defined bin edges

IMPORTANT: Fit on training data, transform on test data to prevent leakage!

Author: Skyulf MLflow Library
"""

import pandas as pd
import numpy as np
from skyulf_mlflow_library.features import SmartBinning

# Set random seed for reproducibility
np.random.seed(42)

print("=" * 80)
print("SKYULF MLFLOW - SMART BINNING EXAMPLES")
print("=" * 80)
print()

# ============================================================================
# Example 1: Equal-Width Binning with Train/Test Split
# ============================================================================
print("\n" + "=" * 80)
print("Example 1: Equal-Width Binning (Train/Test Split)")
print("=" * 80)

# Create sample data with different distributions
df1 = pd.DataFrame({
    'age': np.random.randint(18, 80, 200),
    'salary': np.random.uniform(30000, 150000, 200),
    'experience': np.random.randint(0, 40, 200),
    'target': np.random.choice([0, 1], 200)
})

print("\nOriginal data (first 10 rows):")
print(df1.head(10))
print(f"\nData shape: {df1.shape}")

# Split data first
train_df1 = df1.sample(frac=0.8, random_state=42)
test_df1 = df1.drop(train_df1.index)
print(f"\nTrain: {train_df1.shape}, Test: {test_df1.shape}")

# Fit binning on train data only
binning = SmartBinning(
    columns=['age', 'salary', 'experience'],
    n_bins=5,
    strategy='equal_width',
    labels=['Very Low', 'Low', 'Medium', 'High', 'Very High']
)

train_binned = binning.fit_transform(train_df1.copy())
test_binned = binning.transform(test_df1.copy())

print("\nBinned training data (first 5 rows):")
print(train_binned.head(5))

print("\nBin edges learned from training data:")
for col in ['age', 'salary', 'experience']:
    edges = binning.get_bin_edges(col)
    print(f"{col}: {[f'{e:.1f}' for e in edges]}")

print("\nValue counts in training set:")
for col in ['age', 'salary', 'experience']:
    print(f"\n{col}_binned:")
    print(train_binned[f'{col}_binned'].value_counts().sort_index())

print(f"\nTest set shape: {test_binned.shape}")
print("(Test data binned using training data's edges)")

# ============================================================================
# Example 2: Equal-Frequency Binning with Train/Test Split
# ============================================================================
print("\n" + "=" * 80)
print("Example 2: Equal-Frequency Binning (Train/Test Split)")
print("=" * 80)

# Create data with skewed distribution
df2 = pd.DataFrame({
    'income': np.random.exponential(50000, 200),  # Exponential distribution
    'score': np.random.beta(2, 5, 200) * 100,     # Beta distribution
    'target': np.random.choice([0, 1], 200)
})

print("\nOriginal data statistics:")
print(df2[['income', 'score']].describe())

# Split data
train_df2 = df2.sample(frac=0.8, random_state=42)
test_df2 = df2.drop(train_df2.index)
print(f"\nTrain: {train_df2.shape}, Test: {test_df2.shape}")

# Fit equal-frequency binning on train data
binning_freq = SmartBinning(
    columns=['income', 'score'],
    n_bins=4,
    strategy='equal_frequency',
    labels=['Q1', 'Q2', 'Q3', 'Q4']
)

train_binned2 = binning_freq.fit_transform(train_df2.copy())
test_binned2 = binning_freq.transform(test_df2.copy())

print("\nBinned training data (first 5 rows):")
print(train_binned2.head(5))

print("\nBin edges learned from training quantiles:")
for col in ['income', 'score']:
    edges = binning_freq.get_bin_edges(col)
    print(f"{col}: {[f'{e:.1f}' for e in edges]}")

print("\nValue counts in training set (should be roughly equal):")
for col in ['income', 'score']:
    print(f"\n{col}_binned:")
    print(train_binned2[f'{col}_binned'].value_counts().sort_index())

print(f"\nTest set shape: {test_binned2.shape}")

# ============================================================================
# Example 3: Custom Binning with User-Defined Edges
# ============================================================================
print("\n" + "=" * 80)
print("Example 3: Custom Binning with User-Defined Edges")
print("=" * 80)

# Create sample data
df3 = pd.DataFrame({
    'temperature': np.random.uniform(-10, 45, 100),
    'humidity': np.random.uniform(0, 100, 100)
})

print("\nOriginal data (first 10 rows):")
print(df3.head(10))

# Define custom bin edges based on domain knowledge
custom_bins = {
    'temperature': [-np.inf, 0, 10, 20, 30, np.inf],  # Freezing, Cold, Mild, Warm, Hot
    'humidity': [0, 30, 60, 100]  # Dry, Normal, Humid
}

custom_labels = {
    'temperature': ['Freezing', 'Cold', 'Mild', 'Warm', 'Hot'],
    'humidity': ['Dry', 'Normal', 'Humid']
}

binning_custom = SmartBinning(
    columns=['temperature', 'humidity'],
    strategy='custom',
    bins=custom_bins,
    labels=custom_labels
)

df3_binned = binning_custom.fit_transform(df3)

print("\nBinned data (first 10 rows):")
print(df3_binned.head(10))

print("\nValue counts for each binned column:")
for col in ['temperature', 'humidity']:
    print(f"\n{col}_binned:")
    print(df3_binned[f'{col}_binned'].value_counts().sort_index())

# ============================================================================
# Example 4: KBins Discretization with Sklearn Backend
# ============================================================================
print("\n" + "=" * 80)
print("Example 4: KBins Discretization (Sklearn Backend)")
print("=" * 80)

# Create sample data
df4 = pd.DataFrame({
    'feature1': np.random.normal(100, 15, 100),
    'feature2': np.random.gamma(2, 2, 100),
    'feature3': np.random.uniform(0, 1, 100)
})

print("\nOriginal data statistics:")
print(df4.describe())

# Apply KBins discretization with quantile strategy
binning_kbins = SmartBinning(
    columns=['feature1', 'feature2', 'feature3'],
    n_bins=5,
    strategy='kbins',
    kbins_encode='ordinal',  # or 'onehot', 'onehot-dense'
    kbins_strategy='quantile'  # or 'uniform', 'kmeans'
)

df4_binned = binning_kbins.fit_transform(df4)

print("\nBinned data (first 10 rows):")
print(df4_binned.head(10))

print("\nBin value counts:")
for col in ['feature1', 'feature2', 'feature3']:
    print(f"\n{col}_binned:")
    print(df4_binned[f'{col}_binned'].value_counts().sort_index())

# ============================================================================
# Example 5: One-Hot Encoding of Bins
# ============================================================================
print("\n" + "=" * 80)
print("Example 5: One-Hot Encoding of Bins")
print("=" * 80)

# Create sample data
df5 = pd.DataFrame({
    'price': np.random.uniform(100, 1000, 50),
    'rating': np.random.uniform(1, 5, 50)
})

print("\nOriginal data (first 10 rows):")
print(df5.head(10))

# Apply binning with one-hot encoding using kbins strategy
binning_onehot = SmartBinning(
    columns=['price', 'rating'],
    n_bins=3,
    strategy='kbins',
    kbins_strategy='uniform',
    kbins_encode='onehot-dense',  # One-hot encoding
    drop_original=True  # Drop original continuous columns
)

df5_binned = binning_onehot.fit_transform(df5)

print("\nOne-hot encoded binned data (first 10 rows):")
print(df5_binned.head(10))

print(f"\nResulting columns: {list(df5_binned.columns)}")
print(f"Shape: {df5_binned.shape}")

# ============================================================================
# Example 6: Real-World Use Case - Customer Segmentation
# ============================================================================
print("\n" + "=" * 80)
print("Example 6: Real-World Use Case - Customer Segmentation")
print("=" * 80)

# Create realistic customer data
np.random.seed(123)
df_customers = pd.DataFrame({
    'age': np.random.randint(18, 75, 200),
    'annual_income': np.random.lognormal(10.8, 0.5, 200),  # Realistic income distribution
    'purchase_frequency': np.random.poisson(12, 200),  # Purchases per year
    'avg_purchase_value': np.random.gamma(2, 50, 200),  # Average order value
    'customer_tenure_months': np.random.randint(1, 120, 200)
})

print("\nCustomer data statistics:")
print(df_customers.describe())

# Define business-driven binning strategies
age_bins = [0, 25, 35, 50, 65, 100]
age_labels = ['Gen Z', 'Millennial', 'Gen X', 'Boomer', 'Senior']

income_bins = [0, 30000, 60000, 100000, np.inf]
income_labels = ['Low Income', 'Middle Income', 'Upper Middle', 'High Income']

# Apply custom binning for segmentation
segmentation_binning = SmartBinning(
    columns=['age', 'annual_income'],
    strategy='custom',
    bins={
        'age': age_bins,
        'annual_income': income_bins
    },
    labels={
        'age': age_labels,
        'annual_income': income_labels
    }
)

# Apply equal-frequency binning for behavioral features
behavior_binning = SmartBinning(
    columns=['purchase_frequency', 'avg_purchase_value', 'customer_tenure_months'],
    n_bins=4,
    strategy='equal_frequency',
    labels=['Low', 'Medium', 'High', 'Very High']
)

# Transform data
df_segmented = segmentation_binning.fit_transform(df_customers)
df_segmented = behavior_binning.fit_transform(df_segmented)

print("\nSegmented customer data (first 10 rows):")
print(df_segmented.head(10))

print("\nCustomer segment distribution:")
print("\nAge segments:")
print(df_segmented['age_binned'].value_counts())

print("\nIncome segments:")
print(df_segmented['annual_income_binned'].value_counts())

print("\nPurchase frequency segments:")
print(df_segmented['purchase_frequency_binned'].value_counts())

# Analyze high-value customers
print("\nHigh-value customer profile:")
high_value_mask = (
    (df_segmented['annual_income_binned'].isin(['Upper Middle', 'High Income'])) &
    (df_segmented['purchase_frequency_binned'] == 'Very High')
)
print(f"Number of high-value customers: {high_value_mask.sum()}")
print(f"Percentage of total: {high_value_mask.sum() / len(df_segmented) * 100:.1f}%")

# ============================================================================
# Example 7: Comparing Binning Strategies
# ============================================================================
print("\n" + "=" * 80)
print("Example 7: Comparing Binning Strategies")
print("=" * 80)

# Create data with mixed distributions
df_compare = pd.DataFrame({
    'feature': np.concatenate([
        np.random.normal(20, 5, 50),
        np.random.normal(50, 5, 50),
        np.random.normal(80, 5, 50)
    ])
})

print("\nOriginal feature statistics:")
print(df_compare['feature'].describe())

strategies = ['equal_width', 'equal_frequency', 'kbins']
results = {}

for strategy in strategies:
    if strategy == 'kbins':
        binning = SmartBinning(
            columns=['feature'],
            n_bins=5,
            strategy=strategy,
            kbins_strategy='quantile'
        )
    else:
        binning = SmartBinning(
            columns=['feature'],
            n_bins=5,
            strategy=strategy
        )
    
    result = binning.fit_transform(df_compare)
    results[strategy] = result['feature_binned'].value_counts().sort_index()
    
    print(f"\n{strategy.upper()} strategy:")
    print(f"Bin edges: {binning.get_bin_edges('feature')}")
    print(f"Value counts:\n{results[strategy]}")

print("\n" + "=" * 80)
print("EXAMPLES COMPLETED SUCCESSFULLY!")
print("=" * 80)
print("\nKey Takeaways:")
print("- Equal-width: Good for uniform distributions, easy interpretation")
print("- Equal-frequency: Handles skewed data well, balanced bin sizes")
print("- Custom: Domain knowledge-driven, business-aligned bins")
print("- KBins: Flexible sklearn backend with multiple strategies")
print("- One-hot encoding: Ready for ML models")
print("=" * 80)
