"""
Example: Feature Math - Advanced Feature Engineering

Demonstrates the powerful FeatureMath transformer for creating
complex features through arithmetic, ratios, statistics, text similarity,
and datetime extraction.
"""

import pandas as pd
import numpy as np
from skyulf_mlflow_library.features.transform import FeatureMath

print("=" * 70)
print("Skyulf-MLFlow: Feature Math - Advanced Feature Engineering")
print("=" * 70)
print()

# ==============================================================================
# 1. Arithmetic Operations
# ==============================================================================
print("1. ARITHMETIC OPERATIONS")
print("-" * 70)

df_arithmetic = pd.DataFrame({
    'price': [100, 200, 150, 300],
    'tax': [10, 20, 15, 30],
    'shipping': [5, 10, 7, 15],
    'discount': [20, 40, 30, 60]
})

print("Original Data:")
print(df_arithmetic)
print()

# Create multiple arithmetic features
operations = [
    # Add price + tax + shipping
    {
        'type': 'arithmetic',
        'method': 'add',
        'columns': ['price', 'tax', 'shipping'],
        'output': 'total_cost'
    },
    # Subtract discount from total
    {
        'type': 'arithmetic',
        'method': 'subtract',
        'columns': ['price'],
        'constants': [20],  # Subtract constant $20
        'output': 'price_after_fixed_discount'
    },
    # Multiply price by quantity (using constant)
    {
        'type': 'arithmetic',
        'method': 'multiply',
        'columns': ['price'],
        'constants': [2],  # Assume quantity of 2
        'output': 'subtotal'
    },
]

fm = FeatureMath(operations=operations)
df_result = fm.fit_transform(df_arithmetic)

print("After Arithmetic Operations:")
print(df_result[['price', 'tax', 'shipping', 'total_cost', 'price_after_fixed_discount', 'subtotal']])
print("\n")

# ==============================================================================
# 2. Ratio Operations
# ==============================================================================
print("2. RATIO OPERATIONS")
print("-" * 70)

df_ratio = pd.DataFrame({
    'revenue': [10000, 15000, 12000, 18000],
    'cost': [7000, 9000, 8000, 11000],
    'marketing_spend': [1000, 1500, 1200, 1800]
})

print("Original Data:")
print(df_ratio)
print()

operations = [
    # Profit margin ratio
    {
        'type': 'ratio',
        'numerator': ['revenue'],
        'denominator': ['cost'],
        'output': 'profit_margin_ratio',
        'round': 2
    },
    # Marketing efficiency (revenue per marketing dollar)
    {
        'type': 'ratio',
        'numerator': ['revenue'],
        'denominator': ['marketing_spend'],
        'output': 'marketing_efficiency',
        'round': 2
    },
]

fm = FeatureMath(operations=operations)
df_result = fm.fit_transform(df_ratio)

print("After Ratio Operations:")
print(df_result[['revenue', 'cost', 'profit_margin_ratio', 'marketing_efficiency']])
print("\n")

# ==============================================================================
# 3. Statistical Aggregations
# ==============================================================================
print("3. STATISTICAL AGGREGATIONS")
print("-" * 70)

df_stats = pd.DataFrame({
    'q1_sales': [100, 150, 120, 180],
    'q2_sales': [110, 160, 130, 190],
    'q3_sales': [120, 170, 140, 200],
    'q4_sales': [130, 180, 150, 210]
})

print("Original Quarterly Data:")
print(df_stats)
print()

operations = [
    # Total annual sales
    {
        'type': 'stat',
        'method': 'sum',
        'columns': ['q1_sales', 'q2_sales', 'q3_sales', 'q4_sales'],
        'output': 'annual_sales'
    },
    # Average quarterly sales
    {
        'type': 'stat',
        'method': 'mean',
        'columns': ['q1_sales', 'q2_sales', 'q3_sales', 'q4_sales'],
        'output': 'avg_quarterly_sales',
        'round': 2
    },
    # Best quarter
    {
        'type': 'stat',
        'method': 'max',
        'columns': ['q1_sales', 'q2_sales', 'q3_sales', 'q4_sales'],
        'output': 'best_quarter_sales'
    },
    # Sales volatility (standard deviation)
    {
        'type': 'stat',
        'method': 'std',
        'columns': ['q1_sales', 'q2_sales', 'q3_sales', 'q4_sales'],
        'output': 'sales_volatility',
        'round': 2
    },
    # Sales range (max - min)
    {
        'type': 'stat',
        'method': 'range',
        'columns': ['q1_sales', 'q2_sales', 'q3_sales', 'q4_sales'],
        'output': 'sales_range'
    },
]

fm = FeatureMath(operations=operations)
df_result = fm.fit_transform(df_stats)

print("After Statistical Aggregations:")
print(df_result[['annual_sales', 'avg_quarterly_sales', 'best_quarter_sales', 'sales_volatility', 'sales_range']])
print("\n")

# ==============================================================================
# 4. Text Similarity Features
# ==============================================================================
print("4. TEXT SIMILARITY FEATURES")
print("-" * 70)

df_text = pd.DataFrame({
    'product_name': ['iPhone 13 Pro', 'Samsung Galaxy S21', 'Google Pixel 6'],
    'description': ['iPhone 13 Pro Max', 'Samsung S21 Ultra', 'Pixel 6 Pro']
})

print("Original Data:")
print(df_text)
print()

operations = [
    {
        'type': 'similarity',
        'method': 'ratio',
        'columns': ['product_name', 'description'],
        'output': 'name_desc_similarity',
        'round': 1
    },
]

fm = FeatureMath(operations=operations)
df_result = fm.fit_transform(df_text)

print("After Similarity Computation:")
print(df_result[['product_name', 'description', 'name_desc_similarity']])
print("\nNote: Similarity scores range from 0-100 (higher = more similar)")
print("Install 'rapidfuzz' for faster similarity computation: pip install rapidfuzz")
print("\n")

# ==============================================================================
# 5. DateTime Feature Extraction
# ==============================================================================
print("5. DATETIME FEATURE EXTRACTION")
print("-" * 70)

df_datetime = pd.DataFrame({
    'order_date': pd.to_datetime([
        '2024-01-15 09:30:00',
        '2024-06-20 14:45:00',
        '2024-12-25 18:00:00',
        '2024-09-10 22:30:00'
    ]),
    'ship_date': pd.to_datetime([
        '2024-01-17',
        '2024-06-22',
        '2024-12-27',
        '2024-09-12'
    ])
})

print("Original Data:")
print(df_datetime)
print()

operations = [
    {
        'type': 'datetime',
        'columns': ['order_date'],
        'features': ['year', 'quarter', 'month', 'month_name', 'day', 
                    'weekday', 'is_weekend', 'hour', 'season', 'time_of_day'],
        'prefix': 'order_'
    },
]

fm = FeatureMath(operations=operations)
df_result = fm.fit_transform(df_datetime)

print("After DateTime Feature Extraction:")
datetime_features = [col for col in df_result.columns if col.startswith('order_')]
print(df_result[['order_date'] + datetime_features])
print("\n")

# ==============================================================================
# 6. Complex Multi-Operation Workflow
# ==============================================================================
print("6. COMPLEX MULTI-OPERATION WORKFLOW")
print("-" * 70)

# Create realistic e-commerce dataset
df_ecommerce = pd.DataFrame({
    'product_id': [101, 102, 103, 104],
    'product_name': ['Laptop Pro', 'Phone X', 'Tablet Air', 'Watch Sport'],
    'category': ['Electronics', 'Electronics', 'Electronics', 'Electronics'],
    'base_price': [1200, 800, 500, 300],
    'cost': [900, 600, 350, 200],
    'quantity_sold': [50, 150, 100, 200],
    'returns': [5, 10, 8, 15],
    'order_date': pd.to_datetime(['2024-01-15', '2024-03-20', '2024-06-10', '2024-09-05'])
})

print("Original E-commerce Data:")
print(df_ecommerce[['product_name', 'base_price', 'cost', 'quantity_sold', 'returns']])
print()

# Define comprehensive feature engineering pipeline
operations = [
    # Calculate revenue
    {
        'type': 'arithmetic',
        'method': 'multiply',
        'columns': ['base_price', 'quantity_sold'],
        'output': 'revenue'
    },
    # Calculate total cost
    {
        'type': 'arithmetic',
        'method': 'multiply',
        'columns': ['cost', 'quantity_sold'],
        'output': 'total_cost'
    },
    # Calculate profit
    {
        'type': 'arithmetic',
        'method': 'subtract',
        'columns': ['base_price', 'cost'],
        'output': 'profit_per_unit'
    },
    # Profit margin percentage
    {
        'type': 'ratio',
        'numerator': ['base_price'],
        'denominator': ['cost'],
        'output': 'markup_ratio',
        'round': 2
    },
    # Return rate
    {
        'type': 'ratio',
        'numerator': ['returns'],
        'denominator': ['quantity_sold'],
        'output': 'return_rate',
        'round': 3
    },
    # Extract order month and quarter for seasonality
    {
        'type': 'datetime',
        'columns': ['order_date'],
        'features': ['quarter', 'month', 'season'],
        'prefix': ''
    },
]

fm = FeatureMath(operations=operations, error_handling='skip')
df_result = fm.fit_transform(df_ecommerce)

print("After Feature Engineering:")
feature_cols = ['product_name', 'revenue', 'total_cost', 'profit_per_unit', 
                'markup_ratio', 'return_rate', 'quarter', 'season']
print(df_result[feature_cols])
print("\n")

# ==============================================================================
# 7. Error Handling
# ==============================================================================
print("7. ERROR HANDLING")
print("-" * 70)

df_errors = pd.DataFrame({
    'value1': [10, 20, 30],
    'value2': [5, 0, 10],  # Contains zero for division
})

print("Original Data (with potential division by zero):")
print(df_errors)
print()

# With error_handling='skip' (default), errors are warned but don't stop execution
operations = [
    {
        'type': 'arithmetic',
        'method': 'divide',
        'columns': ['value1', 'value2'],
        'output': 'ratio'
    },
]

fm = FeatureMath(operations=operations, error_handling='skip')
df_result = fm.fit_transform(df_errors)

print("After Division (epsilon prevents true zero division):")
print(df_result)
print("\nNote: Division by zero is handled using epsilon (1e-9) to prevent errors")
print("\n")

# ==============================================================================
# Summary
# ==============================================================================
print("=" * 70)
print("✅ Feature Math Demo Complete!")
print("=" * 70)
print()
print("Key Capabilities:")
print("1. Arithmetic: add, subtract, multiply, divide")
print("2. Ratio: numerator/denominator calculations")
print("3. Statistics: sum, mean, min, max, std, median, count, range")
print("4. Similarity: text matching with rapidfuzz or difflib fallback")
print("5. DateTime: 15+ temporal features (year, quarter, season, etc.)")
print()
print("Advantages:")
print("✓ Define multiple operations in one transformer")
print("✓ Batch process complex feature engineering")
print("✓ Consistent API across all operation types")
print("✓ Built-in error handling and validation")
print("✓ Division by zero protection")
print("✓ Flexible rounding and fillna options")
print()
print("This is a UNIQUE capability not found in other ML libraries!")
