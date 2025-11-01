# 🚀 Skyulf-MLFlow-Library

[![Python Version](https://img.shields.io/badge/python-3.8%2B-blue.svg)](https://www.python.org/downloads/)
[![License](https://img.shields.io/badge/license-MIT-green.svg)](LICENSE)
[![Tests](https://img.shields.io/badge/tests-133%20total%20(132%20passing%2C%201%20skipped)-brightgreen.svg)]()
[![Coverage](https://img.shields.io/badge/coverage-56%25-yellow.svg)]()

> **My personal ML toolkit built from years of frustration with scattered libraries and repetitive code**

Hey there! This library started because I got tired of copy-pasting the same preprocessing code across every ML project. You know the drill - load data, handle missing values, encode categories, scale features, train model, save it somewhere... rinse and repeat.

So I built something better. **Skyulf-MLFlow-Library** pulls together the best parts of scikit-learn, pandas, and imbalanced-learn into one clean API. Think of it as the ML utility belt I wish I'd had when I started.

## 💡 Why I Built This

Look, I've been there. You're building your 10th customer churn model and you're writing the same damn preprocessing pipeline again. Or you're trying to remember if you used `OneHotEncoder` or `get_dummies` in that project from six months ago. Or worse - your model works great in notebooks but completely fails in production because the preprocessing doesn't match.

This library solves those headaches:
- **No more repetitive boilerplate** - Write it once, use it everywhere
- **Consistent patterns** - Same API whether you're imputing, encoding, or engineering features  
- **Production-ready from day one** - Bundle transformers with models to avoid train/serve skew
- **Smart defaults** - Things just work without tweaking 20 parameters

I work on this nights and weekends when I hit something annoying in my ML work and think "there's gotta be a better way." If it helps you too, awesome! Pull requests always welcome.

### 🎯 What's Actually Useful Here?

**Real automation that saves time:**
- Domain detection that actually understands your data (e-commerce vs healthcare vs finance)
- Automatically suggests what features to engineer based on data type
- Quality reports that tell you what's broken before your model does

**One API to rule them all:**
- Everything follows the same `.fit()`, `.transform()` pattern (because muscle memory is real)
- Pass DataFrames in, get DataFrames out (no more "wait, is this a numpy array or pandas?")
- Chain operations without fighting type compatibility

**Features I actually use in production:**
- Create ratios, aggregations, and datetime features with simple config instead of custom functions
- Handle high-cardinality categories without blowing up memory (target encoding, hashing)
- Pick the best features automatically from 9 different selection methods
- Save entire pipelines (not just models) so preprocessing works the same in production

**Not vaporware:**
- 133 tests covering the important stuff
- Type hints everywhere so your IDE actually helps
- Been running in real projects for months
- Decent error messages when things go wrong

---

## ✨ Features

### 🎯 **Smart Data Understanding** (Unique!)
- **Domain Analyzer**: Automatically detects if your data is e-commerce, healthcare, finance, time-series, etc.
- **Text Insights**: Analyzes text columns and provides recommendations
- **Quality Reports**: Generates comprehensive data quality assessments
- **EDA Automation**: Smart exploratory data analysis based on domain

### 🔧 **Advanced Feature Engineering**
- **FeatureMath**: Create complex features with operations like:
  ```python
  {'type': 'arithmetic', 'method': 'add', 'columns': ['price', 'tax'], 'output': 'total'}
  {'type': 'ratio', 'numerator': ['sales'], 'denominator': ['cost'], 'output': 'profit_margin'}
  {'type': 'datetime', 'method': 'extract_hour', 'columns': ['timestamp']}
  ```
- **SmartBinning**: Multiple strategies (equal_width, equal_frequency, kmeans, custom)
- **Feature Selection**: 9+ methods (SelectKBest, RFE, variance threshold, model-based)
- **Polynomial Features**: Automatic interaction and polynomial term generation

### 🏷️ **Comprehensive Encoding**
- One-Hot Encoding (with sparse support)
- Label Encoding (with unseen category handling)
- Ordinal Encoding (preserves order)
- Target Encoding (for high-cardinality features)
- Hash Encoding (for memory efficiency)

### 🧹 **Preprocessing Made Easy**
- **Imputation**: Mean, median, mode, constant, forward-fill, backward-fill
- **Scaling**: Standard, MinMax, Robust, MaxAbs
- **Cleaning**: Drop missing (rows/columns), remove duplicates, outlier detection
- **Sampling**: SMOTE, over-sampling, under-sampling (requires imbalanced-learn)

### 📊 **Model Management** (Production-Ready!)
- **Model Registry**: SQLite-based versioning system
- **Metrics Calculator**: Comprehensive evaluation for classification & regression
- **Metadata Tracking**: Store hyperparameters, tags, descriptions
- **Easy Deployment**: Load models by name or version

### 💾 **Data Ingestion**
- Multi-format support: CSV, Excel, JSON, Parquet, SQL
- Automatic format detection
- Consistent save/load interface
- Built-in error handling

---

## 📦 Installation

### From Source (Current)
```bash
# Clone the repository
git clone https://github.com/flyingriverhorse/skyulf_mlflow_library.git
cd skyulf_mlflow_library

# Install in development mode
pip install -e .

# Or install with all optional dependencies
pip install -e ".[all]"
```

### PyPI Installation (Coming Soon)
```bash
# Basic installation
pip install skyulf-mlflow

# With optional dependencies
pip install skyulf-mlflow[all]        # Everything
pip install skyulf-mlflow[sampling]   # SMOTE and imbalanced-learn
pip install skyulf-mlflow[sql]        # SQL database support
pip install skyulf-mlflow[excel]      # Excel file support
pip install skyulf-mlflow[parquet]    # Parquet file support
```

### Requirements
- Python 3.8+
- pandas >= 1.5.0
- numpy >= 1.21.0
- scikit-learn >= 1.0.0
- scipy >= 1.7.0

---

## 🚀 Complete Tutorial

### End-to-End Example: Predict Customer Churn

Here's a real workflow from messy data to trained model. Follow along:

```python
import pandas as pd
import numpy as np
from skyulf_mlflow_library.preprocessing import SimpleImputer, StandardScaler
from skyulf_mlflow_library.features.encoding import OneHotEncoder
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, classification_report

# Step 1: Load your data (usually from CSV, here's a sample)
df = pd.DataFrame({
    'customer_id': range(1, 101),
    'age': np.random.randint(18, 70, 100),
    'monthly_spend': np.random.uniform(20, 200, 100),
    'tenure_months': np.random.randint(1, 60, 100),
    'city': np.random.choice(['NYC', 'LA', 'SF', 'Chicago', 'Boston'], 100),
    'support_calls': np.random.randint(0, 10, 100),
    'churned': np.random.randint(0, 2, 100)
})

# Add some missing values (real data is messy!)
df.loc[df.sample(frac=0.1).index, 'age'] = None
df.loc[df.sample(frac=0.1).index, 'monthly_spend'] = None

print(f"Dataset: {len(df)} customers, {df['churned'].sum()} churned")
print(f"Missing values: {df.isnull().sum().sum()}")

# Step 2: Split into train/test BEFORE any preprocessing
# (This prevents data leakage)
train_df = df.sample(frac=0.8, random_state=42)
test_df = df.drop(train_df.index)

print(f"\nTrain set: {len(train_df)} samples")
print(f"Test set: {len(test_df)} samples")

# Step 3: Handle missing data (fit on train, apply to both)
imputer = SimpleImputer(strategy='mean')
train_df = imputer.fit_transform(train_df)  # Learn from training data
test_df = imputer.transform(test_df)        # Apply same transformation

# Step 4: Scale numeric features (fit on train, apply to both)
scaler = StandardScaler(columns=['age', 'monthly_spend', 'tenure_months', 'support_calls'])
train_df = scaler.fit_transform(train_df)
test_df = scaler.transform(test_df)

# Step 5: Encode categorical features (fit on train, apply to both)
encoder = OneHotEncoder(columns=['city'], drop_first=True)
train_df = encoder.fit_transform(train_df)
test_df = encoder.transform(test_df)

# Step 6: Prepare X and y for training
X_train = train_df.drop(['churned', 'customer_id'], axis=1)
y_train = train_df['churned']
X_test = test_df.drop(['churned', 'customer_id'], axis=1)
y_test = test_df['churned']

print(f"\nFeatures after preprocessing: {X_train.columns.tolist()}")
print(f"Shape: X_train={X_train.shape}, X_test={X_test.shape}")

# Step 7: Train the model
model = RandomForestClassifier(n_estimators=100, max_depth=10, random_state=42)
model.fit(X_train, y_train)

# Step 8: Evaluate
y_pred_train = model.predict(X_train)
y_pred_test = model.predict(X_test)

print(f"\n🎯 Results:")
print(f"Training accuracy: {accuracy_score(y_train, y_pred_train):.1%}")
print(f"Test accuracy: {accuracy_score(y_test, y_pred_test):.1%}")
print(f"\nTest Set Performance:")
print(classification_report(y_test, y_pred_test, target_names=['Stayed', 'Churned']))
```

**Key points:**
1. **Split first, transform second** - Prevents data leakage
2. **Fit on train, transform on test** - Same transformations in production
3. **DataFrames throughout** - No fighting with numpy arrays vs pandas
4. **One consistent API** - Everything uses `.fit()` and `.transform()`

### Understand Your Data Automatically

Ever wonder what type of dataset you're working with? Is it e-commerce? Healthcare? Finance? The domain analyzer figures it out:

```python
from skyulf_mlflow_library.eda import DomainAnalyzer

# Got a dataset? Let's see what it actually is
df = pd.read_csv('mystery_data.csv')
analyzer = DomainAnalyzer()
result = analyzer.analyze(df)

print(f"This looks like: {result.primary_domain}")
print(f"Confidence: {result.confidence_score:.0%}")
print(f"\nYou should probably:")
for tip in result.recommendations[:3]:
    print(f"  - {tip}")

# Real output example:
# This looks like: e-commerce
# Confidence: 85%
# You should probably:
#   - Segment customers by purchase frequency
#   - Look at seasonal trends in sales
#   - Create features for cart abandonment rate
```

**Why this is useful:** Instead of spending an hour exploring the data, you get instant insights about what kind of problem you're solving and what features might help.

### Create Features the Smart Way

Tired of writing functions to calculate ratios? Or extracting hour from datetime? Just describe what you want:

```python
from skyulf_mlflow_library.features.transform import FeatureMath

# Say you've got e-commerce data
df = pd.DataFrame({
    'order_id': [1, 2, 3, 4, 5],
    'revenue': [100, 150, 200, 180, 220],
    'cost': [60, 80, 120, 100, 140],
    'num_items': [2, 3, 5, 4, 6],
    'order_date': pd.to_datetime(['2024-01-15 09:30', '2024-01-15 14:20', 
                                   '2024-01-16 10:15', '2024-01-16 16:45',
                                   '2024-01-17 11:00'])
})

# Create useful features with simple config
operations = [
    # Profit margin (revenue / cost)
    {
        'type': 'ratio',
        'numerator': ['revenue'],
        'denominator': ['cost'],
        'output': 'profit_margin'
    },
    # Average price per item
    {
        'type': 'ratio',
        'numerator': ['revenue'],
        'denominator': ['num_items'],
        'output': 'avg_item_price'
    },
    # What hour do orders happen?
    {
        'type': 'datetime',
        'method': 'extract_hour',
        'columns': ['order_date'],
        'output': 'order_hour'
    },
    # What day of week?
    {
        'type': 'datetime',
        'method': 'extract_dayofweek',
        'columns': ['order_date'],
        'output': 'order_weekday'
    }
]

fm = FeatureMath(operations=operations)
df_enhanced = fm.fit_transform(df)

print(df_enhanced[['revenue', 'cost', 'profit_margin', 'order_hour']].head())
#   revenue  cost  profit_margin  order_hour
# 0      100    60           1.67           9
# 1      150    80           1.88          14
# 2      200   120           1.67          10
# 3      180   100           1.80          16
# 4      220   140           1.57          11
```

**Why this matters:** Creating derived features usually means writing custom functions for every project. This way, you write config once and reuse it everywhere.

### Bin Continuous Variables (4 Ways)

Sometimes you want to group ages or incomes into buckets. Here's how to do it with different strategies:

```python
from skyulf_mlflow_library.features.transform import SmartBinning

# Sample data: customer ages
df = pd.DataFrame({
    'customer_id': range(1, 11),
    'age': [22, 25, 34, 45, 52, 38, 29, 67, 41, 55],
    'annual_income': [35000, 42000, 58000, 78000, 95000, 61000, 48000, 88000, 72000, 105000]
})

# Strategy 1: Equal width bins (each bin spans same range)
binner = SmartBinning(strategy='equal_width', columns=['age'], n_bins=3)
df_binned = binner.fit_transform(df)
print(df_binned[['age', 'age_binned']])
#    age      age_binned
# 0   22  (21.955, 37.0]  # 22-37 years
# 1   25  (21.955, 37.0]
# 2   34  (21.955, 37.0]
# 3   45    (37.0, 52.0]  # 37-52 years
# 4   52    (37.0, 52.0]

# Strategy 2: Equal frequency (same number of people per bin)
binner = SmartBinning(strategy='equal_frequency', columns=['age'], n_bins=3)
df_binned = binner.fit_transform(df)
# Each bin will have ~3-4 customers

# Strategy 3: K-means clustering (finds natural groups in data)
binner = SmartBinning(strategy='kbins', columns=['age'], n_bins=3, kbins_strategy='kmeans')
df_binned = binner.fit_transform(df)
# Groups customers by age similarity

# Strategy 4: Custom bins with labels (you decide the ranges)
binner = SmartBinning(
    strategy='custom',
    columns=['age'],
    bins={'age': [0, 30, 45, 65, 100]},
    labels={'age': ['Young', 'Middle-Aged', 'Senior', 'Elderly']}
)
df_binned = binner.fit_transform(df)
print(df_binned[['age', 'age_binned']])
#    age  age_binned
# 0   22       Young
# 1   25       Young
# 2   34       Young
# 3   45  Middle-Aged
# 4   52  Middle-Aged
```

**Which one to use?**
- `equal_width`: Simple and interpretable, but can have empty bins
- `equal_frequency`: Each bin has similar sample size, good for training balance
- `kmeans`: Finds natural clusters, best when data has obvious groups
- `custom`: You know the domain (e.g., age groups, income brackets)

### Pick The Best Features Automatically

Got 50 features but only need the top 10? Let the library figure it out:

```python
from skyulf_mlflow_library.features.selection import FeatureSelector

# You've got a dataset with too many columns
df = pd.DataFrame({
    'feature_1': np.random.rand(100),
    'feature_2': np.random.rand(100),
    'feature_3': np.random.rand(100),
    # ... 20 more features ...
    'target': np.random.randint(0, 2, 100)
})

# Pick the top 5 features using correlation
selector = FeatureSelector(
    method='select_k_best',
    k=5,
    problem_type='classification'
)
X = df.drop('target', axis=1)
y = df['target']
X_selected = selector.fit_transform(X, y)

print(f"Started with {len(X.columns)} features")
print(f"Kept these {len(X_selected.columns)}: {list(X_selected.columns)}")
print(f"Dropped these: {selector.dropped_columns_}")

# Output:
# Started with 23 features
# Kept these 5: ['feature_3', 'feature_7', 'feature_12', 'feature_15', 'feature_19']
# Dropped these: ['feature_1', 'feature_2', 'feature_4', ...]
```

**Try different methods** to see what works best:
- `select_k_best`: Fast, uses statistical tests
- `rfe`: Recursive elimination, slower but thorough
- `from_model`: Uses Random Forest importance
- See [full list of 9 methods](docs/FEATURE_SELECTION_METHODS.md)

### Save Models (And Actually Find Them Later)

Ever saved a model as `model_v2_final_FINAL_actually_final.pkl`? Yeah, me too. Here's a better way:

```python
from skyulf_mlflow_library.modeling import ModelRegistry
from sklearn.ensemble import RandomForestClassifier

# Train a model
model = RandomForestClassifier(n_estimators=100)
model.fit(X_train, y_train)

# Save it with proper versioning
registry = ModelRegistry('./my_models')

registry.save_model(
    model=model,
    name='churn_predictor',
    metrics={'accuracy': 0.87, 'f1': 0.82},
    tags={'team': 'data-science', 'status': 'production'},
    description='Random Forest with 100 trees, trained on Q4 2024 data'
)
# Saved as version 1

# Train a better model next week
model_v2 = RandomForestClassifier(n_estimators=200, max_depth=10)
model_v2.fit(X_train, y_train)

registry.save_model(
    model=model_v2,
    name='churn_predictor',  # Same name
    metrics={'accuracy': 0.91, 'f1': 0.88},
    tags={'team': 'data-science', 'status': 'staging'},
    description='Improved version with more trees and tuned depth'
)
# Automatically saved as version 2

# Later, in production code:
model = registry.load_model('churn_predictor', version='latest')  # Gets v2
predictions = model.predict(new_customer_data)

# Or load a specific version:
old_model = registry.load_model('churn_predictor', version=1)

# See all versions:
versions = registry.list_versions('churn_predictor')
for v in versions:
    print(f"Version {v['version']}: accuracy={v['metrics']['accuracy']}")
# Version 1: accuracy=0.87
# Version 2: accuracy=0.91
```

**Why this is better than manual saving:**
- No more `model_v2_final.pkl` nonsense
- Track metrics with each version
- Load any version by name, not by remembering filenames
- See full history: who trained what, when, and how well it performed

---

## 📚 Comprehensive Examples

Check out the [`examples/`](examples/) folder for detailed, runnable examples:

- **[01_basic_usage.py](examples/01_basic_usage.py)** - Getting started with the basics
- **[02_comprehensive_pipeline.py](examples/02_comprehensive_pipeline.py)** - Full ML workflow
- **[04_feature_math.py](examples/04_feature_math.py)** - Advanced feature engineering ⭐
- **[05_smart_binning.py](examples/05_smart_binning.py)** - Intelligent binning strategies
- **[09_feature_selection.py](examples/09_feature_selection.py)** - Feature selection methods
- **[10_full_library_showcase.py](examples/10_full_library_showcase.py)** - Everything in one place ⭐
- **[11_domain_analyzer.py](examples/11_domain_analyzer.py)** - Smart domain detection ⭐
- **[12_eda_overview.py](examples/12_eda_overview.py)** - Automated EDA capabilities

---

## 🆚 "Why Not Just Use Scikit-Learn?"

Fair question! Here's the honest answer:

**You should still use scikit-learn.** This library is built on top of it. I'm not trying to replace sklearn - that would be crazy. It's battle-tested and maintained by hundreds of experts.

**What this adds:**

| Scikit-Learn | This Library |
|--------------|--------------|
| `from sklearn.preprocessing import StandardScaler`<br>`scaler = StandardScaler()`<br>`X_scaled = scaler.fit_transform(X)` | `from skyulf_mlflow_library.preprocessing import StandardScaler`<br>`scaler = StandardScaler(columns=['age', 'income'])`<br>`df = scaler.fit_transform(df)`  # DataFrame in, DataFrame out |
| Save model:<br>`import joblib`<br>`joblib.dump(model, 'model_v2_FINAL.pkl')` | Save model:<br>`registry.save_model(model, 'churn_predictor',`<br>`  metrics={'accuracy': 0.92})` |
| Feature engineering:<br>Write custom functions for every ratio,<br>datetime extraction, aggregation | Feature engineering:<br>Use declarative config, reuse across projects |

**When to use this:**
- You're tired of writing the same preprocessing code
- You want to save models with proper versioning
- You need to create lots of derived features
- You want DataFrames throughout your pipeline

**When NOT to use this:**
- You're happy with your current workflow (don't fix what works!)
- You need bleeding-edge algorithms (stick to sklearn/XGBoost)
- You're running in a resource-constrained environment

---

## 🌟 Unique Features You Won't Find Elsewhere

### 1. **Automatic Domain Detection**
```python
# Knows if your data is e-commerce, healthcare, finance, etc.
analyzer = DomainAnalyzer()
result = analyzer.analyze(df)  # Detects patterns and gives recommendations
```

### 2. **FeatureMath - Complex Feature Engineering Made Simple**
```python
# Create ratios, aggregations, datetime features with declarative syntax
operations = [
    {'type': 'ratio', 'numerator': ['sales'], 'denominator': ['visits'], 'output': 'conversion_rate'},
    {'type': 'stat', 'method': 'mean', 'columns': ['price_1', 'price_2', 'price_3'], 'output': 'avg_price'}
]
```

### 3. **Text Insights with Recommendations**
```python
from skyulf_mlflow_library.eda import get_text_insights

insights = get_text_insights(df['description'], 'description')
# Returns: length stats, category detection, NLP recommendations
```

### 4. **Smart Binning with Multiple Strategies**
- Equal width/frequency (like pandas)
- KMeans-based (adaptive to data distribution)
- Custom with intelligent labels

### 5. **Production-Ready Model Registry**
- SQLite-based (no external dependencies)
- Version control built-in
- Metadata and metrics tracking
- Easy rollback and A/B testing

---

## 🛠️ How to Use

### Basic Pattern

All transformers follow the same pattern:

```python
from skyulf_mlflow_library.preprocessing import SimpleImputer

# 1. Initialize
imputer = SimpleImputer(strategy='mean', columns=['age', 'salary'])

# 2. Fit
imputer.fit(train_df)

# 3. Transform
transformed_df = imputer.transform(test_df)

# Or combine steps
transformed_df = imputer.fit_transform(train_df)
    'age': [25, 30, None, 40, 35],
    'salary': [50000, 60000, 55000, None, 70000],
    'city': ['NYC', 'LA', 'NYC', 'SF', 'LA'],
    'target': [0, 1, 0, 1, 1]
})

# Separate features and target
X = df.drop('target', axis=1)
y = df['target']

# Clean data
X = remove_duplicates(X)

# Impute missing values
imputer = SimpleImputer(strategy='mean')
X_imputed = imputer.fit_transform(X)

# Encode categorical variables
encoder = OneHotEncoder(columns=['city'])
X_encoded = encoder.fit_transform(X_imputed)

# Scale numeric features
scaler = StandardScaler(columns=['age', 'salary'])
X_scaled = scaler.fit_transform(X_encoded)

print(X_scaled.head())
```

### Example 2: Complete ML Pipeline

```python
from skyulf_mlflow_library.data_ingestion import DataLoader
from skyulf_mlflow_library.pipeline import make_pipeline
from skyulf_mlflow_library.preprocessing import SimpleImputer, StandardScaler
from skyulf_mlflow_library.features.encoding import OneHotEncoder
from skyulf_mlflow_library.features.transform import FeatureMath
from skyulf_mlflow_library.utils import train_test_split
from skyulf_mlflow_library.modeling import MetricsCalculator
from sklearn.ensemble import RandomForestClassifier

# 1. Load data
loader = DataLoader('data.csv')
df = loader.load()

X = df.drop('target', axis=1)
y = df['target']

# 2. Split data with stratification
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, stratify=y, random_state=42
)

# 3. Build preprocessing pipeline
preprocessor = make_pipeline(
    SimpleImputer(strategy='mean'),
    OneHotEncoder(columns=['category']),
    StandardScaler()
)

# 4. Fit and transform
X_train_processed = preprocessor.fit_transform(X_train)
X_test_processed = preprocessor.transform(X_test)

# 5. Feature engineering
feature_engineer = FeatureMath(operations=[
    {
        'type': 'stat',
        'method': 'mean',
        'columns': ['feature1', 'feature2', 'feature3'],
        'output': 'avg_features'
    },
    {
        'type': 'ratio',
        'numerator': 'feature1',
        'denominator': 'feature2',
        'output': 'feature_ratio'
    }
])

X_train_eng = feature_engineer.fit_transform(X_train_processed)
X_test_eng = feature_engineer.transform(X_test_processed)

# 6. Train model
model = RandomForestClassifier(n_estimators=100, random_state=42)
model.fit(X_train_eng, y_train)

# 7. Evaluate
y_pred = model.predict(X_test_eng)
y_pred_proba = model.predict_proba(X_test_eng)

metrics = MetricsCalculator()
results = metrics.calculate_classification_metrics(
    y_test, y_pred, y_pred_proba
)

print("Classification Metrics:")
for metric, value in results.items():
    print(f"  {metric}: {value:.4f}")
```

### Example 3: Model Registry

```python
from skyulf_mlflow_library.modeling import ModelRegistry

# Initialize registry
registry = ModelRegistry(db_path='models.db')

# Save model
model_id = registry.save_model(
    model=model,
    name='fraud_detector',
    version='1.0',
    problem_type='classification',
    description='Random Forest classifier for fraud detection',
    metrics={'accuracy': 0.95, 'f1': 0.93},
    hyperparameters={'n_estimators': 100, 'max_depth': 10},
    tags=['production', 'v1']
)

# Load model later
loaded_model = registry.load_model(model_id)

# Get all models
models = registry.list_models()
print(models)
```

### Example 4: Quick Data Quality Report

```python
import pandas as pd
from skyulf_mlflow_library.eda import generate_quality_report

sample = pd.DataFrame(
    {
        "age": [25, 30, None, 40, 35],
        "city": ["NYC", "LA", "NYC", "SF", "LA"],
        "notes": [
            "Customer reported great service",
            "Repeat client",
            "Customer reported great service",
            "New account",
            "VIP segment customer",
        ],
    }
)

report = generate_quality_report(sample)

print(report["summary"])
print(report["recommendations"][0]["title"])
```

---

## 📚 Documentation

### Core Modules

#### 1. **Data Ingestion** (`skyulf_mlflow_library.data_ingestion`)
- `DataLoader`: Universal data loader for multiple formats
- `save_data()`, `load_data()`: Convenience functions

#### 2. **Preprocessing** (`skyulf_mlflow_library.preprocessing`)
- **Imputation**: `SimpleImputer`, `KNNImputer`
- **Scaling**: `StandardScaler`, `MinMaxScaler`, `RobustScaler`, `MaxAbsScaler`
- **Cleaning**: `drop_missing_rows()`, `drop_missing_columns()`, `remove_duplicates()`, `remove_outliers()`
- **Sampling** (optional): `SMOTE`, `RandomOverSampler`, `RandomUnderSampler`

#### 3. **Feature Engineering** (`skyulf_mlflow_library.features`)
- **Encoding**: `OneHotEncoder`, `LabelEncoder`, `OrdinalEncoder`, `TargetEncoder`, `HashEncoder`
- **Transformations**: `FeatureMath`, `PolynomialFeatures`, `InteractionFeatures`, `SmartBinning`

#### 4. **Utilities** (`skyulf_mlflow_library.utils`)
- `train_test_split()`: Enhanced splitting with stratification and groups
- `train_val_test_split()`: Three-way data splitting

#### 5. **Modeling** (`skyulf_mlflow_library.modeling`)
- `MetricsCalculator`: Calculate classification and regression metrics
- `ModelRegistry`: Save, load, and manage models with versioning

#### 6. **Pipeline** (`skyulf_mlflow_library.pipeline`)
- `Pipeline`: Scikit-learn compatible pipeline
- `make_pipeline()`: Convenience function for pipeline creation

#### 7. **Lightweight EDA** (`skyulf_mlflow_library.eda`)
- `generate_quality_report()`: Compute dataset health metrics, text insights, and suggestions
- `get_text_insights()`: Analyze object columns for NLP readiness and data quality flags
- `DomainAnalyzer`: Score column patterns, optional ML signals, and recommend next steps based on inferred business domain
- `infer_domain()`: Convenience helper returning `DomainInferenceResult` for quick dataset inspection

---

## 🎯 Use Cases

### 1. **Imbalanced Classification**
```python
from skyulf_mlflow_library.preprocessing import SMOTE
from sklearn.ensemble import RandomForestClassifier

# Handle class imbalance
smote = SMOTE(random_state=42)
X_resampled, y_resampled = smote.fit_resample(X_train, y_train)

# Train on balanced data
model = RandomForestClassifier()
model.fit(X_resampled, y_resampled)
```

### 2. **High-Cardinality Features**
```python
from skyulf_mlflow_library.features.encoding import HashEncoder

# Encode millions of unique categories efficiently
encoder = HashEncoder(columns=['user_id', 'product_id'], n_components=16)
X_encoded = encoder.fit_transform(X)
```

### 3. **Target Encoding for Supervised Learning**
```python
from skyulf_mlflow_library.features.encoding import TargetEncoder

# Encode categories using target statistics (with smoothing)
encoder = TargetEncoder(columns=['category'], smoothing=1.0)
X_encoded = encoder.fit_transform(X, y)
```

### 4. **Polynomial Feature Engineering**
```python
from skyulf_mlflow_library.features.transform import PolynomialFeatures

# Create polynomial features up to degree 3
poly = PolynomialFeatures(degree=3, interaction_only=False)
X_poly = poly.fit_transform(X)
```

---

## 🏗️ Architecture

```
skyulf-mlflow/
├── skyulf_mlflow/
│   ├── core/               # Base classes, types, exceptions
│   ├── data_ingestion/     # Data loading utilities
│   ├── preprocessing/      # Imputation, scaling, cleaning, sampling
│   ├── features/          # Encoding and transformations
│   │   ├── encoding/      # OneHot, Label, Ordinal, Target, Hash
│   │   └── transform/     # FeatureMath, Polynomial, Binning
│   ├── modeling/          # Metrics, model registry
│   ├── pipeline/          # Pipeline system
│   └── utils/             # Splitting and utilities
├── examples/              # Example scripts
├── tests/                 # Test suite
└── docs/                  # Documentation
```

---

### Chaining Operations

```python
# Method 1: Step by step
df = imputer.fit_transform(df)
df = scaler.fit_transform(df)
df = encoder.fit_transform(df)

# Method 2: Store fitted transformers for later
imputer.fit(train_df)
scaler.fit(train_df)
encoder.fit(train_df)

# Apply to test data
test_df = imputer.transform(test_df)
test_df = scaler.transform(test_df)
test_df = encoder.transform(test_df)
```

---

## 🧪 Testing & Quality

- **133 Tests**: Comprehensive test coverage
- **56% Coverage**: Core functionality tested  
- **All Tests Passing**: ✅ 132/133 (1 skipped due to optional dependency)
- **Type Hints**: Throughout the codebase
- **Error Handling**: Proper exceptions and validation

```bash
# Run tests
pytest tests/

# With coverage
pytest tests/ --cov=skyulf_mlflow --cov-report=term
```

---

## 📖 Documentation

### 📂 Structure
```
docs/
├── api/                          # API documentation
│   ├── feature_engineering.md
│   ├── domain_analysis.md
│   └── eda.md
├── CONTRIBUTING.md               # How to contribute
└── PYPI_PUBLISHING.md           # Publishing guide

examples/
├── 01_basic_usage.py
├── 04_feature_math.py           # ⭐ Advanced features
├── 10_full_library_showcase.py # ⭐ Everything
└── 11_domain_analyzer.py        # ⭐ Unique feature
```

### 🔗 Quick Links
- [Contributing Guide](skyulf_mlflow/CONTRIBUTING.md)
- [API Documentation](docs/api/)
- [Examples](examples/)

---

## 🗺️ Roadmap

This is an **ongoing side project**. Future improvements I'm planning:

### Short-term (When I have time 😊)
- [ ] More feature engineering operations
- [ ] Additional encoding strategies
- [ ] Enhanced text analysis
- [ ] More examples and tutorials
- [ ] Better documentation

### Medium-term
- [ ] Pipeline visualization
- [ ] AutoML capabilities
- [ ] Time series specific features
- [ ] Deep learning integration
- [ ] Web UI for exploration

### Long-term
- [ ] Cloud deployment helpers
- [ ] Real-time prediction serving
- [ ] Model monitoring and drift detection
- [ ] Custom transformer builder

**Want to help?** Contributions are very welcome! See [CONTRIBUTING.md](skyulf_mlflow/CONTRIBUTING.md)

---

## 💭 Philosophy

This library is built on these principles:

1. **🎯 Practicality Over Perfection**: Features that solve real problems
2. **🤝 Ease of Use**: Consistent API, good defaults, clear errors
3. **🔧 Composability**: Mix and match components freely
4. **📚 Learning by Doing**: Each feature comes from actual ML work
5. **🌱 Continuous Improvement**: Always evolving based on experience

---

## 🤝 Contributing

I built this for myself and my projects, but I'd love your input!

**Ways to contribute:**
- 🐛 Report bugs or issues
- 💡 Suggest new features
- 📝 Improve documentation
- 🧪 Add more tests
- ⭐ Star the repo if you find it useful!

See [CONTRIBUTING.md](skyulf_mlflow/CONTRIBUTING.md) for:
- How to add new features
- Testing guidelines
- Code style conventions
- Pull request process

### Development Setup
```bash
# Clone repository
git clone https://github.com/flyingriverhorse/skyulf_mlflow_library.git
cd skyulf_mlflow_library

# Install development dependencies
pip install -e ".[dev]"

# Run tests
pytest tests/

# Check coverage
pytest tests/ --cov=skyulf_mlflow --cov-report=html
```

---

## ⚠️ Current Status

**Version**: 0.1.1 (Alpha)

This is a **work in progress**. I update it whenever I:
- Build a new ML project and need a feature
- Find a better way to do something
- Get inspired by other libraries
- Have some free time on weekends 😄

**What this means:**
- ✅ Core functionality is solid and tested
- ✅ API is relatively stable
- ⚠️ Some features are still evolving
- ⚠️ Documentation is improving continuously
- 🚧 New features added regularly

**Use it if:**
- You want practical ML tools
- You like the "batteries included" approach
- You don't mind some rough edges
- You want to contribute and help shape it

---

## 📊 Stats

- **Tests**: 133 (132 passing)
- **Coverage**: 56%
- **Examples**: 12
- **Python**: 3.8 - 3.12
- **Dependencies**: Minimal (mostly stdlib + scientific stack)

---

## 📋 Changelog

See [CHANGELOG.md](CHANGELOG.md) for version history.

### Latest: v0.1.1 (2025-11-01)
- ✅ Added EDA module with domain analysis
- ✅ Added feature selection with multiple strategies
- ✅ Added comprehensive test suite (133 tests, 56% coverage)
- ✅ Added 12 runnable examples
- ✅ Added CONTRIBUTING guide

### v0.1.0 (2025-10-31)
- ✅ Initial alpha release
- ✅ Core preprocessing and feature engineering
- ✅ Model registry and metrics
- ✅ Basic documentation

---

## 🙏 Acknowledgments

This library stands on the shoulders of giants:

- **[scikit-learn](https://scikit-learn.org/)**: The foundation of ML in Python
- **[pandas](https://pandas.pydata.org/)**: Data manipulation made easy
- **[numpy](https://numpy.org/)**: Numerical computing powerhouse
- **[imbalanced-learn](https://imbalanced-learn.org/)**: Handling class imbalance
- **[scipy](https://scipy.org/)**: Scientific computing tools

And countless blog posts, Stack Overflow answers, and GitHub repos that taught me better ways to do things. Thank you all! 🙏

---

## 📜 License

MIT License - see [LICENSE](LICENSE) file for details.

**TL;DR**: Use it however you want, just don't sue me if something breaks 😊

Copyright (c) 2025 Murat Unsal

---

## 📧 Contact & Support

**Author**: Murat Unsal

- 📫 Email: murath.unsal@unsal.com
- 🐙 GitHub: [flyingriverhorse/MLops](https://github.com/flyingriverhorse/skyulf_mlflow_library)
- 💬 Issues: [Report a bug or request a feature](https://github.com/flyingriverhorse/skyulf_mlflow_library/issues)

---

## ⭐ Star History

If this library helps you, consider giving it a star! It motivates me to keep improving it 🌟

---

## 🎉 Final Words

This is a **passion project** born from real ML work. I built the tools I wished existed when I started doing data science. If it helps you too, that's awesome!

**Remember:**
- Start simple, add complexity when needed
- Good features > Many features
- Documentation is love ❤️
- Tests are your safety net 🛡️

**Happy Machine Learning! 🚀**

---

<div align="center">

Made with ☕ and 💻 during late nights and weekends

**If you find this useful, give it a ⭐ and spread the word!**

</div>
