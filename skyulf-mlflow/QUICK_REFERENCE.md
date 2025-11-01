# Skyulf-MLFlow Quick Reference

**One-page cheat sheet for common operations**

## 🚀 Installation

```bash
pip install -e .                    # Basic
pip install -e ".[all]"             # Everything
```

## 📚 Import Patterns

```python
# Preprocessing
from skyulf_mlflow_library.preprocessing import SimpleImputer, StandardScaler, MinMaxScaler

# Encoding
from skyulf_mlflow_library.features.encoding import OneHotEncoder, LabelEncoder, OrdinalEncoder

# Feature Engineering
from skyulf_mlflow_library.features.transform import FeatureMath, SmartBinning, PolynomialFeatures
from skyulf_mlflow_library.features.selection import FeatureSelector

# EDA
from skyulf_mlflow_library.eda import DomainAnalyzer, get_text_insights

# Modeling
from skyulf_mlflow_library.modeling import RandomForestClassifier, MetricsCalculator, ModelRegistry
```

## 🔧 Common Operations

### Imputation
```python
imputer = SimpleImputer(strategy='mean')  # or 'median', 'most_frequent', 'constant'
df = imputer.fit_transform(df)
```

### Scaling
```python
scaler = StandardScaler(columns=['age', 'salary'])
df = scaler.fit_transform(df)
```

### Encoding
```python
encoder = OneHotEncoder(columns=['city', 'category'])
df = encoder.fit_transform(df)
```

### Feature Math
```python
operations = [
    {'type': 'arithmetic', 'method': 'add', 'columns': ['a', 'b'], 'output': 'sum'},
    {'type': 'ratio', 'numerator': ['sales'], 'denominator': ['cost'], 'output': 'margin'}
]
fm = FeatureMath(operations=operations)
df = fm.fit_transform(df)
```

### Binning
```python
binner = SmartBinning(strategy='equal_frequency', columns=['age'], n_bins=5)
df = binner.fit_transform(df)
```

### Feature Selection
```python
selector = FeatureSelector(
    method='select_k_best',
    k=10,
    problem_type='classification',
    target_column='target'
)
df = selector.fit_transform(df)
```

### Domain Analysis
```python
analyzer = DomainAnalyzer()
result = analyzer.analyze(df)
print(f"Domain: {result.primary_domain}")
```

### Model Training
```python
model = RandomForestClassifier(n_estimators=100)
model.fit(X_train, y_train)
predictions = model.predict(X_test)
```

### Metrics
```python
calc = MetricsCalculator(problem_type='classification')
metrics = calc.calculate(y_true, y_pred)
```

### Model Registry
```python
registry = ModelRegistry('./models')
model_id = registry.save_model(
    model=my_model,
    name='my_classifier',
    problem_type='classification',
    metrics={'accuracy': 0.95}
)
loaded_model = registry.load_model('my_classifier')
```

## 🎯 FeatureMath Operations

### Arithmetic
```python
{'type': 'arithmetic', 'method': 'add', 'columns': ['a', 'b'], 'output': 'sum'}
{'type': 'arithmetic', 'method': 'subtract', 'columns': ['a', 'b'], 'output': 'diff'}
{'type': 'arithmetic', 'method': 'multiply', 'columns': ['a', 'b'], 'output': 'product'}
{'type': 'arithmetic', 'method': 'divide', 'columns': ['a', 'b'], 'output': 'ratio'}
```

### Ratios
```python
{'type': 'ratio', 'numerator': ['sales'], 'denominator': ['cost'], 'output': 'profit_margin'}
```

### Statistics
```python
{'type': 'stat', 'method': 'mean', 'columns': ['a', 'b', 'c'], 'output': 'average'}
{'type': 'stat', 'method': 'sum', 'columns': ['a', 'b', 'c'], 'output': 'total'}
{'type': 'stat', 'method': 'min', 'columns': ['a', 'b', 'c'], 'output': 'minimum'}
{'type': 'stat', 'method': 'max', 'columns': ['a', 'b', 'c'], 'output': 'maximum'}
```

### Datetime
```python
{'type': 'datetime', 'method': 'extract_hour', 'columns': ['timestamp'], 'output': 'hour'}
{'type': 'datetime', 'method': 'extract_day', 'columns': ['timestamp'], 'output': 'day'}
{'type': 'datetime', 'method': 'extract_month', 'columns': ['timestamp'], 'output': 'month'}
```

## 🏷️ Encoding Strategies

| Encoder | Use Case | Example |
|---------|----------|---------|
| OneHot | Low cardinality categorical | city, gender |
| Label | Ordinal categories | low/medium/high |
| Ordinal | Ordered categories | small < medium < large |
| Target | High cardinality | zip codes, user IDs |
| Hash | Very high cardinality | text, IDs |

## 📊 Binning Strategies

| Strategy | Description | Best For |
|----------|-------------|----------|
| equal_width | Equal-sized intervals | Uniform distributions |
| equal_frequency | Equal samples per bin | Skewed distributions |
| kmeans | Cluster-based | Complex distributions |
| custom | User-defined edges | Domain knowledge |

## 🎯 Feature Selection Methods

| Method | Description | Parameters |
|--------|-------------|------------|
| select_k_best | Top K features | k=10 |
| select_percentile | Top N% features | percentile=50 |
| variance_threshold | Remove low variance | threshold=0.01 |
| rfe | Recursive elimination | n_features_to_select |

## 🧪 Common Patterns

### Train-Test Split
```python
from skyulf_mlflow_library.utils import train_test_split

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
```

### Pipeline Pattern
```python
# Fit on training data
imputer.fit(X_train)
scaler.fit(X_train)
encoder.fit(X_train)

# Transform training data
X_train = imputer.transform(X_train)
X_train = scaler.transform(X_train)
X_train = encoder.transform(X_train)

# Transform test data (using fitted transformers)
X_test = imputer.transform(X_test)
X_test = scaler.transform(X_test)
X_test = encoder.transform(X_test)
```

### Or use fit_transform
```python
# For training data
X_train = imputer.fit_transform(X_train)
X_train = scaler.fit_transform(X_train)
X_train = encoder.fit_transform(X_train)

# For test data, must use transform only
X_test = imputer.transform(X_test)
X_test = scaler.transform(X_test)
X_test = encoder.transform(X_test)
```

## 🔍 Debugging Tips

### Check if fitted
```python
if hasattr(transformer, 'is_fitted_') and transformer.is_fitted_:
    print("Transformer is fitted")
```

### Get fitted parameters
```python
print(f"Columns: {transformer.columns_}")
print(f"Parameters: {transformer.get_params()}")
```

### Error handling
```python
from skyulf_mlflow_library.exceptions import (
    PreprocessingError,
    FeatureEngineeringError,
    TransformerNotFittedError
)

try:
    result = transformer.transform(df)
except TransformerNotFittedError:
    print("Must call fit() first!")
```

## 📖 Get Help

```python
# In Python
help(SimpleImputer)
help(FeatureMath)

# Or check docstrings
SimpleImputer.__doc__
```

## 🔗 Resources

- **Examples**: `examples/` folder
- **Contributing**: `skyulf_mlflow_library/CONTRIBUTING.md`
- **API Docs**: `docs/api/`
- **Tests**: `tests/` folder (great for learning!)

## 💡 Pro Tips

1. **Always fit on training data only**
2. **Transform both train and test with same fitted transformer**
3. **Check data types before encoding** (categorical vs numeric)
4. **Use domain analyzer** for quick insights
5. **Start simple, add complexity** as needed
6. **Check test coverage** when adding features
7. **Run examples** to learn patterns

---

**Quick Command Reference**

```bash
# Run tests
pytest tests/

# Check coverage
pytest tests/ --cov=skyulf_mlflow_library

# Run specific test
pytest tests/test_features/test_transform.py

# Install dev dependencies
pip install -e ".[dev]"

# Run all examples
python examples/01_basic_usage.py
```

---

**Made with ❤️ - Keep this handy!**
