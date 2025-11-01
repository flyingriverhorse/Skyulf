# Contributing to Skyulf-MLFlow

This guide explains how to add new features to the Skyulf-MLFlow library following our established patterns and best practices.

## Table of Contents
- [Adding a New Feature](#adding-a-new-feature)
- [Code Structure](#code-structure)
- [Testing Guidelines](#testing-guidelines)
- [Examples Guidelines](#examples-guidelines)
- [API Design Principles](#api-design-principles)
- [Documentation Standards](#documentation-standards)

---

## Adding a New Feature

When adding a new feature to the library, follow these steps:

### 1. Implement the Feature

**Location**: `skyulf_mlflow/<module>/<submodule>/`

All transformers and processors should inherit from `BaseTransformer`:

```python
from skyulf_mlflow_library.core.base import BaseTransformer
from skyulf_mlflow_library.core.exceptions import FeatureEngineeringError
from skyulf_mlflow_library.core.types import DataFrame

class MyNewTransformer(BaseTransformer):
    """
    Brief description of what this transformer does.
    
    Parameters
    ----------
    columns : list of str, optional
        Columns to transform. If None, applies to all numeric columns.
    param1 : str, default='value'
        Description of parameter 1.
    param2 : int, default=5
        Description of parameter 2.
    
    Examples
    --------
    >>> from skyulf_mlflow_library.features.transform import MyNewTransformer
    >>> import pandas as pd
    >>> 
    >>> df = pd.DataFrame({'A': [1, 2, 3], 'B': [4, 5, 6]})
    >>> transformer = MyNewTransformer(columns=['A'], param1='test')
    >>> result = transformer.fit_transform(df)
    """
    
    def __init__(
        self,
        columns: Optional[List[str]] = None,
        param1: str = 'value',
        param2: int = 5,
        **kwargs
    ):
        super().__init__(columns=columns, **kwargs)
        self.param1 = param1
        self.param2 = param2
        
    def fit(self, df: DataFrame, y=None) -> 'MyNewTransformer':
        """Fit the transformer to the data."""
        self._validate_input(df)
        
        # Determine which columns to transform
        if self.columns is None:
            self.columns_ = df.select_dtypes(include=['number']).columns.tolist()
        else:
            self.columns_ = self.columns
            
        # Store any fitted parameters
        self.fitted_params_ = {}
        
        self.is_fitted_ = True
        return self
    
    def transform(self, df: DataFrame) -> DataFrame:
        """Transform the data."""
        self._check_is_fitted()
        df = df.copy()
        
        # Apply transformation
        for col in self.columns_:
            df[f'{col}_transformed'] = df[col] * self.param2
            
        return df
```

**Key Points**:
- Inherit from `BaseTransformer`
- Implement `fit()` and `transform()` methods
- Store fitted parameters with trailing underscore (e.g., `columns_`, `is_fitted_`)
- Use type hints from `skyulf_mlflow.core.types`
- Raise appropriate exceptions from `skyulf_mlflow.core.exceptions`

### 2. Update `__init__.py`

**Location**: `skyulf_mlflow/<module>/__init__.py`

Export your new class:

```python
from .my_module import MyNewTransformer

__all__ = [
    'MyNewTransformer',
    # ... other exports
]
```

### 3. Write Tests

**Location**: `tests/test_<module>/test_<feature>.py`

Create comprehensive tests following this structure:

```python
"""Tests for MyNewTransformer."""

import pytest
import pandas as pd
import numpy as np

from skyulf_mlflow_library.features.transform import MyNewTransformer


class TestMyNewTransformerBasic:
    """Test basic functionality."""
    
    def test_initialization(self):
        """Test that transformer initializes correctly."""
        transformer = MyNewTransformer(param1='test', param2=10)
        
        assert transformer.param1 == 'test'
        assert transformer.param2 == 10
    
    def test_fit_transform_basic(self):
        """Test basic fit_transform."""
        df = pd.DataFrame({
            'A': [1, 2, 3, 4, 5],
            'B': [10, 20, 30, 40, 50]
        })
        
        transformer = MyNewTransformer(columns=['A'], param2=2)
        result = transformer.fit_transform(df)
        
        assert 'A_transformed' in result.columns
        assert result['A_transformed'].tolist() == [2, 4, 6, 8, 10]
    
    def test_fit_then_transform(self):
        """Test that transform works after fit."""
        df_train = pd.DataFrame({'A': [1, 2, 3]})
        df_test = pd.DataFrame({'A': [4, 5, 6]})
        
        transformer = MyNewTransformer()
        transformer.fit(df_train)
        result = transformer.transform(df_test)
        
        assert 'A_transformed' in result.columns


class TestMyNewTransformerEdgeCases:
    """Test edge cases and error handling."""
    
    def test_transform_without_fit_raises_error(self):
        """Test that transform without fit raises error."""
        df = pd.DataFrame({'A': [1, 2, 3]})
        transformer = MyNewTransformer()
        
        with pytest.raises((ValueError, AttributeError, Exception)):
            transformer.transform(df)
    
    def test_with_missing_values(self):
        """Test handling of missing values."""
        df = pd.DataFrame({
            'A': [1, 2, np.nan, 4, 5]
        })
        
        transformer = MyNewTransformer(columns=['A'])
        result = transformer.fit_transform(df)
        
        # Define expected behavior with NaN
        assert result is not None
    
    def test_empty_dataframe(self):
        """Test with empty DataFrame."""
        df = pd.DataFrame()
        transformer = MyNewTransformer()
        
        # Should either handle gracefully or raise appropriate exception
        with pytest.raises(Exception):
            transformer.fit_transform(df)
```

**Testing Best Practices**:
- Test initialization
- Test basic functionality
- Test fit, transform, and fit_transform
- Test with specific columns vs. auto-detect
- Test edge cases (empty data, missing values, single value)
- Test error conditions
- Use descriptive test names
- Group related tests in classes
- Aim for >60% coverage of your new module

### 4. Create Example

**Location**: `examples/XX_<feature_name>.py`

Create a runnable example demonstrating your feature:

```python
"""
Example: My New Transformer - Brief Description

Demonstrates how to use MyNewTransformer for common use cases.
"""

import pandas as pd
import numpy as np
from skyulf_mlflow_library.features.transform import MyNewTransformer

print("=" * 70)
print("Skyulf-MLFlow: My New Transformer Example")
print("=" * 70)
print()

# ==============================================================================
# 1. Basic Usage
# ==============================================================================
print("1. BASIC USAGE")
print("-" * 70)

df = pd.DataFrame({
    'feature1': [1, 2, 3, 4, 5],
    'feature2': [10, 20, 30, 40, 50],
    'feature3': [100, 200, 300, 400, 500]
})

print("Original Data:")
print(df)
print()

# Apply transformer
transformer = MyNewTransformer(columns=['feature1', 'feature2'], param2=2)
df_result = transformer.fit_transform(df)

print("After Transformation:")
print(df_result)
print("\n")

# ==============================================================================
# 2. Advanced Usage
# ==============================================================================
print("2. ADVANCED USAGE")
print("-" * 70)

# Demonstrate more complex scenarios
transformer_advanced = MyNewTransformer(param1='custom', param2=10)
df_advanced = transformer_advanced.fit_transform(df)

print("Advanced Result:")
print(df_advanced)
print("\n")

# ==============================================================================
# 3. Real-World Use Case
# ==============================================================================
print("3. REAL-WORLD USE CASE")
print("-" * 70)

# Show a practical example
print("Use case description...")
print("\n")

# ==============================================================================
# Summary
# ==============================================================================
print("=" * 70)
print("SUMMARY")
print("=" * 70)
print("Key Points:")
print("  • Feature 1: Description")
print("  • Feature 2: Description")
print("  • Feature 3: Description")
```

**Example Best Practices**:
- Start with simple usage
- Progress to advanced features
- Include real-world use cases
- Print outputs to show results
- Add comments explaining what's happening
- Keep examples runnable and self-contained

---

## Code Structure

### Module Organization

```
skyulf_mlflow/
├── core/                      # Core base classes and types
│   ├── base.py               # BaseTransformer
│   ├── types.py              # Type definitions
│   └── exceptions.py         # Custom exceptions
├── data_ingestion/           # Data loading and saving
├── eda/                      # Exploratory data analysis
│   ├── domain.py            # Domain analysis
│   ├── quality.py           # Data quality checks
│   └── text.py              # Text analysis
├── features/                 # Feature engineering
│   ├── encoding/            # Categorical encoding
│   ├── selection/           # Feature selection
│   └── transform/           # Feature transformations
├── modeling/                 # Model training and evaluation
│   ├── classifiers.py       # Classification models
│   ├── regressors.py        # Regression models
│   ├── metrics.py           # Metrics calculation
│   └── registry.py          # Model registry
├── preprocessing/            # Data preprocessing
│   ├── cleaning.py          # Data cleaning
│   ├── imputation.py        # Missing value handling
│   ├── scaling.py           # Feature scaling
│   └── sampling.py          # Sampling methods
└── utils/                    # Utility functions
```

### Naming Conventions

- **Classes**: PascalCase (e.g., `FeatureSelector`, `SmartBinning`)
- **Functions**: snake_case (e.g., `drop_missing_rows`, `calculate_metrics`)
- **Variables**: snake_case (e.g., `column_names`, `fitted_params`)
- **Constants**: UPPER_SNAKE_CASE (e.g., `DEFAULT_K_FALLBACK`)
- **Private methods**: `_leading_underscore` (e.g., `_validate_input`)
- **Fitted attributes**: `trailing_underscore_` (e.g., `columns_`, `is_fitted_`)

---

## Testing Guidelines

### Test Organization

```
tests/
├── test_eda_domain.py
├── test_eda_quality.py
├── test_eda_text.py
├── test_preprocessing.py
├── test_utils.py
├── test_features/
│   ├── test_encoding.py
│   ├── test_selection.py
│   ├── test_transform.py
│   └── test_binning.py
├── test_ingestion/
│   └── test_data_loader.py
├── test_modeling/
│   ├── test_metrics_basic.py
│   └── test_registry_basic.py
└── test_preprocessing/
    └── test_cleaning.py
```

### Running Tests

```bash
# Run all tests
python -m pytest tests/

# Run with coverage
python -m pytest tests/ --cov=skyulf_mlflow --cov-report=term

# Run specific test file
python -m pytest tests/test_features/test_transform.py

# Run with verbose output
python -m pytest tests/ -v

# Run specific test
python -m pytest tests/test_features/test_transform.py::test_feature_math_basic
```

### Coverage Goals

- **Minimum**: 60% coverage for new modules
- **Target**: 70%+ coverage
- **Critical paths**: 90%+ coverage for core functionality

---

## Examples Guidelines

### Example Numbering

Use sequential numbering for examples:
- `01_basic_usage.py` - Simple introduction
- `02_comprehensive_pipeline.py` - Full workflow
- `03_data_consistency.py` - Specific feature
- `04_feature_math.py` - Advanced transformations
- etc.

### Example Structure

1. **Header** - Title and description
2. **Imports** - All necessary imports
3. **Sections** - Numbered sections with separators
4. **Output** - Print results to demonstrate functionality
5. **Summary** - Key takeaways

---

## API Design Principles

### Consistency

All transformers should follow the scikit-learn API:

```python
# Initialize
transformer = MyTransformer(param1=value1)

# Fit
transformer.fit(X_train)

# Transform
X_transformed = transformer.transform(X_test)

# Or combine
X_transformed = transformer.fit_transform(X_train)
```

### Parameter Naming

- **`columns`**: Which columns to process
- **`drop_original`**: Whether to drop original columns
- **`suffix`**: Suffix for new columns
- **`inplace`**: Whether to modify in-place
- **`random_state`**: Random seed for reproducibility

### Common Patterns

#### FeatureMath API
```python
operations = [
    {
        'type': 'arithmetic',      # Operation category
        'method': 'add',            # Specific method
        'columns': ['col1', 'col2'],
        'output': 'result_col'
    }
]
```

#### Encoder API
```python
encoder = LabelEncoder(column='category')
# Creates 'category_encoded' column, optionally drops original
```

#### Text Insights API
```python
insights = get_text_insights(series, column_name='text_col')
# Returns dict with keys: 'avg_text_length', 'text_category', 'eda_recommendations'
```

### Error Handling

Use appropriate exception types:

```python
from skyulf_mlflow_library.core.exceptions import (
    FeatureEngineeringError,
    PreprocessingError,
    DataProcessingError
)

if invalid_condition:
    raise FeatureEngineeringError("Clear error message")
```

---

## Documentation Standards

### Docstring Format

Use NumPy-style docstrings:

```python
def my_function(param1, param2):
    """
    Brief one-line description.
    
    More detailed description of what the function does,
    including any important details or caveats.
    
    Parameters
    ----------
    param1 : type
        Description of param1.
    param2 : type, optional
        Description of param2. Default is None.
    
    Returns
    -------
    type
        Description of return value.
    
    Raises
    ------
    ExceptionType
        Description of when this is raised.
    
    Examples
    --------
    >>> result = my_function(value1, value2)
    >>> print(result)
    expected_output
    
    Notes
    -----
    Additional notes, algorithms, or references.
    """
```

### Code Comments

- Explain **why**, not **what**
- Document complex logic
- Add section separators for clarity
- Keep comments up-to-date

---

## Checklist for New Features

Before submitting a new feature, ensure:

- [ ] Feature implemented in appropriate module
- [ ] Inherits from `BaseTransformer` (if applicable)
- [ ] Exported in `__init__.py`
- [ ] Comprehensive docstrings with examples
- [ ] Type hints added
- [ ] Tests created with >60% coverage
- [ ] Tests include edge cases and error conditions
- [ ] Example file created in `examples/`
- [ ] Example is runnable and well-commented
- [ ] All tests pass (`pytest tests/`)
- [ ] Code follows naming conventions
- [ ] No breaking changes to existing API

---

## Common Issues and Solutions

### Issue: Tests fail with "not fitted" error
**Solution**: Ensure `is_fitted_` is set in `fit()` method

### Issue: Transform doesn't preserve DataFrame structure
**Solution**: Always return a copy: `df = df.copy()`

### Issue: Coverage is low
**Solution**: Add tests for edge cases, error conditions, and all parameters

### Issue: Example fails on user's machine
**Solution**: Keep examples self-contained, avoid external dependencies

---

## Getting Help

- Check existing code for patterns and examples
- Review test files to understand expected behavior
- Look at similar features for guidance
- Ensure all tests pass before submitting

---

## Version Control

When committing changes:

```bash
# Good commit messages
git commit -m "feat: Add SmartBinning transformer with multiple strategies"
git commit -m "test: Add comprehensive tests for FeatureSelector"
git commit -m "docs: Add example for text analysis features"
git commit -m "fix: Correct handling of missing values in imputer"
```

Use conventional commits format:
- `feat:` - New feature
- `fix:` - Bug fix
- `docs:` - Documentation
- `test:` - Tests
- `refactor:` - Code refactoring
- `style:` - Code style/formatting
- `chore:` - Maintenance tasks

---

**Happy Contributing! 🚀**
