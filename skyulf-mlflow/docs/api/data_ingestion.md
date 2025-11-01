# Data Ingestion API Documentation

## Overview

The data ingestion module provides utilities for loading and saving data from various sources including CSV, Excel, JSON, Parquet, and SQL databases.

**Available Classes:**
- **DataLoader**: Load data from files, databases, and Python objects
- **DataSaver**: Save data to various file formats

**Convenience Functions:**
- **load_data()**: Quick data loading
- **save_data()**: Quick data saving

---

## 📥 DataLoader

Universal data loader supporting multiple file formats with automatic format detection.

### Class Signature

```python
from skyulf_mlflow_library.data_ingestion import DataLoader

DataLoader(
    source: Union[str, Path],
    file_type: Optional[str] = None,
    **kwargs
)
```

**Parameters:**
- `source` (str | Path): Path to data file, database query, or URL
- `file_type` (str, optional): File type ('csv', 'excel', 'json', 'parquet', 'sql'). Auto-detected from extension if not provided
- `**kwargs`: Additional arguments passed to pandas read functions

**Supported Formats:**
- CSV (`.csv`, `.txt`)
- Excel (`.xlsx`, `.xls`, `.xlsm`)
- JSON (`.json`)
- Parquet (`.parquet`, `.pq`)
- SQL databases (requires `file_type='sql'` and connection)

---

### When to Use DataLoader

✅ **Use when:**
- Loading data from standard file formats
- Need automatic format detection
- Want consistent API across different sources
- Loading from SQL databases
- Loading from URLs
- Need flexible parameter passing to pandas

❌ **Don't use when:**
- Working with streaming data
- Need real-time data ingestion
- Handling non-tabular data
- Working with very large files (>10GB) - consider chunking

---

## 📖 Loading Methods

### 1. Load from CSV

```python
from skyulf_mlflow_library.data_ingestion import DataLoader

# Basic loading (auto-detects format)
loader = DataLoader('data.csv')
df = loader.load()

# With custom delimiter
loader = DataLoader('data.csv', sep=';')
df = loader.load()

# With encoding
loader = DataLoader('data.csv', encoding='latin1')
df = loader.load()

# Skip rows and select columns
loader = DataLoader('data.csv', skiprows=5, usecols=['col1', 'col2'])
df = loader.load()

# Parse dates
loader = DataLoader('data.csv', parse_dates=['date_column'])
df = loader.load()

# Handle missing values
loader = DataLoader('data.csv', na_values=['NA', 'missing', ''])
df = loader.load()
```

**Common CSV Parameters:**
```python
sep=','          # Column separator
delimiter=';'    # Alias for sep
encoding='utf-8' # File encoding
skiprows=5       # Skip first N rows
nrows=1000       # Read only N rows
usecols=['a','b']# Select columns
parse_dates=['date'] # Parse date columns
na_values=['NA']  # Additional NA values
decimal=','      # Decimal point character
thousands='.'    # Thousands separator
```

**When to Use CSV:**
- Most common data format
- Excel exports
- Database dumps
- Log files
- Data from other tools

---

### 2. Load from Excel

```python
from skyulf_mlflow_library.data_ingestion import DataLoader

# Load first sheet (default)
loader = DataLoader('data.xlsx')
df = loader.load()

# Load specific sheet by name
loader = DataLoader('data.xlsx', sheet_name='Sales')
df = loader.load()

# Load specific sheet by index
loader = DataLoader('data.xlsx', sheet_name=0)
df = loader.load()

# Load multiple sheets
loader = DataLoader('data.xlsx', sheet_name=['Sales', 'Inventory'])
dict_of_dfs = loader.load()  # Returns dictionary

# Load all sheets
loader = DataLoader('data.xlsx', sheet_name=None)
all_sheets = loader.load()  # Returns dictionary

# Skip rows
loader = DataLoader('data.xlsx', skiprows=3)
df = loader.load()

# Select columns
loader = DataLoader('data.xlsx', usecols='A:D')
df = loader.load()
```

**Common Excel Parameters:**
```python
sheet_name='Sheet1'  # Sheet name or index
header=0            # Row with column names
skiprows=3          # Skip first N rows
usecols='A:D'       # Column range
dtype={'col': int}  # Column data types
```

**When to Use Excel:**
- Data from business users
- Multi-sheet workbooks
- Formatted reports
- Financial data

---

### 3. Load from JSON

```python
from skyulf_mlflow_library.data_ingestion import DataLoader

# Load JSON file
loader = DataLoader('data.json')
df = loader.load()

# Records format: [{"col1": val1, "col2": val2}, ...]
loader = DataLoader('data.json', orient='records')
df = loader.load()

# Columns format: {"col1": [val1, val2], "col2": [val3, val4]}
loader = DataLoader('data.json', orient='columns')
df = loader.load()

# Index format
loader = DataLoader('data.json', orient='index')
df = loader.load()

# Values format (only values, no keys)
loader = DataLoader('data.json', orient='values')
df = loader.load()

# Nested JSON with path
loader = DataLoader('data.json')
df = loader.load()
```

**Common JSON Parameters:**
```python
orient='records'  # JSON structure
lines=True       # JSON lines format (one object per line)
dtype={'col': int} # Column types
```

**When to Use JSON:**
- API responses
- NoSQL database exports
- Nested/hierarchical data
- Web scraping results
- Configuration files

---

### 4. Load from Parquet

```python
from skyulf_mlflow_library.data_ingestion import DataLoader

# Load Parquet file
loader = DataLoader('data.parquet')
df = loader.load()

# Load specific columns
loader = DataLoader('data.parquet', columns=['col1', 'col2'])
df = loader.load()

# With filters
loader = DataLoader('data.parquet', filters=[('age', '>', 18)])
df = loader.load()
```

**Common Parquet Parameters:**
```python
columns=['a','b']  # Select columns
filters=[...]     # Row filters
engine='pyarrow'  # Engine to use
```

**When to Use Parquet:**
- Big data workflows
- Data lakes
- When you need compression
- Column-oriented analysis
- Apache Spark integration
- High-performance storage

**Benefits:**
- 10-100x smaller than CSV
- Much faster to read/write
- Preserves data types
- Column pruning
- Predicate pushdown

---

### 5. Load from SQL Database

```python
from skyulf_mlflow_library.data_ingestion import DataLoader
from sqlalchemy import create_engine

# Create database connection
engine = create_engine('postgresql://user:password@localhost:5432/dbname')

# Load full table
query = 'SELECT * FROM users'
loader = DataLoader(query, file_type='sql')
df = loader.load(con=engine)

# Load with WHERE clause
query = 'SELECT * FROM users WHERE age > 18'
loader = DataLoader(query, file_type='sql')
df = loader.load(con=engine)

# Load with JOIN
query = '''
    SELECT u.name, u.email, o.order_date, o.amount
    FROM users u
    LEFT JOIN orders o ON u.id = o.user_id
    WHERE o.order_date > '2024-01-01'
'''
loader = DataLoader(query, file_type='sql')
df = loader.load(con=engine)

# Load with aggregation
query = '''
    SELECT 
        department,
        COUNT(*) as num_employees,
        AVG(salary) as avg_salary
    FROM employees
    GROUP BY department
'''
loader = DataLoader(query, file_type='sql')
df = loader.load(con=engine)

# Parameterized query (prevents SQL injection)
query = 'SELECT * FROM users WHERE age > %(min_age)s'
loader = DataLoader(query, file_type='sql')
df = loader.load(con=engine, params={'min_age': 18})
```

**Database Connection Strings:**

```python
# PostgreSQL
engine = create_engine('postgresql://user:password@host:5432/database')

# MySQL
engine = create_engine('mysql+pymysql://user:password@host:3306/database')

# SQLite
engine = create_engine('sqlite:///path/to/database.db')

# Microsoft SQL Server
engine = create_engine('mssql+pyodbc://user:password@host/database?driver=SQL+Server')

# Oracle
engine = create_engine('oracle+cx_oracle://user:password@host:1521/database')
```

**SQL Loading Parameters:**
```python
con=engine           # REQUIRED: Database connection
params={'key': val}  # Query parameters (prevents SQL injection)
parse_dates=['col']  # Parse date columns
chunksize=10000     # Read in chunks (returns iterator)
```

**When to Use SQL Loading:**
- Loading from relational databases
- Need to filter data at source
- Working with large datasets
- Complex joins and aggregations
- When you need fresh data
- Production databases

**Best Practices:**
```python
# 1. Always use parameterized queries
query = 'SELECT * FROM users WHERE age > %(min_age)s'
df = loader.load(con=engine, params={'min_age': 18})

# 2. Use chunking for large datasets
query = 'SELECT * FROM large_table'
loader = DataLoader(query, file_type='sql')
chunks = loader.load(con=engine, chunksize=10000)
for chunk in chunks:
    process(chunk)

# 3. Filter at database level, not pandas
# Good: Filter in SQL
query = 'SELECT * FROM users WHERE active = true'
# Bad: Load all then filter
# query = 'SELECT * FROM users'
# df = df[df['active'] == True]

# 4. Select only needed columns
query = 'SELECT id, name, email FROM users'  # Good
# query = 'SELECT * FROM users'  # Bad if you don't need all columns
```

---

### 6. Load from URL

```python
from skyulf_mlflow_library.data_ingestion import DataLoader

# Load CSV from URL
url = 'https://example.com/data.csv'
loader = DataLoader(url)
df = loader.load()

# With parameters
url = 'https://example.com/data.csv'
loader = DataLoader(url, sep=';', encoding='utf-8')
df = loader.load()

# Load JSON from API
url = 'https://api.example.com/data.json'
loader = DataLoader(url)
df = loader.load()

# Load from GitHub
url = 'https://raw.githubusercontent.com/user/repo/main/data.csv'
loader = DataLoader(url)
df = loader.load()
```

**When to Use URL Loading:**
- Public datasets
- API endpoints
- GitHub repositories
- Cloud storage with public access
- Web scraping

---

### 7. Load from Python Objects

#### From Dictionary

```python
from skyulf_mlflow_library.data_ingestion import DataLoader

# Dictionary with lists (columns)
data = {
    'name': ['Alice', 'Bob', 'Charlie'],
    'age': [25, 30, 35],
    'city': ['NYC', 'LA', 'SF']
}
df = DataLoader.from_dict(data)

print(df)
#       name  age city
# 0    Alice   25  NYC
# 1      Bob   30   LA
# 2  Charlie   35   SF

# With index
df = DataLoader.from_dict(data, orient='index')

# With custom columns
data = [[1, 2, 3], [4, 5, 6]]
df = DataLoader.from_dict(data, columns=['A', 'B', 'C'])
```

#### From Records (List of Dictionaries)

```python
from skyulf_mlflow_library.data_ingestion import DataLoader

# List of dictionaries (rows)
records = [
    {'name': 'Alice', 'age': 25, 'city': 'NYC'},
    {'name': 'Bob', 'age': 30, 'city': 'LA'},
    {'name': 'Charlie', 'age': 35, 'city': 'SF'}
]
df = DataLoader.from_records(records)

print(df)
#       name  age city
# 0    Alice   25  NYC
# 1      Bob   30   LA
# 2  Charlie   35   SF

# With specific columns
df = DataLoader.from_records(records, columns=['name', 'age'])

# With custom index
df = DataLoader.from_records(records, index='name')
```

**When to Use:**
- Creating test data
- Converting API responses
- Processing nested data structures
- Creating DataFrames programmatically

---

## 💾 DataSaver

Save DataFrames to various file formats.

### Class Signature

```python
from skyulf_mlflow_library.data_ingestion import DataSaver

DataSaver(
    destination: Union[str, Path],
    file_type: Optional[str] = None,
    **kwargs
)
```

**Parameters:**
- `destination` (str | Path): Path where to save the data
- `file_type` (str, optional): File type to save. Auto-detected from extension if not provided
- `**kwargs`: Additional arguments passed to pandas to_* methods

---

### Save to CSV

```python
from skyulf_mlflow_library.data_ingestion import DataSaver
import pandas as pd

df = pd.DataFrame({'a': [1, 2, 3], 'b': [4, 5, 6]})

# Basic save
saver = DataSaver('output.csv')
saver.save(df)

# Without index
saver = DataSaver('output.csv', index=False)
saver.save(df)

# With custom separator
saver = DataSaver('output.csv', sep=';', index=False)
saver.save(df)

# With encoding
saver = DataSaver('output.csv', encoding='utf-8', index=False)
saver.save(df)

# Select columns
saver = DataSaver('output.csv', columns=['a'], index=False)
saver.save(df)
```

**Common CSV Save Parameters:**
```python
sep=','           # Column separator
index=False       # Don't save index
header=True       # Include header
encoding='utf-8'  # File encoding
columns=['a','b'] # Select columns to save
float_format='%.2f' # Format floats
```

---

### Save to Excel

```python
from skyulf_mlflow_library.data_ingestion import DataSaver

# Basic save
saver = DataSaver('output.xlsx')
saver.save(df)

# Without index
saver = DataSaver('output.xlsx', index=False)
saver.save(df)

# Specific sheet name
saver = DataSaver('output.xlsx', sheet_name='Data', index=False)
saver.save(df)

# Save multiple sheets
with pd.ExcelWriter('output.xlsx') as writer:
    df1.to_excel(writer, sheet_name='Sheet1', index=False)
    df2.to_excel(writer, sheet_name='Sheet2', index=False)
```

**Common Excel Save Parameters:**
```python
sheet_name='Sheet1'  # Sheet name
index=False         # Don't save index
columns=['a','b']   # Select columns
```

---

### Save to JSON

```python
from skyulf_mlflow_library.data_ingestion import DataSaver

# Records format (list of dictionaries)
saver = DataSaver('output.json', orient='records')
saver.save(df)

# Pretty print
saver = DataSaver('output.json', orient='records', indent=4)
saver.save(df)

# JSON Lines format (one object per line)
saver = DataSaver('output.json', orient='records', lines=True)
saver.save(df)
```

**Common JSON Save Parameters:**
```python
orient='records'  # Output format
indent=4         # Pretty print with indentation
lines=True       # JSON Lines format
```

---

### Save to Parquet

```python
from skyulf_mlflow_library.data_ingestion import DataSaver

# Basic save
saver = DataSaver('output.parquet')
saver.save(df)

# With compression
saver = DataSaver('output.parquet', compression='gzip')
saver.save(df)

# Other compression options
saver = DataSaver('output.parquet', compression='snappy')
saver.save(df)

saver = DataSaver('output.parquet', compression='brotli')
saver.save(df)
```

**Common Parquet Save Parameters:**
```python
compression='gzip'   # Compression algorithm
engine='pyarrow'    # Engine to use
index=False         # Don't save index
```

**Compression Comparison:**
- `snappy`: Fastest, moderate compression (default)
- `gzip`: Slower, better compression
- `brotli`: Slowest, best compression
- `None`: No compression (fastest, largest files)

---

## 🚀 Convenience Functions

### load_data()

Quick function for loading data without creating a loader object.

```python
from skyulf_mlflow_library.data_ingestion import load_data

# Load CSV
df = load_data('data.csv')

# With parameters
df = load_data('data.csv', sep=';', encoding='latin1')

# Load Excel
df = load_data('data.xlsx', sheet_name='Sales')

# Load JSON
df = load_data('data.json', orient='records')

# Load Parquet
df = load_data('data.parquet', columns=['col1', 'col2'])
```

**When to Use:**
- One-off data loading
- Quick scripts
- Interactive analysis
- When you don't need reusability

---

### save_data()

Quick function for saving data without creating a saver object.

```python
from skyulf_mlflow_library.data_ingestion import save_data
import pandas as pd

df = pd.DataFrame({'a': [1, 2, 3], 'b': [4, 5, 6]})

# Save CSV
save_data(df, 'output.csv', index=False)

# Save Excel
save_data(df, 'output.xlsx', sheet_name='Data', index=False)

# Save JSON
save_data(df, 'output.json', orient='records', indent=4)

# Save Parquet
save_data(df, 'output.parquet', compression='gzip')
```

**When to Use:**
- One-off saves
- Quick exports
- End of scripts
- When you don't need reusability

---

## 📊 Complete Examples

### Example 1: ETL Pipeline

```python
from skyulf_mlflow_library.data_ingestion import DataLoader, DataSaver
from sqlalchemy import create_engine

# 1. Extract: Load from SQL
engine = create_engine('postgresql://user:pass@localhost/db')
query = 'SELECT * FROM sales WHERE date > %(start_date)s'
loader = DataLoader(query, file_type='sql')
df = loader.load(con=engine, params={'start_date': '2024-01-01'})

# 2. Transform: Process data
df['total'] = df['quantity'] * df['price']
df = df[df['total'] > 100]

# 3. Load: Save to Parquet for analysis
saver = DataSaver('processed_sales.parquet', compression='gzip')
saver.save(df)

# Also save CSV for business users
saver = DataSaver('sales_report.csv', index=False, float_format='%.2f')
saver.save(df)
```

### Example 2: Multi-Source Data Integration

```python
from skyulf_mlflow_library.data_ingestion import DataLoader, DataSaver

# Load from different sources
customers = DataLoader('customers.csv').load()
orders = DataLoader('orders.json', orient='records').load()
inventory = DataLoader('inventory.xlsx', sheet_name='Current').load()

# Merge data
import pandas as pd
merged = customers.merge(orders, on='customer_id')
merged = merged.merge(inventory, on='product_id')

# Save integrated data
saver = DataSaver('integrated_data.parquet', compression='snappy')
saver.save(merged)
```

### Example 3: API Data Collection

```python
from skyulf_mlflow_library.data_ingestion import DataLoader, DataSaver

# Load from API endpoint
api_url = 'https://api.example.com/data?date=2024-01-01'
loader = DataLoader(api_url)
df = loader.load()

# Process and save
df['timestamp'] = pd.to_datetime(df['timestamp'])
df = df.sort_values('timestamp')

saver = DataSaver('api_data.parquet', compression='gzip')
saver.save(df)
```

### Example 4: Creating Test Data

```python
from skyulf_mlflow_library.data_ingestion import DataLoader, DataSaver

# Create test data from dict
test_data = {
    'user_id': [1, 2, 3, 4, 5],
    'age': [25, 30, 35, 40, 45],
    'purchases': [10, 20, 15, 25, 30]
}
df = DataLoader.from_dict(test_data)

# Save for testing
saver = DataSaver('test_data.csv', index=False)
saver.save(df)
```

---

## 🎯 Best Practices

### 1. Format Selection

```python
# Choose format based on use case:

# CSV: Universal compatibility
save_data(df, 'data.csv', index=False)

# Parquet: Performance and storage
save_data(df, 'data.parquet', compression='snappy')

# Excel: Business users
save_data(df, 'report.xlsx', sheet_name='Report', index=False)

# JSON: API integration
save_data(df, 'data.json', orient='records', indent=4)
```

### 2. Error Handling

```python
from skyulf_mlflow_library.data_ingestion import DataLoader

try:
    loader = DataLoader('data.csv')
    df = loader.load()
except FileNotFoundError:
    print("File not found")
except pd.errors.EmptyDataError:
    print("File is empty")
except Exception as e:
    print(f"Error loading data: {e}")
```

### 3. Memory Management

```python
# For large files, use chunking
loader = DataLoader('large_file.csv')
chunks = loader.load(chunksize=10000)

for chunk in chunks:
    process(chunk)
    # Each chunk is processed separately
```

### 4. SQL Query Optimization

```python
from sqlalchemy import create_engine

engine = create_engine('postgresql://...')

# Good: Filter at database
query = '''
    SELECT id, name, email
    FROM users
    WHERE active = true
    AND created_date > '2024-01-01'
'''
loader = DataLoader(query, file_type='sql')
df = loader.load(con=engine)

# Bad: Load everything then filter
# query = 'SELECT * FROM users'
# df = loader.load(con=engine)
# df = df[df['active'] == True]
```

### 5. Path Management

```python
from pathlib import Path

# Use pathlib for better path handling
data_dir = Path('data')
data_dir.mkdir(exist_ok=True)

file_path = data_dir / 'output.csv'
saver = DataSaver(file_path, index=False)
saver.save(df)
```

---

## ⚠️ Common Pitfalls

### 1. Encoding Issues

```python
# Problem: Can't read file with special characters
# Solution: Specify encoding
loader = DataLoader('data.csv', encoding='latin1')
df = loader.load()

# Or try UTF-8 with error handling
loader = DataLoader('data.csv', encoding='utf-8', errors='ignore')
df = loader.load()
```

### 2. Memory Issues with Large Files

```python
# Problem: File too large for memory
# Solution: Use chunking
loader = DataLoader('huge_file.csv')
chunks = loader.load(chunksize=10000)

results = []
for chunk in chunks:
    processed = process_chunk(chunk)
    results.append(processed)

df = pd.concat(results, ignore_index=True)
```

### 3. SQL Connection Not Closed

```python
# Problem: Connection leaks
# Solution: Use context manager
from sqlalchemy import create_engine

engine = create_engine('postgresql://...')

try:
    loader = DataLoader(query, file_type='sql')
    df = loader.load(con=engine)
finally:
    engine.dispose()

# Or use with statement
with engine.connect() as con:
    loader = DataLoader(query, file_type='sql')
    df = loader.load(con=con)
```

---

## 📋 Quick Reference

### Load Data
```python
from skyulf_mlflow_library.data_ingestion import DataLoader, load_data

# Object-oriented
loader = DataLoader('data.csv')
df = loader.load()

# Functional
df = load_data('data.csv')

# SQL
df = DataLoader(query, file_type='sql').load(con=engine)

# From dict
df = DataLoader.from_dict({'col': [1, 2, 3]})

# From records
df = DataLoader.from_records([{'col': 1}, {'col': 2}])
```

### Save Data
```python
from skyulf_mlflow_library.data_ingestion import DataSaver, save_data

# Object-oriented
saver = DataSaver('output.csv', index=False)
saver.save(df)

# Functional
save_data(df, 'output.csv', index=False)
```

### Supported Formats
| Format | Load | Save | Best For |
|--------|------|------|----------|
| CSV | ✅ | ✅ | Universal compatibility |
| Excel | ✅ | ✅ | Business users |
| JSON | ✅ | ✅ | APIs, nested data |
| Parquet | ✅ | ✅ | Performance, big data |
| SQL | ✅ | ❌ | Databases |

---

## 🔗 Related Documentation

- [Preprocessing](preprocessing.md) - Clean and prepare loaded data
- [Feature Engineering](feature_engineering.md) - Transform features
- [Model Training](model_training.md) - Train models on loaded data

---

## 💡 Tips

1. **Always specify encoding for non-English data**
   ```python
   loader = DataLoader('data.csv', encoding='utf-8')
   ```

2. **Use Parquet for large datasets**
   ```python
   # 10x faster and 10x smaller than CSV
   save_data(df, 'data.parquet', compression='snappy')
   ```

3. **Filter in SQL, not pandas**
   ```python
   # Good: Filter 1M rows to 1K at database
   query = 'SELECT * FROM users WHERE active = true'
   
   # Bad: Load 1M rows then filter
   # df = load_all()
   # df = df[df['active'] == True]
   ```

4. **Use parameterized SQL queries**
   ```python
   # Prevents SQL injection
   query = 'SELECT * FROM users WHERE age > %(min_age)s'
   df = loader.load(con=engine, params={'min_age': 18})
   ```

5. **Save index=False for CSVs**
   ```python
   # Cleaner output without row numbers
   save_data(df, 'output.csv', index=False)
   ```
