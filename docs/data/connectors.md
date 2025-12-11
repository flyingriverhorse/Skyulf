# Connectors

Connectors normalize access to different data sources so the ingestion pipeline can fetch data and discover schemas in a consistent way. **All connectors return Polars DataFrames** for efficient I/O.

## Connector Interface

All connectors implement the `BaseConnector` interface:

```python
from abc import ABC, abstractmethod
import polars as pl

class BaseConnector(ABC):
    async def connect(self) -> bool: ...
    async def get_schema(self) -> Dict[str, str]: ...
    async def fetch_data(self, query=None, limit=None) -> pl.DataFrame: ...
    async def validate(self) -> bool: ...
```

## Supported Connectors

| Type | Connector | Formats |
|------|-----------|---------|
| `file` | `LocalFileConnector` | CSV, Excel (.xlsx/.xls), Parquet, JSON |
| `postgres`, `mysql`, `sqlite`, `snowflake` | `DatabaseConnector` | SQL via SQLAlchemy |
| `api` | `ApiConnector` | REST endpoints (JSON, CSV responses) |

---

## File Connector

The `LocalFileConnector` handles local file uploads.

```python
import asyncio
from core.data_ingestion.connectors.file import LocalFileConnector

async def load_csv():
    connector = LocalFileConnector("data/customers.csv")
    await connector.connect()
    
    # Get schema (column names → types)
    schema = await connector.get_schema()
    print(schema)  # {'id': 'Int64', 'name': 'Utf8', 'age': 'Int64'}
    
    # Fetch data (returns Polars DataFrame)
    df = await connector.fetch_data(limit=100)
    print(df.head())
    
    return df

asyncio.run(load_csv())
```

### Supported Extensions
- `.csv` - Comma-separated values
- `.xlsx`, `.xls` - Excel spreadsheets
- `.parquet` - Apache Parquet
- `.json` - JSON arrays

---

## SQL Connector

The `DatabaseConnector` connects to SQL databases using SQLAlchemy and fetches data with Polars.

```python
import asyncio
from core.data_ingestion.connectors.sql import DatabaseConnector

async def load_from_postgres():
    connector = DatabaseConnector(
        connection_string="postgresql+psycopg2://user:pass@localhost/mydb",
        table_name="customers"  # Or use 'query' for custom SQL
    )
    await connector.connect()
    
    # Fetch with limit
    df = await connector.fetch_data(limit=1000)
    print(f"Loaded {len(df)} rows")
    
    return df

asyncio.run(load_from_postgres())
```

### Configuration

| Parameter | Description |
|-----------|-------------|
| `connection_string` | SQLAlchemy URL (e.g., `postgresql+psycopg2://user:pass@host/db`) |
| `table_name` | Table to query (generates `SELECT * FROM {table_name}`) |
| `query` | Custom SQL query (overrides `table_name`) |

### Supported Databases
- PostgreSQL (`postgresql+psycopg2://`)
- MySQL (`mysql+pymysql://`)
- SQLite (`sqlite:///path/to/db.sqlite`)
- Snowflake (`snowflake://`)

---

## API Connector

The `ApiConnector` fetches data from REST endpoints.

```python
import asyncio
from core.data_ingestion.connectors.api import ApiConnector

async def load_from_api():
    connector = ApiConnector(
        url="https://api.example.com/users",
        method="GET",
        headers={"Authorization": "Bearer token123"},
        params={"status": "active"},
        data_key="users"  # Extract 'users' array from response
    )
    await connector.connect()
    
    df = await connector.fetch_data(limit=50)
    print(df.head())
    
    return df

asyncio.run(load_from_api())
```

### Configuration

| Parameter | Description |
|-----------|-------------|
| `url` | Endpoint URL |
| `method` | HTTP method (default: `GET`) |
| `headers` | Optional headers dict |
| `params` | Optional query parameters dict |
| `data_key` | Key to extract data from JSON response (e.g., `"items"` for `{"items": [...]}`) |

### Supported Response Types
- `application/json` - Parsed as JSON, converted to DataFrame
- `text/csv` - Parsed as CSV
