"""
Async CRUD Operations for FastAPI

This module provides async equivalents of the Flask db/crud.py operations,
maintaining the same interface and functionality while adding async support.

Usage (examples):
    from fastapi_app.core.database import crud
    await crud.create("users", {"name": "Alice"}, settings)
    await crud.read("users", {"id": 1}, settings, one=True)
"""

from typing import Any, Dict, List, Optional, Union
import re
import logging

from sqlalchemy import text as sa_text
from sqlalchemy.ext.asyncio import AsyncSession

from .adapter import async_session_or_connection, get_db_type, DatabaseType
from .data_sources import async_sqlite_queries, async_postgres_queries, async_data_sources_crud
from config import Settings

logger = logging.getLogger(__name__)

# Basic identifier whitelist (table/collection/column names)
_IDENTIFIER_RE = re.compile(r"^[A-Za-z_][A-Za-z0-9_]*$")


def _check_identifier(name: str) -> None:
    """Validate database identifier to prevent SQL injection."""
    if not _IDENTIFIER_RE.match(name):
        raise ValueError(f"Invalid identifier: {name}")


def _row_to_dict(row):
    """Convert SQLAlchemy Row to dict."""
    if row is None:
        return None
    
    # Handle SQLAlchemy 2.0+ Row objects
    if hasattr(row, '_mapping'):
        return dict(row._mapping)
    
    # Fallback for older versions or other row types
    try:
        return dict(row)
    except Exception:
        try:
            return {k: v for k, v in row.items()}
        except Exception:
            return {}


def _build_sql_from_filter(
    filt: Any, placeholder_style: str = ":", param_counter: int = 0
):
    """
    Build SQL WHERE clause from filter dict.
    Supports nested dicts and logical operators $and and $or.
    """
    if not filt:
        return "", ({} if placeholder_style == ":" else ()), param_counter

    # Handle logical operators
    if isinstance(filt, dict) and any(k in ("$or", "$and") for k in filt.keys()):
        parts = []
        params = {} if placeholder_style in (":", "pyformat") else []
        
        for op in ("$and", "$or"):
            if op in filt:
                subfilters = filt[op]
                if not isinstance(subfilters, list):
                    raise ValueError(f"{op} must be a list of filter dicts")
                
                subparts = []
                for sf in subfilters:
                    clause, subparams, param_counter = _build_sql_from_filter(
                        sf, placeholder_style, param_counter
                    )
                    if clause:
                        clean = clause[7:] if clause.startswith(" WHERE ") else clause
                        subparts.append(f"({clean})")
                        if placeholder_style in (":", "pyformat"):
                            params.update(subparams)
                        else:
                            params.extend(subparams)
                
                joiner = " AND " if op == "$and" else " OR "
                if subparts:
                    parts.append(joiner.join(subparts))

        clause = " WHERE " + (" AND ".join(parts) if parts else "")
        return clause, (params if placeholder_style == ":" else tuple(params)), param_counter

    # Handle simple field: value filters
    parts = []
    params = {} if placeholder_style in (":", "pyformat") else []
    
    if isinstance(filt, dict):
        items = filt.items()
    else:
        items = []

    for k, v in items:
        _check_identifier(k)
        
        if isinstance(v, dict):
            # Handle operators like $in, $gt, etc.
            for op, val in v.items():
                if op == "$in":
                    if placeholder_style == ":":
                        place_names = []
                        for item in val:
                            pname = f"p{param_counter}"
                            param_counter += 1
                            place_names.append(f":{pname}")
                            params[pname] = item
                        parts.append(f"{k} IN ({', '.join(place_names)})")
                    elif placeholder_style == "pyformat":
                        place_names = []
                        for item in val:
                            pname = f"p{param_counter}"
                            param_counter += 1
                            place_names.append(f"%({pname})s")
                            params[pname] = item
                        parts.append(f"{k} IN ({', '.join(place_names)})")
                    else:
                        placeholders = ", ".join(["%s"] * len(val))
                        parts.append(f"{k} IN ({placeholders})")
                        params.extend(list(val))
                else:
                    sql_op = {
                        "$gt": ">",
                        "$lt": "<", 
                        "$gte": ">=",
                        "$lte": "<=",
                        "$ne": "<>",
                        "$eq": "=",
                        "$like": "LIKE",
                        "$ilike": "ILIKE",
                    }.get(op)
                    
                    if sql_op is None:
                        raise ValueError(f"Unsupported operator: {op}")
                    
                    if placeholder_style == ":":
                        pname = f"p{param_counter}"
                        param_counter += 1
                        parts.append(f"{k} {sql_op} :{pname}")
                        params[pname] = val
                    elif placeholder_style == "pyformat":
                        pname = f"p{param_counter}"
                        param_counter += 1
                        parts.append(f"{k} {sql_op} %({pname})s")
                        params[pname] = val
                    else:
                        parts.append(f"{k} {sql_op} %s")
                        params.append(val)
        else:
            # Simple equality
            if placeholder_style == ":":
                parts.append(f"{k} = :{k}")
                params[k] = v
            elif placeholder_style == "pyformat":
                parts.append(f"{k} = %({k})s")
                params[k] = v
            else:
                parts.append(f"{k} = %s")
                params.append(v)

    clause = " WHERE " + " AND ".join(parts) if parts else ""
    return clause, (params if placeholder_style == ":" else tuple(params)), param_counter


def _build_where_clause_sql(filter: Dict[str, Any], placeholder_style: str = ":"):
    """Build WHERE clause from filter dict."""
    clause, params, _ = _build_sql_from_filter(filter or {}, placeholder_style)
    return clause, params


async def create(
    name: str, 
    data: Dict[str, Any], 
    settings: Settings,
    config: Optional[Dict] = None
) -> Any:
    """
    Insert a record/document into table/collection.
    
    Args:
        name: Table or collection name
        data: Data to insert
        settings: Application settings
        config: Optional connection config override
        
    Returns:
        Dict containing the inserted record or insertion metadata
    """
    if not isinstance(data, dict):
        raise ValueError("data must be a dict")

    _check_identifier(name)
    db_type = get_db_type(settings)

    # Delegate data_sources operations to specialized module
    if name == "data_sources":
        return await async_data_sources_crud.create(settings, data)

    if db_type in (DatabaseType.POSTGRES, DatabaseType.POSTGRESQL, DatabaseType.SQLITE):
        async with async_session_or_connection(settings, config) as session:
            cols = ", ".join(data.keys())
            placeholders = ", ".join(f":{k}" for k in data.keys())
            
            if db_type == DatabaseType.SQLITE:
                # SQLite: Insert then fetch by rowid
                sql = sa_text(f"INSERT INTO {name} ({cols}) VALUES ({placeholders})")
                await session.execute(sql, data)
                await session.commit()
                
                # Get last inserted row
                rid_result = await session.execute(sa_text("SELECT last_insert_rowid() AS rid"))
                rid = rid_result.scalar()
                
                if rid is not None:
                    sel_result = await session.execute(
                        sa_text(f"SELECT * FROM {name} WHERE rowid = :rid"),
                        {"rid": rid}
                    )
                    row = sel_result.fetchone()
                    return _row_to_dict(row) if row else None
                return None
            else:
                # PostgreSQL: Use RETURNING
                sql = sa_text(f"INSERT INTO {name} ({cols}) VALUES ({placeholders}) RETURNING *")
                result = await session.execute(sql, data)
                await session.commit()
                row = result.fetchone()
                return _row_to_dict(row) if row else None

    elif db_type in (DatabaseType.MYSQL, DatabaseType.MARIADB, DatabaseType.SNOWFLAKE):
        async with async_session_or_connection(settings, config) as conn:
            cols = ", ".join(data.keys())
            placeholders = ", ".join([f"%({k})s" for k in data.keys()])
            sql = f"INSERT INTO {name} ({cols}) VALUES ({placeholders})"
            
            if hasattr(conn, 'execute'):
                # Async connection (MySQL/Snowflake async wrapper)
                await conn.execute(sql, data)
                await conn.commit()
                # Note: Getting lastrowid in async context depends on the specific driver
                return {"success": True}
            else:
                # Fallback for sync wrapper
                cursor = conn.cursor()
                cursor.execute(sql, data)
                conn.commit()
                lastrowid = getattr(cursor, 'lastrowid', None)
                cursor.close()
                return {"lastrowid": lastrowid}

    elif db_type in (DatabaseType.MONGODB, DatabaseType.MONGO):
        async with async_session_or_connection(settings, config) as db_obj:
            collection = db_obj[name]
            result = await collection.insert_one(data)
            return {"inserted_id": str(result.inserted_id)}

    else:
        raise RuntimeError(f"Unsupported database type: {db_type}")


async def read(
    name: str,
    settings: Settings,
    filter: Optional[Dict[str, Any]] = None,
    config: Optional[Dict] = None,
    one: bool = False,
) -> Union[Dict[str, Any], List[Dict[str, Any]]]:
    """
    Read records/documents from table/collection.
    
    Args:
        name: Table or collection name
        filter: Filter conditions
        settings: Application settings
        config: Optional connection config override
        one: Return single record if True
        
    Returns:
        List of records or single record if one=True
    """
    _check_identifier(name)
    db_type = get_db_type(settings)

    if name == "data_sources":
        return await async_data_sources_crud.read(settings, filter, one=one)

    if db_type in (DatabaseType.POSTGRES, DatabaseType.POSTGRESQL, DatabaseType.SQLITE):
        async with async_session_or_connection(settings, config) as session:
            clause, params = _build_where_clause_sql(filter or {}, placeholder_style=":")
            sql = sa_text(f"SELECT * FROM {name}{clause}")
            result = await session.execute(sql, params)
            rows_raw = result.fetchall()
            rows = [_row_to_dict(r) for r in rows_raw]
            return rows[0] if one and rows else rows

    elif db_type in (DatabaseType.MYSQL, DatabaseType.MARIADB, DatabaseType.SNOWFLAKE):
        async with async_session_or_connection(settings, config) as conn:
            clause, params = _build_where_clause_sql(filter or {}, placeholder_style="pyformat")
            sql = f"SELECT * FROM {name}{clause}"
            
            if hasattr(conn, 'execute'):
                # Async connection
                result = await conn.execute(sql, params)
                # This would need to be adapted based on the specific async driver
                rows = result.fetchall() if hasattr(result, 'fetchall') else []
            else:
                # Sync wrapper
                cursor = conn.cursor()
                cursor.execute(sql, params if isinstance(params, dict) else tuple(params))
                cols = [c[0] for c in cursor.description] if cursor.description else []
                rows = [dict(zip(cols, row)) for row in cursor.fetchall()]
                cursor.close()
            
            return rows[0] if one and rows else rows

    elif db_type in (DatabaseType.MONGODB, DatabaseType.MONGO):
        async with async_session_or_connection(settings, config) as db_obj:
            collection = db_obj[name]
            
            if filter is None:
                cursor = collection.find()
                items = await cursor.to_list(length=None)
            elif isinstance(filter, list):
                # Aggregation pipeline
                cursor = collection.aggregate(filter)
                items = await cursor.to_list(length=None)
            elif isinstance(filter, dict) and "$pipeline" in filter:
                pipeline = filter["$pipeline"]
                if not isinstance(pipeline, list):
                    raise ValueError("$pipeline must be a list")
                cursor = collection.aggregate(pipeline)
                items = await cursor.to_list(length=None)
            else:
                if one:
                    item = await collection.find_one(filter)
                    return item
                cursor = collection.find(filter)
                items = await cursor.to_list(length=None)
            
            return items

    else:
        raise RuntimeError(f"Unsupported database type: {db_type}")


async def update(
    name: str,
    filter: Dict[str, Any],
    update_data: Dict[str, Any],
    settings: Settings,
    config: Optional[Dict] = None,
    many: bool = False,
) -> Any:
    """
    Update matching records/documents.
    
    Args:
        name: Table or collection name
        filter: Filter conditions for records to update
        update_data: Data to update
        settings: Application settings
        config: Optional connection config override
        many: Update multiple records if True
        
    Returns:
        Update result (affected count or result object)
    """
    if not isinstance(filter, dict) or not isinstance(update_data, dict):
        raise ValueError("filter and update_data must be dicts")

    _check_identifier(name)
    db_type = get_db_type(settings)

    if name == "data_sources":
        return await async_data_sources_crud.update(settings, filter, update_data)

    if db_type in (DatabaseType.POSTGRES, DatabaseType.POSTGRESQL, DatabaseType.SQLITE):
        async with async_session_or_connection(settings, config) as session:
            set_parts = ", ".join(f"{k} = :u_{k}" for k in update_data.keys())
            params = {f"u_{k}": v for k, v in update_data.items()}
            clause, where_params = _build_where_clause_sql(filter, placeholder_style=":")
            params.update(where_params)
            
            sql = sa_text(f"UPDATE {name} SET {set_parts}{clause}")
            result = await session.execute(sql, params)
            await session.commit()
            return {"affected_rows": result.rowcount}

    elif db_type in (DatabaseType.MYSQL, DatabaseType.MARIADB, DatabaseType.SNOWFLAKE):
        async with async_session_or_connection(settings, config) as conn:
            set_parts = ", ".join(f"{k} = %({k})s" for k in update_data.keys())
            clause, where_params = _build_where_clause_sql(filter, placeholder_style="pyformat")
            params = dict(update_data)
            if isinstance(where_params, dict):
                params.update(where_params)
            
            sql = f"UPDATE {name} SET {set_parts}{clause}"
            
            if hasattr(conn, 'execute'):
                result = await conn.execute(sql, params)
                await conn.commit()
                return {"affected_rows": getattr(result, 'rowcount', 0)}
            else:
                cursor = conn.cursor()
                cursor.execute(sql, params)
                conn.commit()
                affected = cursor.rowcount
                cursor.close()
                return {"affected_rows": affected}

    elif db_type in (DatabaseType.MONGODB, DatabaseType.MONGO):
        async with async_session_or_connection(settings, config) as db_obj:
            collection = db_obj[name]
            
            if many:
                result = await collection.update_many(filter, {"$set": update_data})
            else:
                result = await collection.update_one(filter, {"$set": update_data})
            
            return {
                "matched_count": result.matched_count,
                "modified_count": result.modified_count
            }

    else:
        raise RuntimeError(f"Unsupported database type: {db_type}")


async def delete(
    name: str,
    filter: Dict[str, Any],
    settings: Settings,
    config: Optional[Dict] = None,
    many: bool = False,
) -> Any:
    """
    Delete matching records/documents.
    
    Args:
        name: Table or collection name
        filter: Filter conditions for records to delete
        settings: Application settings
        config: Optional connection config override
        many: Delete multiple records if True
        
    Returns:
        Delete result (affected count or result object)
    """
    if not isinstance(filter, dict):
        raise ValueError("filter must be a dict")

    _check_identifier(name)
    db_type = get_db_type(settings)

    if name == "data_sources":
        return await async_data_sources_crud.delete(settings, filter)

    if db_type in (DatabaseType.POSTGRES, DatabaseType.POSTGRESQL, DatabaseType.SQLITE):
        async with async_session_or_connection(settings, config) as session:
            clause, params = _build_where_clause_sql(filter, placeholder_style=":")
            sql = sa_text(f"DELETE FROM {name}{clause}")
            result = await session.execute(sql, params)
            await session.commit()
            return {"affected_rows": result.rowcount}

    elif db_type in (DatabaseType.MYSQL, DatabaseType.MARIADB, DatabaseType.SNOWFLAKE):
        async with async_session_or_connection(settings, config) as conn:
            clause, params = _build_where_clause_sql(filter, placeholder_style="pyformat")
            sql = f"DELETE FROM {name}{clause}"
            
            if hasattr(conn, 'execute'):
                result = await conn.execute(sql, params)
                await conn.commit()
                return {"affected_rows": getattr(result, 'rowcount', 0)}
            else:
                cursor = conn.cursor()
                cursor.execute(sql, params if isinstance(params, dict) else tuple(params))
                conn.commit()
                affected = cursor.rowcount
                cursor.close()
                return {"affected_rows": affected}

    elif db_type in (DatabaseType.MONGODB, DatabaseType.MONGO):
        async with async_session_or_connection(settings, config) as db_obj:
            collection = db_obj[name]
            
            if many:
                result = await collection.delete_many(filter)
            else:
                result = await collection.delete_one(filter)
            
            return {"deleted_count": result.deleted_count}

    else:
        raise RuntimeError(f"Unsupported database type: {db_type}")


# Convenience functions that maintain Flask-like interface
async def count(
    name: str,
    settings: Settings,
    filter: Optional[Dict[str, Any]] = None,
    config: Optional[Dict] = None
) -> int:
    """Count records matching filter."""
    _check_identifier(name)
    db_type = get_db_type(settings)

    if db_type in (DatabaseType.POSTGRES, DatabaseType.POSTGRESQL, DatabaseType.SQLITE):
        async with async_session_or_connection(settings, config) as session:
            clause, params = _build_where_clause_sql(filter or {}, placeholder_style=":")
            sql = sa_text(f"SELECT COUNT(*) FROM {name}{clause}")
            result = await session.execute(sql, params)
            return result.scalar() or 0

    elif db_type in (DatabaseType.MYSQL, DatabaseType.MARIADB, DatabaseType.SNOWFLAKE):
        async with async_session_or_connection(settings, config) as conn:
            clause, params = _build_where_clause_sql(filter or {}, placeholder_style="pyformat")
            sql = f"SELECT COUNT(*) FROM {name}{clause}"
            
            if hasattr(conn, 'execute'):
                result = await conn.execute(sql, params)
                return result.fetchone()[0] if hasattr(result, 'fetchone') else 0
            else:
                cursor = conn.cursor()
                cursor.execute(sql, params if isinstance(params, dict) else tuple(params))
                result = cursor.fetchone()
                cursor.close()
                return result[0] if result else 0

    elif db_type in (DatabaseType.MONGODB, DatabaseType.MONGO):
        async with async_session_or_connection(settings, config) as db_obj:
            collection = db_obj[name]
            return await collection.count_documents(filter or {})

    else:
        raise RuntimeError(f"Unsupported database type: {db_type}")