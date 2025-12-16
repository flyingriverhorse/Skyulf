"""
Async CRUD Operations for FastAPI

This module provides async equivalents of the Flask db/crud.py operations,
maintaining the same interface and functionality while adding async support.

Usage (examples):
    from fastapi_app.core.database import crud
    await crud.create("users", {"name": "Alice"}, settings)
    await crud.read("users", {"id": 1}, settings, one=True)
"""
# pylint: disable=broad-exception-caught

import logging
import re
from typing import Any, Dict, List, Optional, Union, cast

from sqlalchemy import column, literal_column, select, table
from sqlalchemy import text as sa_text

from backend.config import Settings

from .adapter import DatabaseType, async_session_or_connection, get_db_type
from .data_sources import async_data_sources_crud

logger = logging.getLogger(__name__)

# Basic identifier whitelist (table/collection/column names)
_IDENTIFIER_RE = re.compile(r"^[A-Za-z_][A-Za-z0-9_]*$")


def _check_identifier(name: str) -> None:
    """Validate database identifier to prevent SQL injection."""
    if not _IDENTIFIER_RE.match(name):
        raise ValueError(f"Invalid identifier: {name}")


def _row_to_dict(row: Any) -> Optional[Dict[str, Any]]:
    """Convert SQLAlchemy Row to dict."""
    if row is None:
        return None

    # Handle SQLAlchemy 2.0+ Row objects
    if hasattr(row, "_mapping"):
        return dict(row._mapping)

    # Fallback for older versions or other row types
    try:
        return dict(row)
    except Exception:
        try:
            return {k: v for k, v in row.items()}
        except Exception:
            return {}


def _init_params_container(style: str):
    return {} if style in (":", "pyformat") else []


def _finalize_params(params, style: str):
    return params if style in (":", "pyformat") else tuple(params)


def _merge_params(target, incoming, style: str) -> None:
    if style in (":", "pyformat"):
        target.update(incoming)
    else:
        target.extend(incoming)


def _coerce_params(params, style: str):
    if style in (":", "pyformat"):
        return params
    return list(params)


def _is_logical_filter(filt: Any) -> bool:
    return isinstance(filt, dict) and any(k in ("$or", "$and") for k in filt.keys())


def _clean_clause_prefix(clause: str) -> str:
    return clause[7:] if clause.startswith(" WHERE ") else clause


def _build_in_clause(field: str, values, style: str, counter: int):
    placeholders = []
    params = _init_params_container(style)

    for value in values:
        if style == ":":
            pname = f"p{counter}"
            counter += 1
            placeholders.append(f":{pname}")
            params[pname] = value
        elif style == "pyformat":
            pname = f"p{counter}"
            counter += 1
            placeholders.append(f"%({pname})s")
            params[pname] = value
        else:
            placeholders.append("%s")
            params.append(value)

    clause = f"{field} IN ({', '.join(placeholders)})"
    return clause, params, counter


def _build_comparison_clause(
    field: str, operator: str, value, style: str, counter: int
):
    sql_op = {
        "$gt": ">",
        "$lt": "<",
        "$gte": ">=",
        "$lte": "<=",
        "$ne": "<>",
        "$eq": "=",
        "$like": "LIKE",
        "$ilike": "ILIKE",
    }.get(operator)

    if sql_op is None:
        raise ValueError(f"Unsupported operator: {operator}")

    params = _init_params_container(style)
    if style == ":":
        pname = f"p{counter}"
        counter += 1
        params[pname] = value
        clause = f"{field} {sql_op} :{pname}"
    elif style == "pyformat":
        pname = f"p{counter}"
        counter += 1
        params[pname] = value
        clause = f"{field} {sql_op} %({pname})s"
    else:
        clause = f"{field} {sql_op} %s"
        params.append(value)

    return clause, params, counter


def _build_simple_equality(field: str, value, style: str):
    params = _init_params_container(style)
    if style == ":":
        params[field] = value
        clause = f"{field} = :{field}"
    elif style == "pyformat":
        params[field] = value
        clause = f"{field} = %({field})s"
    else:
        params.append(value)
        clause = f"{field} = %s"
    return clause, params


def _build_field_clauses(field: str, value, style: str, counter: int):
    parts = []
    params = _init_params_container(style)

    if isinstance(value, dict):
        for op, val in value.items():
            if op == "$in":
                clause, subparams, counter = _build_in_clause(
                    field, val, style, counter
                )
            else:
                clause, subparams, counter = _build_comparison_clause(
                    field, op, val, style, counter
                )
            parts.append(clause)
            _merge_params(params, subparams, style)
    else:
        clause, subparams = _build_simple_equality(field, value, style)
        parts.append(clause)
        _merge_params(params, subparams, style)

    return parts, params, counter


def _build_logical_filter_clause(filt: Dict[str, Any], style: str, counter: int):
    parts: List[str] = []
    params = _init_params_container(style)

    for operator in ("$and", "$or"):
        if operator not in filt:
            continue
        subfilters = filt[operator]
        if not isinstance(subfilters, list):
            raise ValueError(f"{operator} must be a list of filter dicts")

        subparts = []
        for subfilter in subfilters:
            clause, subparams, counter = _build_sql_from_filter(
                subfilter, style, counter
            )
            if clause:
                subparts.append(f"({_clean_clause_prefix(clause)})")
                _merge_params(params, _coerce_params(subparams, style), style)

        if subparts:
            joiner = " AND " if operator == "$and" else " OR "
            parts.append(joiner.join(subparts))

    clause = " WHERE " + (" AND ".join(parts) if parts else "")
    return clause, params, counter


def _build_field_filter_clause(filt: Dict[str, Any], style: str, counter: int):
    parts: List[str] = []
    params = _init_params_container(style)

    for key, value in filt.items():
        _check_identifier(key)
        field_parts, field_params, counter = _build_field_clauses(
            key, value, style, counter
        )
        parts.extend(field_parts)
        _merge_params(params, field_params, style)

    clause = " WHERE " + " AND ".join(parts) if parts else ""
    return clause, params, counter


def _build_sql_from_filter(
    filt: Any, placeholder_style: str = ":", param_counter: int = 0
):
    """Build a WHERE clause for the provided filter."""
    handler = _select_filter_handler(filt)
    clause, params, param_counter = handler(filt, placeholder_style, param_counter)
    finalized_params = _finalize_params(params, placeholder_style)
    return clause, finalized_params, param_counter


def _select_filter_handler(filt: Any):
    if not filt:
        return _handle_empty_filter
    if _is_logical_filter(filt):
        return _handle_logical_filter
    if isinstance(filt, dict):
        return _handle_field_filter
    return _handle_unknown_filter


def _handle_empty_filter(_filt: Any, placeholder_style: str, _counter: int):
    params = _init_params_container(placeholder_style)
    return "", params, _counter


def _handle_logical_filter(filt: Any, placeholder_style: str, counter: int):
    clause, params, counter = _build_logical_filter_clause(
        filt, placeholder_style, counter
    )
    return clause, params, counter


def _handle_field_filter(filt: Any, placeholder_style: str, counter: int):
    clause, params, counter = _build_field_filter_clause(
        filt, placeholder_style, counter
    )
    return clause, params, counter


def _handle_unknown_filter(_filt: Any, placeholder_style: str, counter: int):
    params = _init_params_container(placeholder_style)
    return "", params, counter


def _build_where_clause_sql(filter_dict: Dict[str, Any], placeholder_style: str = ":"):
    """Build WHERE clause from filter dict."""
    clause, params, _ = _build_sql_from_filter(filter_dict or {}, placeholder_style)
    return clause, params


async def create(
    name: str, data: Dict[str, Any], settings: Settings, config: Optional[Dict] = None
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
            # Use SQLAlchemy Core to construct the INSERT statement safely
            # This avoids SQL injection warnings from tools like Semgrep
            tbl = table(name, *[column(c) for c in data.keys()])
            stmt = tbl.insert().values(**data)

            if db_type == DatabaseType.SQLITE:
                # SQLite: Insert then fetch by rowid
                await session.execute(stmt)
                await session.commit()

                # Get last inserted row
                rid_result = await session.execute(
                    sa_text("SELECT last_insert_rowid() AS rid")
                )
                rid = rid_result.scalar()

                if rid is not None:
                    # Use SQLAlchemy Core for SELECT to avoid Semgrep warnings
                    # Equivalent to: SELECT * FROM {name} WHERE rowid = :rid
                    tbl_select = table(name, column("rowid"))
                    stmt_select = select(literal_column("*")).select_from(tbl_select).where(column("rowid") == rid)
                    sel_result = await session.execute(stmt_select)

                    row = sel_result.fetchone()
                    return _row_to_dict(row) if row else None
                return None
            else:
                # PostgreSQL: Use RETURNING
                stmt = stmt.returning(literal_column("*"))
                result = await session.execute(stmt)
                await session.commit()
                row = result.fetchone()
                return _row_to_dict(row) if row else None

    elif db_type in (DatabaseType.MYSQL, DatabaseType.MARIADB, DatabaseType.SNOWFLAKE):
        async with async_session_or_connection(settings, config) as conn:
            cols = ", ".join(data.keys())
            placeholders = ", ".join([f"%({k})s" for k in data.keys()])
            sql_stmt = f"INSERT INTO {name} ({cols}) VALUES ({placeholders})"

            if hasattr(conn, "execute"):
                # Async connection (MySQL/Snowflake async wrapper)
                await conn.execute(sql_stmt, data)
                await conn.commit()
                # Note: Getting lastrowid in async context depends on the specific driver
                return {"success": True}
            else:
                # Fallback for sync wrapper
                cursor = conn.cursor()
                cursor.execute(sql, data)
                conn.commit()
                lastrowid = getattr(cursor, "lastrowid", None)
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
) -> Optional[Union[Dict[str, Any], List[Dict[str, Any]]]]:
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
            clause, params = _build_where_clause_sql(
                filter or {}, placeholder_style=":"
            )
            sql = sa_text(f"SELECT * FROM {name}{clause}")
            result = await session.execute(sql, params)
            rows_raw = result.fetchall()
            rows = [cast(Dict[str, Any], _row_to_dict(r)) for r in rows_raw]
            return cast(
                Union[Dict[str, Any], List[Dict[str, Any]]],
                rows[0] if one and rows else rows,
            )

    elif db_type in (DatabaseType.MYSQL, DatabaseType.MARIADB, DatabaseType.SNOWFLAKE):
        async with async_session_or_connection(settings, config) as conn:
            clause, params = _build_where_clause_sql(
                filter or {}, placeholder_style="pyformat"
            )
            sql_stmt = f"SELECT * FROM {name}{clause}"

            if hasattr(conn, "execute"):
                # Async connection
                result = await conn.execute(sql_stmt, params)
                # This would need to be adapted based on the specific async driver
                rows = cast(
                    List[Dict[str, Any]],
                    result.fetchall() if hasattr(result, "fetchall") else [],
                )
            else:
                # Sync wrapper
                cursor = conn.cursor()
                cursor.execute(
                    sql_stmt, params if isinstance(params, dict) else tuple(params)
                )
                cols = [c[0] for c in cursor.description] if cursor.description else []
                rows = [dict(zip(cols, row)) for row in cursor.fetchall()]
                cursor.close()

            return cast(
                Union[Dict[str, Any], List[Dict[str, Any]]],
                rows[0] if one and rows else rows,
            )

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
                    return cast(Optional[Dict[str, Any]], item)
                cursor = collection.find(filter)
                items = await cursor.to_list(length=None)

            return cast(List[Dict[str, Any]], items)

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
            clause, where_params = _build_where_clause_sql(
                filter, placeholder_style=":"
            )
            params.update(where_params)

            sql = sa_text(f"UPDATE {name} SET {set_parts}{clause}")
            result = await session.execute(sql, params)
            await session.commit()
            return {"affected_rows": result.rowcount}

    elif db_type in (DatabaseType.MYSQL, DatabaseType.MARIADB, DatabaseType.SNOWFLAKE):
        async with async_session_or_connection(settings, config) as conn:
            set_parts = ", ".join(f"{k} = %({k})s" for k in update_data.keys())
            clause, where_params = _build_where_clause_sql(
                filter, placeholder_style="pyformat"
            )
            params = dict(update_data)
            if isinstance(where_params, dict):
                params.update(where_params)

            sql_stmt = f"UPDATE {name} SET {set_parts}{clause}"

            if hasattr(conn, "execute"):
                result = await conn.execute(sql_stmt, params)
                await conn.commit()
                return {"affected_rows": getattr(result, "rowcount", 0)}
            else:
                cursor = conn.cursor()
                cursor.execute(sql_stmt, params)
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
                "modified_count": result.modified_count,
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
            clause, params = _build_where_clause_sql(
                filter, placeholder_style="pyformat"
            )
            sql_stmt = f"DELETE FROM {name}{clause}"

            if hasattr(conn, "execute"):
                result = await conn.execute(sql_stmt, params)
                await conn.commit()
                return {"affected_rows": getattr(result, "rowcount", 0)}
            else:
                cursor = conn.cursor()
                cursor.execute(
                    sql_stmt, params if isinstance(params, dict) else tuple(params)
                )
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
    config: Optional[Dict] = None,
) -> int:
    """Count records matching filter."""
    _check_identifier(name)
    db_type = get_db_type(settings)

    if db_type in (DatabaseType.POSTGRES, DatabaseType.POSTGRESQL, DatabaseType.SQLITE):
        async with async_session_or_connection(settings, config) as session:
            clause, params = _build_where_clause_sql(
                filter or {}, placeholder_style=":"
            )
            sql = sa_text(f"SELECT COUNT(*) FROM {name}{clause}")
            result = await session.execute(sql, params)
            return result.scalar() or 0

    elif db_type in (DatabaseType.MYSQL, DatabaseType.MARIADB, DatabaseType.SNOWFLAKE):
        async with async_session_or_connection(settings, config) as conn:
            clause, params = _build_where_clause_sql(
                filter or {}, placeholder_style="pyformat"
            )
            sql_stmt = f"SELECT COUNT(*) FROM {name}{clause}"

            if hasattr(conn, "execute"):
                result = await conn.execute(sql_stmt, params)
                return result.fetchone()[0] if hasattr(result, "fetchone") else 0
            else:
                cursor = conn.cursor()
                cursor.execute(
                    sql_stmt, params if isinstance(params, dict) else tuple(params)
                )
                result = cursor.fetchone()
                cursor.close()
                return result[0] if result else 0

    elif db_type in (DatabaseType.MONGODB, DatabaseType.MONGO):
        async with async_session_or_connection(settings, config) as db_obj:
            collection = db_obj[name]
            return cast(int, await collection.count_documents(filter or {}))

    else:
        raise RuntimeError(f"Unsupported database type: {db_type}")
