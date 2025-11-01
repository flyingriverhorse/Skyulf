"""
FastAPI-compatible serialization utilities.

JSON-safe data conversion utilities with enhanced async support.
Migrated from Flask sync version with improved type handling.
"""

import asyncio
import logging
from typing import Any, Dict, List, Union, Optional
import pandas as pd
import numpy as np
from datetime import datetime, date
from decimal import Decimal
import json

logger = logging.getLogger(__name__)


class AsyncJSONSafeSerializer:
    """Async-compatible JSON-safe serialization utilities."""

    @staticmethod
    async def clean_for_json(obj: Any) -> Any:
        """
        Convert data structures to JSON-safe format asynchronously.
        
        Args:
            obj: Object to make JSON-safe
            
        Returns:
            JSON-safe version of the object
        """
        if obj is None:
            return None
            
        # Handle pandas DataFrame
        if isinstance(obj, pd.DataFrame):
            # For large DataFrames, yield control to event loop
            if len(obj) > 1000:
                await asyncio.sleep(0)
            return obj.where(pd.notnull(obj), None).to_dict("records")
            
        # Handle pandas Series
        if isinstance(obj, pd.Series):
            if len(obj) > 1000:
                await asyncio.sleep(0)
            return obj.where(pd.notnull(obj), None).tolist()
            
        # Handle dictionaries
        if isinstance(obj, dict):
            result = {}
            for k, v in obj.items():
                result[str(k)] = await AsyncJSONSafeSerializer.clean_for_json(v)
            return result
            
        # Handle lists and tuples
        if isinstance(obj, (list, tuple)):
            if len(obj) > 100:
                await asyncio.sleep(0)
            result = []
            for item in obj:
                result.append(await AsyncJSONSafeSerializer.clean_for_json(item))
            return result
            
        # Handle numpy types
        if isinstance(obj, (np.integer, np.int8, np.int16, np.int32, np.int64)):
            return int(obj)
        if isinstance(obj, (np.floating, np.float16, np.float32, np.float64)):
            return None if pd.isna(obj) else float(obj)
        if isinstance(obj, np.bool_):
            return bool(obj)
        if isinstance(obj, np.ndarray):
            return obj.tolist()
            
        # Handle datetime objects
        if isinstance(obj, (datetime, date)):
            return obj.isoformat()
            
        # Handle Decimal
        if isinstance(obj, Decimal):
            return float(obj)
            
        # Handle generic NaN/infinity
        if isinstance(obj, float):
            if pd.isna(obj) or np.isinf(obj):
                return None
                
        # Handle bytes
        if isinstance(obj, bytes):
            try:
                return obj.decode('utf-8')
            except UnicodeDecodeError:
                return str(obj)
                
        # Handle sets
        if isinstance(obj, set):
            return list(obj)
            
        # Return as-is for basic types
        if isinstance(obj, (str, int, float, bool)):
            return obj
            
        # For other types, try to convert to string
        try:
            return str(obj)
        except Exception:
            logger.warning(f"Could not serialize object of type {type(obj)}")
            return None

    @staticmethod
    async def safe_dict_from_dataframe(
        df: pd.DataFrame, 
        records_format: bool = True,
        max_rows: Optional[int] = None
    ) -> Union[List[Dict], Dict]:
        """
        Convert DataFrame to JSON-safe dictionary format.
        
        Args:
            df: DataFrame to convert
            records_format: If True, return list of records; if False, return dict of columns
            max_rows: Maximum number of rows to include
            
        Returns:
            JSON-safe dictionary representation
        """
        if df.empty:
            return [] if records_format else {}
            
        # Limit rows if specified
        if max_rows and len(df) > max_rows:
            df = df.head(max_rows)
            logger.info(f"Truncated DataFrame to {max_rows} rows for serialization")
            
        # Yield control for large DataFrames
        if len(df) > 1000:
            await asyncio.sleep(0)
            
        # Clean the DataFrame
        clean_df = df.fillna(None)
        
        if records_format:
            # Convert to list of records
            records = clean_df.to_dict("records")
            # Clean each record
            result = []
            for record in records:
                clean_record = await AsyncJSONSafeSerializer.clean_for_json(record)
                result.append(clean_record)
            return result
        else:
            # Convert to dict of columns
            result = {}
            for col in clean_df.columns:
                column_data = clean_df[col].tolist()
                result[str(col)] = await AsyncJSONSafeSerializer.clean_for_json(column_data)
            return result

    @staticmethod
    async def serialize_dataframe_metadata(df: pd.DataFrame) -> Dict[str, Any]:
        """
        Extract and serialize DataFrame metadata.
        
        Args:
            df: DataFrame to analyze
            
        Returns:
            Dictionary containing DataFrame metadata
        """
        if df.empty:
            return {
                "shape": [0, 0],
                "columns": [],
                "dtypes": {},
                "memory_usage": 0,
                "has_nulls": False
            }
            
        # Basic info
        metadata = {
            "shape": list(df.shape),
            "columns": [str(col) for col in df.columns],
            "memory_usage": int(df.memory_usage(deep=True).sum()),
            "has_nulls": df.isnull().any().any(),
            "null_counts": df.isnull().sum().to_dict()
        }
        
        # Data types
        dtypes = {}
        for col, dtype in df.dtypes.items():
            dtypes[str(col)] = str(dtype)
        metadata["dtypes"] = dtypes
        
        # Basic statistics for numeric columns
        numeric_columns = df.select_dtypes(include=[np.number]).columns
        if len(numeric_columns) > 0:
            stats = df[numeric_columns].describe()
            metadata["numeric_stats"] = await AsyncJSONSafeSerializer.clean_for_json(
                stats.to_dict()
            )
            
        # Sample data
        if len(df) > 0:
            sample_size = min(5, len(df))
            sample_data = await AsyncJSONSafeSerializer.safe_dict_from_dataframe(
                df.head(sample_size)
            )
            metadata["sample_data"] = sample_data
            
        return metadata

    @staticmethod
    async def serialize_query_result(
        df: pd.DataFrame,
        include_metadata: bool = True,
        max_rows: Optional[int] = None
    ) -> Dict[str, Any]:
        """
        Serialize a query result DataFrame with optional metadata.
        
        Args:
            df: Query result DataFrame
            include_metadata: Whether to include metadata
            max_rows: Maximum number of rows to serialize
            
        Returns:
            Serialized query result
        """
        result = {
            "success": True,
            "row_count": len(df),
            "column_count": len(df.columns)
        }
        
        # Serialize the data
        result["data"] = await AsyncJSONSafeSerializer.safe_dict_from_dataframe(
            df, records_format=True, max_rows=max_rows
        )
        
        # Add metadata if requested
        if include_metadata:
            result["metadata"] = await AsyncJSONSafeSerializer.serialize_dataframe_metadata(df)
            
        # Add truncation info
        if max_rows and len(df) > max_rows:
            result["truncated"] = True
            result["total_rows"] = len(df)
            result["returned_rows"] = max_rows
            
        return result


class JSONSafeSerializer:
    """Synchronous JSON-safe serializer for backward compatibility."""

    @staticmethod
    def clean_for_json(obj: Any) -> Any:
        """Synchronous version of clean_for_json that works in async context."""
        if obj is None:
            return None
        
        # Handle pandas types
        if hasattr(obj, 'dtype'):
            # pandas Series or numpy array
            if hasattr(obj, 'isna') and obj.isna():
                return None
            if hasattr(obj, 'tolist'):
                return JSONSafeSerializer.clean_for_json(obj.tolist())
            return JSONSafeSerializer.clean_for_json(obj.item() if hasattr(obj, 'item') else obj)
        
        # Handle numpy types
        if hasattr(obj, 'dtype') or str(type(obj)).startswith("<class 'numpy."):
            if hasattr(obj, 'tolist'):
                return obj.tolist()
            elif hasattr(obj, 'item'):
                return obj.item()
            else:
                return str(obj)
        
        # Handle pandas NaN, nat, etc.
        if str(obj) in ['nan', 'NaN', 'NaT', '<NA>', 'inf', '-inf', 'infinity', '-infinity']:
            return None
            
        # Handle float types with proper NaN and infinity checks
        if isinstance(obj, float):
            if obj != obj:  # NaN check
                return None
            if obj == float('inf') or obj == float('-inf'):
                return None
            # Additional check with numpy if available
            try:
                if np.isinf(obj) or np.isnan(obj):
                    return None
            except:
                pass
            return obj
            
        # Handle other numeric and basic types
        if isinstance(obj, (str, int, bool)):
            return obj
            
        if isinstance(obj, dict):
            result = {}
            for k, v in obj.items():
                clean_key = str(k)
                result[clean_key] = JSONSafeSerializer.clean_for_json(v)
            return result
            
        if isinstance(obj, (list, tuple, set)):
            result = []
            for item in obj:
                result.append(JSONSafeSerializer.clean_for_json(item))
            return result
        
        # Handle pandas DataFrame
        if hasattr(obj, 'to_dict'):
            try:
                return JSONSafeSerializer.clean_for_json(obj.to_dict('records'))
            except:
                pass
        
        # Handle datetime objects
        if hasattr(obj, 'isoformat'):
            return obj.isoformat()
            
        # Handle decimal and other numeric types
        if hasattr(obj, '__float__'):
            try:
                val = float(obj)
                return None if val != val else val  # NaN check
            except:
                pass
        
        # Fallback to string representation
        try:
            return str(obj)
        except:
            return None

    @staticmethod
    def safe_dict_from_dataframe(
        df, 
        records_format: bool = True
    ) -> Union[List[Dict], Dict]:
        """Synchronous version of safe_dict_from_dataframe."""
        if df is None or df.empty:
            return [] if records_format else {}
            
        if records_format:
            # Convert to records format and clean each record
            records = df.to_dict('records')
            cleaned_records = []
            for record in records:
                cleaned_record = {}
                for key, value in record.items():
                    cleaned_record[str(key)] = JSONSafeSerializer.clean_for_json(value)
                cleaned_records.append(cleaned_record)
            return cleaned_records
        else:
            # Convert to dict format and clean
            result = {}
            for col in df.columns:
                column_data = df[col].tolist()
                result[str(col)] = JSONSafeSerializer.clean_for_json(column_data)
            return result


class DataTypeConverter:
    """Utility class for data type conversions and validations."""

    @staticmethod
    def infer_column_types(df: pd.DataFrame) -> Dict[str, str]:
        """
        Infer semantic data types for DataFrame columns.
        
        Args:
            df: DataFrame to analyze
            
        Returns:
            Dictionary mapping column names to inferred types
        """
        type_mapping = {}
        
        for col in df.columns:
            series = df[col]
            col_str = str(col)
            
            if series.dtype == 'object':
                # Check for dates
                if series.dropna().apply(lambda x: isinstance(x, (datetime, date))).all():
                    type_mapping[col_str] = 'datetime'
                # Check for numeric strings
                elif series.dropna().str.match(r'^-?\d+\.?\d*$').all():
                    type_mapping[col_str] = 'numeric_string'
                # Check for boolean-like strings
                elif series.dropna().str.lower().isin(['true', 'false', 'yes', 'no', '1', '0']).all():
                    type_mapping[col_str] = 'boolean_string'
                else:
                    type_mapping[col_str] = 'text'
            elif pd.api.types.is_numeric_dtype(series):
                if pd.api.types.is_integer_dtype(series):
                    type_mapping[col_str] = 'integer'
                else:
                    type_mapping[col_str] = 'float'
            elif pd.api.types.is_datetime64_any_dtype(series):
                type_mapping[col_str] = 'datetime'
            elif pd.api.types.is_bool_dtype(series):
                type_mapping[col_str] = 'boolean'
            else:
                type_mapping[col_str] = str(series.dtype)
                
        return type_mapping

    @staticmethod
    async def convert_dataframe_types(
        df: pd.DataFrame, 
        type_mapping: Dict[str, str]
    ) -> pd.DataFrame:
        """
        Convert DataFrame columns to specified types.
        
        Args:
            df: DataFrame to convert
            type_mapping: Dictionary mapping column names to target types
            
        Returns:
            DataFrame with converted types
        """
        result_df = df.copy()
        
        for col, target_type in type_mapping.items():
            if col not in result_df.columns:
                continue
                
            try:
                if target_type == 'integer':
                    result_df[col] = pd.to_numeric(result_df[col], errors='coerce').astype('Int64')
                elif target_type == 'float':
                    result_df[col] = pd.to_numeric(result_df[col], errors='coerce')
                elif target_type == 'datetime':
                    result_df[col] = pd.to_datetime(result_df[col], errors='coerce')
                elif target_type == 'boolean':
                    result_df[col] = result_df[col].astype('boolean')
                elif target_type == 'text':
                    result_df[col] = result_df[col].astype('string')
                    
            except Exception as e:
                logger.warning(f"Failed to convert column {col} to {target_type}: {e}")
                
        return result_df

    @staticmethod
    def get_data_quality_report(df: pd.DataFrame) -> Dict[str, Any]:
        """
        Generate a data quality report for a DataFrame.
        
        Args:
            df: DataFrame to analyze
            
        Returns:
            Data quality report
        """
        if df.empty:
            return {"error": "DataFrame is empty"}
            
        report = {
            "overview": {
                "total_rows": len(df),
                "total_columns": len(df.columns),
                "memory_usage_mb": df.memory_usage(deep=True).sum() / 1024 / 1024
            },
            "columns": {},
            "data_types": DataTypeConverter.infer_column_types(df),
            "quality_score": 0.0
        }
        
        total_quality = 0
        
        for col in df.columns:
            col_str = str(col)
            series = df[col]
            
            col_report = {
                "dtype": str(series.dtype),
                "null_count": int(series.isnull().sum()),
                "null_percentage": float(series.isnull().sum() / len(series) * 100),
                "unique_count": int(series.nunique()),
                "unique_percentage": float(series.nunique() / len(series) * 100)
            }
            
            # Calculate quality score (1.0 = perfect, 0.0 = all nulls)
            quality = 1.0 - (col_report["null_percentage"] / 100)
            col_report["quality_score"] = quality
            total_quality += quality
            
            # Add type-specific stats
            if pd.api.types.is_numeric_dtype(series):
                col_report["min"] = float(series.min()) if not series.empty else None
                col_report["max"] = float(series.max()) if not series.empty else None
                col_report["mean"] = float(series.mean()) if not series.empty else None
                
            report["columns"][col_str] = col_report
            
        # Overall quality score
        report["quality_score"] = total_quality / len(df.columns) if len(df.columns) > 0 else 0
        
        return report


# Utility functions for common serialization tasks
async def serialize_api_response(
    data: Any, 
    success: bool = True, 
    message: Optional[str] = None,
    metadata: Optional[Dict] = None
) -> Dict[str, Any]:
    """
    Serialize data into a standard API response format.
    
    Args:
        data: Data to serialize
        success: Whether the operation was successful
        message: Optional message
        metadata: Optional metadata
        
    Returns:
        Standardized API response
    """
    response = {
        "success": success,
        "timestamp": datetime.now().isoformat(),
    }
    
    if message:
        response["message"] = message
        
    if metadata:
        response["metadata"] = metadata
        
    # Serialize the data
    if data is not None:
        response["data"] = await AsyncJSONSafeSerializer.clean_for_json(data)
    else:
        response["data"] = None
        
    return response


def serialize_api_response_sync(
    data: Any, 
    success: bool = True, 
    message: Optional[str] = None,
    metadata: Optional[Dict] = None
) -> Dict[str, Any]:
    """Synchronous version of serialize_api_response."""
    return asyncio.run(serialize_api_response(data, success, message, metadata))