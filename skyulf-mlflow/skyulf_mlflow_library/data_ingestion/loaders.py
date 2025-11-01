"""
Data Ingestion module for Skyulf MLflow.

This module provides utilities for loading data from various sources
including CSV, Excel, JSON, Parquet, SQL databases, and APIs.
"""

from pathlib import Path
from typing import Any, Dict, List, Optional, Union

import pandas as pd


class DataLoader:
    """
    Universal data loader for various file formats and sources.
    
    Supports loading from:
    - CSV files
    - Excel files (xlsx, xls)
    - JSON files
    - Parquet files
    - SQL databases
    - Python dictionaries and lists
    
    Parameters
    ----------
    source : str or Path
        Path to the data file or database connection string.
    file_type : str, optional
        Type of file ('csv', 'excel', 'json', 'parquet', 'sql').
        If None, infers from file extension.
    **kwargs : dict
        Additional keyword arguments passed to pandas read functions.
    
    Examples
    --------
    >>> from skyulf_mlflow_library.data_ingestion import DataLoader
    >>> 
    >>> # Load CSV file
    >>> loader = DataLoader('data.csv')
    >>> df = loader.load()
    >>> 
    >>> # Load Excel with specific sheet
    >>> loader = DataLoader('data.xlsx', sheet_name='Sheet1')
    >>> df = loader.load()
    >>> 
    >>> # Load with custom separator
    >>> loader = DataLoader('data.txt', file_type='csv', sep='|')
    >>> df = loader.load()
    """
    
    SUPPORTED_FORMATS = {
        'csv': ['.csv', '.txt'],
        'excel': ['.xlsx', '.xls', '.xlsm'],
        'json': ['.json'],
        'parquet': ['.parquet', '.pq'],
    }
    
    def __init__(
        self,
        source: Union[str, Path],
        file_type: Optional[str] = None,
        **kwargs
    ):
        """
        Initialize the data loader.
        
        Parameters
        ----------
        source : str or Path
            Path to the data file or connection string.
        file_type : str, optional
            Type of file to load.
        **kwargs : dict
            Additional arguments for pandas readers.
        """
        self.source = Path(source) if isinstance(source, str) else source
        self.file_type = file_type or self._infer_file_type()
        self.kwargs = kwargs
    
    def _infer_file_type(self) -> str:
        """Infer file type from extension."""
        if not isinstance(self.source, Path):
            raise ValueError("Cannot infer file type from non-path source")
        
        suffix = self.source.suffix.lower()
        
        for file_type, extensions in self.SUPPORTED_FORMATS.items():
            if suffix in extensions:
                return file_type
        
        raise ValueError(
            f"Cannot infer file type from extension '{suffix}'. "
            f"Please specify file_type parameter."
        )
    
    def load(self) -> pd.DataFrame:
        """
        Load data into a pandas DataFrame.
        
        Returns
        -------
        pd.DataFrame
            Loaded data.
        
        Raises
        ------
        FileNotFoundError
            If source file doesn't exist.
        ValueError
            If file type is not supported.
        """
        if isinstance(self.source, Path) and not self.source.exists():
            raise FileNotFoundError(f"File not found: {self.source}")
        
        if self.file_type == 'csv':
            return pd.read_csv(self.source, **self.kwargs)
        
        elif self.file_type == 'excel':
            return pd.read_excel(self.source, **self.kwargs)
        
        elif self.file_type == 'json':
            return pd.read_json(self.source, **self.kwargs)
        
        elif self.file_type == 'parquet':
            return pd.read_parquet(self.source, **self.kwargs)
        
        elif self.file_type == 'sql':
            # Requires connection string in kwargs
            if 'con' not in self.kwargs:
                raise ValueError("SQL loading requires 'con' parameter")
            query = str(self.source)
            return pd.read_sql(query, **self.kwargs)
        
        else:
            raise ValueError(f"Unsupported file type: {self.file_type}")
    
    @staticmethod
    def from_dict(data: Dict[str, List], **kwargs) -> pd.DataFrame:
        """
        Create DataFrame from dictionary.
        
        Parameters
        ----------
        data : dict
            Dictionary with column names as keys and lists as values.
        **kwargs : dict
            Additional arguments for DataFrame constructor.
        
        Returns
        -------
        pd.DataFrame
            Created DataFrame.
        """
        return pd.DataFrame(data, **kwargs)
    
    @staticmethod
    def from_records(records: List[Dict], **kwargs) -> pd.DataFrame:
        """
        Create DataFrame from list of dictionaries.
        
        Parameters
        ----------
        records : list of dict
            List of dictionaries representing rows.
        **kwargs : dict
            Additional arguments for DataFrame constructor.
        
        Returns
        -------
        pd.DataFrame
            Created DataFrame.
        """
        return pd.DataFrame.from_records(records, **kwargs)


class DataSaver:
    """
    Universal data saver for various file formats.
    
    Supports saving to:
    - CSV files
    - Excel files
    - JSON files
    - Parquet files
    
    Examples
    --------
    >>> from skyulf_mlflow_library.data_ingestion import DataSaver
    >>> import pandas as pd
    >>> 
    >>> df = pd.DataFrame({'a': [1, 2, 3], 'b': [4, 5, 6]})
    >>> 
    >>> # Save to CSV
    >>> saver = DataSaver('output.csv')
    >>> saver.save(df)
    >>> 
    >>> # Save to Parquet with compression
    >>> saver = DataSaver('output.parquet', compression='gzip')
    >>> saver.save(df)
    """
    
    def __init__(
        self,
        destination: Union[str, Path],
        file_type: Optional[str] = None,
        **kwargs
    ):
        """
        Initialize the data saver.
        
        Parameters
        ----------
        destination : str or Path
            Path where to save the data.
        file_type : str, optional
            Type of file to save. If None, infers from extension.
        **kwargs : dict
            Additional arguments for pandas writers.
        """
        self.destination = Path(destination) if isinstance(destination, str) else destination
        self.file_type = file_type or self._infer_file_type()
        self.kwargs = kwargs
    
    def _infer_file_type(self) -> str:
        """Infer file type from extension."""
        suffix = self.destination.suffix.lower()
        
        for file_type, extensions in DataLoader.SUPPORTED_FORMATS.items():
            if suffix in extensions:
                return file_type
        
        raise ValueError(
            f"Cannot infer file type from extension '{suffix}'. "
            f"Please specify file_type parameter."
        )
    
    def save(self, data: pd.DataFrame):
        """
        Save DataFrame to file.
        
        Parameters
        ----------
        data : pd.DataFrame
            Data to save.
        
        Raises
        ------
        ValueError
            If file type is not supported.
        """
        # Create parent directory if it doesn't exist
        self.destination.parent.mkdir(parents=True, exist_ok=True)
        
        if self.file_type == 'csv':
            # Default to index=False if not specified
            csv_kwargs = {'index': False, **self.kwargs}
            data.to_csv(self.destination, **csv_kwargs)
        
        elif self.file_type == 'excel':
            # Default to index=False if not specified
            excel_kwargs = {'index': False, **self.kwargs}
            data.to_excel(self.destination, **excel_kwargs)
        
        elif self.file_type == 'json':
            data.to_json(self.destination, **self.kwargs)
        
        elif self.file_type == 'parquet':
            data.to_parquet(self.destination, **self.kwargs)
        
        else:
            raise ValueError(f"Unsupported file type: {self.file_type}")


def load_data(
    source: Union[str, Path],
    file_type: Optional[str] = None,
    **kwargs
) -> pd.DataFrame:
    """
    Convenience function to load data.
    
    Parameters
    ----------
    source : str or Path
        Path to the data file.
    file_type : str, optional
        Type of file to load.
    **kwargs : dict
        Additional arguments for pandas readers.
    
    Returns
    -------
    pd.DataFrame
        Loaded data.
    
    Examples
    --------
    >>> from skyulf_mlflow_library.data_ingestion import load_data
    >>> 
    >>> df = load_data('data.csv')
    >>> df = load_data('data.xlsx', sheet_name='Sheet1')
    """
    loader = DataLoader(source, file_type, **kwargs)
    return loader.load()


def save_data(
    data: pd.DataFrame,
    destination: Union[str, Path],
    file_type: Optional[str] = None,
    **kwargs
):
    """
    Convenience function to save data.
    
    Parameters
    ----------
    data : pd.DataFrame
        Data to save.
    destination : str or Path
        Path where to save the data.
    file_type : str, optional
        Type of file to save.
    **kwargs : dict
        Additional arguments for pandas writers.
    
    Examples
    --------
    >>> from skyulf_mlflow_library.data_ingestion import save_data
    >>> import pandas as pd
    >>> 
    >>> df = pd.DataFrame({'a': [1, 2, 3]})
    >>> save_data(df, 'output.csv', index=False)
    """
    saver = DataSaver(destination, file_type, **kwargs)
    saver.save(data)
