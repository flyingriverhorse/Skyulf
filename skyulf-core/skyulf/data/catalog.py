from abc import ABC, abstractmethod
from typing import Any, Dict, Optional
import pandas as pd

class DataCatalog(ABC):
    """
    Abstract interface for data access.
    Decouples the pipeline from the storage mechanism.
    """

    @abstractmethod
    def load(self, dataset_id: str, **kwargs) -> pd.DataFrame:
        """
        Load a dataset by its identifier.
        
        Args:
            dataset_id: Unique identifier for the dataset (e.g., filename, table name).
            **kwargs: Additional arguments (e.g., version, sample_size).
        """
        pass

    @abstractmethod
    def save(self, dataset_id: str, data: pd.DataFrame, **kwargs) -> None:
        """
        Save a dataset.
        
        Args:
            dataset_id: Unique identifier for the destination.
            data: The pandas DataFrame to save.
        """
        pass

    @abstractmethod
    def exists(self, dataset_id: str) -> bool:
        """Check if a dataset exists."""
        pass
