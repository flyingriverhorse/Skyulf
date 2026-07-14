from abc import ABC, abstractmethod

import pandas as pd

from ..engines import SkyulfDataFrame


class DataCatalog(ABC):
    """
    Abstract interface for data access.
    Decouples the pipeline from the storage mechanism.
    """

    @abstractmethod
    def load(self, dataset_id: str, **kwargs) -> pd.DataFrame | SkyulfDataFrame:
        """
        Load a dataset by its identifier.

        Args:
            dataset_id: Unique identifier for the dataset (e.g., filename, table name).
            **kwargs: Additional arguments (e.g., version, sample_size).
        """

    @abstractmethod
    def save(self, dataset_id: str, data: pd.DataFrame | SkyulfDataFrame, **kwargs) -> None:
        """
        Save a dataset.

        Args:
            dataset_id: Unique identifier for the destination.
            data: The pandas/polars DataFrame to save.
        """

    @abstractmethod
    def exists(self, dataset_id: str) -> bool:
        """Check if a dataset exists."""

    def get_dataset_name(self, dataset_id: str) -> str | None:
        """Returns the human-readable name of the dataset, if available."""
        return None
