from abc import ABC, abstractmethod

import polars as pl


class BaseConnector(ABC):
    """
    Abstract base class for all data connectors.
    Defines the standard interface for connecting, discovering schema, and fetching data.
    """

    @abstractmethod
    async def connect(self) -> bool:
        """
        Establish connection to the data source.
        Returns True if successful, raises Exception otherwise.
        """

    @abstractmethod
    async def get_schema(self) -> dict[str, str]:
        """
        Discover the schema of the data source.
        Returns a dictionary mapping column names to data types.
        """

    @abstractmethod
    async def fetch_data(
        self, query: str | None = None, limit: int | None = None
    ) -> pl.DataFrame:
        """
        Fetch data from the source.

        Args:
            query: Optional query string (SQL, filter, etc.)
            limit: Optional limit on number of rows

        Returns:
            pl.DataFrame: The fetched data
        """

    @abstractmethod
    async def validate(self) -> bool:
        """
        Validate the configuration and connection.
        """
