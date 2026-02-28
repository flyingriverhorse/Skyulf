import io
from typing import Any, Dict, Optional

import httpx
import polars as pl

from .base import BaseConnector


class ApiConnector(BaseConnector):
    """
    Connector for REST APIs.
    Fetches data from a URL and parses it into a Polars DataFrame.
    """

    def __init__(
        self,
        url: str,
        method: str = "GET",
        headers: Optional[Dict[str, str]] = None,
        params: Optional[Dict[str, Any]] = None,
        data_key: Optional[str] = None,
    ):
        self.url = url
        self.method = method
        self.headers = headers or {}
        self.params = params or {}
        self.data_key = (
            data_key  # Key to extract data from JSON response (e.g. "items" or "data")
        )

    async def connect(self) -> bool:
        try:
            async with httpx.AsyncClient() as client:
                response = await client.request(
                    self.method, self.url, headers=self.headers, params=self.params
                )
                response.raise_for_status()
            return True
        except Exception as e:
            raise ConnectionError(f"Failed to connect to API: {str(e)}")

    async def validate(self) -> bool:
        return await self.connect()

    async def get_schema(self) -> Dict[str, str]:
        # For APIs, we often need to fetch at least one record to determine schema
        # We'll fetch a small sample if possible, or just fetch data and infer
        df = await self.fetch_data(limit=1)
        return {col: str(dtype) for col, dtype in df.schema.items()}

    async def fetch_data(
        self, query: Optional[str] = None, limit: Optional[int] = None
    ) -> pl.DataFrame:
        try:
            async with httpx.AsyncClient() as client:
                # Merge query params if provided
                params = self.params.copy()
                if query:
                    # Assuming query might be a filter param, but APIs vary wildly.
                    # For now, we ignore 'query' unless it's structured.
                    pass

                if limit:
                    params["limit"] = limit

                response = await client.request(
                    self.method, self.url, headers=self.headers, params=params
                )
                response.raise_for_status()

                content_type = response.headers.get("content-type", "")

                if "application/json" in content_type:
                    data = response.json()
                    if self.data_key:
                        if self.data_key in data:
                            data = data[self.data_key]
                        else:
                            raise ValueError(
                                f"Key '{self.data_key}' not found in response"
                            )

                    # Polars can read list of dicts
                    return pl.DataFrame(data)

                elif "text/csv" in content_type or "application/csv" in content_type:
                    return pl.read_csv(io.BytesIO(response.content))

                else:
                    # Try to infer
                    try:
                        return pl.DataFrame(response.json())
                    except Exception:
                        pass
                    try:
                        return pl.read_csv(io.BytesIO(response.content))
                    except Exception:
                        raise ValueError(f"Unsupported content type: {content_type}")

        except Exception as e:
            raise RuntimeError(f"Failed to fetch data from API: {str(e)}")
