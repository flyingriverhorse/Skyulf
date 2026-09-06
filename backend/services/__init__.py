"""Backend service layer — currently the data I/O boundary.

``DataService`` is the one place the backend reads and writes tabular data. It
prefers polars for speed and falls back to pandas where polars cannot cope, which
makes this module the polars/pandas boundary the rest of the app depends on: data
ingestion and the EDA routes and tasks all load through it rather than calling
polars or pandas readers directly, so a frame handed to a ``skyulf`` node has a
known engine.
"""

from .data_service import DataService

__all__ = ["DataService"]
