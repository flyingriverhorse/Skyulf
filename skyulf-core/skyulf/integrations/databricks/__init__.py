"""Optional Databricks batch and Delta integrations."""

from .batch import BatchResult, BatchSpec, run_batch

__all__ = ["BatchResult", "BatchSpec", "run_batch"]
