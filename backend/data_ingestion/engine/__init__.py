"""Ingestion-time computation over the freshly-read frame.

A single module, ``profiler``, whose ``DataProfiler.profile`` produces the
dataset and per-column statistics stored on ``DataSource.source_metadata``.
Nothing is re-exported here — import ``DataProfiler`` by path.
"""
