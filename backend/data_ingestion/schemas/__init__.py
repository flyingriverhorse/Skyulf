"""Pydantic request and response models for the ingestion API.

A single module, ``ingestion``, holding the request bodies, the job
acknowledgement and status-polling payloads, and the client-safe
``DataSourceRead`` view that redacts credentials out of ``config``. Nothing is
re-exported here.
"""
