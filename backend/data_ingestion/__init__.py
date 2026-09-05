"""Data ingestion: registering sources, sampling them, and profiling the result.

Nothing is re-exported here — submodules are imported by path, so importing the
package does not pull in Celery, FastAPI, polars or SQLAlchemy.

Layout:
    connectors/  Source readers behind the ``BaseConnector`` contract: local
        files and S3-compatible object storage.
    engine/      ``DataProfiler``, the per-column statistics pass. This
        directory has no ``__init__.py`` and resolves as a namespace package.
    schemas/     Pydantic request/response models, including the redaction that
        keeps credentials out of ``DataSourceRead.config``.
    router.py    HTTP endpoints for both the ``/data/api`` and ``/api/ingestion``
        prefixes.
    service.py   Application layer between those routes and the database.
    tasks.py     Celery worker entry point that runs the ingest and profile pass.
    dependencies.py  FastAPI dependency providers.
    serialization.py JSON-safe conversion helpers.
"""
