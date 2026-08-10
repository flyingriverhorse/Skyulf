# Ray Migration

This folder contains the approved architecture and implementation planning
documents for replacing Celery as Skyulf's execution backend with Ray.

## Documents

- [Architecture design](2026-08-10-ray-migration-design.md)
- Implementation plan: added after the architecture document is reviewed

## Decision Summary

The target architecture keeps FastAPI as the control plane and PostgreSQL as
the user-visible job state store. Ray Jobs, Ray Core, and initially Ray's
joblib integration become the compute plane. S3-compatible storage provides
shared datasets and artifacts across workers.

The migration is incremental. Celery and Ray coexist behind an execution
backend interface until status handling, cancellation, retries, artifact
promotion, and result parity are proven in production-like tests.
