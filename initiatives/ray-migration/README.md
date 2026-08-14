# Ray Migration

This folder contains the approved architecture and implementation planning
documents for replacing Celery as Skyulf's execution backend with Ray.

## Documents

- [Architecture design](2026-08-10-ray-migration-design.md)
- [Implementation roadmap](2026-08-10-00-implementation-roadmap.md)
- [Execution backend foundation](2026-08-10-01-execution-backend-foundation-plan.md)
- [Job attempts and lifecycle](2026-08-10-02-job-attempt-lifecycle-plan.md)
- [Ray Jobs pipeline runtime](2026-08-10-03-ray-jobs-pipeline-runtime-plan.md)
- [Distributed branches and tuning](2026-08-10-04-distributed-compute-plan.md)
- [Operations and deployment](2026-08-10-05-operations-deployment-plan.md)
- [Cutover and Celery removal](2026-08-10-06-cutover-celery-removal-plan.md)

## Decision Summary

The target architecture keeps FastAPI as the control plane and PostgreSQL as
the user-visible job state store. Ray Jobs, Ray Core, and initially Ray's
joblib integration become the compute plane. S3-compatible storage provides
shared datasets and artifacts across workers.

The migration is incremental. Celery and Ray coexist behind an execution
backend interface until status handling, cancellation, retries, artifact
promotion, and result parity are proven in production-like tests.
