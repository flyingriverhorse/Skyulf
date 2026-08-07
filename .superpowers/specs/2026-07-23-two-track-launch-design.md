# Two-Track Launch Design

## Purpose

Skyulf will first build credibility and gather feedback through a public
`skyulf-core` release, including Kaggle notebooks and other reproducible
examples. In parallel, it will prepare the web platform for an invite-only
hosted beta that may receive sensitive or proprietary customer data.

The hosted platform must not be publicly exposed until its authentication,
tenant isolation, production operations, and data-protection foundations are
complete.

## Scope

This design replaces the feature-first public-launch ordering in
`temp/processing/roadmap.md`.

It separates work into two related tracks:

1. A public open-source library release.
2. A security-first hosted private beta.

Explainability and drift-monitoring features remain valuable, but they are not
public-hosting prerequisites. Deep-learning support is deferred until user
demand justifies its infrastructure and maintenance cost.

## Track A: Public `skyulf-core` Release

### Goal

Establish portfolio credibility, attract early users and contributors, collect
expert feedback, and validate the workflows that can later support the hosted
product.

### Release requirements

- Publish a versioned, installable `skyulf-core` package.
- Provide a short, reliable quickstart that succeeds in a clean environment.
- Maintain three reproducible examples:
  - A tabular preprocessing and training pipeline.
  - EDA and drift analysis.
  - Notebook export and execution.
- Publish two or three polished Kaggle notebooks using stable public datasets.
  Each notebook links to the project repository and records package versions.
- Publish clear release notes, a support/compatibility policy, issue templates,
  and a contributor feedback route.
- State clearly that the web platform is not a public hosted service yet.

### Success measures

- Clean installation and example execution are reproducible from documented
  commands.
- Users complete the quickstart and examples without undocumented setup.
- The project receives useful issues, discussions, or contributor feedback.
- Package adoption, repository engagement, and practitioner feedback inform
  future hosted-product priorities.

## Track B: Security-First Hosted Private Beta

### Audience and access

The initial hosted offering is invite-only and limited to a small, named beta
cohort. It does not include self-service sign-up, billing, or unrestricted
programmatic access.

### Production foundation

Before accepting customer data, the platform must provide:

- Separate development and production container configuration. Production
  images must not use reload mode or bind-mount source code.
- PostgreSQL for production metadata and durable object storage for datasets
  and artifacts.
- HTTPS termination, managed secrets, explicitly restricted CORS, upload and
  request limits, and a deliberate policy for API documentation exposure.
- Tested backup and restore procedures for PostgreSQL and artifact storage.
- Structured logs, error reporting, request/job metrics, and alerts for job
  failures, error rate, storage pressure, and authentication anomalies.
- A written data-retention and deletion policy covering datasets, artifacts,
  predictions, logs, and backups.

### Identity and tenant isolation

Authentication and authorization precede customer-facing feature expansion.

- Every request resolves an authenticated principal.
- Every tenant-owned dataset, job, pipeline, model, deployment, artifact, and
  API key has a tenant or owner identifier.
- All service and repository reads and writes enforce that identifier.
- API keys are introduced only after identity exists. They are scoped,
  revocable, securely hashed at rest, and shown only once at creation.
- Audit events record security-sensitive actions, including login, key
  lifecycle, destructive operations, and access-denied decisions.
- Endpoint-level isolation tests prove that one tenant cannot read, modify, or
  delete another tenant's resources.

### Beta admission gate

Do not invite users until all of the following are demonstrated:

- Tenant-isolation tests pass across every resource type and endpoint.
- Secrets, HTTPS, backups, and restore procedures are configured and tested.
- Production monitoring and error-alerting are operating.
- Data-retention and deletion behavior is documented and implemented.
- A support and incident-response owner is identified.

## Feature Sequencing

After the hosted-beta foundation is complete:

1. Add explainability with bounded computation, artifact-size limits, and
   queued workloads for expensive requests. Start with supported model families
   rather than using unbounded generic explainers.
2. Add platform health and audit monitoring. Add model-drift history only once
   prediction/event retention, user consent, and deletion semantics are
   explicit.
3. Expand the invite-only beta based on metrics and direct feedback.
4. Consider self-service onboarding and billing only after the operational
   model is stable.
5. Reconsider deep-learning support only when validated user demand justifies
   GPU/dependency/infrastructure complexity.

## Non-Goals for the First Hosted Beta

- Public anonymous access.
- Self-service sign-up or billing.
- Unbounded SHAP/explainability jobs.
- Deep-learning/GPU platform support.
- Broad integrations or external webhooks beyond what the beta requires.

## Validation and Expansion Metrics

Track the following before widening access:

- Core installation and example reproducibility.
- User feedback, issue quality, and workflow adoption.
- Hosted job success and failure rate.
- API latency and error rate.
- Backup restore-test success.
- Tenant-isolation incidents, with a target of zero.
- Beta-user feedback and retention.

Access expands only when the security admission gate remains satisfied and
these metrics show the hosted platform is operating reliably.
