# SM-32 live operator rehearsal

Date: 2026-09-25. Authorized by the user's request to finish Databricks testing.

Use profile `skyulf`, existing schema `workspace.skyulf_lifecycle_test`, and
existing Bundle jobs train `155738051514173` and score `684955889505992`.
Preserve existing source, prior models and prediction outputs. Register fresh
`sm32_model_polars` and `sm32_model_pandas` with separate prediction outputs so
bootstrap and artifact provenance can be tested without deleting earlier evidence.
No new schema, persistent job or admission table. Every job run is bounded to
900 seconds, with retries disabled and schedules inactive.

1. Save deployed settings, migrate the existing generated project to the current
   template, and upload the verified wheel under a fresh SM-32 workspace path.
2. Run Polars manual training and first approval through the real lifecycle job.
   Verify manual training skips score and approval calls the separate score job.
3. Train and approve an improved candidate. Train later candidates to exercise
   previous_challenger, reject one, and roll back the saved completed promotion.
   Verify rejection skips score and rollback follows its explicit handoff choice.
4. Use the same two jobs with pandas/automatic configuration. Verify first
   champion, improvement, tied candidate and scoring-pin independence.
5. Audit concrete UC aliases, model counts/metrics, prediction counts/provenance,
   operator retry/no-fit behavior and actual condition/Run Job task outcomes.
6. Leave the Bundle in an explicit manual/champion configuration for the user's
   practice, with a reviewable candidate when possible. Publish exact run links
   and UI instructions in an English user guide with Mermaid diagrams.

Deployment/config changes occur only between completed runs. An unexpected
alias outcome requires inspection before any retry. A score failure after a
committed promotion is recovered by scoring without retraining. Do not declare
SM-32 complete until the planned runtime checks have evidence.
