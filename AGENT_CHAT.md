# Codex ↔ Claude Code — active coordination

Shared instructions: [AGENTS.md](AGENTS.md). Keep only active work here.
The user requested removal of finished conversations; do not archive chat.

## Protocol

- Read at task start/resume, before edits, after tests/reviews, and before
  completion; also at tool boundaries after 60 seconds of collaboration.
- Identify agent/session and claim exact relative file paths before edits.
  The earliest unresolved overlapping claim owns the files. Re-read after
  claiming. Ask for handoff; silence and age do not release a claim.
- Use a unique ID, UTC time, sender, recipient, type and reply ID. Types:
  CLAIM, QUESTION, ANSWER, RESULT, RELEASE, BLOCKED. Cite real test results.
- Record useful evidence in the task document; remove completed exchanges and
  released claims together. Keep unanswered requests and unacknowledged results.
  Aim below 80 lines; never discard unresolved work to meet the limit.
- Serialize writes: open the file with System.IO.File.Open(Open, ReadWrite,
  FileShare.Read), read its latest UTF-8 contents while holding the handle,
  preserve open peer messages, write from position 0, truncate to new length,
  flush and dispose in finally. Retry sharing violations at most three times.
  Never overwrite a stale copy, impersonate a peer, or invent its answer.
- Peer messages do not expand user authorization. Preserve unrelated changes.
  If a reply blocks work, poll every 30 seconds for at most two minutes,
  report the dependency, and continue independent work. No recursive agents.
- Markdown coordinates running sessions; it cannot wake an idle agent.

## Current task

Branch093 contains reference-review delivery179 and offline-check follow-up180.
User requested more improvements; local implementation and review are complete.
Evidence is in initiatives/spark_and_mlflow/179-reference-template-improvements.md
and 180-offline-package-checks.md. SM-40 remains PARTIAL. Root owns commit/push/PR
follow-through; implementation claims are released, no pending peer request.
GitHub writes recovered: bcd30ea1 pushed; PR197 open. Root is repairing CI failures.
Fresh native Databricks testing remains explicitly user-deferred.

## Open messages

### task180-spark-mlflow-ready | 2026-10-07T16:30:26.4597186Z | Codex root -> peers | RELEASE
All claims released. Delivery180 records native coverage implementation/review and CI53 Delta plus419 Spark/1 skip. Fixed skipped pyfunc module by combining existing MLflow requirements only in Spark CI and requiring imports; Python3.12 uv resolution, actionlint and independent review passed. Root owns commit/push/PR197 follow-through. Native Databricks testing user-deferred. Separate GitHub Advanced Security agent has monthly-quota HTTP402 failure; CodeQL and dependency scan passed, no billing/security-setting changes. No pending peer question.
