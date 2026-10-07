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

Branch 093: direct YAML templates and optional native feature lifecycle are
implemented and locally verified. User said no existing projects need migration;
new projects emit YAML directly and the unused migration path is removed.
SM-21a remains PARTIAL for native Databricks acceptance, explicitly user-deferred.
SM-21b remains optional/LATER. User requested the delivery182/183 commit;
implementation, template docs and focused evidence form one reviewed change set.

Evidence and operator limits:
- initiatives/spark_and_mlflow/182-native-feature-lifecycle-delivery.md
- initiatives/spark_and_mlflow/182-yaml-defaults-evidence.md
- initiatives/spark_and_mlflow/182-feature-packaging-evidence.md
- initiatives/spark_and_mlflow/183-feature-folder-layout.md

## Open messages

### task183-commit-ready | 2026-10-07T20:18:53Z | Codex root -> peers | RELEASE
Source and template changes are complete; all claims released and no pending questions. User authorized commit of delivery182/183, including direct YAML, optional native lookup lifecycle and src/features/groups isolation. Relevant passing tests and cloud/Delta limits are recorded in the evidence documents above. Final full Ruff/format1579/Ty/CCN and staged diff checks passed again. Git commit runs repository hooks; native workspace acceptance remains user-deferred. No push or cloud action requested.
