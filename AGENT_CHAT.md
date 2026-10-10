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

Task216 implementation is now authorized by the user. Previous planning-only
restriction is superseded; no push. Root coordinates implementation on branch 093.
Task: initiatives/spark_and_mlflow/216-integration-naming-and-public-docs.md.
It explains preprocessing_probe, proposes purpose-based replacements for
misleading local names, preserves meaningful locality and saved-model/import
compatibility, and inventories public integration documentation for a rewrite.
Internal task history and run evidence belong under initiatives, not public docs.
All 13 test-file references in the task were checked against the checkout.
Previous Task215 implementation/native validation is complete; evidence remains
in initiatives/spark_and_mlflow/215-preprocessing-completion.md.

## Open messages
Active claims below. Inference domain release and verified results recorded in Task216.

- T216-START | UTC 2026-10-10T11:32:04.1212075Z | root -> all | CLAIM | reply none: Implement approved Task216 with Ponytail and plan execution. Root owns this log, initiatives/spark_and_mlflow/216-integration-naming-and-public-docs.md and ignored .cache/t216-* evidence. Inference/MLflow, Databricks names and docs will be delegated with separate exact file claims. Root owns final union/static checks and integration review; agents run focused RED/GREEN only. Existing branch 093 and shared checkout retained for continuation.

- T216-LOCAL-DONE | UTC 2026-10-10T12:01:43.8872196Z | root -> all | RESULT | reply T216-START: All released typing/review evidence recorded in Task216. Final union 930 passed, 1 source-drift failure, 1 missing-pyspark skip; exact frozen-source replay passed, yielding 931 unique local passes and 1 skip. Full Ty/Ruff/format/CCN10/schema/MkDocs pass. All142 reviewed hashes current, staged diff checked. Native wheel576 source files exact;858 packaged node IDs collected in isolation. Root completing source commit then native verification. No push.
