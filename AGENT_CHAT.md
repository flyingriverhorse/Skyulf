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

Branch 093. Task191-192 delivery: saved-preprocessing diagnostics, context declarations, captured custom replay, detailed main-template PREPROCESSING_CONTEXT.md, and tracked 50-implementation backlog at docs/contributing/preprocessing_inference_coverage.md. Fit/apply implementations remain the execution owners; no new worker admission or automatic job invocation. No native Databricks run for this diagnostic.

Prepared for the user-requested local DCO commit with 18 reviewed files. Fresh explicit affected union: 603 passed (42 known warnings). Guide fit/save/load/probe and custom factory examples executed; 50 backlog rows exactly match 54 registry names. Full Ruff/CI Ty/format/Lizard/diff checks, strict MkDocs build and applicable pre-commit hooks passed. Independent documentation review corrections applied. All file claims released; completed exchanges consolidated in initiatives/spark_and_mlflow/191-preprocessing-inference-probe.md and 192-preprocessing-docs-commit.md. Commit readback is recorded in the latter report; no push requested.

## Open messages
