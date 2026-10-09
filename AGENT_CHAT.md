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

Branch 093. Task195 adds local saved-state inspection and row context to ten
preprocessing families; reuses existing transforms and keeps worker admission
unchanged. Live inventory: 23 declared implementations, 40 undeclared / 44 IDs.
Three exact-parity follow-ups remain documented (pandas numeric bins/object
replacement and Polars MaxAbs floating-point rounding).

Final source: 1342 local tests passed; full Ruff/format/Ty/Lizard and strict docs
passed. Native final wheel 3e8a5eac, 552 installed source files matched: run
585541621081287/job903098952257050/task19609502034591 SUCCESS; 610 passed, no failures or skips
plus runnable guide. No tables, registered models or endpoints created. Six
review findings repaired and independently checked; reviewer clear. Details and
exact evidence: initiatives/spark_and_mlflow/195-preprocessing-context-batch.md
and docs/contributing/preprocessing_inference_coverage.md. Commit readback will
be recorded in Task195; no push requested.

User side request: ponytail marketplace added and freshly listed; no individual
plugin installed. Original preprocessing task continuation explicitly reaffirmed.

## Open messages

- ID: T195-root-release; UTC: 2026-10-09T12:47:05; sender: Codex root; recipient: all; type: RELEASE; reply: T195-selectors-review-clear,T195-scale-missing-release,T195-stateless-compat-release. All findings/results preserved in Task195, acknowledged and compacted. All implementation/review claims released; final staging/hooks/commit belong to root.
