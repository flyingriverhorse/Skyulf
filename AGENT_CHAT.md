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

Task215, branch 093, source 00254254. Implementation and reviews complete;
1,170 distinct local tests passed with static/schema/docs and source hooks.
Databricks: 1,093 distinct tests passed across two runs with the same wheel.
First run 580851312882650: 1,091 passed, two failed due to an omitted test extra.
Approved retry 100608908263263/job504858919108396: SUCCESS, two passed, guide
passed; added optuna-integration 4.5.0 to the environment, no source change.
Combined evidence verified against 553 runtime files, 56 assets, five markers,
collection hashes and the exact retry node IDs. Strict MkDocs passed.
Evidence: initiatives/spark_and_mlflow/215-preprocessing-completion.md and
tracked docs/contributing/preprocessing_inference_coverage.md. No push.

## Open messages
None. Task215 claims are released; final evidence is recorded in the task document.
