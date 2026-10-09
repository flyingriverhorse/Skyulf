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

Task210: exact GroupImputer integers and Polars zero-width row preservation.
Base0fa6f88d, branch093, clean start. Root owns finalunion/static/native/docs/commit.
Evidence: initiatives/spark_and_mlflow/210-group-integer-empty-context.md. No push.

## Open messages

- ID: T210-handoff; UTC: 2026-10-09T19:21:59.3253312Z; sender: Codex T210; recipient: all; type: RELEASE; reply: T210-root. Local source/tests/docs frozen for requested DCO commit; all file claims released. Final15-file union575passed60warnings59.87s; Ruff/format/fullCI Ty/Lizard/strictMkDocs clean, independentreview no blocker. Full evidence and migration notes in Task210report and coverageMD. New575-test cloud package requires pending user approval; no upload/run yet. Resume task210_native prepared harness after approval, verify exactnodes/sourcehashes, record native result. Read git history for final local commit; do not push.
