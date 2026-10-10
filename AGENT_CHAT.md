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

Task215 preprocessing completion on branch093; base48de280d. User approved all
four bounded improvements and delegated design choices. No push.
Implementation and independent reviews complete. Final scoped local evidence:
1170 distinct tests passed after stale fixture repair and offline CLI reruns.
Ruff/full format/Ty/CCN10/schema and strict MkDocs passed.
Evidence: initiatives/spark_and_mlflow/215-preprocessing-completion.md and
tracked docs/contributing/preprocessing_inference_coverage.md.

## Open messages

- T215-ROOT-FINAL | UTC 2026-10-10T11:05:00Z | root -> all | CLAIM | reply T215-ROOT-ACTIVE: All agent claims released and independent reviews incorporated. Root owns final commit, native package/upload/run/evidence and follow-up docs. Native validation pending. All earlier review handoffs answered; findings preserved in task document.
