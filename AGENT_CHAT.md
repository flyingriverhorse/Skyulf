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

Task216 code/docs committed on branch 093: b23d34f9. No push.
Purpose-based integration APIs preserve old imports/artifacts and wire values.
Public integration guides rewritten; internal evidence stays under initiatives.
Local outcomes: 931 unique tests passed, one missing-pyspark skip. The union's
one source-snapshot failure passed in isolation after typing edits froze.
Full Ty/Ruff/format/CCN10/schema/MkDocs and commit hooks passed.
Task/evidence: initiatives/spark_and_mlflow/216-integration-naming-and-public-docs.md.
Native package ready: 858 tests, 576 exact runtime files, wheel SHA256
56ace36b953f44ee64e797e1075ae2fa5e2de010f43d2e186c431a00f9abe9e6.
Automatic approval review rejected the new source-package upload because prior
approvals covered different payloads. Specific user approval question pending.
No Task216 upload/run occurred. Resume with .cache/t216-submit.py after approval,
then poll and verify with .cache/t216-poll.py and .cache/t216-verify-native.py.

## Open messages
No active file claims. Domain findings and reviews recorded in the task document.
