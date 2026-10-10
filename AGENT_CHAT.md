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

Task216 complete on branch 093. Source/docs: b23d34f9; test repair: 0ec4c406.
Purpose-based integration APIs preserve legacy imports/artifacts and wire values.
Public guides are updated; internal evidence remains under initiatives.
Local: 931 unique tests passed, one missing-pyspark skip. The source-snapshot
failure passed in isolation once source edits froze. Static/docs/commit gates passed.
Databricks: 858 unique tests passed across two runs (854 + 4), zero remaining
failures/skips; real Spark reader and both preprocessing/history guides passed.
Initial run 196440641386730; approved focused retry 106939883844620 SUCCESS.
The same b23d34f9 wheel and 576 runtime files were verified in both runs:
56ace36b953f44ee64e797e1075ae2fa5e2de010f43d2e186c431a00f9abe9e6.
Only two layout-test path constants changed for the retry. All 12 review IDs closed.
Evidence: initiatives/spark_and_mlflow/216-integration-naming-and-public-docs.md
and initiatives/spark_and_mlflow/task216_native/verified.json.
No push. No workspace table, registered model or endpoint created.

## Open messages

No active claims or pending requests. Completed review and execution details are
recorded in the task document; root's retry claims are released.
