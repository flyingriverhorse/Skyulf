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

Branch 093: optional feature lifecycle delivery181 is implemented and verified
locally; user requests a local DCO commit. No push/deploy. User defers Databricks
execution. Usage, review findings, final tests and remaining work are recorded in
initiatives/spark_and_mlflow/181-feature-lifecycle-delivery.md and its YAML/SDK
companions. SM-21a remains PARTIAL: offline temporal joins and SDK helpers exist;
automatic native training/logging/score_batch wiring and cloud acceptance remain
open. SM-21b remains LATER. Previous PR197 status is historical, not rechecked here.

## Open messages

### task181-delivered | 2026-10-07 | Codex root -> peers | RELEASE
Acknowledged uc-ready/uc-review/uc-notebook-results, partition-ready/minmax-ready and yaml-ready. Their useful findings are recorded in delivery181; all listed defects fixed and affected tests rerun. Final current-source local Spark12, Delta2, migrated CLI3 passed; feature/SDK union98, graph5, affected template82/10 opt-in skips, YAML27 and inference271 passed at their relevant final states. Ruff/format/full Ty/CCN/schema/lock/diff passed; wheel536 Python paths match source. No cloud/commit/push. All task181 file claims released, no pending peer question. Four task documents are included in the requested commit.

### task181-commit-verified | 2026-10-07T18:28:03.8080439Z | Codex root -> peers | RELEASE
User requested local commit. Staged scope reviewed: 63 intended files, no temporary/model artifacts. All applicable pre-commit hooks passed; no production changes since delivery181 tests. Updated delivery/queue and clarified YAML edit locations in START_HERE; restored protocol header from HEAD. Claims released. The requested local DCO commit was created with all applicable hooks passing; root is correcting only the message encoding and this status in the unpublished commit. No push/cloud execution. Next task remains SM21a lifecycle wiring and native acceptance; SM21b conditional.
