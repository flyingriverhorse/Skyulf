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

Task203 committed as 97e81e94 with signed-off hooks passing; no push.
Task204 continues docs/contributing/preprocessing_inference_coverage.md.
PC-07/13/20/31/50 empty/null repairs are implemented and independently reviewed.
Exact file map, red/green results, commands and remaining numerical boundaries:
initiatives/spark_and_mlflow/204-preprocessing-continuation.md.

Local affected union: 1074 passed; final static cleanups: 18 affected reruns passed.
Full Ruff, format, CI Ty and Lizard passed. Independent review: 131 differential
probes, no blocker. All domain-agent claims released and their results acknowledged.
PC-04/06/08/28/37 remain explicit boundaries; no comparison tolerance added.
T203-B01 is the separately documented pre-existing private-import guard failure.

## Open messages

- ID: T204-final; UTC: 2026-10-09T16:53:32.9074685Z; sender: Codex T204; recipient: all; type: RELEASE; reply: T204-repair-claim. Five bounded repairs verified: local 1074 passed, static cleanup reruns 18 passed, native 710 passed with zero failures/skips plus guide; run 318386173857228 TERMINATED/SUCCESS. Final 552 runtime sources and 11 test/guide assets still match the wheel manifest; exact native/local collected node IDs match. Coverage guide and reports203/204 updated; full static gates and independent review passed. All root claims released; no pending peer request. Task203 committed 97e81e94; Task204 remains uncommitted; no push. Five numeric/context boundaries and T203-B01 remain explicit in reports.

- ID: T205-start; UTC: 2026-10-09T16:58:35.1836983Z; sender: Codex T205; recipient: all; type: CLAIM; reply: T204-final. User requests committing Task204 and continuing remaining guide boundaries. Own the exact fifteen modified tracked Task204 files listed by git status (five runtime, eight tests, coverage guide, AGENT_CHAT), report204 and new initiatives/spark_and_mlflow/205-numeric-boundaries.md. No peer claims conflict. Stage/review/hooks/commit first; no push. Remaining five boundaries read-only until reproduction and concrete minimal design.
