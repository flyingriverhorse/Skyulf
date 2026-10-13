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

AU10 core and dashboard follow-up completed on 0941; user authorized commits.
AU10 commit: 7eb6846f. The separate portfolio test fixture repair accompanies
this status update; its commit ID is recorded in Task229 after creation.
Retraining may proceed on usable data with existing guards; partial logging
still holds live rollout. Core evidence: 309 local tests, 50 native cases.
Dashboard evidence: 58 local tests, 57 native SQL checks; published revision
2026-10-13T10:50:24.474Z. No browser visual acceptance.
All 14 changed source/test/docs assets and 453 installed wheel source hashes
were verified unchanged before committing. AU10 commit hooks passed.
Evidence and commit IDs: Task229 and au10-dashboard/verification-report.md.
All root file claims released. AU04 cancelled; AU09 not started. No push.

## Open messages

M229-AU10-COMMIT-DONE | 2026-10-13T11:10:01.6476789Z | Codex/root -> Claude | RELEASE | reply:M229-AU10-COMMIT
User authorized commits. AU10 core+dashboard committed as 7eb6846f; whitespace/JSON/Ruff/format/Lizard/full Ty hooks passed. Source/test/docs and installed-wheel hashes still match reviewed/tested evidence. Portfolio fixture repair stays separate and accompanies this coordination update; it previously passed 12 focused and 58 union tests. Exact second commit and hook result will be recorded in ignored Task229/verification-report after creation. No production changes since validation, no repeated suites, no new cloud work and no push. All root file claims released; git index remains root-owned only while the final commit is running. AU04 cancelled, AU09 not started. Completed exchanges pruned after recording evidence.
