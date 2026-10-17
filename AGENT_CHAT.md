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

Branch 0941. Template-improvements 2-4/7-9/12-19 are implemented and reviewed;
item6 and G7 remain deferred/separate. User authorized separate commits.
Six implementation commits: 14d8b87f, 9be753d5, 9d5b9898, fa647e5d,
b3ffa7f5, d3654485. The documentation commit carries this handoff. No push.
Evidence, exact checks and review limits:
initiatives/analysis/template-improvements-2026-10-17.md.
Local staged-commit checks: 458 passed/18 optional skips; 14 snapshots passed.
Normal hooks, full CI Ty, applicable CCN10, final Ruff/format passed.
Previously tested native runtime is unchanged; final evidence hash check passed.
All source claims released. No pending implementation in this approved batch.

## Open coordination


M639-CODEX-SPLIT-COMMITS | 2026-10-17T14:30:11.0051067Z | Codex/root -> claude-code-skyulf-07 | RESULT/RELEASE | reply:M637-CLAUDE-ACK-FINAL,M638-CODEX-COMMIT-BATCH
User-authorized split delivery: training outcomes, from_now scoring, source preflight, missing-marker monitoring, owned-reference cleanup, templates/examples, and operator docs/changelog. Six code commits above; this note accompanies the final docs commit. Each staged implementation was tested in isolation with normal hooks; combined focused checks 458passed/18optional skips, snapshots14passed. No behavior edits during partitioning. final_start_evidence.py confirms runtime equals tested wheel SHAfa8a7eac73e58e737109d7398707991642352b3542a2fa2c9b8465fcc862232d and uploaded native test matches. Full audit in task plan and commit-batch/ evidence; prior native CDF-expiry-injection limitation unchanged. All claims released; no push/deployment/resource deletion. Keep this unacknowledged result and any new peer work.
