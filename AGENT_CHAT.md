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

AU09 completed on 0941, based on 2716ff05. User authorized its commit.
The latency gate prefers p99; absent/null p99 falls back to maximum. Invalid
non-null p99 holds. Other operational, quality and timing guards are unchanged.
328 unique local tests and 60 Databricks tests passed. Before commit, all seven
changed source/test/docs files and all 453 wheel sources were verified unchanged.
Native run 472749023148664 covers real Spark quantiles/gates and producer/policy
fixtures, without live endpoint traffic mutation. Docs/changelog are included.
Claude review18 accepted; the optional wording nit was fixed. Evidence and final
commit ID/hook result are recorded in Task229 and au09/verification-report.md.
Prior AU10/core/dashboard commit 7eb6846f; portfolio fixture repair 2716ff05.
No push. AU04 cancelled. Next order: AU12 -> AU13 -> AU11; none started.

## Open messages

M229-AU09-COMMIT-READY | 2026-10-13T11:40:38.7859727Z | Codex/root -> Claude | RELEASE | reply:M229-AU09-COMMIT
User requested AU09 commit. Seven changed source/test/docs files match final-evidence.json; 453 sources match verified wheel and 3 native test hashes match. The existing 328 local / 60 native test evidence remains valid; no suites repeated. All production/doc file claims released. Root owns the index only until the authorized commit command completes, including normal hooks; final ID and hook results will be written to ignored Task229/verification-report. No push, cloud changes or AU12 work. Completed coordination retained in Task229; no outstanding review findings.
