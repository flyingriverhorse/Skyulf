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

## Latest handoff

C225-CLOSE | 2026-10-11T11:43:23Z | Codex/root -> peers | RELEASE | reply:C225-RESET
User authorized full project cleanup and S02 commit; no push or S03 implementation.
All parent claims released at commit handoff. Detailed results and actual commit
identity are recorded in Task225 and Task219 section 10; Task223 is the delivery tracker.
S02 runtime matches the tested wheel (539 Python paths/bytes); independent review
R1/R2/R3 resolved. Prior affected tests 134 and repair subset 79 passed (not summed).
Fresh Ruff/format/schema, strict MkDocs, full CI Ty/CCN10 and pre-commit passed.
No frontend changes; frontend hooks skipped. Passing pytest was not repeated.
Cloud reset verified: 49 jobs, 199 runs, 188 tables/views, 51 UC models, 18 schemas,
2 volumes, 28 experiments, 4 dashboards and 1 custom endpoint removed.
All 47 selected workspace roots absent (34 direct deletions, 13 already absent).
Fresh inventory plus 94 GETs passed; user objects/jobs/runs/active experiments,
pipelines, online stores and Lakebase projects are empty. Legacy registry disabled.
Platform catalogs/default schemas, 12 foundation endpoints, unrelated sample
dashboard and STOPPED starter warehouse preserved. Local evidence retained under
.superpowers/sdd/225-databricks-reset and prior task directories.
Progress 2/8; next S03 feature-update/rollout/champion acceptance needs fresh fixtures.
No pending peer questions or review requests. Completed exchanges recorded and compacted.
