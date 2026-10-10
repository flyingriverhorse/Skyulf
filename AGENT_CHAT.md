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

C218-COMMIT-DONE | 2026-10-10T21:10:08Z | Codex/root -> peers | RESULT | reply:C218-COMMIT-CLAIM
Task218 DCO commit completed with title:
feat(databricks): add guarded serving rollout and online feature lookup.
See git log for final hash after this coordination-only amend. No push.
All applicable pre-commit hooks passed (schema, whitespace, YAML/JSON, Ruff,
format, full CI Ty and CCN10). Frontend hooks skipped: no frontend changes.
Whole Ruff and strict MkDocs passed.539runtime/676test files still match the
native final6 package; passed pytest groups were not repeated unchanged.
Record: initiatives/spark_and_mlflow/218-rollout-online-features.md.

User next requests streaming but asks what publication selection enables.
Explained: include_online_publication only adds a sync job; it does not make
the endpoint use Lakebase. Native model lookup packaging and published features
are separate prerequisites. Scope is not settled between continuous feature
sync (Delta->Lakebase) and continuous prediction (Delta->prediction table).
User answered the scope question with clarification questions, not a selection.
No streaming implementation/config/cloud changes have been made.

C218-COMMIT-RELEASE | 2026-10-10T21:10:08Z | Codex/root -> peers | RELEASE | reply:C218-COMMIT-DONE
All Task218 source/test/docs/task/queue claims released. No peer question or
active claim. Preserve the separate streaming scope discussion; no agent has
been started for it. This coordination-only update is included in the commit.
