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

Task201 on branch093: fitted sentence encoder assets and all-preprocessor
small-data verification completed. Ponytail full. Root finalizing the signed
commit; no push. No active peer claims, unanswered questions or review findings.

Evidence: docs/contributing/preprocessing_inference_coverage.md and
initiatives/spark_and_mlflow/201-sentence-embedder-assets.md;
per-name results: initiatives/spark_and_mlflow/201-all-preprocessing-native-results.md.
Independent Codex review verified real encoder/tokenizer transport, dependency
pins, native-valid settings, exact group/history values and lifecycle behavior.
Completed exchanges and released claims were removed after recording findings.

Local:691passed/1Windows-symlink skip across21 affected files; new all-owner
matrix135passed, supplementary group/history/lifecycle22passed. Full Ruff,
format, CI Ty and CCN10 passed; final strict docs build passed in9.02s.
Native packaging:149/149passed, zero skips; job988493157672277/run968745626942362.
Native all-owner:157/157passed, zero skips; job994960019987268/run113581190209016.
67registered names/63owners, pandas67 andPolars67,32rows; no new Spark/REST
admission. Both runs verified all552 runtime hashes for wheel063d876b...;
no tables, registered models or endpoints created. Runtime unchanged since tests.

## Open messages
- ID: T201-root-release; UTC: 2026-10-09T15:39:16.7045976Z; sender: Codex root; recipient: all; type: RELEASE; reply: T201-root-final-report. All edit claims released after documentation and native evidence completed. Root owns staging/final commit only; no peer work pending. No push.
