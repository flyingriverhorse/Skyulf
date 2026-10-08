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

Branch 093. Tasks191-192 committed as a50ddd8b: saved-preprocessing diagnostics, context declarations, captured custom replay, the main-template PREPROCESSING_CONTEXT.md guide, and tracked backlog at docs/contributing/preprocessing_inference_coverage.md. Existing fit/apply owns execution. Local affected union: 603 passed; static gates and commit hooks passed. No push.

Task193 native follow-up completed at the user's request: Databricks skyulf profile, serverless STANDARD run 55789884205809, task 396852663331545, API TERMINATED / SUCCESS. All 213 explicit cases passed, none skipped, plus the real guide example. All 552 installed Python source hashes matched a50ddd8b. No runtime fix, Spark UDF execution, REST deployment, UC tables or model registrations. The 50 remaining implementations are still open. Fresh evidence is in the tracked coverage report and initiatives/spark_and_mlflow/193-preprocessing-native-verification.md; raw output in task193_native/output.json. Strict MkDocs build passed. This follow-up contains documentation evidence only; final commit readback is recorded in Task193. All task file claims released.

## Open messages
