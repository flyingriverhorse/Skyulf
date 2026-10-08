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

Branch 093. Saved preprocessing context/probe implementation: a50ddd8b. Task193 native Python verification documented in cf2a420a: 213 passed plus executable guide; earlier local union 603 passed. The tracked backlog at docs/contributing/preprocessing_inference_coverage.md still has 50 open implementations; fit/apply execution is reused, not duplicated.

Task194 real three-route smoke is complete: run 537211044950378, task 308830078979108, API TERMINATED / SUCCESS. Current runtime wheel 4df838c6, 552 installed source files verified. One registered RF model with six preprocessing steps, 8 rows including nulls/unseen categories/large keys; actual Spark UDF batches 1/3 over two partitions, REST and typed UC ai_query all matched all prediction/probability outputs exactly (max error 0.0). All six context probes passed. Temporary endpoint deleted; original endpoint identity and model version preserved by fresh API readback. No runtime repair needed. Evidence: initiatives/spark_and_mlflow/194-three-route-smoke.md and task194_native/verified-result.json, plus the tracked coverage report. Strict MkDocs build passed; documentation-only follow-up, no push. Commit readback belongs in Task194. All task claims released.

## Open messages
