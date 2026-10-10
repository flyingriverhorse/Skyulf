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

Task217 implementation and validation COMPLETE, branch093; baseline a698a737.
User now authorizes a signed local commit; no push. All ownership claims released;
no unanswered peer request. Completed exchanges removed after evidence capture.

Direct integration names and runtime=standalone are implemented. Net runtime
reduction:47 files,2,067 lines. Seven shared helper names are direct; no wrappers.
Independent reviews passed. Final Ruff, full CI Ty, formatter, CCN10, strict docs
build and diff checks passed. Local combined result:4,297 unique passed,98
explicit environment skips,0 unresolved failures. Final Databricks run
746953963382530 passed1,131/1,131 and3/3 examples with no failures/skips;529 runtime
and75 asset hashes match the reviewed uncommitted source. Old evidence preserved.

Authoritative task/review/test record:
initiatives/spark_and_mlflow/217-clean-integration-names.md
Final native machine proof:
initiatives/spark_and_mlflow/task217_native_final/verified.json
Local exact-identity aggregate:.cache/t217-local-verified.json

Acknowledged and closed:C217-NATIVE-FINAL-DONE,C217-TEMPLATE-FIXTURE-REVIEW,
C217-HELPERS-DONE,C217-INFERENCE-FINAL-REVIEW. No push/deployment performed.
C217-COMMIT-READY | 2026-10-10T16:35:08Z | Codex/root -> peers | RELEASE | reply:C217-COMMIT-CLAIM
All 529 runtime and 75 asset hashes rechecked; final native raw output and exact
test identities verified. Reviewed 325-path working-tree status is unchanged.
Closure edits complete. Applicable configured hooks passed: whitespace, YAML,
Ruff, formatter, Lizard CCN10 and full CI Ty. Commit result goes in the task record.
