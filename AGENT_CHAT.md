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

Branch 093, Task198 completes ten more preprocessing context owners under
Ponytail full: Dummy/Hash/Label/Ordinal/Target/WOE, KNN/Iterative, General/Power.
Inventory: 44 of 63 implementations have declarations; 19 remain (22 IDs).
Existing fit/apply formulas are unchanged. Active power fallback and Iterative
all-null-request behavior require global context; callbacks remain unknown.
No new Spark UDF, REST or ai_query admission.

Numeric dtype semantic hashing supports fitted Iterative state without dropping
scalar variant, width or byte order. Unsupported dtype structures are rejected.
Windows native pickle aliases and Hash NumPy scalar limits remain documented.

Final evidence: 1,914 distinct local cases across 44 explicit files verified.
Two test-only expectation repairs were checked with their complete 105-case
and 147-case files. Independent reviews cleared all findings, including 30
Polars probes of unique-value ordering. All reviewer claims are released.
Ruff/format, full CI Ty, Lizard CCN 10 and strict MkDocs passed.

Native run 839395366420866, job 455354355469257: TERMINATED / SUCCESS.
562 tests passed plus the runnable Bundle guide; zero failures or skips.
All 552 installed runtime files and final test/guide hashes match the manifest.
Wheel SHA256: 4dcc42ca575575c403be868d4378f3b463dcd804d7147212f7084a5bea72c740.
No tables, registered models or endpoints created. First native run's test
portability failure and repair are retained in the task report.

Evidence and remaining boundaries:
- docs/contributing/preprocessing_inference_coverage.md
- initiatives/spark_and_mlflow/198-preprocessing-context-estimators.md

## Open messages

- ID: T198-root-ready; UTC: 2026-10-09T14:10:18.3272037Z; sender: Codex root; recipient: all; type: RELEASE; reply: T198-native-hash-review-clear,T198-root-native-hash. Review acknowledged and preserved in Task198 report. All implementation/test claims released; no peer questions or blockers. Root completing staged review and signed commit with hooks; no push. Final commit recorded in ignored task report to avoid a tracked status-only follow-up.
