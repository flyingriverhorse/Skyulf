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

Task200 on branch093, base signed commit a8f98f23, Ponytail full.
Seven remaining training/filter/inspection owners reviewed. Six add native
local metadata; TrainTestSplitter/Split stays unrecorded with no useless hook.
Counts:61/63 declaration machinery;62/63 reviewed lifecycle; sentence_embedder
asset packaging remains open. No new worker admission or fit/apply formulas.

Temporal scalar digest collisions found in real snapshots now fail explicitly.
Supported scalar/container/array digests and NumPy temporal arrays unchanged.
All domain implementation and independent review claims released; root has
acknowledged results and preserved evidence in:
- initiatives/spark_and_mlflow/200-preprocessing-context-training.md
- docs/contributing/preprocessing_inference_coverage.md

Local union:1,438passed across36 explicit files. Test-only invalid-None type
annotation fixed afterward; its entire18casefile passed, fullCI Ty then passed.
Ruff/format/CCN10 also passed. No runtime edits after union. Source/test/guide
hashes packaged for native verification. No unresolved review findings.

Native job750853859501812/run152339084173952:TERMINATED/SUCCESS.399passed
plus Bundle guide, zero failed/skipped. Full duration81.150s; pytest22.56s.
552 runtime files in wheel SHA256:
ea03ef1d8e68bc3907ed3b0235ba4eaa940deb7c38c28cfad755c59b97db2be6.
All552runtime/10test-guide hashes stillmatch. StrictMkDocs passed7.25s.
Root completing staged review and signed commit hooks. No push.

## Open messages

- ID: T200-root-ready; UTC: 2026-10-09T15:00:02.0428312Z; sender: Codex root; recipient: all; type: RELEASE; reply: T200-root-finalization. All Task200 implementation/test/doc claims released. Local1438/native399 plusguide verified; hashes current; independent reviews clear. Ruff/format/fullCI Ty/CCN10/strictMkDocs pass. Final staged review and signed commit hooks next, no push. Commit will be recorded in ignored Task200 report. No open peer questions.
