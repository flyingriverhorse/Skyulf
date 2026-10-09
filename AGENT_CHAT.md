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

Branch 093, Task199 adds eleven preprocessing context owners under Ponytail full.
Base a4ff850e. Scope: PC19/22/25/31/34/35/38/46/47/49/50. Inventory is 55/63;
eight owners (nine registered IDs) remain. Existing transform formulas are reused.
A shared Polars text-only scoring defect was fixed. No new worker admission.

All three domain agents released their implementation/test claims. Root
acknowledges their results and independent reviews; evidence is recorded in
initiatives/spark_and_mlflow/199-preprocessing-context-features.md and the tracked
docs/contributing/preprocessing_inference_coverage.md. No unresolved findings.

Local evidence: 2,154 distinct cases in 44 explicit files. Initial union had
2,140 passes and 14 obsolete negative expectations. After test-only corrections,
both affected complete files passed 177 tests. Independent fixture review clear.
Full CI Ruff/format/Ty/Lizard CCN10 passed. No runtime edit after the union.

Native run 994473515746072, job 830129841788466: TERMINATED / SUCCESS. 712 passed
tests plus the executable Bundle guide. All 552 installed runtime hashes match.
Wheel SHA256 bbb85097d7813c5870b8055a1bc5ee546736c22c8306df48f85a25081dc4b3cf.
Zero failures/skips, zero UC tables/models/endpoints created. Strict MkDocs passed.
Root completing staged review and signed commit with hooks. No push.

## Open messages

- ID: T199-root-ready; UTC: 2026-10-09T14:42:19.2126879Z; sender: Codex root; recipient: all; type: RELEASE; reply: T199-root-finalization. All Task199 implementation/test/doc claims released. Native712 plus guide passed; all hashes current; local2154 verified and independent reviews clear. Ruff/format/fullCI Ty/CCN10 and strictMkDocs pass. Staged28file diff reviewed with no generated artifacts. Root final signed commit with hooks next, no push. Record final commit in ignored Task199 report; no open peer requests.
