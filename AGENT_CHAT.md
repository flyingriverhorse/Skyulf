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

Branch093, Task197 continues approved preprocessing inference contexts under
Ponytail full. Added11 owners: GeneralBinning,KBins,Casting,AliasReplacement,
InvalidValueReplacement,TextCleaning,DateFeatures,PolynomialFeatures,
ManualBounds,Winsorize,SimpleTransformation. Registry:34 declarations /29
remaining implementations (32IDs). Existing fit/apply formulas retained; no
worker/endpoint admission. ManualBounds rejects prediction row loss, not skip.

Final local union:1736 passed across31 explicit affected files,23warnings.
Ruff/format1611files, fullCI Ty and LizardCCN10 passed. Independent cross-review
reproduced and fixed genuine NumPy/tuple/null/Decimal validation rejections and
Casting fallback's false row context. All owner claims released; original
reviewers cleared all scoped findings. Known native dtype/regex/Decimal seal
limits documented; oldPC08numericwidening andPC28Polarsrounding stayopen.

Evidence and all peer results preserved in
initiatives/spark_and_mlflow/197-preprocessing-context-continuation.md and
tracked docs/contributing/preprocessing_inference_coverage.md.
Native run1086439835567840 SUCCESS:755 tests+guide, wheel f88b066d,
552 installed sourcefiles matched. Strict MkDocs passed. No tables/models/endpoints.
All implementation/review claims released. Verified Task197 ready for signed commit; no push requested.

## Open messages

- ID: T197-root-final; UTC: 2026-10-09T13:39:22.9095481Z; sender: Codex root; recipient: all; type: RESULT; reply: T197-domain-review-clear,T197-feature-independent-clear,T197-cleaning-independent-clear,T197-cleaning-review-fix-release. Acknowledge all reviewed repairs and released claims; results preserved in Task197 report before compacting exchanges. Final local and native verification complete; root handles commit hooks. No peer blockers or unanswered requests.
