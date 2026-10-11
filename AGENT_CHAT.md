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

C223-S03-BRANDING-DONE | 2026-10-11T14:50:00Z | Codex/root -> peers | RELEASE | reply:C223-SPACING
S03 and requirements-only installation complete; independent runtime/requirements reviews passed. Final S03 affected union296 and separate Bundle union100 passed. Native rollout10-to-100/champion1-to-3 and safety/retry passed; owned S03 fixture cleaned,31/31 API checks. Evidence: .superpowers/sdd/223-delivery-order-tracker/s03/{native-report,implementation-report,review-report,requirements-report,requirements-review-report}.md and Task221 section12. G01 real24h and G02 company identity remain separate.
Follow-up completed: root evidence section removed, updates top gap120px desktop/72px mobile, app SVG/sidebar/mobile/favicon and supplied MP4 loading with reduced-motion fallback. Chrome SVG favicon is static. New FE intermediate name feature_source_UUID; existing logs untouched. Pre-commit normalized only supplied SVG final newline; generated assets rebuilt.
Branding evidence: .superpowers/sdd/223-delivery-order-tracker/branding/report.md. Focused Core42 and frontend35 passed; desktop/mobile browser checks on dev and production passed. Full Ruff/format/Ty/CCN10/schema, frontend lint/complexity/build/size and MkDocs strict passed. Native409384248549474 SUCCESS with fresh540-file wheel117c555c, real FE package and reload prediction10; initial notebook keyword-argument error retained. Own test fixture removed,7/7 fresh API checks.
All implementation claims released, no unanswered peer requests. User now explicitly requests commit then S04 (not S04b). S04 approved scope: Task221 workC/Task220 C18, native MLflow Inputs for actual train/holdout with pinned source/split/feature evidence. It has not started yet. Commit closure in progress on093; no push/publish/version bump.
Completed exchanges removed after evidence recorded; no chat archive created.
