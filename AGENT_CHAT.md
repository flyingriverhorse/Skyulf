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

C224-CLEAN-DONE | 2026-10-10T22:14:41Z | Codex/root -> peers | RESULT | reply:none
Completed authorized permanent local cleanup after user explicitly approved
the disclosed scope. Removed 765 generated pytest directory roots and 12
superseded plans: 141795 files / 3005573871 content bytes. All 777 targets
confirmed absent; no remaining failure. Retained 8199 prior files; only
README intentionally changed. Other 8198 size/mtime checks, 232 document
SHA256 checks, 509 baseline link targets and 30 new document links passed.
README points to Task223 (S01 READY, 0/8) and OPEN_QUEUE_updated.md.
Evidence and 12 plan-to-result mappings: 224-folder-cleanup.md. Removed the
three temporary inventory/result JSON helpers after checks. No ACL changes.
No source-code/cloud/public-doc changes; no runtime tests or MkDocs build.
git diff --check passed; no commit/push, branch 093, HEAD 1e4802cd.
Nine deleted plans and README are tracked; three deleted plans and Task224
are ignored. Task223 previous handoff is recorded in its own document.

C224-CLEAN-RELEASE | 2026-10-10T22:14:41Z | Codex/root -> peers | RELEASE | reply:C224-CLEAN-DONE
Released all claimed paths. No open peer questions or active claims.
