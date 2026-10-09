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

Task206: PC08 ValueReplacement and PC37 Winsorize integer precision repairs.
Ponytail review uses native guards; no new framework, dependency or artifact field.
Base d6256f69, branch093. User authorized commit; no push.
Evidence and remaining boundaries:
initiatives/spark_and_mlflow/206-replacement-winsorize-precision.md.

Local affected union1139passed; final precision reruns68+43passed, Arrow1passed.
Full Ruff/format/Ty/Lizard passed. Independent review found no blocker.
Native538passed, zero failures/skips, guidepassed, run1017340571205314 SUCCESS.
All552runtimefiles and16assets matched; exact local/native test-node sets match.
Coverage guide records explicit Casting/refit requirements and native boundaries.
All claims released. Strict docs passed; delivery commit is the remaining action.
Exact commit outcome/hash is recorded in report206 after the commit command.

## Open messages

- ID: T206-reviewed; UTC: 2026-10-09T17:43:33.2540514Z; sender: Codex T206; recipient: all; type: RESULT; reply: T206-value-result,T206-winsor-result,T206-value-ty-release,T206-full-ty-pass,T206-winsor-freeze. Acknowledged all owner results/releases and independent review. Evidence consolidated in report206 and coverage guide; completed exchanges removed. Root owns remaining docs/commit; no peer question or active domain claim. No push.
