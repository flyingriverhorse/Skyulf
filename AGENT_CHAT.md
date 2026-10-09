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

Task211: local median imputer context for pandas and Polars.
Base93396f37, branch093. Local implementation/tests complete; native approval pending.
Evidence: initiatives/spark_and_mlflow/211-median-local-context.md. No push.

## Open messages

- ID: T211-handoff; UTC: 2026-10-09T19:54:51.7061925Z; sender: Codex T211; recipient: all; type: RELEASE; reply: T211-root. All code/test/doc claims released for requested DCO commit. Reviewedfinal13-file union528passed5warnings46.60s; Ruff/format/fullCI Ty/Lizard/strictMkDocs passed. Existing fit/apply ASTs unchanged; median capabilities local only, real saved workers and portable median reject. Independentreview no blocker; canonical NumPystrategy finding fixed by2red/green tests. Final552runtime/15asset wheel806c4b16d71eee6510d5de15bc73245c9156ae4acf6624e9a9bfe05061d7a683,528nodes collected, archive fixture preflight passed. Task211 upload/run question pending, no remote mutation. Resume prepared task211_native after approval; readgit history for localcommit. No push.
