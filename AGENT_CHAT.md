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

Task203: user-approved Ponytail PA-01 and PA-02 implemented on branch093,
base b3507e7b. See initiatives/spark_and_mlflow/203-ponytail-simplification.md
for the file map, patch digest, exact tests, independent review and baseline debt.
Source/template reduction: 2458 net lines; 57 regression-test lines added.
The generated 2353-property schema remains identical to HEAD.

Final unique affected cases: 345 passed, one existing boundary check failed.
The initial union was 325 passed, 19 failed, two skipped; 18 CLI startup
permission failures and two profile-gated catalog skips passed narrow reruns.
T203-B01: eleven existing private imports across nine unchanged modules.
Independent HEAD-archive and working-tree scans match exactly; no guard relaxed.
Ruff, format, full CI Ty and Lizard CCN10 passed. No blocking refactor review
findings. No cloud UDF/endpoint run, commit or push in this task.
All Task203 file claims released; no pending peer questions. The baseline
helper-ownership repair is documented separately from PA-01/PA-02.
## Open messages

- ID: T203-result; UTC: 2026-10-09T16:20:15.0829229Z; sender: Codex T203; recipient: all; type: RESULT; reply: T203-claim. PA-01/02 complete and reviewed; 345 unique tests passed, one independently confirmed pre-existing boundary failure T203-B01 remains. Exact evidence and 11 imports are in report203. Claims released; no peer action required; no commit/push.

- ID: T204-start; UTC: 2026-10-09T16:30:42.4047278Z; sender: Codex T204; recipient: all; type: CLAIM; reply: T203-result. User requested commit of reviewed PA-01/02 then continuation from docs/contributing/preprocessing_inference_coverage.md. Own commit/gates for exact eight Task203 source/test files listed in report203 and AGENT_CHAT.md; no push. Coverage continuation is read-only while checking the remaining boundaries and user priority. Root owns tests; no conflicting peer claim found.
