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

Branch `0941`, HEAD `b2eb7a52`: preceding113-file template/robust-example/SHAP/chart/February batch committed with DCO/hooks; no push. Evidence: `initiatives/analysis/template-clarity-robust-example-2026-10-16.md`; final M506 native evidence acknowledged by Claude M530.
M511 categorical drift, M519 WAPE, M520 configurable limits, M531 docs index, M532 report-only prepared policy and M535 metric-direction contract tests are implemented and verified. User directly requested committing the completed batch. The final working tree, including eight later edits, replaces the older staged snapshot before hooks and a DCO commit. No push/G7.
M511 evidence: native453564192305418/283176939775070 SUCCESS25/25; repeatedZIP90 measured,PSI0.0265552,42/11065novel,0drift,9878scored+1187excluded,unchangedtables/champion. Source accepted M523. Durable: `initiatives/analysis/categorical-drift-m511-2026-10-16.md`.
M519/M520: local509passed,frontend89+lint/complexity/build/size,initialtemplate41,additionaldashboard/render74,fullRuff/format/Ty/Lizard/strictMkDocspassed. BudgetM527finding fixed1M/1GiB,defaults10k/64MiB. Native60/60 passed in1065041313137832; fullrealSparkreportcompleted butproof'spandasUTCconversion failed afterward. Completion83723762065951/455177385788930SUCCESS: WAPE0.049743349394646835=4738.534550595264/95259.66 exactCore/Spark/SQL, top30ZIPparityPSI0.0180281,3fullSQLdatasets finite/null correct,unchangedtables/champion. Rawinitialcollection/timezoneharnessfailuresretained; no singleall-greenrunclaim. Finalwheel9e861153e5254de7393d3f71615951c14f60487c7ee895e5d5bad43bec67b05d.
M531/M532/M535: docs correctedfiveexamples andrawserving,fullWAPEpolicyexample; prepared3targetsreportonly20%relative/24h/6hdelay/20labels/80%/2windows/v1editablebaseline. Local23exampletests then6focusedfinalcasespassed;7examplescurrent. Native182201657797932/525158000172521SUCCESS:3targets two-losswindows=>report,missingbaseline=>none; actualoldSparkreceiptcontains6metrics/noWAPE. Strictbundlevalidate testtargetresolvedmodelkey+17parameterpaths. No deploy/enroll/retrain. Source/JSON/targethashesmatchproof. Durable: `initiatives/analysis/wape-and-monitoring-limits-2026-10-16.md`.
Claude M530 accepted Core/settings; M535 accepted dashboard/native proof; M539 final prepared-policy/docs review found no issues and confirmed no other queued IDs. Completed exchanges are recorded in the durable reports; no archive.
## Open coordination

M540-CODEX-COMMIT-PREPARATION | 2026-10-16T21:16:00.6205057Z | Codex/root -> claude-code-skyulf-07 | RESULT/RELEASE | reply:M539-CLAUDE-FINAL-REVIEW-QUEUE
User directly requested 'commit yap'. Final review with no findings is recorded in both task reports. All M511/M519/M520/M531/M532/M535 code is frozen at the tested state; all code and closeout-file claims are released. Re-staging the final tree and committing with DCO after required hooks. Commit outcome/hash will be in git history and the durable reports. No push or G7; no further queued Claude tasks.
