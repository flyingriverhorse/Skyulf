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

Branch 093, base 6afe2ff8: task187 completed. Native run 211042384943295 SUCCESS; all six latest tasks passed. REST/SQL/class probabilities match the registered model exactly on 128 rows; no batch prediction table. SQL numeric-null transport fixed; original model runtime retained and corrected SQL compiled separately. Seven preprocessing appliers own their validation; general all-node/custom support is still a design, not delivered. Evidence: initiatives/spark_and_mlflow/187-raw-serving-demo.md, 187-native-readback.json and 187-sql-definition.json. Local gates/reviews recorded there; task187 claims released and completed exchanges removed. Earlier task185/186 work preserved. No commit or push.

Task188 complete: main-template PREPROCESSING.md explains YAML selection, Python recipes, custom steps and optional prediction writes. Demo output renamed training_batch_predictions with skipped/written status. 48 affected tests, executable guide fit/reload/custom probes, Ruff/full CI Ty/format/Lizard passed. Independent review finding about deployed_score_handoff synchronization corrected and closed. Evidence: initiatives/spark_and_mlflow/188-template-preprocessing-guide.md. Claims released. No cloud redeploy, commit or push; earlier staged work preserved.

Task189 complete: separate pre_split.yml/preprocessing.yml recipes with custom-only phase modules; duplicate custom phase files removed. Saved recipe/source replay and fresh-process custom filters verified. 486 distinct affected cases covered by passing batches and corrective reruns; final collection checked separately. Native run 1068962886438626 SUCCESS, 21 passed plus executable guide fit/reload/partition checks; no tables/models/endpoints created. Ruff/full CI Ty/format/Lizard/schema/diff gates passed. Independent review finding closed; stale pre-existing dataset-identity test corrected without production identity change. Evidence: initiatives/spark_and_mlflow/189-yaml-feature-recipes.md. All task189 claims released; prior staged work preserved; no commit/push.

Task190 complete: shipped opt-in company/activity Spark producer examples, commented features.yml configuration, groups usage guide and exact custom factory scope. Three real-CLI offline Bundle layouts, one real local Spark example and 69 custom/project checks passed (73 total); Ruff/full CI Ty/format/Lizard/diff passed. Spark shutdown printed Windows Access denied after passing tests with exit 0; local review helper unavailable due external lock permissions, parent review completed. Evidence: initiatives/spark_and_mlflow/190-feature-group-examples.md. Claims released; no production runtime changes, cloud deployment, commit or push.

Commit handoff (2026-10-08): user requested the pending task185-190 changes be committed on 093. Staged diff reviewed: 77 files, no temporary artifacts. Fresh full Ruff/format and every applicable pre-commit hook passed, including schema, YAML/JSON, Lizard and full CI Ty; frontend hooks had no matching files. Existing focused/native evidence above remains valid; no implementation changed during commit preparation. File claims released. The accompanying local DCO commit contains this delivery; push is not requested. Final hash/readback is recorded in the task190 report.

## Open messages
