# SM-32 live Bundle operator validation

Date: 2026-09-25. Status: PASSED for the personal serverless SM-32 scope.
Profile: `skyulf`. Existing schema: `workspace.skyulf_lifecycle_test`.
Plan: [live rehearsal](50-sm32-live-rehearsal-plan.md).

Historical snapshot: the practice candidate below was subsequently approved by
the user. Polars champion is now v5; see [report 52](52-sm32-operator-output-and-evidence.md)
for the user approval and later report/form acceptance.

## Scope

The existing `skyulf_lifecycle` Bundle and two persistent jobs are reused:
train `155738051514173`, score `684955889505992`. Both jobs are queued with
one active run; no schedule is active. Existing source, models and predictions
are retained. Fresh `sm32_model_polars` and `sm32_model_pandas` support testing
first-champion behavior without deleting earlier evidence; each has a separate
prediction output. No new schema, persistent job or admission table.

The source is the existing deterministic lifecycle fixture, not a business
accuracy benchmark. Training reads Delta version 0: 120 fit rows and 40
holdout rows. Initial scoring reads the current 170-row source. A deliberately
weak intercept-free model followed by an intercept-enabled model gives a
predictable improvement for testing promotion. The threshold `500.0` is a
rehearsal gate, not a recommended production threshold.

## Live defect and repair

First deployment exposed an unsupported timeout on the rehearsal's If/else
task. That timeout was added by the test harness, not the product template;
the harness now bounds compute tasks and the overall job only.

Initial manual training passed. Approval initialized champion successfully,
but its child score failed: Databricks automatically forwarded the parent's
`lifecycle_action=approve` and evidence into the child Run Job task. The fixed
score adapter correctly refused an explicit lifecycle dispatch, exposing the
missing notebook-boundary filtering in local tests.

The repair filters only inherited lifecycle/evidence fields at the fixed score
notebook boundary. Role/action overrides remain rejected, and direct adapter
API validation is unchanged. Four inherited-parameter tests failed before the
fix and passed afterward. Two further notebook tests preserve override guards.
Review found no blocker.

The corrected wheel was uploaded under a fresh `sm32/r2` path. SHA-256:
`420833172d6392964fd695aa64d277637648d195f11cd836fa894f6d295085f5`.
Packaged adapter source was compared with the working source. The completed
approval was replayed without training/registration, returned the same receipt,
and its child score successfully wrote 170 predictions. The original failed
run remains available as evidence; no alias was manually moved to repair it.

## Verification

- Local affected regression: **194 passed**, five expected legacy deprecation
  warnings, 135.26 seconds (`rehearsals/sm32_live/regression.log`).
- Scoped Ruff and full repository ty passed.
- Strict documentation build passed; the walkthrough's two Mermaid diagrams
  were parsed with the real Mermaid parser.
- Deployed configuration inspection confirmed exactly two persistent jobs,
  correct conditional graph, serialized queues and no active schedule.

All 13 expected-success lifecycle job runs completed, together with five
successful conditional child score runs. The initial pushdown failure remains
in the run table below. This is not company-environment production acceptance.

## Recorded lifecycle job runs

| Scenario | Run | Outcome | Score requested |
| --- | --- | --- | --- |
| `polars_manual_v1` | [503746906366109](https://dbc-45604623-c18b.cloud.databricks.com/?o=7474646244882000#job/155738051514173/run/503746906366109) | SUCCESS | False |
| `polars_approve_v1` | [906338302689820](https://dbc-45604623-c18b.cloud.databricks.com/?o=7474646244882000#job/155738051514173/run/906338302689820) | FAILED | True |
| `polars_approve_v1_recovery` | [755489153097671](https://dbc-45604623-c18b.cloud.databricks.com/?o=7474646244882000#job/155738051514173/run/755489153097671) | SUCCESS | True |
| `polars_manual_v2` | [108692285496124](https://dbc-45604623-c18b.cloud.databricks.com/?o=7474646244882000#job/155738051514173/run/108692285496124) | SUCCESS | False |
| `polars_approve_v2` | [728667696836302](https://dbc-45604623-c18b.cloud.databricks.com/?o=7474646244882000#job/155738051514173/run/728667696836302) | SUCCESS | True |
| `polars_manual_v3` | [238888283154553](https://dbc-45604623-c18b.cloud.databricks.com/?o=7474646244882000#job/155738051514173/run/238888283154553) | SUCCESS | False |
| `polars_reject_v3` | [305664316009276](https://dbc-45604623-c18b.cloud.databricks.com/?o=7474646244882000#job/155738051514173/run/305664316009276) | SUCCESS | False |
| `polars_manual_v4` | [450726548945680](https://dbc-45604623-c18b.cloud.databricks.com/?o=7474646244882000#job/155738051514173/run/450726548945680) | SUCCESS | False |
| `polars_rollback_v2` | [317513046274699](https://dbc-45604623-c18b.cloud.databricks.com/?o=7474646244882000#job/155738051514173/run/317513046274699) | SUCCESS | True |
| `polars_rollback_retry` | [167619432389655](https://dbc-45604623-c18b.cloud.databricks.com/?o=7474646244882000#job/155738051514173/run/167619432389655) | SUCCESS | True |
| `pandas_auto_v1` | [1066520934685111](https://dbc-45604623-c18b.cloud.databricks.com/?o=7474646244882000#job/155738051514173/run/1066520934685111) | SUCCESS | False |
| `pandas_auto_v2` | [384007232635977](https://dbc-45604623-c18b.cloud.databricks.com/?o=7474646244882000#job/155738051514173/run/384007232635977) | SUCCESS | True |
| `pandas_auto_v3` | [107464962193774](https://dbc-45604623-c18b.cloud.databricks.com/?o=7474646244882000#job/155738051514173/run/107464962193774) | SUCCESS | False |
| `polars_practice_v5` | [888140358241905](https://dbc-45604623-c18b.cloud.databricks.com/?o=7474646244882000#job/155738051514173/run/888140358241905) | SUCCESS | False |

The original failed approval had already committed champion. Its `score_requested=true`
means the child was requested, not that scoring succeeded. Recovery reused the
committed approval; the later successful handoff was captured separately.

## Final registry and prediction audit

[Audit run 939680081628471](https://dbc-45604623-c18b.cloud.databricks.com/?o=7474646244882000#job/360955161400049/run/939680081628471)
passed. This was a one-time submitted run, not a third persistent Bundle job.
It checked all eight model versions and their six held-out regression metrics.
No approval/rejection/rollback/retry registered an extra model version.

| Engine | Versions | Champion | Challenger | History retained | Predictions |
| --- | --- | --- | --- | --- | --- |
| Polars | 5 | 1 | 5 | previous_challenger=4; v3 explicit rejection recorded | 180 unique rows, model v1 |
| pandas | 3 | 2 | 3 | previous_champion=1 | 180 unique rows, pinned model v1 |

The audit appended ten new keys to the existing source (170 -> 180 rows).
Each engine's existing prediction output received exactly ten new rows.
Replaying each scorer returned noop=true and retained Delta version 2.
Insert/replay verification called the existing Core run_action score service
inside the one-time notebook. Conditional dispatch, parameter inheritance and
pinned scoring had already been verified through the actual Bundle jobs.

Live combinations: Polars manual approval with champion scoring; pandas
automatic promotion with pinned scoring. Handoff enabled and disabled were
both exercised. The complete four-pair policy matrix remains covered by local
pandas/Polars tests and actual CLI template generation; this report does not
claim every matrix combination was separately deployed live.

## Retained practice configuration

The two Bundle jobs now use Polars, manual_approval, champion scoring,
after_alias_change handoff and incremental_append. No schedule is active.
Model: workspace.skyulf_lifecycle_test.sm32_model_polars.
Source: workspace.skyulf_lifecycle_test.source.
Output: workspace.skyulf_lifecycle_test.sm32_predictions_polars.

The eligible v5 candidate is intentionally not approved. Its training run is
[888140358241905](https://dbc-45604623-c18b.cloud.databricks.com/?o=7474646244882000#job/155738051514173/run/888140358241905).
These values are a snapshot; after another nomination or champion change,
use the current candidate output rather than replaying stale practice inputs.

1. Open that run's train task output and review the comparison/MLflow metrics.
2. On the same train job choose Run now with different parameters.
3. Copy the following values; leave rejection_reason and promotion_receipt_json empty.

```json
{
  "lifecycle_action": "approve",
  "candidate_version": "5",
  "comparison_sha256": "51358cdc74ec62aa14f689cfa2b06532e83f338ecf6a0ff5c9605dc26d2c3c62",
  "expected_champion_version": "1"
}
```

4. Expect champion v5 and a successful child score run. No new source rows means
   score can be a no-op: existing append-mode v1 predictions remain unchanged.
5. To reverse this new promotion, copy next_actions.rollback from that approval
   result into another lifecycle run. Do not use the old v2 receipt for v5.
6. To practice rejection instead, use the pending candidate's reject parameters
   and add a reason before approving it. A rejected version cannot be reopened;
   another training run creates another candidate.

The generic English walkthrough, including two Mermaid diagrams, is in
[docs/user_guide/databricks_bundle_walkthrough.md](../../docs/user_guide/databricks_bundle_walkthrough.md).
The final queue marks SM-32 DONE and SM-33 READY. Company identities/permissions,
policy compute, broader config migration, production operations and combined
acceptance remain in their own queued tasks. Broad Spark/serving work remains later.

## Evidence locations

Exact requests, run states, notebook results and child score outputs are under
rehearsals/sm32_live/. sequence-r2.log records the successful continuation;
sequence.log retains the first failed handoff. audit-result.json records final
UC aliases, per-version metrics, row counts and Delta publication receipts.
The corrected wheel is in wheel_r2/. Plan/report files are ignored by the
repository's broad initiative rule and need explicit inclusion at commit time.
