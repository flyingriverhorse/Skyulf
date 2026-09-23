# SM-22c model lifecycle aliases - 2026-09-23

SM-22c stages an eligible, re-evaluated registered version as
`@challenger`. Explicit promotion requires that committed staging event,
moves the old `@champion` to `@previous_champion`, points `@champion`
to the candidate, and removes `@challenger` and its active marker.
Guarded rollback restores the prior champion and the previous rollback
pointer without restaging the rolled-back model. Concrete version numbers and
prepared/committed event tags retain the transition history.

## Local gate

The real SQLite MLflow 3.16.1 integration gate passed **40 tests** across
promotion, comparison, and registry. The promotion module covers eligible
staging, tampered/missing challenger, three-alias transition, previous-pointer
restoration, stale rollback, a pre-SM-22c receipt, permission denial, and
partial/uncertain writes. Ruff, formatting, and Ty passed for the changed
Python files and Core package. The final wheel's `promotion.py` bytes were
compared with the current source. Installing optional MLflow exposed two
pre-existing Ty diagnostics in the caller-owned-run tracking test; explicit
non-null run assertions made that test stronger, and its focused case passed.

## Unity Catalog gate

The existing isolated test resources in
`workspace.skyulf_sm24a_20260923` were reused: model
`skyulf_sm22b_promotion_r1` versions 1 and 2 and Delta admission table
`skyulf_sm22b_alias_admission_r1`. The new wheel and notebook were uploaded
only under the authenticated user's `sm22c_alias_r1` folder. The final
0.9.0 wheel SHA-256 is
`57755268204A3DBDC6D962B37DF52C37AC0E394E411E2ED829C1C9808FD11933`.

[Final-wheel owner run 269743074025024](https://dbc-45604623-c18b.cloud.databricks.com/?o=7474646244882000#job/693080435047326/run/269743074025024)
completed **SUCCESS** with a 900-second task timeout, zero retries,
serverless environment version 4, and MLflow 3.16.1. The candidate's
heldout MSE was `2.5849394142282116e-27`, versus `100.00000000000088`
for the reference. Event IDs were:

- Staging: `2e3a81c3ac8744b0805486aebd7f813e`
- Promotion: `d8685bdba6a74ccf812e59feeaaf1aca`
- Rollback: `c72b03a5374540b5968cc581796cbe72`

The notebook asserted the challenger alias at version 2, champion at version
2 and previous champion at version 1 after promotion, and the absence of the
challenger alias and active marker. After rollback it asserted champion at
version 1 and no stale previous-champion alias. The same run also rejected a
contending Delta admission holder. The final model alias state is restored to
the original champion version 1; the transition events and uploaded test
files remain for audit.

A first request, [run 912755889386772](https://dbc-45604623-c18b.cloud.databricks.com/?o=7474646244882000#job/656058797751923/run/912755889386772),
failed before compute launch because the deprecated `client: "1"` environment
channel is unsupported in this workspace. A corrected
[run 417916316136066](https://dbc-45604623-c18b.cloud.databricks.com/?o=7474646244882000#job/259434330041715/run/417916316136066)
passed the three-alias path, but its wheel preceded the final
challenger-marker cleanup. Only the final-wheel run is counted as the
complete SM-22c platform gate.

## Operational boundary

All staging, promotion, and rollback writers must share the same
non-expiring Delta admission row and controlled registry permissions.
Unity Catalog does not make the three alias changes atomic. A failure after
any alias mutation reports an unknown outcome and requires inspection of
aliases and the prepared event before retry. `@previous_champion` points
only to the immediate rollback target; it does not replace version history.
The caller still owns the labeled holdout snapshot and first-champion
initialization. No retraining service or Bundle was created here.
