# SM-16 platform validation

Status: **DONE — selected serverless Connect workflow validated on 2026-09-22**.
Baseline: `d43e74ca`, branch `090`, target release `0.9.0`.

The [machine-readable evidence](reports/2026-09-22-sm16-live-evidence.json) was
generated after asserting the actual live results, matching model/table/trial
identities and the r3 wheel checksum. It combines five successful reports and
the explicitly retained contender assertion failure. Individual stage reports
keep `platform_gate_complete: false`; the aggregate records completion of G4.
This certifies the selected regression workflow, not every FE node, cloud
runtime, classification path or production workload. SM-15L remains separate.

## Local preparation evidence — 2026-09-22

- Added `load_registered_bundle` with local registry tests: concrete version
  loading after alias movement, separate stores, preservation of unrelated
  MLflow state, declared artifact paths and rejection of missing/mismatched
  metadata, missing directories and escaping paths.
- Combined local Linux/WSL gate: **29 passed**, no skips, 107.39s (10 existing
  registry tests, 15 loader tests and 4 smoke tests). MLflow **3.16.1**, Python
  **3.12.3**, PySpark **4.0.3**, Delta fixture **4.0.0**, Java **17.0.20.1**.
  The worker probe used Polars training, then loaded the registered model and
  exercised both Spark modes. Local gold tests cover pandas and Polars training.
- Base bundle/integration regression: **77 passed, 28 skipped**; optional
  MLflow/Delta tests are absent in that environment. A known Windows physical
  core detection warning was emitted. Scoped Ruff/format, repository Ty,
  `--help` without optional dependencies, and strict MkDocs passed.
- Independent code review found no blocker. The integration probe's experiment
  artifact root was moved under its temporary directory before the final gate.
- Wheel built locally: `.cache/sm16-dist/skyulf_core-0.9.0-py3-none-any.whl`.
  SHA-256: `ede03a571bf3de480e1acca8ddd9c5179efd832906c410f4a0a98f50709b4779`.
  This was the first uploaded wheel; the corrected r2/r3 hashes and successful
  driver/worker checks are recorded below. Rebuild after code changes.

Exact local commands:

```powershell
wsl -d Ubuntu -- bash .cache/sm15-linux-run.sh -m pytest skyulf-core/tests/integrations/test_mlflow_registry.py skyulf-core/tests/integrations/test_mlflow_registry_bundle.py skyulf-core/tests/integrations/test_platform_smoke.py -q -p no:cacheprovider -o addopts= --basetemp /tmp/sm16-registry-reviewed --tb=short
.venv/Scripts/python.exe -m pytest skyulf-core/tests/integrations skyulf-core/tests/spark/test_inference_bundle.py -q -o addopts= --basetemp .cache/sm16-base-final --tb=short
uv build --wheel --out-dir .cache/sm16-dist skyulf-core
.venv/Scripts/python.exe -m mkdocs build --strict --site-dir .cache/sm16-docs
```

For another Linux test environment, install `requirements-delta.txt` and
`mlflow==3.16.1` with `uv pip`, then run the same three test modules with
`SKYULF_REQUIRE_DELTA=1`. The local wrapper selects the prepared Java runtime
and cached official Delta jars; those are test-environment details, not a
Databricks runtime dependency profile.

## Confirmed local tooling

- Databricks agent-skills plugin `0.2.20` is installed. Its `databricks-core`
  and `databricks-jobs` instructions were inspected.
- Databricks CLI **1.17.0** was verified with `--version` from the existing
  WinGet installation. It is absent from this agent shell's PATH, so an
  explicit executable path is needed until that shell is refreshed.
- The repository root has no `databricks.yml`. The user confirmed it is not
  needed here. No bundle, deployment or authentication was created.
- On 2026-09-22 the user explicitly authorized using the supplied workspace
  and existing `skyulf` or default CLI profile. `auth profiles` returned one
  matching profile: `skyulf`, default, valid, OAuth CLI authentication.
  `current-user me` succeeded without another login.
- Read-only discovery found catalogs `workspace`, `samples`, `system`; the
  `workspace` catalog has `default` and `information_schema` schemas. No classic
  clusters were listed. One stopped serverless SQL warehouse is available.
- A one-time serverless registry/Spark probe request was prepared with a
  900-second timeout and no retries. Proposed resources are the isolated
  `workspace.skyulf_sm16_20260922` schema and a same-named folder under the
  authenticated user's workspace directory. It uses the locally built wheel
  and explicitly requests MLflow 3.16.1 without installing OSS PySpark.
- Automatic approval review initially rejected schema creation. The user then
  explicitly approved these resources, uploads and the bounded serverless job.
  Schema/folder creation and uploads succeeded. Parent run `245612415275039`
  and task run `45737629137632` were submitted with a 900-second timeout.
  [Live run](https://dbc-45604623-c18b.cloud.databricks.com/?o=7474646244882000#job/94637315302832/run/245612415275039).

## Stage 1: registry-to-Spark probe

### First live result and correction — 2026-09-22

- Parent run `245612415275039`, task `45737629137632`: **FAILED** during
  Spark inference identifier validation. The traceback reached `predict_spark`
  after model registration, pinned bundle download and both local gold checks.
- Retained test model:
  `workspace.skyulf_sm16_20260922.probe_b6b83b505a4145c192cc6be6c1ab7569`.
- Imported Skyulf came from the job's ephemeral Python 3.12 `site-packages`.
  Clean worker prediction did not run, so worker wheel delivery is still open.
- Live failure: `CONFIG_NOT_AVAILABLE.WITHOUT_SUGGESTION` when reading
  `spark.sql.caseSensitive`. Serverless restricts configuration access.
- Added one shared identifier-rule helper at all three read sites. Only the
  structured unavailable-configuration condition falls back to conservative
  case-insensitive collision checks; transport and permission failures propagate.
- Reproduced the failure before the fix. The real Spark regression gate passed
  **75 tests** in 76.42s, including pandas/Polars training, both inference modes,
  hidden configuration and ambiguous-case rejection. Independent review found
  no blocker; scoped Ruff and Ty passed.
- Corrected wheel: `.cache/sm16-dist-r2/skyulf_core-0.9.0-py3-none-any.whl`.
  SHA-256: `b10a7172cd268deec9b06168987a4d2ee4a19259417fc230d9a0bb3b5ea1b211`.
  Automatic approval review initially rejected the additional upload/run as
  exceeding the single-run approval. The user explicitly approved the retry;
  `r2` uploads succeeded and parent run `447606109645160` was submitted with
  the same 900-second timeout. Task run `987386852945959` finished **SUCCESS**.

### Successful serverless rerun

[Run `447606109645160`](https://dbc-45604623-c18b.cloud.databricks.com/?o=7474646244882000#job/541401697073405/run/447606109645160)
completed the Polars-trained bundle round-trip through Unity Catalog. Both
`native_features` and `python_pipeline` produced the keyed gold predictions
`[-2, 3, 5]`. The report intentionally keeps `platform_gate_complete: false`.

| Evidence | Value |
| --- | --- |
| Model | `workspace.skyulf_sm16_20260922.probe_3e8e88a3d2f2435a9c2b7b280a184bc2`, version `1` |
| MLflow run | `81e30610dba14c52ae0e7e478b5abf50` |
| Bundle digest | `66890475ca608271a6115ed2505890740bd6669fd9301711f9032390b4ea8084` |
| Python / Spark | `3.12.3` / `4.2.0` |
| MLflow / Arrow | `3.16.1` / `25.0.1` |
| pandas / Polars / scikit-learn | `2.2.3` / `1.44.2` / `1.6.1` |
| Runtime environment | `client.4.10`, `pyspark.sql.connect.session.SparkSession` |
| Skyulf driver installation | `0.9.0`, non-editable ephemeral `site-packages` |

The exact uploaded wheel is the `r2` artifact identified above. Both distributed
prediction actions succeeded; per-worker package/checksum provenance and scale
measurements were not collected by this three-row probe. Raw task output is
retained locally at `.cache/sm16-live-registry-output-r2.json` and in the run UI.
Test models and tracking runs remain in the approved namespace for inspection.

### Restricted identity preparation

The user requested creation of a separate test identity. Created
`skyulf-sm16-restricted-20260922`, application ID
`9559edaa-81e3-434b-a679-35703510c134` (workspace principal `75734274047655`).
It has workspace access, inherited `USE_CATALOG`, explicit `USE_SCHEMA` on the
test schema, `EXECUTE` on only the successful rerun's model and `CAN_READ` on
the isolated test folder. It has no admin role, data-modification grant or
created token/secret. The requesting account retains its existing manager role
and gained the user role on this test principal so it can submit `run_as` jobs.

Run `2921374308246`, task `479913572085165`, completed **SUCCESS** with this
`run_as` identity. Allowed model loading passed and the first test model returned
`RegistryAccessError`. No token/secret was created and no data table was written.
Two worker probes imported non-editable `skyulf-core 0.9.0` from ephemeral
`site-packages`; package-content digests are checked in the next stage.

| Rows / requested partitions | Mode | Action time | Maximum absolute error |
| --- | --- | --- | --- |
| 10,000 / 2 | `native_features` | 9.91s | 8.89e-16 |
| 10,000 / 2 | `python_pipeline` | 2.07s | 8.89e-16 |
| 50,000 / 8 | `native_features` | 20.37s | 8.89e-16 |
| 50,000 / 8 | `python_pipeline` | 2.33s | 8.89e-16 |

Each aggregate checked row count, distinct key count and the independent `2*x`
oracle. These sequential synthetic runs include different validation work and
warm-up effects; they are not a controlled engine benchmark or capacity claim.
Notebook/Connect client Python process-lifetime peak RSS was 2,276,596 KiB
for all four observations. This is not Spark JVM or total cluster memory.
The separate worker import probes recorded 536,040 and 534,688 KiB peak RSS;
those are not peak measurements of the inference task itself. Raw evidence and
the prepared notebook/request are under `.cache/sm16-live-permissions*`.

### Monthly Delta and alias stage

The reusable `examples/databricks_delta_smoke.py` passed **5 real Delta tests**
in 62.76s, plus Ruff/format/Ty and independent review. It retains three unique
test-owned tables and exercises real public batch publication, replay,
stale-version rejection, empty protection and explicit empty replacement.
Parent run `810044894558535`, task `499533558954860`, used the same `r2` wheel
and **FAILED** at the admission provider's `REFRESH TABLE` call with structured
`NOT_SUPPORTED_WITH_SERVERLESS`. It created the seeded test tables
`workspace.skyulf_sm16_20260922.skyulf_delta_smoke_837cbd4e7c274c0f905f84adce1d8c89_{source,target,control}`;
no batch publication began. These resources are retained.

Before that failure, worker package assertions and the alias test completed.
The worker check compared all 225 installed Skyulf package files with the
wheel's package-content digest
`f887a4251f03fddb478b0d263afcd8328d9f7f81398e7da857c00d694a8cf97b`.
The test created alias `sm16_validation`, pinned version 1, registered a
numerically different `3*x` model version and moved the alias. The pinned bundle
still predicted 4 for x=2 while the alias-selected bundle predicted 6. The failure
trace reached the subsequent Delta stage; the notebook's final JSON was not
produced. The successful r3 retry below repeated the content and pinned-load
assertions without moving the alias again.

The admission correction preserves refresh on classic Spark and catches only
the structured serverless restriction. It still reads identity and ownership;
exclusivity comes from the conditional Delta UPDATE and a fresh UUID token.
Other failures propagate. The official serverless limitations also prohibit
DataFrame `persist`/`unpersist`. The batch runner now catches only the same
structured serverless restriction from `persist`, preserves the distributed
frame, and calls `unpersist` only after successful caching. Other cache errors
propagate. Validation/publication can recompute uncached predictions.

The combined real Delta gate passed **48 tests** in 173.18s: 24 batch tests,
19 admission tests and 5 probe tests. It covers both inference modes without
caching, unrelated-error propagation, classic cache cleanup, fresh ownership
reads with refresh rejected, and actual monthly transactions. Independent review
found no blocker. The base integration lane passed 41 with 53 optional skips.

Retry parent run `783094949884769` completed **SUCCESS** using
`.cache/sm16-dist-r3/skyulf_core-0.9.0-py3-none-any.whl`:

- Wheel SHA-256: `e2ce61eb3f11f25dd334934b9b45832cc3901ceba2aa04a5a28bb091232a29c5`.
- Package-content digest (225 files): `2d5da10303be00bd58c2cb5f6f2e468ad54bfd683181c5b511f09fa2aedd78f0`.

Task run `26007003170545` returned all eight monthly checks as true: publication,
replay, stale-version rejection, empty rejection, explicit empty replacement,
empty replay, ownership release and preservation of the prior period. Initial
publication and replay used source snapshot 0 and returned commit 1; empty
replacement and its replay used source snapshot 1 and returned commit 2.
The target ended at version 2 with only the prior-period row.

Retained tables share prefix
`workspace.skyulf_sm16_20260922.skyulf_delta_smoke_4bbeac1816a04bbe854c85903256c41d_`:

| Suffix | Delta table ID | Version after monthly probe | Rows |
| --- | --- | --- | --- |
| `source` | `36b4f01d-3378-48e9-a93f-c593b634acef` | 1 | 1 |
| `target` | `f16a3d46-8a00-4c55-9ac8-8f7690b17647` | 2 | 1 |
| `control` | `fe57f94d-a91a-44c1-a294-d0caaee19f7f` | 10 | 1 |

The report confirms reloading pinned version 1 after alias `sm16_validation`
moved to version 2, with distinct digests and the expected 4 versus 6 predictions.
Two worker partitions each checked all 225 installed package files against the
r3 package-content digest. The full result is saved locally in
`.cache/sm16-live-delta-output-r3.json`.

Independent contention jobs used the same r3 wheel: winner
`973709879452231` and restricted-principal contender `404334394214907`.
They share trial `3d74ef662d5047dfb90a13ec42dc5eaa` and the retained tables above.
The restricted principal has SELECT on source/target and SELECT/MODIFY on the
control table. The winner completed **SUCCESS**, committing version 3 and
releasing ownership. The contender passed the held-owner conflict, unchanged
owner/version-2 assertions and acknowledgement before the winner could commit.
It then acquired admission and reached an actual target write denial.

The contender run is **FAILED**, because its test accepted only condition names
containing `PERMISSION` or `PRIVILEGE`. Databricks returned `UNAUTHORIZED_ACCESS`;
the chained error explicitly states `PERMISSION_DENIED` and missing `MODIFY` on
the exact target. This is a probe assertion mismatch, not a successful job or a
product-code defect. Its final preservation assertions were not executed.
The restricted-only follow-up run `977447944071613`, task `396353814917782`,
completed **SUCCESS**. It verified the winner's rows/metadata, target version 3
and released ownership before and after another denied write, accepting only
the observed exact condition and message. Data and version remained unchanged.
The paired-run contention assertions and this final permission probe jointly
close admission and write-denial gates; the initial failed run remains failed.

Live records: [winner](https://dbc-45604623-c18b.cloud.databricks.com/?o=7474646244882000#job/970586457713217/run/973709879452231),
[contender with assertion mismatch](https://dbc-45604623-c18b.cloud.databricks.com/?o=7474646244882000#job/602255242801236/run/404334394214907),
[final write-denial probe](https://dbc-45604623-c18b.cloud.databricks.com/?o=7474646244882000#job/555471604714601/run/977447944071613).
The signal table is
`workspace.skyulf_sm16_20260922.skyulf_contention_3d74ef662d5047dfb90a13ec42dc5eaa`.
All test resources and the restricted identity are retained for inspection.

References: [Serverless restrictions](https://docs.databricks.com/aws/en/compute/serverless/limitations),
[Databricks identifier rules](https://docs.databricks.com/aws/en/sql/language-manual/sql-ref-identifiers).

### Shared Delta admission preparation

`DeltaTableAdmission` now provides non-expiring conditional ownership through
an operator-provisioned singleton Delta table. Independent real Spark sessions
share that authority. The final focused gate passed **16 tests** in 59.07s,
including a forced race after both readers see a free row and acknowledgement
loss after acquisition commits. The failed caller never enters publication;
its retained claim blocks others until deliberate operator recovery.

The final combined gate passed **18 tests** in 87.00s: the 16 provider tests,
the existing OS-process lock and public `run_batch` publication/replay through
shared Delta admission. Predictions, prior-period rows, the original receipt
and owner release after both calls were verified. Base regression passed
**80 tests**, with **48 optional-runtime skips** and one known Windows physical
core detection warning. Ruff/format/Ty, strict MkDocs and independent review
passed. The later live monthly run verified serverless/UC admission;
independent contention and permission runs are recorded above. See the
[batch guide](../../docs/user_guide/databricks_batch.md) for provisioning and
the explicit cooperative-writer and manual-recovery boundaries.

### Probe contract

Script: [databricks_batch_smoke.py](../../skyulf-core/examples/databricks_batch_smoke.py).

1. Fit a tiny pandas or Polars pipeline: mean SimpleImputer, StandardScaler,
   linear regression with a known `y = 2*x` numerical oracle.
2. Log and register a uniquely suffixed test model through explicit MLflow
   stores. Retain the created model and run for inspection; no alias is moved.
3. Resolve the created version and download its declared Skyulf bundle.
   Compare the package and bundle digests with the pinned model identity.
4. Compare local inference and both Spark inference modes by row key. The gold
   predictions are `[-2, 3, 5]`; the middle result uses the training mean `1.5`.
5. Produce a small report with runtime versions, installed module provenance,
   model/run identity and individual check results. It explicitly records
   `platform_gate_complete: false` and the gates still outstanding.

Only the fixed three-row probe is collected. This is not a scale/RSS test or a
production batch collector. A successful probe is not proof of Delta delivery,
distributed admission, UC permission handling or clean wheel deployment.

After choosing a compatible runtime and installing the built wheel in the job,
the script can be run as a Python task or from an explicitly prepared notebook:

```bash
python databricks_batch_smoke.py \
  --model-prefix TEST_CATALOG.TEST_SCHEMA.skyulf_sm16 \
  --experiment-name /Workspace/APPROVED_TEST_EXPERIMENT \
  --tracking-uri databricks \
  --registry-uri databricks-uc \
  --training-engine polars \
  --output /tmp/skyulf-sm16-evidence.json
```

These are placeholders, not approved destinations. The caller must select the
Spark environment; the script never creates a local Spark master inside a
Databricks job. Both `pandas` and `polars` are training options. Outputs use the
same fitted model in Spark inference. The script prints each uniquely created
model name before registration, so failed probes can still be inspected and
their test-owned resources removed explicitly. Preserve the evidence before
cleaning up; never delete other models, experiments, tables or aliases.

`load_registered_bundle` consumes trusted MLflow packages only: payloads may
contain pickle. Digests detect mismatched artifacts; they do not authenticate
an untrusted producer. Downloads use explicit stores and the pinned version,
including when a previously resolved alias has moved.

## Completed platform gate and evidence boundaries

| Gate | Required evidence | Current state |
| --- | --- | --- |
| Selected environment | Profile, workspace, compute ID/type, DBR/Python/Spark/MLflow/Arrow | `skyulf`; serverless Connect; successful runtime recorded above |
| Package delivery | Exact wheel/checksum, installed distribution on clean driver/workers | Non-editable driver and workers passed; 225-file content checks passed in r3 |
| Real UC round-trip | Concrete model version, alias pinning and trusted bundle download | Register/download passed; pinned version 1 survives alias movement to version 2 |
| Spark parity | Gold data, both FE modes, key-based predictions, job run ID/URL | Passed in live corrected run `447606109645160` |
| Monthly publication | Real test Delta tables, snapshots, commits, replay, other periods preserved | Passed in live run `783094949884769`; source versions 0/1, target commits 1/2 |
| Distributed admission | Shared authority, ownership through commit, competing-job result | Winner committed version 3 only after contender verified held-owner rejection; released afterward |
| UC permissions | Actual denied and allowed access in the approved namespace | Model allow/deny passed; target MODIFY denied with unchanged version/data and released admission in final probe |
| Scale | Synthetic distributed dataset, runtime and scoped memory observations | 10k/50k parity/timing + client Python lifetime peak and separate worker import RSS recorded; not inference-worker or cluster peak |

Do not use `LocalTableLock` on a distributed Databricks driver. Limiting one
job's concurrent runs does not coordinate other jobs or external writers.
An expiring lease without sink-enforced fencing is insufficient. The platform
admission implementation now uses a shared non-expiring Delta claim. Its local
concurrency evidence does not substitute for validation on the selected platform.

## DAB and templates

A job can be submitted through the CLI/SDK without a bundle. A bundle can be
written directly in YAML or use Python resource definitions; a reusable project
template is optional. Even Python-defined bundle resources use a root
`databricks.yml` entry point. A model artifact does not automatically generate
the code, compute selection, permissions and job definition needed for a bundle.

SM-16 used script/wheel uploads and reviewed one-time job requests against the
user-selected workspace. Reusable templates remain SM-20. A future generated
project can contain its own `databricks.yml`; the Skyulf library repository does
not need one for these tests.

References: [Python bundle resources](https://docs.databricks.com/aws/en/dev-tools/bundles/python),
[CLI bundle commands](https://docs.databricks.com/aws/en/dev-tools/cli/bundle-commands).

## Local-engine Delta follow-up

The SM-15 runner currently requires Spark. Delta itself does not require Spark:
pandas/Polars can write through Arrow and `delta-rs`. This is a separate sink
implementation, not an implicit conversion of a Spark dataset to local memory.

Track **SM-15L** before template support advertises local-engine Delta output:

- Separate inference engine (`pandas`, `polars`, `spark`) from the output sink.
- Define explicit local source snapshot provenance and memory limits.
- Carry over period replacement, schema/UTC rules, empty protection, durable
  receipts, retry identity and single-writer admission with real transaction tests.
- Establish supported Delta protocol features and storage credentials. Writing
  a filesystem/object-store Delta path does not by itself prove that a Unity
  Catalog managed table can be written through that path.
- Validate UC access separately; fail unsupported combinations explicitly.

As of 2026-09-22, Databricks documents external managed-table writes through
Unity REST as preview functionality requiring catalog commits and a supported
client. Its supported-client list does not establish delta-rs support. Therefore
a local filesystem delta-rs test cannot close the UC writer requirement. Evaluate
an explicit pandas/Polars inference plus Spark catalog-I/O path separately from
the optional standalone delta-rs sink; keep local data collection bounded and
visible in that API. This is a follow-up design constraint, not implemented support.

References: [delta-rs transactions](https://delta-io.github.io/delta-rs/api/transaction/),
[Databricks external Delta access](https://docs.databricks.com/aws/en/external-access/unity-rest).
