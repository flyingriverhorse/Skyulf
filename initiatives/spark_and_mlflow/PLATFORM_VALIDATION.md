# SM-16 platform validation

Status: **ACTIVE — live registry/Spark parity passed; remaining platform gates open**.
Baseline: `d43e74ca`, branch `090`, target release `0.9.0`.

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
  The wheel was uploaded to the approved workspace test folder. Installation and
  worker execution remain pending the live job. Rebuild after code changes.

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

Run `2921374308246` was submitted with this `run_as` identity and a 900-second
timeout. It checks allowed model loading, denied access to the first test model,
worker package provenance and 10k/50k synthetic prediction aggregates. It does
not write tables. Its result is pending; submitting a job is not permission-test
evidence. Prepared notebook/request are under `.cache/sm16-live-permissions*`.

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
passed. Live serverless/UC admission,
competing jobs and permission enforcement remain pending. See the
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

## Full platform gate still required

| Gate | Required evidence | Current state |
| --- | --- | --- |
| Selected environment | Profile, workspace, compute ID/type, DBR/Python/Spark/MLflow/Arrow | `skyulf`; serverless Connect; successful runtime recorded above |
| Package delivery | Exact wheel/checksum, installed distribution on clean driver/workers | Non-editable driver + worker prediction passed; per-worker provenance pending |
| Real UC round-trip | Concrete model version, alias pinning and trusted bundle download | Register/download/local checks reached; alias gate pending |
| Spark parity | Gold data, both FE modes, key-based predictions, job run ID/URL | Passed in live corrected run `447606109645160` |
| Monthly publication | Real test Delta tables, snapshots, commits, replay, other periods preserved | SM-15 local evidence only |
| Distributed admission | Shared authority, ownership through commit, competing-job result | Implemented; real local Delta tests passed; live gate pending |
| UC permissions | Actual denied and allowed access in the approved namespace | Pending |
| Scale | Synthetic distributed dataset, runtime and driver/worker memory evidence | Pending |

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

For SM-16, prepare the script/wheel and use a reviewed job request after the
user selects the platform. Reusable templates remain SM-20. A future generated
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

Reference: [delta-rs writing](https://delta-io.github.io/delta-rs/usage/writing/).
