# SM-16 platform validation

Status: **ACTIVE — local preparation; no live Databricks evidence yet**.
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
  This proves a local build only; the wheel has not been installed on Databricks
  workers, uploaded or deployed. Rebuild/recompute the hash after code changes.

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
- The user will provide the CLI profile, permitted test `catalog.schema`
  and compute selection. None has been inferred from an earlier pasted host.

## Stage 1: registry-to-Spark probe

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
| Selected environment | Profile, workspace, compute ID/type, DBR/Python/Spark/MLflow/Arrow | Awaiting user selection |
| Package delivery | Exact wheel/checksum, installed distribution on clean driver/workers | Pending |
| Real UC round-trip | Concrete model version, alias pinning and trusted bundle download | Pending live execution |
| Spark parity | Gold data, both FE modes, key-based predictions, job run ID/URL | Local probe preparation |
| Monthly publication | Real test Delta tables, snapshots, commits, replay, other periods preserved | SM-15 local evidence only |
| Distributed admission | Shared authority, ownership through commit, competing-job result | Not implemented/validated |
| UC permissions | Actual denied and allowed access in the approved namespace | Pending |
| Scale | Synthetic distributed dataset, runtime and driver/worker memory evidence | Pending |

Do not use `LocalTableLock` on a distributed Databricks driver. Limiting one
job's concurrent runs does not coordinate other jobs or external writers.
An expiring lease without sink-enforced fencing is insufficient. The platform
admission choice remains an explicit design/validation step after runtime
selection, not a no-op context manager pretending to own a distributed lock.

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
