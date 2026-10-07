# Optional feature production, YAML settings and wider Spark scoring

Date: 2026-10-07. Branch: `093`, based on `43a3a585`. The user requested a local
DCO commit of this delivery. No cloud resources, model aliases or deployed jobs changed.
The user deferred fresh Databricks execution. Earlier cloud evidence does not
validate this delivery.

## Delivered behavior

1. Optional Spark feature production creates one task per data domain and a
   dependent merge task. Initialization pins Delta input versions and declared
   transform-file hashes. Groups may run in parallel; `selected_groups` controls
   recomputation versus reuse. Repairs consume the original initialization
   snapshot. Null/duplicate keys, incompatible timestamp types, feature collisions,
   missing matches and changed join cardinality fail before merged publication.
   Exact and backward as-of joins execute in Spark. Delta outputs are created with
   CDF or upserted with explicit schema checks; prior history is retained.
2. `migrate_config.py` converts supported generated Python/JSON projects into
   `training.yml` and `inference.yml`. `features.yml` separately declares optional
   data domains. Shared defaults and per-model differences use one loader. JSON,
   YAML and Python cannot own the same setting twice. Custom calculations and
   fitted preprocessing remain Python. The initializer still uses its legacy
   format; migration is explicit and preserves byte-exact backups.
3. Certified pandas-worker prediction now admits exact sklearn DecisionTree,
   RandomForest and ExtraTrees regressors and single-target classifiers, including
   reviewed tuned wrappers. Fitted MinMaxScaler has state and recipe checks.
   Model sets can combine numeric named outputs with `weighted_sum` or
   `weighted_mean`, checked weights and explicit float64 output schemas.
4. SM-21a has offline temporal joins and optional native Feature Engineering API
   helpers. Full Skyulf training/logging/scoring lifecycle wiring is **not done**.
   SM-21b online publication/serving lookup remains optional and unimplemented.

Ready-table projects get no feature job or feature notebook until enabled.
Single, competition and model-set projects can all share the same upstream data
domains; feature count does not follow model count. Training remains bounded
pandas fitting. Distributed inference uses pandas batches on Spark workers and
does not distribute estimator training or tuning.

## Edit locations and execution

The generated [FEATURE_TABLES guide](../../skyulf-core/templates/databricks/template/%7B%7B.project_name%7D%7D/FEATURE_TABLES.md)
contains the graph, examples, migration, refresh commands and selective repair.

| User-owned file | Purpose |
| --- | --- |
| `config/features.yml` | Data domains, input/output Delta tables, keys/time and joins |
| `src/feature_groups/*.py` | Spark domain transformations |
| `config/training.yml` | Training, models, tuning and per-model differences after migration |
| `config/inference.yml` | Scoring mode/destinations and model-set composition after migration |
| `src/features/` | Existing custom preprocessing and business functions |

Run feature graph refresh after changing domain configuration or inherited job
controls. It generates `resources/features.job.yml` and a thin notebook, carrying
the training job's compute, target-specific Run-as and optional permissions.
Generated resources should not be hand-edited. Set the downstream training and
score source to the merged table; run feature production before those jobs.
Feature production does not automatically train/promote models.

## Library ownership

- `integrations/databricks/features/`: validated configuration, Spark joins, Delta
  publishing, snapshot/receipt runtime and graph generation.
- `integrations/databricks/jobs/features/`: notebook parameters, bounded task
  evidence and routing to the feature runtime.
- `integrations/databricks/projects/yaml_{config,models,migration}.py`: inert
  parsing, authoritative ownership, adapters and migration. Project, competition,
  branch, model-set, notebook, smoke and preview callers use this boundary.
- `inference/_partition_trees.py`, `_partition_nodes.py` and
  `_model_set_operations.py`: checked tree/scaler state and row-local arithmetic;
  existing admission and model-set scoring call these helpers.
- `integrations/databricks/feature_store/`: optional native SDK lookup,
  training-set, flavor logging and batch-scoring helpers. See
  [SM-21a evidence](181-feature-store-evidence.md) for its exact boundaries.

PyYAML is directly declared in the MLflow/all extras, shared requirements,
generated runtime and development dependencies. The optional `feature-store`
extra requires `databricks-feature-engineering>=0.11,<1.0`; it is not installed
by normal Bundle operation. `uv.lock` is refreshed. The existing Delta CI lane
now executes the feature-publishing integration tests.

## Review findings resolved

Independent domain review and root integration checks found and corrected:

- Duplicate JSON/YAML ownership, including per-model overrides, previously
  allowed silent replacement. Nested/deep YAML and duplicate literal declarations
  also needed bounded, explicit rejection.
- Migration had to reject executable defaults, decorators, annotations and
  custom factories it cannot preserve. Static smoke now validates model types.
- Notebook evidence accepted nonfinite JSON, and direct feature dataclasses could
  accept string keys/columns. Regression tests now reject both.
- Partially qualified tables could alias other configured destinations. Feature
  plans require explicit three-part names and separate inputs/outputs.
- Feature jobs initially omitted target-specific Run-as/permissions. They now
  inherit those controls in shared and personal targets.
- Optional per-directory sync globs and backup exclusions emitted strict CLI
  warnings when absent after migration. One `src/**/*.py` include and generated
  `.gitignore` backup/staging rules pass strict validation before and after migration.
- Real CLI canonicalizes task ordering. The integration assertion now checks
  actual dependency edges rather than serialization order.

The transform snapshot covers its declared entry file, not arbitrary imported
helpers or external reads. This is documented rather than claimed as hermetic
execution. No unresolved peer request remains.

## Verification and evidence

All commands used explicit affected files. Counts below overlap and must not be
summed as a unique suite total. Temporary fixtures, Java installations and logs
live under ignored `tmp_repro_artifacts/task181/`.

| Verification | Result and scope |
| --- | --- |
| Feature/config/runtime/notebook/SDK union | **98 passed**, `root-final-union.log`; graph inheritance subsequently verified with **5 passed** |
| Real generated Bundle CLI | **3 passed**, `feature-cli3.log`; each layout migrated to YAML and strictly validated against local fake identity service for shared and personal targets |
| Final affected template checks | **82 passed, 10 opt-in CLI skips**, `template-final.log`; the new three-layout CLI matrix ran separately |
| Final YAML regressions | **27 passed**; earlier affected 11-file union **258 passed, 14 opt-in skips**, followed by rerun for migration-only changes |
| Affected inference/MinMax consumers | **271 passed**, `minmax-affected.log`; earlier pre-MinMax inference union **291 passed, 1 Windows symlink skip** |
| Actual Spark JVM/workers, current source | **12 passed**, `native-final.log`; all six tree families, MinMax, both arithmetic operations, partition/batch sizes, empty/null inputs and temporal joins |
| Actual Delta transactions on Linux | **2 passed**, `delta-linux.log`; create/upsert/no-op and pinned-version replay/merge |
| Ruff / formatter | Passed repository check and CI Python format scope, 1,563 files |
| Ty / complexity | Passed full CI Ty scope and Lizard CCN <=10 |
| Template schema / lock | `build_schema.py --check` and `uv lock --check --offline` passed |
| Wheel packaging | Offline build passed; all 536 packaged Python files match source paths, and the generated feature notebook source is byte-identical |

Representative final commands:

```powershell
# Base environment; basetemps/log redirection omitted for readability.
.venv\Scripts\python.exe -m pytest skyulf-core/tests/integration/platforms/test_feature_group_config.py skyulf-core/tests/integration/platforms/test_feature_group_graph.py skyulf-core/tests/integration/platforms/test_feature_group_runtime.py skyulf-core/tests/integration/platforms/test_feature_group_notebook.py skyulf-core/tests/integration/platforms/test_databricks_feature_store.py -q -o addopts=
$env:SKYULF_BUNDLE_CLI_TEST_PROFILE = 'offline'
.venv\Scripts\python.exe -m pytest skyulf-core/tests/integration/platforms/test_feature_bundle_generation.py -q -o addopts=
# Real local Java 17 and Spark; no cloud connection.
.venv-spark\Scripts\python.exe -m pytest skyulf-core/tests/spark/test_declarative_model_set.py skyulf-core/tests/spark/test_feature_group_joins.py -q -o addopts=
.venv\Scripts\python.exe -m ruff check .
.venv\Scripts\python.exe -m ruff format --check backend skyulf-core tests run_skyulf.py celery_worker.py
.venv\Scripts\ty.exe check backend skyulf-core/skyulf skyulf-core/tests run_skyulf.py celery_worker.py
.venv\Scripts\lizard.exe backend skyulf-core/skyulf --CCN 10 -w
.venv\Scripts\python.exe skyulf-core/templates/databricks/build_schema.py --check
uv lock --check --offline --cache-dir tmp_repro_artifacts/task181/uv-cache
git diff --check
```

Delta ran in an isolated WSL Ubuntu environment using `requirements-delta.txt`,
Java 17 and the official Delta jars, with `SKYULF_REQUIRE_DELTA=1`. The exact local
script is `tmp_repro_artifacts/task181/run_delta_linux.sh`. Windows Spark pytest
returned exit 0; JVM shutdown printed an additional Windows `Access denied`
message after the passing summary. No test result is based on cloud execution.

Full suites/coverage remain CI-owned. Frontend checks are inapplicable. MkDocs,
push and deployment were not run. Pre-commit validation is recorded below for the
requested local commit. Agent-specific detailed evidence is in
[YAML evidence](181-yaml-config-evidence.md) and
[Feature Engineering evidence](181-feature-store-evidence.md).

Commit follow-up (2026-10-07): the user requested a local DCO commit. The staged
63-file scope contains only this delivery and its evidence. All applicable
pre-commit hooks passed: whitespace/end-of-file, YAML, Ruff/format, full-scope Ty
and Lizard CCN 10. Frontend, schema and JSON hooks had no applicable staged files;
the separate schema check is recorded above. Production code did not change
after the recorded affected tests. The first-run guide now explicitly directs
migrated projects to YAML instead of the removed Python declarations. The shared
coordination log's protocol header was restored from HEAD after an earlier log
compaction dropped it. No push or Databricks execution is part of this commit.

## Remaining work and explicit limits

- **SM-21a remains PARTIAL:** preserve native TrainingSet and lookup metadata
  through Skyulf fit/MLflow packaging, route only feature-packaged artifacts to
  `fe.score_batch`, retain signatures/partition certificates, and verify historical
  lookup plus saved-model replay on Databricks. Generic SDK delegation is not this
  completed lifecycle.
- **SM-21b remains LATER:** add online publication/freshness and serving lookup
  only when an online consumer requires it.
- Feature YAML currently uses literal fully qualified table names; destination
  schemas/permissions must exist. Deployment-placeholder expansion is not added.
- Outputs need one owning producer. Job-level concurrency does not lock other
  jobs. Schema evolution and deletion of disappeared source records are not
  automatic. Sources/output history must remain available for pinned replays.
- As-of joins use the declared effective/available timestamp, without a separate
  arrival-time column or offline lookback limit. Imported transform helpers are
  not frozen by the entry-file digest.
- Custom Python composition, XGBClassifier, RobustScaler and unreviewed estimator
  subclasses are not newly admitted. Portable `predict_spark` codec support and
  distributed estimator fitting are unchanged.
