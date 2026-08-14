# Per-node code escape hatch — feasibility and security

**Date:** 2026-08-11  
**Decision:** **Feasible in phases, but arbitrary edited Python must not execute in Skyulf's current API or Celery workers.** Ship a read-only explanation/export first; permit only constrained edits next; defer arbitrary executable Python until identity, tenant isolation, and a dedicated hardened executor exist.

## Executive verdict

Skyulf is presently a declarative graph interpreter, not a code-artifact runtime. A canvas node becomes a `{node_id, step_type, params, inputs}` record, then the backend resolves the registered calculator/applier for `step_type`; the submitted `params` influence fixed, trusted implementation code rather than replace it [frontend/ml-canvas/src/core/utils/pipelineConverter.ts:165-202, 561-584](../../frontend/ml-canvas/src/core/utils/pipelineConverter.ts#L165-L202) [skyulf-core/skyulf/preprocessing/pipeline.py:94-120, 186-250](../../skyulf-core/skyulf/preprocessing/pipeline.py#L94-L120). Therefore, an editable code string would be a new execution model and an `exec`-like sink, not a small extension of an existing feature.

**Recommended product boundary:**

1. **Phase A (safe now):** display generated, read-only, per-node execution code and improve the current full notebook export. This does not change execution and carries no new server-side code-execution risk.
2. **Phase B (safe-ish, after normal auth/authorization):** offer a “code-like advanced editor” for an allow-listed declarative AST/parameter expressions, compile it back to canonical parameters, and reject everything else. This is not arbitrary Python, and should be marketed honestly as an advanced transform editor.
3. **Phase C (not safe to schedule until foundations land):** arbitrary Python node, code-only after conversion, executed only in a separately isolated per-job runtime. It depends on the critical authentication/multi-tenancy/data-plane blockers already identified in the enterprise plan [initiatives/enterprise-readiness/2026-08-11-backend-blockers.md:21-73, 240-255](../enterprise-readiness/2026-08-11-backend-blockers.md#L21-L73).

A bidirectional “edit any Python, then reconstruct every native panel” is not credible for v1. The realistic UX is a one-way **Convert to code node** door: preserve the original generated source/config as provenance, disable the node-specific form, and provide a code editor plus explicit input/output contract. This matches the way code escape hatches normally avoid pretending arbitrary code is invertible.

## Current architecture and the changes it requires

### What executes now

* Core preprocessing is a two-stage, stateful protocol: a calculator fits a mapping/artifact from `(data, config)`, then an applier applies that artifact. `StatefulTransformer` stores the fitted parameters and invokes `calculator.fit(...)` followed by `applier.apply(...)` [skyulf-core/skyulf/preprocessing/base.py:82-125, 158-206](../../skyulf-core/skyulf/preprocessing/base.py#L82-L125).
* `FeatureEngineer` takes each declarative step's `transformer` and `params`, obtains registered components, and runs a `StatefulTransformer`; it persists only the applier and fitted artifact for later transform [skyulf-core/skyulf/preprocessing/pipeline.py:94-120, 186-250](../../skyulf-core/skyulf/preprocessing/pipeline.py#L94-L120). Its inference path likewise calls the stored applier, not stored source code [skyulf-core/skyulf/preprocessing/pipeline.py:52-75](../../skyulf-core/skyulf/preprocessing/pipeline.py#L52-L75).
* Representative implementations confirm that config is data, not source: StandardScaler reads `with_mean`/`with_std`, fits sklearn's fixed `StandardScaler`, serializes mean/scale/columns, and its applier uses those artifact values [skyulf-core/skyulf/preprocessing/scaling/standard.py:41-95, 98-155](../../skyulf-core/skyulf/preprocessing/scaling/standard.py#L41-L95). One-hot encoding reads options from `config`, instantiates a fixed sklearn encoder, then persists that object and feature names [skyulf-core/skyulf/preprocessing/encoding/one_hot.py:113-172, 190-232](../../skyulf-core/skyulf/preprocessing/encoding/one_hot.py#L113-L172). Feature generation simply turns configured operations into an artifact and dispatches to fixed pandas/Polars functions [skyulf-core/skyulf/preprocessing/feature_generation/generation.py:18-45](../../skyulf-core/skyulf/preprocessing/feature_generation/generation.py#L18-L45).
* The backend follows the same model: the generic transformer runner wraps `node.step_type` and `node.params` in a one-step `FeatureEngineer` [backend/ml_pipeline/_execution/engine/_node_runners.py:809-842](../../backend/ml_pipeline/_execution/engine/_node_runners.py#L809-L842), and unrecognised non-special nodes fall into that runner [backend/ml_pipeline/_execution/engine/__init__.py:227-258](../../backend/ml_pipeline/_execution/engine/__init__.py#L227-L258).

### Required architecture for a code node

A full code feature needs a distinct `custom_python` node kind—not an arbitrary `step_type` or a new `params["code"]` that the existing generic runner evaluates. It needs:

1. a persisted source revision, language/runtime version, dependency allow-list, input/output schema contract, immutable provenance (generated base source + user patch/source), and content hash;
2. a separate executor protocol that materializes only the authorized input partition and accepts only validated tabular outputs (Arrow/Parquet/JSON), never live application objects;
3. graph-level schema/lineage behavior for a code node (unknown until run is acceptable), timeouts/cancellation and resource accounting; and
4. export behavior that emits the exact frozen source (and environment manifest), rather than asking `NodeRegistry` to recreate it.

The existing request schema allows arbitrary dictionaries but no source/artifact type: `NodeConfigModel.params` is `dict[str, Any]` [backend/ml_pipeline/_internal/_schemas.py:28-42](../../backend/ml_pipeline/_internal/_schemas.py#L28-L42). Adding `code` there without changing where it runs would make the trusted execution boundary dangerous.

## Existing export and Phase-A feasibility

The current notebook exporter is a good starting point for **showing an explanation of the actual execution pattern**, but not for claiming to show literal calculator source or the exact expanded pandas/sklearn statements.

* It accepts the canonical graph shape and describes full mode as one cell per preprocessing node [backend/ml_pipeline/_internal/_routers/notebook_export.py:3-22, 411-446](../../backend/ml_pipeline/_internal/_routers/notebook_export.py#L3-L22).
* Full-mode `node_to_cell` serializes the node parameters, resolves the calculator/applier from `NodeRegistry`, then emits `fit`/`apply` calls [backend/ml_pipeline/_internal/_routers/_notebook_builders.py:353-364](../../backend/ml_pipeline/_internal/_routers/_notebook_builders.py#L353-L364). Compact mode instead emits a `SkyulfPipeline` config built from those declarative step records [backend/ml_pipeline/_internal/_routers/_notebook_builders.py:189-196, 269-280](../../backend/ml_pipeline/_internal/_routers/_notebook_builders.py#L189-L196). This is templated registry glue plus substituted config, **not** a decompilation of each node's pandas/sklearn implementation.
* It does preserve topological node order [backend/ml_pipeline/_internal/_routers/notebook_export.py:72-106, 323-364](../../backend/ml_pipeline/_internal/_routers/notebook_export.py#L72-L106), but it iterates only `preprocess` through `node_to_cell`; split/model handling is emitted by separate builders [backend/ml_pipeline/_internal/_routers/notebook_export.py:347-364](../../backend/ml_pipeline/_internal/_routers/notebook_export.py#L347-L364). Therefore, “every node” needs additional generators for loaders, splitters, models, resampling and preview/visual nodes.

There are two different claims the UI must label accurately. A **faithful generated execution representation** can reuse the current registry-call template and substitute the active config. Literal **implementation source** would require surfacing the relevant calculator/applier source (often multiple helpers and pandas/Polars branches) and still would not produce a standalone config-specialized snippet. A readable, directly runnable “exact pandas/sklearn code with this config” needs an explicit source-generator contract per node type, tested against the registered implementation; there is no such contract today. Do not label Phase-A registry calls as the literal code executed.

For Phase C, faithful export is straightforward only if custom source becomes immutable node data: emit it verbatim with a stable wrapper/input-output contract, pinned dependencies, and a hash. It is **not** possible to faithfully export arbitrary edits through the existing templates, because templates discard internal UI/routing fields and regenerate a call to the registered node [backend/ml_pipeline/_internal/_routers/_notebook_builders.py:65-84, 189-196](../../backend/ml_pipeline/_internal/_routers/_notebook_builders.py#L65-L84). Do not silently fall back to a template: exported code must visibly identify a custom-code node and include its exact frozen text.

## Security analysis

### RCE and present blast radius

Executing customer-edited Python is deliberate arbitrary code execution. The current pipeline submission route accepts node params, rebuilds the graph, and dispatches it through Celery or FastAPI `BackgroundTasks` [backend/ml_pipeline/_internal/_routers/run_pipeline.py:263-291, 368-416](../../backend/ml_pipeline/_internal/_routers/run_pipeline.py#L263-L291). The execution service reconstructs the graph then invokes `PipelineEngine` in that process [backend/ml_pipeline/_services/pipeline_execution_service.py:71-86, 169-225](../../backend/ml_pipeline/_services/pipeline_execution_service.py#L71-L86).

Celery is an async task mechanism, **not** a sandbox:

* The same worker registers ingestion, ML-pipeline, and monitoring tasks and initializes the application database/settings [celery_worker.py:12-19, 35-54](../../celery_worker.py#L12-L54). Pipeline tasks open a normal SQLAlchemy session and call `execute_pipeline` directly [backend/ml_pipeline/tasks.py:42-60, 82-98](../../backend/ml_pipeline/tasks.py#L42-L60).
* The development topology runs API and worker from the same image, bind-mounts the repository into each, uses the same Redis, and runs a solo worker [docker-compose.yml:13-48](../../docker-compose.yml#L13-L48). The image includes application source and writable upload/export/temp locations [Dockerfile:19-29](../../Dockerfile#L19-L29). Settings can expose AWS credentials to that process [backend/config/mixins/aws.py:4-15](../../backend/config/mixins/aws.py#L4-L15).
* JSON Celery message serialization reduces unsafe task-message deserialization, but it does not constrain Python that an application later evaluates [backend/celery_app.py:12-20](../../backend/celery_app.py#L12-L20).
* A sandbox must also not hand untrusted serialized artifacts back to trusted code: local artifact loading explicitly warns that `joblib.load` uses pickle and can execute code [backend/ml_pipeline/artifacts/local.py:29-44](../../backend/ml_pipeline/artifacts/local.py#L29-L44).

There is an adjacent egress concern: resolution returns datasource `storage_options` untouched [backend/ml_pipeline/resolution.py:35-58](../../backend/ml_pipeline/resolution.py#L35-L58), while the catalog strips top-level endpoint options but leaves an already nested `client_kwargs.endpoint_url` untouched [backend/data/catalog.py:226-242](../../backend/data/catalog.py#L226-L242). This reinforces that a code executor must default-deny network, not merely trust application-level validation.

### Ranked options

| Option | Safety / feasibility | Recommendation |
|---|---|---|
| RestrictedPython / AST filtering in the existing worker | Not a security boundary: Python object-graph escapes, libraries/native extensions, filesystem and process credentials remain. Useful only to implement Phase-B *syntax*, never to contain hostile tenants. | Do not use as a sandbox. |
| Subprocess with `setrlimit`, timeout, basic namespace restrictions | Better resilience against accidental loops, but shares a kernel and is too easy to misconfigure for hostile multi-tenant Python. | Single-tenant/trusted development only. |
| Separate hardened ephemeral container/Kubernetes Job | Practical first production boundary if enforced: non-root, read-only root filesystem, no host mounts/Docker socket, dropped capabilities, seccomp/AppArmor, per-job scratch, CPU/RAM/PID/time limits, network default-deny, scoped data token. | Recommended production implementation. |
| Per-execution Firecracker/gVisor microVM | Stronger isolation against container/kernel escape; higher platform/latency/operations cost. | Preferred high-assurance/multi-tenant tier; evaluate after container executor. |
| Pyodide/WASM | Strong in-browser/local experimentation but poor fit for current pandas/sklearn/Python package compatibility and server ML workload. | Not a backend escape-hatch runtime. |

**Minimum viable safe arbitrary-code design:** a control-plane service authorizes `(principal, workspace, code revision, dataset version)` and sends only an opaque job ID to a dedicated executor. That executor has its own image/service account/broker queue, zero application DB/Redis/AWS credentials, no source mount and no access to trusted queues; it obtains scoped read-only input and has write-only output upload. It has default-deny egress, resource limits and killable deadlines. Trusted services validate Arrow/Parquet/JSON output schemas and never deserialize executor-produced pickle/joblib. Add audit logs, per-workspace quotas, source retention and explicit user consent.

This is not achievable by extending the current shared worker alone. It is gated by real identity and workspace ownership: the prior blocker investigation reports no tenant/workspace foreign keys and no logical customer partitioning [initiatives/enterprise-readiness/2026-08-11-backend-blockers.md:52-73](../enterprise-readiness/2026-08-11-backend-blockers.md#L52-L73), plus no per-organization quotas [initiatives/enterprise-readiness/2026-08-11-backend-blockers.md:222-238](../enterprise-readiness/2026-08-11-backend-blockers.md#L222-L238).

## UI model and round-trip fidelity

Canvas state is untyped React Flow `node.data`; it is initialized from a node definition's defaults and mutated by shallow merging [frontend/ml-canvas/src/core/store/useGraphStore.ts:456-500](../../frontend/ml-canvas/src/core/store/useGraphStore.ts#L456-L500). Each registered node owns a typed settings component, validation function and default config [frontend/ml-canvas/src/core/types/nodes.ts:21-71](../../frontend/ml-canvas/src/core/types/nodes.ts#L21-L71); graph validation directly calls that definition validator on `node.data` [frontend/ml-canvas/src/core/store/useGraphStore.ts:169-244](../../frontend/ml-canvas/src/core/store/useGraphStore.ts#L169-L244). The converter is a large one-way, type-specific mapping with transformations such as selecting scaler/encoder step types and flattening feature transformations [frontend/ml-canvas/src/core/utils/pipelineConverter.ts:204-389](../../frontend/ml-canvas/src/core/utils/pipelineConverter.ts#L204-L389).

Consequences:

* Generated source can be rendered from the canonical converted node config, but source-to-UI inversion would need a parser and semantic mapper for every node, supported pandas/Polars branch, fitted-state behavior, aliases and arbitrary imports. It cannot faithfully represent a user changing control flow, adding imports, or changing output schema.
* V1 should add `mode: 'native' | 'custom_code'`, `generated_source`, `custom_source`, `base_config_snapshot`, `runtime_contract`, and `source_hash` to node data/persistence. In `custom_code`, hide/lock native settings and use a code-only definition whose validation demands a declared output contract. Provide **Discard custom code / restore native snapshot** as an explicit destructive return path rather than a false “sync back.”
* Phase B can stay bidirectional only for a deliberately small grammar: parameter literals, selected column names, and allow-listed expression operators compile to canonical params. No imports, attribute traversal, calls beyond an allow-list, statements, comprehensions or assignment. This materially reduces injection/RCE risk, but is an advanced parameter editor—not the requested general escape hatch.

## Precedent research

Targeted Brave searches were attempted for Dataiku, KNIME, n8n and Retool but the configured service returned rate-limit/invalid-subscription errors. Direct official documentation was then checked where accessible; inaccessible pages are linked below rather than treated as verified implementation detail.

* **Dataiku:** its official [Code recipes documentation](https://doc.dataiku.com/dss/latest/code_recipes/index.html) is the relevant “recipe is code” precedent. The page was dynamically rendered in this investigation, so validate exact UX/runtime claims in a product trial before using it as a detailed implementation model. Product implication: make conversion explicit rather than attempting arbitrary code-to-visual round trips.
* **KNIME:** the [Python Script node documentation entry](https://hub.knime.com/knime/extensions/org.knime.features.python3.nodes/latest/org.knime.python3.nodes.extension.PythonScriptNodeFactory) is the relevant precedent, but the documentation host denied automated retrieval. Its broad pattern is a separate script node with tabular ports—not arbitrary rewriting of every native node. Validate installation/environment and isolation mechanics in a hands-on spike.
* **Databricks:** official [AutoML documentation](https://docs.databricks.com/aws/en/machine-learning/automl/) says AutoML generates source notebooks for trials so users can review, reproduce and modify them, and distinguishes low-code UI from generated notebooks. This strongly supports Phase A/export and the one-way graduation model; it does not demonstrate executing untrusted notebook edits in Skyulf's shared service.
* **Google Vertex AI Workbench:** official [Workbench documentation](https://cloud.google.com/vertex-ai/docs/workbench/introduction) describes authenticated Jupyter instances, configurable environments/hardware, custom-training notebook execution, VPC/service-perimeter options and Cloud Storage result storage. It supports the conclusion that code runs need an explicit managed runtime/identity/network boundary, not a normal API worker.
* **n8n:** the official [task-runner configuration documentation](https://docs.n8n.io/hosting/configuration/task-runners/) and [Code node documentation](https://docs.n8n.io/code/code-node/) were 403-blocked here. They are nevertheless useful follow-up sources because n8n separates code execution into task runners; verify the precise isolation guarantees and deployment mode before borrowing the pattern.
* **Retool:** official [Queries and code](https://docs.retool.com/queries/quickstart) documents executable query/transformer blocks, including custom Python in workflows; its [JavaScript guide](https://docs.retool.com/queries/guides/javascript) states that browser/context interactions are restricted for safety. This is evidence for an explicit scoped code surface and restricted capabilities, not evidence that arbitrary server Python is safely sandboxed by a UI editor.

## Recommended plan and follow-up spike

| Phase | Scope | Gate / acceptance condition |
|---|---|---|
| A — inspect/export | Per-node read-only registry-call code view, plus an honestly labelled implementation-source link/view; source/config provenance; export reuses and extends the current full-cell builder. | Verify every node family has deterministic generated output and snapshot-test notebook cells. No backend execution change. |
| B — constrained edit | AST/DSL compiler to canonical params, per-node allow-lists, preview/validation and clear “advanced config” wording. | Threat-model the grammar; property tests prove rejected imports/statements/attribute escapes and parity tests prove compiled config equals native behavior. |
| C — custom Python | One-way custom-code node, exact-source export, dedicated isolated executor. | Auth + workspace ownership + tenant-scoped storage/audit/quotas; production deployment; external sandbox review and adversarial escape tests. |

Before estimating Phase C, spike: (1) whether all useful native nodes can expose a stable DataFrame/Arrow contract; (2) cold-start/runtime/package requirements for a hardened job; (3) data-token design and egress policy for S3/data sources; (4) artifact/result formats that avoid pickle across the trust boundary; (5) cancellation/quotas/log redaction; (6) legal/support policy for executing customer code; and (7) a product prototype testing whether power users accept one-way conversion plus restore-from-snapshot.
