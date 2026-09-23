# Spark and Databricks gaps after SM-16

Date: 2026-09-22. Inspected baseline: `63dd3e21`, branch `090`.
This is a source review and roadmap update, not new runtime validation.

**Historical planning snapshot:** the user subsequently parked broad SM-17 and
selected the Databricks integration lane first. The source findings below remain
useful; priority/status statements are superseded by
[04-databricks-integration-plan.md](../04-databricks-integration-plan.md) and the
current queue. No missing node/model support was implemented by changing priority.

## User decisions

- Do not start SM-15L today. It is deferred until explicitly resumed.
- Prioritize completion of existing node and model coverage through SM-17.
- Park SM-18 and all later implementation, including endpoints and templates.
- Account for every existing registered node and configuration. Merely documenting
  an unsupported operation does not satisfy the requested completion scope.
  Keep the owning task open unless implemented or explicitly deferred by the user.
- Do not replace a local algorithm with a different Spark algorithm silently.
  Native distributed fit, native apply, worker Python inference and explicit
  bounded local execution are distinct contracts.

## What the repository actually provides

An AST scan of literal `NodeRegistry.register` calls across `skyulf-core/skyulf`
found **100 source registration IDs: 62 preprocessing and 38 modeling**. This
includes aliases and optional-dependency registrations; it is not 100 independent
algorithms or a claim that every registration loads in the base environment.
The complete ID inventory and task ownership are in [NODE_SUPPORT.md](../NODE_SUPPORT.md).

Only `SimpleImputer` and `StandardScaler` currently declare Spark fit/apply
capabilities. The imputer declaration covers mean/constant; the scaler covers
its mean/std flag combinations. Native FE and Python-worker FE inference do not
provide an automatic fallback for the other nodes. Their artifact/state and
execution contracts must be implemented and tested explicitly.

The new bundle's `inference/_model.py::validate_model` requires a standalone
sklearn estimator from a `sklearn.*` module, regression/classification semantics,
single output and classifier `predict_proba`. Custom prediction appliers require
an adapter. Consequently, existing local XGBoost/LightGBM registrations, clustering
and all special/composite model paths must be audited separately; none is made
portable merely by using a pickle. Native Spark model training and persistence
are additional work, not an existing consequence of distributed prediction.

SM-16 verified one selected Databricks serverless regression workflow: Polars
training, registered bundle loading, both Spark prediction modes, monthly Delta
replacement/replay, shared admission and real access denial. It did not certify
all 100 registrations, Spark training, classification on Databricks, HTTP serving,
online feature lookup or SQL endpoint calls.

Code entry points reviewed:

- `skyulf-core/skyulf/core/capabilities.py` and `registry.py`.
- `preprocessing/imputation/simple.py`, `scaling/standard.py` and the registration inventory.
- `inference/bundle.py`, `_model.py`, `spark.py` and `integrations/mlflow/model.py`.
- `integrations/databricks/{batch,delta,delta_admission,_contracts}.py`.

## Changes to SM-17

| Task | Deliverable and acceptance boundary |
| --- | --- |
| SM-17-00 | Complete per-ID/per-configuration inventory; fit/apply/codec/inference/runtime tests tracked separately; aliases linked, optional packages counted |
| SM-17a | Casting, cleaning, date/math/interaction/polynomial and basic scaling; native expressions and fitted-state parity |
| SM-17b | Remaining imputation strategies, encoding, bins/ranges and robust/quantile statistics; exact/approximate policy, category order, unseen/null behavior |
| SM-17c | Selection, general/power transforms and outliers; selected-column apply separated from distributed fit; row filtering preserves X/y/key alignment |
| SM-17d | Group/window/history and target/WOE OOF contracts; train-only state, deterministic ordering/ties, partition-boundary and leakage tests |
| SM-17e | Tokenization/vectorization/embedding and geo; vocabulary/hash/normalization parity, sparse/vector schemas, optional dependency and worker packaging |
| SM-17f | Splits, resampling and inspection; deterministic membership, training-only row changes, bounded previews and distributed aggregate profiles |
| SM-17g | Current-model inference coverage, including XGBoost/LightGBM, ensembles/calibration, non-probabilistic classifiers, clustering and explicit output contracts; save/load/registry/worker parity |
| SM-17h | Native distributed model fit and transform: explicit Spark ML backend, feature-vector metadata, label/probability mapping, model persistence and MLflow flavor; no implicit sklearn-to-Spark substitution |
| SM-17i | Evaluation, CV/tuning, thresholds and explainability/SHAP contracts for model backends; distinguish parallel trials from distributed fit; explicit bounded execution and metric collection |
| SM-17j | SDK/config usability and preflight: independent platform/FE/model/sink/tracking/registry choices, selected-version bundle loader, actionable unsupported/runtime errors and examples |
| SM-17k | Complete family gates, all-ID coverage guard, fit/save/load/predict integration and selected Databricks runtime checks; no open requested coverage silently converted to DONE |

For native training, prioritize linear/logistic, tree/forest/GBT and clustering
families after mapping actual local semantics. Other model IDs retain individual
tasks. Ridge/lasso parameterizations, SVC/KNN, boosting libraries, ensembles and
resampling do not automatically have equivalent Spark implementations. Each
needs an explicit backend/adapter decision and evidence; equivalence is not
established by matching class names or random seeds. Distributed library training
such as XGBoost is distinct from Python-worker prediction of a locally fitted model.

Spark-native models may require a different artifact kind from the current
pickle-backed Python bundle. Version that contract deliberately, preserve old
bundles, and do not require a Spark session in a local HTTP serving process.

## Non-template patterns reviewed in the supplied example

Reference root:
`C:/Users/Murat/Desktop/codes-main (1)/codes-main/dbml-mlops-template`.
Paths below are relative to `template/{{.project_name}}/{{.project_name}}_classification`.
The review covered the relevant Python services and structure; it did not execute
the example, audit every notebook/test or establish the example's production readiness.
No source was copied into Skyulf and no resources were deployed.

| Reference code | Useful pattern | Skyulf placement / state |
| --- | --- | --- |
| `common/config_service.py.tmpl` | Central config and per-developer resource scope | SM-17j SDK configuration; resource generation remains parked SM-20 |
| `src/training/services/{model_factory,training_service}.py.tmpl` | Model family interface, tuning and evaluation separation | SM-17g/h/i; reuse existing Skyulf modeling abstractions |
| `src/training/services/models/random_forest_classifier_service.py.tmpl` | Explicit sklearn/pandas model service | Confirms Databricks placement need not imply Spark-native fit |
| `common/services/feature_store_service.py.tmpl`, `src/training/services/data_preparation_service.py.tmpl` | UC feature tables, training-set lookups, timestamp keys, optional online publication | Parked SM-21; point-in-time joins and online freshness need their own contracts |
| `src/training/services/registry_service.py.tmpl`, `src/validation/services/model_validation.py.tmpl` | Validation against a champion, version tags and explicit deployment lifecycle | Parked SM-22; retain pinned identities and separate evaluation from promotion |
| `src/inference/services/inference_service.py.tmpl` | Feature Store `score_batch` or MLflow Spark UDF scoring | Optional adapters, not a replacement for Skyulf's existing native FE / worker paths |
| `src/deployment/services/endpoint_service.py.tmpl` | Endpoint configuration, ACLs, readiness, traffic routes and inference logging | Parked SM-19a/d; authentication via supported credential providers, bounded retries |
| `src/monitoring/services/data_quality_service.py.tmpl` | Monitoring setup/refresh, data quality and drift integration | Parked SM-23; reuse Skyulf metrics, keep platform provisioning optional |
| `tests/integration/serving_endpoint_test.py.tmpl`, `run_serving_endpoint_locust_test.py.tmpl` | Real endpoint contract and load tests | Parked SM-19; local pyfunc loading is not a serving-endpoint test |

Adopt concepts, not assumptions: the example's binary classification code uses
probability position 1 and an integer threshold label. Skyulf must preserve its
manifest class order, original labels, multiclass outputs and saved thresholds.
Likewise `randomSplit` does not establish cross-engine row-membership parity;
unbounded local conversions or model loads must not bypass Skyulf resource limits.
The example's automatic first-champion promotion is not a default for Skyulf.

## Batch, live and SQL access

| Access path | Current state | Remaining work |
| --- | --- | --- |
| Local Python prediction | Implemented for the current bundle contract, with pandas/Polars input | Broader node/model coverage in SM-17 |
| Distributed Spark prediction | Native FE + Python model, or worker Python FE + model | Broader coverage and native Spark models in SM-17 |
| Scheduled/monthly Delta batch | Spark runner and retry/publication guards implemented; schedule caller-owned | SM-15L deferred; orchestration conveniences separate from correctness |
| Existing Skyulf HTTP deployment | Existing backend path uses legacy artifacts | Bridge and compatibility in parked SM-18; not proof of Databricks endpoint support |
| Databricks live HTTP serving | MLflow pyfunc packaging exists; endpoint integration not delivered | Parked SM-19a: create/update/readiness, model dependencies, query ACLs, schema/parity, timeout/error behavior |
| SQL `ai_query` | No Skyulf invocation adapter or live validation yet | Parked SM-19b: invoke the existing serving endpoint, named struct/signature, return schema, CAN QUERY, row errors and capacity |
| Continuous/streaming inference | Not provided by the batch runner | Parked optional SM-19c: checkpoint/state/watermark, model updates and idempotency |

`ai_query` is an invocation surface for an existing Model Serving endpoint, not
a third model-training engine or necessarily a separate endpoint deployment.
SQL endpoint calls also do not inherit the direct Delta runner's idempotent
publication guarantees automatically.

Live serving can reuse the Python model + fitted FE bundle for compatible
row-local operations. Window/history-dependent FE needs caller-supplied context
or an explicit lookup design; a one-row REST request cannot reconstruct a whole
Spark partition. Spark-trained model serving needs an independently validated
serving artifact/runtime; Spark batch support alone is insufficient.

## Usability recommendation

Today the API is usable by an engineer, but it is still several explicit steps:
resolve a registry version, load the bundle, construct the period/snapshot spec,
select admission and execute. The SM-16 probe's uploads, grants and wheel checks
were validation work, not the intended everyday user workflow.

SM-17j should provide one documented config-driven entry path and a preflight
report. Keep platform, FE engine, model backend, sink and tracking/registry
independent. Derive schema/feature order/model digest/code version when already
known, but retain explicit period, source snapshot, target version and alias
promotion choices. Give a minimal notebook example and an advanced customization
example. Do not create resources or start jobs implicitly during config validation.
Backend/Canvas controls and generated DAB files remain parked.

## Verified platform references

- [Databricks serverless environment 4](https://docs.databricks.com/aws/en/release-notes/serverless/environment-version/four)
  supports `pyspark.ml` and `mlflow.spark`, with model-size and unsupported-model
  limits. Do not preserve the outdated blanket assumption that serverless cannot
  train Spark ML models; Skyulf still needs its own adapter and runtime tests.
- [Serverless limitations](https://docs.databricks.com/aws/en/compute/serverless/limitations):
  Connect APIs, cache restrictions and bounded UDF memory affect implementation.
- [Custom model serving](https://docs.databricks.com/aws/en/machine-learning/model-serving/custom-models):
  logging a compatible MLflow model is a prerequisite, not endpoint validation.
- [ai_query](https://docs.databricks.com/aws/en/sql/language-manual/functions/ai_query):
  existing endpoint, compatible input schema, permissions and supported runtime.

These references were checked on 2026-09-22. Runtime support must be rechecked
when implementing each adapter, including serverless versus classic compute.
