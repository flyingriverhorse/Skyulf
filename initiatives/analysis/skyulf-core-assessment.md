# skyulf-core vs. the Databricks MLOps Template — Technical Assessment

**Date:** 2026-02
**Subject:** `/Users/BH7043/Skyulf/skyulf-core` (v0.8.5) evaluated as a replacement for the Python layer of `dbml-mlops-template`
**Question asked:** Is skyulf-core usable instead of the template's ML code? Does it need PySpark and MLflow added to become usable?

---

## Table of contents

1. [Executive summary](#1-executive-summary)
2. [The finding that reframes the PySpark question](#2-the-finding-that-reframes-the-pyspark-question)
3. [Side-by-side inventory](#3-side-by-side-inventory)
4. [Architecture comparison](#4-architecture-comparison)
5. [What skyulf-core does better](#5-what-skyulf-core-does-better)
6. [Code review — weaknesses in skyulf-core](#6-code-review--weaknesses-in-skyulf-core)
7. [The measured cost of adding Spark](#7-the-measured-cost-of-adding-spark)
8. [Layer-by-layer replacement analysis](#8-layer-by-layer-replacement-analysis)
9. [Roadmap](#9-roadmap)
10. [Decision summary](#10-decision-summary)
11. [Appendix — reproducing the measurements](#11-appendix--reproducing-the-measurements)

---

## 1. Executive summary

**skyulf-core is better-engineered than the template's Python layer.** It has explicit fitted state, protocol-typed contracts, registry-driven dispatch, a leakage guard the template does not have at all, and a 0.57 test-LOC-to-source-LOC ratio. The template's `src/` is a set of hand-written service classes with duplicated logic across four bundle copies and no equivalent safety net.

**It can replace roughly 60% of the template's Python layer today** — preprocessing, model fitting, cross-validation, drift profiling, expectations. It cannot replace the Databricks-native parts: Unity Catalog I/O, Feature Store lookups, MLflow logging/registry, and the serving-endpoint deployment.

**On the two additions you asked about:**

| Addition | Verdict | Reasoning |
|---|---|---|
| **MLflow** | **Yes — do this first.** High value, low cost. | The abstract seams (`ModelSerializer`, `ModelRegistry`, `ComputeBackend`) already exist and are explicitly documented as "ahead of the Databricks/MLflow phases". Implementing them is ~200–300 LOC and touches no node. |
| **PySpark (as a dataframe engine)** | **No.** Low value, very high cost. | The template does not train with Spark — it calls `.toPandas()` first. And skyulf-core's dual-engine dispatch is positionally hardcoded to exactly two engines across 126 call sites in 49 files. The cost is O(number of nodes), not O(1). |

What you need instead of a Spark engine is a thin **I/O adapter**: read from Unity Catalog with Spark, convert once via Arrow, run skyulf-core on Polars/pandas. That is the same shape the template already uses.

---

## 2. The finding that reframes the PySpark question

Before assessing whether skyulf-core needs Spark, I checked what the template actually does with Spark.

**It does not train with Spark.**

The model services import single-node estimators only:

```python
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from xgboost import XGBClassifier
```

And the training notebook collects to pandas before fitting — `src/training/notebooks/train.ipynb.tmpl`, lines 376–431:

```python
train_pdf = train_df.toPandas()
test_pdf  = test_df.toPandas()
```

The same pattern appears in `src/validation/model_validation.py` at lines 129 and 136.

**So Spark in this template is used for exactly three things:**

1. Reading and writing Unity Catalog tables
2. Feature engineering (`compute_features` / `merge_features`)
3. Feature Store lookups

Training, tuning, validation and inference are all single-node pandas.

**Implication:** adding a Spark *dataframe engine* to skyulf-core would buy you parity with a capability the template itself doesn't use for modelling. The genuine requirement is *interoperability at the boundary* — read Spark, hand off a local frame — which is a converter, not an engine.

---

## 3. Side-by-side inventory

### skyulf-core

| Metric | Value |
|---|---|
| Source files | 330 Python files |
| Source LOC | 78,923 |
| Test files | 150 |
| Test LOC | 45,384 |
| Test:source LOC ratio | **0.57** |

Module breakdown:

| Module | Files | LOC | Purpose |
|---|---|---|---|
| `preprocessing/` | 87 | 13,809 | Transformation nodes (Calculator/Applier pairs) |
| `modeling/` | 38 | 9,955 | Estimators, CV, tuning, metrics |
| `profiling/` | 23 | 5,518 | Drift detection, expectations, EDA |
| `core/` | 11 | 1,340 | Protocols, artifacts, schema, seams |
| `engines/` | 6 | 562 | Polars/pandas abstraction |
| `data/` | 3 | 96 | Loaders |

Top-level: `pipeline.py` (579), `utils.py` (401), `leakage.py` (164), `config_validation.py` (111), `registry.py` (108).

### dbml-mlops-template

| Metric | Value |
|---|---|
| Generated files (all bundles) | 514 |
| Bundle copies | 4 (classification 120, regression 117, churn_model 108, taxi_model 152) |
| `feature_store_service.py` | **byte-identical** across all 4 bundles |
| `data_service.py` | identical across 3 of 4 |
| Duplicated `new_cluster` blocks | 10 |
| `cluster_policy_id_*` variables | 6, all resolving to **1** policy |
| Repeated CI bash block ("Resolve workflow model configuration") | 5 copies |
| Go-escaped `{{ `"{{job.parameters...}}"` }}` sites | **89** |
| Placeholder lines repeating the same 4-column data contract | 26, across 5 config files |

The template has no leakage detection, no config schema validation, no cross-validation, and no fitted-state serialization contract.

---

## 4. Architecture comparison

### The template: three stacked interpolation layers

| Layer | Syntax | Resolved at |
|---|---|---|
| Go templating | `{{ .project_name }}` | `databricks bundle init` |
| DAB variables | `${var.x}`, `${bundle.target}` | `databricks bundle deploy` |
| OmegaConf | `${catalog_name}`, `${suffix}` | job runtime, via `ConfigService` |

This is the root of most of the template's accidental complexity. The 89 Go-escaped Databricks job-parameter references (`{{ `"{{job.parameters.catalog_name}}"` }}`) exist purely because layer 1 and Databricks' own parameter syntax collide.

Its Python layer is a set of service classes (`DataService`, `FeatureStoreService`, `ModelService`, `ConfigService`) copy-pasted across four bundles, with configuration passed as loosely-typed dicts.

### skyulf-core: registry + protocol + explicit state

The central design decision is the **Calculator/Applier split**:

```python
# Calculator: returns DATA, not mutated self
def fit(df, config) -> Mapping: ...

# Applier: consumes that data
def apply(df, params) -> DataFrame: ...
```

Contrast with sklearn, where fitted state hides in `self.mean_` and is only recoverable by pickling the whole estimator. In skyulf-core the fitted state is a plain mapping — inspectable, diffable, serializable, and transportable between processes.

These are formalised as `@runtime_checkable` Protocols in `core/protocols.py`, and each artifact gets a `TypedDict` with `total=False` in `core/artifacts.py` (so the empty-artifact early return stays assignment-compatible with the populated one — a small, correct detail).

Dispatch flows through `NodeRegistry` (`registry.py`), which carries metadata per node: `learns_from_data`, `is_splitter`, and so on. That metadata is what makes the leakage guard possible.

**Verdict:** skyulf-core has a real architecture. The template has a folder convention.

---

## 5. What skyulf-core does better

### 5.1 `leakage.py` — the standout

164 lines, and the best code in the library.

- It **derives** the data-dependent node list from `NodeRegistry` metadata rather than hardcoding a list, so it cannot drift as nodes are added.
- It **fails closed** (lines 155–157): an unknown transformer is treated as potentially leaking, not waved through.
- It has exactly four exemption predicates — `is_explicit_column_drop`, `is_constant_imputation`, `is_explicit_missing_indicator`, `is_explicit_hash_encoding` — and each is justified in a comment against that node's own implementation.

The template has nothing comparable. A user who fits a scaler before splitting gets no warning.

### 5.2 `config_validation.py`

Pydantic with `ConfigDict(extra="allow", strict=True)` — validates pipeline structure while passing node-specific params through untouched. Adds `get_close_matches` typo hints so `"standerd_scaler"` produces a suggestion rather than a `KeyError` three steps later.

The template validates config by crashing at runtime.

### 5.3 The Databricks seams already exist

Three abstract base classes are already in place and unimplemented:

| File | ABC | Default impl |
|---|---|---|
| `core/compute.py` | `ComputeBackend` | `LocalComputeBackend` |
| `core/serialization.py` | `ModelSerializer` | `JoblibModelSerializer` |
| `core/model_registry.py` | `ModelRegistry` | `InMemoryModelRegistry` |

Their docstrings say, verbatim, that they are an "additive, non-breaking seam ahead of the Databricks/MLflow phases." And `engines/registry.py` lines 61–67 already maps `pyspark -> spark`.

**This is why the MLflow work is cheap.** The interfaces were designed for it.

### 5.4 Capabilities the template lacks entirely

- `modeling/cross_validation.py` — nested CV with per-fold preprocessing (the correct way to avoid optimistic estimates; the template has no CV at all)
- `profiling/drift.py` — `DriftCalculator` plus Pydantic result models
- `profiling/expect.py` — `expect_no_nulls`, `expect_value_range`, `expect_unique`, `expect_columns_exist`

The template delegates drift entirely to Lakehouse Monitoring, which means it cannot detect drift before deployment, only after.

---

## 6. Code review — weaknesses in skyulf-core

Ordered by severity.

### 6.1 `pipeline.py:42` — `_HARDCODED_MODEL_MAP` duplicates the registry

There are two sources of truth for model dispatch. Worse, the registry lookup swallows `ValueError` at `debug` level:

```python
try:
    return NodeRegistry.get(name)
except ValueError:
    logger.debug(...)      # a registration bug is now invisible
    return _HARDCODED_MODEL_MAP[name]
```

Two consequences: a genuine registration bug silently falls back instead of failing, and `pipeline.py` is forced to import four concrete model classes at module scope, coupling the orchestrator to specific estimators.

**Fix:** delete the map, register the four models properly, let `ValueError` propagate.

### 6.2 `engines/protocol.py:38-47` — `SkyulfDataFrame` verifies nothing

```python
class SkyulfDataFrame(Protocol):
    ...
    def __getattr__(self, name: str) -> Any: ...
```

A protocol with `__getattr__(name) -> Any` is structurally satisfied by *every object in Python*. The type checker will accept an `int`. The annotation reads as a contract but enforces nothing.

The docstring also over-promises Spark and Dask support that does not exist.

**Fix:** either enumerate the methods actually used (`columns`, `shape`, `select`, …) or be honest and use a `TypeAlias` union of the two concrete frame types.

### 6.3 Process-wide mutable singletons without locks

`set_compute_backend`, `set_model_serializer` and `EngineRegistry.set_active_engine` mutate module-level state with no synchronisation. `NodeRegistry` and `InMemoryModelRegistry` both use a `Lock`; these three don't.

Practical effect: tests become order-dependent, and on a Databricks cluster (where multiple notebook cells and threads share an interpreter) this is a live hazard.

**Fix:** either add locks, or better, make these context managers so scope is bounded:

```python
with compute_backend(SparkComputeBackend(spark)):
    ...
```

### 6.4 `pipeline.py:56-66` — pickle-based reproducibility digest

```python
def _artifact_digest(artifacts) -> str:
    try:
        blob = pickle.dumps(artifacts)
    except Exception:
        blob = repr(artifacts).encode()
    return hashlib.sha256(blob).hexdigest()
```

Two problems. `pickle.dumps` output is **not stable across numpy/sklearn versions** — the same fitted state produces a different digest after a dependency bump, so the "seal" fires false alarms. And the `repr()` fallback can collide (two different arrays with the same truncated repr).

**Fix:** digest the structured artifact mappings directly — sorted keys, canonical JSON for scalars, `array.tobytes()` plus `dtype` and `shape` for arrays. Deterministic across versions and collision-resistant.

### 6.5 `pipeline.py` is doing too much

579 lines covering: fitting orchestration, Mermaid diagram escaping, artifact digesting, pandas conversion, and save/load. Mermaid rendering in particular has no business being in the pipeline module.

**Fix:** extract `visualization.py` and `persistence.py`. Low risk, immediate clarity gain.

---

## 7. The measured cost of adding Spark

This is the quantitative core of the recommendation.

`preprocessing/dispatcher.py` explicitly owns engine branching — which sounds like the right place to add an engine. But look at the signatures:

```python
# line 102
def apply_dual_engine(df, params, polars_func, pandas_func): ...

# line 153
def fit_dual_engine(df, config, polars_func, pandas_func): ...
```

The engine set is **positional and hardcoded to exactly two**. And the body (lines 139–147) branches:

```python
if engine.name == EngineName.POLARS:
    return polars_func(...)
else:
    X_pd = X.to_pandas()   # anything not Polars is silently collected
    return pandas_func(X_pd, ...)
```

A Spark frame reaching this code would be silently collected to the driver. Not an error — a performance cliff.

### What a third engine would actually cost

| Measurement | Count |
|---|---|
| Files using `apply_dual_engine` / `fit_dual_engine` | **49** |
| Call sites | **126** |
| Node files importing pandas or polars directly | 73 of 125 |
| — importing polars | 57 |
| — importing pandas | 59 |
| — engine-agnostic | 52 |
| `pl.` references | 255 across 59 files |
| `pd.` references | 350 across 59 files |
| `.iloc[` / `.loc[` usages | 27 across 12 files |

Those 27 `.iloc`/`.loc` usages are the killer: **`.iloc` has no Spark equivalent.** Spark DataFrames have no positional index. Each one needs redesign, not translation.

**Conclusion:** adding a Spark engine is O(number of nodes) — 87 preprocessing files, 38 modeling files — not O(1) interface implementation. And it buys a capability the target platform doesn't use for training.

### What to do instead

```python
# The boundary, not the engine
spark_df = spark.table("catalog.schema.features")
arrow_tbl = spark_df.toArrow()          # or via toPandas()
df = pl.from_arrow(arrow_tbl)           # zero-copy into Polars
result = pipeline.fit(df, config)       # skyulf-core, unchanged
```

One converter. Zero node changes. This is exactly what the template already does with `.toPandas()`, only faster.

---

## 8. Layer-by-layer replacement analysis

| Template layer | skyulf-core covers it? | Notes |
|---|---|---|
| `ConfigService` (OmegaConf merge, suffix resolution) | ❌ No | Databricks-specific; keep. Could be validated by skyulf's Pydantic approach. |
| `DataService` (UC read/write) | ❌ No | Needs Spark; keep as adapter. |
| `FeatureStoreService` | ❌ No | Databricks Feature Engineering client; keep. |
| Feature engineering (`compute_features`) | ⚠️ Partial | skyulf preprocessing can express most of it, but UC-scale aggregation belongs in Spark. |
| Train/test splitting | ✅ Yes | And skyulf's splitters carry `is_splitter` metadata the leakage guard uses. |
| Preprocessing / encoding / scaling | ✅ Yes — better | 87 node files vs. the template's ad-hoc inline transforms. |
| Model fitting | ✅ Yes | Same sklearn/XGBoost estimators underneath. |
| Hyperparameter tuning | ✅ Yes — better | Nested CV; the template has none. |
| Metrics / evaluation | ✅ Yes | |
| Model validation gates | ⚠️ Partial | Thresholds exist; the "compare against current champion" logic is MLflow-dependent. |
| MLflow logging | ❌ **Gap** | `ModelSerializer` seam exists, unimplemented. |
| UC model registry | ❌ **Gap** | `ModelRegistry` seam exists, unimplemented. |
| Batch inference | ⚠️ Partial | Scoring works; UC write does not. |
| Serving endpoint deployment | ❌ No | Databricks API; keep. |
| Lakehouse Monitoring | ⚠️ Overlap | `profiling/drift.py` overlaps but doesn't replace the managed monitor. |
| Leakage protection | ✅ **skyulf only** | Template has none. |
| Config validation | ✅ **skyulf only** | Template has none. |

**Roughly 60% replaceable.** The remaining 40% is genuinely Databricks-platform work and should stay in a thin adapter layer.

---

## 9. Roadmap

### Phase 1 — MLflow (highest value, lowest cost)

Implement the two existing ABCs. No node changes.

```python
# skyulf/integrations/mlflow_serializer.py
from skyulf.core.serialization import ModelSerializer

class MLflowModelSerializer(ModelSerializer):
    """Persist fitted pipelines as MLflow runs instead of joblib files."""

    def __init__(self, experiment: str) -> None:
        mlflow.set_experiment(experiment)

    def save(self, pipeline, path: str) -> str:
        with mlflow.start_run() as run:
            mlflow.log_params(pipeline.config.flat_params())
            mlflow.log_metrics(pipeline.metrics)
            mlflow.log_dict(pipeline.artifacts_summary(), "artifacts.json")
            mlflow.sklearn.log_model(pipeline.estimator, "model")
            return run.info.run_id

    def load(self, run_id: str):
        return mlflow.sklearn.load_model(f"runs:/{run_id}/model")
```

```python
# skyulf/integrations/uc_registry.py
from skyulf.core.model_registry import ModelRegistry

class UnityCatalogModelRegistry(ModelRegistry):
    """Register models into Unity Catalog (three-level namespace)."""

    def __init__(self, catalog: str, schema: str) -> None:
        mlflow.set_registry_uri("databricks-uc")
        self._prefix = f"{catalog}.{schema}"

    def register(self, name: str, run_id: str) -> int:
        mv = mlflow.register_model(f"runs:/{run_id}/model", f"{self._prefix}.{name}")
        return int(mv.version)

    def set_alias(self, name: str, alias: str, version: int) -> None:
        mlflow.MlflowClient().set_registered_model_alias(
            f"{self._prefix}.{name}", alias, version
        )
```

**Estimate:** 200–300 LOC including tests. Highest return in the whole roadmap.

### Phase 2 — `skyulf-databricks` I/O adapter (separate package)

Keep `skyulf-core` free of a Databricks dependency. A separate package holds the boundary:

```python
def read_uc_table(spark, full_name: str) -> pl.DataFrame:
    """Read a Unity Catalog table into Polars via Arrow (zero-copy)."""
    return pl.from_arrow(spark.table(full_name).toArrow())

def write_uc_table(spark, df: pl.DataFrame, full_name: str, mode: str = "overwrite") -> None:
    """Write a Polars frame back to Unity Catalog."""
    spark.createDataFrame(df.to_arrow().to_pandas()).write.mode(mode).saveAsTable(full_name)
```

**Estimate:** ~150 LOC. Replaces the template's `DataService`.

### Phase 3 — Fix the digest and the model map

- Replace `_artifact_digest`'s pickle with structured, version-stable hashing (§6.4)
- Delete `_HARDCODED_MODEL_MAP`; register the four models; let `ValueError` propagate (§6.1)
- Tighten `SkyulfDataFrame` or downgrade it to a `TypeAlias` (§6.2)
- Make backend setters context managers (§6.3)

**Estimate:** ~1 day. Removes the four highest-risk defects.

### Phase 4 — Optional: `SparkComputeBackend`

*This is the only legitimate use of Spark here.* Not a dataframe engine — a parallel-execution backend for hyperparameter search:

```python
class SparkComputeBackend(ComputeBackend):
    """Fan out independent fit tasks across a Spark cluster."""

    def __init__(self, spark) -> None:
        self._sc = spark.sparkContext

    def map(self, fn, items):
        return self._sc.parallelize(list(items)).map(fn).collect()
```

Each task still runs on Polars/pandas on its executor. **Zero node changes required.** This is the correct way to get Spark's value without paying the engine cost.

### Explicitly not recommended

- ❌ A Spark dataframe engine in `engines/` — see §7
- ❌ Spark ML (`pyspark.ml`) estimators — the template doesn't use them, and they'd fork the modeling module

---

## 10. Decision summary

| Question | Answer |
|---|---|
| Is skyulf-core better code than the template's Python layer? | **Yes**, clearly — architecture, testing, safety. |
| Can it replace the template? | **Partially — ~60%.** The Databricks-native 40% must stay. |
| Does it need MLflow? | **Yes, and it's cheap.** The seams are already there. Do this first. |
| Does it need PySpark as a dataframe engine? | **No.** 126 call sites, 27 `.iloc` usages, and the template doesn't train on Spark anyway. |
| Does it need Spark at all? | Only at the I/O boundary (Phase 2) and optionally as a compute backend for tuning (Phase 4). |
| What's the single highest-value next step? | `MLflowModelSerializer` + `UnityCatalogModelRegistry`. |

**The strategic shape:**

```
┌──────────────────────────────────────────────┐
│  Databricks Asset Bundle (jobs, clusters)    │  ← keep from template
├──────────────────────────────────────────────┤
│  skyulf-databricks  (UC I/O, Feature Store)  │  ← Phase 2, new
├──────────────────────────────────────────────┤
│  skyulf-core  (preprocess, model, CV, drift) │  ← already exists
│    + MLflow serializer / UC registry         │  ← Phase 1
└──────────────────────────────────────────────┘
```

The template becomes what it should have been: orchestration and infrastructure. The ML logic lives in a tested, versioned library instead of being copy-pasted across four bundle folders.

---

## 11. Appendix — reproducing the measurements

All commands were run from the respective repository roots.

```bash
# skyulf-core: source and test size
find skyulf -name '*.py' | wc -l
find skyulf -name '*.py' -exec cat {} + | wc -l
find tests  -name '*.py' | wc -l
find tests  -name '*.py' -exec cat {} + | wc -l

# per-module LOC
for m in preprocessing modeling profiling core engines data; do
  echo -n "$m "; find "skyulf/$m" -name '*.py' -exec cat {} + | wc -l
done

# dual-engine dispatch surface
grep -rl 'apply_dual_engine\|fit_dual_engine' skyulf | wc -l    # 49 files
grep -rn 'apply_dual_engine\|fit_dual_engine' skyulf | wc -l    # 126 call sites

# direct engine coupling
grep -rl 'import polars' skyulf/preprocessing skyulf/modeling | wc -l   # 57
grep -rl 'import pandas' skyulf/preprocessing skyulf/modeling | wc -l   # 59
grep -rn '\.iloc\[\|\.loc\[' skyulf | wc -l                             # 27

# template: duplication
find template -type f | wc -l
md5 template/*/*/common/feature_store_service.py.tmpl
grep -rn 'new_cluster' template --include='*.tmpl' | wc -l
grep -rno '{{ `"{{job\.parameters' template | wc -l                      # 89

# the pivotal evidence: no Spark training
grep -n 'toPandas' template/*/*/src/training/notebooks/train.ipynb.tmpl
grep -n 'toPandas' template/*/*/src/validation/model_validation.py.tmpl
```

### Citation index

| Claim | Location |
|---|---|
| Template collects to pandas before fitting | `src/training/notebooks/train.ipynb.tmpl:376-431` |
| Validation does the same | `src/validation/model_validation.py.tmpl:129,136` |
| Dual-engine signature hardcoded to two engines | `skyulf/preprocessing/dispatcher.py:102,153` |
| Non-Polars silently collected to pandas | `skyulf/preprocessing/dispatcher.py:139-147` |
| Leakage guard fails closed | `skyulf/leakage.py:155-157` |
| Redundant model map | `skyulf/pipeline.py:42` |
| Pickle-based digest | `skyulf/pipeline.py:56-66` |
| Model estimator init | `skyulf/pipeline.py:163-186` |
| Over-permissive dataframe protocol | `skyulf/engines/protocol.py:38-47` |
| `pyspark -> spark` mapping already present | `skyulf/engines/registry.py:61-67` |
| Unimplemented Databricks seams | `skyulf/core/compute.py`, `core/serialization.py`, `core/model_registry.py` |

### Caveats

- Line counts are raw `wc -l` (includes blanks and comments).
- skyulf-core's test suite was **not executed** during this assessment; files were read and counted, not run.
- The template was analysed as `.tmpl` sources plus the four generated bundle copies.
