# Preprocessing inference context: coverage and remaining work

Status snapshot: **2026-10-09**, branch `093`, Task196 dtype parity repairs.
This is a tracked continuation checklist. Update it
when a family gains a reviewed declaration or new validation evidence; do not
treat a passing sample report as closing an entire family.

The generated Bundle contains `PREPROCESSING_CONTEXT.md`: it explains `row`,
`group`, `window`, `global`, `unknown`, the difference between training-time and
request-time aggregation, and a complete `artifact`/`sample`/`report` example.
Its source is
`skyulf-core/templates/databricks/template/{{.project_name}}/PREPROCESSING_CONTEXT.md`.
For related published guides, see [preprocessing placement](../user_guide/preprocessing_placement.md)
and [inference flow](../user_guide/inference_flow.md).

## What has actually been delivered?

- The diagnostic calls existing saved appliers. It does not duplicate transforms
  or fit new state. Shared schema/config checks delegate node-specific state
  inspection to the preprocessing owner.
- It compares full, repeated, chunked, reversed and empty input; detects output
  differences and input/state mutation; reports normal prediction-time skips.
- Context metadata uses existing `ExecutionCapability` declarations or an
  optional applier-owned hook. Custom function builders accept an optional
  `inference_context` declaration, stored with their saved parameters.
- Captured custom functions/classes survive a fresh process without executing
  editable recipe builders. Unknown custom code can pass samples while remaining
  unadmitted for Spark/REST.
- No automatic Bundle probe task, context-aware group/history executor, or new
  distributed/REST admission was added. The existing local fit/apply path remains.

## Count definitions

| Measurement | Count | Interpretation |
| --- | ---: | --- |
| Registered transformer IDs | 67 | Includes aliases and training/inspection helpers |
| Distinct applier implementations | 63 | Four additional names refer to existing classes |
| Implementations with context declaration machinery | 23 | Twenty built-in implementations plus three custom wrappers |
| Remaining without declaration machinery | **40** | **44 registered IDs** after including aliases; listed below |
| Reviewed worker-admitted preprocessing families | 7 | Only their supported pandas configurations; not every mode/engine |

The count of 23 describes available machinery, not unconditional support for
every configuration. Existing worker-subset validators can abstain for valid
local states. Some Polars configurations also have no matching declaration.
Custom wrappers return `unknown` unless their saved definition explicitly
declares context. New project-defined classes are not part of the fixed 63.

### Existing declarations

| Implementation | Present context behavior | Remaining boundary |
| --- | --- | --- |
| `SimpleImputer` | `row` for matched reviewed declarations | Review additional local modes/engines independently |
| `GroupImputer` | `row` for saved group lookups in reviewed declarations | Verify fallback/null-group modes independently |
| `StandardScaler` | `row` for matched reviewed declarations | Exact Polars chunk arithmetic may differ by rounding |
| `MinMaxScaler` | `row` for matched reviewed declarations | Boundaries, clip modes and engines remain configuration-specific |
| `OneHotEncoder` | `row` for matched reviewed declarations | A local default such as `max_categories=20` may report `unknown` |
| `FeatureInteraction` | `row` for matched reviewed declarations | Other feature-generation classes have separate entries below |
| `ClipValues` | `row` for matched reviewed declarations | Preserve configured limits and dtype behavior |
| `LagFeatures` | `window`, with its saved row effect | No independent-chunk probe or new history execution |
| `RollingAggregate` | `window` | No independent-chunk probe or new history execution |
| `Deduplicate` | `global` / filter | Normal row-preserving prediction skips it |
| `ColumnFunction` | Explicit saved declaration, otherwise `unknown` | Validate each custom implementation and captured dependencies |
| `FittedFunction` | Explicit saved declaration, otherwise `unknown` | Validate saved-state reuse and no mutation per implementation |
| `RowFilterFunction` | Explicit saved declaration, otherwise `unknown` | Prediction skip and fixed pre-split eligibility are separate contracts |

### Ten additional local contracts (Task195)

These owners declare `row` for inspected pandas/Polars state and validate their
saved apply parameters through `validate_inference_state`. Existing apply code
is reused. None of these additions grants Spark-worker or endpoint admission.
The scope column is part of the result, not an unconditional parity promise.

| ID | Implementation | Inspected scope and remaining boundary |
| --- | --- | --- |
| PC-02 | `CustomBinning` | Fixed edges, ordinal/bin-index/range labels, missing policies, empty no-ops. Pandas numeric labels retain nullable `Int64`; missing-label mode retains object dtype, including singleton and empty requests. |
| PC-08 | `ValueReplacement` | Fixed scalar/list/tuple and dictionary rules, nulls, typed keys and mapping precedence. Pandas object input stays object. Numeric widening can still change dtype across chunks; arbitrary mapping objects/Series are outside inspection. |
| PC-09 | `DropMissingColumns` | Reuse saved dropped columns even when request missingness differs. Training threshold is reporting metadata, not recomputed. |
| PC-11 | `MissingIndicator` | Saved columns and nonempty string suffix, null/NaN flags, no-op and empty frames. Non-string suffix objects are outside inspection. |
| PC-21 | `CorrelationThreshold` | Saved drop list and enabled/disabled flag; no request correlation calculation. |
| PC-23 | `UnivariateSelection` | Saved candidate/selected columns, no-target artifact and empty selection. Scoring methods without p-values now retain an empty p-value report. |
| PC-24 | `VarianceThreshold` | Frozen selection with constant/all-missing requests, empty selection and saved undefined variance metadata. |
| PC-28 | `MaxAbsScaler` | Saved scale/max-abs vectors, zeros/constants/nulls and empty no-op. Polars nonbinary division can differ by one ULP between full and singleton evaluation. |
| PC-29 | `RobustScaler` | Saved center/scale/quantiles and all flag combinations, null statistics and empty no-op. Exact numerical parity remains sample/runtime-specific. |
| PC-30 | `GeoDistance` | Saved coordinate/output names, both distance methods and units. Existing invalid-coordinate behavior is reported without adding new geospatial validation. |

Task196 fixes PC-02 numeric-label dtypes in the shared binning applier and PC-08
object downcasting. This deliberately changes pandas ordinal output from NumPy
`int64`/`float64` to nullable `Int64`. An object column stays object even when all
replacement values happen to be numeric. If a later step requires numeric input,
configure an explicit `Casting` step instead of relying on batch-dependent
inference. For example, object-backed numbers containing `None` need a numeric
cast before binning. No new artifact fields or transform implementations are needed.
Refit the whole pipeline when upgrading older binning models with downstream
string-key encoders. In particular, legacy `LabelEncoder`/`OrdinalEncoder` keys
such as `"0.0"`, `"1.0"`, `"nan"` do not match the new integer/null representation;
replaying those older encoders can silently select their unknown-category code.
At the sklearn boundary, nullable numeric missing values become `np.nan` even
beside object features; other column values and large integer precision remain.

Keep **PC-08 numeric widening** and **PC-28 Polars arithmetic** open. For PC-08,
replacing `1` with `0.5` in an integer column widens matched chunks to float while
unmatched chunks stay integer. Use a consistent float input dtype for that rule;
the diagnostic continues reporting the integer-input mismatch.
For PC-28, a fitted scale of 5 yielded `6 / 5 = 1.2000000000000002` in the full
Polars result and `1.2` in a singleton, a difference of `2.22e-16`. The diagnostic
retains `output_mismatch`; no tolerance was added to hide it. These follow-ups
are additional to the 40 implementations without declarations. The Polars
rounding case remains a strict diagnostic boundary; it does not justify a second
arithmetic implementation or a weaker comparison.

## Remaining 40 implementations without declarations

Every row below is **OPEN**. The “review focus” is a work item, not a certified
context declaration. Start with P1, then P2; P3 covers training/inspection
semantics, and P4 needs external model packaging work. This ordering does not
imply permission to expand worker admission.

| ID | Priority | Registered names sharing this implementation | Review focus before closing |
| --- | --- | --- | --- |
| PC-01 | P2 | `GeneralBinning` | Resolve each mode's fitted bins, labels, boundary/null behavior |
| PC-03 | P2 | `KBinsDiscretizer` | Saved estimator/bins; chunk-invariant encoding and no refit |
| PC-04 | P1 | `Casting` | Invalid casts, nullable dtypes and schema consistency |
| PC-05 | P1 | `AliasReplacement` | Fixed mapping, unknown/null inputs, output type |
| PC-06 | P1 | `InvalidValueReplacement` | Rule/fallback behavior without request-derived state |
| PC-07 | P1 | `TextCleaning` | Unicode, nulls and deterministic per-row text rules |
| PC-10 | P3 | `DropMissingRows` | Row-filter context plus existing prediction-skip evidence |
| PC-12 | P2 | `DummyEncoder` | Frozen output columns, unseen categories, no batch vocabulary |
| PC-13 | P2 | `HashEncoder` | Stable hash/configuration, output width and null handling |
| PC-14 | P2 | `LabelEncoder` | Saved mapping and unknown/null policies |
| PC-15 | P2 | `OrdinalEncoder` | Saved category order, sentinels and output dtype |
| PC-16 | P2 | `TargetEncoder` | Fold-trained mappings; no target/request-time learning |
| PC-17 | P2 | `WOEEncoder` | Saved mappings/smoothing, unseen values and class assumptions |
| PC-18 | P1 | `DateFeatures` | Time zones, invalid dates and fixed feature schema |
| PC-19 | P2 | `FeatureGenerationNode`, `FeatureMath`, `FeatureGeneration` | Review each mode; distinguish saved group-aggregation lookup from live grouping |
| PC-20 | P1 | `PolynomialFeaturesNode`, `PolynomialFeatures` | Saved terms, names/order and exact chunk behavior |
| PC-22 | P2 | `ModelBasedSelection` | Frozen selection mask, loaded estimator and dependencies |
| PC-25 | P2 | `feature_selection` | Delegate context/validation to the selected nested implementation |
| PC-26 | P2 | `IterativeImputer` | Saved estimators, randomness, repeated transforms and no refit |
| PC-27 | P2 | `KNNImputer` | Saved training neighbors, batching behavior and bounded state/package |
| PC-31 | P2 | `H3Index` | Optional dependency pins, resolution and invalid coordinates |
| PC-32 | P3 | `DatasetProfile` | Separate inspection side effects from effective prediction apply |
| PC-33 | P3 | `DataSnapshot` | Separate snapshot behavior from effective prediction apply |
| PC-34 | P2 | `EllipticEnvelope` | Saved detector, filter effects and inference row preservation |
| PC-35 | P2 | `IQR` | Saved thresholds; filter/change behavior and row alignment |
| PC-36 | P1 | `ManualBounds` | Fixed bounds, row effects and prediction contract |
| PC-37 | P1 | `Winsorize` | Saved limits; do not recompute quantiles from a request |
| PC-38 | P2 | `ZScore` | Saved mean/std; filter behavior, nulls and row alignment |
| PC-39 | P3 | `Oversampling` | Training-only skip; do not generate rows during prediction |
| PC-40 | P3 | `Undersampling` | Training-only skip; do not remove prediction requests |
| PC-41 | P3 | `TrainTestSplitter`, `Split` | Unrecorded training markers/alias; no inference splitting |
| PC-42 | P3 | `feature_target_split` | Training marker and target separation; no prediction execution |
| PC-43 | P2 | `GeneralTransformation` | Resolve each operation and saved parameters, including invalid domains |
| PC-44 | P2 | `PowerTransformer` | Saved transform parameters, domains and loaded estimator |
| PC-45 | P1 | `SimpleTransformation` | Fixed formula/settings, numerical domains and schema |
| PC-46 | P2 | `count_vectorizer` | Frozen vocabulary/output schema; dense output width/memory bounds and package replay |
| PC-47 | P2 | `hashing_vectorizer` | Stable output width/hash and supported frame representation |
| PC-48 | P4 | `sentence_embedder` | Pin/save actual model assets/revision; verify fresh-process/offline replay |
| PC-49 | P2 | `tfidf_vectorizer` | Saved vocabulary/IDF; no request-time vocabulary/IDF fitting |
| PC-50 | P2 | `tokenizer` | Null/text handling, saved analyzer settings, fixed output names/count dtypes and pandas/Polars fallback parity |

Some entries above are intentionally skipped during prediction rather than
remotely executed. Their completion needs accurate skip evidence, not a forced
`row` label. Context-dependent operators may be correctly complete with
`requires_context`; a separate executor would need its own specification/tests.

## Definition of done for each row

1. Inspect the actual saved `apply` path and each relevant mode/engine. Distinguish
   training statistics from request-time data needs. Document unsupported modes.
2. Put the declaration and any step-specific fitted-state validation in the
   existing preprocessing owner. Reuse common schema/package checks and the
   existing transform implementation; no second copy of its formula.
3. Fit once, then exercise saved apply with fit disabled. Compare full/chunked/
   reordered requests where semantically valid; verify required group/history
   behavior where independent chunks are invalid.
4. Cover nulls, unseen categories, empty input, constant columns, schema/order,
   state/input mutation and actual supported pandas/Polars modes. Account for
   numerical rounding explicitly without hiding material differences.
5. Save/reload in a fresh process. Verify required code, dependencies and external
   assets are present; do not rely on an editable project file or runtime cache.
6. If worker admission is separately expanded, add its strict identity/config/
   state checks and native runtime validation. A diagnostic result alone is not
   admission. Keep ordinary local behavior and skip semantics unchanged.
7. Run focused affected tests and CI static scopes, review the change, then update
   this row with engine/configuration scope, test evidence and commit reference.

### Cross-cutting work not included in the 40 count

- [ ] Decide whether/how to expose the diagnostic as an optional training/release
  check. It is currently an explicit library call; no automatic job was added.
- [ ] Specify any future group/window executor, including complete-group keys,
  ordering, late rows and history continuation. The probe does not build one.
- [ ] Expand reviewed engine/configuration coverage within the already declared
  families. A family count is not a matrix of every supported mode.
- [ ] Review each project's custom declarations, packaging and sample coverage.
  The three wrappers cannot certify arbitrary user code.

## Reproduce the inventory

Run from the repository environment; this reads registrations and does not train,
score, collect Spark data or contact Databricks:

```python
import warnings
import skyulf.preprocessing  # register built-in nodes
from skyulf.registry import NodeRegistry

warnings.simplefilter("ignore", DeprecationWarning)
groups = {}
for name in NodeRegistry.list_transformers():
    owner = NodeRegistry.get_applier(name)
    groups.setdefault(owner, []).append(name)

remaining = []
for owner, names in groups.items():
    calculator = NodeRegistry.get_calculator(names[0])
    has_hook = "inference_capability" in vars(owner)
    has_declarations = bool(vars(calculator).get("__execution_capabilities__", ()))
    if not (has_hook or has_declarations):
        remaining.append(names)

print("Registered names:", len(NodeRegistry.list_transformers()))
print("Distinct implementations:", len(groups))
print("Remaining implementations:", len(remaining))
for names in remaining:
    print(", ".join(names))
```

Expected snapshot: 67 names, 63 implementations, 40 remaining implementations.
This counts declaration machinery, not whether a fitted configuration passes
`get_inference_capability` or the worker certificate.

## Validation record

Task191 added 82 context cases and 48 probe/reload cases; the coordinated affected
test union contains 603 distinct cases. It covers built-in saved-state reuse,
custom function/class replay, no refit, differing batch statistics, row/order
changes, pandas metadata aliasing, mutable numpy cells, validator mutation,
engine-context rejection and preservation of strict worker checks. Independent
review findings were reproduced, repaired and checked. At that stage these local
checks left all 50 then-open rows unchanged; native evidence is recorded below.

Task192 reran the explicit affected union: **603 passed**, with 42 existing
warnings from temporal CV, feature names, deprecated aliases and small fixtures.
The complete guide fit/save/load/probe example and its custom factory example
also executed successfully. The report's 50 rows were checked against the live
registry: all 54 remaining names match exactly, including aliases.
The commit also includes the Task191 implementation/tests; future coverage
changes should update this tracked report with fresh evidence.

### Native Databricks verification

On 2026-10-08, Task193 tested runtime commit `a50ddd8b` on Databricks serverless
STANDARD with the `skyulf` profile. Run
[55789884205809](https://dbc-45604623-c18b.cloud.databricks.com/jobs/801276331272715/runs/55789884205809)
finished **TERMINATED / SUCCESS**: **213 passed**, zero failures or skips, in
21.03 seconds of pytest execution. Two expected warnings came from R² metrics
on one-row test fixtures; environment startup time is separate.

The five explicit files were `test_preprocessing_probe.py`,
`test_preprocessing_inference_context.py`, `test_preprocessing_probe_project.py`,
`test_preprocessing_fitted_validation.py` and `test_partition_safety.py`.
They verify saved-state reuse with fit disabled, captured custom replay in fresh
processes, context requirements, mutation/batch-dependence detection and the
unchanged strict partition admission checks. The installed wheel's 552 Python
files matched the tested repository source hashes.

The complete main-template guide example also executed successfully: saved
`GroupImputer` and `OneHotEncoder` reported `row / passed` for full, repeated,
single-row, chunked, reversed and empty inputs. Runtime: Python 3.12.3, pandas
2.2.3, Polars 1.44.2, NumPy 2.1.3 and scikit-learn 1.8.0.

This was a native notebook test of the Python fit/apply and diagnostic paths;
it did not execute a Spark UDF or deploy a REST endpoint. No Unity Catalog tables,
registered models or endpoints were created. It adds runtime evidence for the
existing implementation and left all 50 then-open backlog rows unchanged.

Task194 then exercised the actual inference routes on the same runtime source:
[run 537211044950378](https://dbc-45604623-c18b.cloud.databricks.com/jobs/927702146751887/runs/537211044950378)
finished **SUCCESS**. One freshly fitted and registered random-forest pipeline
used six preprocessing steps: two SimpleImputers, GroupImputer,
FeatureInteraction, StandardScaler and OneHotEncoder. All six probe steps passed.
Eight scoring rows included numeric/category nulls, unseen categories and keys
above 2^53. Actual Spark UDF execution used two partitions, prediction batch sizes
1 and 3, and explicit `env_manager=local`; REST and the typed UC `ai_query`
function used the same registered model version. Every route matched all three
local output columns (`prediction`, `probability_0`, `probability_1`) with maximum
absolute error **0.0**. The temporary endpoint was deleted after testing; the
existing example endpoint kept its original identity and model version. This is
a small correctness check, not load/latency testing or wider node admission.

### Task195 validation: ten additional local context contracts

On 2026-10-09 the deduplicated union of 25 explicit affected test files passed:
**1,342 passed**, 68 expected warnings, no failures. Full repository Ruff,
formatting (1,606 files), the complete CI Ty scope and Lizard CCN <= 10 passed.
Six independent review findings were reproduced and repaired, including NumPy
scalar/boolean state compatibility, tuple replacement rules, legacy bin labels
and internally conflicting generated column names. Existing transform formulas
were retained; the separate univariate missing-p-value reporting crash was fixed.

The final wheel was verified on Databricks serverless PERFORMANCE_OPTIMIZED,
using the previously selected `skyulf` profile:
[run 585541621081287](https://dbc-45604623-c18b.cloud.databricks.com/jobs/903098952257050/runs/585541621081287)
finished **TERMINATED / SUCCESS**. **610 tests passed**, zero failures or skips,
plus the complete Bundle guide example. All 552 installed Python source files
matched the final source manifest. Wheel SHA256:
`3e8a5eac4c8dc17e24164a3228a0611aca28308ea114a4e40027d2d80adf65c5`.

Fresh-process tests fit/save ten actual pipelines per engine, then disable every
involved calculator before loading and probing their artifacts. They cover
pandas and Polars, frozen selections/statistics, full/singleton/chunk/reverse/
empty execution, mutation detection and unchanged partition rejection. Separate
tests retain the known exact-parity failures listed above. These are native
Python validation results; this batch did not admit the ten nodes to Spark UDF,
REST or `ai_query`, and did not create tables, registered models or endpoints.

### Task196 validation: stable pandas dtypes and model prediction

On 2026-10-09, **885 local tests passed**: 768 tests across ten explicitly
selected affected files, plus 117 temporal model/bridge consumer tests. Full
Ruff/format, CI Ty scope and Lizard CCN <= 10 passed. Independent review found
and reproduced the mixed object/nullable-bin sklearn regression; its repair
passed 19 additional precision, sentinel, mutation and empty-input probes.

The final installed wheel passed **430 tests**, zero failures or skips, plus
the runnable Bundle guide on Databricks serverless:
[run 535767378226162](https://dbc-45604623-c18b.cloud.databricks.com/jobs/615731771938126/runs/535767378226162)
finished **TERMINATED / SUCCESS**. All 552 installed runtime files matched source.
Wheel SHA256: `dfd48304f50755d5dc951384bbf85d858c6119478e067aad6f5fb6abe8de1eec`.
The saved-model regression disables fit in a fresh process and checks real
random-forest predictions for full and singleton requests, including missing
and out-of-range bins. This is native Python fit/apply/predict evidence; no new
Spark UDF, REST or `ai_query` admission or route test was added. The remaining
40 undeclared implementations and two documented parity boundaries stay open.
