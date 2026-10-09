# Preprocessing inference context: coverage and remaining work

Status snapshot: **2026-10-09**, branch `093`, Task199 feature/text/detector contexts.
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
| Implementations with context declaration machinery | 55 | Fifty-two built-in implementations plus three custom wrappers |
| Remaining without declaration machinery | **8** | **9 registered IDs** after including aliases; listed below |
| Reviewed worker-admitted preprocessing families | 7 | Only their supported pandas configurations; not every mode/engine |

The count of 55 describes available machinery, not unconditional support for
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
are additional to the 8 implementations without declarations. The Polars
rounding case remains a strict diagnostic boundary; it does not justify a second
arithmetic implementation or a weaker comparison.

### Eleven additional local contracts (Task197)

These declarations inspect saved state and call the existing applier. They do
not broaden Spark UDF, REST or `ai_query` admission. A `row` declaration describes
required input context; strict sample checks can still expose unsupported dtypes,
null behavior or numerical differences.

| ID | Implementation | Inspected scope and remaining boundary |
| --- | --- | --- |
| PC-01 | `GeneralBinning` | Fixed learned bins and supported label settings; shares validation with the existing binning owners. Numeric custom labels retain the existing Polars apply error. |
| PC-03 | `KBinsDiscretizer` | Saved bin edges for uniform, quantile and k-means strategies; inference does not recompute bins. |
| PC-04 | `Casting` | Frozen categories and supported scalar conversions. Legacy pandas categories need the whole request; coercive fallback casts can also depend on invalid neighboring values. Integer output dtypes can differ across chunks. |
| PC-05 | `AliasReplacement` | Saved standard/custom mappings, unseen inputs and nulls. Native engine limitations are retained. |
| PC-06 | `InvalidValueReplacement` | Fixed rules and replacement values. Pandas integer-to-null replacement can change the empty output dtype. |
| PC-07 | `TextCleaning` | Saved operation order, nulls, regex and no-op settings. Pandas slash-date normalization followed by trim fails on a null-only chunk; Polars rejects unsupported regex lookbehind. |
| PC-18 | `DateFeatures` | UTC-aware saved states, epochs, time zones, nulls, tuples and NumPy string settings. Legacy non-UTC state stays `unknown`. Existing generated-name overwrites and Polars duplicate-output errors remain visible. |
| PC-20 | `PolynomialFeaturesNode`, `PolynomialFeatures` | Saved degree/flags/columns/order and prefixes. Apply rebuilds sklearn's combinatorial expansion from configuration; it does not learn request statistics. Existing empty/null-input errors remain. |
| PC-36 | `ManualBounds` | Saved fixed/open limits; `row` context with a filter effect. Ordinary prediction runs this step and rejects requests that would lose rows. It is not automatically skipped. |
| PC-37 | `Winsorize` | Saved training quantiles, never request quantiles. Pandas integer input may produce float full output but integer in-bound chunks; exact probing retains the mismatch. |
| PC-45 | `SimpleTransformation` | Eight existing fixed-formula modes and saved settings, including native no-ops. Numeric domains and exceptional scalar types remain native-engine boundaries. |

Local validators preserve supported NumPy settings instead of rewriting them.
Decimal limits/thresholds are accepted where native apply supports them; this does
not add Decimal artifact sealing or transport support. In particular, pandas
exponential clipping with Decimal can still fail for clipped or null values.
Use ordinary numeric settings for sealed models and validate representative
samples. No new mathematical implementation or relaxed comparison was added.

### Ten additional local contracts (Task198)

This group includes operations whose existing execution genuinely depends on
request boundaries. A declaration can correctly report `global` or `unknown`;
neither means the estimator is being trained again.

| ID | Implementation | Inspected scope and remaining boundary |
| --- | --- | --- |
| PC-12 | `DummyEncoder` | Saved vocabulary, output names and drop-first policy. Current category keys are row-local; legacy pandas datetime rendering can depend on neighboring values and reports `global`. |
| PC-13 | `HashEncoder` | Existing stable hash and saved bucket count. Current key version is row-local; legacy pandas rendering reports `global`. Empty-result dtype mismatches remain exact probe failures. |
| PC-14 | `LabelEncoder` | Saved feature/target mappings and unknown/missing codes. Current category keys are row-local; legacy pandas representations can change across batches. Target decoding is separate from feature-only inference. |
| PC-15 | `OrdinalEncoder` | Saved category ordering, sentinels, feature and optional target encoders; no inference category fitting. Legacy pandas feature keys can require `global` context. |
| PC-16 | `TargetEncoder` | Saved full-data category statistics and output class layout. Training cross-fitting deliberately differs from inference lookup; request labels are not used to relearn mappings. |
| PC-17 | `WOEEncoder` | Saved binary-label mappings and defaults; use a genuine two-class training target to exercise this transform. Legacy pandas keys can require `global` context. |
| PC-26 | `IterativeImputer` | Saved initial statistics and chained estimators. Active rounds report `global`: an entirely null request takes sklearn's initial-fill shortcut. Zero-round fits are row-local; posterior sampling or custom prediction estimators remain `unknown`. |
| PC-27 | `KNNImputer` | Saved training neighbors, masks and supported distance/weight settings. Built-in modes are row-local; custom callbacks remain `unknown`. Numerical parity remains configuration/runtime-specific. |
| PC-43 | `GeneralTransformation` | Simple formulas are row-local. Fitted power rules report `global` because one invalid value can cause a whole column to pass through unchanged. |
| PC-44 | `PowerTransformer` | Saved lambdas/scaler settings are reused. Active fits report `global`: an invalid value causes whole-frame fallback. Genuine empty/no-op fits remain row-local. |

For example, a valid value alongside a negative Box-Cox input can remain raw in
the full request while the same value alone is transformed. Preserve request
boundaries and correct invalid input; independent worker chunks do not preserve
that fallback behavior. The diagnostic reports `requires_context` without running
independent chunks. The existing fallback and transform math are unchanged.

IterativeImputer's fitted sklearn objects also store NumPy dtype objects. Their
plain numeric dtype metadata now has a distinct semantic hash encoding, preserving
kind, scalar variant, width and byte order. Structured, metadata-bearing and
nonnumeric dtype
objects remain rejected. This fixes fingerprint/probe inspection; ordinary local
save/load already used a binary payload checksum. Existing array/scalar encodings
and strict worker admission are unchanged.

Some native NumPy pickle aliases normalize on Windows: a `longdouble` dtype can
reload as `float64` where their storage widths coincide. These remain distinct
semantic identities; do not assume fingerprint equality for those normalized
aliases. Ordinary float64 model reload is covered by the saved-model tests.

Use an ordinary Python integer for HashEncoder's `n_features`. A NumPy integer
retains existing runtime limits: NumPy 2 can raise `OverflowError` when a hash
exceeds its signed range; NumPy 1 can promote buckets to float. The latter can
change pandas chunk dtypes or make Polars reject mixed replacement values at
different checks, depending on unique-value order. The diagnostic keeps these
failures visible; it does not normalize fitted settings or weaken comparisons.

### Eleven additional local contracts (Task199)

These owners reuse their existing transformations and saved state. Context
inspection is still separate from worker admission and strict sample parity.

| ID | Implementation | Inspected scope and remaining boundary |
| --- | --- | --- |
| PC-19 | `FeatureGenerationNode`, `FeatureMath`, `FeatureGeneration` | Fixed formulas and training-fitted group lookups. Inference never aggregates the request. Pandas similarity can depend on datetime rendering across rows and reports `global`; unpinned legacy similarity and uncertain fill/epsilon modes stay `unknown`. |
| PC-22 | `ModelBasedSelection` | Saved selected/candidate columns and drop flag. Estimator importance metadata is not recalculated during prediction. |
| PC-25 | `feature_selection` | Delegates validation and context to the concrete fitted selector. Existing unknown-method no-ops remain unchanged. |
| PC-31 | `H3Index` | Saved coordinates/resolution and native output labels. Optional H3 must be installed; pandas empty-output dtype mismatches remain visible. |
| PC-34 | `EllipticEnvelope` | Saved sklearn detector models and learned covariance state. Active models filter rows; null/nonfinite handling remains native. |
| PC-35 | `IQR` | Saved training bounds; no request quartiles. Active bounds filter rows. |
| PC-38 | `ZScore` | Saved means/stds and threshold; no request statistics. Active maps filter rows. |
| PC-46 | `count_vectorizer` | Saved vocabulary, estimator and output layout; no inference vocabulary fit. Native cache mutations and unusual input dtypes remain diagnostic boundaries. |
| PC-47 | `hashing_vectorizer` | Saved hash/analyzer settings and dense output width. A pristine native estimator can lazily mutate its cache on first transform; the diagnostic reports that mutation. |
| PC-49 | `tfidf_vectorizer` | Saved vocabulary/IDF and output width; no request IDF fitting. Optional callbacks remain undeclared. |
| PC-50 | `tokenizer` | Saved analyzer settings, output names and token counts. Native empty token-count dtypes can differ from nonempty output. |

Normal prediction executes the three detectors and rejects row loss. A benign
sample passing the probe does not make filtering safe for arbitrary requests.
Group-aggregate feature generation, by contrast, reads frozen training mappings;
missing or unseen groups keep the existing fallback behavior.

A real model test also exposed a Polars scoring defect: with only a text column
and `drop_original=True`, dropping that last input left a zero-height frame.
The shared attachment helper now returns the generated feature frame when no
input columns remain. This repairs text-only scoring across its consumers while
retaining the existing tokenization, vocabulary and numerical calculations.

Local selector validation now accepts NumPy string column names produced by real
fits, and local bound validation accepts native empty-string column labels.
Neither repair widens portable worker validation. H3 inspection also retains
native `None`/hashable output labels; engine and model-schema restrictions remain.
Feature overwrite and selector drop flags retain native scalar truthiness,
including `None`; metadata inspection does not rewrite them as booleans. Use
explicit YAML `true`/`false` in project recipes to express the intended behavior.
Detector validation also preserves native hashable column labels and supported
Fraction/Decimal settings. Unused fit-report fields do not impose new runtime
restrictions. This does not add sealing or transport support for those types.

H3 is an optional dependency: provision and pin it in the scoring environment.
The local pipeline's core runtime manifest does not automatically record H3's
version. Native verification for this batch explicitly uses `h3==4.5.0`; this
is not an automatic dependency-packaging feature.

## Remaining 8 implementations without declarations

Every row below is **OPEN**. The “review focus” is a work item, not a certified
context declaration. Start with P1, then P2; P3 covers training/inspection
semantics, and P4 needs external model packaging work. This ordering does not
imply permission to expand worker admission.

| ID | Priority | Registered names sharing this implementation | Review focus before closing |
| --- | --- | --- | --- |
| PC-10 | P3 | `DropMissingRows` | Row-filter context plus existing prediction-skip evidence |
| PC-32 | P3 | `DatasetProfile` | Separate inspection side effects from effective prediction apply |
| PC-33 | P3 | `DataSnapshot` | Separate snapshot behavior from effective prediction apply |
| PC-39 | P3 | `Oversampling` | Training-only skip; do not generate rows during prediction |
| PC-40 | P3 | `Undersampling` | Training-only skip; do not remove prediction requests |
| PC-41 | P3 | `TrainTestSplitter`, `Split` | Unrecorded training markers/alias; no inference splitting |
| PC-42 | P3 | `feature_target_split` | Training marker and target separation; no prediction execution |
| PC-48 | P4 | `sentence_embedder` | Pin/save actual model assets/revision; verify fresh-process/offline replay |

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

### Cross-cutting work not included in the 8 count

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

Expected snapshot: 67 names, 63 implementations, 8 remaining implementations.
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

### Task197 validation: eleven more local context contracts

On 2026-10-09, the deduplicated union of 31 explicit affected files passed:
**1,736 passed**, 23 expected warnings, zero failures or skips. Full repository
Ruff, formatting (1,611 files), the complete CI Ty scope and Lizard CCN <= 10
passed. Independent cross-review reproduced and repaired overly narrow saved
state validation and Casting's request-dependent fallback context. Reviewers
then rechecked the original cases without repeating the full affected suite.

Fresh processes load 11 real fitted random-forest pipelines per engine with
every involved preprocessing calculator disabled. Checks compare saved apply,
real full/singleton model predictions and unchanged fitted-state hashes. They
retain ManualBounds row-loss rejection, Polynomial empty-input limitations,
and strict worker rejection for these newly declared owners. New declarations
bring coverage to **34 of 63 implementations**, with **29 remaining**; configured
boundaries in the scope table remain open.

The same final wheel passed **755 tests**, zero failures or skips, plus the
complete Bundle guide on Databricks serverless PERFORMANCE_OPTIMIZED:
[run 1086439835567840](https://dbc-45604623-c18b.cloud.databricks.com/jobs/358630628755055/runs/1086439835567840)
finished **TERMINATED / SUCCESS**. All 552 installed Python sources matched the
tested working-tree manifest. Pytest execution took 25.00 seconds; the full run,
including setup, took 82.406 seconds. Wheel SHA256:
`f88b066d27e1b37d0a4a7adc303cd183d2edb846e514318fb9e04b472cf8e00d`.
This was native Python fit/apply/predict validation, not an additional Spark UDF,
REST or `ai_query` route test. No tables, registered models or endpoints were
created. The earlier Task194 route evidence remains scoped to its six steps.

### Task198 validation: encoders, fitted imputers and power transforms

On 2026-10-09, **1,914 distinct local cases** across 44 explicit affected files
were verified. The initial union had 1,913 passes and one obsolete expectation
that PowerTransformer lacked a declaration. That negative test now uses the
still-undeclared ModelBasedSelection; its complete 105-case file then passed.
No runtime change followed the union. Full Ruff, formatting, CI Ty and Lizard
CCN <= 10 passed. One final test formatting change preserved its AST.

Fresh processes reload ten actual random-forest pipelines per engine with
calculators disabled, compare predictions against saved expectations and check
unchanged learned-state digests. The tests retain `requires_context` for active
power/iterative configurations, HashEncoder's empty-schema mismatch, and strict
worker rejection. Ordinary row-local modes also compare singleton predictions.
Independent review repaired and rechecked valid Decimal/boolean settings,
missing iterative bounds, and Windows extended dtype compatibility. Existing
array digest fixtures still match; no transform formula was replaced.

The first native run exposed a NumPy-version assumption in one test: 561 passed
and one expected a dtype mismatch where NumPy 2 correctly reported overflow.
The test now pins that native exception and retains NumPy 1's dtype/mixed-value
boundaries. Its entire 147-case file passed locally. Independent review used
30 Polars probes to verify that unique-value order can change the failing check;
these remain failed diagnostics. Runtime code and the 1,914 distinct-case count
are unchanged.

The final wheel then passed **562 tests**, zero failures or skips, plus the
complete Bundle guide on Databricks serverless PERFORMANCE_OPTIMIZED:
[run 839395366420866](https://dbc-45604623-c18b.cloud.databricks.com/jobs/455354355469257/runs/839395366420866)
finished **TERMINATED / SUCCESS**. All 552 installed Python sources matched the
tested working-tree manifest. Pytest took 25.34 seconds; the full run including
setup took 60.836 seconds. Wheel SHA256:
`4dcc42ca575575c403be868d4378f3b463dcd804d7147212f7084a5bea72c740`.
This validates native Python fit/apply/predict. It does not add Spark UDF, REST
or `ai_query` admission or route tests. No tables, registered models or endpoints
were created.

### Task199 validation: generated features, selectors, detectors and text

On 2026-10-09, **2,154 distinct local cases** across 44 explicit affected files
were verified. The initial union passed 2,140 cases; 14 previous negative tests
expected rejection of native-valid scalar flags or ordered tuple names. Their
fixtures now use genuinely malformed arrays/sets, retaining rejection assertions.
Both complete affected files then passed all 177 cases. No runtime code changed
after the union. Full Ruff, formatting, CI Ty and Lizard CCN <= 10 passed.

Fresh processes reload eleven actual random-forest pipelines per engine with
all calculator fit methods disabled. The checks compare real predictions,
singleton requests and unchanged learned-state digests; they retain detector
row-loss errors and the documented text/H3 empty-schema diagnostic failures.
The Polars sole-text scoring repair is covered with actual training targets
absent from prediction input. Independent review reproduced, repaired and
rechecked native scalar/label compatibility and callback context declarations.
The inventory now has **55 of 63 implementations** declared, with **8 remaining**.

The final wheel passed **712 tests**, zero failures or skips, plus the complete
Bundle guide on Databricks serverless PERFORMANCE_OPTIMIZED:
[run 994473515746072](https://dbc-45604623-c18b.cloud.databricks.com/jobs/830129841788466/runs/994473515746072)
finished **TERMINATED / SUCCESS**. All 552 installed Python sources matched the
tested working-tree manifest. The full run including setup took 79.913 seconds.
Wheel SHA256:
`bbb85097d7813c5870b8055a1bc5ee546736c22c8306df48f85a25081dc4b3cf`.
This verifies native Python fit/apply/predict with real saved models. It does
not add Spark UDF, REST or `ai_query` admission or route tests. No tables,
registered models or endpoints were created.
