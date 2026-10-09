# Preprocessing inference context: coverage and remaining work

Status snapshot: **2026-10-09**, branch `093`, Task211 local median declarations.
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
| Implementations with context declaration machinery | 62 | Fifty-nine built-in implementations plus three custom wrappers |
| Without declaration machinery | 1 | Reviewed training-only splitter (two IDs) |
| Implementations with reviewed context/lifecycle behavior | **63** | Includes the unrecorded training splitter without inventing a frame declaration |
| Remaining initial owner reviews | **0** | Configuration, numerical and execution boundaries below still apply |
| Reviewed worker-admitted preprocessing families | 7 | Only their supported pandas configurations; not every mode/engine |

The count of 62 describes available machinery, not unconditional support for
every configuration. Existing worker-subset validators can abstain for valid
local states. Some Polars configurations also have no matching declaration.
Custom wrappers return `unknown` unless their saved definition explicitly
declares context. New project-defined classes are not part of the fixed 63.

### Existing declarations

| Implementation | Present context behavior | Remaining boundary |
| --- | --- | --- |
| `SimpleImputer` | `row` for matched reviewed declarations, including local pandas/Polars median apply | Median reuses saved numeric scalar validation through local-only declarations. The `mode` alias and unsupported empty/nonfinite states remain `unknown`. Task210 preserves Polars zero-column row counts. |
| `GroupImputer` | `row` for saved group lookups, including local pandas/Polars median apply | Mean, median and most-frequent strategies reuse saved lookups/fallbacks; median is local-only. The `mode` alias still normalizes to most-frequent. Task210 preserves integer modes and group keys through UInt64; Int128 retains its previous conversion boundary. |
| `StandardScaler` | `row` for matched reviewed declarations, including local Polars apply | Primitive numeric Polars division uses batch-independent native NumPy arithmetic with existing promotion. Float16/Decimal/Int128 and other unsupported types retain native expression boundaries; local metadata does not grant worker admission. |
| `MinMaxScaler` | `row` for matched reviewed declarations, including local Polars apply | Existing finite affine coefficients and matching feature range are required. All-null/nonfinite fits retain existing validator abstention. |
| `OneHotEncoder` | `row` for matched reviewed declarations, including local Polars apply | Existing dense subset with `max_categories=None`, no infrequent grouping and `include_missing=False`; the default `max_categories=20` and empty artifacts remain `unknown`. |
| `FeatureInteraction` | `row` for matched reviewed declarations, including local Polars apply | Existing degree 2–4 combinations, repetition and bias flags; Task210 preserves zero-column bias row counts. Other feature-generation classes have separate entries. |
| `ClipValues` | `row` for matched reviewed declarations, including local Polars apply | Saved finite fixed limits and matching configuration; native dtype and numerical behavior are unchanged. |
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
| PC-08 | `ValueReplacement` | Fixed scalar/list/tuple and dictionary rules, nulls, typed keys and mapping precedence. Integer numeric rules preserve width and precision; incompatible numeric replacements require explicit Casting, including empty/unmatched requests. Integer null rules use nullable output. Preview reports definite integer/object types and leaves uncertain types unknown. Float/mixed-type runtime boundaries remain; arbitrary mapping objects/Series are outside inspection. |
| PC-09 | `DropMissingColumns` | Reuse saved dropped columns even when request missingness differs. Training threshold is reporting metadata, not recomputed. |
| PC-11 | `MissingIndicator` | Saved columns and nonempty string suffix, null/NaN flags, no-op and empty frames. Non-string suffix objects are outside inspection. |
| PC-21 | `CorrelationThreshold` | Saved drop list and enabled/disabled flag; no request correlation calculation. |
| PC-23 | `UnivariateSelection` | Saved candidate/selected columns, no-target artifact and empty selection. Scoring methods without p-values now retain an empty p-value report. |
| PC-24 | `VarianceThreshold` | Frozen selection with constant/all-missing requests, empty selection and saved undefined variance metadata. |
| PC-28 | `MaxAbsScaler` | Saved scale/max-abs vectors, zeros/constants/nulls and empty no-op. Supported primitive Polars inputs use native Series division across eager/lazy chunks, including NumPy scale scalars. Special input/output types retain native boundaries; legacy bulk reciprocal-overflow behavior remains. |
| PC-29 | `RobustScaler` | Saved center/scale/quantiles and all flag combinations, null statistics and empty no-op. Eager/lazy Polars replay uses native Series division with its existing Float64 cast and bulk rounding, including legacy reciprocal overflow. |
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

Task206 rejects PC-08 fractional numeric rules on integer input before matching
rows. Replacing `1` with `0.5` requires an explicit preceding `Casting` to float.
Integral numeric rules stay within the input integer dtype's range; missing
replacements preserve nullable integer output. This avoids rounding untouched
large integers or wrapping unsigned values. Float/object and mixed nonnumeric
replacement boundaries remain configuration-specific.
For PC-28, the original fitted scale of 5 yielded `6 / 5 = 1.2000000000000002` in the full
Polars result and `1.2` in a singleton, a difference of `2.22e-16`. The diagnostic
retains exact comparisons; no tolerance was added to hide it. Task205 repairs
this ordinary eager-frame case through native Series division. Task208 extends
that behavior to NumPy statistics and lazy frames for the supported numeric types.
Task207 corrects StandardScaler's primitive numeric division separately while
preserving its native dtype promotion; see its remaining boundaries above.

### Eleven additional local contracts (Task197)

These declarations inspect saved state and call the existing applier. They do
not broaden Spark UDF, REST or `ai_query` admission. A `row` declaration describes
required input context; strict sample checks can still expose unsupported dtypes,
null behavior or numerical differences.

| ID | Implementation | Inspected scope and remaining boundary |
| --- | --- | --- |
| PC-01 | `GeneralBinning` | Fixed learned bins and supported label settings; shares validation with the existing binning owners. Numeric custom labels retain the existing Polars apply error. |
| PC-03 | `KBinsDiscretizer` | Saved bin edges for uniform, quantile and k-means strategies; inference does not recompute bins. |
| PC-04 | `Casting` | Frozen categories and supported scalar conversions. Explicit nullable integer choices survive fit/schema/replay; integer values and plain integer tokens stay exact beside decimal/fractional, null or invalid peers. Preview respects fit precedence. Large decimal/scientific tokens retain native parsing limits. Legacy categories and coercive fallback casts can need whole-request context; lowercase integer targets still choose their container from request nulls. |
| PC-05 | `AliasReplacement` | Saved standard/custom mappings, unseen inputs and nulls. Native engine limitations are retained. |
| PC-06 | `InvalidValueReplacement` | Fixed rules and replacement values. Active numeric integer rules require exactly representable replacements; otherwise explicit Casting is required before matching rows. None, NaN or pd.NA retain nullable integer width. Integral float bounds compare as exact integers. Infinity-only integer cleanup remains a no-op. Fractional/nonfinite bounds and nonnumeric replacements retain native boundaries. |
| PC-07 | `TextCleaning` | Saved operation order, nulls, regex and no-op settings. Slash-date normalization preserves object output for pandas object/StringDtype input, including empty and null-only chunks followed by trim. Categorical mapping is unchanged; Polars still rejects unsupported regex lookbehind. |
| PC-18 | `DateFeatures` | UTC-aware saved states, epochs, time zones, nulls, tuples and NumPy string settings. Legacy non-UTC state stays `unknown`. Existing generated-name overwrites and Polars duplicate-output errors remain visible. |
| PC-20 | `PolynomialFeaturesNode`, `PolynomialFeatures` | Saved degree/flags/columns/order and prefixes. Apply rebuilds sklearn's combinatorial expansion from configuration; it does not learn request statistics. Empty requests retain the native output schema; null-input and generated-name collision errors remain. |
| PC-36 | `ManualBounds` | Saved fixed/open limits; `row` context with a filter effect. Ordinary prediction runs this step and rejects requests that would lose rows. It is not automatically skipped. |
| PC-37 | `Winsorize` | Saved training quantiles, never request quantiles. Integer input requires representable integral bounds and retains its dtype; fractional/out-of-range bounds require explicit Casting. Integer 0/100-percentile endpoints remain exact; large-integer interpolation is rejected before float conversion. Float/Decimal behavior remains native. |
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
| PC-13 | `HashEncoder` | Existing stable hash and saved bucket count. Current key version is row-local; legacy pandas rendering reports `global`. Empty output is int64 for Python integer bucket counts from 1 through 2**63. Larger counts and non-Python-integer settings retain native dtype/overflow boundaries. |
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
| PC-31 | `H3Index` | Saved coordinates/resolution and native output labels. Pandas empty output retains object dtype and the input index. Optional H3 remains required, including for valid empty requests. |
| PC-34 | `EllipticEnvelope` | Saved sklearn detector models and learned covariance state. Active models filter rows; null/nonfinite handling remains native. |
| PC-35 | `IQR` | Saved training bounds; no request quartiles. Active bounds filter rows. |
| PC-38 | `ZScore` | Saved means/stds and threshold; no request statistics. Active maps filter rows. |
| PC-46 | `count_vectorizer` | Saved vocabulary, estimator and output layout; no inference vocabulary fit. Native cache mutations and unusual input dtypes remain diagnostic boundaries. |
| PC-47 | `hashing_vectorizer` | Saved hash/analyzer settings and dense output width. A pristine native estimator can lazily mutate its cache on first transform; the diagnostic reports that mutation. |
| PC-49 | `tfidf_vectorizer` | Saved vocabulary/IDF and output width; no request IDF fitting. Optional callbacks remain undeclared. |
| PC-50 | `tokenizer` | Saved analyzer settings, output names and token counts. Empty outputs retain the populated string representation and int64 counts in pandas and Polars. |

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

## Training-only and inspection lifecycle (Task200)

These seven owners have been reviewed against actual training and prediction.
Six gained local declarations; the train/test splitter deliberately has no new
hook because its partitioned `SplitDataset` is not a feature-frame transform.
The existing query returns `unknown` for it, and real prediction never calls it.

| ID | Implementation | Native apply and actual prediction behavior |
| --- | --- | --- |
| PC-10 | `DropMissingRows` | Native `row` / filter. Evaluation can filter, while ordinary prediction skips it and retains every requested row. Configure an imputer or let the model reject missing input. |
| PC-32 | `DatasetProfile` | Statistics are captured during fit. Apply is an active `row` / preserve passthrough; prediction does not generate another profile. |
| PC-33 | `DataSnapshot` | Rows are captured during fit. Apply is an active `row` / preserve passthrough; prediction does not capture new rows. |
| PC-39 | `Oversampling` | Supported ordinary samplers use `global` context and expand training data. Prediction skips them. Combined SMOTETomek and custom callbacks remain `unknown`. |
| PC-40 | `Undersampling` | Supported ordinary samplers use `global` context and filter training data. Prediction skips them. Replacement sampling and custom callbacks remain `unknown`. |
| PC-41 | `TrainTestSplitter`, `Split` | Creates training partitions, is not recorded in `fitted_steps`, and has no probe entry. Reviewed training behavior is not an inference declaration. |
| PC-42 | `feature_target_split` | Native `row` / preserve separation returns `(X, y)`, not a feature frame. It is also unrecorded and absent from the prediction report. |

A skipped fitted step reports `action: skip_preserve_rows`, `status: skipped`,
empty `checks` and `state_validation: unavailable`. Its context can describe the
native training operation; the probe did not execute that operation. This differs
from an unrecorded split marker, which has no entry or per-step state hash.
The original configuration remains part of the saved pipeline.

Do not infer sampler effects from their names. A real SMOTETomek example reduced
six rows to zero, and replacement undersampling duplicated an original row.
The current capability vocabulary cannot honestly express those mixed effects.
Also retain the native target-only Polars limitation: removing the sole target
column leaves zero-width features with height zero beside nonempty labels.
This structural training helper has no newly enabled prediction route.

Temporal scalar snapshot values have a separate integrity boundary. The former
generic object hash could ignore a pandas Timestamp's actual date, making two
different snapshots share a digest. Unsupported `date`, `time`, `timedelta`,
pandas `Period` and NumPy `timedelta64` scalars now raise `TypeError` instead.
Pandas temporal subclasses are included. NumPy
non-object temporal arrays retain their existing byte encoding. A temporal
snapshot can still be a native passthrough, but fingerprint/probe certification
requires representable state. Ordinary local binary-payload checks are separate;
this change does not add a datetime codec or recompute training reports.

## Sentence encoder assets (Task201)

| ID | Status | Implementation | Reviewed scope |
| --- | --- | --- | --- |
| PC-48 | Implemented | `sentence_embedder` | Native PyTorch encoder snapshot, SHA256 identity, exact dependency pins, CPU replay and local pandas/Polars context inspection. |

New fits capture the actual loaded encoder using native `torch.save` into an
in-memory buffer. The immutable bytes contain weights, tokenizer and configuration;
`model_sha256` identifies that exact snapshot. `model_name` remains provenance,
not an instruction to download a replacement at prediction time. No learned
transform or tokenization formula is implemented a second time.

The existing `pipeline.pkl` carries these bytes through `save_local_pipeline`,
model sets and MLflow pyfunc packages. There is no extra asset directory to copy
or absolute model path to repair. Optional encoder dependencies are recorded at
fit, merged with project pins and included in MLflow requirements; incompatible
versions fail before native encoder restoration. The source model directory and
Hugging Face cache are unnecessary when replaying the saved encoder.

Apply restores the snapshot on **CPU**, calls the same native `encode`, and
caches by immutable model bytes. A changed original model or another model with
the same display name cannot replace a fitted snapshot. Neither fit nor apply
adds direct filesystem operations to the preprocessing owner. First-time fitting
a Hub model still requires network/cache access; installing Python dependencies
is separate from offline model replay.

`inference_capability` declares local `row` context for packaged state and empty
no-ops. Inspection checks schema, width, bounded bytes, checksum and package pins
without executing the encoder. Name-only legacy artifacts remain readable and
`unknown`; refit to obtain captured assets. Python/NumPy scalar flag semantics are
preserved; arbitrary truthy objects and noncanonical scalar classes are outside
the inspected subset. An empty apply keeps the fitted output width without
loading a model; a probe still validates its saved state first.

The native snapshot uses executable pickle, like the existing pipeline payload:
load only trusted artifacts. Checksums prove byte integrity, not authenticity.
The **256 MiB** local-pipeline and aggregate model-set budgets remain unchanged;
larger encoders or several copied encoders can exceed them. CPU PyTorch is the
reviewed path; ONNX, OpenVINO, custom model implementations and cross-version
loading are not admitted. Context metadata and offline replay do not add Spark
UDF, REST or `ai_query` eligibility. Exact chunk equality still depends on the
actual encoder, input and runtime; no numerical tolerance was added to the probe.

Native API references: [PyTorch save](https://docs.pytorch.org/docs/stable/generated/torch.save.html)
and [PyTorch load](https://docs.pytorch.org/docs/stable/generated/torch.load.html).

Reviewed declarations still have the configuration and dtype boundaries recorded
above. Context-dependent operators can correctly report `requires_context`; a
separate execution mechanism needs its own specification and tests.

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

### Cross-cutting work beyond initial owner reviews

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
print("Without declaration machinery:", len(remaining))
for names in remaining:
    print(", ".join(names))
```

Expected snapshot: 67 names, 63 implementations, 1 without declaration machinery:
`TrainTestSplitter`/`Split` (reviewed training-only marker).
Reviewed lifecycle coverage is recorded separately.
This counts declaration machinery, not whether a fitted configuration passes
`get_inference_capability` or the worker certificate.

## Validation record

### Task201: sentence snapshots and offline package replay

The affected local union passed **691 tests** across 21 explicit files; one
existing symlink test skipped because Windows symlink privilege was unavailable.
Ruff, full CI formatting/Ty scopes, Lizard CCN <= 10 and strict MkDocs passed.

Databricks [run 968745626942362](https://dbc-45604623-c18b.cloud.databricks.com/jobs/988493157672277/runs/968745626942362)
finished **TERMINATED / SUCCESS**: **149 passed, zero failures or skips** in
47.22 seconds of pytest execution (264.278 seconds total run time). The native
check verified all 552 installed runtime files and the exact test inventory.
Wheel SHA256: `063d876b41b16f9c79175644f690ed1fe1978863489e8e4ac3456486b440bfb3`.
Six warnings came from intentionally tiny one-row holdouts where R² is undefined.

Actual weighted sentence encoders survived fresh offline processes, removed
source paths and empty caches in local artifacts, model sets and MLflow pyfunc
packages. Overwriting the original encoder after fit did not change predictions.
Independent local BERT/fast-tokenizer probes also retained weights, tokenizer,
CPU placement and exact outputs in both engines. These are real model tests;
the separately maintained mocked shape tests are not counted as asset proof.

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

### Task200 validation: training helpers, inspection and state integrity

On 2026-10-09, **1,438 tests passed** across 36 explicit affected files. The
inspection test's invalid-input annotation was corrected after full CI Ty
identified an intentionally supplied `None`; all 18 cases in that file passed
again. Runtime behavior was unchanged. Ruff, formatting, full CI Ty and Lizard
CCN <= 10 passed. Independent review rechecked the original sampler/temporal
findings without repeating the complete affected suite.

Fresh processes reload eight real random-forest pipelines per engine, with
all calculator fit calls and training-only filter/sampler/split apply calls
disabled before loading. Full and singleton predictions match; requested rows
are retained, saved reports are unchanged, and training split markers have no
probe entries. Known mixed sampler effects remain `unknown`. A small native
NumPy-boolean threshold compatibility fix preserves the original row-filter
declaration without changing the missing-row formula.

Actual Timestamp/Period snapshot collisions and NumPy timedelta/integer
collisions were reproduced before the shared fail-closed correction. Supported
scalar/container/array digest fixtures and NumPy temporal-array hashes remain
unchanged. This is an integrity fix, not a new temporal scalar serialization
feature. Context/lifecycle review now covers **62 of 63 owners**; **61 have
declaration machinery**, and the train/test splitter remains an unrecorded
training-only operation. Sentence embedding asset packaging remains open.

The final wheel passed **399 tests**, zero failures or skips, plus the complete
Bundle guide on Databricks serverless PERFORMANCE_OPTIMIZED:
[run 152339084173952](https://dbc-45604623-c18b.cloud.databricks.com/jobs/750853859501812/runs/152339084173952)
finished **TERMINATED / SUCCESS**. All 552 installed Python sources and all ten
test/guide assets matched the manifest. Full duration including setup:81.150s.
Wheel SHA256:
`ea03ef1d8e68bc3907ed3b0235ba4eaa940deb7c38c28cfad755c59b97db2be6`.
This is native Python saved-model fit/apply/predict verification, not additional
Spark UDF, REST or `ai_query` route coverage. No tables, registered models or
endpoints were created.


### Task201 validation: every registered preprocessor on small data

At the user's request, the final installed wheel also ran every registered
preprocessor against actual **32-row** fixtures in pandas and Polars. The matrix
covers **67 registered names / 63 implementations**, with 134 actual fit/save/apply
cases and one registry coverage guard. A further 22 selected cases check saved
group statistics, ordered temporal history and training-only filter/sampler
lifecycle. No calculator refitting is allowed during saved-state replay.

The complete **157 tests passed**, zero failures or skips, on Databricks
serverless:
[run 113581190209016](https://dbc-45604623-c18b.cloud.databricks.com/jobs/994960019987268/runs/113581190209016)
finished **TERMINATED / SUCCESS**. All 552 installed runtime files and all nine
test/verifier/package assets matched the manifest. Pytest took 10.99 seconds;
the complete run took 226.734 seconds. This uses the same Task201 wheel SHA256
`063d876b41b16f9c79175644f690ed1fe1978863489e8e4ac3456486b440bfb3`.

The durable matrix is
`skyulf-core/tests/integration/core/test_preprocessing_small_data_roundtrip.py`.
Each name has one explicit meaningful recipe; aliases, training helpers and
inspection steps are included in the denominator. This does not establish every
configuration, arbitrary custom code or exact pandas-to-Polars numerical parity.
The run uses native Python execution on Databricks and adds no Spark UDF, REST
or `ai_query` route coverage. No tables, registered models or endpoints were
created. The context and execution boundaries elsewhere in this guide remain.

### Task204 validation: empty schemas and null-only text

Task204 repairs five previously recorded boundaries: PC-07 `TextCleaning`,
PC-13 `HashEncoder`, PC-20 `PolynomialFeatures`, PC-31 `H3Index` and PC-50
`tokenizer`. Their scope is recorded in the owner rows above. Existing native
operations supply the output types; no dependency or artifact field was added.
For an empty polynomial request, one zero-valued row supplies sklearn's native
layout and the output is then sliced to zero rows. This learns no request
statistics and preserves the existing null-input and name-collision errors.

The local affected union passed **1,074 tests** across nineteen explicit file/node
selections. After two static cleanups, all eighteen affected Hash/Tokenizer cases
passed again; these are repeats, not additional distinct cases. Full Ruff,
formatting, CI Ty and Lizard CCN <= 10 passed. Independent review found no blocker
and verified 131 differential probes for populated values, dtypes and errors.
Fresh-process saved-model checks disable calculator fit and compare full and
singleton predictions plus learned-state digests.

The final wheel passed **710 tests**, zero failures or skips, plus the complete
Bundle guide on Databricks serverless PERFORMANCE_OPTIMIZED:
[run 318386173857228](https://dbc-45604623-c18b.cloud.databricks.com/jobs/381149866704502/runs/318386173857228)
finished **TERMINATED / SUCCESS**. All 552 installed Python sources matched the
tested working-tree manifest. Pytest took 39.17 seconds; the complete run took
88.803 seconds. Wheel SHA256:
`aa9d35e9bea9c76e7560f31491a34b7b04b3eec4ae61ea382e0cd7ebc0e71617`.
This is native Python fit/apply/predict coverage. No Spark UDF, REST or `ai_query`
admission or route coverage was added; no tables, registered models or endpoints
were created.

The initial review count remains **63 of 63 owners**. PC-04 casting fallbacks,
PC-06 integer/null replacement, PC-08 numeric widening, PC-28 Polars rounding and
PC-37 integer winsorization remain open numeric/context boundaries. Existing
strict diagnostic checks retain their dtype/output failures; no blanket numeric
cast or comparison tolerance was added. Historical Task197-199 empty-schema
observations above are superseded only for the five repaired configurations.

### Task205 validation: exact integers and native scaling

Explicit nullable integer casting choices such as `Int64` and `UInt64` now
survive configuration normalization. Integer/object inputs use native nullable
parsing when it produces an integer result; legacy float parsing and fallback
behavior remain. Lowercase `int64` retains its existing container choice. For a
stable nullable pandas output, request `Int64` explicitly. Already rounded
floating-point inputs and mixed fractional parsing cannot recover lost precision.

Invalid-value rules replacing integers with `None`, `NaN` or `pd.NA` now use
nullable integer output, including empty or entirely unmatched chunks. Polars
represents those missing integer values as null. Integer width and values above
`2**53` remain exact; float-column sentinel behavior and infinity-only integer
no-ops are unchanged. Refit older pipelines where these integer-rule outputs
feed string-key encoders: an old learned key such as `"2.0"` differs from the
new integer key `"2"`. This is an intentional output-type correction.

For ordinary fitted MaxAbsScaler and RobustScaler state, eager Polars execution
uses native Series arithmetic to remove the observed batch/singleton division
difference. No formula, tolerance or saved-state field was added. StandardScaler
remains unchanged because the same substitution changes Float32 promotion;
MaxAbs Decimal/NumPy-statistic and lazy execution boundaries remain explicit.

The affected local union passed **1,210 tests** across eighteen explicit file/node
selections. After a test type-narrowing fix, all thirty-six integer-null cases
passed again; these are repeats within the union. Full Ruff, formatting, CI Ty
and Lizard CCN <= 10 passed. Independent review found no blocker, including 168
Casting baseline comparisons, 112 float replacement comparisons and 180 scaler
bulk comparisons. The 63-owner review count and worker admission are unchanged.

At Task205 delivery, PC-08 replacement and PC-37 winsorization still required an
explicit numeric-type decision. Their full integer batches could round an
untouched `9007199254740993`
to `9007199254740992.0` while an unmatched singleton stays exact. A preceding
`Casting` to `float64` stabilizes ordinary fractional model features, but even
`coerce_on_error=False` does not make large-integer-to-float conversion lossless.
Do not use it for columns requiring exact large integers. Task206's integer
contract below supersedes these two observations without relabeling the
operations global.

The same final wheel passed **596 tests**, zero failures or skips, plus the Bundle
guide on Databricks serverless PERFORMANCE_OPTIMIZED:
[run 130677126334878](https://dbc-45604623-c18b.cloud.databricks.com/jobs/280135714197592/runs/130677126334878)
finished **TERMINATED / SUCCESS**. All 552 installed Python sources matched the
manifest. Pytest took 22.09 seconds; full duration was 81.963 seconds. Wheel SHA256:
`97c96a95300613b14292a178d47a94f87ac8b0d28c1f1f23b1694c1d811efd53`.
Two earlier attempts stopped during collection for missing test fixtures/imports
and Hypothesis; the repaired test package ran unchanged runtime sources. This
verifies native Python execution, not Spark UDF/REST/`ai_query` routes. No tables,
registered models or endpoints were created.

### Task206 integer contract and migration

`ValueReplacement` now keeps integral numeric rules within the source integer
dtype's range. `None`, `NaN` and `pd.NA` use integer nulls. Fractional, infinite
or out-of-range numeric replacements raise an error directing callers to an
explicit `Casting` step. Validation uses the configured rules and dtype before
row matching, so an empty or unmatched request cannot conceal an invalid rule.

`Winsorize` keeps integer output for representable integral bounds. Fractional,
non-finite and out-of-range bounds require explicit Casting, including bounds
that happen not to affect the current rows. Training percentile endpoints at
0 and 100 use exact minima/maxima. Interpolated quantiles on integer data outside
`[-2**53, 2**53]` are rejected before conversion; no custom quantile algorithm
or automatic Decimal/object promotion was added.
Within that range, interpolation still uses pandas's floating-point quantiles.
The integer check validates the computed bound, not the mathematically exact
quantile; rounding can hide a fractional part near `2**53`. Only the 0/100
endpoints carry the exact-integer quantile guarantee.

For genuinely fractional features, cast explicitly to float before fitting
and replaying these steps. This accepts floating-point precision: converting
`9007199254740993` to float64 still loses its final unit. Keep exact identifiers
and other precision-sensitive integers out of that conversion.

Refit and re-save affected older pipelines. Integer output may disagree with
a legacy saved float schema or with downstream string-key encoders trained on
keys such as `"2.0"` instead of `"2"`. Strict diagnostics continue reporting
those mismatches. Saved artifact field layouts and worker admission are unchanged.
Float32 widening, mixed nonnumeric replacements and native float/Decimal
boundaries remain configuration-specific; this is not universal family closure.
At Task206 delivery, `ValueReplacement`'s configuration-only schema preview
still passed input labels through; Task207 corrects that preview below.
Polars still rejects a NaN replacement key on integer input;
a NaN replacement value is supported as an integer null.

The affected local union passed **1,139 tests** across eighteen explicit file/node
selections. After the final static and Arrow UInt64 repairs, the two precision
files passed **68** and **43** tests; the final Arrow check passed again. These
are focused reruns, not additional disjoint suites. Full Ruff, formatting, CI
Ty and Lizard CCN <= 10 passed. Independent review found no blocker and checked
unchanged float/object behavior, exact keys, Arrow types and weighted replay.

The final wheel passed **538 tests**, zero failures or skips, plus the Bundle
guide on Databricks serverless PERFORMANCE_OPTIMIZED:
[run 1017340571205314](https://dbc-45604623-c18b.cloud.databricks.com/jobs/499829535761776/runs/1017340571205314)
finished **TERMINATED / SUCCESS**. All 552 installed Python sources and sixteen
test/guide/support assets matched the manifest. The local collected and remote
passed test-node sets match exactly. Pytest took 17.92 seconds; the complete run
took 70.298 seconds. Wheel SHA256:
`ac19dede21d980479f3c75d30f87ef587344bcc8913ec2897177c2f5c53379c0`.
This verifies native Python fit/apply/predict execution. No Spark UDF, REST or
`ai_query` routes were added; no tables, registered models or endpoints were created.

### Task207 preview, StandardScaler and invalid integer rules

`ValueReplacement` preview now preserves column names and reports only definite
output dtypes. Integer null rules advertise nullable integers, including Arrow
inputs whose runtime null conversion uses pandas nullable types. Existing
object output stays object. If Float32 widening, mixed replacements or an
unsupported rule prevents a definite result, only that column's dtype is
omitted; the Canvas already displays it as `unknown`. Unaffected columns keep
their types. UI-list precedence, nested mappings and collisions between coerced
dictionary keys follow the existing runtime rules. Preview never fits or applies
the pipeline, and fitted runtime schemas remain the validation authority.

StandardScaler's Polars path uses native NumPy division for supported primitive
numeric inputs and Float32/Float64 output, including lazy frames. Native Polars
schema inference supplies the existing promotion; centering, nulls, zero-scale
handling and saved statistics remain unchanged. This corrects chunk-dependent
division without a tolerance or rounding layer. Ordinary bulk results can differ
by a final bit; extreme reciprocal-overflow cases can change more substantially.
For example, dividing `1e-320` by `1e-320` now produces `1` consistently.
Float16, Decimal, Int128 and other unsupported expression types retain their
previous path. MaxAbsScaler/RobustScaler boundaries listed above remain open.
Task207 did not add a Polars context declaration. Task208 adds local Polars apply
metadata below; worker admission remains unchanged.

`InvalidValueReplacement` shares the existing integer scalar guard with
`ValueReplacement`. Active integer rules normalize integral numeric sentinels
and reject fractional, infinite or out-of-range replacements before matching
rows, including empty or unmatched requests. Integral floating comparison
bounds become exact integer bounds: an upper bound of `float(2**53)` must reject
the larger integer `2**53 + 1`. Integer infinity-only cleanup and configurations
without an effective numeric rule remain no-ops. Boolean/string replacements
and fractional/nonfinite bounds retain their native behavior. Refit and re-save
older pipelines whose changed integer output or numeric results affect saved
schemas, encoders or model features; artifact field layouts are unchanged.

Casting's remaining mixed-fraction boundary was reproduced rather than declared
closed. Coercive parsing of an object column containing `2**53 + 1` and `"1.5"`
can round the integer before conversion. A retry that removes fractional values
does not solve mixtures also containing decimal strings such as `"1.0"`.
No partial parser or automatic widening was added. Even strict conversion can
silently round a large integer beside a decimal-form string such as `"1.0"`;
rejecting observed fractional values alone is insufficient. Preserve already
integer or canonical integer-string inputs when exact identity is required.
Explicit nullable targets stabilize the container, while lowercase integer
targets retain their legacy request-dependent container choice.

All **1,600 distinct local tests** in eighteen explicit file selections passed.
The first combined command passed 1,564; twenty-nine tests could not use the
default Windows temporary folder and seven were affected by the backend import
enabling pandas copy-on-write globally. Those thirty-six tests passed in a
separate Core process with an explicit writable temporary folder. Passing groups
were not repeated, and no runtime changes were made for these harness issues.
Full Ruff, formatting, CI Ty and Lizard CCN <= 10 passed. Independent review
passed thirty-five additional targeted probes with no remaining blocker.

The same final wheel passed **1,186 tests**, zero failures or skips, plus the
Bundle guide on Databricks serverless PERFORMANCE_OPTIMIZED:
[run 553129580194887](https://dbc-45604623-c18b.cloud.databricks.com/jobs/972514887883844/runs/553129580194887)
finished **TERMINATED / SUCCESS**. All 552 installed Python sources and twenty-two
test/guide/support assets matched the manifest; local collected and remote passed
test-node sets match exactly. Pytest took 30.47 seconds with 85 warnings; the full
run took 65.703 seconds. Wheel SHA256:
`9bd91a3850fb74f37d44b98be59d3571b5bbbed7f561117689af98ee472f43dc`.
The first attempt passed 1,185 tests but could not set up one test because its
shared fixture file was missing from the archive. Adding the existing conftest
and verifying the isolated fixture plan repaired packaging without changing the
wheel. This validates native Python execution, not Spark UDF, REST or `ai_query`;
no tables, registered models or endpoints were created.

### Task208 Casting and scaler context continuation

Pandas integer Casting now preserves an exact integer or plain integer token
beside a decimal-form peer. For example, `[9007199254740993, "1.0"]` becomes
`[9007199254740993, 1]` with an explicit `Int64` target. With coercion enabled,
`"1.5"` becomes missing without rounding the other value. Invalid tokens and
out-of-range values retain the existing null/raise rules and validation order.
The same correction handles strict UInt64 values beside nulls, where the native
batch parser can return the original object values rather than numeric data.
Narrow/signed targets now raise the appropriate range error for those values.

The correction reuses native scalar parsing only when object/string integer
parsing returns floating-point or unconverted object results. Exact object
scalars reach the existing fractional/range guards without a common float
container. Ordinary numeric inputs and native integer parsing keep their fast
paths. This fallback is slower: a local 10,000-row prototype took about 62ms
versus 4ms for the previous bulk cast. No custom decimal parser was added.

This does not make numeric text universally lossless. A token such as
`"9007199254740993.0"` can itself round inside the native parser, even in strict
mode. Already-rounded floats cannot be recovered. Lowercase integer targets
still select nullable containers from request missingness, and other best-effort
fallbacks retain their documented boundaries. Existing Casting context labels
are unchanged; the exact diagnostic remains responsible for exposing a sample
that fails chunk equivalence. Refit affected saved models even when the dtype
stays `Int64`: corrected values can disagree with previously learned encoders
or model features. Artifact field layouts are unchanged.

Casting schema preview now gives the shared `target_type` precedence over a
conflicting `column_types` entry, matching fit/apply, and skips absent columns.
StandardScaler declares local Polars `row` apply for the four existing mean/std
flag combinations, using the same saved-state validation and config matching.
Invalid/unsupported state still abstains. This metadata adds no Polars worker,
Spark UDF or serving admission.

MaxAbsScaler and RobustScaler now reuse native Series division inside Polars
batch expressions for supported lazy and NumPy-statistic paths. Each column's
saved denominator is captured independently. Existing bulk values, dtype
promotion, nulls, zero handling and special-type fallbacks are retained; singleton
and full requests agree for the reviewed numeric configurations. Their legacy
extreme reciprocal-overflow results remain unchanged. This differs from Task207's
intentional StandardScaler correction to true NumPy division.

All **1,110 local tests** in fourteen explicit affected Core files passed
(123 warnings, 26.14 seconds). Full Ruff, formatting, CI Ty and Lizard CCN <= 10
passed. Independent review found no blocker and passed thirteen additional
boundary probes. A separate 240-case Casting comparison preserved 235 prior
results; five cases intentionally repaired strict UInt64/object-parser behavior.
Native exception wording can change from `float64` to `object` in a remaining
near-tolerance safe-cast error; byte-identical native errors are not promised.

The same final wheel passed **916 tests**, zero failures or skips, plus the
Bundle guide on Databricks serverless PERFORMANCE_OPTIMIZED:
[run 759853836417962](https://dbc-45604623-c18b.cloud.databricks.com/jobs/900860217197590/runs/759853836417962)
finished **TERMINATED / SUCCESS**. All 552 installed Python sources and nineteen
test/guide/support assets matched the manifest; local collected and remote passed
test-node sets match exactly. Pytest took 24.99 seconds with 123 warnings; the full
run took 80.282 seconds. Wheel SHA256:
`956695874a40fd4a4efe094fe2f21be1ac60264f77e269f4baccf7d37429821f`.
This validates native Python fit/apply/predict and context diagnostics. Spark UDF,
REST and `ai_query` admission remain unchanged; no tables, registered models or
endpoints were created.

### Task209 existing families' local Polars declarations

SimpleImputer, GroupImputer, MinMaxScaler, ClipValues, OneHotEncoder and
FeatureInteraction now expose the same reviewed saved-state subsets through
Polars `local` / `row` metadata. Their pandas declarations already existed.
Nine declaration entries cover the six owners' existing strategy selectors;
codec versions, validation, configuration matching and applier identity checks
are retained. No transformation, estimator, resolver or artifact format changed.
Existing pandas/Spark declarations and worker admission are unchanged.

This fills an engine-metadata gap, not every configuration gap. SimpleImputer
median and its `mode` alias, GroupImputer median, OneHotEncoder's default
`max_categories=20`, missing-token mode and empty artifacts still abstain.
The GroupImputer `mode` alias already normalizes to the reviewed most-frequent
strategy. Null/unseen groups reuse saved fallbacks without request aggregation.
Local metadata never invokes fit or apply. A positive `row` declaration describes
input dependencies; the exact diagnostic still detects dtype or shape failures.

Two existing boundaries were reproduced here and repaired in Task210 below:

- GroupImputer mode can lose exact large integers beside nulls. Polars training
  conversion can round `9007199254740993` to `9007199254740992.0`; pandas saved
  lookup mapping can promote exact values to float when fallback groups appear.
  UInt64 maximum-value mode inputs can also fail during apply. These are native
  fit/lookup precision issues, not evidence that inference relearns group means.
- A Polars frame with zero rows **and zero columns** can acquire one row when
  SimpleImputer restores a missing column or FeatureInteraction adds a bias
  literal. Ordinary zero-row frames retaining a column schema pass the reviewed
  empty checks. The diagnostic still fails the zero-column cases; metadata does
  not turn them into successful reports.

All **584 local tests** across eighteen explicit affected files passed
(29 warnings, 46.51 seconds). After tightening a test's pandas/Polars type
narrowing and input-mutation assertion, its 26 cases passed again; this does not
increase the distinct test count. The six-step model also passed fit/save and
fresh-process load/probe/predict on both engines with calculators disabled.
Full Ruff, formatting, CI Ty and Lizard CCN <= 10 passed. Independent review
confirmed that numerical code and previous declaration sequences are unchanged;
two additional probes kept the zero-column diagnostic failures visible.

The same final wheel passed **584 tests**, zero failures or skips, plus the
Bundle guide on Databricks serverless PERFORMANCE_OPTIMIZED:
[run 407701115957407](https://dbc-45604623-c18b.cloud.databricks.com/jobs/461409568158715/runs/407701115957407)
finished **TERMINATED / SUCCESS**. All 552 installed Python sources and twenty-six
test/guide/support assets matched the manifest; local collected and remote passed
test-node sets match exactly. Pytest took 37.54 seconds with 29 warnings; the full
run took 93.908 seconds. Wheel SHA256:
`9236ae2b24cacbd55d9c0dbb153ca03e8c4973aca5369312297dea9d86e6e92d`.
This validates native Python saved-state replay and context diagnostics. No Spark
UDF, REST or `ai_query` routes were added, and no tables, registered models or
endpoints were created.

### Task210 exact group integers and zero-column row counts

GroupImputer now retains native Polars integer group keys when fitting every
strategy, and integer values when fitting most-frequent/mode. Only these selected
columns use nullable pandas integer storage; existing mean/median value conversion
is retained. Signed and unsigned widths through UInt64 are covered. Int128 keeps
its previous conversion path and is not included in this guarantee.

Pandas application keeps group lookup keys and integer replacements exact before
missing or unseen groups introduce nulls. This also covers an empty learned group
map with an integer global fallback, duplicate request indexes, Arrow-backed
integer columns and mixed integer/fractional keys. Native floating outputs and
pure-float replacement errors are retained; Polars replacement casts are unchanged.
No scoring-batch statistics are learned, and context declarations are unchanged.

SimpleImputer missing-column restoration and FeatureInteraction bias generation
now use native Polars repeats sized to the input. Zero-column frames retain their
zero or nonzero row counts, scalar dtypes and lazy execution. The diagnostic's
zero-column reversal retains a clone: Polars has no index or column values to
reorder in this case, and its ordinary reverse operation loses the row count.

Artifact fields and codec versions are unchanged. Exact saved group artifacts
remain readable. Previously rounded fitted keys or values cannot be recovered;
refit and re-save the affected whole model pipeline, including downstream steps,
even if its displayed schema is unchanged. Models trained through the previously
lossy pandas group replay should also be refitted before using corrected features.

All **575 local tests** in fifteen explicit affected files passed (60 warnings,
59.87 seconds). The union includes 46 group precision/control cases, 61 zero-column
cases and four real integer-mode fit/save/fresh-process load/probe/predict cases
across pandas and Polars, with calculators disabled during replay. Full Ruff,
formatting, CI Ty and Lizard CCN <= 10 passed. Independent Ponytail review found
no blocker and passed twenty additional compatibility probes.

The same final wheel passed **575 tests**, zero failures or skips, plus the
Bundle guide on Databricks serverless PERFORMANCE_OPTIMIZED:
[run 601530407504441](https://dbc-45604623-c18b.cloud.databricks.com/jobs/957775592488736/runs/601530407504441)
finished **TERMINATED / SUCCESS**. All 552 installed Python sources and twenty-three
test/guide/support assets matched the manifest; local collected and remote passed
test-node sets match exactly. Pytest took 39.95 seconds with 60 warnings; the full
run took 95.725 seconds. Wheel SHA256:
`b7be400ab3d7ce798fa48d6c4f02c4b3ddc457a36f06c85c9139e3b99cf53ec0`.
This validates native Python fit/save/load/apply/predict and context diagnostics.
Spark UDF, REST and `ai_query` admission remain unchanged; no tables, registered
models or endpoints were created.

### Task211 local median declarations

SimpleImputer and GroupImputer now declare saved median application as `row`
on pandas and Polars. The new entries use execution kind `local` and no codec;
the existing pandas `python_batch` and Spark declarations are unchanged. The
node-owned validators inspect saved medians and bind the recipe to the artifact.
Inference still fills from training statistics, including saved group fallbacks.
No fit/apply implementation, arithmetic, artifact field or generic resolver changed.

Group median uses the existing numeric group/fallback checks. Simple median
reuses the existing artifact shape/count checks and requires finite numeric or
`None` fills. A pandas wholly-null fit that produces an empty artifact remains
`unknown`; a valid Polars `None` median is inspectable and leaves existing missing
values unchanged. Unsupported nonfinite/nonnumeric median values and a conflicting
configured constant are
rejected. The SimpleImputer `mode` recipe alias remains outside this reviewed
subset. Existing modal validation still returns a canonical Python strategy
string when given a NumPy string scalar.

Local context does not authorize partition workers. Actual saved pandas median
pipelines for both owners remain rejected by the worker capability gate; portable
SimpleImputer encoding still accepts only its existing mean/constant vocabulary.
No model refit or artifact migration is required for this metadata-only extension.

All **528 local tests** in thirteen explicit affected files passed (5 warnings,
46.60 seconds). The union includes 49 focused median cases and four median
fit/save/fresh-process load/probe/predict cases across both engines, with fit
disabled during replay and actual worker rejection checked. Full Ruff, formatting,
CI Ty and Lizard CCN <= 10 passed. Independent Ponytail review retained all prior
declaration entries and nineteen baseline validator outcomes; its normalization
finding was reproduced and corrected before the final union.

The matching wheel and 528-test package are prepared for Databricks validation;
Task211 upload/run approval is pending. This does not claim Spark UDF or endpoint
validation.
