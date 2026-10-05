# SM-56 inventory for SM-57 — partition-safe inference

Date: 2026-10-05. Inspected revision: `42bf826f`.
Status: source inventory complete for the listed built-in preprocessing families;
no Spark UDF route is certified by this note. Custom project code and estimators
require their own admission checks. Implementation plan: [SM-57](166-sm57-spark-udf-plan.md).

## Scope and distinctions

The user approved starting SM-57 after closing SM-23 batch monitoring. Training
remains local pandas; large inference should run the fitted pipeline through
`mlflow.pyfunc.spark_udf` on Spark workers. Existing fitted Polars artifacts must
not silently switch engines. Local inference remains available for suitable data.

There are two different artifact paths:

| Path | Current boundary | Consequence |
| --- | --- | --- |
| `inference/spark.py` Python pipeline mode | Portable JSON feature state; SimpleImputer mean/constant and StandardScaler only | Extending this route needs node codecs as well as execution declarations. |
| `inference/local_pipeline.py` + `integrations/mlflow/local_model.py` | Trusted pickle of the fitted pipeline; manifest scope `whole_frame_local` | Additional fitted objects can travel without a portable JSON codec, but that does not prove partition safety. |

`core/capabilities.py` already describes engine, operation, execution kind, row
effect and context. Existing declarations on SimpleImputer and StandardScaler
describe native Spark operations. They are not an implicit pandas-worker
certificate. `require_capability` currently checks engine/operation/configuration;
a partition-safety gate must also check execution kind, row preservation and
row context. Defaults and fitted configuration must be normalized explicitly.

## Current scoring lifecycle

| Location | Observed behavior | SM-57 requirement |
| --- | --- | --- |
| `integrations/databricks/local_workflow.py:_run_scoring_action` | Pins champion version, prepares local predictor, calls local incremental batch, switches rebuild view only after success | Preserve the pin and publication ordering while dispatching an explicit inference mode. |
| `local_incremental.py:bounded_frame` | `.limit(max_rows + 1).toLocalIterator()` collects the selected input into a driver pandas frame | Spark mode must bypass this conversion; collecting bounded scalar diagnostics is distinct from collecting input rows. |
| `local_incremental.py:_incremental_prediction_bridge` | Predicts locally and rebuilds a Spark DataFrame from Python tuples | Keep predictions and keys in Spark through the sink. |
| `local_incremental.py:_commit_increment` | Rechecks table identities/latest target commit, writes transactional Delta metadata, verifies receipt | Reuse these guarantees; a UDF must not bypass idempotency, conflict or replay rules. |
| `model_set_batch.py` | Uses the same bounded input path and publishes the complete set atomically | Model-set needs its own distributed handoff, component gates and combined-output validation. |
| `inference/local_scoring.py` | Applies saved eligibility/output rules around prediction | Feature safety alone is insufficient: callbacks and reused pre-split rules must also be admitted explicitly. |
| `inference/model_set_scoring.py` | Branch rules, key/schema validation, combined outputs and optional temporal history | Reject unsupported callbacks/history before executor work; preserve one outcome per key. |
| `integrations/mlflow/_nullable_transport.py` | Nullable integral/Boolean transport is encoded before Arrow and restored before FE | Reuse named signature transport, including integers beyond `2**53`; do not cast them to float. |
| `templates/databricks/schema/project.json` | One `engine` question says training and prediction use the same library | Add a separate one-time inference-capacity choice; persist inference mode independently. |

`workflow_config.py` currently has no inference-mode field. The generated schema
comes from topic files; edit those and regenerate with `build_schema.py`, rather
than hand-editing the generated compact JSON. Existing projects must retain the
local default. Existing row/byte budgets still protect local training.

## Node inventory

The family-level findings below must distinguish source inspection from executed
parity evidence. A candidate is not a supported node until its actual apply and
pipeline composition pass repartition, null, unknown-category and dtype cases.

Paths in the following tables are relative to `skyulf-core/skyulf/preprocessing/`.
The inspected appliers preserve rows unless explicitly described otherwise.

| Family / nodes | Actual apply behavior | Admission boundary |
| --- | --- | --- |
| OneHotEncoder | Fitted sklearn encoder and output names; dense fixed-category transform (`encoding/one_hot.py:90`) | Candidate with complete input columns, fitted unknown policy, artifact version and output-width tests. |
| DummyEncoder | Fitted category lists, unknown/null all-zero (`encoding/dummy.py:82`) | Candidate; canonical category-key version and upstream dtype stability matter. |
| LabelEncoder / OrdinalEncoder | Fitted encoder lookup/transform, target path skipped when y is absent (`encoding/label.py:90`, `ordinal.py:61`) | Candidate with exact class/null/unknown behavior; missing input subsets must not alter sklearn width. |
| TargetEncoder / WOEEncoder | Fitted inference encodings/mappings; no new-population group fit (`encoding/target.py:34`, `woe.py:43`) | Candidate after multiclass width, key identity and null tests; training cross-fit output is not the inference oracle. |
| HashEncoder | Deterministic per-value hashing; batch unique lookup is an optimization (`encoding/hash.py:29`) | Candidate for supported canonical-key version; legacy stringify requires separate evidence. |
| Count / TF-IDF / Hashing vectorizers | Saved sklearn transform; vocabulary/IDF/options fixed (`vectorization/_common.py:230`) | Conditional candidate; complete text sources, dense width/memory and exact worker dependencies. |
| Tokenizer | Analyzer rebuilt from fixed config (`vectorization/tokenizer.py:34`) | Conditional candidate; empty/all-null/string conversion and output dtype tests. |
| Sentence embedder | Lazily loads external model by name (`vectorization/sentence_embedder.py:75`) | Defer: weights/revision/device are not captured by pickling a name; model memory and download lifecycle unresolved. |
| Casting | Null-dependent integral dtype, inferred datetime dtype and broad failure fallback (`casting.py:357`, `:388`) | No blanket support; mixed timezone and null batches do not guarantee one output schema. |
| General / Custom / KBins binning | Fitted edges, but per-column apply exception skips that whole batch's generated output (`bucketing.py:302`) | Restrict to validated numeric inputs or correct failure semantics; mixed-invalid case is confirmed unsafe. |
| GeoDistance | Per-row arithmetic (`geo/distance.py:42`) | Conditional candidate; complete columns, nullable coordinates and fixed output schema. |
| H3Index | Per-row optional-library call; invalid coordinates become None (`geo/h3_index.py:33`) | Defer until dependency, empty/all-null and output-schema parity pass. |
| AliasReplacement / TextCleaning | Fixed mappings/operations with runtime dtype selection (`cleaning/alias.py:46`, `text.py:108`) | Conditional; upstream batch-dependent dtypes can change which operation runs. |
| ValueReplacement | Mapping key coercion follows runtime dtype; `infer_objects()` changes that dtype (`cleaning/value_replacement.py:147`) | Confirmed composition hazard; no unconditional row-local certificate. |
| InvalidValueReplacement | Fixed numeric masks and replacements (`cleaning/invalid_value.py:132`) | Conditional; null/large-integral/nonfinite and dtype-widening composition tests. |

| Family / nodes | Actual apply behavior | Admission boundary |
| --- | --- | --- |
| Standard / MinMax / MaxAbs / Robust scalers | Use fitted arrays; no apply-time population statistics (`scaling/standard.py:99`, `minmax.py:36`, `maxabs.py:37`, `robust.py:40`) | First candidates; validate complete state, input schema and nullable/Decimal behavior. No standalone Normalizer was found in the inspected families. |
| SimpleImputer, local strategies | Uses saved fill values for mean/median/mode/constant (`imputation/simple.py:107`) | Candidate beyond the portable codec's mean/constant subset; all-null/category/missing-column tests required. |
| GroupImputer | Saved group lookup plus saved global fallback (`imputation/group.py:136`) | Candidate with typed-key, unknown/null group and dtype tests; does not aggregate new input. |
| KNN / Iterative imputation | Stored sklearn transform and training donor/estimator state (`imputation/knn.py:33`, `iterative.py:36`) | Later candidates; state memory/dependencies, complete fitted columns, deterministic configuration and repeated-call parity. JSON absence is not the blocker. |
| SimpleTransformation / GeneralTransformation simple methods | Elementwise log1p/sqrt/cbrt/reciprocal/square/clipped-exp (`transformations/_ops.py:23`) | Candidate restricted to explicit known numeric methods. |
| PowerTransformer / GeneralTransformation power methods | Stored lambda/scaler with whole-matrix/column exception fallback (`transformations/power.py:111`, `general.py:60`) | Confirmed unsafe for unrestricted batches; reject initial admission. |
| FeatureMath / FeatureGeneration arithmetic/ratio | Ordered operations with per-operation exception catches (`feature_generation/_pandas_ops.py:27`, `:240`) | Configuration-specific candidate, not a blanket family certificate. |
| FeatureMath / FeatureGeneration group_agg | Public apply requires fitted `group_agg_mapping` (`feature_generation/generation.py:81`); maps saved training keys (`_pandas_ops.py:207`) | Correct earlier queue inference: frozen mapping is a candidate; actual unfitted inference-time aggregates remain unsupported. Test mixed Boolean/numeric/null key identity. |
| Feature generation similarity/datetime | Saved similarity backend for new fits; UTC mixed datetime parsing (`_pandas_ops.py:135`, `:185`) | Later operation-specific candidate with pinned dependency and output schema. |
| PolynomialFeatures | Recreates/fits sklearn PolynomialFeatures during apply using schema/config, not new statistical estimates (`feature_generation/polynomial.py:20`) | Later candidate with empty/invalid input, width, names and expansion-memory tests; apply-time fit alone is not proof of batch dependence. |
| FeatureInteraction | Saved combinations, elementwise products and optional bias (`feature_generation/interaction.py:76`) | Candidate with complete numeric inputs and collision checks. |
| Variance / Univariate / ModelBased / Correlation selection | Drops stored column lists; no apply-time selection fit (`feature_selection/_common.py:116`, `correlation.py:23`) | Candidate with valid saved lists and recognized facade subtype; unknown subtype must not silently gain support. |
| Winsorize / ClipValues | Stored quantile bounds or explicit bounds; preserves rows (`outliers/winsorize.py:38`, `clip_values.py:97`) | Candidate; these are clipping operations, not new-batch quantiles or row filters. |
| IQR / ZScore / ManualBounds / EllipticEnvelope | Remove rows using fitted/configured predicates/models (`outliers/iqr.py:35`, `zscore.py:36`, `manual_bounds.py:57`, `elliptic.py:55`) | Reject active scalar-UDF chain: row-local predicates alone do not preserve output cardinality. |
| DropMissingColumns / MissingIndicator | Fitted drop list or row-wise missing flags (`drop_and_missing/drop_columns.py:16`, `missing_indicator.py:28`) | Candidate with saved columns, fixed output names and schema. |
| DropMissingRows / Deduplicate | Raw appliers remove rows; deduplication depends on current batch (`drop_and_missing/drop_rows.py:69`, `deduplicate.py:26`) | Raw active apply is unsupported; effective inference skip discussed below. |
| DateFeatures | New state uses explicit UTC/epoch unit (`time_series/date_features.py:49`, `:251`) | Later candidate limited to validated new artifacts and Arrow/timezone parity; legacy mixed-offset results can fail. |
| Lag / Rolling, batch or carry history | Requires other rows and potentially saved/session history; may sort/drop rows (`time_series/lag.py:47`, `rolling.py:68`, `_history_apply.py:65`) | Reject generic pyfunc partition execution. Pickled seed history does not coordinate worker batches or history commits. |
| ColumnFunction / FittedFunction | Arbitrary callback gets whole input batch; checks shape/index, not purity (`function_steps.py:49`, `:153`) | Default reject until separately reviewed code/version/dependencies, immutable state and batch parity establish support. |
| RowFilterFunction | Arbitrary callback produces removal mask (`function_steps.py:311`) | Raw active apply unsupported; effective skip discussed below. |

`FeatureEngineer._transform_steps(preserve_rows=True)` already skips Deduplicate,
DropMissingRows, RowFilterFunction, splitters and resamplers
(`preprocessing/pipeline.py:244`). Other outlier filters still execute and their
row changes are rejected. Certify the effective chain and recorded skip contract;
do not add new silent skips or reject a safe pipeline solely because a skipped
train-only step appears in its training recipe.

These modules import Polars even for pandas execution. The current artifact runtime
also pins Polars. Pandas worker execution therefore does not mean the Polars
package can be omitted from worker dependencies.

## Executed counterexamples

Independent read-only probes used current source at `42bf826f`; these are tiny
applier experiments, not a full pytest suite or Spark certification.

| Probe | Whole input | Concatenated singleton batches | Consequence |
| --- | --- | --- | --- |
| ValueReplacement mappings `{"1": 1}` then `{"1": 9}`, object input `["1", "foo"]` | `[1, "foo"]` | `[9, "foo"]` | Upstream inferred dtype changes downstream key coercion. |
| Binning object `[1, "bad"]`, edges `[0, 2]`, drop original | Original `x` remains | First batch produces `x_binned=0` | Output columns depend on other rows in the batch. |
| Casting mixed naive/aware datetime strings | Object dtype | Naive datetime and timezone-aware datetime dtypes | Arrow output schema cannot be assumed stable. |
| Box-Cox fitted on `[1,2,3,4,8]`, standardize false; input `[2,-1]` | `[2,-1]` | `[0.7025656653894778,-1]` | One invalid row makes a whole-column exception fallback; both PowerTransformer and GeneralTransformation reproduce it. |

These findings are reasons to reject/restrict a configuration or repair it with
focused parity tests. They do not change existing local semantics in this task.

The Box-Cox probe was independently reproduced by root after the source review:

```python
import pandas as pd
from skyulf.preprocessing.transformations.power import (
    PowerTransformerApplier,
    PowerTransformerCalculator,
)

train = pd.DataFrame({"x": [1., 2., 3., 4., 8.]})
score = pd.DataFrame({"x": [2., -1.]})
state = PowerTransformerCalculator().fit(
    train, {"columns": ["x"], "method": "box-cox", "standardize": False}
)
applier = PowerTransformerApplier()
whole = applier.apply(score, state)
split = pd.concat([applier.apply(score.iloc[[i]], state) for i in range(len(score))])
assert not whole.equals(split)
```

The equivalent GeneralTransformation configuration is
`{"transformations": [{"column": "x", "method": "box-cox", "standardize": False}]}`.
Root executed both using `.venv/Scripts/python.exe`; both produced the values in
the table. These are reproductions of unsupported batch behavior, not passing
correctness tests or fixes. The three dtype/binning probes came from the independent
encoding/cleaning inspection with pandas 2.3.2.

## Verification standard

Compare whole-frame fitted inference with concatenated singletons and mixed
batch sizes, then repeat through actual Spark partitions and Arrow. Include
empty/all-null partitions, unknown categories, nullable large integers, malformed
values and compositions of supported nodes. Check values, output columns/dtypes,
row order/key identity and fitted-state immutability. Unknown custom callbacks
must not gain support by appearing in a trusted pickle.

## Kickoff delivery record

Two independent read-only inspections covered the numeric/temporal and
encoding/text/geo families. Root traced the artifact, setup, single/model-set
publication and monitoring integration and reproduced both power counterexamples.
A separate final read-only review found no substantive issue in this inventory,
the implementation plan or queue status. No application behavior was edited for
SM-57, no Spark job was launched, and no new node is advertised as supported.
No pytest suite was required for these source/documentation changes; the executed
probes are explicitly reported as reproductions above. Markdown/staged checks
and ordinary pre-commit gates apply to the documentation commit.
