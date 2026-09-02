# Skyulf Deep Audit (Opus) — Frontend: node config layer & application

> Part of [`opus_core_analysis`](./README.md). Severity: 🔴 Critical · 🟠 High · 🟡 Medium · ⚪ Low. Finding IDs use the `OC-` prefix.

**Scope:** `frontend/ml-canvas/src` — ~400 files / 71,666 lines. All ~34 registered node components, `pipelineConverter.ts`, state stores, API client, canvas, pages, hooks, build config.

---

## Findings

### OC-53
### 🟡 Medium — `select_from_model`'s `max_features` is a Python-only, UI-unreachable param

**Files:** `frontend/ml-canvas/src/modules/nodes/processing/FeatureSelectionNode.tsx:493-503`
vs `skyulf-core/skyulf/preprocessing/feature_selection/_common.py:226-234`, `model_based.py:41`

The UI's Select-From-Model form renders only a `threshold` field. The backend
constructor also reads `config.get("max_features")`, and the node's
`@node_meta(params=...)` declares it:

```python
# _common.py:226-234
threshold = config.get("threshold", "mean")
...
return SelectFromModel(estimator, threshold=threshold,
                       max_features=config.get("max_features"), ...)
```

**Impact:** Users cannot cap the selected feature count from the canvas. The
feature exists only for direct API callers. This is the *inverse* of the usual
drift direction — a backend capability with no UI affordance, cf. [OC-06](./01-cross-cutting.md).

**Fix:** Add a `max_features` numeric input to the `select_from_model` branch.

**Confidence:** 9/10

---

### OC-54
### 🟡 Medium — `DebugNode` is dead code that would silently no-op if ever wired up

**Files:** `src/modules/nodes/base/DebugNode.tsx:26-48`, `src/core/registry/init.ts`,
`src/core/utils/pipelineConverter.ts:557-561`

`DebugNode` declares `definitionType: 'debug_node'` but is never
`registry.register()`'d in `init.ts`, so it cannot be placed on the canvas.
Worse, even if registered, `pipelineConverter.ts` has no `'debug_node'` branch —
it would fall through to `console.warn('Unknown node type: ...')` and
`stepType = 'Unknown'`.

```console
$ grep -rn "DebugNode" src
src/modules/nodes/base/DebugNode.tsx:27    # only the definition itself
```

**Impact:** Currently harmless (unreachable), but a maintenance trap: registering
it later would produce a node that silently disappears from the pipeline.

**Fix:** Delete the file, or add the registry registration **and** a converter
branch together.

**Confidence:** 9/10

---

### OC-55
### 🟡 Medium — `tsc --noEmit` fails: `mermaid` declared but not installed

**File:** `frontend/ml-canvas/src/components/**/MermaidDiagram.tsx` (+ its test)

```console
$ npx tsc --project tsconfig.json --noEmit
exit 2 — 4 errors, all TS2307: Cannot find module 'mermaid'
```

`package.json` declares `"mermaid": "^11.17.2"`, but `node_modules/mermaid` does
not exist. This is a stale/incomplete dependency install rather than a source
defect — but it means **the documented type-check command does not currently
pass on a fresh checkout**, so the repo's own lint/type gate cannot be satisfied
without a `npm install` that reconciles the lockfile.

**Impact:** Type-checking is effectively disabled for anyone who hits this state;
real type regressions would hide behind the 4 pre-existing module errors.

**Fix:** Reconcile `package.json`/lockfile and reinstall; add the `tsc --noEmit`
step to CI so this state cannot persist.

**Confidence:** 9/10

---

### OC-56
### ⚪ Low — `useSchemaPreview` does not cancel in-flight requests on unmount

**File:** `src/core/hooks/useSchemaPreview.ts:30-77`

The debounce timer is cleared on cleanup, but once
`previewPipelineSchema(config)` has started there is no `AbortController`; only
a `requestIdRef` guard discards the response after arrival.

```ts
const handle = window.setTimeout(() => {
  const myRequestId = ++requestIdRef.current;
  void (async () => {
    const response = await previewPipelineSchema(config);
    ...
  })();
}, DEBOUNCE_MS);
return () => window.clearTimeout(handle);   // does not abort the in-flight fetch
```

**Impact:** Wasted network and backend schema-graph work during rapid canvas
edits. Correctness is preserved by the staleness guard.

**Fix:** Thread an `AbortController` through `previewPipelineSchema` and abort in
cleanup.

**Confidence:** 8/10

---

### OC-57
### ⚪ Low — `any`-typed chart props bypass type safety in EDA components

**Files:** `src/components/eda/ThreeDScatterPlot.tsx:35-98` (`traces: any[]`,
`layout={{...} as any}`), `src/components/eda/CanvasScatterPlot.tsx:111-121`
(`context: any`)

Plotly/Chart.js prop and callback shapes are cast to `any` instead of the
libraries' typed interfaces. Because these components are fed **dynamic EDA
data**, a malformed trace/layout shape is exactly the kind of error types would
catch.

**Fix:** Use Plotly's `Data`/`Layout` and Chart.js's `TooltipItem<'scatter'>`.

**Confidence:** 9/10

---

> **Merged, not re-filed:** this agent independently rediscovered the
> `AliasReplacement` `punctuation` silent no-op at 10/10 confidence with a full
> backend trace (`_resolve_alias_type` passes `"punctuation"` through unmatched →
> `_resolve_alias_mapping` returns `{}` → `mapping.get(clean, val)` returns `val`
> unchanged). That is already filed as **[OC-19](./02-encoding-cleaning-imputation-scaling.md)**;
> the independent confirmation raises its confidence to 10/10.
> It also re-verified **[OC-13](./02-encoding-cleaning-imputation-scaling.md)**
> (Drop Rows), confirming the "Missing Value Threshold (%)" slider is *never read
> at all* and the backend behaves as if "drop rows with ANY missing value" were
> permanently on, regardless of the checkbox.

---

## Complete node parity matrix

Legend: **bold** = mismatch. "Already-confirmed" entries are filed under OC-13/14/15/19/25/26 elsewhere in this audit.

| Node component | definitionType | UI param keys & enum values | Python keys/values actually read | Mismatch |
|---|---|---|---|---|
| `DatasetNode.tsx` | `dataset_node` | `datasetId` | converter renames → `dataset_id`; resolved in `backend/ml_pipeline/resolution.py:18-94` | None (intentional rename) |
| `TrainTestSplitNode.tsx` | `TrainTestSplitter` | `test_size`, `validation_size`, `random_state`, `stratify`, `shuffle`, `target_column` | same (`split.py:103-184`) | None |
| `FeatureTargetSplitNode.tsx` | `feature_target_split` | `target_column` | same (`split.py:427-476`) | None |
| `DataPreviewNode.ts` | `data_preview` | none | converter emits `params={}` (`pipelineConverter.ts:557-559`) | None |
| `CastTypeNode.tsx` | `casting` | `column_types`; float/float32/int/int8-64/uint8-64/string/category/bool/datetime | same aliases (`casting.py:21-91,118-180,322-349`) | None |
| `DropColumnsNode.tsx` | `drop_missing_columns` | `columns`, `missing_threshold` | same (`drop_columns.py:68-95`) | None |
| `DropRowsNode.tsx` | `drop_missing_rows` | `drop_if_any_missing`, `missing_threshold` | `subset`/`how`/`threshold` (`drop_rows.py:38-103`) | **OC-13** |
| `DeduplicationNode.tsx` | `deduplicate` | `subset`, `keep`: first/last/none | same; `"none"`→`False` | None |
| `MissingIndicatorNode.tsx` | `MissingIndicator` | `columns`, `flag_suffix` (`_was_missing`) | same; backend default `_missing` differs but is unreachable | None functional |
| `DebugNode.tsx` | `debug_node` | `message` | unregistered; would hit `Unknown` branch | **OC-54** |
| Imputation / Simple | `SimpleImputer` | `strategy` mean/median/most_frequent/constant, `fill_value`, `columns` | same | None |
| Imputation / KNN | `KNNImputer` | `n_neighbors`, `weights`, `columns` | same | None |
| Imputation / Iterative | `IterativeImputer` | `estimator` bayesian_ridge/decision_tree/extra_trees/knn | matches only `DecisionTree`/`ExtraTrees`/`KNeighbors` (`_common.py:96-105`) | **OC-14** |
| Scaling / Standard | `StandardScaler` | `with_mean`, `with_std`, `columns` | same | None |
| Scaling / MinMax | `MinMaxScaler` | `feature_range_min`, `feature_range_max` | `feature_range` tuple | **OC-15** |
| Scaling / Robust | `RobustScaler` | `quantile_range_min`, `quantile_range_max`, `with_centering`, `with_scaling` | `quantile_range` tuple | **OC-15** |
| Scaling / MaxAbs | `MaxAbsScaler` | `columns` | same | None |
| Outliers | `IQR`/`ZScore`/`Winsorize`/`EllipticEnvelope` | `multiplier`/`threshold`/`lower_percentile`+`upper_percentile`/`contamination` | same | None |
| Encoding ×7 | OneHot/Dummy/Label/Ordinal/Target/Hash/WOE | drop_first, handle_unknown, max_categories, include_missing, missing_code, target_column, categories_order, unknown_value, smooth, target_type, n_features, regularization | same per encoder | None |
| `TransformationNode.tsx` | `GeneralTransformation` | yeo-johnson/box-cox/log/square_root/cube_root/reciprocal/square/exponential, `standardize`, `clip_threshold` | same dispatch; `standardize` ignored for power methods | **OC-27** |
| `BinningNode.tsx` | `GeneralBinning` | `strategy` equal_width/equal_frequency/kmeans/custom; `label_format`; `custom_bins`, `custom_labels` | same + `uniform`/`kbins` aliases | None |
| `ResamplingNode.tsx` | `Oversampling`/`Undersampling` | 7 over-, 4 under-sampling methods + tuning | same (`resampling.py`) | None |
| `FeatureGenerationNode.tsx` | `FeatureMath` | arithmetic/datetime_extract/ratio/similarity/group_agg | same dispatch | None new |
| `PolynomialFeaturesNode.tsx` | `PolynomialFeatures` | columns, degree, interaction_only, include_bias, output_prefix, include_input_features | same | None |
| `FeatureInteractionNode.tsx` | `FeatureInteraction` | columns, degree 2-4, interaction_only, include_bias | same | None (but see OC-33) |
| `TimeSeriesNode.tsx` | `RollingAggregate`/`DateFeatures`/`LagFeatures` | lag/rolling/date + aggs + lag options | same | None new |
| `FeatureSelectionNode.tsx` | `feature_selection` | 10 methods, `threshold` | also reads **`max_features`**, `n_features_to_select` | **OC-53**, **OC-25** |
| `TextCleaningNode.tsx` | `TextCleaning` | trim/case/remove_special/regex + sub-modes | same enums | None |
| Count / Tfidf vectorizer | `count_vectorizer`/`tfidf_vectorizer` | lowercase, stop_words, max_features, min_df, max_df, ngram_range, binary/sublinear_tf | same | None |
| `HashingVectorizerNode` | `hashing_vectorizer` | n_features, `norm` l1/l2/**none**, alternate_sign | `norm` must be `None` not `"none"` | **OC-26** |
| `TokenizerNode` | `tokenizer` | ngram_range, analyzer word/char/char_wb, add_token_count | same | None |
| `SentenceEmbedderNode` | `sentence_embedder` | model_name (3 presets), normalize | accepts arbitrary string | None |
| `ValueReplacementNode.ts` | `ValueReplacement` | columns, `replacements[]` | `mapping`/`replacements`/`to_replace`/`value` | None (but see OC-20) |
| `AliasReplacementNode.tsx` | `AliasReplacement` | custom/canonicalize_country_codes/normalize_boolean/**punctuation** | only boolean/country/custom (`alias.py:20-52`) | **OC-19** |
| `InvalidValueReplacementNode.tsx` | `InvalidValueReplacement` | negative_to_nan/zero_to_nan/percentage_bounds/age_bounds/custom_range | normalizes UI aliases correctly | None (previously broken, now fixed) |
| `ClassificationNode.tsx` | `classification` | `model_type` (registry-driven) | matches `hyperparameters/_registry.py` | None new |
| `RegressionNode.tsx` | `regression` | `model_type` | matches `_registry.py` | None new |
| `TextClassificationNode.tsx` | `text_classification` | `model_type` (default `multinomial_nb`) | matches text-tagged ids | None |
| `EnsembleNode.ts` | `EnsembleNode` | voting/stacking, soft/hard, final_estimator, sigmoid/isotonic, 11-13 base keys per task | matches `ensemble.py:87-137`, `_ensemble.py`; both independent base-key lists agree | None |
| `SegmentationNode.tsx` | `SegmentationNode` | `model_type` kmeans/minibatch_kmeans/gaussian_mixture/birch | matches `clustering.py` | None |

> Note: `SegmentationNode.tsx` **does** list all four clustering algorithms here.
> The [OC-06](./01-cross-cutting.md) reachability gap for `birch` /
> `gaussian_mixture` / `minibatch_kmeans` was derived from a registry-id string
> diff; this matrix supersedes it for those three ids. `CustomBinning`,
> `DataSnapshot`, `DatasetProfile`, `FeatureGeneration`, `GeoDistance` and
> `H3Index` remain unreachable.

---

## Build, type & lint status

| Command | Real result |
|---|---|
| `npx tsc --project tsconfig.json --noEmit` | **Exit 2** — 4 × `TS2307: Cannot find module 'mermaid'`. No other type errors. See OC-55. |
| `npm run lint` (`--report-unused-disable-directives --max-warnings 0`) | **Exit 0**, fully clean across `src/` |
| `npx eslint src --ext ts,tsx` (spot check) | Clean, consistent with the full run |

---

## What I checked and found sound

- All imputation / scaling / outlier / encoding key names match 1:1 except the
  already-filed OC-14 and OC-15.
- All transformation, binning, resampling, feature-generation, polynomial,
  interaction, time-series and feature-selection enum lists match Python's
  dispatch, apart from the new OC-53 gap.
- Text cleaning, vectorizer, tokenizer, sentence-embedder and value-replacement
  param shapes match, apart from OC-26.
- **Modeling is clean:** every `model_type` string across Classification,
  Regression, TextClassification, Ensemble and Segmentation resolves against the
  backend model registry. Ensemble maintains *two independent* base-estimator key
  lists (in-node chips vs. auto-detection from wired nodes) and they are
  consistent with each other and with `ensemble.py`.
- App-layer state stores (`useGraphStore`, `useJobStore`) use immutable update
  patterns — no direct-mutation bugs found.
- **No `dangerouslySetInnerHTML`, no `eval`, no `new Function`, no hardcoded
  secrets** in the reviewed app-layer files.
- `useSchemaPreview` already guards against stale responses via a request-id ref;
  it simply doesn't abort the underlying fetch (OC-56).
- Lint passes with `--max-warnings 0`, which also proves there are **no stale
  `eslint-disable` comments**.

---

## Improvement opportunities (not defects)

- **Add an automated schema-diff test** comparing UI param keys against Python
  `config.get(...)` sites. Given how many instances of this bug class exist
  across ~34 node types, this single test would have caught OC-13, OC-14, OC-15,
  OC-19, OC-25, OC-26 and OC-53. See [R1 in the master report](../opus_core_analysis.md).
- Delete or properly wire up `DebugNode.tsx`.
- Add `AbortController` support to `previewPipelineSchema`.
- Replace the `any` casts in the EDA chart components with library types.
- Add `tsc --noEmit` to CI so OC-55 cannot recur.
