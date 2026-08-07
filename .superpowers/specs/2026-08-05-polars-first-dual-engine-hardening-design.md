# Polars-First Core Migration and Dual-Engine Hardening Design

## Context

The v0.7.4 documentation review found that the silhouette sampler's
"bounded intermediate memory" claim is not currently true for
high-cardinality labels. `calculate_clustering_metrics()` calls
`pd.unique(labels_np)`, and `_select_silhouette_sample_indices()` builds a
representative dictionary for every label before it validates the cap. Both
operations can retain state proportional to the number of distinct labels.

This is not a case where changing pandas to Polars solves the problem. The
evaluation path deliberately accepts Pandas and Polars inputs, then converts
them through `SklearnBridge` to NumPy for scikit-learn metrics. At the
sampler boundary, labels are already a NumPy array. A Polars distinct-count
operation would still inspect/materialize the same label population and would
not provide the required bounded-state guarantee.

Skyulf will remain a dual-engine library: Pandas and Polars stay supported
public inputs. Internal code should prefer Polars or NumPy where doing so
preserves semantics and removes an avoidable conversion, but not through a
mechanical repository-wide pandas replacement.

The approved boundary is all production `skyulf-core` modules. That means
every production Pandas use is inventoried and every proven eligible internal
conversion is migrated. It does not mean every import is removed: public
Pandas compatibility, third-party contracts, and NumPy/scikit-learn
boundaries remain explicit, supported decisions.

## Goals

1. Make silhouette sample selection genuinely bounded by the configured cap
   for additional label-selection state.
2. Remove the unnecessary pandas unique-count operation from the sampler path.
3. Preserve deterministic representative coverage and reservoir sampling for
   inputs whose cluster cardinality fits the cap.
4. Define explicit, predictable behavior for cluster cardinality above the
   sample cap.
5. Inventory every production Core Pandas use and migrate each proven eligible
   internal conversion without changing public engine support.
6. Document every retained Pandas use as a public/third-party compatibility,
   NumPy/scikit-learn boundary, or evidence-gated candidate.

## Non-Goals

- Removing Pandas as a supported input engine.
- Replacing every `import pandas` mechanically.
- Changing frontend/backend node parameters, output schemas, or model APIs.
- Making Calinski-Harabasz or Davies-Bouldin sampled metrics.
- Introducing a new DataFrame abstraction in this work.
- Treating the number of removed imports as a success metric.

## Current Evidence

The initial production Core audit establishes the following migration shape:

- `modeling/_evaluation/metrics.py` has one avoidable Pandas operation:
  `pd.unique(labels_np)` after the sklearn bridge. The sampler repair removes
  it with a bounded NumPy/Python scan; the rest of the metric path is a
  NumPy/scikit-learn boundary.
- `profiling/expect.py` converts every non-Pandas frame solely to implement
  column, null, range, and duplicate checks. It is the first high-confidence
  Polars-native migration, provided it preserves Pandas null/NaN and error
  semantics.
- Preprocessing correlation selection, clustering numeric feature filtering,
  and profiling multivariate fallback are conversion candidates, but require
  parity and benchmark evidence before migration because dtype, null, feature
  ordering, and model-result semantics can differ by engine.
- Engine protocols/wrappers, sklearn/imblearn/SHAP boundaries, public
  prediction/report shapes, catalog/split contracts, and documented Pandas
  workflows are retained compatibility boundaries unless later evidence shows
  a safe alternative.

## Immediate Sampler Repair

### Engine boundary

`calculate_clustering_metrics()` will retain its current public input
contract. It will use `SklearnBridge` to obtain NumPy data for scikit-learn,
then use a NumPy/Python-only bounded label scan. No Polars frame is created
at this boundary, and Pandas remains in the module where its existing
type-level/public compatibility uses require it.

### Bounded cluster analysis

An internal helper will scan labels in row order and retain the first index
for each distinct label, up to `silhouette_sample_size` distinct labels.

- For each new label while fewer than the cap are retained, record its first
  index.
- If a new label appears after the cap has been reached, stop immediately and
  raise `ValueError`. The message will explain that the silhouette cap cannot
  represent more than the configured number of clusters and advise increasing
  the cap or reducing cluster cardinality.
- If the scan finishes at exactly the cap on an input larger than the cap,
  preserve the existing validation that a silhouette sample must contain more
  rows than clusters.
- For inputs at or below the cap, the scan can retain all distinct labels;
  this is still bounded by the configured cap.

The helper returns exact cluster count and representative indices only when
the input can be represented within the configured cap. Consequently,
an all-unique input larger than the cap now raises early instead of returning
only `n_clusters`; this is intentional and makes the resource boundary
consistent.

### Sampling behavior

The existing deterministic reservoir phase will reuse the bounded
representatives:

- Every feasible predicted cluster remains represented.
- The selected rows remain unique.
- `random_state` continues to determine the optional reservoir rows.
- `silhouette_sample_size` continues to report the actual scored rows.
- Calinski-Harabasz and Davies-Bouldin remain full-input for representable
  cluster cardinalities.

Additional memory used by label selection is `O(min(cluster_count, cap))`,
not `O(row_count)`, excluding the caller-provided `labels` and feature matrix.
The scan remains `O(row_count)` time.

## Compatibility and Error Contract

Existing behavior remains unchanged for:

- invalid caps below two;
- one-cluster data, where clustering quality metrics are undefined;
- representable cluster counts below the cap;
- exact-cap inputs that currently fail because there is no spare sample row;
- Pandas and Polars feature inputs flowing through `SklearnBridge`.

The only intentional behavior change is early `ValueError` when an input
contains more than the configured cap of distinct predicted labels, including
the formerly silent all-unique case. The error occurs before allocating
state for every distinct label.

## Verification

Test-first implementation will add or extend focused clustering tests for:

1. A high-cardinality label array that raises after observing cap-plus-one
   labels and does not retain row-proportional intermediate state.
2. The existing deterministic, representative, no-duplicate reservoir
   behavior for representable labels.
3. Invalid and exact-cap error cases.
4. Full-input behavior below the cap and single-cluster behavior.
5. Pandas and Polars feature-frame parity where the existing clustering
   fixture supports both engines.

The first Polars-native expectation wave will add parity tests for Pandas, raw
Polars, and `SkyulfPolarsWrapper` inputs. It covers column existence, nulls,
float NaN, inclusive and exclusive bounds, duplicate rows, missing columns,
unsupported frames, and exact `ExpectationError` messages.

After focused tests, run the repository's required Python Ruff, formatter,
and Ty checks plus the relevant broader Core evaluation/profiling suites.
Re-run the documentation-range review that identified the unsupported memory
claim.

No frontend check is required unless planning reveals a public configuration
or result-schema change; the current repair is internal resource behavior
with stable parameters and output keys.

## Documentation and Release Handling

Do not amend the existing `60a48fee` release-note commit. The sampler repair,
and only that repair, is folded into the unreleased v0.7.4 work because it
makes the reviewed bounded-memory statement true. Broader Polars migrations
receive later release entries. Update the ignored platform roadmap's source
citations and narrative only if the final sampler implementation changes the
documented behavior or line references. Any additional user-facing explanation
of the high-cardinality error must be precise and not imply that all
evaluation memory is bounded.

## Core Pandas/Polars Inventory and Migration Waves

The inventory is a mandatory implementation preflight, not an audit-only end
state. Each production Pandas use is classified as one of:

| Category | Decision rule |
| --- | --- |
| Polars-native opportunity | A raw/wrapped Polars path can perform the operation without a semantic downgrade or forced conversion. |
| NumPy/scikit-learn boundary | Data must be NumPy for an external metric/model API; Polars adds no value at this point. |
| Required Pandas compatibility | Public API, documented user workflow, or third-party integration requires Pandas behavior. |
| Needs evidence | Conversion cost, output semantics, or dependency behavior is unclear; benchmark and contract tests are required before a migration decision. |

For each candidate, the audit records source path, call path, current
conversion direction, expected benefit, compatibility risk, tests needed,
benchmark needed, and smallest next decision. Priority favors forced
`to_pandas()` paths and raw-Polars failures, not imports in annotations or
well-defined scikit-learn boundaries.

The implementation sequence is:

| Wave | Scope | Promotion gate |
| --- | --- | --- |
| 0 | Bounded NumPy sampler repair | Focused memory/error/determinism tests; Core static checks; final review of v0.7.4 claim. |
| 1 | Polars-native profiling expectations | Pandas/raw-Polars/wrapper parity, including NaN/null and error-message behavior. |
| 2 | Correlation selection, clustering numeric filtering, and multivariate fallback candidates | Per-candidate parity matrix and representative benchmark demonstrate a real benefit without schema/model drift. |
| 3 | All remaining inventory entries | Migrate only if evidence passes; otherwise document the retained boundary and its rationale. |

Every Wave 2 or Wave 3 candidate requires a scoped implementation decision,
Pandas/Polars parity, schema/error compatibility, conversion visibility, and
representative performance validation before code changes.

## Success Criteria

- The silhouette implementation no longer uses pandas distinct counting or
  retains label-selection state proportional to row count.
- Inputs above the configured cardinality cap fail early with a clear,
  deterministic error.
- Existing representable sampling behavior remains deterministic and
  representative.
- The v0.7.4 documentation claim is supported by the live implementation.
- The production Core inventory covers every Pandas import/use, identifies
  all forced conversion paths, and records a rationale for every retained
  use.
- Simple profiling expectations no longer convert raw or wrapped Polars
  frames to Pandas and remain behaviorally identical to their Pandas path.
- Every later migration is accepted only with evidence-backed parity and
  benchmark results, without weakening dual-engine compatibility.
