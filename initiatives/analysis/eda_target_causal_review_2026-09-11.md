# EDA target and causal graph review — 2026-09-11

The reported `species_encoded` label is reproducible with the bundled Iris
dataset. Core creates a temporary numeric target for analysis and publishes its
internal name as the graph label. The original `species` column is unchanged.
The review also found incorrect results behind that display issue.

Scope: target preparation, causal discovery and serialization, target
associations/interactions, exclusions, sampled outliers, and the web causal
graph. Source reviewed at `106c6b62` (0.8.20). This is a bounded review, not a
complete audit of every profiling module. The initial review changed no
production code or committed tests. The user's particular uploaded dataset was
not accessed. The subsequent authorized repair is recorded below.

## Repair follow-up

OC-235–245 are repaired; final verification is recorded below. Three subagents
handled causal conversion/selection, target statistics/outlier provenance, and
web rendering. The parent integrated target eligibility, exclusions, metadata,
backend serialization tests, documentation, and the production frontend build.

| Findings | Implemented behavior | Regression coverage |
|---|---|---|
| OC-235 | Preserve real causal-learn endpoint orientation | Real `GeneralGraph` in both node orders; public collider analysis |
| OC-236, OC-238 | Do not create nominal target codes for Pearson/Fisher-Z; retain categorical associations and report the omission reason | String, Boolean, native Categorical/Enum, integer classes, reordered Iris, task overrides, occupied encoded-name features |
| OC-237 | Use the actual eligible target for the graph cap; record all/target-correlation/variance selection | Public low-variance target retention, no-target feature-name controls, schema metadata |
| OC-239 | Exclusions override rules and inferred task type | Public repeated analysis and real backend configuration/serialization |
| OC-240 | Isolate group keys from aggregation output names | Both target/feature directions, all statistic aliases, `group`, null and constant controls |
| OC-241 | Carry sampled row positions separately from numeric features | 50,000/50,001 rows, filtered/unfiltered input, unchanged seeded sample and Isolation Forest scores |
| OC-242–244 | Preserve graph identities/edge endpoints, both bidirected arrowheads, and clear empty replacement results | Real Chromium on desktop/mobile; omission and selection messaging with legacy metadata fallback |

Categorical and Boolean variables no longer enter the Fisher-Z graph through
arbitrary codes. The same target eligibility decision controls target Pearson
output. Explicit Classification treats even a numeric target as categorical;
explicit Regression permits a physically numeric target with low cardinality.
An optional `causal_target_exclusion_reason` survives absent graphs. The frontend
keeps the explanatory view accessible and preserves older reports lacking these
new fields.

Outlier `index` means the zero-based row position in the current filtered input,
before sampling. Without filters it is the source dataframe position. Sampling
does not add a helper feature or change anomaly scores.

Independent review also found and repaired **OC-245 (🟡)**: the target name
`count` collided with the generated count in
`profiling/_analyzer/recommendations.py:_target_class_counts`, aborting public
analysis. The reproduction uses 120 rows, independent normal `x`/`z`, and
`count=[0,1,2]*40`; string targets and a literal `count` category also exercise
the collision. The fix isolates the grouping key from the generated count.
Three cases failed before repair; eight new cases plus two existing count
controls pass afterward, preserving null handling and a 5/100 imbalance ratio.

New durable regressions live in:

- `skyulf-core/tests/integration/test_profiling_analyzer.py` (causal cases).
- `skyulf-core/tests/integration/test_profiling_target_contract.py`.
- `skyulf-core/tests/integration/test_profiling_target_outlier_regressions.py`.
- `skyulf-core/tests/integration/test_profiling_column_name_collisions.py`
  (retains source-preservation assertions under the numeric-only contract).
- `tests/integration/test_eda_target_contract.py`.
- `frontend/ml-canvas/src/components/eda/tabs/CausalTab.test.tsx`.
- `frontend/ml-canvas/e2e/eda-causal-graph.spec.ts`.

The user guide and 0.8.20 changelog explain the new behavior. Reloading the app
updates rendering; **rerun Analyze** to regenerate saved numerical results,
causal graphs, rules, and outlier positions. Older stored analyses are not
silently rewritten.

### Verification limits and separate follow-up

The two Python test trees must run separately, matching CI: combining them in
one pytest invocation causes six duplicate test-module import errors. The
first separate Core run passed 7,053 tests but its optional sentence-embedding
case attempted a blocked Hugging Face HEAD request. That case passed with
`HF_HUB_OFFLINE=1` and `TRANSFORMERS_OFFLINE=1`, using the cached model; the
complete Core suite was then rerun offline.

The first backend run passed 3,611 tests and failed two tests on hardcoded
artifact paths outside the workspace. The artifact-routing test now uses
pytest's `tmp_path`, preserving its real routing assertions.

**OC-246 (🟡, open; separate test repair):** the other case,
`tests/integration/test_full_inference_pipeline.py`, is a legacy local-only
script skipped when its external `skyulf-mlflow` directory is absent (including
CI). A temporary path-only adaptation safely ran it inside the workspace and
showed `Pipeline Succeeded`, followed by `Model artifact is not a dict: class
tuple`, then an early successful return before `Simulating Inference`. Its
remaining checks also print instead of asserting. The adaptation was reverted:
this file is unchanged. It needs current artifact unpacking and genuine
inference assertions, and was deselected from final backend verification.
Its earlier pytest pass is **not** counted as proof of inference correctness.

The final queue has **38 open / 4 parked**: the ten original findings closed,
OC-245 was filed and fixed, and OC-246 is the separate open follow-up.

### Final verification

| Check | Result |
|---|---|
| Complete Core suite, cached model in offline mode | **7,054 passed / 80 skipped**, 433 warnings; all 3 snapshots passed |
| Backend suite, excluding unchanged legacy OC-246 | **3,612 passed / 1 deselected**, 3,200 warnings; all 7 snapshots passed |
| Complete frontend Vitest suite | **2,487 passed**, 191 files |
| Causal graph and omitted-target navigation in Chromium | **8 passed**, desktop/mobile |
| Repository Ruff and backend/Core Ty | Passed |
| Changed Python file formatting | Passed |
| Frontend ESLint and CCN ≤10 | Passed |
| Production TypeScript/Vite build and bundle size budgets | Passed; generated assets updated |
| Diff whitespace and tracker inventory | Passed |

Final suite commands from the repository root (run the frontend commands in
`frontend/ml-canvas`):

```powershell
$env:HF_HUB_OFFLINE = '1'
$env:TRANSFORMERS_OFFLINE = '1'
.venv/Scripts/python.exe -m pytest skyulf-core/tests -q --tb=short --basetemp=tmp_repro_artifacts/pytest-eda-fix-core-final -o cache_dir=tmp_repro_artifacts/pytest-eda-fix-core-cache
.venv/Scripts/python.exe -m pytest tests -q --tb=short --deselect tests/integration/test_full_inference_pipeline.py::test_full_inference_pipeline --basetemp=tmp_repro_artifacts/pytest-eda-fix-backend-final -o cache_dir=tmp_repro_artifacts/pytest-eda-fix-backend-cache
npm.cmd test -- --maxWorkers=4 --reporter=dot --silent
npm.cmd exec -- playwright test e2e/eda-causal-graph.spec.ts --workers=1
npm.cmd run lint
npm.cmd run complexity:check
npm.cmd run build
npm.cmd run size-check
```

Full local logs are `tmp_repro_artifacts/eda-fix-core-final.log`,
`eda-fix-backend-final.log`, `eda-fix-frontend-tests-final.log`, and
`eda-fix-frontend-build-final.log` in that directory. Browser red/green logs
are under `tmp_repro_artifacts/eda-causal-audit/`. Frontend tests/build required
the normal unsandboxed execution environment because esbuild could not resolve
the project configuration through sandbox-restricted parent directories.

## Initial review findings

Ten new findings: **4 🟠 / 5 🟡 / 1 ⚪**, filed as **OC-235–244**. Seven are
reproduced through public `EDAAnalyzer.analyze`; three are browser renderer
contract cases using mocked report/history responses. The latter prove how the
UI handles those payloads, not their frequency in actual discovery results.

| ID | Severity | Confirmed behavior | Area |
|---|---|---|---|
| OC-235 | 🟠 | Directed causal edges are serialized backwards | Core |
| OC-236 | 🟠 | Arbitrary category codes change target Pearson values and the causal graph when identical rows are reordered | Core |
| OC-237 | 🟡 | The 15-variable cap can omit the selected target | Core + UI explanation |
| OC-238 | ⚪ | Temporary target names appear as public graph and matrix labels | Core + UI labels |
| OC-239 | 🟡 | An excluded selected target still drives rule discovery and appears in rule text | Core + backend configuration |
| OC-240 | 🟡 | Target names such as `mean`, `n`, `min`, and `median` silently remove associations or box plots | Core |
| OC-241 | 🟠 | Sampled outliers report sample positions as source row indices | Core + UI display |
| OC-242 | 🟠 | Causal node ID sanitization merges distinct column names | Frontend |
| OC-243 | 🟡 | Bidirected causal edges render with only one arrowhead | Frontend |
| OC-244 | 🟡 | Loading an empty historical graph leaves the previous graph visible | Frontend |

### OC-235 — directed causal arrows are reversed

Location: `skyulf-core/skyulf/profiling/_analyzer/causal.py:86–94`.
`_build_causal_edges` reads `[j, i]` before `[i, j]`, reversing the endpoint pair
expected by `_causal_edge_from_endpoints`.

Runtime evidence uses the installed library, not a handwritten mock:

- `GeneralGraph.add_directed_edge(cause, effect)` produces `[[0, -1], [1, 0]]`
  and the library's edge is `cause --> effect`.
- Skyulf converts that same matrix into `effect → cause`.
- Public analysis of 1,200 observations, NumPy seed 117, independent normal
  `A`, `B`, and `C = A + B + Normal(0, 0.3)`, reproduces the error end to end:
  real PC returns `A → C ← B`; the returned profile contains `C → A` and `C → B`.

The matrix convention agrees with the
[official PC return-value documentation](https://causal-learn.readthedocs.io/en/latest/search_methods_index/Constraint-based%20causal%20discovery%20methods/PC.html).
The existing endpoint test at
`skyulf-core/tests/integration/test_profiling_analyzer.py:476` builds its fake
matrix with the same reversed assumption. Its passing result does not establish
compatibility with causal-learn. Repair should include a real-library graph
fixture and the public collider case, then correct the old test convention.

### OC-236 — nominal category codes change statistical results

Locations: `profiling/analyzer.py:329–354,380–395,518–525` and
`profiling/_analyzer/causal.py:126` under `skyulf-core/skyulf/`.

The temporary target is `cast(pl.Categorical).to_physical()`. Those codes are
then included in a Pearson matrix and PC with `indep_test="fisherz"`.
Two fresh Python processes analyzed the same 150 Iris observations, grouping
the rows by class in orders `0,1,2` and `0,2,1`. No observation was changed.

| Measurement | Class order 0,1,2 | Class order 0,2,1 |
|---|---|---|
| Class codes | setosa=0, versicolor=1, virginica=2 | setosa=0, virginica=1, versicolor=2 |
| Petal width / encoded-target Pearson | 0.9565473329 | 0.5803770334 |
| Petal length / encoded-target Pearson | 0.9490346990 | 0.6492418308 |
| Actual PC graph | 4 edges | 5 edges, with different connections/orientations |
| Petal width categorical association, η | 0.9637857283 | 0.9637857283 |

A wrapper captured the input codes and output of the real PC algorithm while
letting it execute unchanged. A control running PC on fixed numeric class codes
returned identical edge sets across the same row reorder. This isolates the
category-code assignment from ordinary row-order numerical noise. The correct
categorical association output also remains stable.

Sorting the category codes alone would only stabilize an arbitrary numeric
ordering. It would not give species a meaningful numerical distance. The
[Fisher-Z documentation](https://causal-learn.readthedocs.io/en/latest/independence_tests_index/fisherz.html)
describes a linear-Gaussian setting; our inference is that these nominal labels
need an explicitly supported categorical/mixed-data strategy, or exclusion from
this method with a clear explanation. Merely renaming the node does not repair
this finding.

### OC-237 — the chosen target can be dropped from wide graphs

Location: `profiling/_analyzer/causal.py:40–55`.
`_limit_columns_for_pc` guesses a target from names containing `target` or
`label`; the actual selected target is not passed to it. Otherwise it retains
the 15 highest-variance variables.

Public reproduction: seed 901, 250 rows, 16 normal numeric features scaled by
20, plus three-class `species`; call `analyze(target_col="species")`. The
profile still identifies `species` as the target but all 15 graph nodes are
numeric features. `species_encoded` is absent.

`frontend/ml-canvas/src/components/eda/CausalGraph.tsx:189` nevertheless says
the graph shows the target and its 14 most correlated features. Pass the actual
target identity through column selection, and describe the no-target case
accurately. Coordinate categorical handling with OC-236.

### OC-238 — internal encoded names escape into public labels

Locations: `profiling/_analyzer/causal.py:128`,
`profiling/analyzer.py:341–354,380–395`, and
`frontend/ml-canvas/src/components/eda/CausalGraph.tsx:123–124`.

Iris analysis returns a node with both ID and label `species_encoded`; the
target-inclusive correlation matrix uses the same temporary column name.
The frontend displays the supplied label verbatim. The API exposes neither
an original-target display mapping nor the class-code mapping in that graph.

Controls: the analyzer's source frame is restored after analysis, the original
`species` values remain intact, and `profile.columns` contains only the original
columns. OC-198 already protects an existing `species_encoded` feature from
being overwritten; this is a separate display/metadata issue. Preserve distinct
internal identities while displaying the original target name and explaining
any supported representation chosen for OC-236.

### OC-239 — excluded targets still participate in rules

Locations: `profiling/analyzer.py:527–540`,
`profiling/_analyzer/rules.py:35–37`, and `backend/eda/tasks.py:115–128`.

The actual `AnalyzeRequest` and `_build_analysis_config` accept
`target_col="species", exclude_cols=["species"]`. Public analysis removes
`species` from column profiles and sample rows, and lists it as excluded, but
still returns a rule tree predicting `setosa` / `virginica` and infers
`Classification` from that target. One returned rule is
`IF sepal_length <= 6.07 THEN setosa (Confidence: 100.0%, Samples: 30)`.

Unlike target associations, `_compute_rule_tree` does not require the target
to remain in the active column set. The raw frame still contains the excluded
column. Make target/exclusion precedence consistent across analysis modules;
do not claim the column is excluded while using it in generated rules.

### OC-240 — ordinary target names collide with aggregation aliases

Locations: `profiling/_analyzer/target.py:67–68,153–165`.

Control fixture: NumPy seed 25, 60 rows; `sepal_length` consists of 30 values
from Normal(4, 0.1) then 30 from Normal(8, 0.1), `petal_length` is independent
normal noise, and `species` contains 30 `setosa` then 30 `virginica` labels.
Public analysis returns two associations (0.9990065931 and 0.1097745073) and
two target interaction box plots.

- Rename only `species` to `mean` or `n`: associations become `{}` and
  interactions become `None`.
- Rename only `species` to `min` or `median`: associations remain correct,
  but interactions become `[]`.

The group key collides with fixed aggregation aliases; caught errors turn
valid analyses into empty results. Use collision-safe aggregation identities.
This is distinct from OC-190's already-fixed rule-tree `value_counts` collision.

### OC-241 — sampled outlier row indices identify other rows

Locations: `profiling/_analyzer/multivariate.py:323–324,366` and
`frontend/ml-canvas/src/components/eda/tabs/OutliersTab.tsx:40,51`.

Public reproduction: seed 74, 50,001 rows, a `row_id` equal to source position,
a normal numeric feature whose final 100 rows are shifted by +40, and an
alternating two-class target. The returned outlier with `index=3501` contains
`values.row_id=49952`. Other examples are `3777 → 49944` and `4863 → 49918`.
Analyzing the first 50,000 rows is the control: every returned top outlier's
index matches its `row_id`.

Sampling discards original positions; `_top_outlier_points` publishes offsets
within that randomized sample. The UI calls these values “Row Index”, so users
can inspect or remove the wrong source row. Preserve row provenance through
sampling and define its relationship to filters. This reproduction used no
filters, avoiding ambiguity about filtered versus source positions.

### OC-242 — sanitization collapses distinct graph nodes

Location: `frontend/ml-canvas/src/components/eda/CausalGraph.tsx:120–140`.

Real Chromium, mocked profile payload: three nodes `petal width`, `petal_width`,
`species_encoded`, with both feature nodes pointing to the target. The renderer
replaces spaces/punctuation with `_`, so both feature IDs become `petal_width`.
Only two DOM nodes remain, `petal width` disappears, and both edge SVG paths
are identical. Source labels are distinct and valid. Use lossless node
identities or an explicit bijective map for nodes and edge endpoints.

### OC-243 — bidirected edges lose their second arrowhead

Location: `frontend/ml-canvas/src/components/eda/CausalGraph.tsx:131–148`.

A schema-supported `A ↔ B` payload renders with `marker-start` set and
`marker-end=null` in Chromium. `markerEnd` is assigned only for `directed`,
while `bidirected` gets only `markerStart`. Render both arrowheads for that
edge type. This is a renderer contract finding; the audit did not establish
how often the current PC configuration emits bidirected edges.

### OC-244 — an empty replacement graph preserves old results

Locations: `frontend/ml-canvas/src/components/eda/CausalGraph.tsx:117` and
`frontend/ml-canvas/src/components/eda/tabs/CausalTab.tsx:49`.

Browser reproduction starts with `old_feature → species_encoded`. Use the
real history control to load report 2, whose mocked response contains target
`empty_target` and `causal_graph={nodes:[],edges:[]}`. The request completes
and the target selector changes, but the previous graph remains identical.
The graph effect returns before clearing state; the tab only treats a null
graph as unavailable.

Clear node/edge state or display the empty state when a replacement graph is
empty. This is a schema-supported history-response case, not evidence that
ordinary current PC execution produces an empty graph object. Simply changing
the target dropdown before running Analyze was **not** filed as a defect.

## Verification and retained local evidence

All seven Core cases executed against unchanged production code; the
observations above are current behavior, not passing fix regressions.

Local ignored reproduction scripts and outputs (not committed):

- `tmp_repro_artifacts/eda_causal_core_audit.py`: `iris`, `direction`, `wide`.
- `tmp_repro_artifacts/eda_causal_iris_012.json`, `eda_causal_iris_021.json`,
  `eda_causal_direction.json`, `eda_causal_wide.json` in the same directory.
- `tmp_repro_artifacts/eda_adjacent_target_audit.py`: aliases, exclusions,
  and the 50,000/50,001-row outlier control; all defect assertions reproduced.
- `tmp_repro_artifacts/eda-causal-audit/probe.mjs`, `results.json`, and three
  PNGs: original-source Chromium probes, **3/3 defect assertions reproduced**.
  The owned Vite helper was stopped after verification.

Commands from the repository root:

```powershell
.venv/Scripts/python.exe tmp_repro_artifacts/eda_causal_core_audit.py iris --order 0,1,2 --output tmp_repro_artifacts/eda_causal_iris_012.json
.venv/Scripts/python.exe tmp_repro_artifacts/eda_causal_core_audit.py iris --order 0,2,1 --output tmp_repro_artifacts/eda_causal_iris_021.json
.venv/Scripts/python.exe tmp_repro_artifacts/eda_causal_core_audit.py direction --output tmp_repro_artifacts/eda_causal_direction.json
.venv/Scripts/python.exe tmp_repro_artifacts/eda_causal_core_audit.py wide --output tmp_repro_artifacts/eda_causal_wide.json
.venv/Scripts/python.exe -m tmp_repro_artifacts.eda_adjacent_target_audit
node tmp_repro_artifacts/eda-causal-audit/probe.mjs
```

The browser command requires the unchanged frontend's Vite server at
`http://127.0.0.1:5173`; API requests are intercepted locally by the probe.

Existing focused regression suites: **83 passed / 41 warnings**:

```powershell
.venv/Scripts/python.exe -m pytest skyulf-core/tests/integration/test_profiling_analyzer.py skyulf-core/tests/integration/test_profiling_column_name_collisions.py skyulf-core/tests/integration/test_profiling_target.py -q --tb=short --basetemp=tmp_repro_artifacts/pytest-eda-causal-audit -o cache_dir=tmp_repro_artifacts/pytest-eda-causal-cache
```

These tests currently miss the newly reproduced cases, particularly the real
causal-learn endpoint convention. No frontend build was required for this
review because source and generated assets were unchanged.

## Original repair priorities

These priorities were recorded by the initial review; the repairs above have
since been implemented.

First fix OC-235, then settle the supported categorical analysis contract for
OC-236 together with target selection and labels (OC-237/238). OC-241 and OC-242
also silently misidentify data and deserve early repair. Exclusion consistency,
aggregation aliases, and renderer edge/empty-state handling can be independent
small fixes with the reproductions above converted into regression coverage.

Existing open OC-110, OC-113, OC-192, OC-50, and OC-51 were not refiled or
re-verified by this pass. No claim is made that the rest of profiling is free of
defects. At the end of the initial review the live queue had **47 open / 4
parked** and no findings had been closed; see the repair follow-up for the
current count.
