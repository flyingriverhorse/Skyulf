# Growth Plan

**Date:** 2026-08-11
**Code baseline:** branch `078` — backend `0.7.8`, frontend `0.7.8`, `skyulf-core` `0.5.8`
**Budget:** 2–3 days/week, paired
**Horizon:** three stages, ~6–8 weeks. Nothing beyond that is planned here.

## Branch reality (read this first)

Two branches matter, and they are not the same:

| Branch | Role | Versions | State |
|---|---|---|---|
| `078` | **Code and docs.** Everything lands here, including this plan | `0.7.8 / 0.7.8 / 0.5.8` | `080` merged in on 2026-08-11 (50 files, all `initiatives/`, zero code) |
| `origin/deploy/demo-mode` | **What the public actually uses** | `0.7.6 / 0.7.6 / 0.5.6` | 26 commits behind, and staying there by decision |

Verified before merging: `git diff --name-only 078...080` → 50 files, all
under `initiatives/`. `git rev-list --left-right --count origin/deploy/demo-mode...080`
→ `5 26`.

Two consequences that shape everything below:

1. **Findings must be verified on `078`.** All Stage 0 items were
   re-reproduced there after initially being found on `080`. This mattered:
   `078`'s 11 commits included Polars/Pandas parity fixes that rewrote the
   *same functions* as T1 without touching the bug.
2. **Fixes do not reach demo visitors automatically.** The demo branch has
   diverged, carries committed build artifacts, and holds features that
   exist nowhere else. Stage 1 decides deliberately how much of that to care
   about — the answer is "less than first proposed."

## The situation

Distribution already exists. Conversion does not — and the pipeline that
would carry a fix to a real user is broken.

| Asset | State | Verified how |
|---|---|---|
| Live demo, no signup | Up, healthy, ~0.7s | `curl -o /dev/null -w "%{http_code}" https://api.skyulf.com` → `200` |
| **What the demo runs** | **`deploy/demo-mode`, 26 commits behind, diverged 2026-08-08** | `git rev-list --left-right --count origin/deploy/demo-mode...080` → `5 26` |
| **Datasets a demo visitor can use** | **Exactly one: Iris.** Upload is deliberately disabled in demo mode | `curl https://api.skyulf.com/api/pipeline/datasets/list` → `[{"id":"iris-demo","name":"Iris Flower Dataset"}]`; `curl .../api/config` → `{"demo_mode":true}` |
| `skyulf-core` on PyPI | 31 releases, 1,222 downloads/month, 232/week | `pypistats.org/api/packages/skyulf-core/recent` |
| Product analytics | **None** | No `plausible`/`posthog`/`umami`/`gtag` anywhere; the three `analytics` grep hits are internal ops monitoring (`SlowNodesPage.tsx`, `monitoring/router.py`) |
| Sample data in the UI | **None** | 8 datasets exist at `skyulf-core/examples/data/`; no UI entry point on any branch |

So the problem is not "nobody can find it." The problem is what happens in
the five minutes after they do, that we cannot measure it, and that we have
no reliable way to ship them a fix.

## Stage 0 — Trust floor

**Why first.** Acquisition work on a platform that silently mislabels
training data buys users who get wrong answers and leave permanently. In an
ML tool, "it gave me garbage" is the one first impression you cannot
recover from, and it is the kind of thing users write about publicly.

**Every item below was re-reproduced on branch `078` on 2026-08-11** (via a
throwaway worktree), not merely on the branch where it was first found.

### T1 — Lag and Rolling nodes return a stale, misaligned `y` (critical)

`X` is sorted and row-filtered; `y` is returned untouched. Every row ends up
with another row's label.

```
LagFeatures, sort_by='t', drop_na=True
  input   X.t=[3,1,2]  X.v=[30,10,20]   y=[300,100,200]
  output  X: 2 rows (t=2,3)              y: 3 rows, unchanged
    row t=2 v=20.0 → y says 300   (truth 200)
    row t=3 v=30.0 → y says 100   (truth 300)

RollingAggregate, sort_by='t'
    t=1 → y says 300 (truth 100)
    t=2 → y says 100 (truth 200)
    t=3 → y says 200 (truth 300)
```

**Root cause — and this differs from what `bug-hunt.md` records.** It is not
a local oversight in two nodes. `@apply_method` (`preprocessing/base.py:52`)
unpacks `(X, y)` and calls the node with both. The node then passes **bare
`X`** into `apply_dual_engine`, which unpacks *again*
(`dispatcher.py:81`) — from a plain DataFrame, so `y` becomes `None`. The
node's `_y` is therefore always `None`, `pack_pipeline_output` returns `X`
alone, and `@apply_method`'s wrapper falls back to the **original** `y`.

Any node composing both layers *and* changing row count or order is exposed.

**The fix already exists in the repo.** `deduplicate.py:52-58` does it right,
with a comment saying why:

```python
# Note: dedup must propagate row drops to y, so we route X+y as a tuple
# through apply_dual_engine which handles unpack/pack.
return apply_dual_engine((X, y) if y is not None else X, params, ...)
```

`lag.py:77` and `rolling.py:121` pass bare `X`. One line each.

**Verified not affected:** `DropMissingRows` and `Deduplicate` both
propagate row drops to `y` correctly (`X:5→4, y:5→4`). The blast radius is
narrower than the root cause first suggests — but it must be enumerated, not
assumed, which is what T5 is for.

### T2 — FeatureSelection's own default is an unknown method

```
FeatureSelectionCalculator().fit(X, {'method':'variance','threshold':0.0})
→ logs "Unknown feature selection method: variance"
→ returns {}   (constant columns survive)
```

The registered default is `variance`; the dispatch table
(`feature_selection/facade.py`) only knows `variance_threshold`. The node
calls its own default "unknown."

### T3 — GeneralBinning's own default produces no bins

```
GeneralBinningCalculator().fit(X, {'columns':['x'],'n_bins':2,'strategy':'uniform'})
→ {'bin_edges': {}, ...}   (applier leaves X unchanged, no error)
```

Registered default is `strategy: "uniform"`; the fit path handles
`equal_width`. The frontend happens to send `equal_width`, so this hits
SDK and registry-driven callers only — i.e. exactly the PyPI audience.

### T4 — The live demo reports a fake version (symptom of a deploy problem)

```
$ curl -s https://api.skyulf.com/health
{"status":"healthy", "version":"0.0.0-dev", "environment":"production", ...}
```

**Diagnosis, corrected.** This is not a hardcoded string anyone forgot to
update. `backend/config/mixins/core.py:6` does
`_APP_VERSION = version("skyulf")` and falls back to `"0.0.0-dev"` at line 9
when the distribution metadata is missing. The demo reporting the fallback
means **the backend is running from source without being installed** in the
deployed image.

So the visible symptom is cosmetic, but what it reveals is not: the
production deployment does not match the packaging the project assumes.
**Descoped** — moved to the demo backlog under Stage 1. It is recorded here
only because it is how the packaging gap was discovered, and because the
same gap would mislead a self-hoster filing a bug report against the wrong
version.

### T5 — The actual deliverable: a registry-wide contract test

T1–T3 share one shape: **a node's own declared defaults do not work.** Fix
the three and the class remains. So the exit criterion for Stage 0 is a
parametrised test over `NodeRegistry` asserting, for every registered node:

1. Fitting/applying with the node's **own `@node_meta` default params**
   produces no empty artifact, no "unknown method" warning, and no silent
   no-op.
2. Any node that changes row count or row order returns a `y` whose length
   and order match `X`.

This test fails on T1, T2, and T3 today. Three bug fixes are worth a week; a
test that makes the whole class impossible is worth considerably more, and
it is the single highest-value artifact in this stage.

### T6 — FeatureMath silently drops datetime features on mixed-offset input

The same silent-no-op family as T2/T3, independently re-verified:

```
_pandas_ops.py:179  dt = pd.to_datetime(df_out[col], errors="coerce")   # no utc=True
→ object dtype → .dt accessor raises → swallowed at :185-186 as a warning
→ _featgen_apply_pandas (:226-228) continues; node still reports success
Result: `when_hour` absent, no failure reported.
```

Include it in Stage 0 because it is the same fix discipline and because
pandas will turn this into a hard `ValueError` in a future version — which
is *still swallowed*, so it degrades rather than self-corrects.

**Audiences:** all four (exempt from rule 4).
**Exit criteria:** T5 passes across the registry; T1–T3 and T6 closed (T4
is handled in Stage 1a); ship as a patch to all three components from `078`.

## Verified backlog (not Stage 0)

Re-verified on 2026-08-11, real, but deliberately **not** in the trust
floor — none of them silently corrupts a model. Recorded here so they are
not re-investigated a third time.

| Finding | Verdict on re-verification | Where it goes |
|---|---|---|
| Cyclic pipeline graphs accepted, then fail late with `Artifact not found: B` | **Confirmed end-to-end.** No cycle check in `useGraphStore.ts:289-453` (`onConnect` sets edges unconditionally at `:451`), no `isValidConnection` anywhere (0 grep hits), and `engine/__init__.py:123` iterates nodes in list order with no topological check | Stage 2/3 — it is an *error-quality* problem, which is complaint rank #1 |
| Upload UI rejects >500MB although the server accepts 10GB | **Confirmed, severity lower than documented.** `FileUpload.tsx:52-54` hardcodes both the limit and the "500MB" message; `MAX_UPLOAD_SIZE` (`config/mixins/files.py:18`, 10GB) has **zero** `.ts`/`.tsx` hits, so the client is wrong for *every* deployment | Stage 2 — cheap, and it wrongly blocks a first upload |
| Cross-process duplicate job creation | **Static analysis only** (needs Postgres + 2 API processes). Claim holds: `_submit_locks` (`run_pipeline.py:43`) is per-process and there is no DB uniqueness (`UNIQUE constraints: []`). **New defect found:** `skip_locked=True` (`jobs.py:113`) means a row locked by a concurrent transaction is silently skipped, so dedup can fail *within* one process too | Stage 3 / enterprise — needs real infra to fix and to test |
| Out-of-order job-list responses revert newer state | **Confirmed but overstated.** Self-heals on the next poll in most paths. Genuinely sticky only when a slow `fetchJobs` resolves after a tick already saw the terminal status and called `stopPolling()` | Stage 3 — low value |

Two of these were documented at a higher severity than re-verification
supports. That is the operating rules working as intended.

## Stage 1 — Demo triage, then measure

**Scope decision (2026-08-11, owner).** This stage originally proposed
reconciling `deploy/demo-mode` with the main line: porting `demo_mode` into
the tested codebase, recovering the demo-only features, and ending the
committed-bundle deploy process. That was rejected as over-scoped, and the
rejection was correct — the demo is a shop window, not production, and a
2–3 day/week budget should not be spent on release engineering for it.

What survives is one item, not a stage.

### 1a — Make the demo stop demonstrating failure

The demo is the first clickable element in `README.md` — above every badge,
above the description:

```html
README.md:8-12
<a href="https://api.skyulf.com">…Try Live Demo…</a>
<sub>🟢 no signup required</sub>
```

And the live demo ships the **same blocked templates** as the main line.
Verified directly against the deployed branch:

```
$ git show origin/deploy/demo-mode:frontend/ml-canvas/src/core/templates/pipelineTemplates.ts
105:  id: 'tabular_classification',
112:    { localId: 'imp',   type: 'imputation_node',        … }
113:    { localId: 'enc',   type: 'encoding',               … }
114:    { localId: 'scl',   type: 'scale_numeric_features', … }
115:    { localId: 'split', type: 'TrainTestSplitter',      … }

$ git ls-tree -r origin/deploy/demo-mode --name-only | grep leakage
frontend/ml-canvas/src/core/utils/pipelineLeakageValidation.ts   ← the blocker, present
```

So the current visitor path is: click the badge → wait out a cold start →
pick "Tabular Classification" → **Run All → hard-blocked by an error they
did not cause.** Four of the five templates. If the demo exists to show
capability, it presently shows the opposite.

**Work:** cherry-pick the A2.2 template fix onto `deploy/demo-mode` and
redeploy. The templates file is 238 lines and the change is node ordering.

**Explicitly not doing:** branch reconciliation, porting `demo_mode` into
the main line, recovering the Slow Nodes page and SHAP deps, removing the
9 committed bundles, or adding CI coverage for the demo. All real, none
worth the budget. Recorded in the backlog below so the findings are not
lost.

**Depends on:** A2.2 landing on `078` first. Nothing else in the plan
depends on *this*.

*Audiences: anyone arriving from the README badge.*

### Demo backlog (found, deliberately not scheduled)

- `demo_mode` exists **only** on `deploy/demo-mode` — `grep -rn "demo_mode"`
  returns 0 on the main line. It gates uploads, filters datasets to Iris,
  and serves `/api/config`, and it is untested by CI. A liability, but a
  dormant one.
- The branch carries unmerged work that exists nowhere else: a Slow Nodes
  page, SHAP runtime deps, and ~110 changed lines in
  `shap_explanation.py`. **At risk if the branch is ever recreated** — the
  one backlog item with a real loss scenario.
- 9 built frontend bundles are committed to the branch, so each promotion
  is a manual rebuild-and-commit.
- `/health` reports `0.0.0-dev` — the `PackageNotFoundError` fallback at
  `core.py:9`, meaning the backend is not pip-installed in the deployed
  image. Cosmetic.
- **Every visitor's uploads are visible to every other visitor**
  (`data_ingestion/router.py:41-42`, `list_sources(user_id=None)`, marked
  `# KNOWN-GAP: Auth not implemented yet`). Contradicts the "privacy-first"
  claim at `README.md:35`. Demo mode disables uploads, which neutralises it
  *on the demo* — but the same code path is what a self-hosting user runs.
  **Reconsider promoting this if self-hosted multi-user is ever claimed.**
- First request to a cold instance hung >60s; `/health` then reported
  `uptime_seconds: 138`. Warm, it serves in 0.21s. Cold start is the
  hypothesis; the hang and the uptime are measured.

### 1b — Measurement

**Why.** Every prioritisation past this point is a guess without it. This is
the specific defect that made the previous roadmap speculative: it had to
invent an ordering because no data existed to derive one.

**The first question is smaller than "activation rate."** Asked directly,
the owner does not know whether the demo gets any traffic at all. That makes
the cheapest possible measurement the right one to start with, because it
decides whether anything else here is worth doing:

- **Does anyone click the README badge?** GitHub's repo traffic panel
  (Insights → Traffic) already records referring paths and clone/view counts
  with **zero code**, and it is available today. Look before building.
- If the answer is "essentially nobody," then the constraint is acquisition,
  not activation — and Stage 2 is the wrong next move regardless of how
  well-evidenced its individual items are. Say so out loud rather than
  proceeding on momentum.

**Only if there is traffic, add event instrumentation.**

**The privacy tension, resolved explicitly.** Skyulf's positioning is
"self-hosted, privacy-first." Bolting analytics onto that would be
self-defeating, so:

- **Demo instance only** (`api.skyulf.com`), which is already a distinct
  deployment with its own config. Self-hosted installations send **nothing,
  ever**, and we say so plainly in the README. That converts the constraint
  into a differentiator instead of an apology, and answers complaint rank #2
  (pricing/vendor opacity) in
  `../enterprise-readiness/2026-08-11-user-complaints-research.md`.
- **Aggregate events only:** page view, dataset selected, node added, run
  started, run succeeded, run failed. No dataset contents, no column names,
  no file names, no PII.

**The one number that matters** (once traffic is confirmed): *activation
rate* — the share of demo visitors who reach a successful pipeline run.

It disambiguates the rest of the funnel:

- Near-zero → the problem is Stage 2 (first-run), fix that next.
- Healthy but no repeat use → the problem is retention, and Stage 3's
  candidate list is the right menu.

**Note on cost.** Instrumenting the demo means deploying to the demo branch,
which is exactly the manual rebuild-and-commit process 1a declined to fix.
That is an accepted cost, not an oversight: one awkward deploy is cheaper
than reworking the release pipeline. If it turns out to need more than one,
revisit the backlog item.

**Exit criteria:** either a recorded answer to "does the badge get clicked"
that is low enough to stop here, or a week of visible activation rate.

## Stage 2 — First-run activation

The one body of work that serves **all four** audiences simultaneously.

### A2.1 — Give the demo something worth running

**Corrected premise.** The original version of this item said a visitor
"has nothing to click." That is wrong, and the truth is more interesting: a
visitor gets **exactly one dataset, Iris**, and cannot upload their own
because demo mode disables it by design.

Iris is a poor advertisement for this specific product: 150 rows, 4 clean
numeric columns, no missing values, no categoricals, no dates, trivially
separable. It exercises essentially none of what Skyulf is *for* —
imputation, encoding, outliers, leakage, feature generation. A visitor
evaluating a preprocessing-heavy tool sees a dataset that needs no
preprocessing.

Meanwhile 8 substantial datasets already ship in the repo
(`credit_card_fraud`, `disaster_tweets`, `forest_cover`, `house_prices`,
`mall_customers`, `online_retail`, `santander`, `spaceship_titanic`) and
**no UI path loads any of them, on any branch.**

Prior research reached a compatible conclusion independently
(`smooth-experience-fixes.md` Top 3 #1): add a "Load sample dataset" entry
to `AddSourceModal`. The data exists; only the entry point is missing.

Since demo uploads are disabled, this sample library *is* the demo's entire
data story. It pairs directly with A2.2: sample data with no working
template is a dead end, and a working template with only Iris demonstrates
nothing. Ship the two together as one path.

*Audiences: analysts, students, ML engineers evaluating.*

### A2.2 — Fix the shipped templates (they are currently blocked)

**This item changed completely after audit.** It was "bind one template to a
dataset." The truth is that **4 of the 5 shipped templates cannot run at
all** — they are blocked by Skyulf's own leakage guard.

Verified by executing the product's own converter and validator against its
own templates:

```
tabular_classification    ds → DropMissingColumns → SimpleImputer → OneHotEncoder → StandardScaler → TrainTestSplitter → training   LEAKAGE_ISSUES=3
tabular_regression        ds → SimpleImputer → IQR → StandardScaler → TrainTestSplitter → training                                  LEAKAGE_ISSUES=3
text_classification       ds → TextCleaning → tfidf_vectorizer → TrainTestSplitter → training                                       LEAKAGE_ISSUES=1
customer_segmentation     ds → SimpleImputer → StandardScaler → training                                                            LEAKAGE_ISSUES=0
ensemble_classification   ds → DropMissingColumns → SimpleImputer → OneHotEncoder → StandardScaler → TrainTestSplitter → training   LEAKAGE_ISSUES=3
```

Confirmed independently: `pipelineTemplates.ts:112-127` places
`imputation_node`, `encoding` and `scale_numeric_features` upstream of
`TrainTestSplitter`, and `pipelineLeakageValidation.ts:23-27` blocks exactly
those. It **blocks**, it does not warn — `useRunControls.ts:65-72`:
`toast.error('Fix validation issues before running experiments')`. The
backend hard-blocks too (`_leakage_validation.py:212`).

`customer_segmentation` passes only because it has no splitter at all
(`pipelineLeakageValidation.ts:128`: `if (splitterIds.size === 0) return []`).

**So the guided happy path is:** empty canvas tells the user to start from a
template (`FlowCanvas.tsx:371-384`) → they pick "Tabular Classification" →
bind data → Run All → **blocked by an error they did not cause and cannot
interpret.**

**The guard is right; the templates are wrong.** They also contradict
`skyulf-core/README.md:192-196` ("Put `TrainTestSplitter` first"). Fix the
templates by moving the splitter upstream — do not weaken the guard.

There is a real lesson here worth stating plainly: leakage-safety is the
positioning asset this project has chosen, and it is genuinely working — it
caught its own authors. That is evidence the feature is valuable, not
evidence it is too strict.

*Audiences: all four. Catastrophic for analysts and students, who cannot
diagnose it.*

### A2.3 — Fix the first-run entry points

Three separate confirmed blockers, all cheap:

- **`start.sh` is not executable.** `git ls-files -s start.sh` → `100644` on
  both `078` and `080`. `README.md:59` tells macOS/Linux users to run
  `./start.sh`; it fails **100% of the time** with `Permission denied`
  (exit 126). The literal first command in the README. Fix with
  `git update-index --chmod=+x start.sh`.
- **`/` is a dashboard of zeros.** `App.tsx:37-38` routes `/` to
  `<Dashboard />`; the canvas is at `/canvas`. Live: `/api/pipeline/stats` →
  `{"total_jobs":0,...}` plus "No recent jobs found." A "Visual MLOps
  Builder" whose front page is four zeros and an empty table.
- **The install is not "3-5 minutes."** `start.sh:295` claims 3-5 min;
  `requirements-fastapi.txt:67` pulls `sentence-transformers`, which drags
  in torch — measured **2.2 GB venv, 529 MB torch alone**. Realistically
  15-40 minutes. Users will assume it hung. Either make the heavy extras
  optional or state the real number.

*Audiences: ML engineers evaluating, students.*

### A2.4 — `skyulf-core` distribution polish

**Corrected premise, and it is good news.** I assumed the README examples
might be broken. They are not: on a clean venv with `skyulf-core==0.5.7`
from PyPI, **every documented example ran verbatim** — quickstart,
`get_fitted_split`, `validate_leakage_safety`, the full EDA attribute set,
and both docs walkthroughs. The missing-extra errors are exemplary
(`Please install 'rich' ... pip install skyulf-core[viz]`).

This is the strongest part of the entire product's first-run story, and
nothing in this plan should disturb it. The remaining issues are narrow:

- **The very first copy-paste fails.** `skyulf-core/README.md:77` opens with
  `pl.read_csv("customers.csv")` — a file the reader does not have →
  `FileNotFoundError`. The `docs/` examples build a DataFrame inline and are
  strictly better. Use that pattern in the README.
- **17 relative links are dead on PyPI**, including the entire Examples
  table (`README.md:254-264`), which is the reader's intended next step.
  `MANIFEST.in` ships no examples either, so `pip install` provides nothing
  to click. 3 mermaid diagrams render as raw code.
- **The docs site is effectively undiscoverable.** `docs.yml:69` deploys
  mkdocs to `deploy/manual`, but links were written for the root:
  `.../Skyulf/user_guide/threshold_tuning.html` → **404**, while
  `www.skyulf.com/manual/user_guide/threshold_tuning.html` → 200. The Docs
  badge lands on the marketing page with no visible path to `/manual/`.

*Audiences: data scientists, students.*

### A2.5 — Say that pipelines export to Python

Complaint rank #1 across the external research — with a detailed founder
testimonial behind it — is **vendor lock-in / no usable code export**.
Skyulf *already ships* notebook export. This is a positioning gap, not a
build task: make it prominent in the README and the demo rather than
leaving it buried.

The cheapest item in this plan and the best-evidenced.

*Audiences: data scientists, ML engineers.*

### A2.6 — Fix the upload size message

`FileUpload.tsx:52-54` hardcodes a 500MB limit *and* the "500MB" text, while
the server accepts 10GB (`config/mixins/files.py:18`) — a 20× discrepancy —
and `MAX_UPLOAD_SIZE` has zero `.ts`/`.tsx` hits. The limit is never stated
up front; it surfaces only as a failure. The file picker also hides three
formats the backend accepts: `accept=".csv,.xlsx,.parquet,.json"`
(`FileUpload.tsx:113`) vs `.xls`/`.txt`/`.feather` also allowed
(`files.py:27-34`).

*Audiences: analysts, ML engineers (self-hosted paths).*

**Exit criteria:** a first-time visitor reaches a trained model on a dataset
that demonstrates real preprocessing, without uploading a file or reading
docs; activation rate re-measured against the 1b baseline.

## Stage 3 — Retention and enterprise: candidates only

**Deliberately not scheduled.** Sequencing these now would recreate exactly
the mistake this folder exists to correct. Ranked by evidence, chosen after
Stage 1 data:

| Candidate | Evidence | Source |
|---|---|---|
| Per-node data preview ("click a node, see the data") | Strong, highly specific; a competing product was built by an ex-user of an incumbent to solve this exact problem | user-complaints-research #4 |
| Actionable errors (what / where / fix / retry) instead of "Check console" | Strong, 4+ sources; `useRunControls.ts:99-105` is the concrete offender | user-complaints-research #1, smooth-experience-fixes Top 3 #3 |
| Inspectable trace for auto/tuning nodes | Strong (H2O, SageMaker Canvas, DataRobot all criticised for opacity) | user-complaints-research #3 |
| Post-upload pipeline recommendation | `profiling/recommendations.py` already computes the heuristics; only assembly is missing | smooth-experience-fixes Top 3 #2 |
| "Live / Reconnecting" indicator | `jobEventsSocket.ts:44-90` plumbing exists with zero rendered consumers | smooth-experience-fixes §C |
| Post-fit diagnostics surfaced on job completion | Reuses existing chart components | training-visualization 15b tier (a) |
| Read-only per-node generated code | Zero new security risk per its own study | code-escape-hatch Phase A |

Enterprise (stage 3 of the funnel) is downstream of all of the above: an
organisation adopts a tool its people already use. The auth/tenancy work in
`enterprise-readiness/` Phase 0 is real and will be needed — but building it
before there are users to belong to an organisation is inventory, not
progress.

## What is already good (do not break these)

An audit that only lists faults produces a distorted plan. These were
verified working and are load-bearing:

- **Every documented `skyulf-core` example runs verbatim** on a clean PyPI
  install — quickstart, `get_fitted_split`, `validate_leakage_safety`, the
  EDA attributes, and both docs walkthroughs. Missing-extra errors name the
  exact fix.
- **The leakage guard works.** It caught the project's own templates. That
  is the differentiator functioning correctly.
- **Zero-dependency local boot.** A clean clone runs with no `.env`, no
  Postgres/Redis/MinIO, no Node — SQLite default, frontend prebuilt and
  committed, DB auto-created. Verified: `HEALTH 200`, `ROOT 200`.
- **The empty canvas is not a void.** It offers "Browse templates" and
  "Drag a node from the sidebar" (`FlowCanvas.tsx:363-386`), and it *is*
  deployed. 35 node types in 5 searchable groups. Undo/redo, auto-layout,
  autosave with a restore banner.
- **Validation messages are written for humans:** `Move ${node} after
  ${splitter} so it only fits on training data.` (`useGraphStore.ts:235`).
- **No signup, exactly as advertised.**

The implication for sequencing: the product's *substance* is in better
shape than its *entry points*. That is a much cheaper problem to have, and
it is why this plan concentrates on first-run rather than features.

## Explicitly not doing now

Ray migration, deep learning, i18n/RTL, the six page redesigns, and the
enterprise track. Each is defensible in isolation; none survives contact
with a 2–3 day/week budget while activation is unmeasured and the trust
floor is broken. The research stays on disk and is promoted when earned.

## Known gaps in this plan

Honest limits, stated so they are not mistaken for coverage:

- Stage 1b's activation-rate target is undefined on purpose — we have no
  baseline, so any number would be invented. We set it after the first
  week of data.
- Stage 2 assumes the leak is at first-run. Stage 1b exists to falsify
  that; if the data says otherwise, Stage 2 gets rewritten.
- Traffic volume to the demo is unknown (the GitHub traffic API was not
  reachable from here). If it turns out to be very low, acquisition — not
  activation — is the real constraint and this plan changes.
- Bug #1 (duplicate job creation) was verified by static analysis only; it
  needs Postgres and two API processes to reproduce properly.
- The demo cold-start cause (A3) is inferred, not proven — the >60s hang
  and the 138s uptime are measured; "Render cold start" is the hypothesis.
- `QUICKSTART.md` is substantially stale (references a non-existent
  `backend/config.py`, `admin`/`admin123` credentials that exist nowhere,
  and a `data/` directory that does not exist) and is orphaned — linked
  from neither `README.md`, `docs/index.md`, nor `mkdocs.yml`. Not
  scheduled: decide whether to fix or delete it during Stage 2.
- `docs/index.md:33-38` tells users to `pip install` into system Python
  with no venv, which fails with PEP 668 on Homebrew Python. Fold into
  A2.3 if cheap.
- **Demo traffic is unmeasured and unknown.** Asked directly, the owner
  does not know whether anyone clicks the README badge. This is the single
  largest gap in the plan: if traffic is near zero, the constraint is
  acquisition rather than activation, and most of Stage 2 is well-evidenced
  work aimed at the wrong problem. Stage 1b's first step exists to close
  this before Stage 2 consumes any budget.

## Branch decision for this folder — done

`initiatives/growth/` had to live wherever the work happens. Resolved
2026-08-11: `080` was merged into `078` (`0093b15d`), verified as 50 files
all under `initiatives/` with zero code touched. A plan on a branch nobody
builds from is exactly the failure this folder was created to stop, and it
is no longer a risk here.

## Execution order (decided)

Derived from the dependency graph, not from preference:

1. **Stage 0 `skyulf-core` fixes → ship to PyPI.** T1, T2, T3, T6 and the
   T5 contract test are all `skyulf-core`, which releases on its own
   `core-v*` tag and **never touches the demo branch**. This path is
   unblocked today, and it is the one actively corrupting results for 1,222
   downloads/month.
2. **Check the GitHub traffic panel** (Stage 1b, first half). Free, no code,
   and it decides whether the rest of the funnel work is justified at all.
   Do this *before* Stage 2, not after.
3. **A2.2 — fix the templates on `078`**, then **1a** — cherry-pick them to
   the demo branch and redeploy. A2.2 must land first; 1a is a follow-on of
   an hour, not a stage.
4. **The rest of Stage 2**, weighted by whatever step 2 revealed.

The cheap, isolated items (`start.sh` chmod, upload-size text) can ride
along with any of the above; they depend on nothing.

Stage 1b's event instrumentation is deliberately **not** in this list. It is
conditional on step 2 showing traffic worth measuring.
