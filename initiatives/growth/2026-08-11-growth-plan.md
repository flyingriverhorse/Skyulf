# Growth Plan

**Date:** 2026-08-11
**Code baseline:** branch `078` — backend `0.7.8`, frontend `0.7.8`, `skyulf-core` `0.5.8`
**Budget:** 2–3 days/week, paired
**Horizon:** three stages, ~6–8 weeks. Nothing beyond that is planned here.

## Branch reality (read this first)

Three branches are live, none contains the others:

| Branch | Role | Versions | Divergence |
|---|---|---|---|
| `078` | **Code.** Bug fixes land here | `0.7.8 / 0.7.8 / 0.5.8` | 11 commits not on `080` |
| `080` | **Docs.** All `initiatives/` research, including this plan | `0.7.7 / 0.7.7 / 0.5.7` | 18 commits not on `078` |
| `deploy/demo-mode` | **What the public actually uses** | `0.7.6 / 0.7.6 / 0.5.6` | 5 commits not on `080`; 26 behind |

Verified: `git rev-list --left-right --count 078...080` → `11 18`;
`... origin/deploy/demo-mode...080` → `5 26`.

Two consequences that shape everything below:

1. **Fixes do not reach users.** Anything fixed on `078` is invisible to
   every real visitor until it is promoted through `deploy/demo-mode`, which
   has diverged, carries committed build artifacts, and holds features that
   exist nowhere else. See Stage 1.
2. **Findings must be verified on `078`, not `080`.** All Stage 0 items
   below were re-reproduced on `078` after initially being found on `080`.
   This mattered: `078`'s 11 commits include Polars/Pandas parity fixes that
   rewrote the *same functions* as T1 without touching the bug.

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
production deployment does not match the packaging the project assumes. Fix
it as part of Stage 1a (deployment reconciliation) rather than as a code
change here — it is listed under Stage 0 only because it is how the problem
was discovered.

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

## Stage 1 — Make the demo shippable, then measure

Stage 1 was originally just "add measurement." Investigating the demo branch
showed that is not possible yet: instrumentation has to be *deployed* to
produce data, and there is currently no dependable path from a fix to a
visitor. So this stage has two parts, in order.

### 1a — Reconcile `deploy/demo-mode` with the main line

The public demo runs code that **does not exist in the main line at all**:

```
$ grep -rn "demo_mode" --include=*.py --include=*.ts --include=*.tsx backend/ frontend/ml-canvas/src/   # on 080
0
$ git grep -n "demo_mode" origin/deploy/demo-mode -- '*.py'
backend/data_ingestion/router.py:34:  def _block_in_demo_mode() -> None:
backend/health/routes.py:54:          return AppConfigResponse(demo_mode=settings.DEMO_MODE)
```

The demo-mode feature — which gates uploads, filters datasets to Iris, and
serves `/api/config` — is unversioned, untested by CI, and reviewed by
nobody, despite being the only thing standing between the public and the
product. Alongside it, `deploy/demo-mode` carries genuine work that exists
nowhere else: a **Slow Nodes page**, **SHAP runtime dependencies**
(matplotlib, seaborn, plotly, shapely, sentence-transformers), and a
`shap_explanation.py` with 110 changed lines.

It also commits **9 built frontend bundles** (`static/ml_canvas/assets/*.js`)
directly to the branch, so every promotion is a manual rebuild-and-commit.

**Work:**

- Port `demo_mode` into the main line behind a setting, defaulting to off,
  so it is tested and reviewable.
- Port the demo-only features back (Slow Nodes page, SHAP deps, iris
  filter). These are 5 commits at risk of being lost the moment the demo
  branch is ever recreated.
- Stop committing built assets; build during deploy.
- Reduce `deploy/demo-mode` to deployment configuration only (`vercel.json`,
  env), so promotion becomes routine rather than a merge negotiation.
- **Install the backend package in the deployed image** so `/health` reports
  a real version instead of the `0.0.0-dev` fallback (Stage 0 item T4).

**Why this is Stage 1 and not later:** every subsequent item — the trust
fixes, the analytics, the sample datasets — is worthless while it cannot
reach a visitor. This is the release pipeline, and it is currently the
narrowest part of the funnel.

*Audiences: all four (nothing ships to anyone without it).*

### 1b — Measurement

**Why.** Every prioritisation past this point is a guess without it. This is
the specific defect that made the previous roadmap speculative: it had to
invent an ordering because no data existed to derive one.

**The privacy tension, resolved explicitly.** Skyulf's positioning is
"self-hosted, privacy-first." Bolting analytics onto that would be
self-defeating, so:

- **Demo instance only** (`api.skyulf.com`), which is already a distinct
  deployment with its own config — so this rides on 1a rather than adding
  new machinery. Self-hosted installations send **nothing, ever**, and we
  say so plainly in the README. That converts the constraint into a
  differentiator instead of an apology, and answers complaint rank #2
  (pricing/vendor opacity) in `../enterprise-readiness/2026-08-11-user-complaints-research.md`.
- **Aggregate events only:** page view, dataset selected, node added, run
  started, run succeeded, run failed. No dataset contents, no column names,
  no file names, no PII.

**The one number that matters:** *activation rate* — the share of demo
visitors who reach a successful pipeline run.

It disambiguates the whole funnel:

- Near-zero → the problem is Stage 2 (first-run), fix that next.
- Healthy but no repeat use → the problem is retention, and Stage 3's
  candidate list is the right menu.

**Exit criteria:** a fix merged on `078` is running on `api.skyulf.com`
without hand-editing bundles; activation rate visible for a full week.

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
data story — which makes it the highest-leverage item in the plan after the
trust floor.

*Audiences: analysts, students, ML engineers evaluating.*

### A2.2 — Bind one starter template to one sample dataset

Templates exist but "require manual dataset binding + target setup even
after selection" (`TemplatesGalleryModal.tsx:14-18`,
`pipelineTemplates.ts:104-127`). One template, pre-bound, so the path is
*load → run → trained model* with no configuration.

*Audiences: all four.*

### A2.3 — `skyulf-core` README leads with a runnable snippet

The README opens with ~13 lines of prose and 12 badges; installation is at
line 26. PyPI is the highest-volume channel there is (1,222/month). The
first screen should be code that runs.

*Audiences: data scientists, students.*

### A2.4 — Say that pipelines export to Python

Complaint rank #1 across the external research — with a detailed founder
testimonial behind it — is **vendor lock-in / no usable code export**.
Skyulf *already ships* notebook export. This is a positioning gap, not a
build task: make it prominent in the README and the demo rather than
leaving it buried.

The cheapest item in this plan and the best-evidenced.

*Audiences: data scientists, ML engineers.*

### A2.5 — Fix the upload size message

`FileUpload.tsx:52-54` hardcodes a 500MB limit *and* the "500MB" text, while
the server accepts 10GB (`config/mixins/files.py:18`) and `MAX_UPLOAD_SIZE`
has zero `.ts`/`.tsx` hits. Wrong for every non-demo deployment, and it
rejects a first upload that would have succeeded.

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
- This plan lives on `080` while the code lives on `078`. That is the same
  drift it warns about, and it is resolved by the decision recorded below
  rather than by ignoring it.

## Branch decision for this folder

`initiatives/growth/` must live wherever the work happens. Since fixes land
on `078`, this folder should be merged there — or `080` merged into `078` —
before Stage 0 begins. A plan on a branch nobody builds from is exactly the
failure this folder was created to stop.
