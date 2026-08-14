# AutoML / Hyperparameter Tuning Research Scan for Skyulf

**Date:** 2026-08-11

## 0. Baseline: what Skyulf already has (verified in-repo)

Before evaluating novelty, I confirmed the current state of Skyulf's tuning engine directly:

- `skyulf-core/skyulf/modeling/_tuning/schemas.py:11` — `TuningConfig.strategy` already supports `"grid" | "random" | "optuna" | "halving_grid" | "halving_random"`.
- `skyulf-core/skyulf/modeling/_tuning/engine.py:917-935` — Optuna integration already wires up **TPE**, **CMA-ES**, and **random** samplers, plus **Hyperband**, **median**, and **no-op** pruners (`_build_optuna_sampler`, `_build_optuna_pruner`).
- `skyulf-core/skyulf/modeling/_tuning/engine.py:989` — uses `optuna.create_study(sampler=..., pruner=...)` via `OptunaSearchCV`.
- Successive-halving (`HalvingGridSearchCV`/`HalvingRandomSearchCV`, i.e. sklearn's ASHA-like early-stopping) is already used (`advanced_tuning_manager.py` imports, `schemas.py:11`).
- `initiatives/enterprise-readiness/2026-08-11-differentiation-strategy.md:160` already flags: **"AutoML/pipeline-suggestion layer"** is a named gap — "manual tuning (grid/random/halving/Optuna) requires a pre-selected model; `EDAAnalyzer` recommendations are heuristic and never assembled into a pipeline." This is the single most load-bearing existing finding this report builds on.
- No evidence anywhere in `skyulf-core` or `backend` of: **multi-objective optimization**, **meta-learning/warm-starting across runs**, or **meta-feature-based model recommendation**. Grep for `meta.feature|warm.start|multi.objective|Pareto` across the whole repo returned zero real matches in Python source (only unrelated vendor JS false-positives).

**Conclusion up front:** Skyulf's single-objective HPO story (TPE/CMA-ES/Hyperband/median-pruner via Optuna, plus sklearn grid/random/halving) is already at parity with what auto-sklearn/Optuna/AutoGluon expose publicly. The genuinely open, differentiated territory is: (1) **what to try first** (meta-learning/warm-starting, algorithm recommendation), and (2) **multi-objective tuning** (accuracy vs. latency/size/fairness) for an enterprise platform — neither of which Skyulf has, and neither of which most public tools expose well in a UI-first, node-based product.

---

## 1. Modern HPO beyond grid/random (mostly already covered — narrow gaps only)

### BOHB: Robust and Efficient Hyperparameter Optimization at Scale
Falkner, Klein, Hutter — ICML 2018. **arXiv:1807.01774**
https://arxiv.org/abs/1807.01774

Combines Bayesian optimization (a TPE-like model) with Hyperband's bandit-based early-stopping to get both strong anytime performance and fast convergence, addressing the weakness that pure Hyperband's random sampling has no guidance and pure BO scales poorly with cheap function evaluations. Confirmed via arXiv abstract og:description.

- **Differentiation vs. existing tools:** None over Skyulf's status quo — Optuna already offers Hyperband pruning combined with TPE sampling (`_build_optuna_sampler`/`_build_optuna_pruner`, `engine.py:917-935`), which is functionally BOHB's idea already wired into Skyulf's engine. **Nothing new to add here.**
- **Effort:** N/A (already implemented).

### ASHA: A System for Massively Parallel Hyperparameter Tuning
Li, Jamieson, Rostamizadeh, Gonina, Ben-Tzur, Hardt, Recht, Talwalkar — MLSys 2020. **arXiv:1810.05934**
https://arxiv.org/abs/1810.05934

Asynchronous Successive Halving Algorithm — removes Hyperband's synchronization bottleneck so idle workers immediately promote/kill trials without waiting for a full rung to finish, making it far better suited to large-scale parallel/distributed tuning than classic Hyperband or sklearn's `HalvingGridSearchCV`/`HalvingRandomSearchCV` (which are synchronous, single-machine).

- **Differentiation:** This is a genuine, concrete gap. Skyulf's `HalvingGridSearchCV`/`HalvingRandomSearchCV` (`advanced_tuning_manager.py`, `_tuning/schemas.py:11`) are **synchronous** — each rung must fully complete before promotion. Once Ray integration lands (per `initiatives/ray-migration/2026-08-10-ray-migration-design.md:84`, which already notes "Skyulf Core supports grid, random, halving, and Optuna-based search"), synchronous halving will leave workers idle waiting on stragglers. **Ray Tune ships a maintained ASHA scheduler out of the box** (`ray.tune.schedulers.ASHAScheduler`), which is the natural asynchronous replacement once distributed trials are real.
- **Integration:** Add `strategy="asha"` (or reuse `strategy_params={"scheduler": "asha"}` under the existing `optuna` strategy, since Optuna also has an `optuna.integration.SuccessiveHalvingPruner`/works with Ray Tune's `ray.tune.search.optuna.OptunaSearch` + `ASHAScheduler` combo) gated behind the Ray backend, mirroring the existing `parallel_backend`/`n_jobs` fields already in `TuningConfig` (`schemas.py:32-33`).
- **Effort:** **M** — mostly plumbing once Ray lands; the async scheduler itself is maintained by Ray, not something Skyulf writes.
- **Library:** `ray[tune]` (already the dependency the Ray migration initiative is adding) or `optuna-integration`'s Ray Tune bridge. No new dependency needed if Ray lands as planned.

### Optuna: A Next-generation Hyperparameter Optimization Framework
Akiba, Sano, Yanase, Ohta, Koyama — KDD 2019. **arXiv:1907.10902**
https://arxiv.org/abs/1907.10902

Already the exact library Skyulf depends on (`skyulf-core/setup.py:61-63`, `uv.lock:1039-1068`, pinned `>=3.0.0`, actual resolved version 4.6.0). **No action needed — already adopted**, confirming this was the right call; nothing to recommend here beyond keeping current.

---

## 2. Meta-learning / warm-starting hyperparameter search — genuinely missing, moderate effort

### Practical Transfer Learning for Bayesian Optimization
Feurer, Letham, Bakshy — 2018 (extends "Initializing Bayesian Hyperparameter Optimization via Meta-Learning," Feurer, Springenberg, Hutter, AAAI 2015). **arXiv:1802.02219**
https://arxiv.org/abs/1802.02219

This is the auto-sklearn team's own line of work: warm-start a new Bayesian optimization run by ranking/weighting prior optimization runs (from other datasets) that are similar by meta-features, and seeding the new search's initial configurations from the best-performing configs on similar past datasets. auto-sklearn's meta-learning component is exactly this — it ships a database of ~140 OpenML datasets' meta-features + best configs to warm-start new searches.

- **Differentiation vs. existing tools:** auto-sklearn does this already but it's baked into a large, opaque system users can't easily inspect or extend. Skyulf's opportunity is a **much smaller, transparent version**: persist `(dataset meta-features, model_type, best_params, score)` tuples from every completed tuning job Skyulf itself runs (this data already half-exists in the `TrainingJob` DB rows the `AdvancedTuningManager` reads — `best_params`, `best_score`, `model_type`, `search_strategy` are already columns per `advanced_tuning_manager.py:81-118`), then use nearest-neighbor lookup on dataset meta-features (n_rows, n_features, class balance, feature-type mix) to seed the next Optuna study's initial trials via `optuna.study.enqueue_trial()` — a small, already-supported Optuna API. This is a "free" enrichment of Skyulf's own historical run data that neither Optuna nor AutoGluon does automatically for you across separate datasets/projects — it becomes a genuine cross-project institutional-memory feature specific to a persistent multi-tenant platform (which generic OSS libraries, invoked fresh each time, structurally cannot offer).
- **Integration:** New small module, e.g. `skyulf-core/skyulf/modeling/_tuning/_warmstart.py`, reading from the same `TrainingJob`-equivalent history (or a lighter local store in skyulf-core itself, if core must stay backend-agnostic) and calling `study.enqueue_trial(params)` before `study.optimize(...)` in `_build_optuna_searcher` (`engine.py:937-1017`).
- **Effort:** **M** — the hard part is meta-feature extraction (can start crude: n_rows, n_cols, target cardinality, % categorical) and a similarity metric; the Optuna-side plumbing (`enqueue_trial`) is trivial.
- **Library:** No new dependency required — reuses Optuna's existing `enqueue_trial`/`study` APIs already imported in `engine.py`.

### OBOE: Collaborative Filtering for AutoML Model Selection
Yang, Akimoto, Kim, Udell — KDD 2019. **arXiv:1808.03233**
https://arxiv.org/abs/1808.03233

Frames "which algorithm/hyperparameter combo should I try first on this new dataset" as a **collaborative-filtering / matrix-completion problem**: build a matrix of (past datasets × pipeline configs) → performance, run a small set of fast "probe" configs on the new dataset to get a few matrix entries, then use low-rank matrix completion to predict the best untried configs — explicitly designed for a strict time budget, since it avoids full Bayesian optimization loops.

- **Differentiation:** This is a materially different idea from Feurer et al.'s meta-feature nearest-neighbor warm-start — no explicit meta-features required, just observed performance patterns. It gives an actual algorithm ("run cheap probes, then predict") for the exact feature the differentiation-strategy doc flags as missing: **"which model family should the pipeline canvas recommend by default?"** (`differentiation-strategy.md:160`). No public AutoML tool (auto-sklearn, TPOT, AutoGluon, H2O) exposes this recommendation as a standalone, inspectable step — it's buried inside their end-to-end search loops.
- **Integration:** A lightweight "Suggest a starting model" node/panel action on the canvas: run 3-5 very cheap probe fits (e.g. default LogisticRegression, default LightGBM with `n_estimators=50`, default KNN) against the incoming dataset, record scores, and match against Skyulf's own accumulated (dataset, model, score) history via the same warm-start data store above to rank untried model families. This directly answers the "AutoML/pipeline-suggestion layer" gap without reimplementing a full AutoML search — it's a recommendation, not an autonomous pipeline builder, matching Skyulf's stated "don't clone AutoML tools" philosophy.
- **Effort:** **M-L** — needs a persisted cross-dataset performance matrix and either a simple collaborative-filtering step (can start with plain cosine-similarity kNN instead of true matrix factorization to keep v1 small) or reuse of `scikit-learn`'s `NMF`/`TruncatedSVD` (already a dependency via sklearn, no new library needed for a v1).
- **Library:** No dedicated maintained OBOE package to wrap (the original `oboe` GitHub repo, rutgerstillman/oboe-ish, is unmaintained since ~2020) — this would be a from-scratch small implementation, not a dependency add.

---

## 3. Automated model selection / algorithm recommendation

Covered above via OBOE. One additional relevant, more recent angle:

### TabPFN: A Transformer That Solves Small Tabular Classification Problems in a Second
Hollmann, Müller, Eggensperger, Hutter — ICLR 2023. **arXiv:2207.01848**
https://arxiv.org/abs/2207.01848

A Prior-Data Fitted Network (transformer) pre-trained once offline on millions of synthetic tabular tasks, then does in-context learning at inference with **zero hyperparameter tuning** — competitive with tuned AutoML systems with up to 230x-5700x speedup, but restricted to small datasets (≤1000 rows, ≤100 numerical features, ≤10 classes in the original paper; later versions relax this).

- **Differentiation:** Interesting but narrow. It's not a tuning-engine improvement per se — it's a candidate **new model family/node** (a "zero-tuning baseline classifier" node) that could sit alongside XGBoost/LightGBM/sklearn nodes as an instant, no-search baseline for small tabular classification datasets, directly useful for the "which model should I try first" moment on the canvas without invoking the tuning engine at all.
- **Integration:** A `TabPFNClassifier` node following the existing `SklearnCalculator`/`SklearnApplier` pattern (TabPFN ships an sklearn-compatible `.fit()`/`.predict_proba()` API via the `tabpfn` PyPI package), gated to small-dataset use cases with a clear warning/fallback for datasets exceeding its regime.
- **Effort:** **S-M** for a first node (thin sklearn-compatible wrapper); **caveat** — GPU strongly recommended for reasonable latency, adds an infra dependency the platform may not want yet.
- **Library:** `tabpfn` (official, maintained by the Prior Labs/automl.org team), optional dependency.

---

## 4. Multi-objective optimization for ML pipelines — clearest concrete gap found

### Multiobjective Tree-structured Parzen Estimator for Computationally Expensive Optimization Problems
Ozaki, Tanigaki, Watanabe, Onishi — GECCO 2020. DOI: 10.1145/3377930.3389817 (journal extension: Ozaki, Tanigaki, Watanabe, Nomura, Onishi, JAIR 2022, DOI: 10.1613/jair.1.13188)
https://dl.acm.org/doi/10.1145/3377930.3389817

Extends single-objective TPE to multi-objective search (MOTPE), approximating Pareto fronts under tight evaluation budgets — this is exactly the algorithm behind **Optuna's built-in multi-objective support** (`optuna.create_study(directions=[...])`, which Optuna has shipped since v2.x using this MOTPE sampler logic internally alongside NSGA-II).

- **Differentiation:** Skyulf's `_tuning/engine.py:989` currently hardcodes `optuna.create_study(sampler=sampler, pruner=pruner, direction="maximize")` — a **single** `direction`, single metric. `TuningConfig.metric` (`schemas.py:12`) is also singular (`str`, e.g. `"accuracy"`). Optuna itself already supports multi-objective studies (`directions=["maximize", "minimize"]` for e.g. accuracy vs. inference latency, or accuracy vs. model size), so **this is a config-schema and engine change, not a new-library problem** — the capability already exists one layer down in the dependency Skyulf already ships.
- **Why it matters for Skyulf specifically:** the platform explicitly has enterprise ambitions (multi-tenancy, licensing per the roadmap) where "smallest/fastest model that's within X% of best accuracy" is a very real ask (edge/latency-sensitive deployment, cost-conscious inference). None of Skyulf's public differentiators mention this; auto-sklearn/TPOT/AutoGluon do not expose multi-objective tuning as a first-class, UI-configurable option either (H2O AutoML doesn't; AutoGluon has an experimental `Ensembles` weighting for latency but not general Pareto search) — so surfacing multi-objective (accuracy vs. latency vs. model size vs., longer-term, fairness-metric) search **on the canvas** would be a genuinely differentiated, evidence-backed enterprise feature.
- **Integration:**
  1. Extend `TuningConfig` (`schemas.py:8-33`) to accept `metrics: list[str]` + `directions: list[Literal["maximize","minimize"]]` alongside (not replacing) the existing singular `metric` field for backward compatibility.
  2. In `_build_optuna_searcher` (`engine.py:937-1017`), branch: if multiple metrics configured, use `optuna.create_study(directions=[...])` instead of `OptunaSearchCV` (which is single-objective only) — this likely means writing a **thin custom Optuna objective function** wrapping the existing `SklearnCalculator`/cross-validation path rather than `OptunaSearchCV`, similar to what `initiatives/deep-learning/2026-08-11-architecture-design.md:173` already proposes doing for DL tuning ("a small `deep_learning/_dl_tuning.py` runs an Optuna study" with "a direct Optuna objective rather than sklearn CV" per `initiatives/training-visualization/2026-08-11-feasibility-and-plan.md:212`) — meaning **the DL tuning work already underway is building the exact objective-function-based Optuna pattern this multi-objective feature would need**, so there's a natural moment to share that code path rather than duplicate it.
  3. Model size/latency as objectives are simple to compute post-fit (artifact byte size via joblib serialization already happens; predict-latency via a timed batch-predict loop).
  4. Frontend: a Pareto-front scatter plot + "pick a point" UI, selecting among trials — genuinely new UI work but conceptually simple (2D/3D scatter, already have plotly per `static/ml_canvas/assets/vendor-plotly` in the frontend bundle).
- **Effort:** **M** for the core-engine/schema change (reusing Optuna's existing multi-objective API + the objective-function pattern already being built for DL tuning); **M** additional for frontend Pareto-front UI.
- **Library:** No new dependency — Optuna (already a dependency, `optuna>=3.0.0`) supports this natively.

---

## 5. What researchers say is still broken/annoying about AutoML tools (2022-2025)

### Why do tree-based models still outperform deep learning on tabular data?
Grinsztajn, Oyallon, Varoquaux — NeurIPS 2022 (Datasets & Benchmarks). **arXiv:2207.08815**
https://arxiv.org/abs/2207.08815

Rigorously benchmarks tree ensembles (XGBoost/GBDT/RandomForest) vs. several tabular deep-learning architectures across 45 datasets and finds tree-based models still win on medium-sized tabular data, attributing this to trees' robustness to uninformative features and ability to learn irregular/non-smooth target functions that neural nets struggle with by default.

- **Relevance to Skyulf:** This is direct, citable validation of Skyulf's current bet — XGBoost/LightGBM/sklearn as the primary tabular modeling stack — **and a caution for the parallel deep-learning module initiative**: DL tabular nodes should be positioned as a complement for specific regimes (large datasets, mixed modality, embeddings) rather than a general tabular-accuracy upgrade, since the empirical literature doesn't support DL-first tabular modeling as of 2022. Worth citing directly in the deep-learning initiative's design docs (`initiatives/deep-learning/2026-08-11-architecture-design.md`) as a scoping guardrail, not a new build task.
- **Effort:** N/A — this is a positioning/messaging finding, not a code change.

### Large Language Models for Automated Data Science: Introducing CAAFE for Context-Aware Automated Feature Engineering
Hollmann, Müller, Hutter — NeurIPS 2023. **arXiv:2305.03403**
https://arxiv.org/abs/2305.03403

Uses an LLM to iteratively propose new engineered features (with code + natural-language justification) based on a dataset's textual description/schema, improving mean ROC AUC from 0.798 → 0.822 across 11/14 datasets — comparable to the lift from switching logistic regression to random forest, and notably each generated feature comes with a human-readable rationale.

- **Differentiation:** This is squarely in "genuinely differentiated, not cloning AutoML" territory: existing AutoML tools (featuretools/TPOT/auto-sklearn) do automated feature engineering via combinatorial transforms (polynomial features, aggregations), not LLM-reasoned, semantically-labeled features with human-readable rationale. For a **visual, node-based canvas** product, an "AI Suggest Feature" node that proposes a transform + plain-language explanation is a strong UX fit (explainability is already a stated Skyulf value via SHAP integration per `differentiation-strategy.md`). This pairs naturally with the platform's existing profiling/EDA layer (`profiling/recommendations.py` per the differentiation doc) rather than the tuning engine.
- **Integration:** New node type calling an LLM API (needs to fit the enterprise/licensing story re: cost + data-privacy for sending schema/samples to a third-party LLM — likely needs a "bring your own key" or on-prem-model option for enterprise buyers), producing a code snippet through the same transform/feature-engineering pipeline abstraction Skyulf already has.
- **Effort:** **L** — new infra (LLM calling, prompt engineering, sandboxed code execution/validation of generated feature code, enterprise data-privacy controls) — bigger than a tuning-engine change, more of a standalone initiative.
- **Library:** Official `caafe` PyPI package exists (from the paper authors) but is a thin wrapper Skyulf would likely want to reimplement rather than depend on directly, given the enterprise data-privacy requirements above.

### The Technological Emergence of AutoML: A Survey of Performant Software and Applications in the Context of Industry
Scriven et al. — 2022. **arXiv:2211.04148**
https://arxiv.org/abs/2211.04148

A survey specifically evaluating AutoML tools against real-world "performant" criteria beyond academic benchmarks — stakeholder needs, human-computer interaction requirements, practical deployment concerns — rather than pure leaderboard accuracy.

- **Relevance:** Useful as a framing citation for why Skyulf's node-based visual + explainability-first approach (rather than a black-box "just run AutoML" button) is aligned with what this survey identifies as unmet industry needs (interpretability, human-in-the-loop control, trust) — supports the existing differentiation-strategy narrative rather than suggesting a new build.
- **Effort:** N/A — positioning citation only.

---

## Top 5 picks (ranked by actionability)

| Rank | Finding | Paper(s) | Effort | Why it's the priority |
|---|---|---|---|---|
| **1** | **Multi-objective tuning (accuracy vs. latency/model-size)** via Optuna's existing multi-objective API | Ozaki et al., GECCO 2020 / JAIR 2022 (MOTPE, underlying Optuna's multi-objective support) | **M** | Zero new dependencies (Optuna already pinned `>=3.0.0`), directly reuses the objective-function pattern the DL-tuning initiative is *already building* (`_dl_tuning.py`), fills a real enterprise gap (latency/size-aware deployment) that public AutoML tools don't expose well, and is a schema-level change (`TuningConfig.metric` → `metrics`/`directions`) with a clear, bounded blast radius. |
| **2** | **ASHA async successive-halving scheduler** for the Ray migration | Li et al., MLSys 2020, arXiv:1810.05934 | **M** | Directly relevant to a concrete, already-planned initiative (Ray integration); Skyulf's current halving strategies are synchronous and will bottleneck under real distributed parallelism — Ray Tune ships the maintained scheduler, so this is glue work, not new algorithm implementation. |
| **3** | **Cross-run warm-starting via Optuna's `enqueue_trial`** using Skyulf's own historical tuning-job data | Feurer et al. (AAAI 2015 / arXiv:1802.02219) | **M** | Directly answers the named "AutoML/pipeline-suggestion layer" gap from the differentiation-strategy doc; leverages data Skyulf *already persists* (`TrainingJob.best_params`/`best_score`/`model_type`); no new dependency; genuinely hard for a stateless OSS library (Optuna/auto-sklearn) to offer across projects the way a persistent multi-tenant platform can. |
| **4** | **OBOE-style cheap-probe model recommendation** ("suggest a starting model" canvas action) | Yang et al., KDD 2019, arXiv:1808.03233 | **M-L** | Same differentiation-strategy gap as #3, complementary algorithm (collaborative filtering over observed performance vs. meta-feature similarity) — worth prototyping *after* #3 since it can reuse the same historical-run data store; no maintained library to wrap, so it's more implementation work. |
| **5** | **TabPFN as a zero-tuning baseline node** for small tabular datasets | Hollmann et al., ICLR 2023, arXiv:2207.01848 | **S-M** | Cheapest to prototype (official maintained `tabpfn` PyPI package, sklearn-compatible API fits the existing `SklearnCalculator` pattern directly); scope-limited (small datasets only) so it's a genuinely bounded, low-risk addition rather than a strategic bet — good for a fast "instant baseline" UX win on the canvas. |

**Honorable mention, not ranked (positioning-only, no code):** Grinsztajn et al. (arXiv:2207.08815) as a citable guardrail for scoping the parallel deep-learning module's marketing/positioning — trees still win on typical tabular sizes, so DL nodes should target genuinely DL-favorable regimes (large data, embeddings, multi-modal) rather than being pitched as a general accuracy upgrade over the existing XGBoost/LightGBM stack.

