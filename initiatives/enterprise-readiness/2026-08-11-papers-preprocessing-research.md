# Papers & Research for skyulf-core Preprocessing Improvements

**Date:** 2026-08-11

Research scan of arXiv (via the arXiv API, `export.arxiv.org/api/query`) plus known canonical papers, focused on
concrete, non-redundant improvement ideas for skyulf-core's Calculator/Applier preprocessing architecture.
Every claim below is backed by a fetched/verified arXiv abstract or well-known citation. Searches that returned
nothing beyond mainstream/already-known methods are reported as such rather than forced into weak findings.

---

## 1. Categorical encoding beyond what skyulf-core has

**Interpretable vs Learned Encoders for High-Cardinality Fraud Detection** — Han, Liu, Zheng, Zhang, Wu (2026),
arXiv:2607.00477. Controlled comparison of 7 encoders (incl. target encoding, entity embeddings, CatBoost,
TabNet) on the IEEE-CIS fraud dataset. Finding: entity embeddings tie statistically with CatBoost's ordered
target statistics and beat plain target/tier-group encoding; TabNet underperforms tree pipelines under scarcity.
- Differentiation: skyulf-core has target/WOE/hash/one-hot but no *learned entity embedding* encoder trainable
  via a small feed-forward net at fit time, serialized as a lookup table.
- Calculator/Applier mapping: Calculator trains a tiny embedding table (category → k-dim vector) via a shallow
  net against the target at fit time; Applier is a pure JSON-serializable lookup + fallback-vector for unseen
  categories. This keeps the "JSON not pickle" constraint if the embedding table is stored as arrays.
- Effort: **Medium** — needs a lightweight torch/numpy training loop, but is conceptually simple (single hidden
  layer per Guo & Berkhahn below).
- Implementation: no turnkey sklearn-transformer-style implementation to wrap directly for tabular numeric target
  regression/classification generically (Keras/PyTorch example code exists in the original paper's repo). Would
  need custom implementation on top of numpy/torch.

**Entity Embeddings of Categorical Variables** — Cheng Guo, Felix Berkhahn (2016), arXiv:1604.06737. The
foundational paper (3rd place Kaggle Rossmann) showing entity embeddings beat one-hot for high-cardinality sparse
categorical features, and that the learned embeddings can be reused as features for *other* models (gradient
boosting, kNN, etc.), not just neural nets.
- Differentiation: general "distance measure for categories" is not in feature-engine/category_encoders.
- Calculator/Applier mapping: same as above; embedding vectors become new numeric columns, replacing single
  encoded columns — natural fit for skyulf-core's column-transform pattern.
- Effort: **Medium**.
- Implementation: reference implementation in paper's GitHub repo (Keras); needs porting.

**A Comparison of Machine Learning Methods for Data with High-Cardinality Categorical Variables** — Fabio
Sigrist (2023), arXiv:2307.02071. Empirically shows GLMM-style random-effects encodings (mixed-effects models)
combined with tree boosting outperform both classical encoders and plain neural nets for high-cardinality
categoricals.
- Differentiation: skyulf-core doesn't have a "random effects"/hierarchical-shrinkage encoder distinct from its
  target encoder — this is conceptually closer to James-Stein/hierarchical Bayes shrinkage (already flagged as a
  known gap), so treat as reinforcing evidence rather than a new idea. **Not a new pick**, but useful supporting
  citation if/when the team implements the already-known James-Stein/hierarchical shrinkage encoder gap.

**A reproducible comparative study of categorical kernels for Gaussian process regression, with new
clustering-based nested kernels** — Carpintero Perez, Da Veiga, Garnier (2025), arXiv:2510.01840. Niche (GP
regression specific); not directly actionable for a general preprocessing library. **Skip.**

---

## 2. Missing-data imputation beyond simple/KNN/iterative

**GAIN: Missing Data Imputation using Generative Adversarial Nets** — Yoon, Jordon, van der Schaar (ICML 2018),
arXiv:1806.02920. The canonical GAN-based imputer: a generator imputes missing components conditioned on
observed ones; discriminator (given a "hint vector") tries to guess which components were imputed. Reported to
significantly outperform classical methods (mean/kNN/MICE) on several UCI benchmarks.
- Differentiation: genuinely not in sklearn/feature-engine — those stick to statistical imputers. GAIN would be
  a first-of-kind deep imputer in skyulf-core.
- Calculator/Applier mapping: Calculator trains the GAN on the fit split only (critical for leakage-safety, which
  is exactly skyulf-core's core design constraint) and serializes generator weights (as JSON-encoded arrays, or a
  companion binary artifact referenced by the JSON manifest — may require relaxing the "pure JSON" constraint for
  this one component, similar to how sentence-transformer embeddings are presumably already handled for NLP).
  Applier runs the frozen generator forward pass on new rows with missing masks.
- Effort: **Large** — training stability (GAN), needs a torch dependency, meaningfully more complex than existing
  imputers, but a strong differentiator versus feature-engine/scikit-learn which have zero GAN-based imputation.
- Implementation: no actively maintained sklearn-compatible package wraps GAIN directly; a few community forks
  exist but nothing production-grade — would need a from-scratch, dependency-light implementation.

**MIWAE: Deep Generative Modelling and Imputation of Incomplete Data** — Mattei & Frellsen (2018),
arXiv:1812.02633. Importance-weighted autoencoder adapted for missing-at-random data; trains VAEs directly on
incomplete data without imputing first, and supports both single and multiple imputation via Monte Carlo. Shown
competitive with/better than state-of-the-art on continuous and binary UCI data.
- Differentiation: same rationale as GAIN — no sklearn/feature-engine equivalent.
- Calculator/Applier mapping: same pattern as GAIN; VAE weights become the fit-time artifact.
- Effort: **Large** — VAE training + importance-weighted ELBO estimation is nontrivial but well-documented;
  training is generally more stable than GANs (a practical advantage over GAIN if only one deep imputer is added).
- Implementation: reference PyTorch code exists from the authors (linked in later papers); would need integration
  work, not a pip-installable sklearn-style wrapper.

**Missing Data Imputation using Optimal Transport** — Muzellec, Josse, Boyer, Cuturi (2020), arXiv:2002.03860.
Non-neural-net-heavy alternative: treats imputation as matching the distribution of two random batches from the
same dataset via optimal transport (Sinkhorn divergence), avoiding full generative-model training. Reported to
match or beat SOTA imputers across MCAR/MAR/MNAR settings on UCI data.
- Differentiation: much lighter-weight than GAIN/MIWAE (no adversarial or variational training loop needed —
  just a differentiable optimal-transport loss minimized via gradient descent on the missing entries themselves),
  and not present in sklearn/feature-engine.
- Calculator/Applier mapping: Calculator computes an OT-based imputation directly by optimizing missing values
  against training-batch distributions (this is more "transductive" than typical fit/predict — may need adapting
  to a fit-once, reusable-transform design, e.g., learn a small reference set + do OT matching against it at
  apply-time).
- Effort: **Medium** — simpler to implement than deep generative options; author code (`github.com/BorisMuzellec/MissingDataOT`) exists as reference, uses PyTorch/POT libraries.
- **Recommendation among the three: if skyulf-core adds exactly one deep/statistical imputer beyond
  simple/KNN/iterative, Optimal-Transport imputation offers the best effort-to-benefit ratio** (lighter
  dependency footprint, no adversarial training instability, competitive published results).

**Missing Data Imputation using Neural Cellular Automata** — Luu, Nguyen, Ngo (2025), arXiv:2509.00651. Very
recent (Sept 2025), claims to beat SOTA generative imputers using Neural Cellular Automata instead of GAN/VAE.
Interesting but too new/unproven and NCA is an unusual, less battle-tested paradigm for a production library.
- Effort: **Large**, and higher risk given lack of maturity/adoption. **Lower priority than GAIN/MIWAE/OT.**

**Diffusion Models for Tabular Data Imputation and Synthetic Data Generation** — (2024/2025), arXiv:2407.02549.
Diffusion-based imputer with attention-conditioning and dynamic masking; unifies imputation + synthetic data
generation. Technically strong but heavyweight (transformer denoiser) — overkill relative to skyulf-core's
classical-ML focus. **Lower priority — flag as a "watch" item, not actionable near-term.**

---

## 3. Outlier / anomaly detection beyond IQR/z-score

Searched arXiv extensively for "isolation forest," "local outlier factor," "minimum covariance determinant,"
"conformal prediction + anomaly detection." **Finding: no genuinely novel, broadly-applicable, easy-to-integrate
tabular anomaly-detection method emerged that isn't already implemented and mature in PyOD.** The classic
algorithms (Isolation Forest, LOF, One-Class SVM, Robust Covariance/MCD, Mahalanobis distance) are all
well-established (pre-2020) and already available via scikit-learn (`IsolationForest`, `LocalOutlierFactor`,
`EllipticEnvelope`) — so building Calculator/Applier wrappers around these would be "wrapping sklearn," a
reasonable but non-research-differentiated feature.

**PyOD: A Python Toolbox for Scalable Outlier Detection** — Zhao, Nasrullah, Li (JMLR 2019), arXiv:1901.01588,
and **PyOD 2** — (2024), arXiv:2412.12154 (adds 12 deep-learning OD models unified in PyTorch + LLM-based model
selection). This is the most relevant concrete opportunity: PyOD is a mature, actively maintained library (25M+
downloads) with 45 algorithms behind one API.
- Differentiation for skyulf-core: not the algorithms themselves (well known) but **wrapping PyOD's unified API
  as Calculator/Applier pairs** would let skyulf-core jump straight to Isolation Forest, LOF, COPOD, ECOD,
  HBOS, and even a few of PyOD2's deep models — well beyond IQR/z-score — with comparatively low
  implementation effort, since PyOD already handles `fit`/`decision_function` cleanly.
- Calculator/Applier mapping: Calculator calls `PyODModel.fit(X_train)`, serializes minimal fitted parameters
  (for parametric models like `EllipticEnvelope`/`HBOS`/`COPOD`) or the trained estimator's simple params to JSON
  where possible; for tree ensembles (Isolation Forest) either accept a pickled sub-artifact (breaking the pure-
  JSON constraint, similar caveat as GAIN/MIWAE) or reimplement a lightweight variant. Applier calls
  `decision_function`/`predict` on new data.
- Effort: **Small–Medium** for classical PyOD models (HBOS, COPOD, ECOD — these are parametric/histogram-based
  and easy to serialize as JSON, no pickling needed), **Medium–Large** if wrapping Isolation Forest/LOF (tree-
  or graph-based, harder to serialize without pickling).
- Implementation: `pyod` package is maintained and pip-installable — direct wrapping opportunity rather than
  reimplementation from scratch.

**Unsupervised Machine Learning for Detecting Structural Anomalies in European Regional Statistics** — Oancea
(2026), arXiv:2605.02884. Practical validation study using an *ensemble-vote* approach (a region flagged
anomalous only if ≥3 of 5 methods — z-score, Mahalanobis, Isolation Forest, LOF, One-Class SVM — agree).
- Differentiation: the specific idea worth borrowing is the **multi-method consensus/voting outlier flag**
  rather than any single new algorithm — i.e., an ensemble Calculator that runs several outlier detectors and
  flags rows only above an agreement threshold, reducing false positives versus a single IQR/z-score rule.
- Effort: **Small** — orchestration wrapper over the already-implemented IQR/z-score plus 1-2 PyOD models; no new
  ML architecture required.
- Implementation: straightforward to implement in-house since it is just a voting layer on top of existing/PyOD
  scores.

**Conformal-prediction-based anomaly detection**: Several 2025-2026 papers (e.g. arXiv:2607.25020, "Localized
Anomaly Detection via Differentiable D-vine Copulas," using Mondrian conformal prediction for anomaly score
calibration) exist, but they target specialized domains (astrophysics, copula models) with heavy statistical
machinery not well-suited to a general tabular preprocessing library. **Not recommended for near-term adoption** —
flag as a "watch" area if skyulf-core later wants calibrated/guaranteed false-positive-rate outlier flags.

---

## 4. Automatic / adaptive binning and discretization

**Optimal binning: mathematical programming formulation** — Guillermo Navas-Palencia (2020, updated 2022),
arXiv:2001.08025. Formalizes optimal binning (binary/continuous/multiclass target) as a convex mixed-integer
program, with automatic monotonic-trend detection via an ML classifier. This is the paper behind the
**OptBinning** open-source Python library.
- Differentiation: skyulf-core currently has basic binning but nothing that (a) enforces a monotonic
  relationship with the target (important for scorecards/credit-risk use cases where skyulf-core already
  supports WOE encoding) or (b) optimizes bin boundaries via MIP rather than simple quantile/equal-width cuts.
  This is a strong complement to the existing WOE encoder.
- Calculator/Applier mapping: Calculator runs OptBinning's solver on the fit split, serializes the resulting bin
  edges + WOE/event-rate per bin as JSON (this is naturally JSON-friendly — bin edges and per-bin stats are
  just numbers). Applier just does a `searchsorted`-style bucket lookup — very lightweight and fully
  train/apply separated, matching skyulf-core's leakage-safety philosophy closely.
- Effort: **Small–Medium** — `optbinning` package (by the paper's author) is maintained, pip-installable, and
  MIT-licensed; wrapping it as a Calculator/Applier pair is largely plumbing work, not new algorithm design.
- Implementation: **direct wrap of the `optbinning` package** — best effort-to-benefit ratio of anything found
  in this scan.

No further genuinely novel adaptive-binning papers emerged beyond OptBinning/ChiMerge/MDLP-style classics (which
predate 2000 and are already well known / present in some form in existing discretizers) — search for
"discretization + supervised + decision tree + binning" on arXiv returned **zero results**, confirming this is a
saturated classical area with OptBinning as the standout modern, well-engineered exception.

---

## 5. Feature selection beyond correlation

**Distance Correlation Sure Independence Screening (DC-SIS) vs mRMR** — Schellhas, Neupane, Thammineni,
Kanumuri, Green (2020), arXiv:2006.12919. Shows DC-SIS (distance correlation-based screening, Székely et al.
2012 method) achieves statistically indistinguishable accuracy to mutual-information-based mRMR feature
selection, at ~90x lower computation time on a Parkinson's vocal dataset.
- Differentiation: skyulf-core has Pearson/Spearman correlation selection but no **mutual-information-based**
  or **distance-correlation-based** (nonlinear dependency) feature selection — both would catch non-monotonic
  relationships that Pearson/Spearman miss.
- Calculator/Applier mapping: Calculator computes mutual information (`sklearn.feature_selection.mutual_info_classif/regression`, already MIT-licensed and maintained) or distance correlation between each feature and the
  target on the fit split, thresholds/ranks, and serializes the selected feature list + scores to JSON. Applier
  is a pure column-selection/reorder step — trivial.
- Effort: **Small** — mutual information is literally one `sklearn` function call; distance correlation needs a
  small custom function (`dcor` package exists on PyPI) but is still simple.
- Implementation: `sklearn.feature_selection.mutual_info_classif`/`mutual_info_regression` already do the heavy
  lifting; `dcor` package for distance correlation. **This is a fast, low-risk win** — arguably the single
  cheapest genuinely new capability found in this whole scan.

**Boruta algorithm — Noise-Augmented Boruta** (Gharoun, Yazdanjoe, Khorshidi, Gandomi, 2023), arXiv:2309.09694,
and **Novel GPU Boruta algorithms** (2026), arXiv:2605.09950. Boruta itself (shadow-feature permutation testing
against Random Forest importance) is well known and the second paper is literally about GPU-accelerating it, i.e.
Boruta itself is mature (already flagged as excluded by the task's known-gaps list is not quite accurate — Boruta
is NOT currently in skyulf-core per the task description's "coverage" list, so it may still be worth adding). The
"Noise-Augmented" variant improves on classic Boruta by injecting noise into shadow features to reduce bias
toward features correlated with dataset-wide noise characteristics; shown to outperform classic Boruta on 4
benchmark datasets.
- Differentiation: genuinely not in sklearn (Boruta lives in the separate `boruta_py`/`BorutaPy` package,
  unmaintained since ~2021) or feature-engine — a maintained, modernized Boruta variant would be a real add.
- Calculator/Applier mapping: Calculator runs the shadow-feature + Random Forest importance iterations on the
  fit split, serializes the final "confirmed/tentative/rejected" feature list to JSON. Applier is column
  selection only.
- Effort: **Medium** — the noise-augmented variant isn't packaged; would need reimplementation on top of
  `scikit-learn`'s `RandomForestClassifier/Regressor`, using classic BorutaPy's algorithm as a base and adding
  the noise-injection step described in arXiv:2309.09694.
- Implementation: `BorutaPy` (unmaintained, but small/forkable codebase) as a starting point; noise augmentation
  is a small delta on top.

**SHAP-based feature selection**: broad search found this to be a widely-used *technique* rather than a single
citable breakthrough paper — most 2025-2026 papers use SHAP-based selection as a baseline/component rather than
proposing a new method (e.g. arXiv:2608.04180 "opioid use disorder prediction" compares LightGBM-SHAP selection
against other paradigms as one baseline among several, finding it not the best performer). **Conclusion: SHAP-
based feature selection is not a strong differentiated pick** — it's essentially "compute SHAP values, rank,
threshold," already easily doable with the maintained `shap` package and arguably lower priority than mutual-
information/DC-SIS or Boruta, since it requires a trained model first (chicken-and-egg with the Calculator/
Applier fit-time design, since it needs an already-fitted downstream model rather than being a pure preprocessing
statistic).

---

## 6. Leakage-safe pipeline design (academic validation of skyulf-core's own architecture)

**LeakageDetector 2.0: Analyzing Data Leakage in Jupyter-Driven Machine Learning Pipelines** — Truong, Zhang,
Marchareddy, Lee, Busold, Socas, AlOmar (2025), arXiv:2509.15971 (and its earlier version, arXiv:2503.14723).
Builds a VS Code extension + LLM-driven fixer that detects "Overlap," "Preprocessing," and "Multi-test" leakage
categories in notebooks. Directly validates that **"Preprocessing leakage"** (fitting scalers/imputers/encoders
on the full dataset before train/test split) is a recognized, formally-named failure mode in the ML engineering
literature — exactly what skyulf-core's Calculator/Applier split is designed to structurally prevent.
- Use for skyulf-core: strong citation to justify/market the Calculator/Applier design in docs/papers — "our
  architecture eliminates by construction the 'Preprocessing Leakage' category formally identified in
  [Truong et al. 2025]." Not a code change, but valuable positioning material.

**bioLeak: Leakage-Aware Modeling and Diagnostics for Machine Learning in R** — Korkmaz (2026), arXiv:2604.10965.
An R package explicitly built around "train-fold-only preprocessing" (i.e., a Calculator/Applier-equivalent
pattern) plus post-hoc leakage audits and HTML reporting, targeting biomedical data with repeated
measurements/batch effects/temporal dependencies.
- Idea worth borrowing: **a post-hoc "leakage audit" report** — i.e., skyulf-core could add a diagnostic feature
  that automatically flags pipelines where a Calculator's fit-time state statistically resembles information
  from the full dataset rather than the train fold only (analogous to bioLeak's audit step), and/or a
  standardized "leakage-safety report" artifact alongside the existing JSON manifests, aimed at group-structured
  data (repeated measurements, batch effects) — connects with skyulf-core's already-known "no group-aware CV"
  gap.
- Effort: **Medium** — would require designing statistical leakage-audit heuristics (e.g., compare per-fold
  statistic drift) rather than adapting existing code.
- Implementation: no direct Python port of bioLeak exists; would be original design work, informed by the paper's
  audit methodology.

**scicode-lint: Detecting Methodology Bugs in Scientific Python Code with LLM-Generated Patterns** —
arXiv:2603.17893 (2026). An LLM-based linter for detecting leakage/incorrect CV/missing seeds in scientific
Python code. Interesting adjacent tooling idea (a "does this notebook correctly use skyulf-core's
Calculator/Applier API" linter) but out of scope for the preprocessing library itself — more of a companion
tool. **Noted but not prioritized.**

---

## 7. Automatic feature interaction discovery

**AutoCross: Automatic Feature Crossing for Tabular Data in Real-World Applications** — Luo, Wang, Zhou, Yao, Tu,
Chen, Yang, Dai (4Paradigm, 2019), arXiv:1904.12857. Beam search over a tree-structured space to automatically
generate high-order categorical feature crosses, with successive mini-batch gradient descent for efficient
evaluation and "multi-granularity discretization" to cross continuous with categorical features. Deployed in
production at 4Paradigm across banking/hospital/internet customers.
- Differentiation: skyulf-core has no automatic feature-crossing/interaction-discovery Calculator at all — this
  is a clean, real gap. Not present in sklearn/feature-engine (`PolynomialFeatures` only does naive exhaustive
  numeric interactions, not smart categorical crossing with beam search).
- Calculator/Applier mapping: Calculator performs beam search over candidate categorical-column crosses on the
  fit split (using a fast proxy metric like AUC-lift with a simple linear model), keeps only the top-k
  discovered crosses, serializes the winning combinations (e.g. `["city", "device_type"]` → new joint category
  mapping) to JSON. Applier just concatenates/looks up the same columns and applies the same encoding used for
  the discovered cross (piggybacking on skyulf-core's existing encoders).
- Effort: **Medium–Large** — beam search + evaluation loop is nontrivial engineering, though algorithmically
  well-specified in the paper.
- Implementation: no maintained open-source package implements AutoCross directly (it was an internal 4Paradigm
  tool); would need from-scratch implementation guided by the paper's pseudocode.

---

## 8. Related survey / positioning references (not code changes, but citation-worthy)

**Why do tree-based models still outperform deep learning on tabular data?** — Grinsztajn, Oyallon, Varoquaux
(NeurIPS 2022), arXiv:2207.08815. Large benchmark (45 datasets) showing tree-based models (XGBoost, Random
Forest) remain SOTA for medium-sized tabular data, and identifying that NN's weakness stems from being less
robust to uninformative features, not preserving data orientation, and struggling with irregular functions.
- Use for skyulf-core: strong evidence-based justification for skyulf-core's classical-ML/tree-model focus (and
  for prioritizing feature engineering quality — like mutual-information selection or optimal binning above —
  over deep tabular architectures) rather than investing engineering effort into deep tabular models.

**Data-centric Artificial Intelligence: A Survey** — Zha, Bhat, Lai, Yang, Jiang, Zhong, Hu (2023),
arXiv:2303.10158. Broad taxonomy of data-centric AI (training data development, inference data development, data
maintenance). Useful high-level framing/citation for skyulf-core's overall value proposition (a data-centric
preprocessing tool), but not a source of a specific new feature.

**Unreflected Use of Tabular Data Repositories Can Undermine Research Quality** — Tschalzev, Purucker, Lüdtke,
Hutter, Bartelt, Stuckenschmidt (2025), arXiv:2503.09159. Documents how "inappropriate preprocessing" choices
(one of three failure modes identified) commonly corrupt benchmark results when using OpenML-style repositories.
Reinforces the case for a rigorously-designed, leakage-safe, inspectable preprocessing library like skyulf-core —
good supporting citation, not a feature request.

---

## Gaps and uncertainties

- Direct web search engines (Google) were not usable in this environment (returned only a stub "Google Search"
  page); all research was conducted via the arXiv API (`export.arxiv.org/api/query`), which is comprehensive for
  arXiv-hosted papers but does **not** cover KDD/VLDB proceedings papers that are not also mirrored on arXiv
  (e.g., the classic Kaufman et al. "Leakage in Data Mining" KDD 2011 paper, or Schelter et al.'s "Automating
  Large-Scale Data Quality Verification" VLDB 2018 / Deequ paper — I attempted to find the latter by exact title
  on arXiv and got zero results, meaning it likely was never posted to arXiv; I could not independently verify
  its content in this session and have therefore excluded it rather than citing from memory).
- Data quality / schema-validation academic literature (Pandera/Great-Expectations-style contracts) returned
  **zero results** on arXiv for the specific query `"data validation" + "schema" + "machine learning pipelines"`
  — this appears to be an area covered more in industry blog posts/VLDB-Industry-track papers than arXiv
  preprints; a follow-up search on ACM Digital Library / VLDB proceedings directly (not accessible from this
  environment) is recommended if the main agent wants deeper academic backing for schema-contract features.
  This sub-topic is reported as: **searched but found nothing substantive on arXiv.**
- Gaussian-rank / rank-based transformation for tabular data (a technique folklore-famous from Kaggle, e.g. Michael Jahrer's Porto Seguro solution) returned **zero arXiv results** — it appears to be a Kaggle-forum-only
  technique without a formal paper, so I cannot cite it per this task's citation-mandatory rules. Noting its
  existence but excluding it as a "pick" since no citable source was found.
  A follow-up: search Kaggle write-ups / blog citations if the main agent wants to include it anyway with a
  non-arXiv citation.
- Conformal-prediction-based anomaly detection is an active area but the papers found are domain-specific
  (astrophysics transients, copula models) rather than general tabular-data-ready — flagged as "watch," not
  actionable now.
- SHAP-based feature selection: could not find a single strong, differentiated arXiv paper proposing a novel
  SHAP-selection algorithm (most papers use it as a baseline) — concluded this is not a strong pick, as noted
  in Section 5.

---

## Top 5 picks (ranked by actionability × differentiation)

1. **Optimal Binning (mathematical-programming / monotonic binning)** — Navas-Palencia (2020/2022),
   arXiv:2001.08025. **Effort: Small–Medium.** Direct wrap of the maintained, MIT-licensed `optbinning` package.
   Naturally JSON-serializable (bin edges + stats), fits skyulf-core's Calculator/Applier pattern almost
   perfectly, and directly complements the existing WOE encoder for credit-risk/scorecard use cases. **Best
   overall effort-to-benefit ratio found in this scan.**

2. **Mutual-information / distance-correlation feature selection** (beyond Pearson/Spearman) — supported by
   Schellhas et al. (2020), arXiv:2006.12919 (DC-SIS vs mRMR comparison). **Effort: Small.** `sklearn.feature_
   selection.mutual_info_classif/regression` already implements the core method; `dcor` package covers distance
   correlation. Captures nonlinear/non-monotonic feature-target relationships that correlation-based selection
   misses — cheapest genuinely new capability identified.

3. **PyOD wrapper for outlier detection (HBOS/COPOD/ECOD/Isolation Forest/LOF)** — Zhao et al. (2019),
   arXiv:1901.01588; PyOD 2 (2024), arXiv:2412.12154. **Effort: Small (parametric models) to Medium (tree/graph-
   based models).** Maintained, pip-installable, 45-algorithm library — moves skyulf-core well beyond IQR/
   z-score with modest wrapping effort rather than new algorithm R&D. Pair with the **multi-method consensus
   voting** idea from Oancea (2026), arXiv:2605.02884, for a low-risk "ensemble outlier flag" feature.

4. **Optimal-Transport-based missing-data imputation** — Muzellec, Josse, Boyer, Cuturi (2020),
   arXiv:2002.03860. **Effort: Medium.** The best cost/benefit deep-imputation option versus GAIN (Yoon et al.
   2018, arXiv:1806.02920, Large effort, GAN instability) or MIWAE (Mattei & Frellsen 2018, arXiv:1812.02633,
   Large effort, VAE). Reference implementation exists (`MissingDataOT`); no adversarial training loop needed;
   genuinely beyond simple/KNN/iterative imputation and not present in sklearn/feature-engine.

5. **Noise-Augmented Boruta feature selection** — Gharoun et al. (2023), arXiv:2309.09694, building on classic
   Boruta (not currently in skyulf-core per task description). **Effort: Medium.** Modernizes an
   unmaintained-but-valuable algorithm (`BorutaPy` hasn't been updated since ~2021) with a measurable
   improvement (noise-injected shadow features reduce bias), giving skyulf-core a maintained, differentiated
   wrapper-embedded feature-selection method beyond correlation and beyond plain permutation importance.

**Honorable mention (positioning/citation value, not code):** Grinsztajn et al. "Why do tree-based models still
outperform deep learning on tabular data?" (NeurIPS 2022, arXiv:2207.08815) and the LeakageDetector papers
(arXiv:2509.15971 / arXiv:2503.14723) — both are excellent evidence-based justifications for skyulf-core's
classical-ML, leakage-safe-by-construction design philosophy, worth citing in project docs/README even though
they don't map to a specific new Calculator/Applier feature.
