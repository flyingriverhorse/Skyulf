# Beyond the Loss Curve: Research-Backed Ideas for Skyulf's DL Module & Training Visualization

**Date:** 2026-08-11

**Scope note:** This report deliberately skips ideas already decided (streaming loss/accuracy curves, confusion matrices, ROC/PR curves, residual plots). It focuses on what's genuinely *next*.

---

## 1. Tabular deep learning architecture choice (what to actually ship first)

The core finding across the tabular-DL literature (2021–2023) is a *consistent, evidence-backed recommendation*: don't chase novel architectures — ship a strong, simple baseline plus one attention-based model, and be honest with users that trees may still win on small/medium tabular data.

- **Grinsztajn, Oyallon, Varoquaux, "Why do tree-based models still outperform deep learning on tabular data?"** (NeurIPS 2022 Datasets & Benchmarks), https://arxiv.org/abs/2207.08815 — 45-dataset, 20,000-compute-hour benchmark. Tree ensembles (XGBoost, RandomForest) remain SOTA on medium-sized tabular data (~10K rows). Identifies *why* NNs struggle: (1) not robust to uninformative/irrelevant features, (2) don't preserve feature-orientation invariances that trees exploit, (3) struggle with irregular/discontinuous target functions. **Actionable:** Skyulf's DL module should surface a warning/diagnostic node — "this dataset has N uninformative features / high feature-count-to-row ratio; a tree-based model in the classical-ML canvas may outperform DL here" — a genuinely differentiating, evidence-grounded nudge rather than blindly promoting the new DL canvas.
- **Gorishniy, Rubachev, Khrulkov, Babenko, "Revisiting Deep Learning Models for Tabular Data"** (NeurIPS 2021), https://arxiv.org/abs/2106.11959 — Establishes a ResNet-like MLP baseline and a tabular-adapted **FT-Transformer** (Feature Tokenizer + Transformer) as the two architectures that should be "hard to beat" baselines going forward; shows most "novel" architectures don't actually beat these two under fair tuning. **Actionable:** Ship exactly these two as skyulf's first two DL node types (`ResNetTabular`, `FT-Transformer`) rather than reinventing something bespoke — this is literally the paper's stated goal (raising the baseline bar for practitioners).
- **Shwartz-Ziv & Armon, "Tabular Data: Deep Learning is Not All You Need"** (2021), https://arxiv.org/abs/2106.03253 — Independent benchmark reaching the same conclusion: XGBoost is still hard to beat, and DL tabular models require much heavier tuning/compute for comparable accuracy. **Actionable:** justifies an explicit "compute cost vs. accuracy" disclosure panel when a user picks a DL node over a classical node for tabular data — set correct expectations rather than oversell the DL module.
- **Arik & Pfister, "TabNet: Attentive Interpretable Tabular Learning"** (AAAI 2021), https://arxiv.org/abs/1908.07442 — Sequential attention mechanism that selects a sparse feature subset at each decision step, producing per-step interpretable feature masks natively (no need for a separate post-hoc explainer). **Actionable:** if skyulf includes TabNet as an architecture option, its attention masks give a "free" interpretability node output (see §5) at zero extra inference cost — a strong differentiator vs. plain MLP/ResNet nodes that need bolted-on interpretability.
- **Hollmann et al., "TabPFN: A Transformer That Solves Small Tabular Classification Problems in a Second"** (ICLR 2023), https://arxiv.org/abs/2207.01848 — A Prior-Data Fitted Network (in-context learning transformer) that does classification on small tabular datasets (≤1000 rows, ≤100 numeric features) with **no training/tuning required**, competitive with tuned AutoML, up to 5700× faster with GPU. **Actionable — high value, low effort:** for skyulf's "small dataset" use case this could be a *zero-configuration DL node* ("Instant Tabular Classifier") — no epochs, no streaming loss curve needed at all (ties back cleanly to the viz study's finding that non-iterative methods don't need streaming curves), giving users a fast DL-flavored option without the complexity of configuring a real training run. Good onboarding/demo feature.
- **Gorishniy et al., "TabR: Tabular Deep Learning Meets Nearest Neighbors in 2023"**, https://arxiv.org/abs/2307.14338 — Retrieval-augmented tabular DL (attends over nearest neighbors in the training set) closes much of the remaining gap to GBDT on medium data. More novel/heavier to implement than FT-Transformer; good **v2 roadmap item**, not v1.
- **Chen et al., "Trompt: Towards a Better Deep Neural Network for Tabular Data"** (ICML 2023), https://arxiv.org/abs/2305.18446 — Prompt-learning-inspired architecture separating "learned per-feature prompts" from a shared table-agnostic backbone; competitive with tree methods on multiple benchmark suites. Worth watching, not prioritizing for v1 (more novel path, more implementation risk).

**Bottom line for architecture prioritization:** ship **FT-Transformer + ResNet-MLP baseline** as the core two node types (per Gorishniy et al. 2021's explicit recommendation), add **TabNet** if interpretability-as-a-feature is wanted, and consider **TabPFN** as a "instant/no-config" node for small datasets. Explicitly do NOT lead with exotic new 2023-25 architectures (TabR/Trompt) for v1 — literature shows the accuracy delta over the two baselines is inconsistent across benchmarks while implementation/tuning cost is much higher.

---

## 2. Training diagnostics beyond the loss curve

- **Li, Xu, Taylor, Studer, Goldstein, "Visualizing the Loss Landscape of Neural Nets"** (NeurIPS 2018), https://arxiv.org/abs/1712.09913 — Introduces "filter normalization" to make 2D/3D loss-surface visualizations comparable across architectures; shows skip connections flatten/smooth the landscape (correlates with easier training and better generalization).
  - *Value-add:* genuinely goes beyond what Lightning/Keras ship (they only give scalar loss curves, never landscape geometry).
  - *Canvas mapping:* a **"Loss Landscape" diagnostic node** — after training, take 2 random/PCA directions in weight space, do a small grid of forward passes, render a contour/3D surface plot showing where the final model sits (sharp vs. flat minimum). Flat minima are correlated with better generalization — could feed a one-line automated verdict: "This model converged to a sharp minimum; consider more regularization or a smaller learning rate."
  - *Effort:* **M** — requires re-running forward passes along 2 directions (cheap for small tabular nets, costly for larger ones — should be opt-in/sampled). No off-the-shelf library; would build on top of `loss-landscapes` (PyPI package implementing this paper) rather than from scratch.

- **Schoenholz, Gilmer, Ganguli, Sohl-Dickstein, "Deep Information Propagation"** (ICLR 2017), https://arxiv.org/abs/1611.01232 — Mean-field theory showing there are hard *depth limits* past which gradient/signal information cannot propagate through randomly-initialized nets ("edge of chaos" analysis), giving a principled way to predict a priori whether a given depth/init/activation combination is trainable at all.
  - *Value-add:* most frameworks only let you observe vanishing/exploding gradients empirically after they happen (via gradient-norm logging). This gives a **pre-flight predictive diagnostic**.
  - *Canvas mapping:* a lightweight pre-training sanity-check node/panel: before starting a real run, do a forward+backward pass on a single random batch with the chosen architecture/init/activation, and report per-layer gradient-norm ratios as a simple "signal propagation health" gauge (red/yellow/green), flagging likely vanishing/exploding gradients *before* the user burns compute.
  - *Effort:* **S–M** — the empirical diagnostic (per-layer gradient-norm histogram over epoch 0) is simple to build from scratch on top of PyTorch hooks; doesn't need the full mean-field theory, just its practical takeaway (watch per-layer gradient norm ratios early).
  - **Practical implementation note:** this is NOT something PyTorch Lightning/Keras callbacks give you by default (they give you scalar loss/lr, not per-layer gradient-norm breakdowns) — genuine differentiator for a "Gradient Health" panel next to the loss curve.

- **Smith, "Cyclical Learning Rates for Training Neural Networks"** (WACV 2017), https://arxiv.org/abs/1506.01186 — Introduces the **LR range test**: linearly ramp LR over a few epochs/iterations and plot loss vs. LR; the point where loss starts diverging tells you the max usable LR, well before wasting a full training run on a bad LR choice.
  - *Value-add:* directly implements the requested "your LR looks too high based on curve shape" idea, but as a **pre-training diagnostic** rather than only a post-hoc curve-shape heuristic — arguably more useful because it prevents the bad run instead of just diagnosing it after the fact.
  - *Canvas mapping:* an optional **"LR Range Finder" pre-flight node** that runs a short (few dozen iterations) ramp-up mini-run and outputs a suggested LR range directly into the real training node's config — genuinely novel UX for a no-code tool (most no-code platforms don't offer this at all).
  - *Effort:* **S** — it's a standard, well-documented recipe (`fastai`, `pytorch-lr-finder` package already implement it); wrapping an existing OSS implementation is low effort.
  - *Automated curve-shape diagnostics (complementary, no dedicated paper needed):* once streaming curves exist, cheap heuristics can be layered on top post-hoc: oscillating/diverging loss → "LR likely too high"; loss plateauing early with train≪val gap → overfitting; train and val loss both flat and high → underfitting/dead network. This is standard practice, not a research finding — flag as "engineering, not research" so it isn't mistaken for a novel discovery.

---

## 3. Data-centric diagnostics (label noise, data quality, "what to monitor")

- **Swayamdipta, Pavlick, Bhagavatula, Le Bras, Choi, Smith, "Dataset Cartography: Mapping and Diagnosing Datasets with Training Dynamics"** (EMNLP 2020), https://arxiv.org/abs/2009.10795 — Uses only signal already available during any normal training run (per-example confidence and its variability across epochs) to plot every example on a 2D map into "easy-to-learn" / "ambiguous" / "hard-to-learn" regions. Hard-to-learn examples strongly correlate with **mislabeled data**; ambiguous examples are the most valuable for generalization.
  - *Value-add:* **very high** — requires zero extra compute beyond normal training (just needs per-example prediction logging each epoch, which is nearly free), yet gives a qualitatively new diagnostic no DL framework ships out of the box.
  - *Canvas mapping:* a **"Data Map" diagnostic node/panel** that appears after training completes — scatter plot of confidence vs. variability per training example, color-coded by region, clickable to inspect/flag "likely mislabeled" rows for the user to review or remove and re-run.
  - *Effort:* **M** — needs per-epoch, per-example prediction logging (memory/storage cost scales with dataset size × epochs — should sample or only track true-class probability, not full logits) plus a scatter-plot UI. No existing wraps-it-all library; core computation is simple to build from scratch (a few dozen lines), the harder part is UI + storage/streaming design integrated with the existing training-viz pipeline.

- **Pruthi, Liu, Kale, Sundararajan, "Estimating Training Data Influence by Tracing Gradient Descent" (TracIn)** (NeurIPS 2020), https://arxiv.org/abs/2002.08484 — Computes per-example "influence" on a given prediction by tracing how loss on a test point moved during training whenever that specific training example was used; needs only gradients + checkpoints, is architecture-agnostic.
  - *Value-add:* answers "which training examples caused this specific wrong prediction / caused this specific correct prediction," a fundamentally different and more actionable question than dataset cartography's global "is this example noisy" view. Complements it well (cartography = global data health, TracIn = local/case-by-case debugging).
  - *Canvas mapping:* an **"Explain this prediction via influential training examples" node** — user selects a misclassified validation row, node returns top-k most helpful/harmful training rows. Great for a support/debugging workflow in a no-code tool ("why did the model get this wrong?").
  - *Effort:* **L** — requires storing model checkpoints at multiple points during training (storage cost) and computing gradient dot-products per training example at inference time (compute cost scales with dataset size); more implementation-heavy than the other diagnostics. Best treated as a v2/advanced feature, gated behind "large" model runs only.

- **Northcutt, Jiang, Chuang, "Confident Learning: Estimating Uncertainty in Dataset Labels"** (JAIR 2021), https://arxiv.org/abs/1911.00068 — Principled framework (the basis of the popular open-source **`cleanlab`** library) for finding likely-mislabeled examples using only the model's predicted class probabilities and the given labels, without any training-dynamics tracking (works even on a single trained model, not just during training).
  - *Value-add:* simpler and cheaper than dataset cartography (doesn't need per-epoch history, just a single trained model's predictions) — good complementary/cheaper alternative when users don't want per-epoch instrumentation overhead.
  - *Canvas mapping:* a **"Label Quality Report" node** usable for BOTH the classical-ML canvas and the DL canvas (since it only needs final predicted probabilities) — ranks training rows by estimated label-error likelihood.
  - *Effort:* **S** — `cleanlab` is a mature, permissively-licensed OSS library; this is a wrap-an-existing-library job, not a build-from-scratch job. **Cross-cutting note:** because this works off any model's predicted probabilities, it's actually a good candidate to ship for the *classical ML* canvas too, not DL-only — worth flagging to whoever owns the classical-ML roadmap.

- **Toneva, Sordoni, Combes, Trischler, Bengio, Gordon, "An Empirical Study of Example Forgetting during Deep Neural Network Learning"** (ICLR 2019), https://arxiv.org/abs/1812.05159 — Defines "forgetting events" (an example flips from correctly- to incorrectly-classified during training); shows a small subset of examples account for most forgetting events, and these are often safely prunable without hurting accuracy — a training-set-compression angle, not just a diagnostics angle.
  - *Value-add:* niche, but a nice complementary metric on the same per-epoch logging infrastructure that dataset cartography would already require. Not essential for v1.
  - *Canvas mapping:* could be folded into the same "Data Map" panel as an additional overlay/metric ("forgettable" examples) rather than a separate node.
  - *Effort:* **S** if the per-epoch logging infra for cartography already exists (this is just a derived statistic from the same logs) — bundle, don't build separately.

---

## 4. Interpretability for deep tabular models (canvas node output)

- **Sundararajan, Taly, Yan, "Axiomatic Attribution for Deep Networks" (Integrated Gradients)** (ICML 2017), https://arxiv.org/abs/1703.01365 — Model-agnostic, gradient-based attribution method satisfying sensitivity + implementation-invariance axioms that most earlier saliency methods violate; needs only a baseline input and calls to the standard gradient operator (no architecture surgery).
  - *Value-add:* directly portable to any tabular DL node (MLP, ResNet, FT-Transformer) as a per-feature attribution report — this is the natural DL-canvas analogue of SHAP-for-classical-ML that skyulf likely already offers, closing feature parity between the two canvases.
  - *Canvas mapping:* an **"Explain Prediction (Integrated Gradients)" node**, output as a per-feature bar chart per row (or averaged over a batch for global feature importance).
  - *Effort:* **S** — wrap **Captum** (Meta's PyTorch interpretability library, actively maintained, implements Integrated Gradients + many other attribution methods out of the box). Don't build from scratch.

- **TabNet's native attention masks** (Arik & Pfister 2019, §1 above) — no additional library needed; the mask tensors are a normal forward-pass output of the architecture itself.
  - *Canvas mapping:* if TabNet is one of the shipped architectures, expose its per-step attention masks as a free interpretability panel (heatmap of which features were attended to at each of TabNet's sequential decision steps) — zero extra compute, differentiates this specific node type from MLP/ResNet nodes which need the (heavier) Integrated-Gradients wrap instead.
  - *Effort:* **S** — purely a matter of exposing an intermediate tensor that the architecture already produces.

**Recommendation:** ship Integrated Gradients (via Captum) as a universal "Explain" node across all DL architectures, and additionally surface TabNet's free native attention masks when that specific architecture is chosen — cheap way to make TabNet feel like the "interpretable choice" among the DL node options.

---

## 5. Early stopping / resource optimization (ties to Ray effort — noted, not researched)

- **Li, Jamieson, Rostamizadeh, Gonina, Ben-Tzur, Hardt, Recht, Talwalkar, "A System for Massively Parallel Hyperparameter Tuning" (ASHA)** (MLSys 2020), https://arxiv.org/abs/1810.05934 — Asynchronous Successive Halving: aggressively early-stops clearly-underperforming trials so compute concentrates on promising configurations, scales well in distributed/parallel settings.
- **Jaderberg et al., "Population Based Training of Neural Networks"** (2017), https://arxiv.org/abs/1711.09846 — Jointly evolves a population of models AND their hyperparameters during a single training run (discovers a hyperparameter *schedule* rather than a fixed value), rather than treating tuning as a separate outer loop.
  - *Note:* Both **ASHA and PBT are the algorithms literally implemented inside Ray Tune** (Ray's hyperparameter-tuning module). Since the platform has a separate Ray integration effort underway, this is the direct connection point: **if/when the DL module exposes hyperparameter search as a canvas feature, it should be built on Ray Tune's existing ASHA/PBT schedulers rather than reimplementing early-stopping/tuning logic** — flagging this dependency for the Ray-integration workstream rather than doing further Ray-specific research here, per the task scope.

---

## Gaps / not pursued further

- No paper found specifically on "vanishing gradient diagnostics for tabular (non-vision/non-sequence) architectures" — the Deep Information Propagation paper (§2) is architecture-general theory, not tabular-specific; treat its practical takeaway (per-layer gradient-norm health check) as broadly applicable rather than tabular-specialized.
- Did not find recent (2023-2025) papers specifically proposing new *training-visualization* techniques beyond loss-landscape and dataset-cartography-style approaches — the most-cited/novel ideas in this space are the 2017-2021 papers cited above; 2023-2025 tabular-DL papers (TabR, Trompt) focus on architecture, not diagnostics/visualization.
- Curriculum learning proper (as in Bengio et al. 2009's original formulation of ordering training examples by difficulty) was intentionally treated as adjacent to dataset cartography/example-forgetting rather than a separate literature push — recommend skipping a dedicated curriculum-learning node for v1 since its user-facing value is unclear without a much larger research investment; the cartography-derived "hard/easy/ambiguous" labels already give users most of the actionable signal curriculum learning research would motivate.
- TabPFN v2 (2024/2025, Nature-published extension) was referenced by name in the community but not independently fetched/verified in this session — flag for a follow-up search on `arxiv.org` for "TabPFN v2" / Hollmann et al. Nature 2025 if a more capable, regression-supporting successor is relevant to the "instant classifier" node idea in §1.

---

## Top 5 picks (ranked by actionability)

1. **FT-Transformer + ResNet-MLP baseline architectures** (Gorishniy et al. 2021, https://arxiv.org/abs/2106.11959) — Effort: **M**. Directly answers "what architecture should v1 ship" with a peer-reviewed, explicitly-stated recommendation; avoids wasted effort on exotic architectures that don't reliably beat these baselines.
2. **Dataset Cartography-style "Data Map" panel** (Swayamdipta et al. 2020, https://arxiv.org/abs/2009.10795) — Effort: **M**. Near-zero extra training compute, and it's the single most differentiating "beyond the loss curve" visualization feature discovered — nothing in PyTorch Lightning/Keras gives this out of the box, and it also flags likely label errors as a side effect.
3. **Integrated Gradients "Explain Prediction" node via Captum** (Sundararajan et al. 2017, https://arxiv.org/abs/1703.01365) — Effort: **S**. Wraps a mature, maintained library (Captum); gives DL-canvas feature parity with whatever SHAP-based explainability classical-ML nodes already offer.
4. **LR Range Finder pre-flight node** (Smith 2017, https://arxiv.org/abs/1506.01186) — Effort: **S**. Cheap, well-documented recipe with existing OSS implementations; prevents wasted training runs rather than just diagnosing them after the fact — strong no-code UX win.
5. **Confident Learning "Label Quality Report" via `cleanlab`** (Northcutt et al. 2021, https://arxiv.org/abs/1911.00068) — Effort: **S**. Lowest-effort item on this list (wrap existing library), and uniquely reusable across BOTH the classical-ML and DL canvases since it only needs final predicted probabilities — worth flagging cross-team.

