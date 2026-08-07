# Skyulf-Core Examples Redesign — Design

Status: Approved by user (2026-07-22)
Branch: `071`

## Goal

Replace the deleted, overly-simple `skyulf-core/examples/` scripts with a small set of
Jupyter notebooks that (a) onboard a brand-new user quickly, and (b) showcase the library's
real depth — deep feature engineering, leakage-safe pipelines, tuning, ensembling, and full
EDA/profiling capabilities — using real, verified Kaggle competition data. Polars + numpy
only, no pandas, anywhere in the new notebooks.

## Structure

`skyulf-core/examples/`
- `00_quickstart.ipynb` — lightweight onboarding, synthetic data, mirrors the previous
  `01_quickstart.py` scope (load → `SkyulfPipeline` config → fit → predict → save/load).
- `01_house_prices_regression.ipynb`
- `02_disaster_tweets_text_classification.ipynb`
- `03_mall_customers_segmentation.ipynb`
- `04_forest_cover_multiclass_ensemble.ipynb`
- `05_santander_imbalanced_classification.ipynb`
- `06_credit_card_fraud_extreme_imbalance.ipynb`
- `07_spaceship_titanic_classification.ipynb` (rebuild of the already-bundled dataset;
  existing CSVs stay as-is)
- `examples/data/<dataset_name>/` — one folder per dataset, each with its CSV(s) and a
  `SOURCE.md` disclosure file (see Data Sourcing below).

Every deep-dive notebook follows the same section skeleton so the set reads as one coherent
showcase, not 7 unrelated scripts:

1. **Problem framing** — what the competition asks, what the notebook demonstrates.
2. **Load data** — `pl.read_csv`, schema check.
3. **EDA (full library usage)** — see EDA section below; every notebook uses
   `EDAAnalyzer`/`EDAVisualizer` plus at least one of `DriftCalculator` or `expect_*` where it
   naturally fits the dataset's story (e.g. drift between train/test distributions, or
   `expect_value_range` as a guard before modeling).
4. **Leakage-safe `SkyulfPipeline` build** — preprocessing steps chosen for the dataset's
   real issues (missingness, skew, high-cardinality categoricals, text, geo, imbalance),
   `TrainTestSplitter` always first.
5. **Model selection & tuning** — `TuningCalculator` (grid/random/optuna) where it adds
   value; ensembling (`VotingClassifier`/`StackingClassifier`) for the harder datasets
   (Forest Cover, Santander, Credit Card Fraud).
6. **Evaluation** — task-appropriate metrics via `skyulf.modeling._evaluation`, compared
   against known public leaderboard/benchmark ranges for that competition (stated as
   "typical range," not a claim of leaderboard rank).
7. **Key takeaways / what to try next** — short, honest, points at real levers (more
   tuning, more features, more data).

## EDA depth (explicit ask: "do properly to get all from it")

Every deep-dive notebook must exercise the **full** `EDAAnalyzer.analyze()` surface, not a
subset, and render it with `EDAVisualizer`:
- `EDAAnalyzer(df).analyze(target_col=..., date_col=..., lat_col=..., lon_col=...)` called
  with every argument that applies to that dataset (e.g. `lat_col`/`lon_col` isn't
  applicable to any of these 7, `date_col` doesn't apply either — noted honestly in-notebook
  rather than faked).
- `EDAVisualizer(profile, df).summary()` (Rich dashboard) **and** `.plot()` (Matplotlib)
  both called and their output kept in the notebook (so it doubles as a rendered reference).
- Notebook explicitly reads and narrates: alerts (`profile.alerts`), recommendations
  (`profile.recommendations`), VIF/multicollinearity, target correlations/interactions
  (leakage-alert threshold explained), outlier analysis, PCA/clustering, and the decision-tree
  rule/feature-importance surrogate (`profile.rule_tree`) — this is the fastest way to show a
  newcomer *why* a feature matters before they hand-roll their own feature importance code.
- Where relevant to the dataset's narrative:
  - Santander/Credit Card Fraud: `DriftCalculator` comparing train vs. held-out slice, to
    show it's available for train/serving-skew monitoring.
  - House Prices/Forest Cover: `expect_value_range`/`expect_no_nulls` as a lightweight
    pre-modeling data-quality gate.
- No pandas fallback — `EDAAnalyzer`/`profiling` module is already polars-native internally,
  confirmed via exploration.

## Datasets & bundling plan (confirmed with user: "mixed" strategy)

| # | Folder | Case | Sourcing | Bundle plan |
|---|--------|------|----------|-------------|
| 1 | `spaceship_titanic` | Binary classification | Already bundled (real, full) | unchanged, 1.1MB |
| 2 | `house_prices` | Regression | GitHub mirror `SrikanthVelpuri/House-Prices-Advanced-Regression-Techniques` (verified: real schema, Id...SalePrice) | full, real, ~0.9MB |
| 3 | `disaster_tweets` | Text/NLP classification | GitHub mirror `tarunannapareddy/Natural-Language-Processing-with-Disaster-Tweets` (verified ~7613 real rows) | full, real, ~1.4MB |
| 4 | `mall_customers` | Clustering/segmentation | GitHub mirror `DACUS1995/BIRCH-Mall-Customers-clustering` (verified 200 rows, exact schema) | full, real, <0.1MB |
| 5 | `forest_cover` | Multiclass classification + ensembling | UCI direct `archive.ics.uci.edu/static/public/31/covertype.zip` (verified 581,012 rows real) | full zip bundled (11.2MB), unzipped at notebook runtime, not committed as raw CSV |
| 6 | `santander` | Imbalanced classification, wide (371 cols) | GitHub mirror `poindextrose/Kaggle-Santander-Customer-Satisfaction` fetched via git blob API, verified authentic (76,020 rows, 3.96% positive) | **stratified subsample**, 15,000 rows, true 3.96% ratio preserved exactly (594 positive of real pool + 14,406 negative, seed=42), ~14.9MB |
| 7 | `credit_card_fraud` | Extreme-imbalance classification (resampling techniques) | GitHub mirror `nsethi31/Kaggle-Data-Credit-Card-Fraud-Detection`, verified authentic full file (284,807 rows, 492 frauds = 0.1727%, exact match to public stats) | **stratified subsample**: all 492 real frauds + 14,500 random non-fraud (seed=42) → ~15,000 rows, enriched to ~3.28% fraud rate, ~5.4MB |

Every subsampled/zipped dataset ships a `SOURCE.md` in its data folder (same pattern as the
existing Spaceship Titanic one) disclosing: original source URL, real full-dataset row
count/statistics, exact subsampling method and seed, and why (repo size). This keeps the
"real Kaggle data" claim honest even where a subsample is used.

## Testing / verification plan

- Every notebook executed end-to-end via `jupyter nbconvert --to notebook --execute
  --inplace`, exit code 0.
- Static checks: no `import pandas` anywhere in `examples/*.ipynb` source cells; `ruff
  check`/`ruff format` over any `.py` helper modules if extracted; `ty check` unaffected
  (notebooks aren't type-checked, but any shared helper `.py` will be).
- Each notebook's final metrics sanity-checked against known public ranges for that
  competition (not required to match exactly, just plausible).
- Confirm `TrainTestSplitter` is always the first preprocessing step (leakage-safe by
  construction, backed by this session's earlier library fix).
- README rewritten to list all 8 notebooks, the dataset table above, and the "why polars,
  why no pandas" framing already established.

## Out of scope

- PTCG AI Battle Challenge / ARC-AGI-3 (not tabular ML, rejected earlier this session).
- Restoring `docs_internal/multi_output_audit.md` or old `.gitignore` exception lines
  automatically — handled explicitly during implementation, not assumed.
- Changing skyulf-core's public API (that work is tracked separately, see the companion
  usability-improvement plan in `temp/skyulf-core-usability-plan.md`, which is planning-only
  and not part of this implementation).
