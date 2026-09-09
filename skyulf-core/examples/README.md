# Skyulf Core — Examples

The nine original Jupyter notebooks show `skyulf-core` on real Kaggle
datasets — regression, text classification, clustering, multiclass +
ensembling, imbalanced classification, extreme-imbalance classification, and
binary classification. Those notebooks are **Polars + NumPy only** (no
pandas anywhere in the pipeline), uses `skyulf`'s own `EDAAnalyzer` /
`EDAVisualizer` for a full exploratory pass before any modeling, builds a
**leakage-safe** preprocessing pipeline (`TrainTestSplitter` is always the
first step — nothing that learns from data runs before the split), and
compares at least two models honestly (including a tuned or ensembled one).

## Running the notebooks

```bash
cd skyulf-core
pip install -e ".[dev,viz,eda,tuning,preprocessing-imbalanced,modeling-xgboost,modeling-lightgbm,explainability]"
python -m pip install jupyter
jupyter nbconvert --to notebook --execute --inplace examples/<name>.ipynb
```

Each notebook is self-contained and reads its dataset from `examples/data/`
(bundled in this repo — see the table below and each `SOURCE.md` /
`README.md` for provenance). No downloads or Kaggle API keys required.

## Standalone leakage-safety notebook and script

[`09_leakage_safety.ipynb`](09_leakage_safety.ipynb) is the interactive,
**pandas-based** walkthrough. Run its cells in order to build a real pipeline
with FeatureGeneration, explicit column dropping, train/test splitting, X/y
separation, encoding, scaling, and a regression model. Inspect preprocessing
tables, predictions, leakage modes, and numerical train-only scaling checks.
It also shows a fresh core pipeline per CV fold and explains why the backend's
`unsupported_graph` fallback differs from ordinary leakage-gate acceptance.

The notebook uses synthetic in-memory data and needs no backend or downloads.
The simpler [`09_leakage_safety.py`](09_leakage_safety.py) remains available for
terminal use without Jupyter. Select a notebook kernel with the core library
installed, or run the script:

From the repository root:

```bash
uv pip install -e ./skyulf-core
python skyulf-core/examples/09_leakage_safety.py
```

The script's numbered sections demonstrate:

1. Default `raise`: reject scaler-before-split before any training, including
   calls through `get_fitted_split()`.
2. `warn`: return diagnostic messages or log them during `fit()`, then continue
   despite the unsafe order.
3. `ignore`: continue without leakage warnings; this does not fix leakage or
   suppress unrelated errors.
4. The recommended `Split -> StandardScaler -> Model` order, prediction, and
   extraction of already-preprocessed train/test frames.
5. Why stateless row rules and constant imputation are allowed before splitting,
   while mean imputation is not.
6. An external `SplitDataset`, with numerical assertions showing that both
   partitions are transformed using only the training mean and standard deviation.
7. Why a config with no splitter returns an advisory even under `raise`.

`leaking_pipeline`, `warning_pipeline`, and `pipeline` are ordinary Python
variables holding `SkyulfPipeline` instances, not separate classes or modes.
The mode is selected by the `on_leakage` argument. Keep its default unless you
deliberately accept an unsafe fit; choose correct ordering rather than hiding
the warning. The script catches the expected rejections so later sections can
still run, and assertions describe the expected results.

The shared JSON contract is in
[`../tests/test_cases/leakage/registry_nodes.json`](../tests/test_cases/leakage/registry_nodes.json).
The core and backend test matrices cover `raise`, `warn`, and `ignore`, safe and
unsafe placements, and parameter-dependent exceptions for all registered nodes.

## Original notebooks (00-08)

| # | Notebook | Task | Dataset | Rows | Highlights |
|---|----------|------|---------|------|------------|
| 00 | `00_quickstart.ipynb` | — | synthetic | — | Full pipeline lifecycle: fit → save → load → predict, no-pandas Arrow/NumPy bridge, geo features (`GeoDistance` haversine + `H3Index` hex grid) |
| 01 | `01_house_prices_regression.ipynb` | Regression | [House Prices](https://www.kaggle.com/c/house-prices-advanced-regression-techniques) | 1,460 | Deep EDA (VIF, target correlations, rule-tree importances), NA-semantic recoding, outlier handling comparison (Winsorize vs. IQR row-removal vs. none), log1p target, tuned Ridge vs. Optuna-tuned GradientBoosting, SHAP explainability, Kaggle submission generation |
| 02 | `02_disaster_tweets_text_classification.ipynb` | Text classification | [NLP with Disaster Tweets](https://www.kaggle.com/c/nlp-getting-started) | 7,613 | `TextMixin` EDA stats, TF-IDF + hash-encoded categorical, Naive Bayes vs. tuned Logistic Regression vs. stacking ensemble, char n-gram typo-robustness experiment, `sentence_embedder` (graceful optional-dependency handling), Kaggle submission generation |
| 03 | `03_mall_customers_segmentation.ipynb` | Clustering | [Mall Customer Segmentation](https://www.kaggle.com/datasets/vjchoudhary7/customer-segmentation-tutorial-in-python) | 200 | Silhouette-based k selection, KMeans vs. GaussianMixture vs. BIRCH, auto-generated cluster profiles |
| 04 | `04_forest_cover_multiclass_ensemble.ipynb` | Multiclass classification | [Forest Cover Type](https://archive.ics.uci.edu/dataset/31/covertype) | 581,012 (bundled as zip; 100k stratified training sample) | Runtime unzip/decompress, RandomForest vs. voting ensemble (RF + ExtraTrees + HistGradientBoosting) |
| 05 | `05_santander_imbalanced_classification.ipynb` | Imbalanced classification | [Santander Customer Satisfaction](https://www.kaggle.com/c/santander-customer-satisfaction) | 15,000 (stratified subsample, true 3.96% positive rate) | Wide-frame (369-col) EDA, `DriftCalculator`, VarianceThreshold + CorrelationThreshold, feature-selection strategy comparison (Univariate ANOVA vs. Model-Based RF importance), SMOTE, ROC-AUC-first evaluation |
| 06 | `06_credit_card_fraud_extreme_imbalance.ipynb` | Extreme-imbalance classification | [Credit Card Fraud Detection](https://www.kaggle.com/mlg-ulb/creditcardfraud) | 14,992 (stratified subsample, enriched ~3.28% vs. real 0.173%) | Class-weighting vs. SMOTE vs. random undersampling, PR-AUC-first evaluation |
| 07 | `07_spaceship_titanic_classification.ipynb` | Binary classification | [Spaceship Titanic](https://www.kaggle.com/c/spaceship-titanic) | 8,693 | Structured-string parsing (`PassengerId`/`Cabin`), domain-knowledge consistency features, feature generation (`FeatureInteraction` + `PolynomialFeatures`), Logistic Regression vs. Random-Search-tuned RF vs. Grid-Search-tuned RF vs. voting vs. stacking ensemble, Kaggle submission generation |
| 08 | `08_online_retail_customer_segmentation.ipynb` | Clustering (RFM segmentation) | [Online Retail](https://archive.ics.uci.edu/dataset/352/online+retail) | 153,150 transactions / 1,800 customers (stratified-by-customer subsample) | Raw-transaction-to-RFM feature engineering, log1p + scaling, KMeans vs. MiniBatchKMeans vs. GaussianMixture vs. BIRCH, business-named segments (Champions/Hibernating/etc.) from per-cluster medians, bonus time-series feature engineering (`DateFeatures`/`LagFeatures`/`RollingAggregate`) |

## Design principles followed in notebooks 00-08

- **No pandas.** Data loading, feature engineering, and inspection all use
  Polars (+ NumPy where needed); `skyulf`'s own dual-engine pipeline nodes
  handle any pandas bridging internally.
- **Split first.** `TrainTestSplitter` (optionally stratified) is always the
  first preprocessing step. Anything that *learns* a statistic from data —
  imputation values, scaler mean/std, encoder categories, TF-IDF vocabulary,
  correlation-threshold drops, SMOTE/undersampling — happens strictly
  *after* the split, inside the pipeline config, so it only ever sees the
  training fold. Deterministic domain-knowledge transforms (NA-semantic
  recoding, ratio/date features, structured-string parsing, log1p target
  transforms) are safe to do *before* the split since they don't learn
  anything from the data.
- **Full EDA, not a token gesture.** Every notebook runs `EDAAnalyzer` /
  `EDAVisualizer` and actually reads the alerts, recommendations, and (where
  relevant) target-correlation / VIF / clustering / outlier output before
  deciding on feature engineering — not just a `.head()` and a shrug.
- **Honest evaluation.** Model comparisons report metrics appropriate to the
  problem (ROC-AUC/PR-AUC/precision/recall for imbalanced tasks, not bare
  accuracy) and disclose dataset-sampling caveats explicitly where a bundled
  sample's class ratio was enriched from the true rate (05, 06).

## Dataset provenance

Every dataset under `examples/data/<name>/` ships its own `SOURCE.md` (or
`README.md` for Spaceship Titanic) documenting exactly where the data came
from, how it was verified, and — for the two subsampled datasets (Santander,
Credit Card Fraud) — precisely how the stratified sample was drawn and how
its class ratio compares to the real, full-scale dataset.

### Leakage execution and node audit

Example 09 also demonstrates native core tuning with per-fold refitting. Its
core/backend diagrams are embedded SVG attachments for offline notebook viewing.
See the [illustrated core/backend guide](../../docs/user_guide/leakage_core_backend.md)
and the [complete preprocessing audit](../../docs/user_guide/preprocessing_leakage_audit.md).

The [preprocessing placement reference](../../docs/user_guide/preprocessing_placement.md)
lists all node registrations and their settings. The same guide is searchable
in the canvas under **How pipelines work > Preprocessing & Leakage**.
