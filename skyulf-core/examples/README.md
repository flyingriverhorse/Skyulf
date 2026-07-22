# Skyulf Core — Examples

Eight end-to-end Jupyter notebooks showing `skyulf-core` on real Kaggle
datasets — regression, text classification, clustering, multiclass +
ensembling, imbalanced classification, extreme-imbalance classification, and
binary classification. Every notebook is **Polars + NumPy only** (no
pandas anywhere in the pipeline), uses `skyulf`'s own `EDAAnalyzer` /
`EDAVisualizer` for a full exploratory pass before any modeling, builds a
**leakage-safe** preprocessing pipeline (`TrainTestSplitter` is always the
first step — nothing that learns from data runs before the split), and
compares at least two models honestly (including a tuned or ensembled one).

## Running the notebooks

```bash
cd skyulf-core
pip install -e ".[dev]"          # or your usual editable install
jupyter nbconvert --to notebook --execute --inplace examples/<name>.ipynb
# or just open them normally:
jupyter lab examples/
```

Each notebook is self-contained and reads its dataset from `examples/data/`
(bundled in this repo — see the table below and each `SOURCE.md` /
`README.md` for provenance). No downloads or Kaggle API keys required.

## Notebooks

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

## Design principles followed in every notebook

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
