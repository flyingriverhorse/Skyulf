# Credit Card Fraud Detection

- **Original dataset**: https://www.kaggle.com/datasets/mlg-ulb/creditcardfraud
- **Source used**: `github.com/nsethi31/Kaggle-Data-Credit-Card-Fraud-Detection` (fetched
  via the GitHub git Blobs API; full file is 102MB). Verified authentic: full file is
  284,807 rows, 492 frauds (`Class=1`), fraud rate 0.1727% — an exact match to the
  well-known public statistics for this dataset (PCA-anonymized `V1`-`V28` features, `Time`,
  `Amount`, `Class`).
- **Bundled file**: `creditcard_sample.csv` — a **stratified subsample**, NOT the full file.
  - Method: **all 492 real fraud rows** (kept in full — there are too few to subsample
    further without losing signal) + 14,500 randomly sampled non-fraud rows (`seed=42`),
    concatenated and shuffled.
  - Result: 14,992 rows, fraud rate **~3.28%** — deliberately enriched from the real
    0.1727% to keep the file small (~5.4MB vs. 102MB) while still leaving enough positive
    examples to train/evaluate on. This means the *raw* imbalance ratio in this file is
    NOT representative of the real competition — the notebook must state this explicitly
    and discuss the real 0.17% rate and its implications (why resampling/class-weighting
    techniques are needed) as text, not just rely on the bundled ratio.
  - Reason: full file (102MB) was judged too large to bundle raw.
- **This is a real-data subsample, not synthetic data** — every row is an authentic row from
  the original dataset.
