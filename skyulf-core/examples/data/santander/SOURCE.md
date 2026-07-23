# Santander Customer Satisfaction

- **Original competition**: https://www.kaggle.com/competitions/santander-customer-satisfaction
- **Source used**: `github.com/poindextrose/Kaggle-Santander-Customer-Satisfaction`
  (fetched via the GitHub git Blobs API since the real train.csv is 59MB, over the Contents
  API's 1MB inline limit). Verified authentic: full file is 76,020 rows x 371 columns,
  TARGET positive rate 3.96% (2,952/76,020 approx, matching well-known public statistics for
  this competition).
- **Bundled file**: `train_sample.csv` — a **stratified subsample**, NOT the full file.
  - Method: 594 positive (`TARGET=1`) rows + 14,406 negative rows, both randomly sampled
    (`seed=42`) from the full verified 76,020-row file, concatenated and shuffled.
  - Result: exactly 15,000 rows, true 3.96% positive rate preserved (matches the full
    dataset's real imbalance ratio, not an enriched/synthetic ratio).
  - Reason: the full 59MB train.csv (+ a similarly sized test.csv) was judged too large to
    bundle raw in this repo; this subsample keeps the same statistical character
    (high-dimensional, ~4% imbalance) at ~14.9MB.
- **This is a real-data subsample, not synthetic data** — every row is an authentic row from
  the original competition file.
