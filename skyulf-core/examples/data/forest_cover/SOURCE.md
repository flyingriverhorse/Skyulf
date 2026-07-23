# Forest Cover Type

- **Original competition/dataset**: https://www.kaggle.com/competitions/forest-cover-type-prediction
  (same underlying data as the UCI Covertype dataset)
- **Source used**: UCI Machine Learning Repository direct download —
  https://archive.ics.uci.edu/static/public/31/covertype.zip (reachable directly in this
  environment; Kaggle's own site and figshare, which `sklearn.datasets.fetch_covtype()`
  relies on, are not).
- **Bundled file**: `covtype.zip` (11.2MB compressed) containing `covtype.data.gz` +
  `covtype.info` (column/label documentation). The notebook unzips/decompresses this at
  runtime rather than committing the ~75MB extracted CSV.
- **Full dataset, no subsampling**: 581,012 rows, 54 features, 7 cover-type classes —
  matches the real, complete dataset exactly.
