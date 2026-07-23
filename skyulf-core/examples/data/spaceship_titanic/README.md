# Spaceship Titanic dataset

Source: Kaggle "Spaceship Titanic" competition
(https://www.kaggle.com/competitions/spaceship-titanic). Bundled here so the
`08_kaggle_spaceship_titanic` example is fully reproducible offline, without
requiring Kaggle API credentials.

- `train.csv` — 8693 rows, labeled (`Transported` target column).
- `test.csv` — 4277 rows, unlabeled (Kaggle's actual holdout for the
  competition leaderboard — we don't have its labels, so the example
  reports out-of-fold local CV performance rather than a real leaderboard
  score).

Provided by Kaggle for the associated public competition; used here for
educational/demonstration purposes only.
