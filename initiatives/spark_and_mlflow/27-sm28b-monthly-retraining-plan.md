# SM-28b: Optional monthly local retraining

## Goal and boundary

Keep the generated Bundle at two jobs (`train`, `score`) and one prediction
table. A new project has no active schedule. Users may opt into a **paused**
monthly schedule on the existing `train` job and unpause it after supplying a
real labeled source, pipeline, experiment and permissions. The schedule never
runs `score`, moves a registry alias, or changes its pinned `model_version`.

## Monthly training run

1. At run start, normalize the current instant to UTC. The cutoff is midnight
   on the first day of that month. The holdout covers the preceding calendar
   month; fit starts a configurable number of calendar months before cutoff.
   A run on any day of the month therefore has the same temporal boundaries.
2. Validate the source as Delta and read its latest concrete version once.
   Pass that version into the existing bounded `LocalTrainingSpec`; the run's
   `dataset_id` records the exact version and window. Filter rows by event time
   and label availability at cutoff. The version is pinned at run start, not
   time-traveled to the cutoff: honest `label_at` values are required.
3. Resolve `@champion` once, if present, and pass its concrete version to the
   existing `train_local_candidate` service. A missing alias means first-model
   training; any other registry failure aborts. The service records metrics and
   a comparison, then registers a candidate without moving aliases.
4. Activation remains explicit: review the comparison, manage aliases through
   the existing lifecycle API, update `model_version` in the Bundle config and
   redeploy before `score` uses the new version. Existing predictions remain.

## Safety and acceptance

- The opt-in train schedule is paused and uses one monthly Quartz time in UTC;
  `max_concurrent_runs: 1` also applies to train. Initial Bundle output still
  contains exactly two jobs and creates no Unity Catalog objects at deploy.
- Tests cover month/year rollover, timezone normalization, invalid lookback,
  Delta snapshot pinning, missing champion, and no implicit promotion/scoring.
- Validate a generated serverless project with Databricks CLI. Live execution
  needs a real labeled source with enough rows in both temporal partitions;
  a static historical test table cannot establish a current-month success.
- Retrying a month may register another candidate if the source version or
  configuration changed. That is visible in MLflow and does not activate it.
  Shared-target scoring writers still need a separate admission decision;
  this task introduces no second score writer.
