# Measure drift and feature data quality

## When you need it

Use drift to compare production inputs with the verified population used to train a concrete model
version. It detects changed distributions, missing declared features and incompatible types without
requiring outcome labels. [Performance loss](databricks_performance.md) instead compares labeled
prediction quality with a baseline. Both can independently qualify for
[guarded retraining](databricks_retraining.md).

## What to edit

Set `monitoring_drift_thresholds` under the producer target's variables in `deployment/targets.yml`.
These are the current defaults:

```yaml
monitoring_drift_thresholds: '{"psi": 0.2, "ks_statistic": 0.1, "wasserstein": 0.1, "kl_divergence": 0.1}'
```

`monitoring_drift_thresholds` is authoritative for this project's enrollment;
`{}` uses Core defaults and partial objects override named metrics. Defaults are
PSI 0.2, KS statistic 0.1, normalized Wasserstein 0.1 and KL divergence 0.1.
A value must strictly exceed its own threshold. The displayed KS p-value is
diagnostic relative to these effect limits; statistical support is checked separately.

| Check | Comparison and limit |
| --- | --- |
| Numeric PSI | Compare distributions using reference-derived bins; use `psi` (0.2). Low-cardinality numeric references use midpoint bins; otherwise reference percentiles supply the bins. |
| Categorical PSI | Compare category frequencies; use the same `psi` limit. |
| KS | Maximum empirical cumulative-distribution gap; compare **KS statistic** with `ks_statistic` (0.1). The diagnostic KS p-value is never compared with this effect-size limit. |
| Wasserstein | Empirical transport distance divided by the reference population standard deviation; with zero standard deviation use the unnormalized distance. Compare with `wasserstein` (0.1). |
| KL divergence | Compare current and reference numeric-bin probabilities with `kl_divergence` (0.1). |
| Missing schema | A declared feature absent from either input records `schema_missing`; unrelated extra columns are not new declared features. |
| Type drift | Incompatible feature types record `type_drift`. Supported numeric/string and boolean representations are normalized where valid; this is not a blanket raw-dtype inequality check. |
| Missing fraction | Nulls and numeric NaNs are missing. Compare with the reference rate, using the quality rule below. |
| Nonfinite fraction | Numeric NaN and infinity contribute to the fraction; infinity is a quality issue. |

Distribution drift requires both an effect above its own limit and supported sample-aware
statistical evidence. Evidence uses Bonferroni correction across feature tests in one report at
significance 0.05; repeated windows are not covered by that correction. `not_detected` is not proof
of equal distributions or adequate test power. Schema/type violations are recorded independently.

The missing-rate reference value is not a configurable alert threshold. An increase from a zero
reference rate is an issue; otherwise an increased rate uses a one-sided Fisher test at 0.01.
Missing input rows without usable saved predictions can also be quality issues. Do not interpret a
displayed reference missing percentage as an allowed limit.

### Choose the reference population

The monitor replays original training membership from saved source and split evidence, prepares
immutable Spark reference tables, and compares current features from the source snapshots recorded
with saved predictions. It does not pick an arbitrary latest table or retrain during measurement.
See [reference preparation](databricks_monitoring_enrollment.md#the-command) and
[store reference tables](databricks_monitoring_setup.md#what-to-edit).

## The command

After changing thresholds, validate and deploy the producer, then run activation/scoring
registration to update its enrollment. Run a fresh observation from the generated project root:

```powershell
databricks bundle validate --strict -t test --profile YOUR_PROFILE
databricks bundle deploy -t test --profile YOUR_PROFILE
databricks bundle run monitoring -t test --profile YOUR_PROFILE
```

Open `monitoring_report` in the run for the saved feature metrics and notes, then **Drift and data
quality** in the shared dashboard. Dashboard refresh alone does not calculate a new drift report.

## What you should see

| Status | Meaning |
| --- | --- |
| healthy | Measured drift/quality checks passed; model accuracy is not an acceptance gate. |
| drift | Distribution or declared feature-schema checks detected drift. |
| stale | The latest scoring commit (batch) or captured request (online) is older than expected_interval_hours, regardless of the last monitoring run. |
| never_observed | No observation exists for the current enrollment configuration. |
| degraded | Quality/measurement issues, no usable predicted outputs, or a nonfinite calculated metric. |
| failed | Monitoring raised a saved error. |
| no_data | The selected reference or current population is empty. |
| disabled | Enrollment explicitly paused. |
| unavailable | A result exists but the scoring or captured-request time is unavailable. |

Counts represent enrollments, not prediction rows. measurement_status retains the
underlying measured verdict when freshness changes. No retraining, performance
acceptance thresholds or notifications are enabled by the dashboard.

Measured status precedence is `no_data` for an empty reference/current population, then `drift` for
a detected feature change, then `degraded` for incomplete checks, quality problems or invalid
measured metrics, otherwise `healthy`. A quality issue does not erase measured drift. Health
freshness is based on scoring commit or captured request time, not the time monitoring reran.

### Read the drift charts

The drift/quality page keeps two history tables and separates charts by model/version/context. Each
identity shows its latest observation's top ten feature PSI values beside their own limits, plus
feature counts with drift or unavailable checks over time. Each feature is counted once within its
observation even if multiple checks flag it. The count axis uses whole numbers; compact UTC
date/time labels keep the chart readable. Hover to see the full window and calculation timestamp.
For example, data from 09:00 to 10:00 appears at 10:00 even if the checks ran at 11:30. Drift and
Performance date filters use the same window end as their time charts. Calculation time remains
available separately in the details. The matching-identity table shows which contexts the shared
filters include. The full model/table inventory remains on Overview.

KS p-value is not an effect-size threshold. Unavailable checks are shown separately,
and missing values never become zero. PSI is only one drift check; the counts
and table include other distribution and schema checks too.

Drift columns show each
metric's value / its own limit. KS p-value is diagnostic only; it is never divided
by the KS statistic's threshold. Missing features and changed types stay visible.
Quality rows identify model version, scoring time and feature, with separate
missing/nonfinite percentages. Both tables retain observation history.

Filters never select an older report; see [latest observations and page
filters](databricks_monitoring_dashboard.md#latest-observations-and-page-filters).
The PSI bars retain at most ten features per identity before feature filtering;
use the history tables for the rest.

The PSI matrix includes numeric and categorical features. Each cell divides a
recorded PSI by its own recorded limit; values above 1 receive a fixed highlight.
An absent or nonpositive limit leaves the cell blank. Each identity retains its
latest 12 full observations and the 10 features with the largest ratios within
that history. Feature and date filters narrow this retained set; they do not
retrieve older observations. Observation labels include the year and a sequence
number, so separate reports for the same window remain separate cells.

The missing-value matrix uses the same observation and feature bounds, ranking
features by their largest missing fraction. Its reference rate is the reference
dataset's missing fraction, **not an alert limit**. A recorded quality issue can
depend on a statistical test or unscored inputs. The tooltip identifies that saved
decision. Nonfinite fractions remain available in the quality table.

Latest PSI bars use the newest saved report for each context, as described in
[latest observations](databricks_monitoring_dashboard.md#latest-observations-and-page-filters).
The drift context summary exposes the newest report status even when charts are empty.
Mature performance-window-only reports stay in policy history; they do not replace
the full feature observation, current model health or confusion matrix.

## Common errors and fixes

| What you see | What to check |
| --- | --- |
| `no_data` | Confirm the pinned reference and selected current window contain rows. Empty data is not proof of no drift. |
| `degraded` or unavailable checks | Read the per-feature evidence and quality notes. Unsupported types, too little statistical evidence, unscored inputs and invalid values must be resolved before treating the population as measured. |
| Large PSI but no supported drift | Inspect effect and statistical evidence separately; meeting one gate does not establish the other. |
| Drift but no training request | Opt into `on_drift: retrain` only after reviewing the [shared request guards](databricks_retraining.md). Quality/schema problems, incomplete checks or stale/freshness guards can block a request. |

### Spark measurement limits

Spark computes regression and confusion-matrix metrics from full-population
aggregates, and probability AUC/AP from exact tied-score counts. Numeric drift
uses exact empirical-CDF KS/Wasserstein effects and reference percentile bins;
rows are not sampled. Ordered CDF and probability curves use an ordered Spark
window over distinct values, which can concentrate work on one executor for
high-cardinality features. Removing the local row cap does not remove compute,
shuffle or executor-memory requirements.

Statistical evidence is bounded separately: KS uses the exact lattice calculation
when the two population sizes multiply to at most 1,000,000, the Core asymptotic
method when either size exceeds 10,000, and explicitly named
`ks_dkw_union_bound` evidence in the remaining range. An inconclusive bound is
unavailable rather than evidence of no drift. Categorical summaries allow at
most 1,024 current categories and use Core's 50-category reference limit;
classification allows at most 256 saved classes. Temporal/nested feature drift
and unsupported cardinalities are explicit failures or unavailable checks.
These limits bound summaries and supported algorithms, not observation rows.

Temporal or nested feature checks and unsupported cardinalities stay explicit; missing values never
become a zero drift score. Numeric infinities or invalid numeric conversions can make the
distribution comparison unavailable while quality evidence remains visible.
