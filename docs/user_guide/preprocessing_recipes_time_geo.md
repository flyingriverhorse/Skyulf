# Time and geography

[Back to the preprocessing catalog](preprocessing_recipes.md)

These steps add calendar features, lags and rolling aggregates, distances and H3
cells. Lags and rolling aggregates depend on ordered history; read the history
section before scoring them.

YAML blocks are list fragments to paste under a selected recipe unless a card
marks them as standalone Core configuration. Shared column, row and runtime rules
are in [read the step cards](preprocessing_recipes.md#read-the-step-cards).

## DateFeatures

**Purpose and effect:** a timestamp adds calendar components such as `event_time_year=2026` and
`event_time_month=10`.

**File:** `config/preprocessing.yml`.

| Parameter | Type | Default | Allowed values | What it changes |
| --- | --- | --- | --- | --- |
| `columns` | list[string] | [] | Existing column names. | Select explicitly; no automatic detection. |
| `features` | list[string] | [year, month, day, dayofweek] | year, month, day, dayofweek, dayofyear, quarter, weekofyear, hour, minute, second, is_weekend, is_month_start, is_month_end; hour_sin/hour_cos, dayofweek_sin/dayofweek_cos, month_sin/month_cos, dayofyear_sin/dayofyear_cos. | Choose calendar parts or cycles; null/empty uses default, unknown names fail during fit with supported choices listed. |
| `drop_original` | boolean | false | true, false | Remove source date columns. |
| `epoch_unit` | string/null | null | s, ms, us, ns; required for numeric epochs. | Interpret numeric timestamps. |
| `timezone` | string | UTC | IANA name, such as Europe/Istanbul or Europe/Copenhagen. | Extract parts and cycles in the selected local calendar. |

**Minimal YAML:**

```yaml
- name: datefeatures
  transformer: DateFeatures
  params: {columns: [observation_time], features: [year, month, dayofweek], drop_original: true}
```

**Common mistake:** Numeric epochs require epoch_unit; new artifacts normalize timestamp
interpretation to UTC before converting to the selected timezone. A naive local
clock string must carry its offset if it represents a non-UTC instant.

**Details:** Newly fitted artifacts interpret naive dates and numeric epochs as
UTC, then convert each instant to `timezone`; offset-aware dates describe the
same instant regardless of their written offset. DST jumps and repeated hours
follow that conversion without localizing an ambiguous wall clock. The fitted
state stores the timezone and calendar selections. Adds `<column>_<feature>` values, with null features
for unparseable timestamps; dtype representation of nullable calendar values
can differ across engines. dayofweek is Monday=0. Empty apply retains generated
columns. Context row. Base dependencies. Legacy artifacts retain their old
timestamp interpretation when their saved state lacks a timezone, until the
whole pipeline is refitted. Existing artifacts with `timezone: UTC` retain UTC.

The explicit `_sin` and `_cos` options emit Float64 values using
`sin(2*pi*position/period)` and `cos(2*pi*position/period)`. Hour uses integer
hours 0–23 and period 24; weekday uses Monday=0 and period 7. Month subtracts
one (January=0) and uses period 12. Day-of-year subtracts one (January 1=0) and
uses 365 or 366 according to each row's local calendar year. These are calendar
components, so minutes do not contribute a fractional hour. Select both members
of a pair to preserve circular position; each output can also be selected alone.

```yaml
- name: local_calendar
  transformer: DateFeatures
  params:
    columns: [observation_time]
    timezone: Europe/Copenhagen
    features: [second, hour_sin, hour_cos, dayofweek_sin, dayofweek_cos]
    drop_original: true
```

## LagFeatures

**Purpose and effect:** ordered values `[10, 20, 30]` and lag 1 add `[null, 10, 20]`.

**File:** `config/preprocessing.yml`.

| Parameter | Type | Default | Allowed values | What it changes |
| --- | --- | --- | --- | --- |
| `columns` | list[string] | [] | Existing column names. | Select explicitly; no automatic detection. |
| `lags` | integer/list[integer] | [1] | Positive; normalized to sorted distinct integers. | Choose previous-row offsets. |
| `drop_na` | boolean | false | true, false; carry rejects true. | Remove rows with missing lag output. |
| `group_by` | string/list[string]/null | null | Existing grouping columns; use a list in carry mode. | Maintain separate entity sequences. |
| `sort_by` | string/null | null | Existing numeric/typed datetime clock for carry. | Order observations; absent uses batch input order. |
| `history_mode` | string | batch | batch, carry | Use the current batch or a bounded saved training tail. |
| `history_max_rows` | integer | 10000 | Positive; carry only. | Bound stored history rows. |
| `history_max_bytes` | integer | 1048576 | Positive; carry only. | Bound stored history bytes. |

**Minimal YAML:**

```yaml
- name: lagfeatures
  transformer: LagFeatures
  params: {columns: [income], lags: [1], group_by: [entity], sort_by: observation_time}
```

**Common mistake:** Sorting can reorder rows; independent batches do not share history unless you
manage continuation explicitly.

**Details:** Appends `<column>_lag_<n>` and keeps original columns. Missing previous
observations produce missing lags; drop_na filters rows with missing lag values
and aligns y. Without sort_by, input order is the sequence. Batch mode sorts
stably when requested and can reorder rows/y. State saves configuration, plus
bounded seed observations in carry mode. Context window; base dependencies.
Target lags require careful availability handling and are forbidden in carry
mode; use only prediction-available inputs in Bundle features.

Schema preview retains known shift-stable types, including nullable integers,
Boolean extensions, strings and floats. Native integer and Boolean labels can
change when missing lags are inserted, so ambiguous types return an unknown
schema. Apply keeps the engine's native behavior, including empty batches.

## RollingAggregate

**Purpose and effect:** `[10, 20, 30]` with window 2 and mean adds `[10, 15, 25]`: **the current row
is included**.

**File:** `config/preprocessing.yml`.

| Parameter | Type | Default | Allowed values | What it changes |
| --- | --- | --- | --- | --- |
| `columns` | list[string] | [] | Existing column names. | Select explicitly; no automatic detection. |
| `window` | integer | 3 | Positive. | Set rolling row-count window. |
| `aggregations` | string/list[string] | [mean] | mean, sum, min, max, std, median. | Choose summary features; unknown names fail during fit with supported choices listed. |
| `min_periods` | integer | 1 | No greater than window. | Require enough nonmissing observations. |
| `group_by` | string/list[string]/null | null | Existing grouping columns; use a list in carry mode. | Maintain separate entity sequences. |
| `sort_by` | string/null | null | Existing numeric/typed datetime clock for carry. | Order observations; absent uses batch input order. |
| `history_mode` | string | batch | batch, carry | Use the current batch or a bounded saved training tail. |
| `history_max_rows` | integer | 10000 | Positive; carry only. | Bound stored history rows. |
| `history_max_bytes` | integer | 1048576 | Positive; carry only. | Bound stored history bytes. |

**Minimal YAML:**

```yaml
- name: rollingaggregate
  transformer: RollingAggregate
  params:
    columns: [income]
    window: 2
    aggregations: [mean, max]
    min_periods: 1
    group_by: [entity]
    sort_by: observation_time
```

**Common mistake:** The current row is included; rolling the known target would leak its label and
is rejected.

**Details:** Unknown aggregation names fail during fit. Adds
`<column>_roll_<aggregation>_<window>` columns, keeping originals.
Missing observations affect min_periods; sample std needs enough observations.
Rolling the known target is rejected to prevent same-row label leakage.
Batch mode may reorder rows/y through sorting. Saves rules and optionally
bounded training history. Context window; base dependencies.

Pandas rolling outputs are double precision. Polars can retain float32 and
integer types for some aggregates. Schema preview predicts only known double
outputs: double inputs, or integer mean/median/std. Other combinations return
an unknown schema; this does not change their fitted or native apply behavior.

## Carry history for LagFeatures and RollingAggregate

Both accept `history_mode` string `batch` default or `carry`,
`history_max_rows` positive int default `10000`, and `history_max_bytes`
positive int default `1048576`. Use a list for grouped carry history, such as
`group_by: [entity]`. Carry requires explicit columns, a nonempty
sort_by, a numeric or typed datetime clock, and nonmissing unique entity/time
keys; parse text clocks first. It rejects drop_na and target leakage. It saves
the required tail per entity within both budgets; budget overflow fails instead
of silently discarding entities. New observations must follow saved history.
Apply does not mutate the model. Successive requests need caller-owned
continuation state if they should see one another; repeating apply without it
reuses the training seed. See [preprocessing context](preprocessing_context.md)
for TemporalHistorySession and publication of state with predictions.

```yaml
- name: rollingaggregate
  transformer: RollingAggregate
  params:
    columns: [income]
    window: 5
    aggregations: [mean]
    group_by: [entity]
    sort_by: observation_time
    history_mode: carry
    history_max_rows: 1000
    history_max_bytes: 48000
- name: dropmissingcolumns
  transformer: DropMissingColumns
  params: {columns: [observation_time], missing_threshold: null}
```

In a Bundle, observation_time must be an actual model input available at scoring,
distinct from protected split event metadata when necessary. Drop the raw clock
after generating features if the estimator cannot consume it. Do not split
ordered history across arbitrary Spark partitions or independent REST requests
and expect batch-equivalent results.

## GeoDistance

**Purpose and effect:** identical coordinate pairs produce distance 0.

**File:** `config/preprocessing.yml`.

| Parameter | Type | Default | Allowed values | What it changes |
| --- | --- | --- | --- | --- |
| `lat1_col` | string | required | Existing numeric latitude column in degrees. | Latitude of the first point; an empty name fails. |
| `lon1_col` | string | required | Existing numeric longitude column in degrees. | Longitude of the first point; an empty name fails. |
| `lat2_col` | string | required | Existing numeric latitude column in degrees. | Latitude of the second point; an empty name fails. |
| `lon2_col` | string | required | Existing numeric longitude column in degrees. | Longitude of the second point; an empty name fails. |
| `method` | string | haversine | haversine, euclidean | Choose spherical or flat-plane approximation. |
| `unit` | string | km | km, mi | Choose output distance unit. |
| `output_column` | string | geo_distance_<unit> | Explicit name, or empty/unset for generated name. | Name the appended/replaced distance column. |

**Minimal YAML:**

```yaml
- name: geodistance
  transformer: GeoDistance
  params: {lat1_col: latitude, lon1_col: longitude, lat2_col: store_lat, lon2_col: store_lon, unit: km}
```

**Common mistake:** Validate coordinates upstream; this step is not a general coordinate-cleaning
rule.

**Details:** Euclidean uses the flat-plane
approximation for coordinates. Saves configuration, appends/replaces the named
float output. Null coordinates produce missing distance; validate geographic
ranges upstream rather than assuming this is a coordinate-cleaning step.
Context row. Base dependencies.

## H3Index

**Purpose and effect:** nearby lat/lon observations map to string H3 cell IDs at the chosen grid
resolution.

**File:** `config/preprocessing.yml`.

| Parameter | Type | Default | Allowed values | What it changes |
| --- | --- | --- | --- | --- |
| `lat_col` | string | required | Existing numeric latitude column in degrees. | Latitude used to compute the H3 cell. |
| `lon_col` | string | required | Existing numeric longitude column in degrees. | Longitude used to compute the H3 cell. |
| `resolution` | integer | 9 | 0–15. | Choose H3 grid resolution. |
| `output_column` | string | h3_index | String. | Name the appended/replaced cell column. |

**Minimal YAML:**

```yaml
- name: h3index
  transformer: H3Index
  params: {lat_col: latitude, lon_col: longitude, resolution: 9, output_column: h3_index}
```

**Common mistake:** Install the h3 extra in both fit and apply environments; missing/invalid
coordinates produce null.

**Details:** Saves only configuration; appends/replaces the output column, keeps inputs and
row order. Missing/invalid coordinates and latitude outside ±90 or longitude
outside ±180 produce null. Context row. Optional dependency `h3` (`geo` extra)
must be installed in fit and apply environments. For example,
`uv pip install "skyulf-core[geo]"`.
