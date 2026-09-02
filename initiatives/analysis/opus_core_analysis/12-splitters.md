# 12 — Splitters (`preprocessing/split.py`)

**Scope:** `skyulf/preprocessing/split.py` (517 lines) — `SplitCalculator`,
`SplitApplier`, `DataSplitter`, and the feature/target splitter.

This module was the one [OC-75](./11-tests-packaging-ci.md#oc-75) blocked from
testing (the repo venv's polars 1.40.1 has no `DataFrame.gather`, which the
polars path at `split.py:283-351` depends on). I built a separate venv with
**polars 1.44.1** to exercise it:

```bash
PYTHONPATH=skyulf-core /tmp/venv_polars/bin/python <script>
```

**Verdict: this module is correct.** I tried hard to break it across eight
behavioural properties and found no defect. Only one very minor robustness item
is filed. New ids continue from the master at **OC-90**.

| ID | Severity | Issue | Location |
| --- | --- | --- | --- |
| [OC-90](#oc-90) | ⚪ Low | `SplitCalculator.fit` silently drops unrecognised config keys, so a mistyped key disables the feature with no signal | `split.py:174-183` |

---

<a id="oc-90"></a>
### OC-90
### ⚪ Low — Unrecognised split config keys are silently dropped

**File:** `skyulf-core/skyulf/preprocessing/split.py:174-183`

`SplitCalculator.fit` copies a fixed whitelist into the artifact:

```python
for key in ("test_size", "validation_size", "random_state",
            "shuffle", "stratify", "target_column"):
    if key in config:
        artifact[key] = config[key]
```

The comment explains the intent well (keep the artifact shape stable rather than
"echoing arbitrary user keys back"), and that intent is right. The cost is that
a key which is *nearly* correct is discarded in total silence.

This is not hypothetical — **it caught me during this audit**. I passed
`stratify_col="target"` (the name used internally by `DataSplitter`, and an
entirely plausible guess) instead of `stratify`. Nothing warned; the split ran
happily unstratified, and the class balance silently drifted:

```
with stratify_col (silently ignored):  train=0.208  test=0.225  val=0.150
with stratify      (correct key):      train=0.200  test=0.200  val=0.200
```

For an SDK user driving this node from Python, a one-word slip yields a
plausible-looking but unstratified split with no error and no log line. The
frontend is unaffected (it sends the correct keys — verified below), so this is
a Python-API robustness issue only, hence ⚪ Low.

**Fix:** log a `debug`/`warning` for keys present in `config` but not in the
whitelist. Three lines, and it converts a silent wrong answer into a visible one.

---

## Checked and found sound

Every property below was executed against **both engines**. Unless noted, pandas
and polars produced **byte-identical** results.

### Cross-engine parity is exact

Not merely "same shape" — the *same rows*. 100 rows, `test_size=0.2`,
`validation_size=0.2`, `random_state=42`, stratified:

```
pandas: train=60 test=20 val=20
polars: train=60 test=20 val=20
  train      identical rows across engines: True
  test       identical rows across engines: True
  validation identical rows across engines: True
```

This is a genuinely strong result. The two paths are structurally different —
pandas splits frames via `train_test_split`, polars computes indices and
`gather`s them (`split.py:285-321`) — yet they agree row-for-row. It also means
[OC-76](./11-tests-packaging-ci.md#oc-76)'s complaint (parity tests never compare
applied output) does **not** conceal a defect here.

### The three-way split partitions the data exactly

200 rows, 20% test / 20% validation:

```
pandas: overlap tr/te=0  tr/va=0  te/va=0  union=200/200
polars: overlap tr/te=0  tr/va=0  te/va=0  union=200/200
```

No leakage between partitions, no row lost, no row duplicated.

### Stratification is exact, not approximate

Source positive rate 0.200 (160/40 imbalance), stratified 3-way split:

```
pandas: train=0.200  test=0.200  val=0.200
polars: train=0.200  test=0.200  val=0.200
```

This matches a hand-built reference `train_test_split` chain exactly (8 positives
in test, 8 in validation, 24 in train). The relative-size arithmetic at
`split.py:387` and `:408` —
`relative_val_size = validation_size / (1 - test_size)` — is the correct formula,
and it is applied to the post-test remainder, so the validation fraction is
measured against the *original* frame as a user would expect.

### Degenerate and edge configurations are handled deliberately

| Case | Behaviour |
| --- | --- |
| `test_size + validation_size >= 1` | Raises `ValueError` with an explicit message naming both values and explaining "there is no data left for training". Tested at `(0.6,0.4)`, `(0.7,0.4)`, `(0.5,0.5)` — all rejected. |
| Least-populated class has 1 member | `_safe_stratify` / `_safe_stratify_polars` log a warning and **disable** stratification rather than crashing. Both engines returned the identical `train=8 test=3` split. |
| `shuffle=False` (time-ordered data) | Order is strictly preserved: `train=[0,1,2,…,79]`, `test` begins at `80`, `ordered=True`. Identical on both engines — important, since the polars path could easily have permuted here. |
| `stratify=True` with no target column, plain-frame input | Emits a precise warning — *"no target_column is configured for this plain-DataFrame input, so there is no column to stratify on. Stratification will be disabled."* — and proceeds. No crash on the `"__implicit_target__"` sentinel. |

### The frontend contract is correctly bridged

Worth stating explicitly because it *looks* like a type mismatch of exactly the
kind this repo has shipped before. `TrainTestSplitNode.tsx:13` declares
`stratify: **boolean**`, while `DataSplitter.__init__` takes
`stratify_col: **str | None**`. That is reconciled properly in `_build_splitter`
(`split.py:103-112`):

```python
stratify   = params.get("stratify", False)
target_col = params.get("target_column")
stratify_col = target_col if stratify else None
if stratify and not target_col:
    stratify_col = "__implicit_target__"
```

The UI's checkbox plus its `target_column` dropdown (correctly `disabled={!config.stratify}`,
`TrainTestSplitNode.tsx:169`) map onto the backend's single column-name argument
without loss. **No frontend/backend drift here.**

## Improvements

- **Promote the parity check above into a test.** `test_engine_parity.py`
  currently compares fitted artifacts, not applied output
  ([OC-76](./11-tests-packaging-ci.md#oc-76)). The row-identity assertion used
  here is ~10 lines and would lock in the strongest property this module has.
- **Guard the polars floor.** The entire polars path silently requires
  `DataFrame.gather`. Until [OC-75](./11-tests-packaging-ci.md#oc-75) is fixed,
  anyone on polars < 1.43.2 gets an `AttributeError` from deep inside
  `_split_polars` rather than a dependency error.
- **`shuffle=False` deserves a first-class time-series splitter.** It currently
  works by accident of `train_test_split`'s contiguous behaviour; a named
  `TimeSeriesSplit` node would make the intent explicit and support rolling
  windows.
