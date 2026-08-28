# skyulf-core — findings register

**Subject:** `/Users/BH7043/Skyulf/skyulf-core`
**Date of audit:** this session
**Scope:** 330 source files, 78,923 LOC; 3,535 tests
**Method:** source reading, `ruff --select ALL`, `mypy`, full test-suite
execution, and targeted runtime experiments. Every finding below has a
`file:line` citation and, where the defect is behavioural, a reproduction.

---

## Read this first

**Two things you probably do not know about the current state of the library:**

1. **Your test suite is red.** 11 tests fail on `main` in your own `.venv`
   ([F-03](#f-03--the-test-suite-is-red-11-failures-on-main)).
2. **Nine of those failures are one real bug**: the Polars train/test split path
   is completely broken and has been since Polars 1.0
   ([F-01](#f-01--the-polars-splitting-path-is-broken)).

Beyond that, there are two silent correctness bugs
([F-02](#f-02--tuningconfigrandom_state-is-silently-ignored),
[F-05](#f-05--metric-calculation-swallows-every-exception-and-returns-silently))
and one measured quality gap
([F-04](#f-04--no-class-imbalance-handling-anywhere-in-the-classification-path)).

None of this changes the overall verdict, which is in
[`skyulf-core-assessment.md`](skyulf-core-assessment.md) and is favourable. The
library's core design — registry-derived behaviour, leakage detection, per-fold
preprocessing — is better than the Databricks template it would replace. What
follows is a fix list for a good codebase, not an indictment of a bad one.

**Severity key**

| Level | Meaning |
|---|---|
| 🔴 **Critical** | Produces a wrong result, a wrong decision, or a crash — silently or on a common path |
| 🟠 **High** | Will bite in production, or blocks a stated architectural goal |
| 🟡 **Medium** | Real defect, bounded impact; fix when next in the area |
| ⚪ **Low** | Hygiene; safe to batch |

---

## Contents

1. [Summary table](#1-summary-table)
2. [Critical findings](#2-critical-findings) — F-01 to F-05
3. [High findings](#3-high-findings) — F-06 to F-12
4. [Medium findings](#4-medium-findings) — F-13 to F-22
5. [Low findings](#5-low-findings) — F-23 to F-31
6. [Things I checked that are fine](#6-things-i-checked-that-are-fine)
7. [Suggested fix order](#7-suggested-fix-order)
8. [How to reproduce every finding](#8-how-to-reproduce-every-finding)

---

## 1. Summary table

| ID | Sev | Area | Finding |
|---|---|---|---|
| [F-01](#f-01--the-polars-splitting-path-is-broken) | 🔴 | Preprocessing | `DataFrame.gather()` does not exist in Polars — the Polars split path always raises |
| [F-02](#f-02--tuningconfigrandom_state-is-silently-ignored) | 🔴 | Tuning | `TuningConfig.random_state` never reaches the final model; it is always 42 |
| [F-03](#f-03--the-test-suite-is-red-11-failures-on-main) | 🔴 | CI | 11 tests fail on `main`; nothing appears to be gating this |
| [F-04](#f-04--no-class-imbalance-handling-anywhere-in-the-classification-path) | 🔴 | Modeling | No `class_weight` / `scale_pos_weight` in any default or search space |
| [F-05](#f-05--metric-calculation-swallows-every-exception-and-returns-silently) | 🔴 | Evaluation | `except Exception: pass` silently drops F1 / ROC-AUC / PR-AUC |
| [F-06](#f-06--a-failed-cv-fold-scores--inf-and-the-run-still-returns-a-model) | 🟠 | Tuning | All-folds-failed still yields a "best" model |
| [F-07](#f-07--the-engine-abstraction-is-bypassed-in-nine-modules) | 🟠 | Engines | `._df` private unwrapping in 9 modules across 6 packages |
| [F-08](#f-08--the-dataframe-protocol-verifies-nothing) | 🟠 | Engines | `SkyulfDataFrame.__getattr__ -> Any` disables all type checking |
| [F-09](#f-09--dual-engine-dispatch-cannot-accept-a-third-engine) | 🟠 | Engines | Dispatch hardcoded to two engines, silent pandas fallback |
| [F-10](#f-10--_hardcoded_model_map-duplicates-the-registry) | 🟠 | Pipeline | Hand-maintained 4-entry map shadows a 24-entry registry |
| [F-11](#f-11--173-function-level-imports-signal-circular-dependency-pressure) | 🟠 | Structure | 173 deferred imports; 4 explicitly labelled circular-dependency workarounds |
| [F-12](#f-12--drift-detection-uses-only-the-ks-p-value-and-discards-the-statistic) | 🟠 | Profiling | KS statistic computed then thrown away; p-value alone is sample-size dependent |
| [F-13](#f-13--threshold-tuning-exists-but-is-never-on-the-default-path) | 🟡 | Evaluation | The fix for F-04's symptom ships unused |
| [F-14](#f-14--unscoped-unlocked-global-singletons) | 🟡 | Core | 9 `global` statements; no locking, no scoping |
| [F-15](#f-15--pickle-based-reproducibility-digest) | 🟡 | Pipeline | Digest unstable across versions; `repr` fallback can collide |
| [F-16](#f-16--mypy-reports-23-errors-including-a-real-optional-handling-bug) | 🟡 | Types | `SplitPayload?` indexed and iterated without a `None` check |
| [F-17](#f-17--20-mutable-class-attributes-are-not-classvar) | 🟡 | Types | Shared-mutable-state risk in registries and pipelines |
| [F-18](#f-18--_tuningenginepy-is-1572-lines) | 🟡 | Structure | Six responsibilities in one module |
| [F-19](#f-19--pipelinepy-mixes-four-responsibilities) | 🟡 | Structure | Fitting, diagramming, digesting, persistence |
| [F-20](#f-20--eight-loggererror-calls-inside-except-blocks-discard-the-traceback) | 🟡 | Errors | `logger.error` where `logger.exception` is needed |
| [F-21](#f-21--33-hardcoded-random_state-42-defaults) | 🟡 | Reproducibility | The mechanism behind F-02, and a trap on its own |
| [F-22](#f-22--the-default-engine-is-pandas-not-polars) | 🟡 | Engines | Contradicts the Polars-first design and the dual-engine dispatch order |
| [F-23](#f-23--56-blind-except-handlers) | ⚪ | Errors | 56 `BLE001`; 6 are bare `try/except/pass` |
| [F-24](#f-24--only-the-first-fold-error-is-kept) | ⚪ | Tuning | Subsequent, possibly more informative, errors discarded |
| [F-25](#f-25--optional-test-dependencies-are-undeclared) | ⚪ | Packaging | 17 collection errors in a clean env (`hypothesis`, `pytest-benchmark`) |
| [F-26](#f-26--a-test-spawns-a-subprocess-that-cannot-import-skyulf) | ⚪ | Tests | `test_encoding_hash` fails on `PYTHONPATH`, not on logic |
| [F-27](#f-27--three-databricks-seams-are-stubs-with-confident-docstrings) | ⚪ | Docs | Reads as implemented to a newcomer |
| [F-28](#f-28--11-unused-imports-used-as-availability-probes) | ⚪ | Hygiene | Should be `importlib.util.find_spec` |
| [F-29](#f-29--13-stale-noqa-suppressions-and-2-blanket-noqa) | ⚪ | Hygiene | Suppressions outliving their cause |
| [F-30](#f-30--three-valueerrors-that-should-be-typeerrors) | ⚪ | Errors | `TRY004` — wrong exception class for type validation |
| [F-31](#f-31--miscellaneous-small-defects) | ⚪ | Mixed | Unused protocol, unused unpack, `logger.exception` outside a handler, unsorted `__all__` |

**Totals:** 5 critical, 7 high, 10 medium, 9 low.

---

## 2. Critical findings

### F-01 — The Polars splitting path is broken

**Severity:** 🔴 Critical
**File:** `skyulf/preprocessing/split.py:317`, `:320`, `:321`, `:348`, `:351`, `:352`

**The bug.** Polars provides `Series.gather()` but **not** `DataFrame.gather()`.
The splitter assumes the API is symmetric and calls `.gather()` on frames:

```python
# skyulf/preprocessing/split.py:348-353
            validation = df.gather(val_idx)

        return SplitDataset(
            train=df.gather(train_idx),
            test=df.gather(test_idx),
            validation=validation,
        )
```

Every one of those raises `AttributeError: 'DataFrame' object has no attribute
'gather'` on Polars ≥ 1.0 (your pinned version is 1.40.1).

Verified directly:

```
$ python -c "import polars as pl; print(hasattr(pl.DataFrame({'a':[1]}), 'gather'))"
False
$ python -c "import polars as pl; print(hasattr(pl.DataFrame({'a':[1]})['a'], 'gather'))"
True
```

Note that line 336 — `df.get_column(self.stratify_col).gather(tv_idx)` — is
**correct**, because it operates on a Series. The five broken sites are exactly
the ones where the receiver is a frame. That asymmetry is what made the mistake
easy to miss in review.

**Blast radius.** This is not an edge case. It is the primary split operation on
the primary engine:

- `DataSplitter.split()` on a Polars frame — broken
- `DataSplitter.split_xy()` on a Polars frame — broken
- Stratified and validation-bearing variants — broken
- `TrainTestSplit` as a pipeline step on a raw Polars frame — broken

It causes **9 of your 11 failing tests** ([F-03](#f-03--the-test-suite-is-red-11-failures-on-main)).

**Second-order consequence, which is the part that worries me.** `leakage.py`
guarantees leak-free preprocessing *by locating a splitter node in the config*.
The recommended, leak-safe workflow is: put a `TrainTestSplitter` in your
pipeline, and the checker verifies nothing data-dependent runs before it. On
Polars that workflow currently cannot execute at all. So the users following your
best-practice guidance are exactly the users who hit this.

**Fix.** Polars frames support positional row selection through `__getitem__`:

```pycon
>>> df[[3, 0]]      # returns rows 3 and 0, in that order
```

Replace the five frame-level `.gather(idx)` calls with `df[list(idx)]` /
`X[list(idx)]` (wrap in `list()` — `train_test_split` hands back a NumPy array,
and the list form is unambiguous). Leave the Series-level calls at lines 306,
317 (`y.gather`), 320, 321 (`y.gather`) and 336 alone — they are correct.

**I verified this fix.** Injecting exactly that shim before collection:

```python
if not hasattr(pl.DataFrame, "gather"):
    pl.DataFrame.gather = lambda self, idx: self[list(idx)]
```

turns the affected files completely green:

```
$ pytest skyulf-core/tests/integration/test_split.py \
         skyulf-core/tests/unit/test_pipeline.py -q
51 passed in 2.74s
```

So the change is confined to those five call sites; nothing else depends on the
broken behaviour.

Then add a regression test that runs the splitter under **both** engines with the
same input and asserts identical row selection. The reason this bug survived is
that the Polars path had tests, they were correct, they failed — and nothing
failed the build.

---

### F-02 — `TuningConfig.random_state` is silently ignored

**Severity:** 🔴 Critical
**Files:** `skyulf/modeling/_tuning/engine.py:262-265`, plus 33 calculators in
`skyulf/modeling/classification.py` and `regression.py`

**The bug.** The refit guards against overwriting a user-supplied seed:

```python
# skyulf/modeling/_tuning/engine.py:~262
final_params = {**self.model_calculator.default_params, **best_params}

if "random_state" not in final_params and hasattr(tuning_config, "random_state"):
    final_params["random_state"] = tuning_config.random_state
```

But **every calculator hardcodes `"random_state": 42` in its `default_params`**
(33 occurrences). So `"random_state" not in final_params` is *never* true, and
`tuning_config.random_state` is never applied. The seed the user asked for is
silently discarded.

**Reproduction — this is not a reading of the code, I ran it:**

```python
for seed in (1, 999):
    cfg = TuningConfig(strategy="optuna", metric="roc_auc",
                       search_space={"n_estimators": IntDistribution(10, 20)},
                       n_trials=2, random_state=seed, n_jobs=1)
    model, _ = TuningCalculator(RandomForestClassifierCalculator()).fit(
        X_tr, y_tr, config=cfg, validation_data=(X_v, y_v))
    print(seed, "->", model.random_state)
```

```
TuningConfig.random_state=1    ->  fitted model.random_state=42
TuningConfig.random_state=999  ->  fitted model.random_state=42
```

**Why critical.** A seed parameter that exists in the public API and does nothing
is worse than no seed parameter. Three concrete consequences:

1. **Variance studies are impossible.** Anyone repeating a run across seeds to
   estimate uncertainty is varying only the data split — the model itself is
   frozen at 42. They will systematically *underestimate* their model's variance
   and over-trust a single result.
2. **Reproducibility is accidental, not guaranteed.** It happens to be
   reproducible, but for the wrong reason, and it will stop being so the moment
   someone adds `random_state` to a search space (at which point `best_params`
   *does* override, and the behaviour changes silently again).
3. **It affects the benchmark you commissioned.** In the head-to-head, skyulf's
   models were refit at `random_state=42` for every seed while the template's
   varied. Not enough to change the conclusions — the seed-to-seed spread was
   dominated by the data split — but it is a real asymmetry and I have noted it in
   [`benchmark-review.md`](benchmark-review.md).

**Fix.** Reverse the precedence so config beats defaults, and make the
tuned-parameter case explicit:

```python
final_params = {**self.model_calculator.default_params, **best_params}

# The caller's seed must win over the calculator's baked-in default, but not
# over a seed the search itself selected.
if "random_state" not in best_params and tuning_config.random_state is not None:
    final_params["random_state"] = tuning_config.random_state
```

Then add a test asserting `model.random_state == config.random_state` — a
three-line test that would have caught this.

Longer term, do [F-21](#f-21--33-hardcoded-random_state-42-defaults): remove the
33 hardcoded 42s and let seeding be a single, explicit concern.

---

### F-03 — The test suite is red: 11 failures on `main`

**Severity:** 🔴 Critical (as a process defect)
**Command:** `python -m pytest skyulf-core/tests -q -o addopts=""` in
`/Users/BH7043/Skyulf` using `.venv`

```
11 failed, 3470 passed, 54 skipped, 56 warnings in 87.05s
```

| Failing test | Cause |
|---|---|
| `test_split.py::test_data_splitter_split_polars_round_trips_back_to_polars` | F-01 |
| `test_split.py::test_data_splitter_polars_split_preserves_dtypes` | F-01 |
| `test_split.py::test_data_splitter_polars_split_xy_preserves_dtypes` | F-01 |
| `test_split.py::test_data_splitter_split_polars_with_validation_round_trips` | F-01 |
| `test_split.py::test_data_splitter_split_xy_polars_round_trips_back_to_polars` | F-01 |
| `test_split.py::test_data_splitter_split_xy_polars_with_validation_round_trips` | F-01 |
| `test_split.py::test_feature_target_split_applier_handles_polars_split_dataset_input` | F-01 |
| `test_pipeline.py::test_train_test_split_step_actually_splits_a_raw_polars_dataframe` | F-01 |
| `test_pipeline.py::test_predict_rejects_input_containing_fitted_target_column[DataFrame1]` | F-01 |
| `test_encoding_hash.py::test_pandas_bucket_assignment_stable_across_process_hash_seeds` | F-26 (environmental) |
| `test_wrapped_polars_frames.py::test_sentence_embedder_accepts_wrapped_polars_frame` | Network — model download blocked |

**The finding is not the count.** 3,470 passing tests is a genuinely strong
suite, and I want to be clear that the engineering here is good. The finding is
that **nine tests are failing on a real bug and the codebase does not appear to
notice.** The tests were written, they are correct, they caught the regression —
and then nothing acted on the signal.

That is a CI gap, and it is more serious than any single bug in this document,
because it is the mechanism that lets bugs like F-01 persist.

**Fix.**

1. Fix F-01; that clears 9.
2. Fix F-26; that clears 1.
3. Mark the sentence-embedder test with
   `@pytest.mark.network` and deselect it by default, so an offline or
   firewalled environment gets a clean run.
4. Add a CI job that runs the suite and fails the build. If one already exists,
   find out why it is not blocking — that is the real bug.
5. Consider `--strict-markers` and treating collection errors as failures, so
   [F-25](#f-25--optional-test-dependencies-are-undeclared) cannot silently
   disable 17 modules.

---

### F-04 — No class-imbalance handling anywhere in the classification path

**Severity:** 🔴 Critical
**Files:** `skyulf/modeling/classification.py:300-315` (RF), `:428-439` (GB),
`:490-504` (XGB), `:78-99` (LR); and `skyulf/modeling/hyperparameters/`

**The gap.** Not one classifier sets `class_weight` or `scale_pos_weight` in
`default_params`, and neither parameter appears in any tuned search space:

```python
# skyulf/modeling/classification.py:300
class RandomForestClassifierCalculator(SklearnCalculator):
    def __init__(self):
        super().__init__(
            model_class=RandomForestClassifier,
            default_params={
                "n_estimators": 50,
                "max_depth": 10,
                "min_samples_split": 5,
                "min_samples_leaf": 2,
                "n_jobs": -1,
                "random_state": 42,
            },                          # <- no class_weight
            problem_type="classification",
        )
```

**Measured impact.** 50,000-row synthetic set, 15.7% positive rate, against the
Databricks template's equivalent service (which sets `class_weight="balanced"`):

| Model | ROC-AUC (ranking) | F1 (operating point) |
|---|---|---|
| Random Forest — template | 0.9552 | **0.9371** |
| Random Forest — skyulf | 0.9453 | **0.8670** |

AUC is near-identical, so **skyulf's model ranks just as well**. The whole
7-point F1 gap is the decision threshold sitting in the wrong place because the
minority class was never up-weighted.

The control case confirms the mechanism: Logistic Regression is the one
classifier where *neither* library sets a class weight, and across five seeds its
F1 delta was exactly `0.0000 ± 0.0000`.

**Why this matters more than average for Danica.** Claims, lapse, fraud and
churn are all rare-event problems. A classifier that ignores imbalance by default
is the characteristic failure of the domain — and it fails *quietly*, because the
headline AUC looks fine.

**The counter-argument, stated fairly.** skyulf *does* address imbalance — at the
preprocessing layer, via `Oversampling` and `Undersampling` nodes. Keeping
estimators pure and resampling upstream is a legitimate architecture. But it is
not equivalent: resampling distorts predicted probabilities, must be applied
per-fold to stay leak-free, and costs memory, whereas `class_weight` is free and
exact. And the nodes are opt-in, so the *default* remains unweighted.

**Fix.**

1. Add `"class_weight": None` to every applicable classifier's `default_params`,
   so the knob is at least visible.
2. Add `class_weight` to the tunable space for tree ensembles and logistic
   regression as a categorical over `[None, "balanced"]`. One line per model, and
   it lets the tuner *discover* the right answer per dataset rather than guessing.
3. For XGBoost/LightGBM, add `scale_pos_weight` as a tuned float. Do **not**
   hardcode a constant the way the template does — its `scale_pos_weight=13.0` is
   right for a ~7% positive rate and wrong for everything else, which is precisely
   why its XGBoost advantage nearly vanished at 15.7%.
4. Warn from the classification evaluator when the minority class is under ~20%
   and no imbalance handling is configured.

Pair this with [F-13](#f-13--threshold-tuning-exists-but-is-never-on-the-default-path);
together they close the gap completely.

---

### F-05 — Metric calculation swallows every exception and returns silently

**Severity:** 🔴 Critical
**Files:** `skyulf/modeling/_evaluation/metrics.py:225`, `:278`, `:302`;
`skyulf/modeling/_evaluation/classification.py:48`

```python
# skyulf/modeling/_evaluation/metrics.py:~215
        metrics["f1"] = float(
            f1_score(y_arr, predictions, average="binary",
                     pos_label=pos_label, zero_division=0)
        )
except Exception:
    pass
```

**Two distinct failure modes.**

1. **Silent metric loss.** If anything inside raises — a label dtype mismatch, a
   single-class fold, an unexpected `pos_label` — the block aborts and `metrics`
   comes back **missing keys**, with no log line. A consumer doing
   `metrics.get("roc_auc", 0.0)` then records a total failure as a *score of
   zero*; a consumer doing `metrics["roc_auc"]` gets a `KeyError` far from the
   cause.
2. **Order-dependent partial dictionaries.** The assignments are sequential
   inside one `try`, so an exception while computing `roc_auc` also discards `f1`
   even though `f1` computed successfully. Which metrics survive depends on their
   order in the source file — a genuinely surprising coupling.

`classification.py:48` has the same shape around `predict_proba`. That one is
more defensible (`predict_proba` is genuinely optional, and the `# nosec B110`
comment shows it was a considered decision), but the effect is that `y_prob`
becomes `None` and every probability-based metric silently disappears.

**Why critical.** Model selection reads these metrics. A metric that is
absent-or-zero rather than loudly broken can promote the wrong model, and the
logs will not say why.

**Fix.**

- Wrap each metric individually, not the group.
- Record the failure rather than dropping the key:
  `metrics["roc_auc"] = float("nan")` plus `logger.warning("roc_auc failed: %s", exc)`.
  `nan` propagates visibly through comparisons; a missing key does not.
- Narrow the caught type to `ValueError`/`TypeError`; let genuinely unexpected
  exceptions propagate.
- For `predict_proba`, keep the catch but log at `info` and set an explicit
  `metrics["proba_available"] = False`.

---

## 3. High findings

### F-06 — A failed CV fold scores `-inf`, and the run still returns a model

**Severity:** 🟠 High
**File:** `skyulf/modeling/_tuning/engine.py:824-832`

```python
except Exception as e:
    if fold_errors is not None and not fold_errors:
        fold_errors.append(str(e))
    ...
    return -float("inf")
```

Returning `-inf` for a broken fold is the right *local* choice — one bad
candidate should not win. The problem is that no global check follows. If
**every** candidate fails on **every** fold — a systematically malformed feature
matrix, a label encoding problem, a missing optional dependency — then all scores
are `-inf`, Optuna still nominates a "best" trial, `_refit_best_model` still
runs, and the caller receives a fitted model and an ordinary-looking
`TuningResult`.

The user sees a successful tuning run whose hyperparameters were chosen by
nothing at all.

**Fix.** After the search, assert at least one trial produced a finite score;
otherwise raise, surfacing the collected `fold_errors`. Additionally record
`n_failed_trials` on `TuningResult` and warn above ~20% — a partially failing
search is also a result the user needs to see.

---

### F-07 — The engine abstraction is bypassed in nine modules

**Severity:** 🟠 High
**Files:** `skyulf/utils.py:123`, `skyulf/engines/registry.py:134`,
`skyulf/preprocessing/dispatcher.py:36`,
`skyulf/preprocessing/feature_selection/correlation.py:48`,
`skyulf/preprocessing/vectorization/_common.py:63,70`,
`skyulf/profiling/expect.py:52`,
`skyulf/modeling/_evaluation/clustering.py:32`

Nine sites across six packages reach into a wrapped frame's private `._df` to
get the native object back. `ruff` flags 19 `SLF001` violations in total,
including `_scaler` (6×), `_SOLVER_PENALTIES`, `_ROW_DROPPING_TYPES` and
`_RESAMPLING_TYPES`.

**Why this is High and not style.** `._df` unwrapping is the load-bearing
workaround for [F-08](#f-08--the-dataframe-protocol-verifies-nothing): because
the protocol cannot express engine-specific operations safely, callers escape it.
Each escape is a place where engine-agnostic code silently becomes
engine-specific — and each is invisible to the type checker, which is exactly how
[F-01](#f-01--the-polars-splitting-path-is-broken) happened: a Polars-only method
called on a frame, with nothing in the type system to object.

The count is the real finding. One or two escapes is pragmatism; nine across six
packages means the abstraction is not carrying its weight.

**Fix.** Add a public, documented unwrap on the engine adapter —
`frame.to_native()` — and route all nine call sites through it. That does not by
itself fix the type safety, but it converts an invisible convention into a
greppable, testable API, and it gives you a single place to add a warning when
someone unwraps a distributed frame (see
[F-09](#f-09--dual-engine-dispatch-cannot-accept-a-third-engine)).

---

### F-08 — The dataframe protocol verifies nothing

**Severity:** 🟠 High
**File:** `skyulf/engines/protocol.py:38-47`

```python
def __getattr__(self, name: str) -> Any:
    """
    Allow engine-specific attribute access (e.g. pandas `.loc`, `.iloc`,
    `.select_dtypes`, polars `.with_columns`, `.is_empty`). Concrete
    engine adapters expose these natively; the Protocol stays minimal
    but transparent to the type checker.
    """
```

A `Protocol` with an unrestricted `__getattr__ -> Any` accepts **every**
attribute name. `mypy` can no longer tell you that `.iloc` is unavailable on a
Polars frame, that `.gather` is unavailable on a Polars *frame* (F-01), or that
`.slect_dtypes` is a typo.

The docstring is candid that this is deliberate. But the consequence is that
`SkyulfDataFrame` provides documentation value only, not safety — and it is the
one place in the design where safety would pay for itself, because it is the
seam between engines.

**This finding is now empirically confirmed.** F-01 is precisely the bug class
this protocol was positioned to prevent, and it shipped.

**Fix.** Split the protocol in two:

- `SkyulfDataFrame` — strict, only operations every engine implements, **no**
  `__getattr__`
- `PandasBackedFrame(SkyulfDataFrame)` — adds `.loc`, `.iloc`, `.select_dtypes`

Code needing pandas semantics then declares it in its signature, and the 27
`.iloc`/`.loc` sites become visible in the type system instead of hidden. Do the
same for a `PolarsBackedFrame`. This is the single highest-leverage structural
change in this document, and a hard prerequisite if a Spark engine is ever on the
table.

---

### F-09 — Dual-engine dispatch cannot accept a third engine

**Severity:** 🟠 High
**File:** `skyulf/preprocessing/dispatcher.py:102`, `:139-147`, `:153`

`apply_dual_engine` and `fit_dual_engine` take exactly two positional callables —
the Polars implementation and the pandas implementation. Lines 139-147 handle
"anything else" by **silently collecting to pandas**.

So the engine layer is not *N*-engine; it is two-engine with a pandas catch-all.
Adding Spark is therefore `O(number of nodes)`, not `O(1)`: 126 call sites across
49 files, plus the 27 pandas-only indexing sites from F-08.

The silent collection is the sharper edge. On a Spark frame it means pulling a
distributed dataset to the driver with no warning — unnoticeable at 10,000 rows,
fatal at 100 million. And `EngineRegistry._TOP_LEVEL_TO_ENGINE`
(`engines/registry.py:63-67`) *already* maps `"pyspark" -> "spark"`, so a Spark
frame will be detected, routed, and then quietly collected.

**Fix.** Change the signature to a mapping keyed by engine name
(`{"polars": fn, "pandas": fn}`) and raise `NotImplementedError` for unmapped
engines instead of falling back. That turns a silent, dangerous default into a
loud, correct one and makes a third engine additive rather than invasive. Do this
**before** any Spark work, not during it.

---

### F-10 — `_HARDCODED_MODEL_MAP` duplicates the registry

**Severity:** 🟠 High
**File:** `skyulf/pipeline.py:42-48`

```python
_HARDCODED_MODEL_MAP: dict[str, tuple[type[BaseModelCalculator], type[BaseModelApplier]]] = {
    "logistic_regression": (LogisticRegressionCalculator, LogisticRegressionApplier),
    "random_forest_classifier": (RandomForestClassifierCalculator, RandomForestClassifierApplier),
    "ridge_regression": (RidgeRegressionCalculator, RidgeRegressionApplier),
    "random_forest_regressor": (RandomForestRegressorCalculator, RandomForestRegressorApplier),
}
```

There are **24 registered estimators**. This fallback lists **4**. It is a
second, hand-maintained source of truth for something `NodeRegistry` already
knows, and nothing keeps them in sync. As registrations get completed, the
hardcoded entry either silently wins or silently diverges depending on lookup
order.

Worth naming plainly: this is the same defect I criticised the Databricks
template for — duplicated definitions with no sync mechanism. And it sits three
files away from `leakage.py`, which is the best example in either codebase of
doing the opposite.

**Fix.** Delete the map. If some estimators lack a paired applier, make that a
registry concern: add the pairing at registration, or have the registry raise a
clear "no applier registered for X". Then add a test that walks every registered
estimator and asserts a resolvable calculator/applier pair, so the gap cannot
reopen.

---

### F-11 — 173 function-level imports signal circular-dependency pressure

**Severity:** 🟠 High
**Evidence:** `ruff --select PLC0415` → 173 hits. Concentrated in
`profiling/visualizer.py` (15), `preprocessing/feature_generation/_polars_ops.py`
(12), `modeling/_tuning/engine.py` (9), `profiling/_analyzer/multivariate.py` (7).

Four are explicitly labelled as circular-dependency workarounds:

```
skyulf/modeling/ensemble.py:234:  # Lazy import avoids a circular dependency (the hyperparameters package
skyulf/modeling/base.py:228:      # Import here to avoid circular dependency if any
skyulf/modeling/base.py:600:      # Import here to avoid circular dependency
skyulf/engines/registry.py:13:    # to avoid circular imports if engines import protocol.
```

Some deferred imports are legitimate — deferring an expensive optional
dependency like `shap` or `statsmodels` is good practice. But 173 is well past
that, and `base.py:228`'s "if any" tells you the author was not certain the cycle
existed and deferred defensively.

**Why High.** Deferred imports move failures from import time to call time, so a
broken dependency surfaces mid-run instead of at startup. They also hide the real
module graph, which makes the file splits in
[F-18](#f-18--_tuningenginepy-is-1572-lines) and
[F-19](#f-19--pipelinepy-mixes-four-responsibilities) riskier than they need to
be.

**Fix.** Map the actual cycles (`pydeps`, or `python -X importtime`), then break
them structurally — usually by extracting the shared types into a leaf module
that both sides import. Keep deferred imports **only** for genuinely optional
heavy dependencies, and add a comment naming the dependency and why. Enable
`PLC0415` in `ruff` afterwards so new ones need a waiver.

---

### F-12 — Drift detection uses only the KS p-value and discards the statistic

**Severity:** 🟠 High
**File:** `skyulf/profiling/drift.py:196`

```python
# 2. KS Test
ks_stat, ks_p = ks_2samp(ref_data, curr_data)
ks_drift = ks_p < thresholds["ks"]
metrics.append(
    DriftMetric(
        metric="ks_test_p_value",
        value=float(ks_p),
        ...
```

`ks_stat` is computed and never used (`ruff` RUF059 confirms it is dead).

**Why that is a methodological problem, not a lint nit.** The KS *p-value* is a
function of both effect size and sample size. On a large monitoring window, an
utterly trivial distributional shift produces `p < 0.05`, and the detector fires.
On a small window, a large shift may not reach significance, and it stays silent.
So the alerting threshold implicitly depends on how much data happened to arrive.

The KS *statistic* is the effect size — the maximum CDF distance — and it is
sample-size robust. It is the quantity you actually want to threshold for drift
monitoring, and it is already sitting there in the tuple.

In production this manifests as alert fatigue: monitoring on a busy feature
fires constantly, everyone stops reading the alerts, and the one real drift event
is missed. I flagged the mirror-image problem in the Databricks template's
Lakehouse Monitoring config; this is the same mistake from the other direction.

**Fix.** Emit both metrics — `ks_statistic` and `ks_test_p_value` — and make the
drift decision on the statistic with a configurable threshold (0.1 is a common
starting point), optionally requiring significance as a secondary condition. Keep
the p-value in the report for diagnostics.

---

## 4. Medium findings

### F-13 — Threshold tuning exists but is never on the default path

**Severity:** 🟡 Medium
**Files:** `skyulf/modeling/_evaluation/thresholds.py`; the
`calibrated_classifier` estimator registration

The library **already ships the fix** for F-04's symptom: a threshold-selection
module and a calibration wrapper. Neither is invoked by the standard
`evaluate_classification` path, neither appears in the tuning flow, and neither is
mentioned in the docs. A user has no way to discover them.

This is a defaults-and-documentation problem rather than a missing capability,
which makes it unusually cheap for the value.

**Fix.** After tuning a binary classifier, optionally select the threshold that
maximises the configured metric on the validation split, store it on the result,
and apply it in `predict`. Gate behind `TuningConfig(tune_threshold=True)` so
nothing changes silently. Then document it.

---

### F-14 — Unscoped, unlocked global singletons

**Severity:** 🟡 Medium
**Files:** `skyulf/core/compute.py:50-52`, `skyulf/core/serialization.py:63-65`,
`set_active_engine` in the engines package. `ruff` counts 9 `global` statements
(`PLW0603`).

```python
def set_compute_backend(backend: ComputeBackend) -> None:
    global _DEFAULT_BACKEND
```

Process-wide mutable state, set by a bare function, no lock, no way to scope a
change to one block. Two consequences: concurrent pipelines in one process (a
Celery worker, a notebook running parallel experiments) can reconfigure each
other mid-run; and tests must remember to restore the previous value or they leak
state into whatever runs next.

Given that your backend runs Celery workers, the first is not hypothetical.

**Fix.** Keep the setters, but add context-manager forms
(`with compute_backend(x): ...`) implemented over `contextvars` rather than a
module global. `contextvars` behaves correctly under threads *and* asyncio, and
makes the test-isolation problem disappear.

---

### F-15 — Pickle-based reproducibility digest

**Severity:** 🟡 Medium
**File:** `skyulf/pipeline.py:56-66`

```python
def _artifact_digest(obj: Any) -> bytes:
    """Stable digest of a fitted artifact.

    Pickle is deterministic for the same fitted estimator (same numpy arrays),
    which is what we want for a reproducibility seal. Falls back to ``repr`` for
    the rare object that refuses to pickle.
    """
    try:
        return hashlib.sha256(pickle.dumps(obj)).digest()
    except Exception:
        return hashlib.sha256(repr(obj).encode("utf-8")).digest()
```

The docstring's claim is true *within* one environment and false *across*
environments. Pickle output embeds class module paths and protocol details, so
upgrading scikit-learn changes the digest for a byte-identical model. A
reproducibility seal that reports a difference after a routine dependency bump
gets ignored within a month.

**The `repr` fallback is the worse half.** For most estimators, `repr` prints
constructor parameters only. So two models with identical hyperparameters and
*completely different fitted weights* hash the same. A digest that can collide on
exactly the thing it exists to distinguish is worse than no digest, because it
reports success.

**Fix.** Digest semantic content: sorted hyperparameters, plus fitted arrays
(`coef_`, `feature_importances_`, tree structures) via `ndarray.tobytes()`, plus
the training-data digest. Delete the `repr` fallback and raise instead — an
artifact that cannot be digested should fail the seal, not silently pass it.

---

### F-16 — `mypy` reports 23 errors, including a real Optional-handling bug

**Severity:** 🟡 Medium
**Command:** `mypy skyulf --ignore-missing-imports`

| Code | Count |
|---|---|
| `attr-defined` | 5 |
| `arg-type` | 5 |
| `var-annotated` | 4 |
| `valid-type` | 3 |
| `return-value` | 2 |
| `misc`, `index`, `has-type`, `assignment` | 1 each |

The ones that look like genuine latent bugs:

```
skyulf/modeling/base.py:408:            Value of type SplitPayload? is not indexable          [index]
skyulf/preprocessing/pipeline.py:454:   SplitPayload? has no attribute "__iter__"             [attr-defined]
skyulf/preprocessing/pipeline.py:455:   SplitPayload? has no attribute "__iter__"             [attr-defined]
skyulf/preprocessing/pipeline.py:478:   SplitPayload? has no attribute "__iter__"             [attr-defined]
skyulf/preprocessing/pipeline.py:526:   "object" has no attribute "shape"                     [attr-defined]
skyulf/preprocessing/pipeline.py:527:   "object" has no attribute "ne"                        [attr-defined]
```

`SplitPayload?` is `Optional[SplitPayload]` being indexed and unpacked at four
sites with no `None` check. If the split is absent — which is exactly what
happens when a splitter fails, see
[F-01](#f-01--the-polars-splitting-path-is-broken) — these raise `TypeError:
cannot unpack non-sequence None` rather than a diagnosable error. These four
sites are in the same code path as the nine failing tests, so this is likely
where a fixed F-01 would surface its *next* problem.

Also worth noting: `profiling/_analyzer/dates.py:150` has a
`type: ignore[attr-defined]` that does not cover the error actually raised
(`has-type`) — a suppression that is silently doing nothing.

**Fix.** Fix the six listed errors, then add `mypy` to CI at the current error
count as a ratchet so the number can only go down.

---

### F-17 — 20 mutable class attributes are not `ClassVar`

**Severity:** 🟡 Medium
**Files:** `skyulf/registry.py:10-13` (4), `skyulf/preprocessing/pipeline.py`
(10), `skyulf/modeling/classification.py:85,216`, `skyulf/modeling/ensemble.py:137`,
`skyulf/engines/registry.py:57,63`, `skyulf/modeling/_tuning/engine.py:638`

`ruff` RUF012. Mutable class-level containers without a `ClassVar` annotation are
ambiguous: a reader cannot tell whether the dict is shared across all instances
(intentional, as for the registries) or was meant to be per-instance (a bug — the
classic shared-mutable-default trap).

For `NodeRegistry` and `EngineRegistry` the sharing is clearly intentional and
correct. For the ten in `preprocessing/pipeline.py` it is not obvious from
reading, which is the problem.

**Fix.** Annotate the intentional ones `ClassVar[dict[str, ...]]` and move any
that should be per-instance into `__init__`. Then enable RUF012 so the
distinction stays explicit. This is cheap and it makes an entire category of
future bug impossible.

---

### F-18 — `_tuning/engine.py` is 1,572 lines

**Severity:** 🟡 Medium

The largest file in the library by a wide margin — the next is 818. It holds the
Optuna strategy, the grid/random strategies, CV splitter construction, the
per-fold scoring loop, the refit, boosting-specific callback plumbing, and the
logging protocol.

Nothing here is *wrong*; this file contains some of the most careful code in the
repo, including the `PredefinedSplit` handling I verified. But at this size it is
hard to review, hard to test in isolation, and hard for a new engineer — or an AI
assistant — to hold in context. It is also where two of the five critical
findings live (F-02, F-06), which is not a coincidence.

**Fix.** Split along seams that already exist: `strategies/` (one module per
search strategy), `splitters.py` (the `_build_*_cv` family), `scoring.py` (the
fold loop), `refit.py`. Each becomes independently testable and the public
`TuningCalculator` surface does not change.

---

### F-19 — `pipeline.py` mixes four responsibilities

**Severity:** 🟡 Medium
**File:** `skyulf/pipeline.py` (579 lines)

Fitting, Mermaid diagram generation (`_mermaid_escape`), artifact digesting
(`_artifact_digest`), and persistence in one module. Diagram rendering in
particular has no business sitting next to the fit path.

**Fix.** Move `_mermaid_escape` and the diagram builder to `pipeline_diagram.py`,
and `_artifact_digest` to `pipeline_seal.py` (alongside the F-15 rewrite). That
also shrinks the file enough to make F-10's deletion obvious.

---

### F-20 — Eight `logger.error` calls inside `except` blocks discard the traceback

**Severity:** 🟡 Medium
**Files:** `modeling/_tuning/engine.py:1234`, `preprocessing/resampling.py:160,285`,
`preprocessing/transformations/power.py:86,104`,
`profiling/_analyzer/causal.py:132`, `profiling/correlations.py:119`,
`profiling/distributions.py:81`

`ruff` TRY400. Inside an exception handler, `logger.error(...)` records the
message but **not** the traceback; `logger.exception(...)` records both. When one
of these fires in a Databricks job, you get a one-line message and no stack — and
the code around it is exactly the code most likely to fail on unfamiliar data
(power transforms, resampling, causal analysis).

**Fix.** Mechanical: replace with `logger.exception` and drop any manual
`{e}` interpolation. `ruff --fix --unsafe-fixes --select TRY400` does it; review
the diff.

---

### F-21 — 33 hardcoded `random_state: 42` defaults

**Severity:** 🟡 Medium
**Files:** `skyulf/modeling/classification.py`, `regression.py`, `ensemble.py`

This is the mechanism behind [F-02](#f-02--tuningconfigrandom_state-is-silently-ignored),
but it is a trap on its own. A user who sets a global seed, or passes
`random_state` through a config, still gets 42 in every model unless they
override each calculator individually. Reproducibility becomes something that
happens *to* the user rather than something they control.

**Fix.** Remove `random_state` from `default_params` entirely and inject it in
one place during model construction, sourced from the config with a documented
fallback. Seeding then has exactly one owner. Do this immediately after F-02 —
F-02 is the urgent patch, F-21 is the durable fix.

---

### F-22 — The default engine is pandas, not Polars

**Severity:** 🟡 Medium
**File:** `skyulf/engines/registry.py:58`

```python
class EngineRegistry:
    _engines: dict[str, type[BaseEngine]] = {}
    _active_engine: str = "pandas"  # Default
```

The dispatcher is written Polars-first (`apply_dual_engine(polars_func,
pandas_func)`), the nodes carry `TODO(pandas-removal)` markers, and the
performance argument for the library rests on Polars. Yet a user who does not
call `set_active_engine` gets pandas.

This is not wrong — pandas is the safer default and the Polars path is less
covered (see F-01, which proves the point). But it means the *documented*
performance characteristics are not the *default* performance characteristics,
and most users will never know.

**Fix.** Either switch the default to Polars once F-01 is fixed and Polars parity
is green in CI, or state explicitly in the README that pandas is the default and
`set_active_engine("polars")` is opt-in. Silence is the only wrong answer.

---

## 5. Low findings

### F-23 — 56 blind `except` handlers

**Severity:** ⚪ Low individually, Medium in aggregate
**Evidence:** `ruff` BLE001 → 56; S110 (`try/except/pass`) → 6; S112
(`try/except/continue`) → 2. Raw count of `except Exception` → 73.

Many are legitimately defensive and annotated with `# noqa: BLE001` plus a
reason — SHAP explainability, drift profiling, best-effort schema inference. That
is good practice and I do not want to discourage it. The problematic subset is
the one that discards the exception object entirely;
[F-05](#f-05--metric-calculation-swallows-every-exception-and-returns-silently)
is the worst instance.

**Fix.** Adopt the rule: a broad catch must re-raise, log the exception, or carry
a comment explaining why the failure is genuinely uninteresting. Enable BLE001 in
`ruff` so new ones need an explicit waiver. Most existing sites already comply —
this is about preventing regression, not a cleanup campaign.

---

### F-24 — Only the first fold error is kept

**Severity:** ⚪ Low
**File:** `skyulf/modeling/_tuning/engine.py:825`

```python
if fold_errors is not None and not fold_errors:
    fold_errors.append(str(e))
```

`and not fold_errors` means: record the first error, ignore every subsequent one.
If fold 1 fails for an incidental reason and folds 2-5 fail for the real reason,
the user only ever sees the incidental one — which is the opposite of helpful
when diagnosing F-06.

**Fix.** Append all errors (they are strings; the cost is nil), or keep a
`Counter` of distinct messages so the dominant cause is visible.

---

### F-25 — Optional test dependencies are undeclared

**Severity:** ⚪ Low

Collecting the suite in a clean environment yields **17 collection errors**, all
`ModuleNotFoundError: No module named 'hypothesis'`. The property-based tests —
`test_engine_parity.py`, `test_schema_inference_fuzz.py`, and several
transformation/imputation integration modules — simply do not run. Separately,
`addopts` passes `--benchmark-skip`, which errors out unless `pytest-benchmark`
is installed.

The tests are fine. The problem is that a contributor who installs the declared
dev extras sees 17 red errors with no way to know they are environmental — and in
CI, if collection errors are not fatal, your **engine parity tests can silently
stop running**. Given F-01 is an engine parity bug, that is not a small thing.

**Fix.** Add `hypothesis` and `pytest-benchmark` to the dev/test extra in
`pyproject.toml`, or guard each module with
`pytest.importorskip("hypothesis")` so the skip is explicit and counted.

---

### F-26 — A test spawns a subprocess that cannot import `skyulf`

**Severity:** ⚪ Low
**File:** `tests/integration/test_encoding_hash.py::test_pandas_bucket_assignment_stable_across_process_hash_seeds`

The test verifies hash-bucket stability across `PYTHONHASHSEED` values by
launching `sys.executable -c "...import skyulf..."`. It fails with:

```
ModuleNotFoundError: No module named 'skyulf'
```

The subprocess inherits neither an editable install nor a `PYTHONPATH`. The
behaviour under test is genuinely important — an unstable hash encoder silently
changes feature meaning between runs — so this test should not be left broken.

**Fix.** Pass `env={**os.environ, "PYTHONPATH": str(repo_root), "PYTHONHASHSEED": seed}`
to `subprocess.run`, or install the package editable in CI. Do not skip it.

---

### F-27 — Three Databricks seams are stubs with confident docstrings

**Severity:** ⚪ Low
**Files:** `skyulf/core/compute.py`, `skyulf/core/serialization.py`,
`skyulf/core/model_registry.py`

Each notes it exists "ahead of the Databricks/MLflow phases". The abstraction
points are well chosen — this is *good* forward design, and it is why the
assessment concluded MLflow integration would be cheap. The only finding is that
an unimplemented interface with a confident docstring reads as implemented.

**Fix.** Have the stubs raise
`NotImplementedError("Databricks backend not yet implemented — see <issue>")`
rather than silently no-op, and put the status in the first line of the module
docstring where it cannot be missed.

---

### F-28 — 11 unused imports used as availability probes

**Severity:** ⚪ Low
**File:** `skyulf/profiling/_analyzer/_utils.py:87-110`

Eleven imports (`KMeans`, `PCA`, `IsolationForest`, `SimpleImputer`,
`StandardScaler`, `DecisionTreeClassifier`, `DecisionTreeRegressor`, `f_oneway`,
`kstest`, `shapiro`, `adfuller`) exist only to test whether a package is
installed. It works, but it imports and initialises real objects to answer a
yes/no question, and it reads like dead code to anyone maintaining the file.

**Fix.** Use `importlib.util.find_spec("sklearn.cluster") is not None`, which
ruff suggests directly. Cheaper and self-documenting.

---

### F-29 — 13 stale `noqa` suppressions and 2 blanket `noqa`

**Severity:** ⚪ Low
**Evidence:** `ruff` RUF100 → 13 unused `noqa`; PGH004 → 2 blanket `noqa`;
PGH003 → 1 blanket `type: ignore`

Suppressions that have outlived their cause. Individually harmless, collectively
they erode trust in every other suppression in the file — a reader can no longer
assume a `noqa` means something.

**Fix.** `ruff check --fix --select RUF100`, then give the two blanket `noqa` and
one blanket `type: ignore` specific rule codes.

---

### F-30 — Three `ValueError`s that should be `TypeError`s

**Severity:** ⚪ Low
**Files:** `modeling/_evaluation/metrics.py:53`,
`modeling/_evaluation/thresholds.py:72`, `pipeline.py:294`

`ruff` TRY004. Each validates an argument's *type* and raises `ValueError`.
Callers that correctly catch `TypeError` for type problems will miss these.

**Fix.** Change to `TypeError`. Check for downstream `except ValueError` handlers
first — this is a behavioural change, small as it is.

---

### F-31 — Miscellaneous small defects

**Severity:** ⚪ Low

| Item | Location | Note |
|---|---|---|
| `logger.exception` called outside an exception handler | `preprocessing/dispatcher.py:87` | Works only because callers happen to be in an `except` block; logs `NoneType: None` if that ever stops being true. Pass the exception explicitly. |
| Unused private protocol `_AnalyzerState` | `profiling/_analyzer/_utils.py:54` | Dead abstraction — delete or use it. |
| 23 unsorted `__all__` blocks | package `__init__` files | `ruff --fix --select RUF022`. |
| 10 uses of `.values` on pandas objects | various | `PD011`; prefer `.to_numpy()`, which is explicit about dtype and copy semantics. |
| 20 unnecessary lambdas | various | `PLW0108`; minor readability. |
| 130 magic-value comparisons | various | `PLR2004`; the ones inside statistical thresholds are worth naming as constants. |
| `for` loop variable rebound | `preprocessing/cleaning/value_replacement.py:124` | `PLW2901`. Reviewed — this one is idiomatic and safe; listing it only so you know it was checked and dismissed. |
| `S105` "hardcoded password" on `_MISSING_TOKEN` | `preprocessing/encoding/one_hot.py:22` | False positive; add a `# noqa: S105` with a reason if you enable `S`. |

---

## 6. Things I checked that are fine

Recording these matters as much as the defects — each was a real suspicion the
code disproved, and several are places where skyulf-core is doing something
better than most production ML libraries.

| Suspicion | Verdict |
|---|---|
| skyulf secretly runs 5-fold CV while the template uses one split, inflating its cost | **Fine.** `TuningConfig.cv_enabled` defaults `True`, but when `validation_data` is supplied, `engine.py:483-505` builds a `PredefinedSplit` (`-1` train / `0` val) — one fold, matching the caller's split exactly. |
| skyulf refits on train+val, giving itself more data than the baseline | **Fine.** The concatenation is local to the search; `X_np` is never rebound, so `engine.py:445` refits on train only. |
| `_instantiate_model` silently drops tuned nested parameters | **Fine, and better than most.** It filters kwargs against the constructor signature *and* routes nested `a__b` keys through `set_params` specifically so an ensemble's tuned base-model params survive. |
| Leakage detection is a hardcoded blocklist that will drift | **Fine, and the best code in the repo.** `leakage.py` derives both the data-dependent set and the splitter set from each node's own `@node_meta`, so it cannot drift. It **fails closed** on unknown transformers. Its four exemptions each carry a written justification, and `is_explicit_column_drop` correctly distinguishes threshold-based dropping (unsafe pre-split) from an explicit column list (safe). It even emits `NO_SPLIT_DIAGNOSTIC` when no splitter exists at all, rather than silently passing. |
| CV refits only the model, leaking preprocessing statistics across folds | **Fine — explicitly solved.** `modeling/fold_preprocessing.py` defines a `FoldPreprocessor` protocol whose docstring names this exact leak, and CV/tuning refit the preprocessor per fold. Most production ML code gets this wrong. |
| Mutable default arguments (the classic Python trap) | **Fine.** `ruff --select ALL` found zero B006/B008 violations across 330 files. |
| Test coverage is thin | **Fine.** 3,535 tests; 45,384 test LOC against 78,923 source LOC (0.57 ratio). The tests are also *good* — they caught F-01 correctly. |
| The `TODO(pandas-removal)` markers are abandoned work | **Fine.** Five markers, each stating the specific numerical reason the node still needs pandas (quantile interpolation semantics, per-column dropna). Documented constraints, not neglect. |
| Pickle use is a security risk | **Fine.** One `S301`, in `_artifact_digest`, on in-process trusted objects, correctly annotated `# nosec`. (It has a *different* problem — see F-15 — but not a security one.) |
| The project's own lint config is being ignored | **Fine.** `ruff check skyulf` against the committed config passes clean. Everything in this document came from rules you have not enabled, not from rules you are violating. |

---

## 7. Suggested fix order

Ordered by (impact ÷ effort), not by severity alone.

| # | Item | Effort | Why here |
|---|---|---|---|
| 1 | **F-01** Polars `gather` | ~1 hour | Five-line fix; unbreaks the primary split path and clears 9 failing tests |
| 2 | **F-02** seed propagation | ~1 hour | Three-line fix; removes a silently no-op public API |
| 3 | **F-03** get CI green and blocking | ~half day | Without this, everything else regresses. The highest-value item in the list |
| 4 | **F-05** metric swallowing | ~2 hours | Trivial; removes a silent-wrong-answer path |
| 5 | **F-26** subprocess `PYTHONPATH` | ~30 min | Clears the 10th failure |
| 6 | **F-25** declare test deps | ~30 min | Re-enables 17 modules, including engine parity |
| 7 | **F-04** class imbalance | ~1 day | Largest measured quality gap; mostly dict entries |
| 8 | **F-13** wire up threshold tuning | ~1 day | Code already exists; compounds with F-04 |
| 9 | **F-06** all-folds-failed guard | ~2 hours | Prevents a meaningless model being returned |
| 10 | **F-24** keep all fold errors | ~15 min | One line; do it inside F-06 |
| 11 | **F-12** KS statistic | ~2 hours | Fixes a real drift-alerting flaw cheaply |
| 12 | **F-20** `logger.exception` | ~30 min | Mechanical; large debugging payoff |
| 13 | **F-21** remove the 33 hardcoded seeds | ~half day | The durable fix behind F-02 |
| 14 | **F-16** mypy Optional errors | ~half day | The `SplitPayload?` sites sit in F-01's blast radius |
| 15 | **F-10** delete the hardcoded map | ~half day | Removes a divergence source; needs the registry test |
| 16 | **F-17** `ClassVar` annotations | ~2 hours | Cheap; closes a whole bug category |
| 17 | **F-29**, **F-28**, **F-30**, **F-31** | ~2 hours | Batch with `ruff --fix`; review the diff |
| 18 | **F-15** semantic digest | ~1 day | Fixes a claim the library cannot currently support |
| 19 | **F-07** public `to_native()` | ~1 day | Converts an invisible convention into an API |
| 20 | **F-09** engine dispatch mapping | ~2 days | Prerequisite for any third engine; do before Spark |
| 21 | **F-08** split the frame protocol | ~3 days | Highest leverage, largest blast radius — do deliberately |
| 22 | **F-11** break the import cycles | ~2 days | Do before the file splits |
| 23 | **F-18**, **F-19** file splits | ~2 days | Pure refactor; schedule when the area is quiet |
| 24 | **F-14** `contextvars` scoping | ~1 day | Needed before parallel in-process execution |
| 25 | **F-23** enable BLE001 | ~half day | Mostly waivers on existing sites |
| 26 | **F-22** decide the default engine | ~1 hour | Do after F-01 and Polars parity is green |
| 27 | **F-27** stub docstrings | ~1 hour | Do alongside the MLflow work |

**If you only do one thing:** items 1-6. That is roughly **two days**, and it
turns the suite green, removes both silent correctness bugs, and restores 17
disabled test modules.

**If you only do one week:** items 1-14. Every finding that can produce a wrong
answer is closed.

---

## 8. How to reproduce every finding

```bash
cd /Users/BH7043/Skyulf/skyulf-core

# ---- The two critical runtime bugs -------------------------------------
# F-01: Polars has Series.gather but not DataFrame.gather
python -c "import polars as pl; d=pl.DataFrame({'a':[1,2]}); \
           print('DataFrame.gather:', hasattr(d,'gather'), \
                 '| Series.gather:', hasattr(d['a'],'gather'))"
grep -rn "\.gather(" --include=*.py skyulf

# F-02: the caller's seed never reaches the model (prints 42 both times)
#   full script in this document, section F-02

# ---- Test suite state (F-03, F-25, F-26) -------------------------------
cd /Users/BH7043/Skyulf
.venv/bin/python -m pytest skyulf-core/tests -q -o addopts=""     # 11 failed
.venv/bin/python -m pytest skyulf-core/tests --collect-only -q -o addopts="" \
    2>&1 | tail -3                                                # 17 errors

# ---- Static analysis ---------------------------------------------------
cd /Users/BH7043/Skyulf/skyulf-core
uvx ruff check skyulf                       # clean against your own config
uvx ruff check skyulf --select ALL \
    --ignore D,ANN,COM,Q,I,E501,T20,PT,ARG,FBT,TD,FIX,ERA,S101,\
PLR0913,C901,PLR0912,PLR0915,TRY003,EM,N8 --statistics

# Per-finding rule codes
uvx ruff check skyulf --select RUF012 --output-format concise   # F-17
uvx ruff check skyulf --select SLF001 --output-format concise   # F-07
uvx ruff check skyulf --select PLC0415 --output-format concise  # F-11
uvx ruff check skyulf --select TRY400 --output-format concise   # F-20
uvx ruff check skyulf --select F401   --output-format concise   # F-28
uvx ruff check skyulf --select RUF059 --output-format concise   # F-12
uvx ruff check skyulf --select TRY004 --output-format concise   # F-30

uvx --with pandas --with polars --with numpy --with scikit-learn \
    mypy skyulf --ignore-missing-imports                        # F-16

# ---- Source-reading findings -------------------------------------------
grep -n "class_weight\|scale_pos_weight" skyulf/modeling/classification.py  # F-04 (no hits)
sed -n '215,232p' skyulf/modeling/_evaluation/metrics.py                    # F-05
sed -n '818,832p' skyulf/modeling/_tuning/engine.py                         # F-06, F-24
sed -n '30,50p'   skyulf/engines/protocol.py                                # F-08
sed -n '95,160p'  skyulf/preprocessing/dispatcher.py                        # F-09
sed -n '40,50p'   skyulf/pipeline.py                                        # F-10
sed -n '193,200p' skyulf/profiling/drift.py                                 # F-12
grep -rn "^def set_\|global _" --include=*.py skyulf/core skyulf/engines    # F-14
sed -n '55,67p'   skyulf/pipeline.py                                        # F-15
grep -rn '"random_state": 42' --include=*.py skyulf | wc -l                 # F-21 -> 33
sed -n '55,70p'   skyulf/engines/registry.py                                # F-22
find skyulf -name "*.py" -exec wc -l {} + | sort -rn | head -5              # F-18, F-19

# ---- Section 6: the things that are fine -------------------------------
sed -n '437,451p' skyulf/modeling/_tuning/engine.py    # refit uses train only
sed -n '483,505p' skyulf/modeling/_tuning/engine.py    # PredefinedSplit
sed -n '120,164p' skyulf/leakage.py                    # registry-derived, fails closed
sed -n '1,32p'    skyulf/modeling/fold_preprocessing.py
```

The F-04 measurement (the AUC/F1 divergence) came from a 50,000-row synthetic
benchmark; its method, confounds and full results are in
[`benchmark-review.md`](benchmark-review.md).

---

## Related documents

- [`skyulf-core-assessment.md`](skyulf-core-assessment.md) — the overall
  evaluation, the measured Spark-migration cost, and the layer-by-layer
  comparison against the Databricks template
- [`benchmark-review.md`](benchmark-review.md) — review of the head-to-head
  benchmark, including the evidence behind F-04
- [`../mlops-platform/`](../mlops-platform/) — the platform proposal that would
  consume this library
