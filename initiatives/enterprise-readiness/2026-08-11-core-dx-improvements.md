# skyulf-core: standalone developer-experience (DX) audit

**Date:** 2026-08-11
**Scope:** `skyulf-core/skyulf/` package as a `pip install skyulf-core` library used directly in notebooks/scripts, outside of the FastAPI/Celery/visual-canvas machinery. Verified by actually importing and running the code (not just reading it).
**Not covered:** canvas/pipeline-authoring UX (see `2026-08-11-node-flexibility.md`), general tech debt (`2026-08-11-technical-debt-deep-dive.md`).

## Summary

`skyulf-core` is further along on "standalone DX" than a typical internal-only
library: it ships a real PyPI-style README, a flat re-exported
`skyulf.preprocessing` namespace, 9 example notebooks
(`skyulf-core/examples/00_quickstart.ipynb` … `08_online_retail_customer_segmentation.ipynb`),
and calculators/appliers that genuinely work stand-alone with plain
dict configs and no registry/decorator ceremony. The biggest remaining gaps
are (1) no sklearn `BaseEstimator`/`TransformerMixin` protocol, so it cannot
be dropped into `sklearn.pipeline.Pipeline` today, (2) `config`/`params` are
untyped `dict[str, Any]` everywhere, so IDEs can't autocomplete keys and typos
are silently ignored, and (3) validation is close to absent — bad column
names and wrong param types do not raise, they silently no-op.

## Findings

### 1. Calculators/Appliers silently no-op on bad config instead of raising

**Severity: High (friction) / Small-Medium (fix cost)**

**Location:** `skyulf-core/skyulf/utils.py:401` (`user_picked_no_columns`); `skyulf-core/skyulf/preprocessing/scaling/standard.py:114-119`; same pattern repeated across most calculators (`_select_subset_pandas`/`_select_subset_polars` helpers, `resolve_valid_columns`).

Reproduction (run directly against the installed package):

```python
import pandas as pd
from skyulf.preprocessing.scaling import StandardScalerCalculator

df = pd.DataFrame({"a": [1.0, 2.0, 3.0, 4.0]})
calc = StandardScalerCalculator()
artifact = calc.fit(df, {"columns": ["nonexistent"]})
print(artifact)   # {}  -- no error, no warning
```

```python
artifact = calc.fit(df, {"columns": "a"})  # str instead of list[str]
print(artifact)   # fits fine — "a" is iterated char-by-char if it happened
                   # to match multi-char columns, otherwise silently `{}`
```

**Actual:** A misspelled column name or a `str` passed where `list[str]` is
expected produces an empty artifact dict (`{}`) and the pipeline continues
with the untouched original data — no exception, no log line, no `KeyError`.
A new user has no signal that their config was ignored.

**Expected:** `fit()` should validate `columns` against `X.columns` (and
validate the container type) and raise a `ValueError`/custom
`SkyulfConfigError` naming the missing column(s)/bad type, mirroring
sklearn's `"columns are missing: {...}"` style errors.

**Before/after:**

```python
# before (utils.py:401)
def user_picked_no_columns(config):
    if "columns" not in config:
        return True
    return not config["columns"]

# after
def user_picked_no_columns(config, X=None):
    cols = config.get("columns")
    if cols is None:
        return True
    if not isinstance(cols, (list, tuple)):
        raise TypeError(f"'columns' must be a list[str], got {type(cols).__name__}: {cols!r}")
    if X is not None:
        missing = [c for c in cols if c not in X.columns]
        if missing:
            raise ValueError(f"columns not found in input data: {missing}")
    return not cols
```

---

### 2. No sklearn `BaseEstimator`/`TransformerMixin` protocol — cannot drop into `sklearn.pipeline.Pipeline`

**Severity: High (adoption blocker for the target notebook audience) / Medium fix cost**

**Location:** `skyulf-core/skyulf/preprocessing/base.py:82-140` (`BaseCalculator`, `BaseApplier` both subclass plain `abc.ABC`, not `sklearn.base.BaseEstimator`); `skyulf-core/skyulf/core/protocols.py` (`CalculatorProtocol`/`ApplierProtocol` are `Protocol`s with `fit`/`apply`, not `fit`/`transform`).

Reproduction:

```python
from sklearn.pipeline import Pipeline
from skyulf.preprocessing.scaling import StandardScalerCalculator
import pandas as pd

df = pd.DataFrame({"a": [1.0, 2.0, 3.0, 4.0]})
p = Pipeline([("scale", StandardScalerCalculator())])
p.fit(df)
```

```
TypeError: argument of type 'NoneType' is not iterable
  File ".../skyulf/preprocessing/scaling/standard.py", line 115, in fit
    if user_picked_no_columns(config):
  File ".../skyulf/utils.py", line 401, in user_picked_no_columns
    if "columns" not in config:
```

**Actual:** `Pipeline([...])` "builds" without error (duck-typed `fit`
exists), giving a false impression of compatibility, but `Pipeline.fit`
calls `fit(X, y, **fit_params)` — sklearn passes `config=None` positionally
where Skyulf expects a dict — and the call blows up deep inside a private
helper with a confusing `NoneType is not iterable` message that gives no
hint that the real problem is "this isn't an sklearn transformer."

**Expected:** Either (a) real sklearn compatibility — implement
`get_params`/`set_params`/`fit(X, y=None)`/`transform(X)` on top of the
existing `fit`/`apply`, likely via a thin adapter class
(`SklearnTransformerAdapter(calculator_cls, config)`), or (b) if full
compatibility is out of scope, fail fast with a clear message when a
Calculator is used outside `SkyulfPipeline`/`FeatureEngineer` (e.g. detect
`config is None` and raise `TypeError("StandardScalerCalculator.fit expects "
"a config dict; use FeatureEngineer or SkyulfPipeline for sklearn-style "
".fit(X, y) usage, or call fit(X, {...}) directly.")`).

**Cost:** Medium — an adapter wrapping one calculator+applier pair in
`BaseEstimator`/`TransformerMixin` is straightforward; doing it for all ~90
nodes with divergent `fit`/`apply` signatures (some return `SplitDataset`,
some are analyzers with no `apply`) needs a generic bridge plus per-family
exceptions.

---

### 3. `config`/`params` are bare `dict[str, Any]` everywhere — no IDE autocomplete, no typo protection

**Severity: Medium / Small-Medium fix cost**

**Location:** Every calculator/applier signature, e.g. `skyulf-core/skyulf/preprocessing/scaling/standard.py:114` (`def fit(self, X: Any, _y: Any, config: dict[str, Any])`), `.../base.py:82-95` (`BaseCalculator.fit(... config: dict[str, Any])`). Output *artifacts* are `TypedDict`s (`skyulf-core/skyulf/core/artifacts.py`, re-exported via `preprocessing/_artifacts.py:1-8`), but the **input** config accepted by `fit`/`apply` has no matching `TypedDict`/`dataclass` anywhere — `node_meta(params={...})` (`skyulf-core/skyulf/core/meta/decorators.py:15-38`) stores a *default-value* dict for UI purposes, not a schema.

**Actual:** A user typing `StandardScalerCalculator().fit(df, cfg)` gets zero
autocomplete on what keys `cfg` supports (`columns`, `with_mean`, `with_std`)
— they must open the source file or find the node in `node_meta` metadata.
Typos like `{"colums": [...]}` or `{"with_men": False}` are silently ignored
(same root cause as Finding 1) rather than caught by the type checker or at
runtime.

**Expected:** Add a `TypedDict` per node (mirroring the existing `*Artifact`
TypedDicts) for the *input* config, e.g.:

```python
class StandardScalerConfig(TypedDict, total=False):
    columns: list[str]
    with_mean: bool
    with_std: bool

class StandardScalerCalculator(BaseCalculator):
    def fit(self, X: Any, _y: Any, config: StandardScalerConfig) -> StandardScalerArtifact:
        ...
```
This is purely a typing improvement (TypedDicts are structurally dicts, so
it's backward compatible) — `mypy`/`pyright` and IDEs immediately start
flagging `config={"colums": [...]}` as an unknown key without any runtime
behavior change.

**Cost:** Small per node, but ~90 nodes to cover exhaustively; could be
staged by category (scaling/encoding first, since those are the ones a
notebook user reaches for first).

---

### 4. Public API surface is good today (`skyulf.preprocessing.X` works, flat re-exports) — worth calling out as a *strength*, but top-level `skyulf` import doesn't expose calculators at all

**Severity: Low (mostly positive) / N/A**

**Location:** `skyulf-core/skyulf/preprocessing/__init__.py:1-125` re-exports all ~138 public names (scaling, encoding, feature_generation, time_series, etc.) flatly, so `from skyulf.preprocessing import StandardScalerCalculator` works without knowing the `scaling` subfolder exists — this matches the sklearn/feature-engine convention of "one import path per concept." Compare `skyulf-core/skyulf/__init__.py:1-30`, which only exposes `SkyulfPipeline`, `SplitDataset`, `FeatureEngineer`, `NodeRegistry`, profiling helpers, and `validate_leakage_safety` — no individual calculators/appliers.

**Actual:** A user who does `import skyulf; skyulf.<tab>` sees pipeline-level
classes but not a single transformer, encoder, or scaler — for those they
must already know to go one level down into `skyulf.preprocessing`. This is
reasonable (matches `sklearn.preprocessing.StandardScaler` vs top-level
`sklearn`), so it's not really a bug, but it's worth documenting since new
users coming from `feature_engine.encoding.OneHotEncoder`-style imports may
initially expect `skyulf.StandardScaler` to work given the flat `__init__.py`
already imports `FeatureEngineer` at top level.

**Expected (optional polish):** README's "Quick start" already models
`SkyulfPipeline`-first usage (`skyulf-core/README.md` Quick start section),
which is the right onboarding path; no change strictly required. If a more
"import a single transformer" flow is desired, add one line to the README
showing `from skyulf.preprocessing import StandardScalerCalculator,
StandardScalerApplier` as an alternative to the full pipeline, since today
this path is undocumented even though it works (see Finding 6).

---

### 5. Registration via `@NodeRegistry.register` + `@node_meta` is a decorator side-effect, not a constructor requirement — standalone use genuinely works, but this isn't documented

**Severity: Medium (documentation gap) / Small fix cost**

**Location:** `skyulf-core/skyulf/preprocessing/scaling/standard.py:110-118` (decorators stacked directly on the class); `skyulf-core/skyulf/registry.py:17-54` (`NodeRegistry.register` just populates class dicts at import time, no instance-level requirement).

Verified directly:

```python
import pandas as pd
from skyulf.preprocessing.time_series import LagFeaturesCalculator, LagFeaturesApplier

df = pd.DataFrame({"t": [1, 2, 3, 4], "x": [10, 20, 30, 40]})
calc = LagFeaturesCalculator()
artifact = calc.fit(df, {"columns": ["x"], "lags": [1]})
app = LagFeaturesApplier()
out = app.apply(df, {"columns": ["x"], "lags": [1]})
```

Works with no FastAPI/Celery/registry lookup needed — `NodeRegistry` is only
consulted when a caller looks a node up *by string name* (as
`SkyulfPipeline`/canvas execution does), not when instantiating classes
directly.

**Actual:** This is a genuine strength: `BaseCalculator`/`BaseApplier`
subclasses are plain, side-effect-free (module import registers the node
globally, but that's harmless) classes usable exactly like sklearn
transformers, minus the sklearn protocol (Finding 2). However this fact is
not stated anywhere in the README or docstrings — a new user has to guess or
read source to learn they don't need `SkyulfPipeline`/`FeatureEngineer` at
all for single-node use.

**Expected:** Add a short "Using a single node directly" section to
`skyulf-core/README.md` right after "Quick start," showing the exact snippet
above, so users porting one transformer into an existing notebook pipeline
don't assume they must adopt the full `SkyulfPipeline` config format.

---

### 6. Documentation is materially better than typical internal libs, but has no per-node API reference

**Severity: Low / Medium fix cost**

**Location:** `skyulf-core/README.md` (full installation/quick-start/extras
matrix), `skyulf-core/examples/00_quickstart.ipynb` through
`08_online_retail_customer_segmentation.ipynb` (8 domain-specific
notebooks + `examples/README.md`), `mkdocs.yml:38` (`paths: [., skyulf-core]`
mkdocstrings config) and `mkdocs.yml:79` (`skyulf-core Configuration:
user_guide/configuration.md`).

**Actual:** There is a real README, real notebooks, and an mkdocstrings hook
into the root docs site — this is well ahead of a typical internal-only
package. What's missing is a generated **per-node API reference** (e.g. a
page per calculator listing its config keys/defaults, sourced from
`node_meta(params=...)` at `skyulf-core/skyulf/core/meta/decorators.py:15-38`,
which already carries a machine-readable `params` default dict per node but
is only consumed by the canvas backend, not rendered into docs).

**Expected:** A small doc-generation script that walks
`NodeRegistry._metadata` (`skyulf-core/skyulf/registry.py:10`) and emits a
Markdown table per node (id, category, description, default params) would
give notebook users a searchable reference without hand-writing docs for ~90
nodes.

**Cost:** Medium — the data already exists in `NodeRegistry`/`node_meta`;
this is a doc-generation script, not new instrumentation.

---

### 7. Test-only sample-dataset loader (`tests/utils/dataset_loader.py`) is a good "quickstart" utility candidate not exposed to users

**Severity: Low / Small fix cost**

**Location:** `skyulf-core/tests/utils/dataset_loader.py:1-53`
(`load_sample_dataset(name, engine="pandas"|"polars")`, reading from
`tests/data/*.csv`); referenced already in the master fix list
(`2026-08-11-master-fix-list.md:140`) as "Example CSVs already ship in the
repo (`skyulf-core/examples/data/`)" for the *canvas* UI's "Load sample
dataset" feature — but the *Python-API* version of this same idea (a public
`skyulf.datasets.load_sample("customers")`) does not exist.

**Actual:** A notebook user wanting "give me a quick DataFrame to try
`StandardScalerCalculator` on" has to bring their own CSV; the only sample
data shipped (`tests/data/`, `examples/data/`) is not importable from the
installed `skyulf-core` wheel (tests aren't packaged), so it's unusable
outside a git checkout.

**Expected:**

```python
# skyulf/datasets.py (new, ships in the wheel)
from importlib.resources import files
import pandas as pd

def load_sample(name: str = "customers") -> pd.DataFrame:
    """Load a small bundled example dataset, e.g. for trying nodes in a REPL."""
    return pd.read_csv(files("skyulf.datasets") / f"{name}.csv")
```
mirrors `sklearn.datasets.load_iris()`/`seaborn.load_dataset()` conventions
and reuses data already curated for tests/examples instead of authoring new
CSVs.

**Cost:** Small — copy 2-3 of the existing `examples/data/*.csv` files into a
packaged `skyulf/datasets/` directory and add `include_package_data`/
`package_data` wiring in `pyproject.toml`.

---

### 8. Config-dict interface is workable for now; a builder/dataclass API is a real but lower-priority ask

**Severity: Low / Large fix cost (if pursued)**

**Location:** `skyulf-core/README.md` Quick start (`SkyulfPipeline({...})`
nested dict config); `skyulf-core/skyulf/preprocessing/pipeline.py`
(`FeatureEngineer` consumes the same list-of-dict-steps shape).

**Actual:** The dict-of-steps format is identical to what the canvas backend
sends, which is good for parity/consistency (one config format, two
producers: UI or hand-written Python) — but it means typos like
`"transfromer": "OneHotEncoder"` or `"colums"` inside `params` fail silently
or with a generic `KeyError`/registry lookup error rather than a schema
validation error naming the offending step index.

**Expected (optional, larger effort):** A thin builder API layered *on top*
of the existing dict format (not replacing it, to preserve canvas/back-end
parity) — e.g.:

```python
from skyulf.builder import PipelineBuilder

pipeline = (
    PipelineBuilder()
    .split(target_column="purchased", test_size=0.2, random_state=42)
    .impute(columns=["income"], strategy="median")
    .one_hot(columns=["city"], drop_original=True)
    .model("logistic_regression", max_iter=500, random_state=42)
    .build()
)
```
`.build()` would emit the exact same nested dict `SkyulfPipeline` already
accepts, so this is additive sugar, not a rewrite, and could validate step
names/kwargs against `NodeRegistry` at build time (turning Finding 3's typo
class of bug into an immediate `TypeError` with a helpful message and
suggestion, e.g. via `difflib.get_close_matches`).

**Cost:** Large if it aims to cover all ~90 nodes with typed builder methods;
Medium if scoped to the ~15 most commonly used preprocessing/modeling nodes
first (scalers, common encoders, imputers, `TrainTestSplitter`, the handful
of modeling types in the Quick start example).

## Ranked by (friction removed) × (implementation cost)

| # | Finding | Friction removed | Cost | Priority |
|---|---|---|---|---|
| 1 | Silent no-op on bad column/type in `fit`/`apply` config | High — turns silent data-corruption-class bugs into loud, actionable errors for every user, canvas or standalone | S–M | **1** |
| 5 | Document that single-node standalone usage works (no registry ceremony needed) | High — removes the #1 unknown for a notebook user evaluating the library | S (docs only) | **2** |
| 7 | Public `skyulf.datasets.load_sample()` | Medium — classic "try it in 30 seconds" quickstart affordance | S | **3** |
| 3 | TypedDict configs per node for IDE autocomplete | Medium — compounds with Finding 1 to catch typos at type-check time | S per node, M in aggregate | **4** |
| 6 | Auto-generated per-node API reference from `NodeRegistry`/`node_meta` | Medium — turns tribal source-reading into a searchable page | M | **5** |
| 2 | sklearn `BaseEstimator`/`TransformerMixin` adapter | High for the specific "drop into existing sklearn Pipeline" audience, but a narrower audience than 1/5/7 | M (single adapter) – L (full coverage) | **6** |
| 4 | Top-level `skyulf` import doesn't include calculators (mostly a non-issue) | Low | S | **7** |
| 8 | Builder/dataclass config API | Low-Medium (dict format already usable) | L | **8** |
