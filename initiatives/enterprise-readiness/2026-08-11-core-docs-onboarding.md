# Enterprise Readiness — `skyulf-core` Documentation & Onboarding Audit

**Date:** 2026-08-11
**Status:** Investigation complete (subagent audit, spot-checked against real files)
**Scope:** Documentation artifacts only — README, docstrings, generated API
docs, examples, CONTRIBUTING, packaging metadata, changelog — for the
standalone `skyulf-core` Python package. Does **not** re-litigate the
architecture/extensibility findings already covered in
[2026-08-11-node-flexibility.md](2026-08-11-node-flexibility.md); this doc
is about whether existing docs *explain* that architecture well enough for
a newcomer.

## Summary

Contrary to the initial hypothesis, `skyulf-core` is **unusually well
documented for a repo at this stage** — it has a long, example-rich README,
a dedicated "extending custom nodes" guide with a copy-pasteable node
template, per-series changelogs, 9 runnable example notebooks, and a full
mkdocs + mkdocstrings site. The real gaps are narrower and more surgical:
docstring coverage is inconsistent inside the actual node implementation
files (as opposed to the docs describing them), the top-level `pyproject.toml`
is nearly empty (all metadata is dynamically pulled from `setup.py`, an
unusual and fragile split), and there's no enforced/measured docstring
coverage gate, so quality will drift silently over time.

---

### 1. README.md exists and is strong, but lives one level away from where a Python packager expects it

**Location:** `/Users/BH7043/Skyulf/skyulf-core/README.md` (438 lines)

**Finding:** A full README exists and covers nearly everything asked for:
installation (`README.md:22-52`, all 11 extras enumerated with rationale),
a runnable quickstart (`README.md:56-89`), the calculator/applier + pipeline
mental model with a mermaid diagram (`README.md:93-124`), explicit guidance
on when to drop to `Calculator`/`Applier` directly vs. use `SkyulfPipeline`
(`README.md:126-129`), a data-leakage-safety section with its own sequence
diagram (`README.md:150-183`), Polars-native/no-pandas guarantees
(`README.md:185-192`), a table of 9 example notebooks (`README.md:196-208`),
and an Automated EDA walkthrough (`README.md:210-243`). It does **not**
itself explain "how to add a new node" — that's intentionally deferred to
`docs/user_guide/extending_custom_nodes.md` (not linked from the README body
directly, though the mkdocs nav surfaces it).

**Severity:** Low. This is the one part of the audit that came back
better than expected.

**Recommendation:** Add one short "Extending Skyulf" section/link near the
bottom of `README.md` (next to "Features") pointing directly at
`docs/user_guide/extending_custom_nodes.md` (or its published URL) so a
GitHub-only visitor (who won't browse `docs/`) can find the node-authoring
guide without knowing it exists. One paragraph + one link is sufficient;
no need to duplicate content.

---

### 2. A real "add a new node" guide exists and is good — but is discoverable only via the platform mkdocs site, not from inside `skyulf-core/`

**Location:** `/Users/BH7043/Skyulf/docs/user_guide/extending_custom_nodes.md` (127 lines)

**Finding:** This file is exactly the doc a new contributor needs: it names
the Calculator/Applier split up front, gives a full copy-pasteable
`MyNodeCalculator`/`MyNodeApplier` template with `@NodeRegistry.register` +
`@node_meta` (`extending_custom_nodes.md:11-64`), explains what the
decorators actually do under the hood (`:66-70`), points to a real reference
implementation (`one_hot.py`, `:72-74`), covers modeling estimators
(`:76-82`), documents the duck-typed/no-subclassing escape hatch via
`StatefulTransformer` + `CalculatorProtocol`/`ApplierProtocol`
(`:84-113`), and ends with testing guidance (`:115-127`). This is a
genuinely complete document — better than most OSS "CONTRIBUTING" sections.
**However**, this file lives under the top-level `/docs/` mkdocs tree, not
inside `skyulf-core/`, and `skyulf-core/README.md` never links to it
directly (see Finding 1). A contributor who clones only `skyulf-core/` as a
subtree, or who is browsing the package on PyPI/GitHub without the parent
monorepo context, will not discover this file exists.

**Severity:** Medium — the content problem is solved; the discoverability
problem is not.

**Recommendation:** Either (a) mirror/symlink a short pointer file at
`skyulf-core/CONTRIBUTING.md` that says "see the Calculator/Applier guide
at `docs/user_guide/extending_custom_nodes.md`" with a live URL to the
published mkdocs page, or (b) if `skyulf-core` is ever split into its own
repo for standalone PyPI publishing, copy this file into
`skyulf-core/docs/` so it ships with the package's own source tree.

---

### 3. Docstring coverage on the actual node source files is inconsistent — passes on Calculator/Applier class-level docstrings, fails on private helper methods

**Location:** Sampled via AST function-count across 10 files in
`skyulf-core/skyulf/{preprocessing,modeling}`:

| File | Functions | Documented | Coverage |
|---|---|---|---|
| `preprocessing/imputation/_common.py` | 6 | 5 | 83% |
| `preprocessing/imputation/simple.py` | 7 | 0 | 0% |
| `preprocessing/imputation/knn.py` | 5 | 0 | 0% |
| `preprocessing/imputation/iterative.py` | 5 | 0 | 0% |
| `preprocessing/outliers/_common.py` | 2 | 2 | 100% |
| `preprocessing/outliers/manual_bounds.py` | 7 | 2 | 29% |
| `preprocessing/outliers/elliptic.py` | 6 | 1 | 17% |
| `preprocessing/outliers/iqr.py` | 5 | 0 | 0% |
| `preprocessing/outliers/zscore.py` | 5 | 0 | 0% |
| `preprocessing/encoding/one_hot.py` | 12 | 5 | 42% |
| `modeling/classification.py` | 26 | 5 | 19% |

**Detail:** `skyulf/preprocessing/imputation/simple.py:1` and
`skyulf/preprocessing/imputation/knn.py:1` do have a *module*-level
docstring ("Simple imputer node...", "KNN imputer node...") and the
Applier/Calculator *classes* often carry a docstring (e.g.
`SimpleImputerApplier` at `simple.py:21-25` has a full explanatory
docstring), but nearly every `apply`/`_apply_polars`/`_apply_pandas`/`fit`
*method* underneath has zero docstring — see `knn.py:18-20` (`apply`, no
docstring), `knn.py:23-28` (`_apply_polars`, no docstring). This is the
opposite pattern from `skyulf/preprocessing/base.py`, whose shared
`apply_method`/`fit_method` decorators (`base.py:32-61`) are fully
documented with prose explaining behavior, arguments, and when to skip them
— showing the team clearly *can* write good docstrings when it's core
infra, but the convention isn't applied uniformly to leaf node files.

**Severity:** Medium. Doesn't block usage (behavior is inferable from the
short, uniform helper functions and the working example in
`extending_custom_nodes.md`), but it does mean the "reverse-engineer 5
existing files" problem cited in the node-flexibility doc's scope note is
real *inside the code*, even though the standalone extending-guide (Finding
2) solves it for someone who finds that guide first.

**Recommendation:** Add a docstring linting gate (e.g. `ruff` with `D` rules
enabled, or `pydocstyle`) scoped initially to `skyulf/preprocessing/**` and
`skyulf/modeling/**`, requiring at minimum a one-line docstring on every
public method (`fit`, `apply`, `predict_output_schema`) and every
`NodeRegistry`-registered class. Private `_apply_polars`/`_apply_pandas`
engine-dispatch pairs can be exempted via a documented rule (their public
`apply` docstring plus the `_common.py` shared helpers already explain the
pattern), but that exemption should be explicit in `CONTRIBUTING.md`/the
node-authoring guide rather than an accident of omission.

---

### 4. No project-wide "General Coding Rules" document was found to check compliance against

**Location:** Searched `.github/instructions/*.md`,
`docs/contributing/writing_docs.md`, root `CONTRIBUTING.md` — no file
titled or describing "General Coding Rules" with a "every function must have
a short docstring" clause was found anywhere in the repo (checked
`.github/instructions/lint-typecheck-format.instructions.md`,
`backend-frontend-sync.instructions.md`, `codacy.instructions.md`; none
mandate docstrings).

**Severity:** Low-Medium — this means task point #7's premise (a written
rule this repo already has) could not be verified as an actual file; the
inconsistent coverage in Finding 3 may simply reflect no rule existing yet,
rather than a rule being violated.

**Recommendation:** If such a convention exists only as tribal knowledge or
in an agent system prompt outside this repo, codify it explicitly in
`docs/contributing/writing_docs.md` (which currently exists but should be
checked for a docstring-coverage clause) so both human contributors and
future coding agents have a single source of truth to enforce against, and
so this audit's compliance claim can be re-verified precisely next time.

---

### 5. Generated API documentation exists and is genuinely good (mkdocs + mkdocstrings, already built and published)

**Location:** `/Users/BH7043/Skyulf/mkdocs.yml:1-40`, `/Users/BH7043/Skyulf/docs/`
(11 top-level docs + `guides/`, `user_guide/`, `reference/`, `superpowers/`,
`ray-migration/` subfolders), and a pre-built static site at
`/Users/BH7043/Skyulf/site/` (e.g. `site/architecture.html`,
`site/index.html`) served at the README's badge-linked URL
`https://flyingriverhorse.github.io/Skyulf`.

**Finding:** `mkdocs.yml:32-38` configures the `mkdocstrings` plugin with
Google-style docstring parsing and `paths: [., skyulf-core]`, meaning the
site does pull live API reference from source docstrings — so Finding 3's
docstring gaps propagate directly into thin generated API-reference pages
for the affected files/methods, even though the *hand-written* guide pages
(architecture, EDA, extending_custom_nodes, etc.) are thorough.

**Severity:** Low for existence (it's there and good); ties back to Finding
3 for content quality on auto-generated pages specifically.

**Recommendation:** None needed for the docs infra itself — it's already a
credible OSS documentation setup. Prioritize Finding 3's docstring gate
specifically because it will directly and visibly improve
`mkdocstrings`-rendered pages, which is a different value proposition than
"hidden internal code that no one reads."

---

### 6. Example notebooks are comprehensive, real-dataset-based, and cover the full feature surface end-to-end

**Location:** `skyulf-core/examples/` — 9 notebooks (`00_quickstart.ipynb`
through `08_online_retail_customer_segmentation.ipynb`), indexed in both
`skyulf-core/README.md:196-208` and `skyulf-core/examples/README.md`
(69 lines, dataset sourcing notes per notebook).

**Finding:** This is not a gap. Notebooks cover classification, regression,
text classification, clustering, multiclass ensembles, imbalanced
classification (two flavors), structured-string feature parsing, and
RFM segmentation — with explicit callouts for leakage-safe practices,
Optuna tuning, and SHAP. All confirmed to use Polars/NumPy only, no pandas
(per the README's own claim at `README.md:191-192`, consistent with what's
described).

**Severity:** None — informational only, included because the task asked
to check for this explicitly.

**Recommendation:** None required. Optional nice-to-have: a single ultra-
minimal "5-line, no dataset download" snippet directly in
`examples/README.md` for someone who wants to sanity-check an install
without opening Jupyter, but this is a polish item, not a gap.

---

### 7. `pyproject.toml` is nearly empty — almost all packaging metadata lives in `setup.py` instead, an unusual and fragile split

**Location:** `/Users/BH7043/Skyulf/skyulf-core/pyproject.toml` (9 lines total):
```
[build-system]
requires = ["setuptools>=64", "wheel"]
build-backend = "setuptools.build_meta"

[project]
name = "skyulf-core"
dynamic = ["version", "description", "readme", "authors", "dependencies", "optional-dependencies", "classifiers", "urls"]
requires-python = ">=3.12"
```
versus `/Users/BH7043/Skyulf/skyulf-core/setup.py` (79 lines), which
actually defines `version="0.5.7"`, `description`, `long_description`
(pulled from README), `author`, `project_urls`, `install_requires`, all 12
`extras_require` groups, and `classifiers`.

**Finding:** This works (setuptools resolves `dynamic` fields from
`setup.py`), but it's a legacy/hybrid pattern most modern Python tooling
(uv, hatch, PDM, even pip's own docs) steers people away from — and it means
anyone skimming `pyproject.toml` alone (which is the first file most
PyPI-savvy users or tools like `pip-audit`/`poetry` inspect) sees almost
nothing. The `classifiers` list itself is also thin: only 3 entries
(`setup.py:76-79` — Python 3, Python 3.12, OS Independent). Missing:
`Development Status ::`, `Intended Audience ::`, `Topic :: Scientific/
Engineering :: Artificial Intelligence`, `License :: OSI Approved :: Apache
Software License` (the license is stated in prose in the README but not as
a classifier or an SPDX `license` field in either file).

**Severity:** Medium for OSS/PyPI credibility — a reviewer judging
"would this look credible on PyPI" by opening `pyproject.toml` first (a
common quick gut-check) will see an almost-empty file and could
under-rate the project, even though the actual published metadata (via
`setup.py`) is fine.

**Recommendation:** Migrate fully to PEP 621 static metadata in
`pyproject.toml` (drop `setup.py` or reduce it to nothing beyond what
`find_packages`/native extensions require), and expand `classifiers` to
include at least: `Development Status :: 4 - Beta`, `Intended Audience ::
Developers`, `Intended Audience :: Science/Research`, `Topic :: Scientific/
Engineering :: Artificial Intelligence`, `License :: OSI Approved :: Apache
Software License`, and `Typing :: Typed` (justified — `package_data`
already ships `py.typed` at `setup.py:24`).

---

### 8. Changelog exists, is well-organized by release series, but lives at the monorepo root, not inside `skyulf-core/`

**Location:** `/Users/BH7043/Skyulf/CHANGELOG.md` (root, 21 lines, indexing
into `/Users/BH7043/Skyulf/changelog/0.1.x.md` through `0.7.x.md`).
`setup.py:16` links `Changelog` project URL to GitHub Releases instead of
this file.

**Finding:** The changelog is genuinely good — versioned, dated, organized
by series with human-readable one-line summaries per series (e.g. "0.7.x —
Model Explainability, Segmentation Models & Unified Training Pipeline").
But it documents the whole Skyulf platform (frontend + backend + core)
undifferentiated — there's no `skyulf-core`-specific changelog filter, so a
pure library user pulling `pip install skyulf-core` and wanting to know
"what changed in core between 0.5.7 and 0.6.0" has to read through
platform/canvas/UI entries mixed in with core library changes.

**Severity:** Low. A real changelog trumps most projects at this maturity;
the platform-vs-core mixing is a minor navigation cost, not an absence.

**Recommendation:** Either add a `[core]`/`[backend]`/`[frontend]` tag
prefix to each changelog bullet (cheap, no restructuring), or point
`setup.py`'s `Changelog` project URL at the root `CHANGELOG.md` directly
(currently points at GitHub Releases, which may not be kept in sync with
the richer `changelog/*.md` files) so PyPI visitors reliably land on the
detailed version.

---

## Priority summary

| # | Finding | Severity | Effort to fix |
|---|---|---|---|
| 3 | Inconsistent method-level docstrings inside node source files | Medium | Medium (lint rule + backfill) |
| 7 | Nearly-empty `pyproject.toml` / thin classifiers | Medium | Small |
| 2 | Node-authoring guide undiscoverable from `skyulf-core/` alone | Medium | Small |
| 4 | No written docstring convention to audit against | Low-Medium | Small |
| 8 | Changelog mixes core/platform entries, stale Changelog URL | Low | Small |
| 1 | README lacks a direct link to the extending-nodes guide | Low | Trivial |
| 5 | mkdocstrings-generated pages inherit docstring gaps | Low | (fixed by #3) |
| 6 | Example notebooks — no gap found | None | N/A |

**Overall assessment:** `skyulf-core`'s documentation is closer to
"needs polish and a packaging cleanup" than "needs to be written from
scratch." The biggest real risk to new-contributor experience is Finding 3
(docstring coverage on the actual node implementations a contributor will
copy from), since the excellent standalone guide (Finding 2) can go
unread by anyone browsing code first.
