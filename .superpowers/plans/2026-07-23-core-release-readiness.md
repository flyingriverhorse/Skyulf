# Core Release Readiness Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Make `skyulf-core` a reproducibly installable, clearly documented, and community-ready public library release.

**Architecture:** Keep the standalone `skyulf-core` package as the only runtime product in scope. Add a distribution-level CI check so the built wheel is installed outside the repository, correct public documentation to match package extras and curated notebooks, and document release/Kaggle publication procedures without adding SaaS functionality.

**Tech Stack:** Python 3.12, setuptools, pytest, GitHub Actions, MkDocs Material, Jupyter notebooks, PyPI Trusted Publishing.

## Global Constraints

- Support Python `>=3.12` only.
- Do not add backend, FastAPI, Celery, or frontend dependencies to `skyulf-core`.
- Do not publish a new package version, create a PyPI release, or upload notebooks to Kaggle as part of this implementation.
- Preserve the existing PyPI Trusted Publishing release workflow and `core-v<version>` tag format.
- Keep all examples reproducible from repository-bundled data; no Kaggle API key may be required.
- The hosted platform remains private and is out of scope for this plan.

---

## File Structure

| Path | Responsibility |
|---|---|
| `.github/scripts/verify_skyulf_core_distribution.py` | Builds an isolated temporary virtual environment, installs the wheel, and verifies the public import surface outside the repository. |
| `.github/workflows/skyulf-core-tests.yml` | Runs the wheel build and isolated-install check on every core CI run. |
| `skyulf-core/README.md` | Primary package landing page: accurate installation extras, quickstart, and curated notebook direction. |
| `skyulf-core/examples/README.md` | Accurate inventory and executable instructions for all bundled notebooks. |
| `docs/user_guide/installation.md` | Canonical website installation guidance consistent with package metadata. |
| `docs/guides/core-release.md` | Maintainer release checklist for versioning, distribution validation, trusted publishing, and post-release checks. |
| `docs/guides/kaggle-publication.md` | Repeatable manual process for publishing the three curated public notebooks. |
| `mkdocs.yml` | Navigation entries for public-library release and Kaggle guides. |
| `.github/ISSUE_TEMPLATE/feature_request.yml` | Removes the broken reference to a nonexistent root roadmap. |
| `SECURITY.md` | Supplies an actionable private GitHub reporting channel for public contributors. |

## Task 1: Verify the Built Package, Not the Checkout

**Files:**
- Create: `.github/scripts/verify_skyulf_core_distribution.py`
- Modify: `.github/workflows/skyulf-core-tests.yml`
- Test: CI job `skyulf-core test suite`

**Interfaces:**
- Consumes: a wheel path passed as the sole positional argument.
- Produces: exit code `0` only after a fresh virtual environment installs the wheel and imports `SkyulfPipeline`, `EDAAnalyzer`, `DriftCalculator`, and `NodeRegistry`.

- [ ] **Step 1: Demonstrate the missing distribution check**

Run:

```bash
rm -rf skyulf-core/dist
python -m build --outdir skyulf-core/dist skyulf-core
python .github/scripts/verify_skyulf_core_distribution.py skyulf-core/dist/*.whl
```

Expected: FAIL because `.github/scripts/verify_skyulf_core_distribution.py` does not exist.

- [ ] **Step 2: Add the isolated-install verifier**

Create `.github/scripts/verify_skyulf_core_distribution.py`:

```python
"""Verify that a built skyulf-core wheel works outside its source checkout."""

from __future__ import annotations

import subprocess
import sys
import tempfile
import venv
from pathlib import Path


def run(command: list[str], *, cwd: Path) -> None:
    """Run a checked command in the isolated verification directory."""
    subprocess.run(command, cwd=cwd, check=True)


def main() -> None:
    """Install the supplied wheel into a fresh venv and import public APIs."""
    if len(sys.argv) != 2:
        raise SystemExit("usage: verify_skyulf_core_distribution.py <wheel-path>")

    wheel = Path(sys.argv[1]).resolve()
    if wheel.suffix != ".whl" or not wheel.is_file():
        raise SystemExit(f"wheel does not exist: {wheel}")

    with tempfile.TemporaryDirectory(prefix="skyulf-core-wheel-") as directory:
        root = Path(directory)
        environment = root / "venv"
        venv.EnvBuilder(with_pip=True).create(environment)
        executable = "Scripts/python.exe" if sys.platform == "win32" else "bin/python"
        python = environment / executable

        run([str(python), "-m", "pip", "install", str(wheel)], cwd=root)
        run(
            [
                str(python),
                "-c",
                (
                    "from skyulf import ("
                    "SkyulfPipeline, EDAAnalyzer, DriftCalculator, NodeRegistry"
                    "); print('skyulf-core wheel import passed')"
                ),
            ],
            cwd=root,
        )


if __name__ == "__main__":
    main()
```

- [ ] **Step 3: Add the distribution job step after the existing test command**

In `.github/workflows/skyulf-core-tests.yml`, add this step immediately after
`Run skyulf-core tests`:

```yaml
      - name: Build and verify distributable wheel
        run: |
          echo "::group::Build skyulf-core wheel"
          python -m pip install --upgrade build
          rm -rf skyulf-core/dist
          python -m build --outdir skyulf-core/dist skyulf-core
          echo "::endgroup::"
          bash .github/scripts/run_check.sh "isolated skyulf-core wheel install" \
            python .github/scripts/verify_skyulf_core_distribution.py \
              skyulf-core/dist/*.whl
```

- [ ] **Step 4: Verify the new check locally**

Run:

```bash
rm -rf skyulf-core/dist
python -m pip install --upgrade build
python -m build --outdir skyulf-core/dist skyulf-core
python .github/scripts/verify_skyulf_core_distribution.py skyulf-core/dist/*.whl
```

Expected: `skyulf-core wheel import passed`.

- [ ] **Step 5: Run relevant static and regression checks**

Run:

```bash
ruff check .github/scripts/verify_skyulf_core_distribution.py
ruff format --check .github/scripts/verify_skyulf_core_distribution.py
ty check .github/scripts/verify_skyulf_core_distribution.py
pytest skyulf-core/tests/test_public_api_exports.py -q
```

Expected: all commands exit `0`.

- [ ] **Step 6: Commit**

```bash
git add .github/scripts/verify_skyulf_core_distribution.py .github/workflows/skyulf-core-tests.yml
git commit -m "ci: verify installed skyulf-core wheel"
```

## Task 2: Make Public Installation and Example Guidance Accurate

**Files:**
- Modify: `skyulf-core/README.md`
- Modify: `skyulf-core/examples/README.md`
- Modify: `docs/user_guide/installation.md`
- Test: `mkdocs build`

**Interfaces:**
- Consumes: extras declared in `skyulf-core/setup.py`.
- Produces: consistent documented commands for `viz`, `eda`, `text`, `nlp`, `geo`, `tuning`, `preprocessing-imbalanced`, `modeling-xgboost`, `modeling-lightgbm`, and `explainability`.

- [ ] **Step 1: Capture current documentation mismatches**

Run:

```bash
rg -n "Eight end-to-end|all.*optional|modeling-lightgbm|explainability|\\[text\\]|\\[nlp\\]|\\[geo\\]" \
  skyulf-core/README.md skyulf-core/examples/README.md docs/user_guide/installation.md
```

Expected: the output shows the examples README says “Eight” despite nine
notebooks and the website installation page omits currently supported extras.

- [ ] **Step 2: Update the package README’s installation and curation copy**

In `skyulf-core/README.md`:

1. Keep the existing individual extra commands.
2. Add the aggregate command directly after them:

```bash
# All non-geo optional runtime features
pip install skyulf-core[all]
```

3. Add this sentence beneath it:

```markdown
`all` intentionally excludes the native geospatial stack; add `[geo]` only when you need geospatial nodes.
```

4. Replace the final Kaggle sentence with:

```markdown
For the first public showcase, publish notebooks 01 (House Prices), 02 (Disaster Tweets), and 07 (Spaceship Titanic); each produces a competition-ready submission file.
```

- [ ] **Step 3: Correct the example inventory and installation command**

In `skyulf-core/examples/README.md`:

1. Change `Eight end-to-end Jupyter notebooks` to `Nine end-to-end Jupyter notebooks`.
2. Replace the setup command block with:

```bash
cd skyulf-core
pip install -e ".[dev,viz,eda,tuning,preprocessing-imbalanced,modeling-xgboost,modeling-lightgbm,explainability]"
python -m pip install jupyter
jupyter nbconvert --to notebook --execute --inplace examples/<name>.ipynb
```

3. Keep the existing note that no Kaggle credentials are required for local execution.

- [ ] **Step 4: Complete the website installation reference**

In `docs/user_guide/installation.md`:

1. Add rows for `text`, `nlp`, `geo`, `modeling-lightgbm`, and
   `explainability` using their exact `setup.py` extra names.
2. Replace the “Install everything at once” command and explanation with:

````markdown
Install all non-geospatial optional runtime features:

```bash
pip install skyulf-core[all]
```

Install geospatial functionality separately because it has native dependencies:

```bash
pip install skyulf-core[geo]
```
````

3. Remove the “Full platform (Docker)” section from this library installation
guide and replace it with one link to
`../guides/platform_setup.md`, labelled “Run the separate self-hosted platform”.

- [ ] **Step 5: Validate documentation and imports**

Run:

```bash
python -m mkdocs build --strict
pytest skyulf-core/tests/test_public_api_exports.py -q
```

Expected: the docs build has no broken internal link and the public export test passes.

- [ ] **Step 6: Commit**

```bash
git add skyulf-core/README.md skyulf-core/examples/README.md docs/user_guide/installation.md
git commit -m "docs: clarify skyulf-core installation and examples"
```

## Task 3: Document a Repeatable Core Release and Kaggle Publication Process

**Files:**
- Create: `docs/guides/core-release.md`
- Create: `docs/guides/kaggle-publication.md`
- Modify: `skyulf-core/README.md`
- Modify: `mkdocs.yml`
- Test: `mkdocs build --strict`

**Interfaces:**
- Consumes: the existing `skyulf-core/setup.py` version and `.github/workflows/release.yml` trusted-publishing workflow.
- Produces: a maintainer-run checklist; no release or Kaggle upload is automated.

- [ ] **Step 1: Add the core release checklist**

Create `docs/guides/core-release.md` with these mandatory sections:

````markdown
# Releasing skyulf-core

## Preconditions

- The target change is merged to `master`.
- The skyulf-core test workflow, static checks, security scan, and documentation build are green.
- The version in `skyulf-core/setup.py` follows semantic versioning and is not already tagged as `core-v<version>`.
- `CHANGELOG.md` and the applicable `changelog/<series>.md` entry describe user-visible changes.

## Local distribution verification

```bash
rm -rf skyulf-core/dist
python -m pip install --upgrade build twine
python -m build --outdir skyulf-core/dist skyulf-core
twine check skyulf-core/dist/*
python .github/scripts/verify_skyulf_core_distribution.py skyulf-core/dist/*.whl
```

## Publishing

Merge the version change to `master`. The `Release skyulf-core` workflow uses PyPI Trusted Publishing, publishes the already-validated distribution, and creates the `core-v<version>` tag. Do not add a PyPI API token to the repository.

## Post-release checks

```bash
python -m venv /tmp/skyulf-core-release-check
/tmp/skyulf-core-release-check/bin/python -m pip install --upgrade pip skyulf-core
/tmp/skyulf-core-release-check/bin/python -c "from skyulf import SkyulfPipeline; print(SkyulfPipeline)"
```
````

- [ ] **Step 2: Add the Kaggle publication guide**

Create `docs/guides/kaggle-publication.md` with:

````markdown
# Publishing Skyulf Core Kaggle Notebooks

## Curated notebooks

Publish these repository notebooks in this order:

1. `skyulf-core/examples/01_house_prices_regression.ipynb`
2. `skyulf-core/examples/02_disaster_tweets_text_classification.ipynb`
3. `skyulf-core/examples/07_spaceship_titanic_classification.ipynb`

They demonstrate regression, text classification, and tabular classification while each generates a submission artifact.

## Pre-publication verification

Run each notebook locally from the repository checkout before uploading it:

```bash
python -m pip install -e "skyulf-core[eda,viz,tuning,preprocessing-imbalanced,modeling-xgboost,modeling-lightgbm,explainability]"
python -m pip install jupyter
jupyter nbconvert --to notebook --execute --inplace skyulf-core/examples/01_house_prices_regression.ipynb
jupyter nbconvert --to notebook --execute --inplace skyulf-core/examples/02_disaster_tweets_text_classification.ipynb
jupyter nbconvert --to notebook --execute --inplace skyulf-core/examples/07_spaceship_titanic_classification.ipynb
```

## Publication rules

- Publish from the tested notebook revision and name the exact released `skyulf-core` version in the notebook introduction.
- Link to `https://github.com/flyingriverhorse/Skyulf` and `https://pypi.org/project/skyulf-core/`.
- Preserve each dataset source and sampling caveat already documented in `skyulf-core/examples/data/`.
- Do not include credentials, API tokens, local file paths, customer data, or unpublished SaaS details.
- Add a short note that Skyulf Core is the public Python library and that the hosted platform is not publicly available.
````

- [ ] **Step 3: Wire the guides into the public documentation**

In `skyulf-core/README.md`, append this sentence to the existing public
showcase paragraph:

```markdown
 See the [Kaggle publication guide](../docs/guides/kaggle-publication.md) for the manual release procedure.
```

In `mkdocs.yml`, add these entries to the `Guides` section immediately after
`Getting Started`:

```yaml
    - Core Release Process: guides/core-release.md
    - Kaggle Publication: guides/kaggle-publication.md
```

- [ ] **Step 4: Build the documentation**

Run:

```bash
python -m mkdocs build --strict
```

Expected: exit code `0`.

- [ ] **Step 5: Commit**

```bash
git add docs/guides/core-release.md docs/guides/kaggle-publication.md skyulf-core/README.md mkdocs.yml
git commit -m "docs: add core release and Kaggle publication guides"
```

## Task 4: Make Community and Security Contact Surfaces Actionable

**Files:**
- Modify: `.github/ISSUE_TEMPLATE/feature_request.yml`
- Modify: `SECURITY.md`
- Test: YAML parse and link check through `mkdocs build --strict`

**Interfaces:**
- Consumes: GitHub Issues, Discussions, and the repository Security Advisories feature.
- Produces: correct contribution links and a private vulnerability-reporting path.

- [ ] **Step 1: Verify the broken roadmap link**

Run:

```bash
test ! -f ROADMAP.md
rg -n "ROADMAP.md" .github/ISSUE_TEMPLATE/feature_request.yml
```

Expected: both commands confirm the template currently references a nonexistent file.

- [ ] **Step 2: Fix the feature request prompt**

Replace the `value` under the feature request template’s markdown block with:

```yaml
      value: |
        Thanks for suggesting a new feature!
        Please search [existing issues](https://github.com/flyingriverhorse/Skyulf/issues) and [GitHub Discussions](https://github.com/flyingriverhorse/Skyulf/discussions) before submitting.
```

- [ ] **Step 3: Make private reporting explicit**

Replace the `Reporting a vulnerability` section in `SECURITY.md` with:

```markdown
## Reporting a vulnerability

Do not open a public issue for a suspected vulnerability. Use the repository's
[private security advisory form](https://github.com/flyingriverhorse/Skyulf/security/advisories/new)
and include reproduction steps, impact, affected components, and a safe proof
of concept. We aim to acknowledge reports within 3–5 business days.
```

- [ ] **Step 4: Validate the changed community files**

Run:

```bash
python - <<'PY'
from pathlib import Path
import yaml

yaml.safe_load(Path(".github/ISSUE_TEMPLATE/feature_request.yml").read_text())
print("feature request template is valid YAML")
PY
python -m mkdocs build --strict
```

Expected: `feature request template is valid YAML` and a successful docs build.

- [ ] **Step 5: Commit**

```bash
git add .github/ISSUE_TEMPLATE/feature_request.yml SECURITY.md
git commit -m "docs: clarify public contribution and security reporting"
```

## Final Release-Readiness Verification

- [ ] **Step 1: Run the repository checks affected by this plan**

Run:

```bash
ruff check .github/scripts/verify_skyulf_core_distribution.py
ruff format --check .github/scripts/verify_skyulf_core_distribution.py
ty check .github/scripts/verify_skyulf_core_distribution.py
pytest skyulf-core/tests/test_public_api_exports.py -q
python -m mkdocs build --strict
```

Expected: all commands exit `0`.

- [ ] **Step 2: Run the installed-wheel verification**

Run:

```bash
rm -rf skyulf-core/dist
python -m build --outdir skyulf-core/dist skyulf-core
python .github/scripts/verify_skyulf_core_distribution.py skyulf-core/dist/*.whl
```

Expected: `skyulf-core wheel import passed`.

- [ ] **Step 3: Confirm the release boundary**

Run:

```bash
git status --short
```

Expected: only intentional, committed core-release readiness changes; no hosted-SaaS feature, credential, customer-data, or deployment change.
