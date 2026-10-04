# Skyulf — Command Cheatsheet

Quick reference for local dev, quality checks, releases, and CI parity.

- **Workspace root:** `c:\Users\Murat\Desktop\Skyulf`
- **Python:** uv-managed venv at `.venv` (use `uv pip`, never plain `pip`)
- **Activate venv (PowerShell):**
  ```powershell
  Set-ExecutionPolicy -Scope Process -ExecutionPolicy RemoteSigned
  .\.venv\Scripts\Activate.ps1
  ```

> Tip: the `ty` type checker is a normal dev dependency installed **into `.venv`**
> (pinned in `pyproject.toml` and `requirements-ci.txt`). With the venv activated
> just run `ty check ...`, or `.\.venv\Scripts\python.exe -m ty check ...`.

---

## 1. Dependency Management (uv)

**Rule:** never run plain `pip`. The `.venv` is uv-managed; plain `pip` bypasses
uv's resolver/lockfile and can leave orphaned `.dist-info` dirs. Installing a
package is only half the job — you must also **declare it** in `pyproject.toml`
(and the matching `requirements-*.txt`) so CI and fresh clones get it.

```powershell
# Add a RUNTIME dependency: installs + writes to pyproject [project.dependencies] + uv.lock
uv add "slowapi>=0.1.9"

# Add a DEV/TEST-only dependency: goes to [dependency-groups].dev
uv add --dev "pytest-mock>=3.14"

# Add to an optional-dependencies extra (e.g. geo, eda)
uv add --optional geo "geopandas>=1.1.2,<1.2.0"

# Remove a dependency (updates pyproject + lockfile)
uv remove slowapi

# Install WITHOUT touching pyproject (ad-hoc, not persisted to deps)
uv pip install <pkg>

# Re-lock only (regenerate uv.lock, no install) — pyscan reads the lock, not the venv
uv lock
```

**Manual edit alternative.** If you prefer editing `pyproject.toml` by hand, add
the pin to the right table, then install + re-lock:

```powershell
# 1. Add the line, e.g. under [project.dependencies] or [dependency-groups].dev:
#    "rapidfuzz>=3.6.1,<4.0.0",
# 2. Install it into the venv from the edited manifest
.\.venv\Scripts\python.exe -m uv pip install "rapidfuzz>=3.6.1,<4.0.0"
# 3. Regenerate the lockfile
uv lock
```

**CI parity — keep `requirements-*.txt` in sync.** CI installs from the
`requirements-*.txt` files, not from `pyproject.toml`. After adding a dep, also
add the same pin to the file CI consumes:

| Dependency kind        | pyproject table                   | requirements file          |
| ---------------------- | --------------------------------- | -------------------------- |
| App runtime (FastAPI)  | `[project.dependencies]`          | `requirements.txt` |
| Dev / test / lint      | `[dependency-groups].dev`         | `requirements-dev.txt`     |
| CI gate tooling        | `[dependency-groups].dev`         | `requirements-ci.txt`      |
| Optional extra (geo…)  | `[project.optional-dependencies]` | `requirements-geo.txt` etc.|

> ⚠️ **`uv sync` foot-gun:** `uv sync` PRUNES any installed package not declared
> in `pyproject.toml`. Use `uv pip install -r requirements.txt` to ADD
> without pruning, or `uv lock` when you only need the lockfile refreshed.
>
> ⚠️ **RECORD warning:** if `uv pip install` prints `Failed to uninstall ... due
> to missing RECORD file`, the package can become a broken namespace package
> (`import pkg` works but attribute access raises `AttributeError`). Fix with
> `uv pip install --force-reinstall <pkg>` and verify with
> `python -c "import pkg; print(pkg.__file__)"`.

---

## 2. Quality Checks (current toolchain — Ruff + ty)

The project migrated off `black` + `flake8` + `isort` to a single **Ruff** binary.

```powershell
# Lint (import sort + critical errors)
.\.venv\Scripts\python.exe -m ruff check .

# Format check (no changes) / apply
.\.venv\Scripts\python.exe -m ruff format --check .
.\.venv\Scripts\python.exe -m ruff format .

# Type check (ty is installed in the venv)
.\\.venv\Scripts\python.exe -m ty check backend skyulf-core/skyulf skyulf-core/tests run_skyulf.py celery_worker.py

# Complexity reports: CCN > 8 is informational in CI.
.\.venv\Scripts\python.exe -m lizard skyulf-core/skyulf --CCN 8 -w
.\.venv\Scripts\python.exe -m lizard backend --CCN 8 -w

# Complexity gates: CCN > 10 fails across all Core and backend source files.
.\.venv\Scripts\python.exe -m lizard skyulf-core/skyulf --CCN 10 -w
.\.venv\Scripts\python.exe -m lizard backend --CCN 10 -w
```

<details>
<summary>Deprecated (pre-0.6.x) — black / flake8</summary>

```powershell
black --check backend skyulf-core tests run_skyulf.py celery_worker.py
flake8 . --count --select=E9,F63,F7,F82 --show-source --statistics
# mypy (replaced by ty)
mypy backend skyulf-core/skyulf skyulf-core/tests run_skyulf.py celery_worker.py
```
</details>

---

## 3. Tests (pytest)

Local development uses explicit test files or node IDs. Start with the failing
regression, then run the affected files and direct integration consumers once
per completed batch. Reuse passing results while their code is unchanged;
reviewers add independent checks for uncovered cases. See `AGENTS.md` for the
batch and review rules. Do not run pytest without a selector locally.

```powershell
# Focused Core regression file
.\.venv\Scripts\python.exe -m pytest skyulf-core/tests/integration/core/test_function_step_mutation.py -q --tb=short

# Affected saved-model behavior and its integration consumer
.\.venv\Scripts\python.exe -m pytest skyulf-core/tests/integration/platforms/test_local_pipeline_policy_identity.py skyulf-core/tests/integration/platforms/test_local_pipeline_artifact.py -q --tb=short

# Check moved imports and fixtures without executing tests
.\.venv\Scripts\python.exe -m pytest skyulf-core/tests/integration --collect-only -q
```

GitHub CI retains the full Core/backend/frontend suites and coverage gates.
The Core branch-coverage floor is **90%**. Reproducing that full run locally is
reserved for an explicit user request; it is not the default repair command.
Pre-commit runs static checks, not pytest, and remains enabled.

---

## 4. Frontend (ml-canvas)
```powershell
Set-Location frontend\ml-canvas
npm run lint
# Pass the affected test file or an unambiguous filename filter.
npm run test -- useBranchColors
npm run build

# Run the affected browser flow only.
npm run test:e2e -- e2e/threshold-tuning.spec.ts
```

---

## 5. Pre-commit Hooks

Pre-commit hooks run in Git's context where the venv is **not** activated.
All system hooks use `uv run --no-sync python -m ...` so `uv` (which is on the
system PATH) locates the project venv automatically — no PATH or activation needed.

```powershell
# Install hook into Git (run once per clone)
.\.venv\Scripts\pre-commit install

# Run all hooks against the whole repo
.\.venv\Scripts\pre-commit run --all-files

# Validate the config YAML
.\.venv\Scripts\python.exe -c "import yaml; yaml.safe_load(open('.pre-commit-config.yaml', encoding='utf-8')); print('YAML Valid!')"
```

---

## 6. Git & GitHub CLI

```powershell
# Install gh
winget install --id GitHub.cli --silent --accept-source-agreements --accept-package-agreements

# Auth
gh --version
gh auth status
gh auth login

# Open a PR
gh pr list --head 057 --state open --json number,title,url
gh pr create --base master --head 057 --title "v0.5.7 - ..." --body-file temp/pr_body_057.md
```

### Merge a feature branch locally
```powershell
git checkout master
git pull
git merge --no-ff 057
git push
```

### Update local master (safe fast-forward)
```powershell
git fetch origin
git checkout master
git pull --ff-only origin master
git status -sb
```

### Trigger an empty CI / docs redeploy
```powershell
git commit --allow-empty -m "ci: trigger"
git push
```

---

## 7. DCO Sign-off

In VS Code: Settings (`Ctrl + ,`) → search `signoff` → enable **Git: Always Signoff**.

Or via Git hook:
```bash
echo 'SOB=$(git var GIT_AUTHOR_IDENT | sed -n "s/^\(.*>\).*$/Signed-off-by: \1/p")' >> .git/hooks/prepare-commit-msg
echo 'grep -qs "^$SOB" "$1" || echo "" >> "$1"' >> .git/hooks/prepare-commit-msg
echo 'grep -qs "^$SOB" "$1" || echo "$SOB" >> "$1"' >> .git/hooks/prepare-commit-msg
chmod +x .git/hooks/prepare-commit-msg
```

---

## 8. Docs / GitHub Pages

```powershell
git commit --allow-empty -m "ci: trigger docs redeploy after gh-pages branch switch"
git push

gh api repos/flyingriverhorse/Skyulf/pages --jq '{status: .status, url: .html_url, source: .source, custom_domain: .custom_domain, https_enforced: .https_enforced}'
```

---

## 9. WSL / Ubuntu

```powershell
wsl.exe --install Ubuntu
```
