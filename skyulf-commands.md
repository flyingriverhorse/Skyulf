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
| App runtime (FastAPI)  | `[project.dependencies]`          | `requirements-fastapi.txt` |
| Dev / test / lint      | `[dependency-groups].dev`         | `requirements-dev.txt`     |
| CI gate tooling        | `[dependency-groups].dev`         | `requirements-ci.txt`      |
| Optional extra (geo…)  | `[project.optional-dependencies]` | `requirements-geo.txt` etc.|

> ⚠️ **`uv sync` foot-gun:** `uv sync` PRUNES any installed package not declared
> in `pyproject.toml`. Use `uv pip install -r requirements-fastapi.txt` to ADD
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
.\.venv\Scripts\python.exe -m ty check backend skyulf-core/skyulf skyulf-core/tests run_fastapi.py run_skyulf.py celery_worker.py

# Complexity gate
.\.venv\Scripts\python.exe -m lizard skyulf-core/skyulf -C 9 -w
```

<details>
<summary>Deprecated (pre-0.6.x) — black / flake8</summary>

```powershell
black --check backend skyulf-core tests run_fastapi.py run_skyulf.py celery_worker.py
flake8 . --count --select=E9,F63,F7,F82 --show-source --statistics
# mypy (replaced by ty)
mypy backend skyulf-core/skyulf skyulf-core/tests run_fastapi.py run_skyulf.py celery_worker.py
```
</details>

---

## 3. Tests (pytest)

```powershell
# Full suite (stop on first failure, short traceback)
.\.venv\Scripts\python.exe -m pytest tests/ -x --tb=short -q

# Core package with coverage floor (CI parity)
.\.venv\Scripts\python.exe -m pytest skyulf-core/tests -q --cov=skyulf --cov-branch --cov-fail-under=45 --disable-warnings

# Core coverage report (term)
.\.venv\Scripts\python.exe -m pytest skyulf-core/tests -q --cov=skyulf --cov-branch --cov-report=term
```

---

## 4. Frontend (ml-canvas)
```powershell
Set-Location frontend\ml-canvas
npm run lint
npm run test -- --run
npm run build

# Single Vitest file
npx vitest run useBranchColors
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

## 6. Build & Publish skyulf-core (PyPI)

```powershell
# Install build tooling
.\.venv\Scripts\python.exe -m uv pip install build twine

# Clean build
Remove-Item -Recurse -Force dist -ErrorAction SilentlyContinue
.\.venv\Scripts\python.exe -m build

# Upload (replace version)
.\.venv\Scripts\twine.exe upload dist/skyulf_core-0.1.16*
```

### Release a new core version (tag-triggered workflow)
1. Bump version in `skyulf-core/setup.py` (e.g. `0.1.16` → `0.1.17`).
2. Commit and push:
   ```powershell
   git add skyulf-core/setup.py
   git commit -m "chore: bump skyulf-core to 0.1.17"
   git push
   ```
3. Tag and push — the publish workflow fires automatically:
   ```powershell
   git tag core-v0.1.17
   git push origin core-v0.1.17
   ```

---

## 7. Release Drafter (GitHub Release Draft)

The release drafter reads the version from `pyproject.toml`, finds the matching
`## vX.Y.Z` block in `changelog/X.Y.x.md`, and publishes/updates a GitHub Release
draft automatically. It fires on every push to `master` that touches `pyproject.toml`
or `changelog/**`, or on manual `workflow_dispatch`.

### Step-by-step: cut a new app release

```
1. Write the changelog entry
2. Bump the version in pyproject.toml
3. Commit & push → release draft is auto-created on GitHub
4. Review the draft on GitHub, then "Publish release"
```

#### 1. Write the changelog entry

Add (or finish) the `## vX.Y.Z` section at the top of the matching series file:

```
changelog/
  0.6.x.md   ← edit this for a v0.6.x release
  0.7.x.md   ← create this file for a v0.7.0 release
```

The header must be **exactly** `## vX.Y.Z` or `## vX.Y.Z — Short title`.
Everything between that header and the next `## v` block becomes the release body.

```markdown
## v0.6.2 — Short release title

### 🔧 Backend
- **Thing:** what changed and why it matters.

### 🐛 Bug Fixes
- **Fix:** description.
```

#### 2. Bump the version in `pyproject.toml`

```toml
# pyproject.toml  ← the script reads [project].version
[project]
version = "0.6.2"   # ← change this
```

The script (`draft_release_from_changelog.py`) derives the series file from
the version — e.g. `0.6.2` → `changelog/0.6.x.md`. If the series file doesn't
exist yet, create it (see §7 below for a new minor series).

#### 3. Commit & push

```powershell
git add pyproject.toml changelog/0.6.x.md
git commit -m "chore: release v0.6.2"
git push
```

The `release-drafter.yml` workflow fires because `pyproject.toml` was touched.
It creates or updates the draft at
`https://github.com/flyingriverhorse/Skyulf/releases`.

#### Trigger manually (without a push)

```powershell
gh workflow run release-drafter.yml --ref master
```

---

### Starting a new minor series (e.g. v0.7.0)

1. Create `changelog/0.7.x.md` with a `## v0.7.0 — Title` header.
2. Bump `pyproject.toml` → `version = "0.7.0"`.
3. Commit & push — the drafter picks up the new file automatically.

---

## 8. Git & GitHub CLI

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

## 9. DCO Sign-off

In VS Code: Settings (`Ctrl + ,`) → search `signoff` → enable **Git: Always Signoff**.

Or via Git hook:
```bash
echo 'SOB=$(git var GIT_AUTHOR_IDENT | sed -n "s/^\(.*>\).*$/Signed-off-by: \1/p")' >> .git/hooks/prepare-commit-msg
echo 'grep -qs "^$SOB" "$1" || echo "" >> "$1"' >> .git/hooks/prepare-commit-msg
echo 'grep -qs "^$SOB" "$1" || echo "$SOB" >> "$1"' >> .git/hooks/prepare-commit-msg
chmod +x .git/hooks/prepare-commit-msg
```

---

## 10. Docs / GitHub Pages

```powershell
git commit --allow-empty -m "ci: trigger docs redeploy after gh-pages branch switch"
git push

gh api repos/flyingriverhorse/Skyulf/pages --jq '{status: .status, url: .html_url, source: .source, custom_domain: .custom_domain, https_enforced: .https_enforced}'
```

---

## 11. WSL / Ubuntu

```powershell
wsl.exe --install Ubuntu
```
