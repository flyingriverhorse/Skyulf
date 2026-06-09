# Releasing Skyulf

This document covers both release tracks:

- **App releases** — the FastAPI backend + frontend + ML canvas (`v0.6.x`, etc.) published as GitHub Releases via the automated Release Drafter.
- **Library releases** — `skyulf-core` published to PyPI via a tag-triggered workflow.

---

## App release (`vX.Y.Z`)

### How it works

The release drafter workflow (`.github/workflows/release-drafter.yml`) fires automatically on every push to `master` that touches `pyproject.toml` or any file in `changelog/`. It runs `.github/scripts/draft_release_from_changelog.py`, which:

1. Reads `[project].version` from `pyproject.toml` (e.g. `"0.6.2"`).
2. Derives the series file: `0.6.2` → `changelog/0.6.x.md`.
3. Extracts the `## v0.6.2` block as the release body.
4. Creates or updates a **draft** GitHub Release at the matching tag.

You then review and publish the draft on GitHub.

---

### Step-by-step: cut a new patch/minor release

#### 1. Write (or finish) the changelog entry

Open the matching series file and add a new `## vX.Y.Z` header **at the top**, above any existing versions:

```
changelog/
  0.6.x.md   ← for a v0.6.x release
  0.7.x.md   ← create this file for a new v0.7.0 minor
```

The header syntax must be **exactly** one of:

```markdown
## v0.6.2
## v0.6.2 — Short human-readable title
```

Everything between that header and the next `## v` block becomes the GitHub Release body. Use the standard categories:

```markdown
## v0.6.2 — Short release title

### 🆕 New Features
- **Feature:** description of what it does and why it matters.

### 🔧 Backend
- **Change:** what changed.

### 🎨 Frontend
- **Change:** what changed.

### 🐛 Bug Fixes
- **Fix:** what was broken and how it was fixed.

### 🧰 Tooling
- **Change:** toolchain / CI / config changes.
```

#### 2. Bump the version in `pyproject.toml`

```toml
[project]
version = "0.6.2"   # ← update this
```

> The script derives the series file purely from this version string.
> If the series file (`changelog/0.6.x.md`) does not exist the workflow fails.

#### 3. Commit & push to `master`

```powershell
git add pyproject.toml changelog/0.6.x.md
git commit -m "chore: release v0.6.2"
git push
```

The `release-drafter.yml` workflow triggers because `pyproject.toml` was modified. After ~30 seconds a draft appears at:

```
https://github.com/flyingriverhorse/Skyulf/releases
```

#### 4. Publish the draft

Review the draft on GitHub — tweak the body if needed — then click **"Publish release"**.

---

### Trigger the drafter manually (without a push)

```powershell
gh workflow run release-drafter.yml --ref master
```

---

### Starting a new minor series (e.g. v0.7.0)

1. Create `changelog/0.7.x.md` — the file **must** contain a `## v0.7.0` header:

   ```markdown
   # Changelog — 0.7.x Series

   ## v0.7.0 — New minor title

   ### 🆕 New Features
   - **First feature:** description.
   ```

2. Bump `pyproject.toml` → `version = "0.7.0"`.
3. Commit & push — the drafter picks up the new file automatically.

---

### Troubleshooting

| Symptom | Cause | Fix |
|---|---|---|
| Workflow fails: `Could not find changelog header` | `## vX.Y.Z` header is missing or misformatted in the series file | Check the exact header format — no extra spaces, correct version number |
| Workflow fails: `Changelog file ... not found` | Series file doesn't exist | Create `changelog/X.Y.x.md` with the required header |
| Draft not updating | Workflow didn't trigger | Push touched neither `pyproject.toml` nor `changelog/**`; run manually with `gh workflow run` |
| Wrong content in release | Wrong version in `pyproject.toml` | Ensure `pyproject.toml` version matches the changelog header exactly |

---

## `skyulf-core` library release (PyPI)

`skyulf-core` is published to PyPI separately from the app. Its release is triggered by pushing a Git tag matching `core-vX.Y.Z`. The workflow (`.github/workflows/release.yml`) builds a wheel + sdist and uploads via PyPI Trusted Publishing (OIDC — no API token needed).

### Step-by-step

#### 1. Bump the version in `skyulf-core/setup.py`

```python
setup(
    name="skyulf-core",
    version="0.3.1",   # ← update this
    ...
)
```

#### 2. Commit and push

```powershell
git add skyulf-core/setup.py
git commit -m "chore: bump skyulf-core to 0.3.1"
git push
```

#### 3. Tag and push — the publish workflow fires automatically

```powershell
git tag core-v0.3.1
git push origin core-v0.3.1
```

The `release.yml` workflow runs, builds `skyulf_core-0.3.1-py3-none-any.whl` + `.tar.gz`, and uploads them to PyPI.

#### 4. Verify on PyPI

```
https://pypi.org/project/skyulf-core/
```

---

### Building locally (optional, for verification before tagging)

```powershell
# Install build tooling (once)
uv pip install build twine

# Build
Set-Location skyulf-core
Remove-Item -Recurse -Force dist -ErrorAction SilentlyContinue
.\.venv\Scripts\python.exe -m build

# Check metadata
.\.venv\Scripts\twine.exe check dist/*
```

---

### Relationship between app version and core version

The two versions are **independent**:

| Track | File | Tag format | Published to |
|---|---|---|---|
| App (`backend` + `frontend`) | `pyproject.toml` `[project].version` | _(no Git tag — GitHub Release only)_ | GitHub Releases |
| Library (`skyulf-core`) | `skyulf-core/setup.py` `version=` | `core-vX.Y.Z` | PyPI |

An app release does **not** require a core release, and vice versa.
