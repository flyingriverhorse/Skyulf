# 0.7.5 Version Initialization Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Initialize branch `075` with application/frontend version `0.7.5` and `skyulf-core` version `0.5.6`.

**Architecture:** Keep the root `pyproject.toml` as the application version source of truth, use the existing frontend synchronization script for npm metadata, and retain `skyulf-core/setup.py` as the core package version source. Refresh `uv.lock` only to synchronize workspace metadata, without upgrading third-party dependencies.

**Tech Stack:** Python packaging (`pyproject.toml`, setuptools, uv), Node.js/npm package metadata

## Global Constraints

- Application and frontend version must be exactly `0.7.5`.
- `skyulf-core` version must be exactly `0.5.6`.
- Do not add features, changelog entries, or unrelated dependency updates.
- Commit with a DCO sign-off and the required Copilot co-author trailer.

---

### Task 1: Initialize Release Versions

**Files:**
- Modify: `pyproject.toml:5`
- Modify: `skyulf-core/setup.py:10`
- Modify: `frontend/ml-canvas/package.json:4`
- Modify: `frontend/ml-canvas/package-lock.json:3,9`
- Modify: `uv.lock`

**Interfaces:**
- Consumes: Root `[project].version`, core setuptools `version`, and `frontend/ml-canvas/scripts/sync-version.mjs`.
- Produces: Consistent `0.7.5` application/frontend metadata and `0.5.6` core package metadata.

- [ ] **Step 1: Run the desired-version assertion and verify it fails**

Run:

```bash
python - <<'PY'
import json
import re
from pathlib import Path

root = Path("pyproject.toml").read_text()
core = Path("skyulf-core/setup.py").read_text()
package = json.loads(Path("frontend/ml-canvas/package.json").read_text())

assert re.search(r'^version = "0\\.7\\.5"$', root, re.MULTILINE)
assert re.search(r'version="0\\.5\\.6"', core)
assert package["version"] == "0.7.5"
PY
```

Expected: FAIL because the files still contain `0.7.4` and `0.5.5`.

- [ ] **Step 2: Update the root and core version sources**

Change `pyproject.toml`:

```toml
version = "0.7.5"
```

Change `skyulf-core/setup.py`:

```python
version="0.5.6",
```

- [ ] **Step 3: Synchronize frontend npm metadata**

Run:

```bash
cd frontend/ml-canvas && npm run sync-version
```

Expected: `Synced version to 0.7.5 (from pyproject.toml).`

- [ ] **Step 4: Refresh workspace lock metadata**

Run:

```bash
uv lock
```

Expected: the workspace package `skyulf` changes to `0.7.5`; no third-party package versions change.

- [ ] **Step 5: Verify no third-party lock versions changed**

Run:

```bash
python - <<'PY'
import subprocess
import tomllib
from pathlib import Path

before = tomllib.loads(
    subprocess.run(
        ["git", "show", "HEAD:uv.lock"],
        check=True,
        capture_output=True,
        text=True,
    ).stdout
)
after = tomllib.loads(Path("uv.lock").read_text())

def third_party_packages(lock):
    return {
        (package["name"], str(package.get("source", {}))): package
        for package in lock["package"]
        if package["name"] != "skyulf"
    }

assert third_party_packages(before) == third_party_packages(after)
PY
```

Expected: PASS with no output.

- [ ] **Step 6: Run version and metadata verification**

Run:

```bash
python - <<'PY'
import json
import re
from pathlib import Path

root = Path("pyproject.toml").read_text()
core = Path("skyulf-core/setup.py").read_text()
package = json.loads(Path("frontend/ml-canvas/package.json").read_text())
lock = json.loads(Path("frontend/ml-canvas/package-lock.json").read_text())

assert re.search(r'^version = "0\\.7\\.5"$', root, re.MULTILINE)
assert re.search(r'version="0\\.5\\.6"', core)
assert package["version"] == "0.7.5"
assert lock["version"] == "0.7.5"
assert lock["packages"][""]["version"] == "0.7.5"
PY
cd frontend/ml-canvas && npm run check-version
cd ../.. && python skyulf-core/setup.py --version
uv lock --check
```

Expected:

```text
OK — package.json/package-lock.json already match pyproject.toml (0.7.5).
0.5.6
```

`uv lock --check` exits successfully.

- [ ] **Step 7: Run affected repository static checks**

Run:

```bash
source .venv/bin/activate
ruff check .
ruff format --check backend skyulf-core tests run_fastapi.py run_skyulf.py celery_worker.py
ty check backend skyulf-core/skyulf skyulf-core/tests run_fastapi.py run_skyulf.py celery_worker.py
git diff --check
```

Expected: all checks pass with no formatting or type diagnostics.

- [ ] **Step 8: Review the final diff**

Run:

```bash
git diff --stat
git diff -- pyproject.toml skyulf-core/setup.py frontend/ml-canvas/package.json frontend/ml-canvas/package-lock.json uv.lock
```

Expected: only the specified version metadata changes.

- [ ] **Step 9: Commit the version initialization**

Run:

```bash
git add pyproject.toml skyulf-core/setup.py frontend/ml-canvas/package.json frontend/ml-canvas/package-lock.json uv.lock
git commit -s -m "chore: initialize 0.7.5 versions" \
  -m "Set skyulf-core to 0.5.6 and synchronize application, frontend, and lock metadata." \
  -m "Co-authored-by: Copilot <223556219+Copilot@users.noreply.github.com>"
```

Expected: one signed commit containing only the version initialization files.
