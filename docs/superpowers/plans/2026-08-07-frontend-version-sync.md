# Frontend Version Sync Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Synchronize the frontend package version with the root Skyulf application version `0.7.4`.

**Architecture:** Keep the root `pyproject.toml` as the single source of truth and use the existing Node.js synchronization script. The script updates both frontend package manifests atomically without changing dependencies or runtime behavior.

**Tech Stack:** Node.js, npm, JSON package manifests, TypeScript, ESLint, Vite, Vitest

## Global Constraints

- The frontend version must follow the root application version `0.7.4`.
- The independent `skyulf-core` version `0.5.5` must not be copied to the frontend.
- Only frontend version fields may change; dependency versions and application behavior must remain unchanged.

---

### Task 1: Synchronize and Validate the Frontend Version

**Files:**
- Modify: `frontend/ml-canvas/package.json:4`
- Modify: `frontend/ml-canvas/package-lock.json`
- Use: `frontend/ml-canvas/scripts/sync-version.mjs`
- Use: `pyproject.toml:5`

**Interfaces:**
- Consumes: root `[project].version = "0.7.4"` from `pyproject.toml`
- Produces: frontend manifest versions equal to `0.7.4`

- [ ] **Step 1: Confirm the version check currently detects drift**

Run:

```bash
cd frontend/ml-canvas
npm run check-version
```

Expected: exit code `1` with a message stating that `pyproject.toml` is
`0.7.4` and the frontend manifests are out of sync.

- [ ] **Step 2: Synchronize both frontend manifests**

Run:

```bash
cd frontend/ml-canvas
npm run sync-version
```

Expected: `Synced version to 0.7.4 (from pyproject.toml).`

- [ ] **Step 3: Inspect the generated manifest diff**

Run:

```bash
git --no-pager diff -- frontend/ml-canvas/package.json frontend/ml-canvas/package-lock.json
```

Expected: only these values change from `0.7.3` to `0.7.4`:

```json
{
  "version": "0.7.4",
  "packages": {
    "": {
      "version": "0.7.4"
    }
  }
}
```

- [ ] **Step 4: Confirm the version drift is resolved**

Run:

```bash
cd frontend/ml-canvas
npm run check-version
```

Expected: exit code `0` with
`OK — package.json/package-lock.json already match pyproject.toml (0.7.4).`

- [ ] **Step 5: Run the frontend static and production checks**

Run:

```bash
cd frontend/ml-canvas
npm run lint
npx tsc --project tsconfig.json --noEmit
npm run build
```

Expected: all commands exit with code `0`.

- [ ] **Step 6: Run the frontend test suite**

Run:

```bash
cd frontend/ml-canvas
npm test
```

Expected: Vitest exits with code `0`.

- [ ] **Step 7: Commit only the synchronized frontend manifests**

Run:

```bash
git add frontend/ml-canvas/package.json frontend/ml-canvas/package-lock.json
git commit -m "chore: sync frontend version to 0.7.4" \
  -m "Co-authored-by: Copilot <223556219+Copilot@users.noreply.github.com>"
```

Expected: a commit containing only the two frontend manifest changes.
