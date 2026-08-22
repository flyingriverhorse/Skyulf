#!/usr/bin/env bash
# Promote commits from the main line onto deploy/demo-mode safely.
#
# Usage:
#   bash scripts/promote_to_demo.sh <commit> [<commit>...] [--push]
#
# What it does:
#   1. Creates an isolated worktree of deploy/demo-mode (synced with origin)
#   2. Cherry-picks the given commits (with -x provenance)
#   3. HARD GUARD: verifies the demo-specific arrangements survived
#      (upload block, Iris-only datasets, demo_mode config, SlowNodesPage)
#   4. HARD GUARD: refuses version-file changes (the demo never bumps)
#   5. Runs the frontend test suite (known pre-existing failures tolerated)
#   6. Rebuilds the committed static/ml_canvas bundles from demo sources
#   7. Commits the rebuilt bundles; pushes only when --push is given
#
# Aborts (and leaves the worktree for inspection) on any guard failure.
set -euo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
BRANCH="deploy/demo-mode"
WT="$REPO_ROOT/.demo-promote-worktree"
KNOWN_FAILING_TESTS="src/pages/DataSources.test.tsx"
PUSH=0
COMMITS=()

for arg in "$@"; do
  case "$arg" in
    --push) PUSH=1 ;;
    -h|--help) grep '^#' "$0" | sed 's/^# \{0,1\}//'; exit 0 ;;
    *) COMMITS+=("$arg") ;;
  esac
done

if [ "${#COMMITS[@]}" -eq 0 ]; then
  echo "FAIL: no commits given. Usage: bash scripts/promote_to_demo.sh <commit> [...] [--push]"
  exit 2
fi

log() { echo "[promote-to-demo] $*"; }
die() { echo "[promote-to-demo] FAIL: $*"; echo "[promote-to-demo] worktree kept at: $WT"; exit 1; }

# --- demo invariants: the arrangements that must never be lost -------------
check_invariants() {
  local dir="$1"
  local -a checks=(
    "backend/data_ingestion/router.py:_block_in_demo_mode:upload block in demo mode"
    "backend/data_ingestion/router.py:IRIS_SOURCE_ID:Iris-only dataset gating"
    "backend/config/mixins/core.py:DEMO_MODE:demo mode setting"
    "backend/health/routes.py:demo_mode:/api/config demo flag"
    "frontend/ml-canvas/src/pages/SlowNodesPage.tsx:SlowNodesPage:demo-only Slow Nodes page"
    "frontend/ml-canvas/src/core/hooks/useAppConfig.ts:useAppConfig:frontend demo config hook"
  )
  for check in "${checks[@]}"; do
    local file="${check%%:*}"
    local rest="${check#*:}"
    local marker="${rest%%:*}"
    local what="${rest#*:}"
    if [ ! -f "$dir/$file" ] || ! grep -q "$marker" "$dir/$file"; then
      echo "INVARIANT BROKEN: $what ($file no longer contains '$marker')"
      return 1
    fi
  done
  return 0
}

# --- setup ------------------------------------------------------------------
cd "$REPO_ROOT"
git fetch origin "$BRANCH"
[ -d "$WT" ] && git worktree remove --force "$WT" 2>/dev/null || true
git worktree add "$WT" "$BRANCH" >/dev/null
cd "$WT"

LOCAL=$(git rev-parse "$BRANCH")
REMOTE=$(git rev-parse "origin/$BRANCH")
[ "$LOCAL" = "$REMOTE" ] || {
  git reset --hard "origin/$BRANCH" >/dev/null
  log "local $BRANCH was behind origin — synced to origin first"
}
BASE=$(git rev-parse HEAD)

# --- cherry-pick -------------------------------------------------------------
for sha in "${COMMITS[@]}"; do
  log "cherry-picking $sha"
  git cherry-pick -x "$sha" || {
    git cherry-pick --abort 2>/dev/null || true
    die "cherry-pick of $sha conflicts — resolve manually on $BRANCH"
  }
done

# --- guard: demo arrangements intact ----------------------------------------
log "checking demo invariants"
check_invariants "$WT" || die "demo-specific arrangements were damaged — aborting"

# --- guard: no version drift -------------------------------------------------
if git diff "$BASE" --stat -- pyproject.toml skyulf-core/setup.py frontend/ml-canvas/package.json | grep -q .; then
  die "promoted commits change version files — the demo never bumps versions; cherry-pick the code parts only"
fi

# --- frontend tests ----------------------------------------------------------
log "running frontend test suite"
cd "$WT/frontend/ml-canvas"
[ -d node_modules ] || npm ci --silent
TEST_OUT="$(npx vitest run 2>&1 || true)"
echo "$TEST_OUT" | tail -4
NEW_FAILURES="$(echo "$TEST_OUT" | grep -oE '❯ [^ ]+\.tsx? \([0-9]+ tests? \| [0-9]+ failed\)' | awk '{print $2}' | grep -v -F "$KNOWN_FAILING_TESTS" || true)"
[ -z "$NEW_FAILURES" ] || die "new test failures beyond known ones ($KNOWN_FAILING_TESTS): $NEW_FAILURES"

# --- rebuild committed bundles ------------------------------------------------
log "rebuilding static/ml_canvas from demo sources"
npm run build >/dev/null
cd "$WT"
if [ -n "$(git status --porcelain static/ml_canvas)" ]; then
  git add static/ml_canvas
  git commit -m "chore(demo): rebuild frontend bundles for promoted changes" >/dev/null
  log "rebuilt bundles committed"
else
  log "no bundle changes needed"
fi

# --- final invariant pass on what will be pushed ------------------------------
check_invariants "$WT" || die "post-build invariant check failed"

log "result: $(git rev-list --count "$BASE"..HEAD) commit(s) ahead of origin/$BRANCH"
git --no-pager log --oneline "$BASE"..HEAD

if [ "$PUSH" = "1" ]; then
  git push origin "$BRANCH"
  log "pushed to origin/$BRANCH — redeploy the demo host if it does not auto-pull"
else
  log "NOT pushed. Review with: git -C $WT log --oneline origin/$BRANCH..$BRANCH"
  log "Publish with: bash scripts/promote_to_demo.sh --push <same commits> (or push manually from $WT)"
fi
