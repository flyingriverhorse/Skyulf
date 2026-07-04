#!/usr/bin/env bash
# Runs a CI check command, streams its normal output to the job log, and
# appends a single ✅/❌ line to the GitHub Actions Job Summary
# ($GITHUB_STEP_SUMMARY) so the workflow run page shows a scannable
# pass/fail checklist instead of requiring a full log read to see what
# happened. Exit status is preserved so the calling step still fails
# normally when the wrapped command fails.
#
# Usage: run_check.sh "<label>" <command> [args...]
#   e.g. run_check.sh "Ruff lint" ruff check .
set -uo pipefail

label="$1"
shift

"$@"
status=$?

summary="${GITHUB_STEP_SUMMARY:-/dev/null}"
if [ "$status" -eq 0 ]; then
  echo "- ✅ ${label}" >> "$summary"
else
  echo "- ❌ ${label} (exit ${status})" >> "$summary"
fi

exit "$status"
