# Coverage reporting investigation — 2026-09-07

## Finding

The reported Sonar new-code coverage was 16.8%, with all listed frontend files
at 0%. The Sonar workflow imported Python Cobertura reports only. The frontend
ran Vitest without coverage, had no installed V8 coverage provider, and configured
neither an LCOV reporter nor `sonar.javascript.lcov.reportPaths`.

Several reported-zero files already had substantial unit coverage. The first
local report measured 100% lines in `nodeSearch`, `nodeDisplayNames`,
`nextNodePosition`, `preprocessingGroups`, and `splitConnections`, 97.3% in
`connectionValidation`, and 97.4% in `ValidationField`.

## Changes

- Added `npm run test:coverage` and the V8 provider matching locked Vitest 4.1.9.
- Generate HTML and LCOV under `frontend/ml-canvas/coverage/`; generated reports
  are ignored by Git. LCOV source names are repository-relative, using native
  platform separators; the Linux CI report uses forward slashes.
- The frontend workflow runs coverage and uploads the report. A token-gated
  frontend coverage job in PR Check supplies its LCOV artifact to Sonar before
  scanning, alongside the existing Python XML reports. Missing frontend artifacts
  fail the job. Sonar now knows the LCOV report path.
- Excluded colocated `.spec` test files from Sonar production sources consistently
  with existing `.test` exclusions. No production coverage exclusions or reduced
  thresholds were added.
- Added 16 frontend behavior tests covering compatible next steps, grouped
  insertion, duplicate/stale/read-only/cancelled connections, drag feedback,
  endpoint information, component browsing/search, and clipboard outcomes.
- Added four Python cases for empty/all-null PII samples and single non-null
  phone samples.

This follows [Sonar's JavaScript/TypeScript coverage import contract](https://docs.sonarsource.com/sonarqube/latest/analysis/test-coverage/javascript-typescript-test-coverage)
and [Vitest's coverage configuration](https://vitest.dev/guide/coverage).

## Local measurements

These are **whole-file line coverage** from the unit test report, not Sonar's
new-code metric. All frontend source files remain included, even if untested.
Overall frontend line coverage increased from 49.39% to 50.43% with the new tests.

| File | Line coverage |
| --- | ---: |
| ConnectionGuidance.tsx | 92.3% |
| ConnectionHoverCard.tsx | 66.7% |
| ConnectionPicker.tsx | 96.1% |
| ConnectionPort.tsx | 100% |
| NodeDetails.tsx | 100% |
| Sidebar.tsx | 94.9% |
| ValidationField.tsx | 97.4% |
| useGraphStore.ts | 70.1% |
| connectionValidation.ts | 97.3% |
| nextNodePosition.ts | 100% |
| nodeDisplayNames.ts | 100% |
| nodeSearch.ts | 100% |
| preprocessingGroups.ts | 100% |
| splitConnections.ts | 100% |

## Remaining gaps and limits

`FlowCanvas`, `CustomEdge`, and `CustomNodeWrapper` still have 0% Vitest coverage.
Their existing Playwright tests exercise browser behavior, but this LCOV report
does not collect Playwright coverage. They need focused component tests or a
separately verified browser coverage import. Other node settings also retain
uncovered branches; the HTML report contains the full inventory.

Python text profiling now has 100% lines and branches in the focused suites.
The clustering crosstab's defensive rejection of noninteger IDs/zero counts
remains uncovered: normal clustering labels are integers and grouped counts
are positive. No artificial mocks or exclusions were added to hide that gap.

The hosted Sonar scan has not been run here. Its final new-code percentage also
depends on its comparison baseline and cannot be inferred from whole-file values.

## Verification

- 937 frontend tests in 118 files passed with coverage enabled.
- After final test-only typing/accessibility cleanup, all eight affected tests
  passed again; production code and measured behavior were unchanged.
- All 308 LCOV source paths resolve to actual repository source files.
- Both YAML workflows parse; the V8 provider and Vitest lock versions match.
- Frontend lint, TypeScript/build, and bundle size checks passed.
- Python profiling: 55 tests passed; clustering: 56 tests passed.
- Changed Python tests passed Ruff, format, and ty checks.
- Existing build chunk warnings remain. No CI workflow was triggered or commit made.
