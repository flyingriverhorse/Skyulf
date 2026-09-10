# Frontend scanner follow-up and complexity interpretation

**Base:** `5a63fe55` on `0819`. **Date:** 2026-09-10.

The user supplied four critical scanner labels, a file-level table of
complexity changes, and then three job-log regex performance warnings.
These are separate questions: a complexity delta does not
establish a security defect, and a scanner label needs a reachable data flow.

## Reading the complexity table

`+39` for `trainingConfig.ts` is a change reported for the file, not a claim that
one function has CCN 39. That helper and `PipelineDiffView.test.tsx` were added
in `5ff4a74b`. Their positive entries include extracted code and new test callbacks;
the parent files may show corresponding reductions. New functions each have a
baseline CCN of one in ESLint, so splitting responsibilities can increase a sum
even when the largest function becomes simpler. Added tests contribute separately.

The external tool's exact calculation and comparison base were not supplied.
Do not equate its `+39` with ESLint's sum, or infer whole-project improvement
from individual positive/negative rows. The required local gate is per-function
CCN <= 10; the informational report remains at 8. Readability, responsibility
boundaries and behavior checks still matter independently of either number.

Local ESLint measurement before and after this follow-up, with an invocation-only
threshold of zero to enumerate every function:

| File | Largest function CCN, before -> after | Function CCN sum, before -> after |
|---|---:|---:|
| `comparisonTable/trainingConfig.ts` | 8 -> 8 | 35 -> 35 |
| `experiments/PipelineDiffView.test.tsx` | 3 -> 3 | 43 -> 43 |
| `core/utils/operationalContext.ts` | 10 -> 10 | 23 -> 22 |
| `operationalContext/recordParsers.ts` | 3 -> 3 | 31 -> 31 |

No function was split merely to address these four scanner patterns. Removing
the redundant kind-check helper accounts for the one-point sum reduction.

Two unchanged files from the supplied table further illustrate the distinction:
`useInferenceController.tsx` (external delta +132) contains 57 functions/callbacks
with local ESLint sum 128 and maximum 8; `EnsembleFormSections.tsx` (delta +81)
contains 38 functions/callbacks with sum 76 and maximum 8. The dashboard delta
and the local per-function maximum are different measurements.

## Four reported patterns

- **Dynamic RegExp in a test:** its interpolated value was the length of a
  static test array, not untrusted pattern text. Replaced it with a string
  substring matcher that still checks the exact parenthesized selection count.
- **Operational parser dispatch:** the original URL kind was checked against
  a fixed whitelist before indexing the parser object. Replaced the runtime
  object dispatch with a Map of explicitly registered parsers. The private mapped
  type still requires all ten kinds and each corresponding reference shape;
  every parser body and the public URL contract are unchanged. Missing/unknown
  kinds return null directly through the registry lookup.
- **CV and tuning formatter lookups:** both now use Map.get with the same
  formatter bodies, lookup order, source precedence, zero/null/empty handling
  and fallback. The UI supplies a fixed list of field labels. Direct helper
  calls with inherited names such as `constructor` and `__proto__` previously
  returned invalid values or threw; they now return the unsupported-field `-`.

The inherited-name helper behavior was reproduced on the original source.
It is useful hardening, but the current UI's fixed labels do not demonstrate
an application command-injection route. No command interpreter or untrusted
regular-expression pattern was found in these paths. The external scanner is
not available in this session; its four issue statuses require a fresh scan.
No scanner rule, complexity limit, dependency or lint exemption was changed.

## Lookup follow-up verification (before the log regex change)

- Original source: 107 passing tests and four expected failures in the new
  direct-helper inherited-name cases. The other three suites passed.
- Final focused selection: **111 tests / 4 files passed**, including all ten
  operational kinds, numeric and optional IDs, comparison rows and pipeline diff.
- Full Vitest: **2,411 tests / 187 files passed**.
- Normal ESLint and the whole-source strict CCN 10 check: **passed**.
- Informational CCN 8: **133 functions / 96 files**, all optional CCN 9/10.
  The inventory's source locations were refreshed from this report.
- `npm.cmd run build`: **passed**, including TypeScript compilation.
  Main asset: `index-BNqQe-D6.js`. All **11** unchanged bundle budgets passed.
- Chromium: **9 tests passed** in `comparison-error-log.spec.ts` and
  `graph-preprocessing-context.spec.ts`, including comparisons and operational
  links after reload at desktop/compact widths. HTTP is mocked.
- Independent source review: **no blocking findings**; supported behavior and
  types preserved. No external clean-scan claim was made.

Expected error-path/jsdom logging occurred in the full unit run; the browser
run logged a WebSocket proxy reset. Both runs completed without failed tests.
Evidence is retained under `tmp_repro_artifacts/security-review-*`, including
the original/final metrics, regression logs and independent review.

Release notes are under v0.8.19. The functional audit queue remains **63 open /
4 parked**, with existing findings and deferred DRIFT-01 unchanged.

## Job-log regex performance follow-up

The three numeric expressions in `jobDetails/JobLogs.tsx` (original lines
118/123/128) had quadratic search behavior on long digit runs without a required
suffix. Each failed start rescanned most of the same digits. All three now use
`-?(?<!\d)\d+...`: the fixed-width guard rejects starts inside a digit run, while
its position after the optional minus preserves cases such as `run1-2s`.
An original first match could not begin inside a digit run because it could
extend left to an earlier valid match. Suffix syntax, match priority, colors
and all log controls remain unchanged; no functions were split or limits raised.

Bounded Node measurements from the same comparison run, for 16,000 digits and
an unmatched suffix (`!`):

| Expression | Original | Guarded |
|---|---:|---:|
| Duration | 354.922 ms | 0.555 ms |
| Percentage | 104.368 ms | 0.856 ms |
| Bare decimal | 107.110 ms | 0.563 ms |

These individual samples illustrate the reproduced issue and are not production
latency guarantees. The script isolates each measurement with a five-second
process timeout. A seeded differential check of **100,000 messages** found
identical match locations/text for all three expressions and identical complete
tokenizer segments. Independent review also checked **327,156 strings** against
the three old/new patterns and found no blocking issue in the final diff/tests.

Fresh verification after the regex change:

- **61 tests / 2 files passed** both before and after the change. The 18 new
  rendering cases cover signs, scientific notation, whitespace, malformed suffixes,
  embedded numbers, priorities and long digit runs without a flaky timing assertion.
- Full Vitest: **2,429 tests / 188 files passed**.
- Chromium: **7 tests passed** in `layout-jobs-segmentation.spec.ts`; the new
  case renders **100,000 digits**, verifies highlighted values and changes wrapping
  through the real job-detail view. HTTP is mocked; this checks completion and
  interaction, not a latency SLA.
- Normal ESLint, source-wide CCN 10 and TypeScript/build passed. CCN 8 still
  reports **133 functions / 96 files**, with unchanged inventory locations.
- Main asset rebuilt as `index-BVvggaqQ.js`; all **11** bundle budgets passed.

Evidence is in `tmp_repro_artifacts/job-log-*`. Expected jsdom/error-path logs
occurred in unit tests; all tests passed. External scanner issue status still
requires a fresh scan.

The linear search claim applies to these three expressions. The surrounding
tokenizer still searches the remaining suffix once per token, so arbitrary
messages with many short tokens can retain quadratic total work. That separate
algorithm was not changed. Lookbehind already exists elsewhere in the app;
[MDN documents its browser support](https://developer.mozilla.org/en-US/docs/Web/JavaScript/Reference/Regular_expressions/Lookbehind_assertion).
This check used Chromium and does not establish support for older browsers.

## Codacy temporary artifact follow-up

Commit `6a9dbbe5` included seven local files from `tmp_repro_artifacts/`, and
Codacy reported 21 findings in `verify_security_followup.py`: 15 assertion
warnings, three subprocess warnings, and one each for list-comprehension style,
missing docstring and import ordering. The script validates a particular pending
diff using local report files and an exact asset hash. It is not reusable CI or
application code; no production or workflow caller was found.

The assertion warning is technically correct for Python optimized mode. The
subprocess calls use argument lists (no shell), fixed Git arguments at the call
sites and no externally supplied input. These findings do not establish 18
application security defects. Their appearance exposed a repository hygiene
mistake: temporary verification evidence was committed with the source changes.

At the user's request, the seven files were deleted from disk and their removals
staged in Git. The root `.gitignore` now ignores `/tmp_repro_artifacts/`, and
`.codacy.yml` excludes that directory, consistent with the existing exclusions
for `temp/` and local
Ruff/ty checks. Production files, tests and dependency scanning remain in their
existing scope. The verified behavior and performance results above remain the
durable record; some referenced scripts/reports were deleted by this cleanup,
and the remaining temporary logs are local evidence only.

Verification confirmed all seven files are absent, none remain tracked, Git
ignores future files in the directory, the Codacy YAML parses with only this
additional exclusion, and staged/unstaged whitespace checks pass. Application,
dependency and built asset files are unchanged; frontend tests were not rerun
for this repository-only cleanup. Codacy must rescan the committed changes
before its dashboard can confirm closure of the 21 artifact findings.

The user also listed one issue in `frontend/ml-canvas/package-lock.json` without
its message. A read-only npm audit completed and returned 14 vulnerable package
entries (2 critical, 3 high, 8 moderate, 1 low); this is a separate tool/result,
not confirmation of Codacy's one finding. No dependency was changed based on an
assumed match. The exact Codacy advisory is requested for a scoped follow-up.

## Dependency audit continuation

This first pass reduced the npm result from 14 to 2 entries. The subsequent
[Plotly peer follow-up](#plotly-peer-dependency-follow-up) resolves those two;
the current installed-tree audit reports zero known vulnerable package entries.

**Baseline:** `988b5a43` on branch `0820`, 2026-09-10. A fresh registry-backed
`npm audit` reproduces **14 vulnerable package entries: 2 critical, 3 high,
8 moderate and 1 low**. These include transitive and parent-package entries,
not fourteen distinct application vulnerabilities. The Codacy lockfile advisory
has not been supplied, so this remains an independently verified npm audit.

The compatible update changes **27 existing lock entries**, without adding or
removing packages. Vitest, its UI and coverage provider resolve together at
4.1.11; the coverage provider retains its exact version pin. Other updates
use existing parent dependency ranges, including query-string's updated range
for its decoder. The main affected versions are:

| Package or family | Before | After |
|---|---|---|
| Vitest, UI, coverage and internal packages | 4.1.9 | 4.1.11 |
| Browserslist / baseline-browser-mapping | 4.28.1 / 2.10.37 | 4.28.9 / 2.11.21 |
| query-string / decode-uri-component | 9.3.1 / 0.4.1 | 9.5.1 / 0.5.0 |
| fflate | 0.8.2 | 0.8.3 |
| js-yaml | 4.3.1 | 4.3.2 |
| nanoid (OC-214) | 3.3.17 | 3.3.18 |
| postcss-selector-parser (OC-215) | 6.1.2 | 6.1.4 |

Both the first-pass lockfile audit and the post-install npm audit report **2 critical,
0 high, 0 moderate and 0 low** entries. Audit still exits 1 because the critical
dependency finding below remains open. Independent review verified all **52**
dependency/peer constraints referencing the updated packages and found no
blocking or important issue.

The two critical entries are one advisory reported for `maplibre-gl` and its
parent `plotly.js`. The installed Plotly 3.3.1 requires MapLibre `^4.7.1`, while
the advisory lists 6.4.1 as patched. Registry inspection of current Plotly 4.1.0
still finds a `^5.24.0` requirement, so a latest-Plotly upgrade alone does not
resolve the advisory. No forced MapLibre major override or Plotly downgrade
is included in the compatible update scope.

`src/core/plotly.ts` uses `plotly.js-gl3d-dist-min` through
`react-plotly.js/factory`; the full Plotly package is installed to satisfy the
wrapper's peer dependency. The reported sanitizer handles map attribution
HTML, and no direct MapLibre import was found in application source. A temporary
read-only build plugin recorded module metadata across **69 emitted chunks**:
zero MapLibre modules and two full-Plotly module records, both with zero rendered
bytes. Production artifacts are byte-identical to HEAD, including
`index-BVvggaqQ.js` and `vendor-plotly-EQ6XkBLq.js`. These observations do not
establish a reachable application exploit; the installed critical dependency
finding remains open. Its follow-up needs an upstream compatible fix or a
separately reviewed change to the unused full-Plotly peer dependency.

Sources: [Vitest advisory](https://github.com/advisories/GHSA-82fw-gwwq-j7x9),
[MapLibre advisory](https://github.com/advisories/GHSA-jrc7-96c5-q579),
[nanoid advisory](https://github.com/advisories/GHSA-2v37-7h3g-55p8), and
[selector-parser patch](https://github.com/postcss/postcss-selector-parser/releases/tag/6.1.3).

Fresh verification:

- Baseline and final full Vitest: **2,429 tests / 188 files passed**. The final
  run used V8 coverage: statements 78.99%, branches 74.42%, functions 78.59%,
  lines 79.96%; the HTML and LCOV reporters completed.
- Normal ESLint, source-wide CCN 10, TypeScript/production build and all
  **11** unchanged bundle budgets passed.
- A clean `npm ci --ignore-scripts --no-audit --no-fund` installation in a
  separate temporary directory passed with the same manifest and lockfile hash.
  The main installed tree also resolves the updated versions successfully.
- **17 Chromium tests passed** across `ccn10-inspection-data-analysis.spec.ts`,
  `experiment-results-ccn.spec.ts`, `comparison-error-log.spec.ts` and
  `layout-jobs-segmentation.spec.ts`, using a fresh Vite server on port 5187.
  HTTP is mocked; data preview, chart interactions, filtering, keyboard
  navigation and segmentation submission use the real frontend.
- A production-preview run additionally passed 15 of those scenarios. The
  remaining two require `window.__skyulfTest`, deliberately removed from
  production, and passed in the final development-server run. The temporary
  preview configuration initially needed an explicit frontend working directory.

The first sandboxed Vitest attempt could not load the esbuild config because
an ancestor directory was unreadable; rerunning with the required filesystem
access passed. In-place `npm ci` was blocked by the user's running Vite server
locking esbuild.exe. `npm install --ignore-scripts` completed the main tree,
with cleanup warnings for locked old native files; the separate clean install
verified reproducibility without stopping that server. Expected error-path
and jsdom logging occurred in the passing unit run.

OC-214 and OC-215 are closed in the audit archive; the live queue now contains
**61 open / 4 parked** findings. The other audit priorities are unchanged.
Release notes are under v0.8.20. Local evidence is under ignored
`tmp_repro_artifacts/frontend-deps-*` and `frontend-npm-audit-2026-09-10-*`;
none is intended for version control. Codacy must rescan before its own
lockfile finding can be confirmed closed or matched to an npm advisory.

## Plotly peer dependency follow-up

**2026-09-10, second pass:** the first-pass dependencies are now recorded in
`b8bca3e2`. The remaining MapLibre advisory was brought in by full Plotly's
automatic peer installation, while application rendering already used the
official GL3D distribution through the React wrapper's factory.

The dependency is now `"plotly.js": "npm:plotly.js-gl3d-dist-min@^3.5.0"`.
This uses [documented npm alias syntax](https://docs.npmjs.com/cli/v11/using-npm/package-spec/#aliases)
and the wrapper's [supported custom-bundle factory](https://github.com/plotly/react-plotly.js#customizing-the-plotlyjs-bundle).
The installed GL3D **3.5.0 version, tarball URL and integrity are unchanged**.
There is no forced MapLibre major override or dependency-scanner exclusion.

The relevant file responsibilities are:

| File | Change |
|---|---|
| `package.json` / `package-lock.json` | Satisfy the wrapper peer with the existing GL3D distribution; remove full Plotly's unused dependency tree. |
| `src/core/plotly.ts` | Import the alias and continue sharing one instance between rendering and export. |
| `vite.config.ts` | Group the alias and factory entry together; avoid the default wrapper's full-package import. |
| `src/vite-env.d.ts` | Remove the obsolete slim-package declaration; use the existing Plotly types. |
| `e2e/plotly-bundle.spec.ts` | Exercise real 3D traces, PNG export and return to 2D. |
| `README.md` | Document the alias, factory requirement and future bundle changes. |

Compared with the first pass, **253 lock entries are removed and none added**.
The retained Plotly entry points to GL3D; nine other entries only change their
development classification. Independent review checked **2,425 dependency
references**, with no missing required dependency or semver violation.

Fresh final verification:

- Main-tree `npm audit`: **0 known vulnerable package entries**, exit 0;
  MapLibre is absent from the lockfile and installed tree. Across both passes
  the result is **14 -> 2 -> 0**. This reflects npm's package/advisory matching,
  not a separate security audit of code embedded in prebuilt distributions.
- Clean isolated `npm ci --ignore-scripts --no-audit --no-fund`: passed;
  the lockfile hash matches the main tree, and `npm ls` confirms the React
  wrapper uses the aliased GL3D peer without a second Plotly installation.
- Full Vitest: **2,429 tests / 188 files passed**. ESLint, source-wide CCN 10,
  explicit lint of the new browser test, TypeScript/build and all **11**
  unchanged bundle budgets passed.
- **18 Chromium tests passed** on a fresh development server, including the
  four previously listed specs and the new Plotly regression. That regression
  also passed before the alias change, preserving an observed working baseline.
- The new regression additionally passed against the **built production
  preview**, covering Vite chunking: 3D trace values match the fixture, the
  downloaded PNG has the expected signature and 1200 x 880 dimensions, and
  switching back to 2D restores the canvas. The exported production image was
  visually checked and shows the expected axes and four points.
- Production assets remain byte-identical to HEAD; the dependency change adds
  no browser payload. Independent review found no blocking implementation issue.

The Plotly/MapLibre npm follow-up is closed. The live audit queue remains
**61 open / 4 parked**, and Codacy still needs to rescan before its separately
reported lockfile finding can be confirmed closed. Local logs and isolated-install
evidence are under ignored
`tmp_repro_artifacts/plotly-peer-*`; release notes are under v0.8.20.
