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
