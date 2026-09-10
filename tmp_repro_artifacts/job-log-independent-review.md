# Independent focused review: job-log numeric regexes

Reviewed the final diff in `JobLogs.tsx`, the new adjacent `JobLogs.test.tsx`,
the new browser test in `e2e/layout-jobs-segmentation.spec.ts`, and the bounded
benchmark/differential script and its recorded results. No tracked files edited.

## Result

No blocking correctness finding in the three-regex change or added tests.

- The guard is after the optional minus in all three expressions, preserving
  negative tokens following digits. A first match cannot start inside a digit
  run: extending left across its preceding digits gives an earlier valid match.
  The new guard therefore preserves first-match index and text.
- Each run is considered a bounded number of times as integer, fraction, or
  exponent. Separators keep their backtracking additive. These three expressions
  have linear search cost; no remaining superlinear path was identified inside
  the guarded expressions.
- The 18 unit cases assert complete rendered message text and actual token
  classes/boundaries. They cover signs, scientific notation, malformed exponents,
  embedded numbers, whitespace, rule precedence, word boundaries, and long runs.
  The long-run unit case checks correctness, not a strict timing bound.
- The browser case renders a 100,000-digit missing-suffix input through the real
  log view, checks all digits and numeric token styling, then changes wrapping
  and checks the resulting class/control state. The selectors agree with the
  component markup and the mock overrides the shared jobs route before navigation.
  This checks completion within Playwright's timeout, not a latency guarantee.
- An independent earlier differential check passed 327,156 input strings across
  all three old/new regex pairs, including exhaustive short strings and seeded
  mixed numeric/Unicode inputs. The parent's script additionally compares all
  tokenizer segments against HEAD for 100,000 messages; its extraction, rule
  indices, and equality assertions are valid. Its final output reports success.
- Recorded focused tests report 61 passed across 2 files. The final benchmark
  records 16,000-digit duration failure at 354.922 ms before and 0.555 ms after.
  These are individual bounded measurements, not stable production benchmarks.
- `git diff --check` for the two tracked changed files passed in this review.
  Full suites and browser execution remain the primary agent's verification.

## Scope and compatibility limits

The surrounding tokenizer still rescans remaining suffixes for each token, so
this change does not establish linear time for arbitrary complete log messages.

Negative lookbehind is already used in `pipelineSummary.ts:8` and TypeScript
targets ES2022. However, Vite's default target includes Safari 14 and lookbehind
support arrived in Safari 16.4 (Apple's Safari 16.4 release notes). Existing usage
supports consistency with the application, but does not prove support on every
default Vite browser target. Do not claim expanded browser compatibility.

Source: https://developer.apple.com/documentation/safari-release-notes/safari-16_4-release-notes
