# Dual-Engine (Polars / Pandas) Correctness

Investigation into whether Skyulf's dual-engine design is actually correct
end-to-end — engine → preprocessing nodes → modeling → persistence →
inference → monitoring — and whether the documented leakage guarantee holds.

Driven by a direct question: *if a user starts with Polars, can they stay on
Polars the whole way through, converting to numpy or pandas only where that is
genuinely required? And can we honestly claim leakage-proof?*

| Document | What it covers |
|---|---|
| [2026-08-11-audit-findings.md](2026-08-11-audit-findings.md) | All 33 findings (3 CRITICAL, 12 HIGH, 12 MED, 6 LOW) with executable reproductions, an explicit list of what is **proven correct**, and a tiered fix order with target versions. |
| [2026-08-11-leakage-enforcement-plan.md](2026-08-11-leakage-enforcement-plan.md) | Plan to strengthen leakage enforcement so the documented claim becomes true, rather than softening the docs. Includes the required documentation changes. |

## Method

Four parallel Opus-5 audit agents with disjoint scopes (engine-parity,
pandas-purity, leakage, inference/experiments), plus an adversarial
rubber-duck pass instructed to find holes rather than agree. **Every finding
was reproduced with an executable probe**, not inferred from reading code.
Probes ran in throwaway worktrees; repo source was untouched during auditing.

The rubber-duck pass earned its keep: it disproved one previously-accepted
HIGH finding (the "backend engine is pandas-only despite Polars being the
default" claim) and surfaced two real bugs in its place.

## Headline result

The architecture is sound. Polars runs end-to-end — **zero** nodes are
Polars-incapable and **zero** silently downgrade Polars to pandas — and the
Polars → numpy → sklearn handoff produces **bit-identical predictions** to the
pandas path. Pandas users stay 100% pandas.

But six real correctness bugs mean **parity cannot yet be claimed**, and three
CRITICAL inference bugs affect pandas users too — including one where
reordering the JSON keys of a prediction request silently changes the answer.

Every bug found is **pre-existing** and represents a **test coverage gap, not a
regression**: the existing 195-test parity suite passes clean against all of
them, because it exercises nulls but never float `NaN`, and raw frames but
never wrapped frames.

## Status

Investigation complete. **No fixes applied yet** — see the tier table at the
end of the findings document for the recommended order and target versions.
Start with Tier 1 (3 CRITICAL, engine-independent, blocks deployment).
