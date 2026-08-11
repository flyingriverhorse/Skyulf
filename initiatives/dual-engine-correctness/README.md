# Dual-Engine (Polars / Pandas) Correctness

Investigation into whether Skyulf's dual-engine design is actually correct
end-to-end — engine → preprocessing nodes → modeling → persistence →
inference → monitoring — and whether the documented leakage guarantee holds.

Driven by a direct question: *if a user starts with Polars, can they stay on
Polars the whole way through, converting to numpy or pandas only where that is
genuinely required? And can we honestly claim leakage-proof?*

| Document | What it covers |
|---|---|
| [2026-08-11-audit-findings.md](2026-08-11-audit-findings.md) | All 49 findings (4 CRITICAL, 18 HIGH, 17 MED, 10 LOW) with executable reproductions, an explicit list of what is **proven correct**, and a tiered fix order with target versions. |
| [2026-08-11-leakage-enforcement-plan.md](2026-08-11-leakage-enforcement-plan.md) | Plan to strengthen leakage enforcement so the documented claim becomes true, rather than softening the docs. Includes the required documentation changes. |
| [2026-08-11-backend-polars-migration-plan.md](2026-08-11-backend-polars-migration-plan.md) | **Skyulf is presented as Polars-backed, but the backend is 100% pandas** (`backend/data/catalog.py` reads only via `pd.read_csv`/`pd.read_parquet`). Phased plan to close that gap, plus a categorised inventory of every `.to_pandas()` site in `skyulf-core` and which are worth converting. |

## Method

Four parallel Opus-5 audit agents with disjoint scopes (engine-parity,
pandas-purity, leakage, inference/experiments), plus an adversarial
rubber-duck pass instructed to find holes rather than agree. **Every finding
was reproduced with an executable probe**, not inferred from reading code.
Probes ran in throwaway worktrees; repo source was untouched during auditing.

The rubber-duck pass earned its keep: it disproved one previously-accepted
HIGH finding (the "backend engine is pandas-only despite Polars being the
default" claim) and surfaced two real bugs in its place.

It also cuts the other way — one agent wrongly reported the **experiments
subsystem** as non-existent (it grepped for MLflow/W&B, found none, and
stopped). Agent conclusions of the form "this does not exist" are not
trustworthy without independent verification against the frontend routes and
API surface. A fifth agent is auditing that area now; see §6 of the findings.

## Headline result

The architecture is sound. Polars runs end-to-end — of the **100 registered
nodes**, **zero** are Polars-incapable and **zero** silently downgrade Polars
to pandas — and the Polars → numpy → sklearn handoff produces **bit-identical
predictions** to the pandas path. Pandas users stay 100% pandas.

But six real correctness bugs mean **parity cannot yet be claimed**, and three
CRITICAL inference bugs affect pandas users too — including one where
reordering the JSON keys of a prediction request silently changes the answer.

Every bug found is **pre-existing** and represents a **test coverage gap, not a
regression**: the existing 195-test parity suite passes clean against all of
them, because it exercises nulls but never float `NaN`, and raw frames but
never wrapped frames.

## Coverage denominators

The registry holds **100 nodes** (34 Modeling, 30 Preprocessing, 9 Feature
Engineering, 6 Cleaning, 5 Data Operations, 5 Feature Selection, 5 Text,
4 Ensemble, 2 Inspection). Engine-capability and pandas-purity checks covered
all 100. Other figures quoted in the findings — 61 transformers, 38 models,
41 replay tests — are **probed subsets**, not full coverage. Don't quote them
as if they were.

## Status

Investigation complete across engine, nodes, modeling, persistence, inference, monitoring and
experiments.
**No fixes applied yet** — see the tier table at the end of the findings document for the
recommended order and target versions. Start with Tier 1 (3 CRITICAL, engine-independent, blocks
deployment).

The **experiments subsystem** audit is now complete too (§6 of the findings). It was initially
mis-reported as "nothing to audit" — an agent grepped for MLflow/W&B, found none, and stopped.
It turned out to be the **worst-affected layer**: 16 findings, **12 LIVE today**, and mostly not
engine bugs at all but metric-semantics and UI-state defects that make users read the wrong number.
