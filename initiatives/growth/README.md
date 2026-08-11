# Growth

**Status:** Active. This is the plan we are actually executing.

This folder answers one question in one order:

1. **What brings people in?**
2. **What makes them keep using it?**
3. **What makes an enterprise start using it?**

Everything else in `initiatives/` is *research*. This folder is the only
place that says what we do next.

## Entry point

[`2026-08-11-growth-plan.md`](2026-08-11-growth-plan.md) — the plan.

## Operating rules

These exist because the previous planning effort produced 19,172 lines of
documentation and zero shipped fixes, then began contradicting itself. Each
rule below is a direct countermeasure to an observed failure, not a
preference.

### 1. Nothing enters this plan without a verified repro or a measured number

Every item cites either a command that was run and its output, or a metric
with its source. "An audit says X" is not evidence; "I ran X and got Y" is.

*Why:* the research in `enterprise-readiness/` is unusually good, but it was
never re-checked against the code. Re-verification found real bugs that were
**worse** than documented (see plan item T1) and claimed bugs whose stated
root cause was wrong.

### 2. No version numbers in this plan

Versions live in `pyproject.toml`, `frontend/ml-canvas/package.json`, and
`skyulf-core/setup.py`. This plan says "next patch" or "next minor".

*Why:* `initiatives/roadmap/` hardcoded a 19-release version ledger. Two
commits were then spent renumbering prose, and the numbers still did not
match the repository afterwards. A plan that must be maintained in lockstep
with three version files will lose that race every time.

### 3. Horizon is three stages, roughly six to eight weeks

We do not write down what happens after that. When a stage completes, we
rewrite the plan from what we learned.

*Why:* the previous roadmap sequenced R1–R19 plus an enterprise track — well
over two years of work at the actual available budget of 2–3 days per week.
Everything past the first few months was speculation formatted to look like
commitment.

### 4. Every item must serve at least two audiences

The four audiences are: data scientists who already write Python, non-coder
analysts, ML engineers evaluating the tool, and students. Trust-floor items
(correctness) are exempt — they serve everyone by definition.

*Why:* the target is "a bit of all of them," which is only coherent if we
refuse work that serves exactly one.

### 5. We measure before we prioritise past Stage 1

*Why:* there is currently no product analytics of any kind, so nobody can
tell whether the funnel leaks at acquisition, activation, or retention.
Every priority call made without that number is a guess.

### 6. Verify on the branch the work will land on

Findings are re-reproduced on the branch that will carry the fix, not merely
on the branch where they were noticed.

*Why:* three branches are live and none contained the others — `078` (code,
`0.7.8`), `080` (docs, `0.7.7`), and `deploy/demo-mode` (what the public
uses, `0.7.6`, 26 commits behind). Stage 0's bugs were first found on `080`
and had to be re-verified on `078`, which had meanwhile rewritten the very
same functions for Polars/Pandas parity without touching the bug. A finding
verified on the wrong branch is not a finding.

*Resolved for this folder:* `080` was merged into `078` on 2026-08-11, so
the plan now lives on the branch that carries the work. `deploy/demo-mode`
is still behind by design — see the plan's Stage 1 scope decision.

### 7. A claim that something *does not exist* needs stronger proof than a grep

Absence is the hardest thing to verify and the easiest to assert. Before
accepting "there is no X," check the frontend routes, the API surface and
the rendered components — not just whether a well-known library is imported.

*Why:* three separate absence-claims in this repo's planning docs were
false, and each was reached by grepping for the wrong token.

- An agent grepped for MLflow and Weights & Biases, found neither, and
  concluded *"no experiment tracker exists — nothing to audit."* Skyulf has
  a **9,473-line** native experiments subsystem with SHAP, pipeline diff and
  branch comparison. It was missed because it isn't a third-party
  integration.
- A fix to `iqr.py`/`zscore.py` was reported as applied. It was never
  applied (dual-engine F-06, re-verified here).
- The claim that a defect "cannot happen because the guard catches it" held
  for the guard but not for the templates *shipped alongside it*.

The inverse rule also holds: a **positive** claim that something is fixed
requires running it, not reading the commit that says so.

## Relationship to the other initiative folders

| Folder | Disposition |
|---|---|
| `dual-engine-correctness/` | **Promoted.** Its Tier 1 (F-01/F-02/F-03) is part of Stage 0, and it keeps ownership of the version ledger and release notes for those fixes. The only prior doc that met this folder's evidence bar unaided. |
| `enterprise-readiness/` | **Mined, not executed.** Raw material for this plan; items promoted only after re-verification. Its `master-fix-list.md` phases are *not* a schedule. |
| `training-visualization/` | **One slice promoted** (post-fit diagnostics, Stage 3). Rest parked behind the deep-learning work. |
| `code-escape-hatch/` | **One slice promoted** (read-only code view, Stage 3), plus a labelling correction applied to A2.5. Phase C blocked on auth/tenancy. |
| `deep-learning/` | **Nothing actionable.** Two findings recorded in the salvage ledger for if/when it starts. |
| `ray-migration/` | **Nothing.** 6,241 lines, six gated plans, no independently cheap phase. Its own rule blocks it: no measurable benefit has been measured. |
| `roadmap/` | **Not maintained.** Superseded by this folder; its version ledger is known inaccurate. |

Every folder above was read and explicitly dispositioned on 2026-08-11.
Recording "nothing here" is a deliverable: it stops the same 6,000 lines
being re-mined next quarter.
