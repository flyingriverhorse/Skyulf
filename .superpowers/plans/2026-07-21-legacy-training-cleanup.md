# Legacy Basic Training / Advanced Tuning Cleanup Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Fully remove the legacy `basic_training`/`advanced_tuning` step_type duplication (old hidden canvas nodes, settings components, and all backward-compat alias-checking code) from the Skyulf ML platform, and rename the still-active `job_type` API dispatch concept from `"basic_training"/"advanced_tuning"` to `"training"/"tuning"` so nothing in the codebase reads as "legacy" anymore. This is a dev-only environment — all existing saved canvases/DB rows may be wiped; there is no backward-compatibility requirement going forward.

**Architecture:** This is a two-layer rename+removal, done backend-first then frontend, because the frontend's `pipelineConverter.ts` must emit values the backend already accepts before the backend stops recognizing the old ones:
1. **Backend job_type rename** (Task 1): `job_type: "basic_training"|"advanced_tuning"|"preview"` → `job_type: "training"|"tuning"|"preview"` everywhere it's constructed/consumed/dispatched. This value is a *separate* concept from a node's `step_type` — it just tags whether a job is a fixed single run or a hyperparameter search — and was never itself dead code, only confusingly named.
2. **Backend step_type removal** (Task 2): delete the `StepType.BASIC_TRAINING`/`StepType.ADVANCED_TUNING` enum members entirely and strip every "treat legacy step_type as equivalent to canonical `training`" branch across ~10 files. Only `StepType.TRAINING` remains as a valid training-node step_type.
3. **Frontend canonical emission** (Task 4): two canvas nodes — `SegmentationNode` and `EnsembleNode` — currently emit the **literal legacy `step_type`** (`basic_training`/`advanced_tuning`) as their real, primary wire format (not just back-compat!). They must be migrated to emit the canonical `step_type: "training"` + `params.run_mode` shape first, matching the pattern `ClassificationNode`/`RegressionNode`/`TextClassificationNode` already use, before the backend can safely drop the old enum members.
4. **Frontend legacy node/component deletion** (Task 5) and **definitionType alias-check removal** (Task 6): delete the old hidden `BasicTrainingNode`/`AdvancedTuningNode`/generic `TrainingNode` + their settings components, and strip every `definitionType === 'basic_training'`-style check. Along the way, fix a **real, currently-existing bug** discovered during investigation: several of these definitionType allow-lists (`EXECUTION_MODE_AWARE_TYPES`, `useBranchColors`'s `MODEL_SOURCE_TYPES`/`TERMINAL_TYPES`, `useGraphStore`'s connection-validation lists, `useRunControls`'s `TRAINING_TYPES`, `perfThresholds.getPerfFamily`) were **never updated** when the canonical `Classification`/`Regression`/`TextClassification`/`Training` node types were introduced — they still only recognize the legacy `'basic_training'`/`'advanced_tuning'` definitionTypes, so today's canonical nodes silently fall through (no execution_mode toggle, no branch coloring, wrong perf bucketing, no connection-validation warning). This plan fixes that gap as part of the same edit, since it's the same lines.
5. **Frontend job_type rename** (Task 7): mirrors Task 1 on the frontend — `JobInfo.job_type`, `RunPipelineRequest.job_type`, and every UI branch reading it (`JobCard.tsx`, `JobDetailsView.tsx`, `jobMeta.ts`, `JobListSidebar.tsx`, `useTrainingNodeContext.ts`, `useRunControls.ts`, `TrainingSettings.tsx`, `EnsembleSettings.tsx`).
6. **Test updates** (Task 3 backend, Task 8 frontend) and **final verification** (Task 9): wipe the dev DB, run both full suites, do a real end-to-end smoke test exercising every affected node type (Classification/Regression fixed+tuned, Segmentation, Ensemble Voting/Stacking fixed+tuned), then update the changelog.

**Tech Stack:** Python 3.11+ FastAPI backend (`backend/ml_pipeline`), pytest; React + TypeScript frontend (`frontend/ml-canvas`), Vitest + `tsc --noEmit`.

## Global Constraints

- This is a dev-only environment. No data migration or backward-compatibility shim is required — the dev DB and any saved canvases may be wiped/regenerated freely.
- The **target job_type values are `"training"` and `"tuning"`** (NOT `"fixed"`/`"tuned"`) — this matches an already-partially-existing convention in the codebase (`ModelRegistryService.get_next_version(..., "tuning")` in `advanced_tuning_manager.py:44`, and `_internal/_routers/jobs.py:115`'s `Literal["training", "tuning"]` query filter, and `_execution/jobs.py:194,196`'s `job_type in ["basic_training", "training"]` / `["advanced_tuning", "tuning"]` acceptance). Do not invent a different pair of strings.
- `job_type` and `run_mode` remain **two separate fields** with two separate value sets: `run_mode` stays `"fixed"`/`"tuned"` (DB column, already correct, untouched by this plan) and `job_type` becomes `"training"`/`"tuning"`/`"preview"` (API/dispatch-only field, changed by this plan). Do not conflate them or try to unify further.
- After this plan, `StepType` (backend `constants.py` and frontend `core/constants/stepTypes.ts`) has exactly 3 members: `DATA_LOADER`, `FEATURE_ENGINEERING`, `TRAINING`. No `BASIC_TRAINING`/`ADVANCED_TUNING` members remain anywhere.
- Every place that currently branches on `definitionType`/`step_type` being `'basic_training'`/`'advanced_tuning'`/`'model_training'`/`'hyperparameter_tuning'` must be updated to recognize the canonical set instead: `'training'`, `'classification'`, `'regression'`, `'text_classification'` (and, where the check is specifically about "is this a trainable/mode-aware node", also add these canonical types even if the current code doesn't reference `'model_training'`/`'hyperparameter_tuning'` legacy variants — see the constant lists this touches in Task 6).
- Full verification after every task: backend `cd /Users/BH7043/Skyulf && python -m pytest tests -q` (record pass count), frontend `cd frontend/ml-canvas && npx tsc --noEmit && npx vitest run` (record pass count). Do not proceed to the next task with a red suite.
- Every commit message follows the existing repo convention seen in prior commits on this branch: `type(scope): summary` (e.g. `refactor(jobs): rename job_type dispatch values to training/tuning`).
- **Execution order note (post rubber-duck review):** execute **Task 4 before Task 2** (i.e. run tasks in the order 1, 4, 2, 3, 5, 6, 7, 8, 9). Task 4 (migrating `SegmentationNode`/`EnsembleNode` to canonical emission) has no real dependency on Task 2 — the backend already accepts canonical `step_type: "training"` today — so doing Task 4 first closes the window where those two nodes would emit a `step_type` the backend has already stopped accepting. Task numbers/headings below are left as originally written for readability; only the *execution order* changes.
- **Do not touch historical DB-migration table names.** `backend/database/engine.py` contains schema-migration deltas referencing old table names `basic_training_jobs`/`advanced_tuning_jobs`. These are migration history, not `step_type`/`job_type` values — leave them exactly as-is in Task 3's sweep; renaming them would corrupt migration history.
- **skyulf-core is out of scope.** The SDK source has zero legacy `step_type`/`job_type` references. Two test function names in `skyulf-core/tests/test_ensemble.py` (`test_advanced_tuning_runs_*`) reference the concept in their names only (they test `run_mode` behavior, not the enum) — leave them. Optionally run `pytest skyulf-core/tests -q` once in Task 9 as a confirmation that it's unaffected.
- **"Zero legacy references" (Tasks 3/8/9) means zero in live source paths**, not literally zero on disk. Known, acceptable exceptions: the `engine.py` migration table names above, the two `skyulf-core/tests` function names above, and the stale compiled bundle (see below) until it is rebuilt.

---

### Task 1: Backend — rename `job_type` dispatch values to `"training"`/`"tuning"`

**Files:**
- Modify: `backend/ml_pipeline/_internal/_schemas.py:41-42` (`PipelineConfigModel.job_type` default)
- Modify: `backend/ml_pipeline/_internal/_routers/run_pipeline.py:121-146` (`_resolve_branch_target_node_id`, `_resolve_model_and_job_type`)
- Modify: `backend/ml_pipeline/_execution/jobs.py:34-77` (`create_job`), `:182-224` (`list_jobs`)
- Modify: `backend/ml_pipeline/_execution/schemas.py:26` (`JobInfo.job_type` Literal)
- Modify: `backend/ml_pipeline/_execution/strategies.py:163-192` (`JobStrategyFactory`)
- Modify: `backend/ml_pipeline/_execution/basic_training_manager.py:98`
- Modify: `backend/ml_pipeline/_execution/advanced_tuning_manager.py:97`
- Test: `tests/test_job_manager_base.py`, `tests/test_backend_strategies.py` (update fixture/assertion strings if they reference `"basic_training"`/`"advanced_tuning"` as a job_type value — check both files for `job_type=` or `_strategies[` usages)

**Interfaces:**
- Consumes: nothing from other tasks (this is the first task).
- Produces: `job_type` values are now the plain strings `"training"`, `"tuning"`, `"preview"` everywhere in the backend. `JobStrategyFactory._strategies` is now keyed by the plain strings `"training"`/`"tuning"` (not `StepType.BASIC_TRAINING`/`StepType.ADVANCED_TUNING`). Task 2 will remove the `StepType.BASIC_TRAINING`/`ADVANCED_TUNING` enum members entirely — nothing in this task's changes may still reference them once Task 1 is done (this task must stop using those enum members even though they still technically exist until Task 2 runs, so Task 2 has zero remaining references to update in these specific files).

- [ ] **Step 1: `_internal/_schemas.py` — change the job_type default**

Change:
```python
class PipelineConfigModel(BaseModel):
    pipeline_id: str
    nodes: list[NodeConfigModel]
    metadata: dict[str, Any] = {}
    target_node_id: str | None = None
    # "basic_training", "advanced_tuning", or "preview".
    job_type: str | None = StepType.BASIC_TRAINING
```
to:
```python
class PipelineConfigModel(BaseModel):
    pipeline_id: str
    nodes: list[NodeConfigModel]
    metadata: dict[str, Any] = {}
    target_node_id: str | None = None
    # "training", "tuning", or "preview".
    job_type: str | None = "training"
```
Remove the now-unused `from backend.ml_pipeline.constants import StepType` import at the top of this file if `StepType` is no longer referenced anywhere else in it (check with `grep -n StepType backend/ml_pipeline/_internal/_schemas.py`).

- [ ] **Step 2: `_internal/_routers/run_pipeline.py` — `_resolve_model_and_job_type`**

Change the function body (lines ~121-146) from:
```python
def _resolve_model_and_job_type(
    sub: PipelineConfig, target_node_id: str | None, requested_job_type: Any
) -> tuple[str, Any]:
    """Determine model type and job type from the sub-pipeline's terminal node."""
    model_type = "unknown"
    job_type = requested_job_type or StepType.BASIC_TRAINING
    for n in sub.nodes:
        if n.node_id != target_node_id:
            continue
        if n.step_type == StepType.BASIC_TRAINING:
            model_type = n.params.get("model_type", n.params.get("algorithm", "unknown"))
            job_type = StepType.BASIC_TRAINING
        elif n.step_type == StepType.ADVANCED_TUNING:
            model_type = n.params.get("algorithm", n.params.get("model_type", "unknown"))
            job_type = StepType.ADVANCED_TUNING
        elif n.step_type == StepType.TRAINING:
            model_type = n.params.get("algorithm", n.params.get("model_type", "unknown"))
            job_type = (
                StepType.ADVANCED_TUNING
                if n.params.get("run_mode", "fixed") == "tuned"
                else StepType.BASIC_TRAINING
            )
        elif n.step_type == "data_preview":
            model_type = "preview"
            job_type = "preview"
        break
    return model_type, job_type
```
to:
```python
def _resolve_model_and_job_type(
    sub: PipelineConfig, target_node_id: str | None, requested_job_type: Any
) -> tuple[str, Any]:
    """Determine model type and job type from the sub-pipeline's terminal node."""
    model_type = "unknown"
    job_type = requested_job_type or "training"
    for n in sub.nodes:
        if n.node_id != target_node_id:
            continue
        if n.step_type == StepType.TRAINING:
            model_type = n.params.get("algorithm", n.params.get("model_type", "unknown"))
            job_type = "tuning" if n.params.get("run_mode", "fixed") == "tuned" else "training"
        elif n.step_type == "data_preview":
            model_type = "preview"
            job_type = "preview"
        break
    return model_type, job_type
```
(Leave `_resolve_branch_target_node_id`'s `terminal_types` set alone in this task — that set still references `StepType.BASIC_TRAINING`/`ADVANCED_TUNING` and is handled in Task 2, since it's a step_type check, not a job_type value.)

- [ ] **Step 3: `_execution/jobs.py` — `create_job` and `list_jobs`**

In `create_job` (lines ~34-77), change the `Literal` type and the three branches:
```python
        job_type: Literal["basic_training", "advanced_tuning", "preview"],
```
→
```python
        job_type: Literal["training", "tuning", "preview"],
```
and:
```python
        if job_type == "basic_training":
```
→
```python
        if job_type == "training":
```
and:
```python
        elif job_type == "advanced_tuning":
```
→
```python
        elif job_type == "tuning":
```
(the `elif job_type == "preview":` branch is unchanged).

In `list_jobs` (lines ~182-224), change:
```python
        if job_type in ["basic_training", "training"]:
            jobs = await BasicTrainingManager.list_training_jobs(session, limit, skip)
        elif job_type in ["advanced_tuning", "tuning"]:
            jobs = await AdvancedTuningManager.list_tuning_jobs(session, limit, skip)
```
→
```python
        if job_type == "training":
            jobs = await BasicTrainingManager.list_training_jobs(session, limit, skip)
        elif job_type == "tuning":
            jobs = await AdvancedTuningManager.list_tuning_jobs(session, limit, skip)
```

- [ ] **Step 4: `_execution/schemas.py` — `JobInfo.job_type` Literal**

Change:
```python
    job_type: Literal["training", "tuning", "preview", "basic_training", "advanced_tuning"]
```
to:
```python
    job_type: Literal["training", "tuning", "preview"]
```

- [ ] **Step 5: `_execution/strategies.py` — re-key `JobStrategyFactory`**

Change:
```python
class JobStrategyFactory:
    _strategies: dict[str, JobStrategy] = {
        StepType.BASIC_TRAINING: BasicTrainingStrategy(),
        StepType.ADVANCED_TUNING: AdvancedTuningStrategy(),
        # Add more strategies here as needed
    }

    @classmethod
    def get_strategy_by_job(cls, job: MLJob) -> JobStrategy:
        run_mode = getattr(job, "run_mode", None)
        if run_mode == "fixed":
            return cls._strategies[StepType.BASIC_TRAINING]
        elif run_mode == "tuned":
            return cls._strategies[StepType.ADVANCED_TUNING]
        else:
            raise ValueError(f"Unknown job run_mode: {run_mode!r} (job type: {type(job)})")
```
to:
```python
class JobStrategyFactory:
    _strategies: dict[str, JobStrategy] = {
        "training": BasicTrainingStrategy(),
        "tuning": AdvancedTuningStrategy(),
        # Add more strategies here as needed
    }

    @classmethod
    def get_strategy_by_job(cls, job: MLJob) -> JobStrategy:
        run_mode = getattr(job, "run_mode", None)
        if run_mode == "fixed":
            return cls._strategies["training"]
        elif run_mode == "tuned":
            return cls._strategies["tuning"]
        else:
            raise ValueError(f"Unknown job run_mode: {run_mode!r} (job type: {type(job)})")
```
Remove the `StepType` import at the top of `strategies.py` if it becomes unused after this change (check with `grep -n StepType backend/ml_pipeline/_execution/strategies.py`).

- [ ] **Step 6: job managers — `job_type=` value**

In `basic_training_manager.py:98`, change:
```python
            job_type=StepType.BASIC_TRAINING.value,
```
to:
```python
            job_type="training",
```
In `advanced_tuning_manager.py:97`, change:
```python
            job_type=StepType.ADVANCED_TUNING.value,
```
to:
```python
            job_type="tuning",
```
Remove the now-unused `from backend.ml_pipeline.constants import StepType` import from each file if `StepType` is no longer referenced elsewhere in it (check both files with `grep -n StepType`).

- [ ] **Step 7: Update backend tests referencing the old job_type strings**

Run:
```bash
cd /Users/BH7043/Skyulf && grep -rn '"basic_training"\|"advanced_tuning"' tests/ | grep -i job_type
```
For every match, update the fixture/assertion to use `"training"`/`"tuning"` instead. Also check `tests/test_job_manager_base.py` and `tests/test_backend_strategies.py` specifically for any `_strategies[StepType.BASIC_TRAINING]`-style lookups and update them to `_strategies["training"]` (or `["tuning"]`).

- [ ] **Step 8: Run the full backend test suite**

```bash
cd /Users/BH7043/Skyulf && python -m pytest tests -q
```
Expected: all tests pass (record the pass count — should match the pre-task baseline count, since this is a pure rename with no behavior change).

- [ ] **Step 9: Commit**

```bash
cd /Users/BH7043/Skyulf && git add backend/ && git commit -m "refactor(jobs): rename job_type dispatch values from basic_training/advanced_tuning to training/tuning

Co-authored-by: Copilot <223556219+Copilot@users.noreply.github.com>"
```

---

### Task 2: Backend — remove legacy `StepType.BASIC_TRAINING`/`ADVANCED_TUNING` enum members and all step_type alias-checks

**Files:**
- Modify: `backend/ml_pipeline/constants.py` (remove the 2 enum members + comment)
- Modify: `backend/ml_pipeline/_internal/_routers/run_pipeline.py:108-113` (`_resolve_branch_target_node_id`'s `terminal_types` set)
- Modify: `backend/ml_pipeline/_internal/_routers/preview.py:125-148` (`_pick_target_node_id`), `:428` (`_group_preview_subs`), `:449` (`_branch_terminal_group_key`)
- Modify: `backend/ml_pipeline/_internal/_helpers.py:30-54` (`_resolve_branch_context`)
- Modify: `backend/ml_pipeline/_execution/graph_utils.py:10` (`TERMINAL_STEP_TYPES`), `:552-553`ish (`extract_job_details`'s `ntype in [...]` list — grep for it, exact line may have shifted)
- Modify: `backend/ml_pipeline/_execution/_schema_validator.py:44-58` (`_OPTIONAL_PARAM_KEYS`)
- Modify: `backend/ml_pipeline/_execution/_schema_graph.py:33-38` (`_PASSTHROUGH_STEP_TYPES`)
- Modify: `backend/ml_pipeline/_execution/engine/_node_runners.py:403-417` (`_resolve_run_mode`)
- Modify: `backend/ml_pipeline/_execution/engine/__init__.py:75-77` (`_pipeline_has_training_node`), `:248` (`_dispatch_node`)
- Modify: `backend/ml_pipeline/_internal/_routers/notebook_export.py:61` (`_MODELING_STEPS`)
- Modify: `backend/ml_pipeline/_internal/_routers/_notebook_branched.py:63-70` (`_SPLIT_OR_MODEL`)
- Modify: `backend/ml_pipeline/_internal/_routers/_notebook_builders.py:161-166` (`_is_tuning_model`)
- Test: run full backend suite; fix any test that constructs a node with `step_type="basic_training"`/`"advanced_tuning"` (grep for these string literals across root `tests/` (NOT `backend/tests/` — that directory does not exist; the real suite lives at repo-root `tests/`, run via `pytest tests -q` per `pyproject.toml`))

**Interfaces:**
- Consumes: Task 1 must be complete and committed first (this task assumes `job_type` is already `"training"`/`"tuning"` everywhere, so no file in this task's scope still needs a `job_type` value change — only `step_type` checks remain).
- Produces: `StepType` enum has exactly 3 members (`DATA_LOADER`, `FEATURE_ENGINEERING`, `TRAINING`). Every "is this node a training/tuning terminal" check across the backend recognizes only `StepType.TRAINING` (plus `"data_preview"` where that was already also checked). Task 4 (frontend) depends on this: once this task lands, the backend no longer accepts `basic_training`/`advanced_tuning` as a step_type at all, so `pipelineConverter.ts`'s `SegmentationNode`/`EnsembleNode` branches (fixed in Task 4) MUST be migrated in the same overall effort — but since backend and frontend are independently deployed/tested here, land this task, verify backend tests green, then move immediately to Task 4 before any manual end-to-end use of Segmentation/Ensemble.

- [ ] **Step 1: `constants.py` — remove the enum members**

Change:
```python
from enum import StrEnum


class StepType(StrEnum):
    DATA_LOADER = "data_loader"
    FEATURE_ENGINEERING = "feature_engineering"
    TRAINING = "training"

    # Legacy/Aliases — retained (not removed) so already-saved pipeline JSON /
    # job rows that still reference these values keep loading and executing
    # unchanged. New pipelines only ever write ``TRAINING`` + a ``run_mode``
    # param; the dispatcher normalizes these two old values to that shape on
    # the way in (see ``PipelineEngine._resolve_run_mode``). Do not delete —
    # same precedent as the (now-removed) MODEL_TRAINING/MODEL_TUNING aliases.
    BASIC_TRAINING = "basic_training"
    ADVANCED_TUNING = "advanced_tuning"
```
to:
```python
from enum import StrEnum


class StepType(StrEnum):
    DATA_LOADER = "data_loader"
    FEATURE_ENGINEERING = "feature_engineering"
    TRAINING = "training"
```

- [ ] **Step 2: `run_pipeline.py` — trim `terminal_types`**

Change:
```python
    terminal_types = {
        StepType.BASIC_TRAINING,
        StepType.ADVANCED_TUNING,
        StepType.TRAINING,
        "data_preview",
    }
```
to:
```python
    terminal_types = {
        StepType.TRAINING,
        "data_preview",
    }
```

- [ ] **Step 3: `preview.py` — trim 3 checks**

In `_pick_target_node_id` (~line 137-144), change:
```python
    if (
        target.step_type
        in [
            StepType.BASIC_TRAINING,
            StepType.ADVANCED_TUNING,
            StepType.TRAINING,
        ]
        and target.inputs
    ):
```
to:
```python
    if target.step_type == StepType.TRAINING and target.inputs:
```

In `_group_preview_subs` (~line 428), change:
```python
    training_types = {StepType.BASIC_TRAINING, StepType.ADVANCED_TUNING, StepType.TRAINING}
```
to:
```python
    training_types = {StepType.TRAINING}
```

In `_branch_terminal_group_key` (~line 449), change:
```python
    if leaf.step_type in {StepType.BASIC_TRAINING, StepType.ADVANCED_TUNING, StepType.TRAINING}:
```
to:
```python
    if leaf.step_type == StepType.TRAINING:
```

- [ ] **Step 4: `_helpers.py` — trim `_resolve_branch_context`**

Change:
```python
    if sub_config.nodes:
        leaf = sub_config.nodes[-1]
        if leaf.step_type in {StepType.BASIC_TRAINING, StepType.ADVANCED_TUNING, StepType.TRAINING}:
            # Only scan for model_type when the terminal IS a training/tuning node.
            for n in sub_config.nodes:
                if n.step_type in {
                    StepType.BASIC_TRAINING,
                    StepType.ADVANCED_TUNING,
                    StepType.TRAINING,
                }:
                    model_type = n.params.get("model_type") or n.params.get("algorithm") or ""
```
to:
```python
    if sub_config.nodes:
        leaf = sub_config.nodes[-1]
        if leaf.step_type == StepType.TRAINING:
            # Only scan for model_type when the terminal IS a training node.
            for n in sub_config.nodes:
                if n.step_type == StepType.TRAINING:
                    model_type = n.params.get("model_type") or n.params.get("algorithm") or ""
```

- [ ] **Step 5: `graph_utils.py` — trim `TERMINAL_STEP_TYPES` and `extract_job_details`**

Change:
```python
TERMINAL_STEP_TYPES = {"basic_training", "advanced_tuning", "training"}
```
to:
```python
TERMINAL_STEP_TYPES = {"training"}
```
Then find the `ntype in [...]` list inside `extract_job_details` (grep `grep -n 'StepType.BASIC_TRAINING' backend/ml_pipeline/_execution/graph_utils.py` to get the exact current line) and remove the `StepType.BASIC_TRAINING`, `StepType.ADVANCED_TUNING`, and `"hyperparameter_tuning"` entries from it, keeping `"train_test_split"`, `"TrainTestSplitter"`, `"feature_target_split"`, and `StepType.TRAINING`.

- [ ] **Step 6: `_schema_validator.py` — trim `_OPTIONAL_PARAM_KEYS`**

Change:
```python
    "basic_training": {"target_column"},
    "advanced_tuning": {"target_column"},
    "training": {"target_column"},
```
to:
```python
    "training": {"target_column"},
```
Update the comment above the dict (currently says "basic_training / advanced_tuning: target_column is metadata...") to say "training: target_column is metadata that tells the training strategy which column to use as y. ..." (drop the legacy names from the prose).

- [ ] **Step 7: `_schema_graph.py` — trim `_PASSTHROUGH_STEP_TYPES`**

Change:
```python
_PASSTHROUGH_STEP_TYPES = {
    "basic_training",
    "advanced_tuning",
    "training",
    "data_preview",
}
```
to:
```python
_PASSTHROUGH_STEP_TYPES = {
    "training",
    "data_preview",
}
```

- [ ] **Step 8: `engine/_node_runners.py` — simplify `_resolve_run_mode`**

Change:
```python
    def _resolve_run_mode(self, node: NodeConfig) -> str:
        """Derive ``'fixed'`` (plain, single hyperparameter set) vs ``'tuned'``
        (hyperparameter search) for this node.

        New pipelines set an explicit ``run_mode`` param on a
        ``StepType.TRAINING`` node. Old saved pipelines / job rows still carry
        the legacy ``basic_training``/``advanced_tuning`` step types —
        normalize those to the equivalent ``run_mode`` here so they keep
        executing unchanged (Phase 2b backward-compat shim).
        """
        if node.step_type == StepType.BASIC_TRAINING:
            return "fixed"
        if node.step_type == StepType.ADVANCED_TUNING:
            return "tuned"
        return node.params.get("run_mode", "fixed")
```
to:
```python
    def _resolve_run_mode(self, node: NodeConfig) -> str:
        """Derive ``'fixed'`` (plain, single hyperparameter set) vs ``'tuned'``
        (hyperparameter search) for this ``StepType.TRAINING`` node, from its
        ``run_mode`` param.
        """
        return node.params.get("run_mode", "fixed")
```

- [ ] **Step 9: `engine/__init__.py` — trim 2 checks**

Change:
```python
    def _pipeline_has_training_node(self) -> bool:
        """Checks if the current pipeline workflow includes a model training step."""
        return any(
            node.step_type in [StepType.BASIC_TRAINING, StepType.ADVANCED_TUNING, StepType.TRAINING]
            for node in self._node_configs.values()
        )
```
to:
```python
    def _pipeline_has_training_node(self) -> bool:
        """Checks if the current pipeline workflow includes a model training step."""
        return any(
            node.step_type == StepType.TRAINING for node in self._node_configs.values()
        )
```
Change:
```python
        if node.step_type in (StepType.BASIC_TRAINING, StepType.ADVANCED_TUNING, StepType.TRAINING):
            return self._run_training(node, job_id=job_id)
```
to:
```python
        if node.step_type == StepType.TRAINING:
            return self._run_training(node, job_id=job_id)
```

- [ ] **Step 10: notebook export files — trim legacy step-type sets**

In `notebook_export.py:61`, change:
```python
_MODELING_STEPS = {"basic_training", "advanced_tuning", "training"}
```
to:
```python
_MODELING_STEPS = {"training"}
```

In `_notebook_branched.py:63-70`, change:
```python
_SPLIT_OR_MODEL = {
    "data_loader",
    "feature_target_split",
    "TrainTestSplitter",
    "basic_training",
    "advanced_tuning",
    "training",
}
```
to:
```python
_SPLIT_OR_MODEL = {
    "data_loader",
    "feature_target_split",
    "TrainTestSplitter",
    "training",
}
```

In `_notebook_builders.py:161-166`, change:
```python
def _is_tuning_model(model: _NodeIn) -> bool:
    """True for a legacy `advanced_tuning` node, or a unified `training`
    node whose `run_mode` param is `"tuned"`."""
    if model.step_type == "advanced_tuning":
        return True
    return model.step_type == "training" and model.params.get("run_mode") == "tuned"
```
to:
```python
def _is_tuning_model(model: _NodeIn) -> bool:
    """True for a `training` node whose `run_mode` param is `"tuned"`."""
    return model.step_type == "training" and model.params.get("run_mode") == "tuned"
```

- [ ] **Step 11: Fix any backend tests still constructing legacy step_type nodes**

```bash
cd /Users/BH7043/Skyulf && grep -rln 'step_type.*basic_training\|step_type.*advanced_tuning\|"basic_training"\|"advanced_tuning"' tests/
```
For each file found, update the fixture to use `step_type="training"` (+ appropriate `run_mode` param if the test needs to distinguish fixed vs. tuned) instead of the legacy step_type string.

- [ ] **Step 12: Run the full backend test suite**

```bash
cd /Users/BH7043/Skyulf && python -m pytest tests -q
```
Expected: all tests pass. Also run a repo-wide grep to confirm no references remain:
```bash
cd /Users/BH7043/Skyulf && grep -rn "BASIC_TRAINING\|ADVANCED_TUNING" --include=*.py backend/ tests/ | grep -v "\.pyc"
```
Expected: no output (or only comments/docstrings you've already updated — there should be zero remaining after this task).

- [ ] **Step 13: Commit**

```bash
cd /Users/BH7043/Skyulf && git add backend/ && git commit -m "refactor(step-type): remove legacy StepType.BASIC_TRAINING/ADVANCED_TUNING enum members and all alias-check branches

Co-authored-by: Copilot <223556219+Copilot@users.noreply.github.com>"
```

---

### Task 3: Backend — final sweep for any remaining legacy references

**Files:**
- Whole `backend/` tree (grep-only task, plus targeted fixes)

**Interfaces:**
- Consumes: Tasks 1-2 complete.
- Produces: a backend tree with zero remaining `basic_training`/`advanced_tuning`/`BASIC_TRAINING`/`ADVANCED_TUNING` references outside of historical changelog entries or the git log itself.

- [ ] **Step 1: Grep the whole backend tree**

```bash
cd /Users/BH7043/Skyulf && grep -rn "basic_training\|advanced_tuning\|BASIC_TRAINING\|ADVANCED_TUNING" --include=*.py backend/ tests/
```

- [ ] **Step 2: Triage each remaining hit**

For each hit found:
- If it's in a test file not already covered by Task 1/2's Step 7/11 — fix it the same way (update to `"training"`/`"tuning"` job_type value, or `step_type="training"` node fixture, as appropriate for what it's testing).
- If it's a stray comment/docstring — update the wording to remove the legacy name.
- If it's in `backend/ml_pipeline/model_registry/service.py` or similar as a `job_type` parameter name (not a value) — no change needed, the parameter name itself was never legacy, only the values passed to it were.
- **Do NOT touch `backend/database/engine.py:208-209`** — these reference historical migration table names (`basic_training_jobs`/`advanced_tuning_jobs`) which are migration history, not live `step_type`/`job_type` values. Renaming them would corrupt migration history. Leave as-is.
- If it's a hit inside `docs/user_guide/segmentation.md` — update the documented example from `step_type: "basic_training"` to `step_type: "training"` with a `run_mode` field, matching the canonical shape Task 4 produces.

- [ ] **Step 3: Re-run full backend suite to confirm still green**

```bash
cd /Users/BH7043/Skyulf && python -m pytest tests -q
```

- [ ] **Step 4: Commit (only if Step 2 made any changes)**

```bash
cd /Users/BH7043/Skyulf && git add backend/ && git commit -m "chore(backend): sweep remaining legacy basic_training/advanced_tuning references

Co-authored-by: Copilot <223556219+Copilot@users.noreply.github.com>"
```
(If Step 2 found nothing to change, skip this commit — note in your report that the sweep was clean.)

---

### Task 4: Frontend — migrate `SegmentationNode` and `EnsembleNode` to canonical `training`/`run_mode` emission

**Files:**
- Modify: `frontend/ml-canvas/src/core/utils/pipelineConverter.ts` (SegmentationNode branch ~line 459-468, EnsembleNode branch ~line 505-575)
- Test: `frontend/ml-canvas/src/core/utils/pipelineConverter.test.ts`, `frontend/ml-canvas/src/core/utils/pipelineConverter.snapshot.test.ts`

**Interfaces:**
- Consumes: Tasks 1-3 complete (backend now only accepts `step_type: "training"` for any training-family node, with `job_type` values `"training"`/`"tuning"`).
- Produces: `SegmentationNode` and `EnsembleNode` both emit `step_type: BackendStepType.TRAINING` with a `run_mode` param (`"fixed"` or `"tuned"`), exactly like `ClassificationNode`/`RegressionNode`/`TextClassificationNode` already do. Task 6 will delete the legacy definitionType sets that referenced the old behavior these branches used to have.

- [ ] **Step 1: Write/update failing tests for the SegmentationNode conversion**

In `pipelineConverter.test.ts`, find the existing test(s) that assert on a `SegmentationNode` conversion's output `step_type` (search for `'SegmentationNode'`). Update the assertion from expecting `step_type: 'basic_training'` (or `BackendStepType.BASIC_TRAINING`) to expecting:
```ts
expect(result.nodes[0].step_type).toBe('training');
expect(result.nodes[0].params.run_mode).toBe('fixed');
```
Run to confirm it now fails against current code:
```bash
cd frontend/ml-canvas && npx vitest run src/core/utils/pipelineConverter.test.ts
```
Expected: FAIL (current code still emits `'basic_training'`).

- [ ] **Step 2: Update the `SegmentationNode` branch**

Change (around line 459-468):
```ts
      } else if (node.data.definitionType === 'SegmentationNode') {
          stepType = BackendStepType.BASIC_TRAINING;
          params = {
              // No target_column — clustering is unsupervised. The backend
              // treats an empty string as the "no target" sentinel.
              target_column: '',
              model_type: node.data.model_type,
              hyperparameters: node.data.hyperparameters,
              cv_enabled: false,
              execution_mode: node.data.execution_mode,
              // Optional column (e.g. species name) excluded from training
              // but kept for post-hoc cluster interpretation — see
              // `reference_crosstab` in the evaluation report.
              reference_column: node.data.reference_column || undefined,
          };
```
to:
```ts
      } else if (node.data.definitionType === 'SegmentationNode') {
          stepType = BackendStepType.TRAINING;
          params = {
              run_mode: 'fixed',
              // No target_column — clustering is unsupervised. The backend
              // treats an empty string as the "no target" sentinel.
              target_column: '',
              model_type: node.data.model_type,
              hyperparameters: node.data.hyperparameters,
              cv_enabled: false,
              execution_mode: node.data.execution_mode,
              // Optional column (e.g. species name) excluded from training
              // but kept for post-hoc cluster interpretation — see
              // `reference_crosstab` in the evaluation report.
              reference_column: node.data.reference_column || undefined,
          };
```
(Segmentation has no tuned mode — it always submits `run_mode: 'fixed'`, matching its pre-existing behavior of never distinguishing fixed/tuned.)

- [ ] **Step 3: Run the Segmentation test again to confirm it passes**

```bash
cd frontend/ml-canvas && npx vitest run src/core/utils/pipelineConverter.test.ts
```
Expected: PASS.

- [ ] **Step 4: Write/update failing tests for the EnsembleNode conversion**

In `pipelineConverter.test.ts`, find the existing test(s) asserting on an `EnsembleNode` conversion (search `'EnsembleNode'`). There should be at least one fixed-mode and one advanced/tuned-mode case (if only one exists, add the missing one). Update/add assertions:
```ts
// fixed-mode ensemble
expect(fixedResult.nodes[0].step_type).toBe('training');
expect(fixedResult.nodes[0].params.run_mode).toBe('fixed');

// tuned-mode ensemble (node.data.run_mode === 'advanced')
expect(tunedResult.nodes[0].step_type).toBe('training');
expect(tunedResult.nodes[0].params.run_mode).toBe('tuned');
```
Run to confirm failure:
```bash
cd frontend/ml-canvas && npx vitest run src/core/utils/pipelineConverter.test.ts
```
Expected: FAIL.

- [ ] **Step 5: Update the `EnsembleNode` branch**

Change the two `stepType` assignments (around lines 505 and 551) from:
```ts
          if (node.data.run_mode === 'advanced') {
              stepType = BackendStepType.ADVANCED_TUNING;
              params = {
                  target_column: node.data.target_column,
```
to:
```ts
          if (node.data.run_mode === 'advanced') {
              stepType = BackendStepType.TRAINING;
              params = {
                  run_mode: 'tuned',
                  target_column: node.data.target_column,
```
and:
```ts
          } else {
              stepType = BackendStepType.BASIC_TRAINING;
              params = {
                  target_column: node.data.target_column,
                  model_type: node.data.model_type,
```
to:
```ts
          } else {
              stepType = BackendStepType.TRAINING;
              params = {
                  run_mode: 'fixed',
                  target_column: node.data.target_column,
                  model_type: node.data.model_type,
```
(Leave every other field in both param objects unchanged — this only adds `run_mode` and changes `stepType`.)

- [ ] **Step 6: Run the Ensemble tests again to confirm they pass**

```bash
cd frontend/ml-canvas && npx vitest run src/core/utils/pipelineConverter.test.ts
```
Expected: PASS.

- [ ] **Step 7: Update `pipelineConverter.snapshot.test.ts` if it has Segmentation/Ensemble fixtures**

```bash
grep -n "SegmentationNode\|EnsembleNode" frontend/ml-canvas/src/core/utils/pipelineConverter.snapshot.test.ts
```
If found, update the expected snapshot step_type/params the same way as Steps 2 and 5 above (or regenerate the snapshot with `npx vitest run -u src/core/utils/pipelineConverter.snapshot.test.ts` and manually review the diff to confirm only the intended `step_type`/`run_mode` fields changed — never blindly accept a snapshot update without reading the diff).

- [ ] **Step 8: Run the full frontend suite + tsc**

```bash
cd frontend/ml-canvas && npx tsc --noEmit && npx vitest run
```
Expected: 0 tsc errors, all tests pass.

- [ ] **Step 9: Commit**

```bash
cd /Users/BH7043/Skyulf && git add frontend/ml-canvas/ && git commit -m "refactor(pipelineConverter): emit canonical training step_type + run_mode for Segmentation and Ensemble nodes

Co-authored-by: Copilot <223556219+Copilot@users.noreply.github.com>"
```

---

### Task 5: Frontend — delete legacy hidden node/settings components

**Files:**
- Delete: `frontend/ml-canvas/src/modules/nodes/modeling/BasicTrainingNode.ts`
- Delete: `frontend/ml-canvas/src/modules/nodes/modeling/AdvancedTuningNode.ts`
- Delete: `frontend/ml-canvas/src/modules/nodes/modeling/BasicTrainingSettings.tsx`
- Delete: `frontend/ml-canvas/src/modules/nodes/modeling/AdvancedTuningSettings.tsx`
- Delete: `frontend/ml-canvas/src/modules/nodes/modeling/TrainingNode.ts` (generic Phase-3-Part-A node, fully superseded by the 4 task-scoped nodes — confirmed via investigation it is never used by `ClassificationNode`/`RegressionNode`/`TextClassificationNode`, they each define their own `createModelingNode(...)` and only reuse the `TrainingSettings` component function, not this node object)
- Modify: `frontend/ml-canvas/src/core/registry/init.ts` (remove imports on lines ~7-9 and registrations on lines ~82-84)
- Modify: `frontend/ml-canvas/src/modules/nodes/bodyPreview.test.ts` (remove the `BasicTrainingNode` import at line ~27, the `ModelTrainingConfig` type import at line ~28, and the `it('createModelingNode default preview shows model and target', ...)` test block at lines ~167-173 that calls `BasicTrainingNode.bodyPreview!(...)` — this test's coverage of "does createModelingNode's default bodyPreview show model+target" is still meaningful, so instead of deleting the whole test, rewrite it against `ClassificationNode` — see Step 4 below)
- Check (comment-only, no code change needed unless grep shows otherwise): `core/types/nodes.ts:45`, `core/factories/nodeFactory.ts:25`, `components/layout/PropertiesPanel.tsx:38`, `modules/nodes/modeling/SegmentationNode.tsx:7`, `modules/nodes/modeling/components/HyperparameterInput.tsx:13`, `modules/nodes/modeling/components/SearchSpaceInput.tsx:13`, `modules/nodes/modeling/components/BestParamsModal.tsx:21-22`, `modules/nodes/modeling/SegmentationSettings.tsx:36`

**Interfaces:**
- Consumes: Task 4 complete (canonical emission for Segmentation/Ensemble already lands, independent of this task, but do this after Task 4 to keep the git history in a logical "emission fixed, then dead code removed" order).
- Produces: `BasicTrainingNode`, `AdvancedTuningNode`, `TrainingNode` (generic), and their settings components no longer exist in the tree. Nothing outside `init.ts` and `bodyPreview.test.ts` imports them (confirmed by investigation — this is the complete reference list).

- [ ] **Step 1: Confirm no other importers exist**

```bash
cd frontend/ml-canvas && grep -rln "from.*BasicTrainingNode\|from.*AdvancedTuningNode\|from.*modeling/TrainingNode'\|from.*BasicTrainingSettings\|from.*AdvancedTuningSettings" src/
```
Expected output: only `src/core/registry/init.ts` and `src/modules/nodes/bodyPreview.test.ts`. If anything else appears, STOP and report — do not proceed with deletion until every importer is accounted for.

- [ ] **Step 2: Update `init.ts`**

Remove these 3 import lines (find by content, exact line numbers may have shifted since investigation):
```ts
import { BasicTrainingNode } from '...';
import { AdvancedTuningNode } from '...';
import { TrainingNode } from '...';
```
Remove this comment block and the 3 registration lines:
```ts
  // Legacy Basic Training / Advanced Tuning node types are kept registered
  // (unchanged, `hidden: true`) so canvases saved before Phase 3's unified
  // `TrainingNode` still load and render correctly — see the unification
  // plan doc. The generic `TrainingNode` itself is also now `hidden: true`
  // (Phase 3 Part B, plan §0.6): it stays registered so canvases saved
  // during Part A keep loading/executing, but new canvases should use one
  // of the 4 task-scoped nodes below (Classification / Regression / Text
  // Classification / Segmentation) for better discoverability — the
  // task-type split is the intended end state, the generic node was a
  // stepping stone.
  registry.register({ ...BasicTrainingNode, hidden: true });
  registry.register({ ...AdvancedTuningNode, hidden: true });
  registry.register({ ...TrainingNode, hidden: true });
```
Leave the subsequent `registry.register(ClassificationNode);` etc. lines untouched.

- [ ] **Step 3: Delete the 5 files**

```bash
cd frontend/ml-canvas && rm -f \
  src/modules/nodes/modeling/BasicTrainingNode.ts \
  src/modules/nodes/modeling/AdvancedTuningNode.ts \
  src/modules/nodes/modeling/BasicTrainingSettings.tsx \
  src/modules/nodes/modeling/AdvancedTuningSettings.tsx \
  src/modules/nodes/modeling/TrainingNode.ts
```

- [ ] **Step 4: Rewrite the `bodyPreview.test.ts` test against `ClassificationNode`**

Remove the import lines:
```ts
import { BasicTrainingNode } from './modeling/BasicTrainingNode';
import type { ModelTrainingConfig } from './modeling/BasicTrainingSettings';
```
Add instead:
```ts
import { ClassificationNode } from './modeling/ClassificationNode';
import type { TrainingConfig } from './modeling/TrainingSettings';
```
Find the test block (~lines 167-173):
```ts
  it('createModelingNode default preview shows model and target', () => {
    const config: ModelTrainingConfig = { /* ...whatever fields the original test set... */ };
    const preview = BasicTrainingNode.bodyPreview!(config);
    // ...assertions referencing BasicTrainingNode...
  });
```
Rewrite it to use `ClassificationNode` and `TrainingConfig` in place of `BasicTrainingNode`/`ModelTrainingConfig`, keeping the same field values and the same assertions (only the node/type names change — the underlying `createModelingNode` default `bodyPreview` behavior being tested is identical for every node built via the same factory, so the assertions themselves do not need to change, only which node object is invoked).

- [ ] **Step 5: Run the full frontend suite + tsc**

```bash
cd frontend/ml-canvas && npx tsc --noEmit && npx vitest run
```
Expected: 0 tsc errors, all tests pass.

- [ ] **Step 6: Commit**

```bash
cd /Users/BH7043/Skyulf && git add -A frontend/ml-canvas/ && git commit -m "refactor(nodes): delete legacy BasicTrainingNode/AdvancedTuningNode/generic TrainingNode and their settings components

Co-authored-by: Copilot <223556219+Copilot@users.noreply.github.com>"
```

---

### Task 6: Frontend — remove legacy `definitionType` alias-checks and fix the canonical-type recognition gap

**Files:**
- Modify: `frontend/ml-canvas/src/core/constants/stepTypes.ts` (remove `BASIC_TRAINING`/`ADVANCED_TUNING` enum members)
- Modify: `frontend/ml-canvas/src/core/types/executionMode.ts:26-29` (`EXECUTION_MODE_AWARE_TYPES`)
- Modify: `frontend/ml-canvas/src/core/hooks/useBranchColors.ts:14-25` (`TERMINAL_TYPES`, `MODEL_SOURCE_TYPES`)
- Modify: `frontend/ml-canvas/src/core/store/useGraphStore.ts:183,196` (`modelTypes`, `modelSourceTypes`)
- Modify: `frontend/ml-canvas/src/components/layout/toolbar/_hooks/useRunControls.ts:10` (`TRAINING_TYPES`)
- Modify: `frontend/ml-canvas/src/core/perf/perfThresholds.ts:39-42` (`getPerfFamily`)
- Modify: `frontend/ml-canvas/src/core/templates/pipelineTemplates.ts:118,145,173` (canned template `type:` values)
- Modify: `frontend/ml-canvas/src/core/utils/pipelineConverter.ts:13-58` (`MODEL_SOURCE_TYPES`, `LEGACY_FIXED_TRAINING_TYPES`, `LEGACY_TUNED_TRAINING_TYPES`, `ALL_TRAINING_DISPATCH_TYPES`, and the `isAdvanced` resolution at ~line 433)
- Modify: `frontend/ml-canvas/src/modules/nodes/modeling/EnsembleSettings.tsx:776,804-808,970-971` (base-learner-detection arrays)
- Test: `frontend/ml-canvas/src/core/types/executionMode.test.ts`, `frontend/ml-canvas/src/core/utils/pipelineConverter.test.ts`, `frontend/ml-canvas/src/core/hooks/useUpstreamData.test.ts`, `frontend/ml-canvas/src/core/hooks/useBranchColors.test.ts`, `frontend/ml-canvas/src/core/store/useGraphStore.test.ts`

**Interfaces:**
- Consumes: Tasks 4-5 complete (canonical emission fixed, legacy nodes already deleted from the registry so no canvas can produce a `'basic_training'`/`'advanced_tuning'` `definitionType` going forward).
- Produces: every "is this a trainable/mode-aware/model-source node" allow-list in the frontend recognizes the canonical `'training'`, `'classification'`, `'regression'`, `'text_classification'` set (Segmentation is intentionally excluded from execution-mode/mode-aware lists exactly as it was before, since clustering has no `run_mode` toggle — only add it to lists where it was already separately handled, e.g. `MODEL_SOURCE_TYPES` for ensemble base-learner detection, which already included `'segmentation'`... check: it currently does NOT include `'segmentation'`, so leave that alone — Segmentation was never a valid Ensemble base-learner source and this plan does not change that scope).

- [ ] **Step 1: `stepTypes.ts` — remove the 2 enum members and update comments**

Change:
```ts
// Matching backend/ml_pipeline/constants.py
export enum StepType {
  DATA_LOADER = 'data_loader',
  FEATURE_ENGINEERING = 'feature_engineering',
  BASIC_TRAINING = 'basic_training',
  ADVANCED_TUNING = 'advanced_tuning',
  // Backend Phase 2b superset ("training", run_mode: fixed|tuned). Used only
  // as the canvas `definitionType` for the unified `TrainingNode` (Phase 3);
  // it is never submitted as a job `job_type` — the node still submits
  // BASIC_TRAINING/ADVANCED_TUNING depending on its `run_mode`, same as
  // before, so the backend dispatcher doesn't need to understand this value.
  TRAINING = 'training',
  // Task-scoped canvas nodes (Phase 3 Part B, plan §0.6). Like TRAINING,
  // these are canvas-only `definitionType`s — never submitted as a job
  // `job_type`. Each still resolves to BASIC_TRAINING/ADVANCED_TUNING based
  // on `run_mode`, exactly like TRAINING; the backend doesn't need to know
  // which task-scoped node produced the job.
  CLASSIFICATION = 'classification',
  REGRESSION = 'regression',
  TEXT_CLASSIFICATION = 'text_classification',
}
```
to:
```ts
// Matching backend/ml_pipeline/constants.py
export enum StepType {
  DATA_LOADER = 'data_loader',
  FEATURE_ENGINEERING = 'feature_engineering',
  // Every training-family canvas node (the generic node plus the
  // task-scoped Classification/Regression/Text Classification/Segmentation/
  // Ensemble nodes) submits this same canonical step_type, discriminated by
  // a `run_mode: 'fixed' | 'tuned'` param — the backend doesn't need to know
  // which canvas node produced the job.
  TRAINING = 'training',
  CLASSIFICATION = 'classification',
  REGRESSION = 'regression',
  TEXT_CLASSIFICATION = 'text_classification',
}
```

- [ ] **Step 2: `executionMode.ts` — expand `EXECUTION_MODE_AWARE_TYPES`**

Change:
```ts
export const EXECUTION_MODE_AWARE_TYPES: ReadonlySet<string> = new Set([
  'basic_training',
  'advanced_tuning',
]);
```
to:
```ts
export const EXECUTION_MODE_AWARE_TYPES: ReadonlySet<string> = new Set([
  'training',
  'classification',
  'regression',
  'text_classification',
]);
```
Update the doc comment above (currently says `training nodes (basic_training, advanced_tuning) and auto-parallel terminals`) to reflect the new canonical type names instead of the old ones.

- [ ] **Step 2b: Write a test proving the gap is fixed**

In `executionMode.test.ts`, find the existing tests using `definitionType: 'basic_training'` (lines ~49,52 per investigation) and change those fixtures to `definitionType: 'classification'` (or `'training'`) — same assertions, just the canonical type name. Then add one new test case asserting the fix:
```ts
it('supportsExecutionModeToggle recognizes canonical task-scoped node types', () => {
  expect(supportsExecutionModeToggle('classification')).toBe(true);
  expect(supportsExecutionModeToggle('regression')).toBe(true);
  expect(supportsExecutionModeToggle('text_classification')).toBe(true);
  expect(supportsExecutionModeToggle('training')).toBe(true);
});
```
Run:
```bash
cd frontend/ml-canvas && npx vitest run src/core/types/executionMode.test.ts
```
Expected: PASS (this specific file's tests, after the Step 2 code change is also in place — do Step 2 and this together, then run once).

- [ ] **Step 3: `useBranchColors.ts` — update both sets**

Change:
```ts
const TERMINAL_TYPES = new Set([...EXECUTION_MODE_AWARE_TYPES]);
```
No change needed to this line itself — it derives from `EXECUTION_MODE_AWARE_TYPES`, which Step 2 already fixed.

Change:
```ts
const MODEL_SOURCE_TYPES = new Set([
  'model_training',
  'basic_training',
  'hyperparameter_tuning',
  'advanced_tuning',
]);
```
to:
```ts
const MODEL_SOURCE_TYPES = new Set([
  'training',
  'classification',
  'regression',
  'text_classification',
]);
```
Update the doc comment above (currently references the legacy names and `MODEL_SOURCE_TYPES` in `pipelineConverter.ts`, which Step 8 below also updates to match) to use canonical names.

Add/update a test in `useBranchColors.test.ts` (fixtures at lines 47,60,74,118,119,141,142,144,175 per investigation) — change any `'basic_training'`/`'advanced_tuning'` definitionType fixture to `'classification'`/`'training'`, keeping the same assertions.

- [ ] **Step 4: `useGraphStore.ts` — update both connection-validation lists**

Change:
```ts
      const modelTypes = ['basic_training', 'advanced_tuning'];
```
to:
```ts
      const modelTypes = ['training', 'classification', 'regression', 'text_classification'];
```
Change:
```ts
      const modelSourceTypes = ['basic_training', 'advanced_tuning', 'model_training', 'hyperparameter_tuning'];
```
to:
```ts
      const modelSourceTypes = ['training', 'classification', 'regression', 'text_classification'];
```
Update `useGraphStore.test.ts` (lines ~198-199 per investigation construct nodes with `addNode('basic_training', ...)`) to use `addNode('classification', ...)` or `'training'` instead, keeping the same assertions about the connection-validation warning firing.

- [ ] **Step 5: `useRunControls.ts` — update `TRAINING_TYPES`**

Change:
```ts
const TRAINING_TYPES = new Set(['basic_training', 'advanced_tuning']);
```
to:
```ts
const TRAINING_TYPES = new Set(['training', 'classification', 'regression', 'text_classification']);
```

- [ ] **Step 6: `perfThresholds.ts` — update `getPerfFamily` to accept `run_mode` (per rubber-duck review: thread it through rather than collapsing to one bucket)**

Change:
```ts
export function getPerfFamily(definitionType: string): PerfFamily {
  if (definitionType === 'advanced_tuning') return 'tuner';
  if (definitionType === 'basic_training') return 'trainer';
  return 'preprocess';
}
```
to:
```ts
export function getPerfFamily(definitionType: string, runMode?: string): PerfFamily {
  if (['training', 'classification', 'regression', 'text_classification'].includes(definitionType)) {
    // Node-level `data.run_mode` uses the UI toggle values 'basic'/'advanced'
    // (see ClassificationNode.tsx/RegressionNode.tsx/TextClassificationNode.tsx/
    // TrainingNode.ts/EnsembleNode.ts defaultConfig: `run_mode: 'basic'`) —
    // NOT the backend dispatch values 'fixed'/'tuned'. Do not confuse the two.
    return runMode === 'advanced' ? 'tuner' : 'trainer';
  }
  return 'preprocess';
}
```

Then update its sole caller in `frontend/ml-canvas/src/components/canvas/CustomNodeWrapper.tsx:91` (note: actual path is `components/canvas/CustomNodeWrapper.tsx`, NOT `components/canvas/nodes/CustomNodeWrapper.tsx` — the file already destructures `data` in scope at line 15, so `data.run_mode` is directly available):

Change:
```ts
      : bucketDuration(perfDurationMs, getPerfFamily(definitionType));
```
to:
```ts
      : bucketDuration(perfDurationMs, getPerfFamily(definitionType, data.run_mode as string | undefined));
```

This correctly distinguishes fixed vs tuned runs of canonical nodes for perf bucketing instead of collapsing all tuned runs into the 'trainer' bucket (which would make tuned runs — which legitimately take longer — misleadingly show as "slow"/red-ringed against the trainer threshold).

- [ ] **Step 7: `pipelineTemplates.ts` — update canned template node types**

```bash
grep -n "'basic_training'\|'advanced_tuning'" frontend/ml-canvas/src/core/templates/pipelineTemplates.ts
```
For each of the 3 occurrences (lines ~118,145,173 per investigation), change `type: 'basic_training'` to `type: 'classification'` (or `'regression'`/`'training'` — pick whichever matches that specific template's actual intent by reading the surrounding template definition's other params, e.g. if the template's `target_column`/model choice implies a classification task, use `'classification'`; check each one individually rather than blanket-replacing).

- [ ] **Step 8: `pipelineConverter.ts` — remove legacy sets, simplify dispatch**

Change:
```ts
const MODEL_SOURCE_TYPES = new Set([
  'model_training',
  'basic_training',
  'hyperparameter_tuning',
  'advanced_tuning',
  'training',
  'classification',
  'regression',
  'text_classification',
]);
```
to:
```ts
const MODEL_SOURCE_TYPES = new Set([
  'training',
  'classification',
  'regression',
  'text_classification',
]);
```
Change (remove the two legacy sets and simplify `ALL_TRAINING_DISPATCH_TYPES` to just equal `RUN_MODE_TRAINING_TYPES`):
```ts
// Legacy hidden node `definitionType`s (Phase 7 merge): these predate the
// `run_mode` field entirely (see `BasicTrainingNode.ts`/`AdvancedTuningNode.ts`
// `defaultConfig` — neither ever sets `run_mode`), so their fixed/tuned intent
// must be read off the definitionType itself rather than `node.data.run_mode`.
const LEGACY_FIXED_TRAINING_TYPES = new Set<string>([
  'model_training',
  BackendStepType.BASIC_TRAINING,
]);
const LEGACY_TUNED_TRAINING_TYPES = new Set<string>([
  'hyperparameter_tuning',
  BackendStepType.ADVANCED_TUNING,
]);

// All definitionTypes that flow through the single shared fixed/tuned
// training dispatch below (Phase 7 merge of the legacy + unified branches).
const ALL_TRAINING_DISPATCH_TYPES = new Set<string>([
  ...RUN_MODE_TRAINING_TYPES,
  ...LEGACY_FIXED_TRAINING_TYPES,
  ...LEGACY_TUNED_TRAINING_TYPES,
]);
```
to:
```ts
// All definitionTypes that flow through the single shared fixed/tuned
// training dispatch below: the generic TrainingNode plus the 3 task-scoped
// supervised nodes.
const ALL_TRAINING_DISPATCH_TYPES = RUN_MODE_TRAINING_TYPES;
```
Then simplify the `isAdvanced` resolution (around line 419-436) — since no legacy node without `run_mode` remains, this no longer needs the definitionType fallback:
```ts
      } else if (ALL_TRAINING_DISPATCH_TYPES.has(node.data.definitionType as string)) {
          // Unified TrainingNode, the task-scoped Classification/Regression/
          // Text Classification nodes (Phase 3 Part B), AND the legacy
          // hidden BasicTrainingNode/AdvancedTuningNode (Phase 7 merge):
          // all dispatch through the same fixed/tuned param-building
          // helpers and emit the canonical `training` step_type.
          //
          // The legacy nodes never populate `node.data.run_mode` at all
          // (only the newer unified TrainingNode defaults one) - so the
          // "advanced" intent must be inferred from the node's
          // `definitionType` itself for those old node shapes. Do not
          // simplify this to `node.data.run_mode === 'advanced'` alone;
          // that would silently reclassify every saved legacy
          // AdvancedTuningNode as a fixed/basic run.
          const isAdvanced = node.data.run_mode
              ? node.data.run_mode === 'advanced'
              : LEGACY_TUNED_TRAINING_TYPES.has(node.data.definitionType as string);
          if (isAdvanced) {
```
to:
```ts
      } else if (ALL_TRAINING_DISPATCH_TYPES.has(node.data.definitionType as string)) {
          // The generic TrainingNode and the 3 task-scoped Classification/
          // Regression/Text Classification nodes all dispatch through the
          // same fixed/tuned param-building helpers and emit the canonical
          // `training` step_type.
          const isAdvanced = node.data.run_mode === 'advanced';
          if (isAdvanced) {
```
Update the `buildFixedTrainingParams` doc comment (~line 145-148) from mentioning `"basic_training"` to just describing it as the fixed-mode param builder shared by the unified `TrainingNode` and task-scoped nodes.

Update `pipelineConverter.test.ts` and `pipelineConverter.snapshot.test.ts`: every fixture using `definitionType: 'model_training'`, `'basic_training'`, `'hyperparameter_tuning'`, or `'advanced_tuning'` (lines listed in the investigation report: 40,133,169,206,243,358,420,455,460,526,545,562,567,607,612 in the `.test.ts` file, and 47,94 in the snapshot file) must be updated to use a canonical `definitionType` instead (`'training'`, `'classification'`, `'regression'`, or `'text_classification'`, matching what the surrounding test case is actually exercising — e.g. a test titled "legacy fixed training node" should become "generic TrainingNode fixed run" using `definitionType: 'training'`, `run_mode: 'basic'` on `node.data`). Read each test case's `it(...)` description to preserve its intent, don't just find-and-replace the string blindly.

- [ ] **Step 9: `EnsembleSettings.tsx` — update base-learner-detection arrays**

Change (line ~776):
```ts
      if (src && ['basic_training', 'advanced_tuning', 'model_training', 'hyperparameter_tuning'].includes(src.data.definitionType as string)) {
```
to:
```ts
      if (src && ['training', 'classification', 'regression', 'text_classification'].includes(src.data.definitionType as string)) {
```
Change (lines ~804-808):
```ts
    const advancedNode = incomingModels.find(
      (m) =>
        m.data.definitionType === 'advanced_tuning' ||
        m.data.definitionType === 'hyperparameter_tuning' ||
        m.data.run_mode === 'advanced'
    );
```
to:
```ts
    const advancedNode = incomingModels.find((m) => m.data.run_mode === 'advanced');
```
Change (lines ~970-971):
```ts
        ['basic_training', 'advanced_tuning', 'model_training', 'hyperparameter_tuning'].includes(
          src.data.definitionType as string,
        )
```
to:
```ts
        ['training', 'classification', 'regression', 'text_classification'].includes(
          src.data.definitionType as string,
        )
```

- [ ] **Step 10: Update `useUpstreamData.test.ts`**

Change the fixture at line ~55 from `data: { definitionType: 'basic_training' }` to `data: { definitionType: 'training' }` (or `'classification'`, matching whatever that specific test case is about).

- [ ] **Step 11: Run the full frontend suite + tsc**

```bash
cd frontend/ml-canvas && npx tsc --noEmit && npx vitest run
```
Expected: 0 tsc errors, all tests pass.

- [ ] **Step 12: Grep for any remaining legacy string literal**

```bash
cd frontend/ml-canvas && grep -rn "'basic_training'\|'advanced_tuning'\|'model_training'\|'hyperparameter_tuning'\|BASIC_TRAINING\|ADVANCED_TUNING" src/ --include=*.ts --include=*.tsx
```
Every remaining hit at this point should ONLY be `job_type` API values (handled in Task 7) — grep should show nothing else. If any `definitionType`/`step_type` check still appears, fix it following the same pattern as Steps 1-9 above before moving on.

- [ ] **Step 13: Commit**

```bash
cd /Users/BH7043/Skyulf && git add frontend/ml-canvas/ && git commit -m "refactor(definitionType): remove legacy basic_training/advanced_tuning alias-checks; fix canonical-type recognition gap in execution-mode/branch-color/perf/connection-validation logic

Co-authored-by: Copilot <223556219+Copilot@users.noreply.github.com>"
```

---

### Task 7: Frontend — rename `job_type` values to `"training"`/`"tuning"`

**Files:**
- Modify: `frontend/ml-canvas/src/core/api/jobs.ts:12` (`JobInfo.job_type`), `:40` (`RunPipelineRequest.job_type`)
- Modify: `frontend/ml-canvas/src/core/hooks/useTrainingNodeContext.ts:12` (`JobType` type)
- Modify: `frontend/ml-canvas/src/modules/nodes/modeling/TrainingSettings.tsx:315` (`runJob` call)
- Modify: `frontend/ml-canvas/src/modules/nodes/modeling/EnsembleSettings.tsx:1134` (`runJob` call)
- Modify: `frontend/ml-canvas/src/components/layout/toolbar/_hooks/useRunControls.ts:105` (`job_type: 'basic_training'` in `jobsApi.runPipeline`)
- Modify: `frontend/ml-canvas/src/components/panels/jobs/JobCard.tsx:73,120`
- Modify: `frontend/ml-canvas/src/components/panels/jobs/JobDetailsView.tsx:428,449`
- Modify: `frontend/ml-canvas/src/components/pages/ExperimentsPage/utils/jobMeta.ts:90,176`
- Modify: `frontend/ml-canvas/src/components/pages/ExperimentsPage/components/JobListSidebar.tsx:81`
- Test: `frontend/ml-canvas/src/core/hooks/useJobPolling.test.ts:12` (fixture), plus targeted tests for `JobCard.tsx`/`JobDetailsView.tsx`/`jobMeta.ts` if they exist — search `grep -rln "job_type" src/**/*.test.ts src/**/*.test.tsx`

**Interfaces:**
- Consumes: Task 1 (backend already emits/accepts `job_type: "training"|"tuning"|"preview"`).
- Produces: frontend `JobInfo.job_type`/`RunPipelineRequest.job_type` types and every UI branch reading `job.job_type` now use `"training"`/`"tuning"` — end-to-end consistent with the backend from Task 1.

- [ ] **Step 1: `core/api/jobs.ts` — update both type unions**

Change:
```ts
  job_type: 'basic_training' | 'advanced_tuning' | 'eda' | 'ingestion';
```
to:
```ts
  job_type: 'training' | 'tuning' | 'eda' | 'ingestion';
```
Change:
```ts
export interface RunPipelineRequest extends PipelineConfigModel {
  target_node_id?: string;
  job_type?: 'basic_training' | 'advanced_tuning' | 'preview';
}
```
to:
```ts
export interface RunPipelineRequest extends PipelineConfigModel {
  target_node_id?: string;
  job_type?: 'training' | 'tuning' | 'preview';
}
```

- [ ] **Step 2: `useTrainingNodeContext.ts` — update `JobType`**

Change:
```ts
type JobType = 'basic_training' | 'advanced_tuning';
```
to:
```ts
type JobType = 'training' | 'tuning';
```

- [ ] **Step 3: `TrainingSettings.tsx` — update the `runJob` call**

Change:
```ts
    await runJob(isAdvanced ? 'advanced_tuning' : 'basic_training', resolvedTask === 'other' ? 'classification' : resolvedTask);
```
to:
```ts
    await runJob(isAdvanced ? 'tuning' : 'training', resolvedTask === 'other' ? 'classification' : resolvedTask);
```

- [ ] **Step 4: `EnsembleSettings.tsx` — update the `runJob` call**

Change (this call was already fixed earlier this session to pass `'ensemble'` as the task argument — only the job_type argument changes here):
```ts
          onClick={() => { void runJob(isAdvanced ? 'advanced_tuning' : 'basic_training', 'ensemble'); }}
```
to:
```ts
          onClick={() => { void runJob(isAdvanced ? 'tuning' : 'training', 'ensemble'); }}
```

- [ ] **Step 5: `useRunControls.ts` — update the "Run All" job_type**

Change:
```ts
      const response = await jobsApi.runPipeline({
        ...pipelineConfig,
        job_type: 'basic_training',
      });
```
to:
```ts
      const response = await jobsApi.runPipeline({
        ...pipelineConfig,
        job_type: 'training',
      });
```
Also update `TRAINING_TYPES` in this same file if Task 6 hasn't already renamed it (it should already be done — verify with `grep -n TRAINING_TYPES` in this file; if it still says `'basic_training'`/`'advanced_tuning'`, that means Task 6's Step 5 wasn't applied yet — do not proceed until it is, since these are different concepts (`TRAINING_TYPES` here is a `definitionType` check, unrelated to this task's `job_type` rename) but both need to be correct before this task's tests can pass.

- [ ] **Step 6: `JobCard.tsx` — update the 2 job_type checks**

Change:
```ts
        {job.job_type === 'advanced_tuning' && job.search_strategy && (
```
to:
```ts
        {job.job_type === 'tuning' && job.search_strategy && (
```
Change:
```ts
          ) : job.job_type === 'advanced_tuning' && !!(job.result as Record<string, unknown>).best_params ? (
```
to:
```ts
          ) : job.job_type === 'tuning' && !!(job.result as Record<string, unknown>).best_params ? (
```

- [ ] **Step 7: `JobDetailsView.tsx` — update the 2 job_type checks**

Change:
```ts
                                {job.job_type === 'basic_training' && !!(job.result as Record<string, unknown>).metrics && (
```
to:
```ts
                                {job.job_type === 'training' && !!(job.result as Record<string, unknown>).metrics && (
```
Change:
```ts
                                {job.job_type === 'advanced_tuning' && (
```
to:
```ts
                                {job.job_type === 'tuning' && (
```

- [ ] **Step 8: `jobMeta.ts` — update 2 job_type checks**

Change (line ~90):
```ts
  if (job.job_type === 'advanced_tuning') {
```
to:
```ts
  if (job.job_type === 'tuning') {
```
Change (line ~176, `hasTuningMetadata`):
```ts
export function hasTuningMetadata(job: { job_type?: string; search_strategy?: string }): boolean {
  return job.search_strategy != null || job.job_type === 'advanced_tuning';
}
```
to:
```ts
export function hasTuningMetadata(job: { job_type?: string; search_strategy?: string }): boolean {
  return job.search_strategy != null || job.job_type === 'tuning';
}
```
Update the doc comment above `hasTuningMetadata` (mentions "checking `job_type === 'advanced_tuning'`") to say `'tuning'` instead.

- [ ] **Step 9: `JobListSidebar.tsx` — update the job_type check**

Change:
```ts
                    {job.status === 'completed' && (job.job_type === 'basic_training' || job.job_type === 'advanced_tuning') && (
```
to:
```ts
                    {job.status === 'completed' && (job.job_type === 'training' || job.job_type === 'tuning') && (
```

- [ ] **Step 10: Update test fixtures**

```bash
cd frontend/ml-canvas && grep -rln "job_type.*basic_training\|job_type.*advanced_tuning\|job_type: 'basic_training'\|job_type: 'advanced_tuning'" src/
```
For each match (expect at least `useJobPolling.test.ts:12`), update the fixture value to `'training'`/`'tuning'`.

- [ ] **Step 11: Run the full frontend suite + tsc**

```bash
cd frontend/ml-canvas && npx tsc --noEmit && npx vitest run
```
Expected: 0 tsc errors, all tests pass.

- [ ] **Step 12: Final grep sweep**

```bash
cd frontend/ml-canvas && grep -rn "basic_training\|advanced_tuning\|BASIC_TRAINING\|ADVANCED_TUNING" src/
```
Expected: no output at all.

- [ ] **Step 13: Commit**

```bash
cd /Users/BH7043/Skyulf && git add frontend/ml-canvas/ && git commit -m "refactor(job_type): rename API job_type values from basic_training/advanced_tuning to training/tuning end-to-end

Co-authored-by: Copilot <223556219+Copilot@users.noreply.github.com>"
```

---

### Task 8: Frontend — final sweep for any remaining legacy references

**Files:**
- Whole `frontend/ml-canvas/src/` tree (grep-only task, plus targeted fixes)

**Interfaces:**
- Consumes: Tasks 4-7 complete.
- Produces: zero remaining `basic_training`/`advanced_tuning`/`model_training`/`hyperparameter_tuning`/`BASIC_TRAINING`/`ADVANCED_TUNING` references in `src/` outside of git history/changelog.

- [ ] **Step 1: Repo-wide grep**

```bash
cd frontend/ml-canvas && grep -rn "basic_training\|advanced_tuning\|model_training\|hyperparameter_tuning\|BASIC_TRAINING\|ADVANCED_TUNING" src/
```

- [ ] **Step 2: Triage and fix each remaining hit**

Same pattern as Task 3's Step 2 — fix test fixtures, comments, or any missed code path. If a hit is inside `node_modules` or a build artifact, ignore it (the grep above is already scoped to `src/` so this shouldn't occur, but double-check the path prefix on every result).

- [ ] **Step 3: Re-run full suite + tsc**

```bash
cd frontend/ml-canvas && npx tsc --noEmit && npx vitest run
```

- [ ] **Step 4: Commit (only if Step 2 made any changes)**

```bash
cd /Users/BH7043/Skyulf && git add frontend/ml-canvas/ && git commit -m "chore(frontend): sweep remaining legacy basic_training/advanced_tuning references

Co-authored-by: Copilot <223556219+Copilot@users.noreply.github.com>"
```

---

### Task 9: Final verification — wipe dev DB, full e2e smoke test, changelog

**Files:**
- Delete/regenerate: dev SQLite DB file (find via `grep -n "database" backend/config.py` or similar — likely `backend/mlops_database.db`, same file dropped during the earlier Phase 4/13 cleanup this session)
- Modify: `CHANGELOG.md`, `changelog/0.7.x.md` (or the next unreleased version file, check which is currently "Active" in `CHANGELOG.md`'s table)

**Interfaces:**
- Consumes: all of Tasks 1-8 complete and committed.
- Produces: a fully working dev environment with zero legacy references anywhere, confirmed via full test suites AND real end-to-end pipeline execution across every affected node type, plus an accurate changelog entry.

- [ ] **Step 1: Wipe and regenerate the dev database**

```bash
cd /Users/BH7043/Skyulf && find backend -iname "*.db" -maxdepth 2
```
Delete the file(s) found (this environment is dev-only, confirmed safe by the user). Start the backend app briefly to confirm it recreates the schema on startup via `Base.metadata.create_all` (same pattern used in the earlier Phase 4/13 verification this session) — check `backend/main.py` or wherever app startup lives for this call, then run:
```bash
cd backend && python -c "from backend.main import app" 2>&1 | tail -20
```
or start the dev server briefly and confirm no startup errors, then stop it.

- [ ] **Step 2: Full backend test suite**

```bash
cd /Users/BH7043/Skyulf && python -m pytest tests -q
```
Expected: all tests pass. Record the exact pass count.

- [ ] **Step 3: Full frontend test suite + tsc**

```bash
cd frontend/ml-canvas && npx tsc --noEmit && npx vitest run
```
Expected: 0 tsc errors, all tests pass. Record the exact pass count.

- [ ] **Step 3b: Confirm skyulf-core is unaffected (out of scope, quick confirmation only)**

```bash
cd /Users/BH7043/Skyulf && python -m pytest skyulf-core/tests -q
```
Expected: same pass count as before this plan started (skyulf-core source has zero legacy `step_type`/`job_type` references; the two `test_advanced_tuning_runs_*` function names in `skyulf-core/tests/test_ensemble.py` test `run_mode` behavior, not the enum, and are left as-is).

- [ ] **Step 3c: Rebuild the frontend bundle before manual/browser smoke testing**

The compiled `static/ml_canvas/assets/index-*.js`/`index.html` bundle still contains the pre-change legacy strings until rebuilt. If Step 4 below is exercised through a browser against the built bundle rather than the Vite dev server, none of this plan's frontend source changes will actually be in effect.

```bash
cd frontend/ml-canvas && npm run build
```
Confirm the build succeeds with no errors, and that the output lands in the path the backend serves from (check `backend/main.py`'s static-file mount). If Step 4's smoke test instead uses the Vite dev server (`npm run dev`) directly, this rebuild is not required — but state explicitly in your task report which of the two you used.

- [ ] **Step 4: Real end-to-end pipeline execution smoke test**

Using the same approach as the earlier "test one ensemble and one normal for each" smoke test this session (real `PipelineEngine`/API executions, not just conversion-layer unit tests), exercise:
1. Classification, fixed mode (`run_mode: 'fixed'`)
2. Classification, tuned mode (`run_mode: 'tuned'`)
3. Regression, fixed mode (shares the same dispatch branch as Classification but is cheap insurance given how many files this plan touches)
4. Segmentation (clustering, always fixed)
5. Ensemble Voting Classifier, fixed mode
6. Ensemble Voting Classifier, tuned mode
7. Ensemble Stacking Classifier, fixed mode (or Regressor — cover at least one Stacking variant)
8. A notebook export for one tuned Training node and one Ensemble node (Task 2 Step 10 changed `_MODELING_STEPS`, `_SPLIT_OR_MODEL`, and `_is_tuning_model` in `_notebook_builders.py`, which distinguish tuned vs fixed for export — this path is only unit-covered otherwise)
9. A multi-branch parallel pipeline run using canonical Classification/Regression nodes with the `execution_mode` toggle set to parallel (this directly exercises the `EXECUTION_MODE_AWARE_TYPES` canonical-recognition bug fix from Task 6 — none of the other scenarios above verify that fix's actual runtime effect, only that the toggle *appears* in the UI)

For each, submit a real pipeline config (via the actual `/pipeline/run` endpoint or direct `PipelineEngine` invocation, matching however the earlier ensemble-crash smoke test in this session was structured) and confirm: job completes with `status: "completed"`, no exceptions, `job_type` in the response is `"training"` or `"tuning"` as expected, and `run_mode`/`model_type` are correctly recorded on the resulting `TrainingJob` row.

If ANY of these 9 fail, this is a genuine regression from this plan's changes — do not proceed to Step 5 until all 9 pass. Use `superpowers:systematic-debugging` if a failure isn't immediately obvious from the error message.

- [ ] **Step 5: Update the changelog and user-facing docs**

Find the currently "Active" series row in `CHANGELOG.md`'s table (as of this plan being written, that's `0.7.x`) and its detail file (`changelog/0.7.x.md`). Add a new paragraph under the current top-of-file version entry (or a new version entry, matching whatever convention the file already uses for consecutive same-day changes — check how the ensemble-UI work earlier this session was appended, and follow the same pattern) describing:
- The legacy `BASIC_TRAINING`/`ADVANCED_TUNING` step_type and the old hidden `BasicTrainingNode`/`AdvancedTuningNode`/generic `TrainingNode` canvas nodes (kept since the original training-unification work for backward compatibility with pre-unification saved canvases) have now been fully removed — every training-family node (Classification, Regression, Text Classification, Segmentation, Ensemble) exclusively emits the canonical `training` step_type + `run_mode` param.
- The `job_type` API/dispatch field (fixed-run vs. tuned-run) has been renamed from `"basic_training"/"advanced_tuning"` to `"training"/"tuning"` for naming consistency, with no behavior change.
- A real, previously-existing bug is now fixed as a side effect: the execution-mode (merge/parallel) toggle, canvas branch coloring, connection-type validation, and perf-overlay bucketing now correctly recognize the canonical Classification/Regression/Text Classification/Training node types (they previously only recognized the old legacy node types and silently didn't apply to the newer canonical nodes).
- This is a dev-only breaking change: pre-existing saved canvases/job history using the old legacy node types or `job_type` values are no longer supported and must be recreated.

Also update `docs/user_guide/segmentation.md:149`, which documents `step_type: "basic_training"` as the accepted value for a Segmentation node's exported config — change it to the canonical `step_type: "training"` + `run_mode` shape produced by Task 4.

Update `CHANGELOG.md`'s summary table row for the active series to mention this cleanup in one clause, matching the existing style of that row.

- [ ] **Step 6: Commit**

```bash
cd /Users/BH7043/Skyulf && git add CHANGELOG.md changelog/ docs/user_guide/segmentation.md && git commit -m "docs(changelog): document full removal of legacy basic_training/advanced_tuning step_type and job_type rename

Co-authored-by: Copilot <223556219+Copilot@users.noreply.github.com>"
```

- [ ] **Step 7: Update the progress ledger**

Append to `.superpowers/sdd/progress.md`:
```
## Legacy training cleanup plan (2026-07-21) — 9 tasks
Task 1: job_type renamed to training/tuning (backend)
Task 2: StepType.BASIC_TRAINING/ADVANCED_TUNING enum + all alias-checks removed (backend)
Task 3: backend final sweep clean
Task 4: SegmentationNode/EnsembleNode migrated to canonical training/run_mode emission
Task 5: legacy BasicTrainingNode/AdvancedTuningNode/generic TrainingNode + settings deleted
Task 6: legacy definitionType alias-checks removed; canonical-type recognition gap fixed
  (execution_mode toggle, branch coloring, connection validation, perf bucketing)
Task 7: frontend job_type renamed to training/tuning end-to-end
Task 8: frontend final sweep clean
Task 9: dev DB wiped, full e2e smoke test (9 scenarios) passed, changelog updated
STATUS: COMPLETE — zero legacy basic_training/advanced_tuning references remain anywhere.
```

---

## Self-Review Notes (for the plan author, not a task)

- **Spec coverage:** every file identified during investigation (backend: constants.py, run_pipeline.py, preview.py, _helpers.py, graph_utils.py, _schema_validator.py, _schema_graph.py, engine/_node_runners.py, engine/__init__.py, notebook_export.py, _notebook_branched.py, _notebook_builders.py, jobs.py, schemas.py, strategies.py, basic_training_manager.py, advanced_tuning_manager.py, _schemas.py; frontend: stepTypes.ts, executionMode.ts, useBranchColors.ts, useGraphStore.ts, useRunControls.ts, perfThresholds.ts, pipelineTemplates.ts, pipelineConverter.ts, EnsembleSettings.tsx, TrainingSettings.tsx, jobs.ts, useTrainingNodeContext.ts, JobCard.tsx, JobDetailsView.tsx, jobMeta.ts, JobListSidebar.tsx, useJobPolling.test.ts, init.ts, BasicTrainingNode.ts, AdvancedTuningNode.ts, TrainingNode.ts, BasicTrainingSettings.tsx, AdvancedTuningSettings.tsx, bodyPreview.test.ts) has an explicit task.
- **Ordering dependency respected (updated per rubber-duck review):** execute in the order **Task 1, Task 4, Task 2, Task 3, Task 5, Task 6, Task 7, Task 8, Task 9** — Task 4 (frontend canonical-emission fix for Segmentation/Ensemble) is moved ahead of Task 2 (backend enum removal) so there is never a window where the frontend emits a step_type the backend has already stopped accepting. Task numbering in the headings below is left as originally written for readability; only the execution order changes (see the Global Constraints note above).
- **Discovered-during-investigation bug fix included:** Task 6 explicitly folds in the `EXECUTION_MODE_AWARE_TYPES`/`MODEL_SOURCE_TYPES`/`modelTypes`/`TRAINING_TYPES`/`getPerfFamily` canonical-type recognition gap fix, since it's the same lines being touched for the legacy-removal work and was called out as a real, currently-existing bug during scoping.
- **No placeholders:** every step shows the exact before/after code, not a description of what to change.
