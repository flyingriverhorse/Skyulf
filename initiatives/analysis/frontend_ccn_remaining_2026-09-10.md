# Remaining frontend CCN violations

**Measured:** 2026-09-10. **Base:** `c56d6cca` on branch `0819` plus batch 10 changes.
**Scope:** all TypeScript/TSX under `frontend/ml-canvas/src`, using the
repository ESLint configuration. **Target:** CCN <= 10 per function.
Source locations refer to the batch 10 working tree measured after its refactors.

- **20 files** contain functions above the target.
- **22 functions** exceed CCN 10.
- **Highest CCN: 24.**
- The source-wide strict gate currently fails on these violations.

The accepted frontend limit is 10 for the strict check and subsequent
refactors. The informational report stays at 8: it lists 139 functions in
101 files, including 117 functions at CCN 9 or 10 that pass the strict gate.
Those 117 remain optional improvement candidates and are omitted from the
required backlog below. Batch 10 reduced strict violations from 56 in 44
files to 22 in 20 files; informational violations changed from 147 to 139.

This is a measured snapshot, not a list of confirmed bugs or a prescribed
refactor order. Prefer cohesive simplification and relevant behavior checks;
CCN alone does not establish incorrect behavior. Files without violations
are omitted. New source files automatically enter the same check.

## Reproduce and update

Run from `frontend/ml-canvas`:

```powershell
npm.cmd run complexity:report
npm.cmd run complexity:check
```

The report uses CCN 8 and emits informational warnings. The strict command
uses CCN 10 and exits 1 while violations remain. This inventory comes from
the strict command, so its totals intentionally differ from the report.
After another fix batch, rerun both and refresh this inventory. Line
numbers refer to the measured source tree above and can move after edits.

## Files, highest CCN first

The maximum is the largest function CCN in that file, not a file-wide sum.

| File | Highest CCN | Functions > 10 | Violations (line:column = CCN) |
|---|---:|---:|---|
| [src/pages/Jobs.tsx](../../frontend/ml-canvas/src/pages/Jobs.tsx) | 24 | 1 | L32:35 = **24** |
| [src/modules/nodes/processing/VectorizerNodes.tsx](../../frontend/ml-canvas/src/modules/nodes/processing/VectorizerNodes.tsx) | 21 | 1 | L105:101 = **21** |
| [src/core/hooks/useJobPolling.ts](../../frontend/ml-canvas/src/core/hooks/useJobPolling.ts) | 20 | 1 | L114:22 = **20** |
| [src/core/hooks/useTrainingNodeContext.ts](../../frontend/ml-canvas/src/core/hooks/useTrainingNodeContext.ts) | 20 | 2 | L23:1 = **14**, L82:5 = **20** |
| [src/components/pages/ExperimentsPage.tsx](../../frontend/ml-canvas/src/components/pages/ExperimentsPage.tsx) | 19 | 1 | L40:42 = **19** |
| [src/modules/nodes/modeling/components/StrategySettingsModal.tsx](../../frontend/ml-canvas/src/modules/nodes/modeling/components/StrategySettingsModal.tsx) | 18 | 1 | L47:76 = **18** |
| [src/pages/SlowNodesPage.tsx](../../frontend/ml-canvas/src/pages/SlowNodesPage.tsx) | 17 | 1 | L89:40 = **17** |
| [src/pages/Dashboard.tsx](../../frontend/ml-canvas/src/pages/Dashboard.tsx) | 16 | 1 | L39:36 = **16** |
| [src/components/eda/JobsHistoryModal.tsx](../../frontend/ml-canvas/src/components/eda/JobsHistoryModal.tsx) | 15 | 2 | L39:66 = **15**, L147:30 = **15** |
| [src/components/pages/ExperimentsPage/hooks/useEvaluationFetch.ts](../../frontend/ml-canvas/src/components/pages/ExperimentsPage/hooks/useEvaluationFetch.ts) | 15 | 1 | L56:43 = **15** |
| [src/core/hooks/useExecutionWarnings.ts](../../frontend/ml-canvas/src/core/hooks/useExecutionWarnings.ts) | 14 | 1 | L27:13 = **14** |
| [src/modules/nodes/modeling/TrainTestSplitNode.tsx](../../frontend/ml-canvas/src/modules/nodes/modeling/TrainTestSplitNode.tsx) | 14 | 1 | L20:138 = **14** |
| [src/components/layout/Navbar.tsx](../../frontend/ml-canvas/src/components/layout/Navbar.tsx) | 13 | 1 | L10:33 = **13** |
| [src/core/hooks/useTuningTrials.ts](../../frontend/ml-canvas/src/core/hooks/useTuningTrials.ts) | 13 | 1 | L253:51 = **13** |
| [src/modules/nodes/processing/TimeSeriesNode.tsx](../../frontend/ml-canvas/src/modules/nodes/processing/TimeSeriesNode.tsx) | 12 | 1 | L284:1 = **12** |
| [src/components/layout/NotificationCenter.tsx](../../frontend/ml-canvas/src/components/layout/NotificationCenter.tsx) | 11 | 1 | L49:6 = **11** |
| [src/components/shared/ModalShell.tsx](../../frontend/ml-canvas/src/components/shared/ModalShell.tsx) | 11 | 1 | L50:54 = **11** |
| [src/modules/nodes/modeling/components/HyperparameterInput.tsx](../../frontend/ml-canvas/src/modules/nodes/modeling/components/HyperparameterInput.tsx) | 11 | 1 | L27:24 = **11** |
| [src/modules/nodes/modeling/components/SearchSpaceInput.tsx](../../frontend/ml-canvas/src/modules/nodes/modeling/components/SearchSpaceInput.tsx) | 11 | 1 | L17:66 = **11** |
| [src/modules/nodes/processing/TransformationNode.tsx](../../frontend/ml-canvas/src/modules/nodes/processing/TransformationNode.tsx) | 11 | 1 | L141:30 = **11** |

## Function details, highest CCN first

Function descriptions below are the ESLint labels. Anonymous callbacks are
identified by their exact line and column; follow the source link to inspect
the surrounding component or callback assignment.

| CCN | File / line | Column | Function (ESLint label) |
|---:|---|---:|---|
| 24 | [src/pages/Jobs.tsx:32](../../frontend/ml-canvas/src/pages/Jobs.tsx#L32) | 35 | Arrow function |
| 21 | [src/modules/nodes/processing/VectorizerNodes.tsx:105](../../frontend/ml-canvas/src/modules/nodes/processing/VectorizerNodes.tsx#L105) | 101 | Arrow function |
| 20 | [src/core/hooks/useJobPolling.ts:114](../../frontend/ml-canvas/src/core/hooks/useJobPolling.ts#L114) | 22 | Async arrow function |
| 20 | [src/core/hooks/useTrainingNodeContext.ts:82](../../frontend/ml-canvas/src/core/hooks/useTrainingNodeContext.ts#L82) | 5 | Async arrow function |
| 19 | [src/components/pages/ExperimentsPage.tsx:40](../../frontend/ml-canvas/src/components/pages/ExperimentsPage.tsx#L40) | 42 | Arrow function |
| 18 | [src/modules/nodes/modeling/components/StrategySettingsModal.tsx:47](../../frontend/ml-canvas/src/modules/nodes/modeling/components/StrategySettingsModal.tsx#L47) | 76 | Arrow function |
| 17 | [src/pages/SlowNodesPage.tsx:89](../../frontend/ml-canvas/src/pages/SlowNodesPage.tsx#L89) | 40 | Arrow function |
| 16 | [src/pages/Dashboard.tsx:39](../../frontend/ml-canvas/src/pages/Dashboard.tsx#L39) | 36 | Arrow function |
| 15 | [src/components/eda/JobsHistoryModal.tsx:39](../../frontend/ml-canvas/src/components/eda/JobsHistoryModal.tsx#L39) | 66 | Arrow function |
| 15 | [src/components/eda/JobsHistoryModal.tsx:147](../../frontend/ml-canvas/src/components/eda/JobsHistoryModal.tsx#L147) | 30 | Arrow function |
| 15 | [src/components/pages/ExperimentsPage/hooks/useEvaluationFetch.ts:56](../../frontend/ml-canvas/src/components/pages/ExperimentsPage/hooks/useEvaluationFetch.ts#L56) | 43 | Async arrow function |
| 14 | [src/core/hooks/useExecutionWarnings.ts:27](../../frontend/ml-canvas/src/core/hooks/useExecutionWarnings.ts#L27) | 13 | Arrow function |
| 14 | [src/core/hooks/useTrainingNodeContext.ts:23](../../frontend/ml-canvas/src/core/hooks/useTrainingNodeContext.ts#L23) | 1 | Function 'findUpstreamDatasetId' |
| 14 | [src/modules/nodes/modeling/TrainTestSplitNode.tsx:20](../../frontend/ml-canvas/src/modules/nodes/modeling/TrainTestSplitNode.tsx#L20) | 138 | Arrow function |
| 13 | [src/components/layout/Navbar.tsx:10](../../frontend/ml-canvas/src/components/layout/Navbar.tsx#L10) | 33 | Arrow function |
| 13 | [src/core/hooks/useTuningTrials.ts:253](../../frontend/ml-canvas/src/core/hooks/useTuningTrials.ts#L253) | 51 | Arrow function |
| 12 | [src/modules/nodes/processing/TimeSeriesNode.tsx:284](../../frontend/ml-canvas/src/modules/nodes/processing/TimeSeriesNode.tsx#L284) | 1 | Function 'validateTimeSeries' |
| 11 | [src/components/layout/NotificationCenter.tsx:49](../../frontend/ml-canvas/src/components/layout/NotificationCenter.tsx#L49) | 6 | Arrow function |
| 11 | [src/components/shared/ModalShell.tsx:50](../../frontend/ml-canvas/src/components/shared/ModalShell.tsx#L50) | 54 | Arrow function |
| 11 | [src/modules/nodes/modeling/components/HyperparameterInput.tsx:27](../../frontend/ml-canvas/src/modules/nodes/modeling/components/HyperparameterInput.tsx#L27) | 24 | Arrow function |
| 11 | [src/modules/nodes/modeling/components/SearchSpaceInput.tsx:17](../../frontend/ml-canvas/src/modules/nodes/modeling/components/SearchSpaceInput.tsx#L17) | 66 | Arrow function |
| 11 | [src/modules/nodes/processing/TransformationNode.tsx:141](../../frontend/ml-canvas/src/modules/nodes/processing/TransformationNode.tsx#L141) | 30 | Arrow function |
