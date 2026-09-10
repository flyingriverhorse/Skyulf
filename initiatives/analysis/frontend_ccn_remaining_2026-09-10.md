# Remaining frontend CCN violations

**Measured:** 2026-09-10. **Base:** `534d1e94` on branch `0819` plus batch 6 changes.
**Scope:** all TypeScript/TSX under `frontend/ml-canvas/src`, using the
repository ESLint configuration. **Target:** CCN <= 10 per function.
Source locations refer to the batch 6 working tree measured after its refactors.

- **77 files** contain functions above the target.
- **105 functions** exceed CCN 10.
- **Highest CCN: 32.**
- The source-wide strict gate currently fails on these violations.

The accepted frontend limit is 10 for the strict check and subsequent
refactors. The informational report stays at 8: it lists 167 functions in
109 files, including 62 functions at CCN 9 or 10 that pass the strict gate.
Those 62 remain optional improvement candidates and are omitted from the
required backlog below. Batch 6 reduced strict violations from 112 in 80
files to 105 in 77 files; informational violations fell from 172 to 167.

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
| [src/components/eda/VariableCard.tsx](../../frontend/ml-canvas/src/components/eda/VariableCard.tsx) | 32 | 1 | L15:58 = **32** |
| [src/core/store/useGraphStore.ts](../../frontend/ml-canvas/src/core/store/useGraphStore.ts) | 32 | 3 | L178:8 = **17**, L273:1 = **32**, L652:17 = **12** |
| [src/components/Layout.tsx](../../frontend/ml-canvas/src/components/Layout.tsx) | 29 | 1 | L10:33 = **29** |
| [src/components/panels/JobsDrawer.tsx](../../frontend/ml-canvas/src/components/panels/JobsDrawer.tsx) | 28 | 2 | L38:37 = **28**, L153:39 = **13** |
| [src/core/utils/operationalContext.ts](../../frontend/ml-canvas/src/core/utils/operationalContext.ts) | 28 | 3 | L124:8 = **19**, L177:1 = **28**, L276:8 = **11** |
| [src/modules/nodes/modeling/SegmentationSettings.tsx](../../frontend/ml-canvas/src/modules/nodes/modeling/SegmentationSettings.tsx) | 28 | 3 | L55:6 = **28**, L71:33 = **15**, L177:23 = **12** |
| [src/core/utils/graphDiff.ts](../../frontend/ml-canvas/src/core/utils/graphDiff.ts) | 27 | 1 | L154:8 = **27** |
| [src/modules/nodes/processing/ScalingNode.tsx](../../frontend/ml-canvas/src/modules/nodes/processing/ScalingNode.tsx) | 27 | 2 | L32:117 = **27**, L82:21 = **19** |
| [src/components/canvas/ConnectionPort.tsx](../../frontend/ml-canvas/src/components/canvas/ConnectionPort.tsx) | 26 | 1 | L9:36 = **26** |
| [src/components/data/DatasetPreviewModal.tsx](../../frontend/ml-canvas/src/components/data/DatasetPreviewModal.tsx) | 26 | 1 | L39:72 = **26** |
| [src/components/layout/resultsPanel/MergeWarningsBanner.tsx](../../frontend/ml-canvas/src/components/layout/resultsPanel/MergeWarningsBanner.tsx) | 26 | 1 | L53:28 = **26** |
| [src/components/shared/NodeInspectorModal.tsx](../../frontend/ml-canvas/src/components/shared/NodeInspectorModal.tsx) | 26 | 1 | L42:70 = **26** |
| [src/modules/nodes/processing/OutlierNode.tsx](../../frontend/ml-canvas/src/modules/nodes/processing/OutlierNode.tsx) | 26 | 3 | L34:117 = **26**, L94:26 = **22**, L222:33 = **15** |
| [src/components/pages/ExperimentsPage/components/SegmentationView.tsx](../../frontend/ml-canvas/src/components/pages/ExperimentsPage/components/SegmentationView.tsx) | 25 | 1 | L46:50 = **25** |
| [src/pages/ModelRegistry.tsx](../../frontend/ml-canvas/src/pages/ModelRegistry.tsx) | 25 | 1 | L23:40 = **25** |
| [src/pages/drift/DriftTable.tsx](../../frontend/ml-canvas/src/pages/drift/DriftTable.tsx) | 25 | 2 | L172:54 = **12**, L300:34 = **25** |
| [src/components/panels/jobs/JobCard.tsx](../../frontend/ml-canvas/src/components/panels/jobs/JobCard.tsx) | 24 | 1 | L43:48 = **24** |
| [src/modules/nodes/inspection/DataPreviewComponents.tsx](../../frontend/ml-canvas/src/modules/nodes/inspection/DataPreviewComponents.tsx) | 24 | 2 | L27:21 = **14**, L82:136 = **24** |
| [src/pages/DataSources.tsx](../../frontend/ml-canvas/src/pages/DataSources.tsx) | 24 | 3 | L21:38 = **24**, L71:44 = **15**, L384:38 = **20** |
| [src/pages/Jobs.tsx](../../frontend/ml-canvas/src/pages/Jobs.tsx) | 24 | 1 | L32:35 = **24** |
| [src/components/layout/PropertiesPanel.tsx](../../frontend/ml-canvas/src/components/layout/PropertiesPanel.tsx) | 23 | 2 | L21:42 = **11**, L336:64 = **23** |
| [src/components/pages/ExperimentsPage/components/ShapExplainabilityView.tsx](../../frontend/ml-canvas/src/components/pages/ExperimentsPage/components/ShapExplainabilityView.tsx) | 22 | 1 | L49:56 = **22** |
| [src/core/utils/pipelineLeakageValidation.ts](../../frontend/ml-canvas/src/core/utils/pipelineLeakageValidation.ts) | 22 | 2 | L299:1 = **22**, L347:8 = **18** |
| [src/pages/DataDriftPage.tsx](../../frontend/ml-canvas/src/pages/DataDriftPage.tsx) | 22 | 1 | L33:40 = **22** |
| [src/components/pages/ExperimentsPage/components/JobListSidebar.tsx](../../frontend/ml-canvas/src/components/pages/ExperimentsPage/components/JobListSidebar.tsx) | 21 | 2 | L21:48 = **13**, L60:27 = **21** |
| [src/core/utils/connectionValidation.ts](../../frontend/ml-canvas/src/core/utils/connectionValidation.ts) | 21 | 1 | L52:8 = **21** |
| [src/modules/nodes/processing/InvalidValueReplacementNode.tsx](../../frontend/ml-canvas/src/modules/nodes/processing/InvalidValueReplacementNode.tsx) | 21 | 1 | L20:154 = **21** |
| [src/modules/nodes/processing/VectorizerNodes.tsx](../../frontend/ml-canvas/src/modules/nodes/processing/VectorizerNodes.tsx) | 21 | 1 | L105:101 = **21** |
| [src/modules/nodes/shared/ColumnMultiSelect.tsx](../../frontend/ml-canvas/src/modules/nodes/shared/ColumnMultiSelect.tsx) | 21 | 1 | L80:8 = **21** |
| [src/components/pages/ExperimentsPage/components/ClassificationChartsForSplit.tsx](../../frontend/ml-canvas/src/components/pages/ExperimentsPage/components/ClassificationChartsForSplit.tsx) | 20 | 1 | L70:55 = **20** |
| [src/core/hooks/useJobPolling.ts](../../frontend/ml-canvas/src/core/hooks/useJobPolling.ts) | 20 | 1 | L114:22 = **20** |
| [src/core/hooks/useTrainingNodeContext.ts](../../frontend/ml-canvas/src/core/hooks/useTrainingNodeContext.ts) | 20 | 2 | L23:1 = **14**, L82:5 = **20** |
| [src/modules/nodes/processing/DropColumnsNode.tsx](../../frontend/ml-canvas/src/modules/nodes/processing/DropColumnsNode.tsx) | 20 | 1 | L22:129 = **20** |
| [src/components/pages/ExperimentsPage.tsx](../../frontend/ml-canvas/src/components/pages/ExperimentsPage.tsx) | 19 | 1 | L40:42 = **19** |
| [src/components/data/PipelineVersionsModal.tsx](../../frontend/ml-canvas/src/components/data/PipelineVersionsModal.tsx) | 18 | 3 | L51:1 = **18**, L83:76 = **11**, L246:25 = **12** |
| [src/components/eda/tabs/BivariateTab.tsx](../../frontend/ml-canvas/src/components/eda/tabs/BivariateTab.tsx) | 18 | 1 | L31:58 = **18** |
| [src/components/pages/ExperimentsPage/utils/classificationCharts.ts](../../frontend/ml-canvas/src/components/pages/ExperimentsPage/utils/classificationCharts.ts) | 18 | 3 | L220:1 = **13**, L240:1 = **16**, L286:8 = **18** |
| [src/modules/nodes/modeling/components/StrategySettingsModal.tsx](../../frontend/ml-canvas/src/modules/nodes/modeling/components/StrategySettingsModal.tsx) | 18 | 1 | L47:76 = **18** |
| [src/core/utils/format.ts](../../frontend/ml-canvas/src/core/utils/format.ts) | 17 | 2 | L43:8 = **12**, L271:39 = **17** |
| [src/modules/nodes/processing/BinningNode.tsx](../../frontend/ml-canvas/src/modules/nodes/processing/BinningNode.tsx) | 17 | 1 | L56:117 = **17** |
| [src/pages/SlowNodesPage.tsx](../../frontend/ml-canvas/src/pages/SlowNodesPage.tsx) | 17 | 1 | L89:40 = **17** |
| [src/components/canvas/FlowCanvas.tsx](../../frontend/ml-canvas/src/components/canvas/FlowCanvas.tsx) | 16 | 2 | L91:37 = **16**, L288:21 = **12** |
| [src/components/data/AddSourceModal.tsx](../../frontend/ml-canvas/src/components/data/AddSourceModal.tsx) | 16 | 1 | L13:62 = **16** |
| [src/components/data/IngestionJobsModal.tsx](../../frontend/ml-canvas/src/components/data/IngestionJobsModal.tsx) | 16 | 2 | L28:28 = **11**, L106:21 = **16** |
| [src/core/hooks/useKeyboardShortcuts.ts](../../frontend/ml-canvas/src/core/hooks/useKeyboardShortcuts.ts) | 16 | 1 | L123:21 = **16** |
| [src/modules/nodes/processing/CastTypeNode.tsx](../../frontend/ml-canvas/src/modules/nodes/processing/CastTypeNode.tsx) | 16 | 1 | L33:120 = **16** |
| [src/pages/Dashboard.tsx](../../frontend/ml-canvas/src/pages/Dashboard.tsx) | 16 | 1 | L39:36 = **16** |
| [src/components/eda/JobsHistoryModal.tsx](../../frontend/ml-canvas/src/components/eda/JobsHistoryModal.tsx) | 15 | 2 | L39:66 = **15**, L147:30 = **15** |
| [src/components/pages/ExperimentsPage/hooks/useEvaluationFetch.ts](../../frontend/ml-canvas/src/components/pages/ExperimentsPage/hooks/useEvaluationFetch.ts) | 15 | 1 | L56:43 = **15** |
| [src/components/pages/DeploymentsPage.tsx](../../frontend/ml-canvas/src/components/pages/DeploymentsPage.tsx) | 14 | 1 | L61:42 = **14** |
| [src/components/pages/ExperimentsPage/components/RegressionChartsForSplit.tsx](../../frontend/ml-canvas/src/components/pages/ExperimentsPage/components/RegressionChartsForSplit.tsx) | 14 | 1 | L36:58 = **14** |
| [src/core/hooks/useExecutionWarnings.ts](../../frontend/ml-canvas/src/core/hooks/useExecutionWarnings.ts) | 14 | 1 | L27:13 = **14** |
| [src/modules/nodes/modeling/TrainTestSplitNode.tsx](../../frontend/ml-canvas/src/modules/nodes/modeling/TrainTestSplitNode.tsx) | 14 | 1 | L20:138 = **14** |
| [src/modules/nodes/processing/AliasReplacementNode.tsx](../../frontend/ml-canvas/src/modules/nodes/processing/AliasReplacementNode.tsx) | 14 | 1 | L100:144 = **14** |
| [src/components/layout/Navbar.tsx](../../frontend/ml-canvas/src/components/layout/Navbar.tsx) | 13 | 1 | L10:33 = **13** |
| [src/components/pages/ExperimentsPage/components/PerClassConfusionMatrix.tsx](../../frontend/ml-canvas/src/components/pages/ExperimentsPage/components/PerClassConfusionMatrix.tsx) | 13 | 2 | L32:57 = **13**, L157:58 = **13** |
| [src/components/pages/experiments/pipelineDiffLayout.ts](../../frontend/ml-canvas/src/components/pages/experiments/pipelineDiffLayout.ts) | 13 | 1 | L42:45 = **13** |
| [src/core/hooks/useTuningTrials.ts](../../frontend/ml-canvas/src/core/hooks/useTuningTrials.ts) | 13 | 1 | L253:51 = **13** |
| [src/core/utils/pipelineCycleValidation.ts](../../frontend/ml-canvas/src/core/utils/pipelineCycleValidation.ts) | 13 | 1 | L25:8 = **13** |
| [src/modules/nodes/processing/MissingIndicatorNode.tsx](../../frontend/ml-canvas/src/modules/nodes/processing/MissingIndicatorNode.tsx) | 13 | 1 | L17:144 = **13** |
| [src/modules/nodes/processing/PolynomialFeaturesNode.tsx](../../frontend/ml-canvas/src/modules/nodes/processing/PolynomialFeaturesNode.tsx) | 13 | 1 | L48:13 = **13** |
| [src/pages/drift/JobSelector.tsx](../../frontend/ml-canvas/src/pages/drift/JobSelector.tsx) | 13 | 1 | L18:56 = **13** |
| [src/pages/drift/_hooks/useDriftReport.ts](../../frontend/ml-canvas/src/pages/drift/_hooks/useDriftReport.ts) | 13 | 1 | L63:48 = **13** |
| [src/components/eda/tabs/PCATab.tsx](../../frontend/ml-canvas/src/components/eda/tabs/PCATab.tsx) | 12 | 1 | L24:46 = **12** |
| [src/components/pages/ExperimentsPage/utils/jobMeta.ts](../../frontend/ml-canvas/src/components/pages/ExperimentsPage/utils/jobMeta.ts) | 12 | 2 | L78:8 = **11**, L128:8 = **12** |
| [src/core/utils/canvasExport.ts](../../frontend/ml-canvas/src/core/utils/canvasExport.ts) | 12 | 1 | L49:1 = **12** |
| [src/modules/nodes/processing/DeduplicationNode.tsx](../../frontend/ml-canvas/src/modules/nodes/processing/DeduplicationNode.tsx) | 12 | 1 | L17:135 = **12** |
| [src/modules/nodes/processing/DropRowsNode.tsx](../../frontend/ml-canvas/src/modules/nodes/processing/DropRowsNode.tsx) | 12 | 1 | L16:120 = **12** |
| [src/modules/nodes/processing/TimeSeriesNode.tsx](../../frontend/ml-canvas/src/modules/nodes/processing/TimeSeriesNode.tsx) | 12 | 1 | L284:1 = **12** |
| [src/components/layout/NotificationCenter.tsx](../../frontend/ml-canvas/src/components/layout/NotificationCenter.tsx) | 11 | 1 | L49:6 = **11** |
| [src/components/shared/ModalShell.tsx](../../frontend/ml-canvas/src/components/shared/ModalShell.tsx) | 11 | 1 | L50:54 = **11** |
| [src/core/utils/canvasLeakageIssues.ts](../../frontend/ml-canvas/src/core/utils/canvasLeakageIssues.ts) | 11 | 1 | L9:1 = **11** |
| [src/modules/nodes/modeling/components/HyperparameterInput.tsx](../../frontend/ml-canvas/src/modules/nodes/modeling/components/HyperparameterInput.tsx) | 11 | 1 | L27:24 = **11** |
| [src/modules/nodes/modeling/components/SearchSpaceInput.tsx](../../frontend/ml-canvas/src/modules/nodes/modeling/components/SearchSpaceInput.tsx) | 11 | 1 | L17:66 = **11** |
| [src/modules/nodes/processing/FeatureInteractionNode.tsx](../../frontend/ml-canvas/src/modules/nodes/processing/FeatureInteractionNode.tsx) | 11 | 1 | L44:13 = **11** |
| [src/modules/nodes/processing/TransformationNode.tsx](../../frontend/ml-canvas/src/modules/nodes/processing/TransformationNode.tsx) | 11 | 1 | L141:30 = **11** |
| [src/pages/drift/SummaryCards.tsx](../../frontend/ml-canvas/src/pages/drift/SummaryCards.tsx) | 11 | 1 | L10:58 = **11** |

## Function details, highest CCN first

Function descriptions below are the ESLint labels. Anonymous callbacks are
identified by their exact line and column; follow the source link to inspect
the surrounding component or callback assignment.

| CCN | File / line | Column | Function (ESLint label) |
|---:|---|---:|---|
| 32 | [src/components/eda/VariableCard.tsx:15](../../frontend/ml-canvas/src/components/eda/VariableCard.tsx#L15) | 58 | Arrow function |
| 32 | [src/core/store/useGraphStore.ts:273](../../frontend/ml-canvas/src/core/store/useGraphStore.ts#L273) | 1 | Function 'confirmConnection' |
| 29 | [src/components/Layout.tsx:10](../../frontend/ml-canvas/src/components/Layout.tsx#L10) | 33 | Arrow function |
| 28 | [src/components/panels/JobsDrawer.tsx:38](../../frontend/ml-canvas/src/components/panels/JobsDrawer.tsx#L38) | 37 | Arrow function |
| 28 | [src/core/utils/operationalContext.ts:177](../../frontend/ml-canvas/src/core/utils/operationalContext.ts#L177) | 1 | Function 'parseRef' |
| 28 | [src/modules/nodes/modeling/SegmentationSettings.tsx:55](../../frontend/ml-canvas/src/modules/nodes/modeling/SegmentationSettings.tsx#L55) | 6 | Arrow function |
| 27 | [src/core/utils/graphDiff.ts:154](../../frontend/ml-canvas/src/core/utils/graphDiff.ts#L154) | 8 | Function 'diffGraphs' |
| 27 | [src/modules/nodes/processing/ScalingNode.tsx:32](../../frontend/ml-canvas/src/modules/nodes/processing/ScalingNode.tsx#L32) | 117 | Arrow function |
| 26 | [src/components/canvas/ConnectionPort.tsx:9](../../frontend/ml-canvas/src/components/canvas/ConnectionPort.tsx#L9) | 36 | Function 'ConnectionPort' |
| 26 | [src/components/data/DatasetPreviewModal.tsx:39](../../frontend/ml-canvas/src/components/data/DatasetPreviewModal.tsx#L39) | 72 | Arrow function |
| 26 | [src/components/layout/resultsPanel/MergeWarningsBanner.tsx:53](../../frontend/ml-canvas/src/components/layout/resultsPanel/MergeWarningsBanner.tsx#L53) | 28 | Arrow function |
| 26 | [src/components/shared/NodeInspectorModal.tsx:42](../../frontend/ml-canvas/src/components/shared/NodeInspectorModal.tsx#L42) | 70 | Arrow function |
| 26 | [src/modules/nodes/processing/OutlierNode.tsx:34](../../frontend/ml-canvas/src/modules/nodes/processing/OutlierNode.tsx#L34) | 117 | Arrow function |
| 25 | [src/components/pages/ExperimentsPage/components/SegmentationView.tsx:46](../../frontend/ml-canvas/src/components/pages/ExperimentsPage/components/SegmentationView.tsx#L46) | 50 | Arrow function |
| 25 | [src/pages/ModelRegistry.tsx:23](../../frontend/ml-canvas/src/pages/ModelRegistry.tsx#L23) | 40 | Arrow function |
| 25 | [src/pages/drift/DriftTable.tsx:300](../../frontend/ml-canvas/src/pages/drift/DriftTable.tsx#L300) | 34 | Arrow function |
| 24 | [src/components/panels/jobs/JobCard.tsx:43](../../frontend/ml-canvas/src/components/panels/jobs/JobCard.tsx#L43) | 48 | Arrow function |
| 24 | [src/modules/nodes/inspection/DataPreviewComponents.tsx:82](../../frontend/ml-canvas/src/modules/nodes/inspection/DataPreviewComponents.tsx#L82) | 136 | Arrow function |
| 24 | [src/pages/DataSources.tsx:21](../../frontend/ml-canvas/src/pages/DataSources.tsx#L21) | 38 | Arrow function |
| 24 | [src/pages/Jobs.tsx:32](../../frontend/ml-canvas/src/pages/Jobs.tsx#L32) | 35 | Arrow function |
| 23 | [src/components/layout/PropertiesPanel.tsx:336](../../frontend/ml-canvas/src/components/layout/PropertiesPanel.tsx#L336) | 64 | Arrow function |
| 22 | [src/components/pages/ExperimentsPage/components/ShapExplainabilityView.tsx:49](../../frontend/ml-canvas/src/components/pages/ExperimentsPage/components/ShapExplainabilityView.tsx#L49) | 56 | Arrow function |
| 22 | [src/core/utils/pipelineLeakageValidation.ts:299](../../frontend/ml-canvas/src/core/utils/pipelineLeakageValidation.ts#L299) | 1 | Function 'isFixedOperation' |
| 22 | [src/modules/nodes/processing/OutlierNode.tsx:94](../../frontend/ml-canvas/src/modules/nodes/processing/OutlierNode.tsx#L94) | 26 | Arrow function |
| 22 | [src/pages/DataDriftPage.tsx:33](../../frontend/ml-canvas/src/pages/DataDriftPage.tsx#L33) | 40 | Arrow function |
| 21 | [src/components/pages/ExperimentsPage/components/JobListSidebar.tsx:60](../../frontend/ml-canvas/src/components/pages/ExperimentsPage/components/JobListSidebar.tsx#L60) | 27 | Arrow function |
| 21 | [src/core/utils/connectionValidation.ts:52](../../frontend/ml-canvas/src/core/utils/connectionValidation.ts#L52) | 8 | Function 'connectionIssue' |
| 21 | [src/modules/nodes/processing/InvalidValueReplacementNode.tsx:20](../../frontend/ml-canvas/src/modules/nodes/processing/InvalidValueReplacementNode.tsx#L20) | 154 | Arrow function |
| 21 | [src/modules/nodes/processing/VectorizerNodes.tsx:105](../../frontend/ml-canvas/src/modules/nodes/processing/VectorizerNodes.tsx#L105) | 101 | Arrow function |
| 21 | [src/modules/nodes/shared/ColumnMultiSelect.tsx:80](../../frontend/ml-canvas/src/modules/nodes/shared/ColumnMultiSelect.tsx#L80) | 8 | Function 'ColumnMultiSelect' |
| 20 | [src/components/pages/ExperimentsPage/components/ClassificationChartsForSplit.tsx:70](../../frontend/ml-canvas/src/components/pages/ExperimentsPage/components/ClassificationChartsForSplit.tsx#L70) | 55 | Arrow function |
| 20 | [src/core/hooks/useJobPolling.ts:114](../../frontend/ml-canvas/src/core/hooks/useJobPolling.ts#L114) | 22 | Async arrow function |
| 20 | [src/core/hooks/useTrainingNodeContext.ts:82](../../frontend/ml-canvas/src/core/hooks/useTrainingNodeContext.ts#L82) | 5 | Async arrow function |
| 20 | [src/modules/nodes/processing/DropColumnsNode.tsx:22](../../frontend/ml-canvas/src/modules/nodes/processing/DropColumnsNode.tsx#L22) | 129 | Arrow function |
| 20 | [src/pages/DataSources.tsx:384](../../frontend/ml-canvas/src/pages/DataSources.tsx#L384) | 38 | Arrow function |
| 19 | [src/components/pages/ExperimentsPage.tsx:40](../../frontend/ml-canvas/src/components/pages/ExperimentsPage.tsx#L40) | 42 | Arrow function |
| 19 | [src/core/utils/operationalContext.ts:124](../../frontend/ml-canvas/src/core/utils/operationalContext.ts#L124) | 8 | Function 'serializeOperationalContext' |
| 19 | [src/modules/nodes/processing/ScalingNode.tsx:82](../../frontend/ml-canvas/src/modules/nodes/processing/ScalingNode.tsx#L82) | 21 | Arrow function |
| 18 | [src/components/data/PipelineVersionsModal.tsx:51](../../frontend/ml-canvas/src/components/data/PipelineVersionsModal.tsx#L51) | 1 | Function 'summariseGraph' |
| 18 | [src/components/eda/tabs/BivariateTab.tsx:31](../../frontend/ml-canvas/src/components/eda/tabs/BivariateTab.tsx#L31) | 58 | Arrow function |
| 18 | [src/components/pages/ExperimentsPage/utils/classificationCharts.ts:286](../../frontend/ml-canvas/src/components/pages/ExperimentsPage/utils/classificationCharts.ts#L286) | 8 | Function 'findBestThreshold' |
| 18 | [src/core/utils/pipelineLeakageValidation.ts:347](../../frontend/ml-canvas/src/core/utils/pipelineLeakageValidation.ts#L347) | 8 | Function 'findPreprocessingBeforeSplitIssues' |
| 18 | [src/modules/nodes/modeling/components/StrategySettingsModal.tsx:47](../../frontend/ml-canvas/src/modules/nodes/modeling/components/StrategySettingsModal.tsx#L47) | 76 | Arrow function |
| 17 | [src/core/store/useGraphStore.ts:178](../../frontend/ml-canvas/src/core/store/useGraphStore.ts#L178) | 8 | Function 'collectGraphValidationIssues' |
| 17 | [src/core/utils/format.ts:271](../../frontend/ml-canvas/src/core/utils/format.ts#L271) | 39 | Arrow function |
| 17 | [src/modules/nodes/processing/BinningNode.tsx:56](../../frontend/ml-canvas/src/modules/nodes/processing/BinningNode.tsx#L56) | 117 | Arrow function |
| 17 | [src/pages/SlowNodesPage.tsx:89](../../frontend/ml-canvas/src/pages/SlowNodesPage.tsx#L89) | 40 | Arrow function |
| 16 | [src/components/canvas/FlowCanvas.tsx:91](../../frontend/ml-canvas/src/components/canvas/FlowCanvas.tsx#L91) | 37 | Arrow function |
| 16 | [src/components/data/AddSourceModal.tsx:13](../../frontend/ml-canvas/src/components/data/AddSourceModal.tsx#L13) | 62 | Arrow function |
| 16 | [src/components/data/IngestionJobsModal.tsx:106](../../frontend/ml-canvas/src/components/data/IngestionJobsModal.tsx#L106) | 21 | Arrow function |
| 16 | [src/components/pages/ExperimentsPage/utils/classificationCharts.ts:240](../../frontend/ml-canvas/src/components/pages/ExperimentsPage/utils/classificationCharts.ts#L240) | 1 | Function 'multiclassMetricValue' |
| 16 | [src/core/hooks/useKeyboardShortcuts.ts:123](../../frontend/ml-canvas/src/core/hooks/useKeyboardShortcuts.ts#L123) | 21 | Arrow function |
| 16 | [src/modules/nodes/processing/CastTypeNode.tsx:33](../../frontend/ml-canvas/src/modules/nodes/processing/CastTypeNode.tsx#L33) | 120 | Arrow function |
| 16 | [src/pages/Dashboard.tsx:39](../../frontend/ml-canvas/src/pages/Dashboard.tsx#L39) | 36 | Arrow function |
| 15 | [src/components/eda/JobsHistoryModal.tsx:39](../../frontend/ml-canvas/src/components/eda/JobsHistoryModal.tsx#L39) | 66 | Arrow function |
| 15 | [src/components/eda/JobsHistoryModal.tsx:147](../../frontend/ml-canvas/src/components/eda/JobsHistoryModal.tsx#L147) | 30 | Arrow function |
| 15 | [src/components/pages/ExperimentsPage/hooks/useEvaluationFetch.ts:56](../../frontend/ml-canvas/src/components/pages/ExperimentsPage/hooks/useEvaluationFetch.ts#L56) | 43 | Async arrow function |
| 15 | [src/modules/nodes/modeling/SegmentationSettings.tsx:71](../../frontend/ml-canvas/src/modules/nodes/modeling/SegmentationSettings.tsx#L71) | 33 | Arrow function |
| 15 | [src/modules/nodes/processing/OutlierNode.tsx:222](../../frontend/ml-canvas/src/modules/nodes/processing/OutlierNode.tsx#L222) | 33 | Arrow function |
| 15 | [src/pages/DataSources.tsx:71](../../frontend/ml-canvas/src/pages/DataSources.tsx#L71) | 44 | Arrow function |
| 14 | [src/components/pages/DeploymentsPage.tsx:61](../../frontend/ml-canvas/src/components/pages/DeploymentsPage.tsx#L61) | 42 | Arrow function |
| 14 | [src/components/pages/ExperimentsPage/components/RegressionChartsForSplit.tsx:36](../../frontend/ml-canvas/src/components/pages/ExperimentsPage/components/RegressionChartsForSplit.tsx#L36) | 58 | Arrow function |
| 14 | [src/core/hooks/useExecutionWarnings.ts:27](../../frontend/ml-canvas/src/core/hooks/useExecutionWarnings.ts#L27) | 13 | Arrow function |
| 14 | [src/core/hooks/useTrainingNodeContext.ts:23](../../frontend/ml-canvas/src/core/hooks/useTrainingNodeContext.ts#L23) | 1 | Function 'findUpstreamDatasetId' |
| 14 | [src/modules/nodes/inspection/DataPreviewComponents.tsx:27](../../frontend/ml-canvas/src/modules/nodes/inspection/DataPreviewComponents.tsx#L27) | 21 | Arrow function |
| 14 | [src/modules/nodes/modeling/TrainTestSplitNode.tsx:20](../../frontend/ml-canvas/src/modules/nodes/modeling/TrainTestSplitNode.tsx#L20) | 138 | Arrow function |
| 14 | [src/modules/nodes/processing/AliasReplacementNode.tsx:100](../../frontend/ml-canvas/src/modules/nodes/processing/AliasReplacementNode.tsx#L100) | 144 | Arrow function |
| 13 | [src/components/layout/Navbar.tsx:10](../../frontend/ml-canvas/src/components/layout/Navbar.tsx#L10) | 33 | Arrow function |
| 13 | [src/components/pages/ExperimentsPage/components/JobListSidebar.tsx:21](../../frontend/ml-canvas/src/components/pages/ExperimentsPage/components/JobListSidebar.tsx#L21) | 48 | Arrow function |
| 13 | [src/components/pages/ExperimentsPage/components/PerClassConfusionMatrix.tsx:32](../../frontend/ml-canvas/src/components/pages/ExperimentsPage/components/PerClassConfusionMatrix.tsx#L32) | 57 | Arrow function |
| 13 | [src/components/pages/ExperimentsPage/components/PerClassConfusionMatrix.tsx:157](../../frontend/ml-canvas/src/components/pages/ExperimentsPage/components/PerClassConfusionMatrix.tsx#L157) | 58 | Arrow function |
| 13 | [src/components/pages/ExperimentsPage/utils/classificationCharts.ts:220](../../frontend/ml-canvas/src/components/pages/ExperimentsPage/utils/classificationCharts.ts#L220) | 1 | Function 'binaryMetricValue' |
| 13 | [src/components/pages/experiments/pipelineDiffLayout.ts:42](../../frontend/ml-canvas/src/components/pages/experiments/pipelineDiffLayout.ts#L42) | 45 | Arrow function |
| 13 | [src/components/panels/JobsDrawer.tsx:153](../../frontend/ml-canvas/src/components/panels/JobsDrawer.tsx#L153) | 39 | Arrow function |
| 13 | [src/core/hooks/useTuningTrials.ts:253](../../frontend/ml-canvas/src/core/hooks/useTuningTrials.ts#L253) | 51 | Arrow function |
| 13 | [src/core/utils/pipelineCycleValidation.ts:25](../../frontend/ml-canvas/src/core/utils/pipelineCycleValidation.ts#L25) | 8 | Function 'findCycleIssues' |
| 13 | [src/modules/nodes/processing/MissingIndicatorNode.tsx:17](../../frontend/ml-canvas/src/modules/nodes/processing/MissingIndicatorNode.tsx#L17) | 144 | Arrow function |
| 13 | [src/modules/nodes/processing/PolynomialFeaturesNode.tsx:48](../../frontend/ml-canvas/src/modules/nodes/processing/PolynomialFeaturesNode.tsx#L48) | 13 | Method 'settings' |
| 13 | [src/pages/drift/JobSelector.tsx:18](../../frontend/ml-canvas/src/pages/drift/JobSelector.tsx#L18) | 56 | Arrow function |
| 13 | [src/pages/drift/_hooks/useDriftReport.ts:63](../../frontend/ml-canvas/src/pages/drift/_hooks/useDriftReport.ts#L63) | 48 | Arrow function |
| 12 | [src/components/canvas/FlowCanvas.tsx:288](../../frontend/ml-canvas/src/components/canvas/FlowCanvas.tsx#L288) | 21 | Arrow function |
| 12 | [src/components/data/PipelineVersionsModal.tsx:246](../../frontend/ml-canvas/src/components/data/PipelineVersionsModal.tsx#L246) | 25 | Arrow function |
| 12 | [src/components/eda/tabs/PCATab.tsx:24](../../frontend/ml-canvas/src/components/eda/tabs/PCATab.tsx#L24) | 46 | Arrow function |
| 12 | [src/components/pages/ExperimentsPage/utils/jobMeta.ts:128](../../frontend/ml-canvas/src/components/pages/ExperimentsPage/utils/jobMeta.ts#L128) | 8 | Function 'getDisplayScore' |
| 12 | [src/core/store/useGraphStore.ts:652](../../frontend/ml-canvas/src/core/store/useGraphStore.ts#L652) | 17 | Method 'equality' |
| 12 | [src/core/utils/canvasExport.ts:49](../../frontend/ml-canvas/src/core/utils/canvasExport.ts#L49) | 1 | Function 'computeFitTransform' |
| 12 | [src/core/utils/format.ts:43](../../frontend/ml-canvas/src/core/utils/format.ts#L43) | 8 | Function 'getMetricDescription' |
| 12 | [src/modules/nodes/modeling/SegmentationSettings.tsx:177](../../frontend/ml-canvas/src/modules/nodes/modeling/SegmentationSettings.tsx#L177) | 23 | Async arrow function |
| 12 | [src/modules/nodes/processing/DeduplicationNode.tsx:17](../../frontend/ml-canvas/src/modules/nodes/processing/DeduplicationNode.tsx#L17) | 135 | Arrow function |
| 12 | [src/modules/nodes/processing/DropRowsNode.tsx:16](../../frontend/ml-canvas/src/modules/nodes/processing/DropRowsNode.tsx#L16) | 120 | Arrow function |
| 12 | [src/modules/nodes/processing/TimeSeriesNode.tsx:284](../../frontend/ml-canvas/src/modules/nodes/processing/TimeSeriesNode.tsx#L284) | 1 | Function 'validateTimeSeries' |
| 12 | [src/pages/drift/DriftTable.tsx:172](../../frontend/ml-canvas/src/pages/drift/DriftTable.tsx#L172) | 54 | Arrow function |
| 11 | [src/components/data/IngestionJobsModal.tsx:28](../../frontend/ml-canvas/src/components/data/IngestionJobsModal.tsx#L28) | 28 | Arrow function |
| 11 | [src/components/data/PipelineVersionsModal.tsx:83](../../frontend/ml-canvas/src/components/data/PipelineVersionsModal.tsx#L83) | 76 | Arrow function |
| 11 | [src/components/layout/NotificationCenter.tsx:49](../../frontend/ml-canvas/src/components/layout/NotificationCenter.tsx#L49) | 6 | Arrow function |
| 11 | [src/components/layout/PropertiesPanel.tsx:21](../../frontend/ml-canvas/src/components/layout/PropertiesPanel.tsx#L21) | 42 | Arrow function |
| 11 | [src/components/pages/ExperimentsPage/utils/jobMeta.ts:78](../../frontend/ml-canvas/src/components/pages/ExperimentsPage/utils/jobMeta.ts#L78) | 8 | Function 'mapJobMetricToDropdown' |
| 11 | [src/components/shared/ModalShell.tsx:50](../../frontend/ml-canvas/src/components/shared/ModalShell.tsx#L50) | 54 | Arrow function |
| 11 | [src/core/utils/canvasLeakageIssues.ts:9](../../frontend/ml-canvas/src/core/utils/canvasLeakageIssues.ts#L9) | 1 | Function 'pathEdges' |
| 11 | [src/core/utils/operationalContext.ts:276](../../frontend/ml-canvas/src/core/utils/operationalContext.ts#L276) | 8 | Function 'describeOperationalRef' |
| 11 | [src/modules/nodes/modeling/components/HyperparameterInput.tsx:27](../../frontend/ml-canvas/src/modules/nodes/modeling/components/HyperparameterInput.tsx#L27) | 24 | Arrow function |
| 11 | [src/modules/nodes/modeling/components/SearchSpaceInput.tsx:17](../../frontend/ml-canvas/src/modules/nodes/modeling/components/SearchSpaceInput.tsx#L17) | 66 | Arrow function |
| 11 | [src/modules/nodes/processing/FeatureInteractionNode.tsx:44](../../frontend/ml-canvas/src/modules/nodes/processing/FeatureInteractionNode.tsx#L44) | 13 | Method 'settings' |
| 11 | [src/modules/nodes/processing/TransformationNode.tsx:141](../../frontend/ml-canvas/src/modules/nodes/processing/TransformationNode.tsx#L141) | 30 | Arrow function |
| 11 | [src/pages/drift/SummaryCards.tsx:10](../../frontend/ml-canvas/src/pages/drift/SummaryCards.tsx#L10) | 58 | Arrow function |
