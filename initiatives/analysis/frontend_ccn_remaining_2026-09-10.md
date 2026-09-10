# Frontend CCN status and optional candidates

**Measured:** 2026-09-10. **Base:** `5a63fe55` on branch `0819` plus the static-analysis follow-up.
**Scope:** all TypeScript/TSX under `frontend/ml-canvas/src`, using the
repository ESLint configuration. **Required limit:** CCN <= 10 per function.

- **0 functions in 0 files exceed CCN 10.** The source-wide strict gate passes.
- **Highest source function CCN: 10.**
- Batch 11 removed the final **22 violations in 20 files** (previous maximum 24).
- The informational CCN 8 report lists **133 functions in 96 files**.
- All 133 are optional CCN 9/10 improvement candidates; none fail the gate.

The required CCN backlog is complete. Keep the strict limit at 10 and the
informational report at 8. Readable functions at 9 or 10 need no automatic
refactor; improve them when responsibility or testing concerns justify it.
CCN is a control-flow metric, not evidence that a function is correct or faulty.

The separate [audit queue](opus_core_analysis-open_queue.md) still tracks
functional findings and deferred work. Completing CCN cleanup does not close
those findings. See the [final batch plan](frontend_ccn_refactor_batch11_2026-09-10.md)
for preserved behavior and executed verification.
The [static-analysis follow-up](frontend_static_analysis_review_2026-09-10.md)
explains file-level deltas, lookup cleanup and job-log regex fixes; the counts remain unchanged.

## Reproduce and update

Run from `frontend/ml-canvas`:

```powershell
npm.cmd run complexity:check
npm.cmd run complexity:report
```

The strict command exits 0 at CCN <= 10. The report emits warnings above 8.
Both use the same source scope; new source files automatically enter the checks.
Refresh this snapshot after changes; line numbers below refer to this measured tree.

## Optional files, highest CCN first

The maximum is the largest function CCN in that file, not a file-wide sum.

| File | Highest CCN | Functions at 9/10 | Locations (line:column = CCN) |
|---|---:|---:|---|
| [src/components/canvas/ConnectionGuidance.tsx](../../frontend/ml-canvas/src/components/canvas/ConnectionGuidance.tsx) | 10 | 1 | L7:8 = **10** |
| [src/components/canvas/FlowCanvas.tsx](../../frontend/ml-canvas/src/components/canvas/FlowCanvas.tsx) | 10 | 3 | L303:21 = **9**, L345:26 = **10**, L448:23 = **10** |
| [src/components/data/AddSourceModal.tsx](../../frontend/ml-canvas/src/components/data/AddSourceModal.tsx) | 10 | 1 | L34:24 = **10** |
| [src/components/data/DatasetPreviewModal.tsx](../../frontend/ml-canvas/src/components/data/DatasetPreviewModal.tsx) | 10 | 1 | L227:23 = **10** |
| [src/components/data/PipelineVersionsModal.tsx](../../frontend/ml-canvas/src/components/data/PipelineVersionsModal.tsx) | 10 | 3 | L48:1 = **9**, L67:1 = **10**, L263:33 = **10** |
| [src/components/eda/DecompositionTree.tsx](../../frontend/ml-canvas/src/components/eda/DecompositionTree.tsx) | 10 | 4 | L81:24 = **10**, L209:36 = **9**, L247:29 = **10**, L328:25 = **9** |
| [src/components/eda/DistributionChart.tsx](../../frontend/ml-canvas/src/components/eda/DistributionChart.tsx) | 10 | 1 | L28:41 = **10** |
| [src/components/eda/JobsHistoryModal.tsx](../../frontend/ml-canvas/src/components/eda/JobsHistoryModal.tsx) | 10 | 2 | L39:66 = **10**, L172:1 = **9** |
| [src/components/eda/tabs/BivariateTab.tsx](../../frontend/ml-canvas/src/components/eda/tabs/BivariateTab.tsx) | 10 | 2 | L32:1 = **9**, L67:58 = **10** |
| [src/components/layout/PropertiesPanel.tsx](../../frontend/ml-canvas/src/components/layout/PropertiesPanel.tsx) | 10 | 3 | L21:42 = **9**, L148:6 = **10**, L453:1 = **9** |
| [src/components/pages/ExperimentsPage/components/ClassificationChartsForSplit.tsx](../../frontend/ml-canvas/src/components/pages/ExperimentsPage/components/ClassificationChartsForSplit.tsx) | 10 | 2 | L230:59 = **10**, L358:84 = **10** |
| [src/components/pages/ExperimentsPage/components/MetricsComparisonChart.tsx](../../frontend/ml-canvas/src/components/pages/ExperimentsPage/components/MetricsComparisonChart.tsx) | 10 | 1 | L46:56 = **10** |
| [src/components/pages/ExperimentsPage/components/PerClassConfusionMatrix.tsx](../../frontend/ml-canvas/src/components/pages/ExperimentsPage/components/PerClassConfusionMatrix.tsx) | 10 | 1 | L33:57 = **10** |
| [src/components/pages/ExperimentsPage/components/ShapExplainabilityView.tsx](../../frontend/ml-canvas/src/components/pages/ExperimentsPage/components/ShapExplainabilityView.tsx) | 10 | 1 | L89:56 = **10** |
| [src/components/pages/ExperimentsPage/hooks/useEvaluationFetch.ts](../../frontend/ml-canvas/src/components/pages/ExperimentsPage/hooks/useEvaluationFetch.ts) | 10 | 1 | L73:43 = **10** |
| [src/components/pages/ExperimentsPage/utils/classificationCharts.ts](../../frontend/ml-canvas/src/components/pages/ExperimentsPage/utils/classificationCharts.ts) | 10 | 2 | L59:8 = **9**, L409:45 = **10** |
| [src/components/pages/ExperimentsPage/utils/classificationCharts/thresholdMetrics.ts](../../frontend/ml-canvas/src/components/pages/ExperimentsPage/utils/classificationCharts/thresholdMetrics.ts) | 10 | 3 | L6:1 = **10**, L60:1 = **9**, L89:1 = **9** |
| [src/components/pages/ExperimentsPage/utils/jobMeta.ts](../../frontend/ml-canvas/src/components/pages/ExperimentsPage/utils/jobMeta.ts) | 10 | 2 | L114:8 = **9**, L173:8 = **10** |
| [src/components/pages/experiments/pipelineDiffLayout.ts](../../frontend/ml-canvas/src/components/pages/experiments/pipelineDiffLayout.ts) | 10 | 2 | L127:20 = **10**, L163:8 = **9** |
| [src/components/panels/jobs/jobCard/JobIdentity.tsx](../../frontend/ml-canvas/src/components/panels/jobs/jobCard/JobIdentity.tsx) | 10 | 1 | L7:8 = **10** |
| [src/components/shared/NodeInspectorModal.tsx](../../frontend/ml-canvas/src/components/shared/NodeInspectorModal.tsx) | 10 | 1 | L42:70 = **10** |
| [src/core/hooks/useTrainingNodeContext.ts](../../frontend/ml-canvas/src/core/hooks/useTrainingNodeContext.ts) | 10 | 2 | L67:1 = **9**, L119:5 = **10** |
| [src/core/hooks/useTuningTrials.ts](../../frontend/ml-canvas/src/core/hooks/useTuningTrials.ts) | 10 | 2 | L178:8 = **10**, L215:13 = **9** |
| [src/core/utils/canvasPersistence.ts](../../frontend/ml-canvas/src/core/utils/canvasPersistence.ts) | 10 | 1 | L60:8 = **10** |
| [src/core/utils/chartUtils.ts](../../frontend/ml-canvas/src/core/utils/chartUtils.ts) | 10 | 1 | L5:30 = **10** |
| [src/core/utils/nodeSearch.ts](../../frontend/ml-canvas/src/core/utils/nodeSearch.ts) | 10 | 1 | L34:22 = **10** |
| [src/core/utils/operationalContext.ts](../../frontend/ml-canvas/src/core/utils/operationalContext.ts) | 10 | 1 | L136:8 = **10** |
| [src/core/utils/pipelineLeakageValidation.ts](../../frontend/ml-canvas/src/core/utils/pipelineLeakageValidation.ts) | 10 | 3 | L188:8 = **9**, L299:1 = **10**, L361:8 = **10** |
| [src/core/utils/preprocessingSerializerAudit20260908.test.ts](../../frontend/ml-canvas/src/core/utils/preprocessingSerializerAudit20260908.test.ts) | 10 | 1 | L130:71 = **10** |
| [src/modules/nodes/modeling/SegmentationSettings.tsx](../../frontend/ml-canvas/src/modules/nodes/modeling/SegmentationSettings.tsx) | 10 | 1 | L45:6 = **10** |
| [src/modules/nodes/modeling/segmentationSettings/SegmentationActionFooter.tsx](../../frontend/ml-canvas/src/modules/nodes/modeling/segmentationSettings/SegmentationActionFooter.tsx) | 10 | 1 | L9:8 = **10** |
| [src/modules/nodes/processing/DeduplicationNode.tsx](../../frontend/ml-canvas/src/modules/nodes/processing/DeduplicationNode.tsx) | 10 | 1 | L44:135 = **10** |
| [src/modules/nodes/processing/DropColumnsNode.tsx](../../frontend/ml-canvas/src/modules/nodes/processing/DropColumnsNode.tsx) | 10 | 1 | L105:129 = **10** |
| [src/modules/nodes/processing/DropRowsNode.tsx](../../frontend/ml-canvas/src/modules/nodes/processing/DropRowsNode.tsx) | 10 | 1 | L43:120 = **10** |
| [src/modules/nodes/processing/MissingIndicatorNode.tsx](../../frontend/ml-canvas/src/modules/nodes/processing/MissingIndicatorNode.tsx) | 10 | 1 | L56:144 = **10** |
| [src/modules/nodes/processing/TextCleaningNode.tsx](../../frontend/ml-canvas/src/modules/nodes/processing/TextCleaningNode.tsx) | 10 | 2 | L33:6 = **9**, L171:132 = **10** |
| [src/modules/nodes/processing/TimeSeriesNode.tsx](../../frontend/ml-canvas/src/modules/nodes/processing/TimeSeriesNode.tsx) | 10 | 2 | L209:1 = **10**, L284:1 = **10** |
| [src/modules/nodes/processing/VectorizerNodes.tsx](../../frontend/ml-canvas/src/modules/nodes/processing/VectorizerNodes.tsx) | 10 | 2 | L149:1 = **10**, L231:101 = **9** |
| [src/modules/nodes/processing/outlier/OutlierFeedback.tsx](../../frontend/ml-canvas/src/modules/nodes/processing/outlier/OutlierFeedback.tsx) | 10 | 1 | L33:1 = **10** |
| [src/pages/Dashboard.tsx](../../frontend/ml-canvas/src/pages/Dashboard.tsx) | 10 | 1 | L39:36 = **10** |
| [src/pages/DataSources.tsx](../../frontend/ml-canvas/src/pages/DataSources.tsx) | 10 | 3 | L33:38 = **10**, L83:44 = **9**, L191:32 = **9** |
| [src/pages/auditLog/AuditRow.tsx](../../frontend/ml-canvas/src/pages/auditLog/AuditRow.tsx) | 10 | 1 | L77:50 = **10** |
| [src/pages/auditLog/auditHistoryModel.ts](../../frontend/ml-canvas/src/pages/auditLog/auditHistoryModel.ts) | 10 | 1 | L52:8 = **10** |
| [src/pages/drift/DriftTable.tsx](../../frontend/ml-canvas/src/pages/drift/DriftTable.tsx) | 10 | 4 | L56:27 = **10**, L148:6 = **10**, L248:1 = **10**, L318:54 = **10** |
| [src/pages/drift/_utils/csvExport.ts](../../frontend/ml-canvas/src/pages/drift/_utils/csvExport.ts) | 10 | 1 | L20:58 = **10** |
| [src/components/appLayout/Sidebar.tsx](../../frontend/ml-canvas/src/components/appLayout/Sidebar.tsx) | 9 | 2 | L115:1 = **9**, L133:17 = **9** |
| [src/components/canvas/connectionPort/portPresentation.ts](../../frontend/ml-canvas/src/components/canvas/connectionPort/portPresentation.ts) | 9 | 1 | L13:8 = **9** |
| [src/components/eda/tabs/PCATab.tsx](../../frontend/ml-canvas/src/components/eda/tabs/PCATab.tsx) | 9 | 1 | L64:46 = **9** |
| [src/components/eda/tabs/TargetAnalysisTab.tsx](../../frontend/ml-canvas/src/components/eda/tabs/TargetAnalysisTab.tsx) | 9 | 1 | L36:68 = **9** |
| [src/components/eda/tabs/TimeSeriesTab.tsx](../../frontend/ml-canvas/src/components/eda/tabs/TimeSeriesTab.tsx) | 9 | 1 | L31:60 = **9** |
| [src/components/eda/variableCard/VariableCardStatus.tsx](../../frontend/ml-canvas/src/components/eda/variableCard/VariableCardStatus.tsx) | 9 | 1 | L17:8 = **9** |
| [src/components/layout/MainLayout.tsx](../../frontend/ml-canvas/src/components/layout/MainLayout.tsx) | 9 | 1 | L26:37 = **9** |
| [src/components/layout/NotificationCenter.tsx](../../frontend/ml-canvas/src/components/layout/NotificationCenter.tsx) | 9 | 1 | L151:45 = **9** |
| [src/components/layout/resultsPanel/MergeWarningsBanner.tsx](../../frontend/ml-canvas/src/components/layout/resultsPanel/MergeWarningsBanner.tsx) | 9 | 1 | L147:1 = **9** |
| [src/components/layout/resultsPanel/SplitTabs.tsx](../../frontend/ml-canvas/src/components/layout/resultsPanel/SplitTabs.tsx) | 9 | 1 | L25:19 = **9** |
| [src/components/pages/ExperimentsPage.tsx](../../frontend/ml-canvas/src/components/pages/ExperimentsPage.tsx) | 9 | 2 | L41:42 = **9**, L746:1 = **9** |
| [src/components/pages/ExperimentsPage/components/JobListSidebar.tsx](../../frontend/ml-canvas/src/components/pages/ExperimentsPage/components/JobListSidebar.tsx) | 9 | 1 | L21:48 = **9** |
| [src/components/pages/ExperimentsPage/components/ShapDependenceView.tsx](../../frontend/ml-canvas/src/components/pages/ExperimentsPage/components/ShapDependenceView.tsx) | 9 | 1 | L29:52 = **9** |
| [src/components/pages/ExperimentsPage/components/ShapInteractionView.tsx](../../frontend/ml-canvas/src/components/pages/ExperimentsPage/components/ShapInteractionView.tsx) | 9 | 1 | L27:53 = **9** |
| [src/components/pages/ExperimentsPage/components/jobListSidebar/JobMetadata.tsx](../../frontend/ml-canvas/src/components/pages/ExperimentsPage/components/jobListSidebar/JobMetadata.tsx) | 9 | 1 | L6:8 = **9** |
| [src/components/pages/ExperimentsPage/components/segmentationView/SegmentationSummary.tsx](../../frontend/ml-canvas/src/components/pages/ExperimentsPage/components/segmentationView/SegmentationSummary.tsx) | 9 | 1 | L44:8 = **9** |
| [src/components/pages/ExperimentsPage/utils/artifactCoverage.ts](../../frontend/ml-canvas/src/components/pages/ExperimentsPage/utils/artifactCoverage.ts) | 9 | 1 | L61:8 = **9** |
| [src/components/panels/jobsDrawer/ParallelRunProgress.tsx](../../frontend/ml-canvas/src/components/panels/jobsDrawer/ParallelRunProgress.tsx) | 9 | 1 | L5:8 = **9** |
| [src/components/shared/useModalFocus.ts](../../frontend/ml-canvas/src/components/shared/useModalFocus.ts) | 9 | 1 | L76:21 = **9** |
| [src/components/ui/FormField.tsx](../../frontend/ml-canvas/src/components/ui/FormField.tsx) | 9 | 1 | L29:43 = **9** |
| [src/core/hooks/useClipboard.ts](../../frontend/ml-canvas/src/core/hooks/useClipboard.ts) | 9 | 1 | L80:27 = **9** |
| [src/core/hooks/useJobPolling.ts](../../frontend/ml-canvas/src/core/hooks/useJobPolling.ts) | 9 | 1 | L65:1 = **9** |
| [src/core/hooks/useNodeInspection.ts](../../frontend/ml-canvas/src/core/hooks/useNodeInspection.ts) | 9 | 1 | L9:8 = **9** |
| [src/core/hooks/useRecommendations.test.ts](../../frontend/ml-canvas/src/core/hooks/useRecommendations.test.ts) | 9 | 1 | L8:13 = **9** |
| [src/core/store/useGraphStore.ts](../../frontend/ml-canvas/src/core/store/useGraphStore.ts) | 9 | 1 | L318:18 = **9** |
| [src/core/store/useJobStore.ts](../../frontend/ml-canvas/src/core/store/useJobStore.ts) | 9 | 1 | L113:23 = **9** |
| [src/core/templates/pipelineTemplates.ts](../../frontend/ml-canvas/src/core/templates/pipelineTemplates.ts) | 9 | 1 | L54:8 = **9** |
| [src/core/utils/format.ts](../../frontend/ml-canvas/src/core/utils/format.ts) | 9 | 1 | L300:39 = **9** |
| [src/core/utils/leakageFeedback.ts](../../frontend/ml-canvas/src/core/utils/leakageFeedback.ts) | 9 | 1 | L25:8 = **9** |
| [src/core/utils/nextNodePosition.ts](../../frontend/ml-canvas/src/core/utils/nextNodePosition.ts) | 9 | 1 | L4:8 = **9** |
| [src/core/utils/operationalContext/recordIdentity.ts](../../frontend/ml-canvas/src/core/utils/operationalContext/recordIdentity.ts) | 9 | 1 | L4:8 = **9** |
| [src/modules/nodes/inspection/DataPreviewComponents.tsx](../../frontend/ml-canvas/src/modules/nodes/inspection/DataPreviewComponents.tsx) | 9 | 1 | L79:136 = **9** |
| [src/modules/nodes/modeling/TrainTestSplitNode.tsx](../../frontend/ml-canvas/src/modules/nodes/modeling/TrainTestSplitNode.tsx) | 9 | 1 | L101:138 = **9** |
| [src/modules/nodes/modeling/components/BestParamsModal.tsx](../../frontend/ml-canvas/src/modules/nodes/modeling/components/BestParamsModal.tsx) | 9 | 1 | L26:64 = **9** |
| [src/modules/nodes/modeling/components/SearchSpaceInput.tsx](../../frontend/ml-canvas/src/modules/nodes/modeling/components/SearchSpaceInput.tsx) | 9 | 2 | L32:66 = **9**, L41:30 = **9** |
| [src/modules/nodes/modeling/components/StrategySettingsModal.tsx](../../frontend/ml-canvas/src/modules/nodes/modeling/components/StrategySettingsModal.tsx) | 9 | 1 | L195:76 = **9** |
| [src/modules/nodes/modeling/segmentationSettings/useSegmentationSubmission.ts](../../frontend/ml-canvas/src/modules/nodes/modeling/segmentationSettings/useSegmentationSubmission.ts) | 9 | 1 | L38:23 = **9** |
| [src/modules/nodes/processing/AliasReplacementNode.tsx](../../frontend/ml-canvas/src/modules/nodes/processing/AliasReplacementNode.tsx) | 9 | 1 | L146:144 = **9** |
| [src/modules/nodes/processing/BinningNode.tsx](../../frontend/ml-canvas/src/modules/nodes/processing/BinningNode.tsx) | 9 | 2 | L56:1 = **9**, L129:117 = **9** |
| [src/modules/nodes/processing/FeatureSelectionNode.tsx](../../frontend/ml-canvas/src/modules/nodes/processing/FeatureSelectionNode.tsx) | 9 | 1 | L65:16 = **9** |
| [src/modules/nodes/processing/InvalidValueReplacementNode.tsx](../../frontend/ml-canvas/src/modules/nodes/processing/InvalidValueReplacementNode.tsx) | 9 | 1 | L94:154 = **9** |
| [src/modules/nodes/processing/OutlierNode.tsx](../../frontend/ml-canvas/src/modules/nodes/processing/OutlierNode.tsx) | 9 | 1 | L15:57 = **9** |
| [src/modules/nodes/processing/ScalingNode.tsx](../../frontend/ml-canvas/src/modules/nodes/processing/ScalingNode.tsx) | 9 | 1 | L14:57 = **9** |
| [src/modules/nodes/processing/ValueReplacementSettings.tsx](../../frontend/ml-canvas/src/modules/nodes/processing/ValueReplacementSettings.tsx) | 9 | 1 | L29:6 = **9** |
| [src/modules/nodes/shared/ColumnMultiSelect.tsx](../../frontend/ml-canvas/src/modules/nodes/shared/ColumnMultiSelect.tsx) | 9 | 1 | L80:8 = **9** |
| [src/pages/DataDriftPage.tsx](../../frontend/ml-canvas/src/pages/DataDriftPage.tsx) | 9 | 2 | L43:1 = **9**, L111:40 = **9** |
| [src/pages/Jobs.tsx](../../frontend/ml-canvas/src/pages/Jobs.tsx) | 9 | 1 | L465:1 = **9** |
| [src/pages/ModelRegistry.tsx](../../frontend/ml-canvas/src/pages/ModelRegistry.tsx) | 9 | 4 | L58:13 = **9**, L155:25 = **9**, L295:43 = **9**, L379:33 = **9** |
| [src/pages/SlowNodesPage.tsx](../../frontend/ml-canvas/src/pages/SlowNodesPage.tsx) | 9 | 1 | L409:6 = **9** |
| [src/pages/drift/JobSelector.tsx](../../frontend/ml-canvas/src/pages/drift/JobSelector.tsx) | 9 | 1 | L23:1 = **9** |
| [src/pages/drift/SelectedJobMeta.tsx](../../frontend/ml-canvas/src/pages/drift/SelectedJobMeta.tsx) | 9 | 1 | L14:64 = **9** |

## Optional function details

Function names are the ESLint labels. Anonymous callbacks are identified
by their exact line and column. All functions below pass the required gate.

| CCN | File / line | Column | Function (ESLint label) |
|---:|---|---:|---|
| 10 | [src/components/canvas/ConnectionGuidance.tsx:7](../../frontend/ml-canvas/src/components/canvas/ConnectionGuidance.tsx#L7) | 8 | Function 'ConnectionGuidance' |
| 10 | [src/components/canvas/FlowCanvas.tsx:345](../../frontend/ml-canvas/src/components/canvas/FlowCanvas.tsx#L345) | 26 | Arrow function |
| 10 | [src/components/canvas/FlowCanvas.tsx:448](../../frontend/ml-canvas/src/components/canvas/FlowCanvas.tsx#L448) | 23 | Arrow function |
| 10 | [src/components/data/AddSourceModal.tsx:34](../../frontend/ml-canvas/src/components/data/AddSourceModal.tsx#L34) | 24 | Async arrow function |
| 10 | [src/components/data/DatasetPreviewModal.tsx:227](../../frontend/ml-canvas/src/components/data/DatasetPreviewModal.tsx#L227) | 23 | Arrow function |
| 10 | [src/components/data/PipelineVersionsModal.tsx:67](../../frontend/ml-canvas/src/components/data/PipelineVersionsModal.tsx#L67) | 1 | Function 'summariseGraph' |
| 10 | [src/components/data/PipelineVersionsModal.tsx:263](../../frontend/ml-canvas/src/components/data/PipelineVersionsModal.tsx#L263) | 33 | Arrow function |
| 10 | [src/components/eda/DecompositionTree.tsx:81](../../frontend/ml-canvas/src/components/eda/DecompositionTree.tsx#L81) | 24 | Arrow function |
| 10 | [src/components/eda/DecompositionTree.tsx:247](../../frontend/ml-canvas/src/components/eda/DecompositionTree.tsx#L247) | 29 | Async arrow function |
| 10 | [src/components/eda/DistributionChart.tsx:28](../../frontend/ml-canvas/src/components/eda/DistributionChart.tsx#L28) | 41 | Arrow function |
| 10 | [src/components/eda/JobsHistoryModal.tsx:39](../../frontend/ml-canvas/src/components/eda/JobsHistoryModal.tsx#L39) | 66 | Arrow function |
| 10 | [src/components/eda/tabs/BivariateTab.tsx:67](../../frontend/ml-canvas/src/components/eda/tabs/BivariateTab.tsx#L67) | 58 | Arrow function |
| 10 | [src/components/layout/PropertiesPanel.tsx:148](../../frontend/ml-canvas/src/components/layout/PropertiesPanel.tsx#L148) | 6 | Arrow function |
| 10 | [src/components/pages/ExperimentsPage/components/ClassificationChartsForSplit.tsx:230](../../frontend/ml-canvas/src/components/pages/ExperimentsPage/components/ClassificationChartsForSplit.tsx#L230) | 59 | Arrow function |
| 10 | [src/components/pages/ExperimentsPage/components/ClassificationChartsForSplit.tsx:358](../../frontend/ml-canvas/src/components/pages/ExperimentsPage/components/ClassificationChartsForSplit.tsx#L358) | 84 | Arrow function |
| 10 | [src/components/pages/ExperimentsPage/components/MetricsComparisonChart.tsx:46](../../frontend/ml-canvas/src/components/pages/ExperimentsPage/components/MetricsComparisonChart.tsx#L46) | 56 | Arrow function |
| 10 | [src/components/pages/ExperimentsPage/components/PerClassConfusionMatrix.tsx:33](../../frontend/ml-canvas/src/components/pages/ExperimentsPage/components/PerClassConfusionMatrix.tsx#L33) | 57 | Arrow function |
| 10 | [src/components/pages/ExperimentsPage/components/ShapExplainabilityView.tsx:89](../../frontend/ml-canvas/src/components/pages/ExperimentsPage/components/ShapExplainabilityView.tsx#L89) | 56 | Arrow function |
| 10 | [src/components/pages/ExperimentsPage/hooks/useEvaluationFetch.ts:73](../../frontend/ml-canvas/src/components/pages/ExperimentsPage/hooks/useEvaluationFetch.ts#L73) | 43 | Async arrow function |
| 10 | [src/components/pages/ExperimentsPage/utils/classificationCharts.ts:409](../../frontend/ml-canvas/src/components/pages/ExperimentsPage/utils/classificationCharts.ts#L409) | 45 | Arrow function |
| 10 | [src/components/pages/ExperimentsPage/utils/classificationCharts/thresholdMetrics.ts:6](../../frontend/ml-canvas/src/components/pages/ExperimentsPage/utils/classificationCharts/thresholdMetrics.ts#L6) | 1 | Function 'binaryMetricValue' |
| 10 | [src/components/pages/ExperimentsPage/utils/jobMeta.ts:173](../../frontend/ml-canvas/src/components/pages/ExperimentsPage/utils/jobMeta.ts#L173) | 8 | Function 'getTaskForModelType' |
| 10 | [src/components/pages/experiments/pipelineDiffLayout.ts:127](../../frontend/ml-canvas/src/components/pages/experiments/pipelineDiffLayout.ts#L127) | 20 | Arrow function |
| 10 | [src/components/panels/jobs/jobCard/JobIdentity.tsx:7](../../frontend/ml-canvas/src/components/panels/jobs/jobCard/JobIdentity.tsx#L7) | 8 | Function 'JobIdentity' |
| 10 | [src/components/shared/NodeInspectorModal.tsx:42](../../frontend/ml-canvas/src/components/shared/NodeInspectorModal.tsx#L42) | 70 | Arrow function |
| 10 | [src/core/hooks/useTrainingNodeContext.ts:119](../../frontend/ml-canvas/src/core/hooks/useTrainingNodeContext.ts#L119) | 5 | Async arrow function |
| 10 | [src/core/hooks/useTuningTrials.ts:178](../../frontend/ml-canvas/src/core/hooks/useTuningTrials.ts#L178) | 8 | Function 'useTuningTrials' |
| 10 | [src/core/utils/canvasPersistence.ts:60](../../frontend/ml-canvas/src/core/utils/canvasPersistence.ts#L60) | 8 | Function 'loadCanvasSnapshotDiagnostic' |
| 10 | [src/core/utils/chartUtils.ts:5](../../frontend/ml-canvas/src/core/utils/chartUtils.ts#L5) | 30 | Async arrow function |
| 10 | [src/core/utils/nodeSearch.ts:34](../../frontend/ml-canvas/src/core/utils/nodeSearch.ts#L34) | 22 | Arrow function |
| 10 | [src/core/utils/operationalContext.ts:136](../../frontend/ml-canvas/src/core/utils/operationalContext.ts#L136) | 8 | Function 'parseOperationalContext' |
| 10 | [src/core/utils/pipelineLeakageValidation.ts:299](../../frontend/ml-canvas/src/core/utils/pipelineLeakageValidation.ts#L299) | 1 | Function 'isFixedOperation' |
| 10 | [src/core/utils/pipelineLeakageValidation.ts:361](../../frontend/ml-canvas/src/core/utils/pipelineLeakageValidation.ts#L361) | 8 | Function 'findPreprocessingBeforeSplitIssues' |
| 10 | [src/core/utils/preprocessingSerializerAudit20260908.test.ts:130](../../frontend/ml-canvas/src/core/utils/preprocessingSerializerAudit20260908.test.ts#L130) | 71 | Arrow function |
| 10 | [src/modules/nodes/modeling/SegmentationSettings.tsx:45](../../frontend/ml-canvas/src/modules/nodes/modeling/SegmentationSettings.tsx#L45) | 6 | Arrow function |
| 10 | [src/modules/nodes/modeling/segmentationSettings/SegmentationActionFooter.tsx:9](../../frontend/ml-canvas/src/modules/nodes/modeling/segmentationSettings/SegmentationActionFooter.tsx#L9) | 8 | Function 'SegmentationActionFooter' |
| 10 | [src/modules/nodes/processing/DeduplicationNode.tsx:44](../../frontend/ml-canvas/src/modules/nodes/processing/DeduplicationNode.tsx#L44) | 135 | Arrow function |
| 10 | [src/modules/nodes/processing/DropColumnsNode.tsx:105](../../frontend/ml-canvas/src/modules/nodes/processing/DropColumnsNode.tsx#L105) | 129 | Arrow function |
| 10 | [src/modules/nodes/processing/DropRowsNode.tsx:43](../../frontend/ml-canvas/src/modules/nodes/processing/DropRowsNode.tsx#L43) | 120 | Arrow function |
| 10 | [src/modules/nodes/processing/MissingIndicatorNode.tsx:56](../../frontend/ml-canvas/src/modules/nodes/processing/MissingIndicatorNode.tsx#L56) | 144 | Arrow function |
| 10 | [src/modules/nodes/processing/TextCleaningNode.tsx:171](../../frontend/ml-canvas/src/modules/nodes/processing/TextCleaningNode.tsx#L171) | 132 | Arrow function |
| 10 | [src/modules/nodes/processing/TimeSeriesNode.tsx:209](../../frontend/ml-canvas/src/modules/nodes/processing/TimeSeriesNode.tsx#L209) | 1 | Function 'TimeSeriesSettings' |
| 10 | [src/modules/nodes/processing/TimeSeriesNode.tsx:284](../../frontend/ml-canvas/src/modules/nodes/processing/TimeSeriesNode.tsx#L284) | 1 | Function 'validateMethodSettings' |
| 10 | [src/modules/nodes/processing/VectorizerNodes.tsx:149](../../frontend/ml-canvas/src/modules/nodes/processing/VectorizerNodes.tsx#L149) | 1 | Function 'VocabularyParameters' |
| 10 | [src/modules/nodes/processing/outlier/OutlierFeedback.tsx:33](../../frontend/ml-canvas/src/modules/nodes/processing/outlier/OutlierFeedback.tsx#L33) | 1 | Function 'OutlierRowSummary' |
| 10 | [src/pages/Dashboard.tsx:39](../../frontend/ml-canvas/src/pages/Dashboard.tsx#L39) | 36 | Arrow function |
| 10 | [src/pages/DataSources.tsx:33](../../frontend/ml-canvas/src/pages/DataSources.tsx#L33) | 38 | Arrow function |
| 10 | [src/pages/auditLog/AuditRow.tsx:77](../../frontend/ml-canvas/src/pages/auditLog/AuditRow.tsx#L77) | 50 | Arrow function |
| 10 | [src/pages/auditLog/auditHistoryModel.ts:52](../../frontend/ml-canvas/src/pages/auditLog/auditHistoryModel.ts#L52) | 8 | Function 'describeAuditHistory' |
| 10 | [src/pages/drift/DriftTable.tsx:56](../../frontend/ml-canvas/src/pages/drift/DriftTable.tsx#L56) | 27 | Arrow function |
| 10 | [src/pages/drift/DriftTable.tsx:148](../../frontend/ml-canvas/src/pages/drift/DriftTable.tsx#L148) | 6 | Arrow function |
| 10 | [src/pages/drift/DriftTable.tsx:248](../../frontend/ml-canvas/src/pages/drift/DriftTable.tsx#L248) | 1 | Function 'DriftRow' |
| 10 | [src/pages/drift/DriftTable.tsx:318](../../frontend/ml-canvas/src/pages/drift/DriftTable.tsx#L318) | 54 | Arrow function |
| 10 | [src/pages/drift/_utils/csvExport.ts:20](../../frontend/ml-canvas/src/pages/drift/_utils/csvExport.ts#L20) | 58 | Arrow function |
| 9 | [src/components/appLayout/Sidebar.tsx:115](../../frontend/ml-canvas/src/components/appLayout/Sidebar.tsx#L115) | 1 | Function 'ThemeControl' |
| 9 | [src/components/appLayout/Sidebar.tsx:133](../../frontend/ml-canvas/src/components/appLayout/Sidebar.tsx#L133) | 17 | Arrow function |
| 9 | [src/components/canvas/FlowCanvas.tsx:303](../../frontend/ml-canvas/src/components/canvas/FlowCanvas.tsx#L303) | 21 | Arrow function |
| 9 | [src/components/canvas/connectionPort/portPresentation.ts:13](../../frontend/ml-canvas/src/components/canvas/connectionPort/portPresentation.ts#L13) | 8 | Function 'portAppearance' |
| 9 | [src/components/data/PipelineVersionsModal.tsx:48](../../frontend/ml-canvas/src/components/data/PipelineVersionsModal.tsx#L48) | 1 | Function 'countCanvasNodes' |
| 9 | [src/components/eda/DecompositionTree.tsx:209](../../frontend/ml-canvas/src/components/eda/DecompositionTree.tsx#L209) | 36 | Arrow function |
| 9 | [src/components/eda/DecompositionTree.tsx:328](../../frontend/ml-canvas/src/components/eda/DecompositionTree.tsx#L328) | 25 | Async arrow function |
| 9 | [src/components/eda/JobsHistoryModal.tsx:172](../../frontend/ml-canvas/src/components/eda/JobsHistoryModal.tsx#L172) | 1 | Function 'HistoryRow' |
| 9 | [src/components/eda/tabs/BivariateTab.tsx:32](../../frontend/ml-canvas/src/components/eda/tabs/BivariateTab.tsx#L32) | 1 | Function 'BivariatePlot' |
| 9 | [src/components/eda/tabs/PCATab.tsx:64](../../frontend/ml-canvas/src/components/eda/tabs/PCATab.tsx#L64) | 46 | Arrow function |
| 9 | [src/components/eda/tabs/TargetAnalysisTab.tsx:36](../../frontend/ml-canvas/src/components/eda/tabs/TargetAnalysisTab.tsx#L36) | 68 | Arrow function |
| 9 | [src/components/eda/tabs/TimeSeriesTab.tsx:31](../../frontend/ml-canvas/src/components/eda/tabs/TimeSeriesTab.tsx#L31) | 60 | Arrow function |
| 9 | [src/components/eda/variableCard/VariableCardStatus.tsx:17](../../frontend/ml-canvas/src/components/eda/variableCard/VariableCardStatus.tsx#L17) | 8 | Function 'VariableCardStatus' |
| 9 | [src/components/layout/MainLayout.tsx:26](../../frontend/ml-canvas/src/components/layout/MainLayout.tsx#L26) | 37 | Arrow function |
| 9 | [src/components/layout/NotificationCenter.tsx:151](../../frontend/ml-canvas/src/components/layout/NotificationCenter.tsx#L151) | 45 | Arrow function |
| 9 | [src/components/layout/PropertiesPanel.tsx:21](../../frontend/ml-canvas/src/components/layout/PropertiesPanel.tsx#L21) | 42 | Arrow function |
| 9 | [src/components/layout/PropertiesPanel.tsx:453](../../frontend/ml-canvas/src/components/layout/PropertiesPanel.tsx#L453) | 1 | Function 'mergeConflictDetails' |
| 9 | [src/components/layout/resultsPanel/MergeWarningsBanner.tsx:147](../../frontend/ml-canvas/src/components/layout/resultsPanel/MergeWarningsBanner.tsx#L147) | 1 | Function 'FanInWarning' |
| 9 | [src/components/layout/resultsPanel/SplitTabs.tsx:25](../../frontend/ml-canvas/src/components/layout/resultsPanel/SplitTabs.tsx#L25) | 19 | Arrow function |
| 9 | [src/components/pages/ExperimentsPage.tsx:41](../../frontend/ml-canvas/src/components/pages/ExperimentsPage.tsx#L41) | 42 | Arrow function |
| 9 | [src/components/pages/ExperimentsPage.tsx:746](../../frontend/ml-canvas/src/components/pages/ExperimentsPage.tsx#L746) | 1 | Function 'ArtifactComparisonViews' |
| 9 | [src/components/pages/ExperimentsPage/components/JobListSidebar.tsx:21](../../frontend/ml-canvas/src/components/pages/ExperimentsPage/components/JobListSidebar.tsx#L21) | 48 | Arrow function |
| 9 | [src/components/pages/ExperimentsPage/components/ShapDependenceView.tsx:29](../../frontend/ml-canvas/src/components/pages/ExperimentsPage/components/ShapDependenceView.tsx#L29) | 52 | Arrow function |
| 9 | [src/components/pages/ExperimentsPage/components/ShapInteractionView.tsx:27](../../frontend/ml-canvas/src/components/pages/ExperimentsPage/components/ShapInteractionView.tsx#L27) | 53 | Arrow function |
| 9 | [src/components/pages/ExperimentsPage/components/jobListSidebar/JobMetadata.tsx:6](../../frontend/ml-canvas/src/components/pages/ExperimentsPage/components/jobListSidebar/JobMetadata.tsx#L6) | 8 | Function 'JobMetadata' |
| 9 | [src/components/pages/ExperimentsPage/components/segmentationView/SegmentationSummary.tsx:44](../../frontend/ml-canvas/src/components/pages/ExperimentsPage/components/segmentationView/SegmentationSummary.tsx#L44) | 8 | Function 'SegmentationSummary' |
| 9 | [src/components/pages/ExperimentsPage/utils/artifactCoverage.ts:61](../../frontend/ml-canvas/src/components/pages/ExperimentsPage/utils/artifactCoverage.ts#L61) | 8 | Function 'getArtifactCoverage' |
| 9 | [src/components/pages/ExperimentsPage/utils/classificationCharts.ts:59](../../frontend/ml-canvas/src/components/pages/ExperimentsPage/utils/classificationCharts.ts#L59) | 8 | Function 'applyThreshold' |
| 9 | [src/components/pages/ExperimentsPage/utils/classificationCharts/thresholdMetrics.ts:60](../../frontend/ml-canvas/src/components/pages/ExperimentsPage/utils/classificationCharts/thresholdMetrics.ts#L60) | 1 | Function 'classRates' |
| 9 | [src/components/pages/ExperimentsPage/utils/classificationCharts/thresholdMetrics.ts:89](../../frontend/ml-canvas/src/components/pages/ExperimentsPage/utils/classificationCharts/thresholdMetrics.ts#L89) | 1 | Function 'countBinaryPredictions' |
| 9 | [src/components/pages/ExperimentsPage/utils/jobMeta.ts:114](../../frontend/ml-canvas/src/components/pages/ExperimentsPage/utils/jobMeta.ts#L114) | 8 | Function 'getDisplayScore' |
| 9 | [src/components/pages/experiments/pipelineDiffLayout.ts:163](../../frontend/ml-canvas/src/components/pages/experiments/pipelineDiffLayout.ts#L163) | 8 | Function 'engineGraphToSide' |
| 9 | [src/components/panels/jobsDrawer/ParallelRunProgress.tsx:5](../../frontend/ml-canvas/src/components/panels/jobsDrawer/ParallelRunProgress.tsx#L5) | 8 | Function 'ParallelRunProgress' |
| 9 | [src/components/shared/useModalFocus.ts:76](../../frontend/ml-canvas/src/components/shared/useModalFocus.ts#L76) | 21 | Arrow function |
| 9 | [src/components/ui/FormField.tsx:29](../../frontend/ml-canvas/src/components/ui/FormField.tsx#L29) | 43 | Arrow function |
| 9 | [src/core/hooks/useClipboard.ts:80](../../frontend/ml-canvas/src/core/hooks/useClipboard.ts#L80) | 27 | Arrow function |
| 9 | [src/core/hooks/useJobPolling.ts:65](../../frontend/ml-canvas/src/core/hooks/useJobPolling.ts#L65) | 1 | Function 'summarizeJobs' |
| 9 | [src/core/hooks/useNodeInspection.ts:9](../../frontend/ml-canvas/src/core/hooks/useNodeInspection.ts#L9) | 8 | Function 'useNodeInspection' |
| 9 | [src/core/hooks/useRecommendations.test.ts:8](../../frontend/ml-canvas/src/core/hooks/useRecommendations.test.ts#L8) | 13 | Arrow function |
| 9 | [src/core/hooks/useTrainingNodeContext.ts:67](../../frontend/ml-canvas/src/core/hooks/useTrainingNodeContext.ts#L67) | 1 | Function 'findUpstreamDatasetId' |
| 9 | [src/core/hooks/useTuningTrials.ts:215](../../frontend/ml-canvas/src/core/hooks/useTuningTrials.ts#L215) | 13 | Arrow function |
| 9 | [src/core/store/useGraphStore.ts:318](../../frontend/ml-canvas/src/core/store/useGraphStore.ts#L318) | 18 | Method 'chainSiblings' |
| 9 | [src/core/store/useJobStore.ts:113](../../frontend/ml-canvas/src/core/store/useJobStore.ts#L113) | 23 | Async arrow function |
| 9 | [src/core/templates/pipelineTemplates.ts:54](../../frontend/ml-canvas/src/core/templates/pipelineTemplates.ts#L54) | 8 | Function 'buildGraphFromTemplate' |
| 9 | [src/core/utils/format.ts:300](../../frontend/ml-canvas/src/core/utils/format.ts#L300) | 39 | Arrow function |
| 9 | [src/core/utils/leakageFeedback.ts:25](../../frontend/ml-canvas/src/core/utils/leakageFeedback.ts#L25) | 8 | Function 'getLeakageErrorMessage' |
| 9 | [src/core/utils/nextNodePosition.ts:4](../../frontend/ml-canvas/src/core/utils/nextNodePosition.ts#L4) | 8 | Function 'nextNodePosition' |
| 9 | [src/core/utils/operationalContext/recordIdentity.ts:4](../../frontend/ml-canvas/src/core/utils/operationalContext/recordIdentity.ts#L4) | 8 | Function 'recordIdentity' |
| 9 | [src/core/utils/pipelineLeakageValidation.ts:188](../../frontend/ml-canvas/src/core/utils/pipelineLeakageValidation.ts#L188) | 8 | Function 'isTargetOnlyEncoding' |
| 9 | [src/modules/nodes/inspection/DataPreviewComponents.tsx:79](../../frontend/ml-canvas/src/modules/nodes/inspection/DataPreviewComponents.tsx#L79) | 136 | Arrow function |
| 9 | [src/modules/nodes/modeling/TrainTestSplitNode.tsx:101](../../frontend/ml-canvas/src/modules/nodes/modeling/TrainTestSplitNode.tsx#L101) | 138 | Arrow function |
| 9 | [src/modules/nodes/modeling/components/BestParamsModal.tsx:26](../../frontend/ml-canvas/src/modules/nodes/modeling/components/BestParamsModal.tsx#L26) | 64 | Arrow function |
| 9 | [src/modules/nodes/modeling/components/SearchSpaceInput.tsx:32](../../frontend/ml-canvas/src/modules/nodes/modeling/components/SearchSpaceInput.tsx#L32) | 66 | Arrow function |
| 9 | [src/modules/nodes/modeling/components/SearchSpaceInput.tsx:41](../../frontend/ml-canvas/src/modules/nodes/modeling/components/SearchSpaceInput.tsx#L41) | 30 | Arrow function |
| 9 | [src/modules/nodes/modeling/components/StrategySettingsModal.tsx:195](../../frontend/ml-canvas/src/modules/nodes/modeling/components/StrategySettingsModal.tsx#L195) | 76 | Arrow function |
| 9 | [src/modules/nodes/modeling/segmentationSettings/useSegmentationSubmission.ts:38](../../frontend/ml-canvas/src/modules/nodes/modeling/segmentationSettings/useSegmentationSubmission.ts#L38) | 23 | Async arrow function |
| 9 | [src/modules/nodes/processing/AliasReplacementNode.tsx:146](../../frontend/ml-canvas/src/modules/nodes/processing/AliasReplacementNode.tsx#L146) | 144 | Arrow function |
| 9 | [src/modules/nodes/processing/BinningNode.tsx:56](../../frontend/ml-canvas/src/modules/nodes/processing/BinningNode.tsx#L56) | 1 | Function 'BinningStrategyControls' |
| 9 | [src/modules/nodes/processing/BinningNode.tsx:129](../../frontend/ml-canvas/src/modules/nodes/processing/BinningNode.tsx#L129) | 117 | Arrow function |
| 9 | [src/modules/nodes/processing/FeatureSelectionNode.tsx:65](../../frontend/ml-canvas/src/modules/nodes/processing/FeatureSelectionNode.tsx#L65) | 16 | Method 'bodyPreview' |
| 9 | [src/modules/nodes/processing/InvalidValueReplacementNode.tsx:94](../../frontend/ml-canvas/src/modules/nodes/processing/InvalidValueReplacementNode.tsx#L94) | 154 | Arrow function |
| 9 | [src/modules/nodes/processing/OutlierNode.tsx:15](../../frontend/ml-canvas/src/modules/nodes/processing/OutlierNode.tsx#L15) | 57 | Arrow function |
| 9 | [src/modules/nodes/processing/ScalingNode.tsx:14](../../frontend/ml-canvas/src/modules/nodes/processing/ScalingNode.tsx#L14) | 57 | Arrow function |
| 9 | [src/modules/nodes/processing/TextCleaningNode.tsx:33](../../frontend/ml-canvas/src/modules/nodes/processing/TextCleaningNode.tsx#L33) | 6 | Arrow function |
| 9 | [src/modules/nodes/processing/ValueReplacementSettings.tsx:29](../../frontend/ml-canvas/src/modules/nodes/processing/ValueReplacementSettings.tsx#L29) | 6 | Arrow function |
| 9 | [src/modules/nodes/processing/VectorizerNodes.tsx:231](../../frontend/ml-canvas/src/modules/nodes/processing/VectorizerNodes.tsx#L231) | 101 | Arrow function |
| 9 | [src/modules/nodes/shared/ColumnMultiSelect.tsx:80](../../frontend/ml-canvas/src/modules/nodes/shared/ColumnMultiSelect.tsx#L80) | 8 | Function 'ColumnMultiSelect' |
| 9 | [src/pages/DataDriftPage.tsx:43](../../frontend/ml-canvas/src/pages/DataDriftPage.tsx#L43) | 1 | Function 'DriftToolbar' |
| 9 | [src/pages/DataDriftPage.tsx:111](../../frontend/ml-canvas/src/pages/DataDriftPage.tsx#L111) | 40 | Arrow function |
| 9 | [src/pages/DataSources.tsx:83](../../frontend/ml-canvas/src/pages/DataSources.tsx#L83) | 44 | Arrow function |
| 9 | [src/pages/DataSources.tsx:191](../../frontend/ml-canvas/src/pages/DataSources.tsx#L191) | 32 | Arrow function |
| 9 | [src/pages/Jobs.tsx:465](../../frontend/ml-canvas/src/pages/Jobs.tsx#L465) | 1 | Function 'JobFilters' |
| 9 | [src/pages/ModelRegistry.tsx:58](../../frontend/ml-canvas/src/pages/ModelRegistry.tsx#L58) | 13 | Arrow function |
| 9 | [src/pages/ModelRegistry.tsx:155](../../frontend/ml-canvas/src/pages/ModelRegistry.tsx#L155) | 25 | Arrow function |
| 9 | [src/pages/ModelRegistry.tsx:295](../../frontend/ml-canvas/src/pages/ModelRegistry.tsx#L295) | 43 | Arrow function |
| 9 | [src/pages/ModelRegistry.tsx:379](../../frontend/ml-canvas/src/pages/ModelRegistry.tsx#L379) | 33 | Arrow function |
| 9 | [src/pages/SlowNodesPage.tsx:409](../../frontend/ml-canvas/src/pages/SlowNodesPage.tsx#L409) | 6 | Arrow function |
| 9 | [src/pages/drift/JobSelector.tsx:23](../../frontend/ml-canvas/src/pages/drift/JobSelector.tsx#L23) | 1 | Function 'JobSelectorTrigger' |
| 9 | [src/pages/drift/SelectedJobMeta.tsx:14](../../frontend/ml-canvas/src/pages/drift/SelectedJobMeta.tsx#L14) | 64 | Arrow function |
