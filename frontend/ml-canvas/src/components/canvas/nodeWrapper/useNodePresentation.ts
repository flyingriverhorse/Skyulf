import type { NodeDefinition } from '../../../core/types/nodes';
import { useGraphStore } from '../../../core/store/useGraphStore';
import { useJobStore } from '../../../core/store/useJobStore';
import { useReadOnlyMode } from '../../../core/hooks/useReadOnlyMode';
import { useCanvasLeakageFeedback } from '../../../core/contexts/CanvasLeakageContext';
import { isAutoParallelType, supportsExecutionModeToggle, getExecutionMode } from '../../../core/types/executionMode';
import { useNodeValidation } from './useNodeValidation';
import { useNodePerformance } from './nodePerformance';

function useNodeExecution(id: string) {
  const nodeResult = useGraphStore((state) => state.executionResult?.node_results?.[id]);
  // Fallback summary entries for trainer/tuner nodes whose jobs run
  // via Celery — they never populate `executionResult.node_results`.
  // The `useNodeJobSummaries` hook keeps this map fresh from
  // `/jobs/node-summaries` after every job event. For parallel runs
  // the array contains one entry per branch; merge runs have one.
  const jobSummaries = useGraphStore((state) => state.nodeJobSummaries[id]);
  // Mirror of the canvas branch labels so multi-branch trainer cards
  // can show "Path B · Xgboost" letters that match the colored edges.
  const branchEdgeLabels = useGraphStore((state) => state.branchEdgeLabels);
  // Whether a fresh job is currently in flight for this node — lets
  // the card tag the existing (now-stale) summary as "previous run"
  // in its tooltip until the new job completes and overwrites it.
  const isJobInFlight = useJobStore((state) =>
    state.jobs.some(
      (j) => j.node_id === id && (j.status === 'running' || j.status === 'queued'),
    ),
  );
  return { nodeResult, jobSummaries, branchEdgeLabels, isJobInFlight };
}

function useNodeSchema(id: string, definitionType: string) {
  const predictedSchema = useGraphStore((state) => state.predictedSchemas[id]);
  const brokenRefs = useGraphStore((state) => state.brokenSchemaRefs[id]);
  const hasBrokenRefs = (brokenRefs?.length ?? 0) > 0;
  const brokenRefTooltip = hasBrokenRefs
    ? `⚠ Column name not found in upstream output\n` +
    (brokenRefs ?? [])
      .map((r) => `  • "${r.column}" (in field '${r.field}')`)
      .join('\n') +
    `\n\nThe canvas automatically previews each node's output schema in the background.\n` +
    `These column names were not found in the predicted upstream output — they may be\n` +
    `misspelled or the upstream step may have renamed / dropped them.\n\n` +
    `The pipeline can still run, but may fail at this step.`
    : null;
  // predictedSchema is `null` (not undefined) when the server explicitly
  // returned null — meaning this calculator is data-dependent (e.g. encoders,
  // feature-selection). `undefined` means the API hasn't responded yet.
  const schemaIsDataDependent = predictedSchema === null && definitionType !== 'dataset_node';
  return { predictedSchema, brokenRefs, hasBrokenRefs, brokenRefTooltip, schemaIsDataDependent };
}

function useNodeLeakage(id: string) {
  const { nodeIssues, openGuide } = useCanvasLeakageFeedback();
  const leakageIssues = nodeIssues[id] ?? [];
  const leakageSeverity = leakageIssues.some(issue => issue.severity === 'error')
    ? 'error' : leakageIssues.length > 0 ? 'warning' : null;
  return { leakageIssues, leakageSeverity, openGuide };
}

function runsInParallel(definitionType: string, data: Record<string, unknown>, incomingSourceCount: number) {
  const isTrainingNode = supportsExecutionModeToggle(definitionType);
  const isAutoParallel = isAutoParallelType(definitionType);
  const isParallel =
    (isTrainingNode && getExecutionMode(data) === 'parallel') ||
    (isAutoParallel && incomingSourceCount > 1);

  return isParallel;
}

function useNodeMerge(id: string, definitionType: string, data: Record<string, unknown>,
  definition: NodeDefinition<unknown> | undefined) {
  const incomingSourceCount = useGraphStore((state) => state.incomingSourceCounts[id] ?? 0);
  const mergeWarningSeverity: 'risk' | 'safe' | null = useGraphStore((state) => {
    const warnings = state.executionResult?.merge_warnings ?? [];
    const w = warnings.find((mw) => mw.node_id === id);
    if (!w) return null;
    return (w.overlap_columns?.length ?? 0) > 0 ? 'risk' : 'safe';
  });
  const isAutoParallel = isAutoParallelType(definitionType);
  const isParallel = runsInParallel(definitionType, data, incomingSourceCount);
  const canMerge = (definition?.inputs?.length ?? 0) > 0 && !isAutoParallel;
  const showMergeBadge = (canMerge && incomingSourceCount > 1) || isParallel;
  return { incomingSourceCount, mergeWarningSeverity, isParallel, showMergeBadge };
}

/** Subscribe to each node's execution and validation feedback without changing graph data. */
export function useNodePresentation(id: string, data: Record<string, unknown>,
  definitionType: string, definition: NodeDefinition<unknown> | undefined) {
  const readOnly = useReadOnlyMode();
  const execution = useNodeExecution(id);
  const schema = useNodeSchema(id, definitionType);
  const leakage = useNodeLeakage(id);
  const merge = useNodeMerge(id, definitionType, data, definition);
  const validation = useNodeValidation(definition, data);
  const perf = useNodePerformance(definitionType, data.run_mode as string | undefined,
    execution.nodeResult, execution.jobSummaries);
  return { readOnly, execution, schema, leakage, merge, validation, perf };
}

export type NodePresentation = ReturnType<typeof useNodePresentation>;
