import type { PreviewDataRows, PreviewData, PreviewResponse } from '../../../../core/api/client';
import type { ResultsPane } from './PanelPanes';

/** Convert a PreviewData payload into a {tabName -> rows} map. */
export function toDatasetMap(previewData: PreviewData | null | undefined): Record<string, PreviewDataRows> {
  if (!previewData) return {};
  if (Array.isArray(previewData)) return { Result: previewData };
  if (typeof previewData === 'object') return previewData;
  return {};
}

/** True dataset totals fall back to the legacy synthetic key and then preview length. */
export function getCurrentDataset(
  executionResult: PreviewResponse | null,
  effectiveTab: string | null,
  datasets: Record<string, PreviewDataRows>,
  totals: Record<string, number>,
) {
  if (!executionResult || !effectiveTab) return { currentRows: [], currentTotal: 0, columns: [] };
  const currentRows = datasets[effectiveTab] || [];
  const currentTotal = totals[effectiveTab] ?? totals._total ?? currentRows.length;
  const columns = currentRows.length > 0 ? Object.keys(currentRows[0] ?? {}) : [];
  return { currentRows, currentTotal, columns };
}

/** Restrict step pills to the active branch when branch membership is supplied. */
export function getAppliedSteps(executionResult: PreviewResponse | null, activeBranch: string | null): string[] {
  const allNodeIds = executionResult?.node_results ? Object.keys(executionResult.node_results) : [];
  const branchNodeIds = executionResult?.branch_node_ids;
  return (branchNodeIds && activeBranch && branchNodeIds[activeBranch])
    ? branchNodeIds[activeBranch]
    : allNodeIds;
}

/** An explicit pane choice survives later validation or result changes. */
export function getPaneSummary(
  pane: ResultsPane | null,
  validationCount: number,
  lastRunError: string | null,
  executionResult: PreviewResponse | null,
  mergeWarningCount: number,
) {
  const issueCount = validationCount + (lastRunError ? 1 : 0) + (executionResult ? mergeWarningCount : 0);
  const defaultPane: ResultsPane = validationCount > 0 || lastRunError ? 'issues' : 'data';
  return { issueCount, activePane: pane ?? defaultPane };
}

/** Collapse takes precedence over maximization, preserving the stored height preference. */
export function getPanelHeight(expanded: boolean, maximized: boolean, panelHeight: number): number | string {
  if (!expanded) return 40;
  return maximized ? '100%' : panelHeight;
}
