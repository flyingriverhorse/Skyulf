import React, { useState, useMemo } from 'react';
import { collectGraphValidationIssues, useGraphStore } from '../../core/store/useGraphStore';
import { useViewStore } from '../../core/store/useViewStore';
import { AlertTriangle, ChevronUp, ChevronDown, Maximize2, Minimize2, Table, XCircle } from 'lucide-react';
import type { PreviewDataRows, PreviewData } from '../../core/api/client';
import { generateBranchColors } from '../../core/hooks/useBranchColors';
import { clickableProps } from '../../core/utils/a11y';
import { useConfirm } from '../shared';
import { BranchTabs } from './resultsPanel/BranchTabs';
import { SplitTabs } from './resultsPanel/SplitTabs';
import { MergeWarningsBanner } from './resultsPanel/MergeWarningsBanner';
import { ResultsTable } from './resultsPanel/ResultsTable';

/** Convert a PreviewData payload into a {tabName -> rows} map. */
function toDatasetMap(previewData: PreviewData | null | undefined): Record<string, PreviewDataRows> {
  if (!previewData) return {};
  if (Array.isArray(previewData)) return { Result: previewData as PreviewDataRows };
  if (typeof previewData === 'object') return previewData as Record<string, PreviewDataRows>;
  return {};
}

/** Which pane of the results panel is showing. */
type ResultsPane = 'data' | 'issues' | 'steps';

/** Shows preview results alongside canvas validation and run failure summaries. */
export const ResultsPanel: React.FC = () => {
  const executionResult = useGraphStore((state) => state.executionResult);
  const canvasNodes = useGraphStore((state) => state.nodes);
  const canvasEdges = useGraphStore((state) => state.edges);
  const lastRunError = useGraphStore((state) => state.lastRunError);
  const onNodesChange = useGraphStore((state) => state.onNodesChange);
  const chainSiblings = useGraphStore((state) => state.chainSiblings);
  const confirm = useConfirm();
  const { isResultsPanelExpanded, setResultsPanelExpanded } = useViewStore();
  const [activeBranch, setActiveBranch] = useState<string | null>(null);
  const [activeTab, setActiveTab] = useState<string | null>(null);
  const [mergeWarningsOpen, setMergeWarningsOpen] = useState<boolean>(false);
  const [pane, setPane] = useState<ResultsPane | null>(null);
  const [isMaximized, setIsMaximized] = useState<boolean>(false);
  const validationHeadingId = React.useId();
  const validationIssues = useMemo(
    () => collectGraphValidationIssues(canvasNodes, canvasEdges),
    [canvasNodes, canvasEdges],
  );

  // Map node id → readable label (falls back to a prettified definitionType
  // so users see "Drop Rows" instead of "drop_rows-04475cca-eef7-4fdb-...").
  const nodeLabelMap = useMemo(() => {
    const map: Record<string, string> = {};
    for (const n of canvasNodes) {
      const data = (n.data ?? {}) as Record<string, unknown>;
      const label =
        (data.label as string) ||
        (data.title as string) ||
        (typeof data.definitionType === 'string'
          ? (data.definitionType as string)
              .replace(/_/g, ' ')
              .replace(/\b\w/g, (c) => c.toUpperCase())
          : n.id);
      map[n.id] = label;
    }
    return map;
  }, [canvasNodes]);

  // Branch labels (only when backend returned multiple branches)
  const branchLabels = useMemo(() => {
    const bp = executionResult?.branch_previews;
    if (!bp) return [];
    const keys = Object.keys(bp);
    return keys.length > 1 ? keys : [];
  }, [executionResult]);

  // Use the canvas-assigned colors (keyed by label string) so the colored dot
  // next to each branch tab always matches the edge color the user sees on the
  // canvas. Falls back to freshly-generated colors when the canvas hasn't been
  // rendered yet (e.g. loading a saved pipeline for the first time).
  const branchLabelColors = useGraphStore((s) => s.branchLabelColors);
  const branchColors = useMemo(() => {
    const fallback = generateBranchColors(branchLabels.length);
    return branchLabels.map((l, i) => branchLabelColors[l] ?? fallback[i] ?? '#888');
  }, [branchLabels, branchLabelColors]);

  // Pick default branch when branches change
  React.useEffect(() => {
    if (branchLabels.length > 0 && (!activeBranch || !branchLabels.includes(activeBranch))) {
      setActiveBranch(branchLabels[0] ?? null);
    } else if (branchLabels.length === 0 && activeBranch !== null) {
      setActiveBranch(null);
    }
  }, [branchLabels, activeBranch]);

  // Dataset tabs for the active branch (or top-level preview when single-branch)
  const datasets = useMemo(() => {
    if (branchLabels.length > 0 && activeBranch && executionResult?.branch_previews) {
      return toDatasetMap(executionResult.branch_previews[activeBranch]);
    }
    return toDatasetMap(executionResult?.preview_data);
  }, [executionResult, branchLabels, activeBranch]);

  // True row totals (rows in `datasets` are capped at 50 for transport).
  // Falls back to the preview row count when the backend didn't supply a
  // total — keeps older clients/responses functional.
  const totals = useMemo<Record<string, number>>(() => {
    if (branchLabels.length > 0 && activeBranch && executionResult?.branch_preview_totals) {
      return executionResult.branch_preview_totals[activeBranch] ?? {};
    }
    return executionResult?.preview_totals ?? {};
  }, [executionResult, branchLabels, activeBranch]);

  const tabNames = Object.keys(datasets);

  // Derive the effective tab synchronously so that switching branches or
  // receiving a fresh executionResult never produces an in-between render
  // with `activeTab` pointing at a key that doesn't exist in `datasets`
  // (which previously flashed "No preview data available" before the
  // default-picking effect caught up). The state setter still drives user
  // intent; this just absorbs the one-frame mismatch.
  const effectiveTab = useMemo<string | null>(() => {
    if (activeTab && tabNames.includes(activeTab)) return activeTab;
    if (tabNames.length === 0) return null;
    if (tabNames.includes('train')) return 'train';
    if (tabNames.includes('X')) return 'X';
    return tabNames[0] ?? null;
  }, [activeTab, tabNames]);

  // Set default split tab when datasets change
  React.useEffect(() => {
    if (tabNames.length > 0 && (!activeTab || !tabNames.includes(activeTab))) {
      if (tabNames.includes('train')) setActiveTab('train');
      else if (tabNames.includes('X')) setActiveTab('X');
      else setActiveTab(tabNames[0] ?? null);
    }
  }, [tabNames, activeTab]);

  // Engine-emitted merge advisories (sibling fan-in etc.) - surfaced so users
  // immediately see when a downstream node is silently merging parallel
  // branches that share an ancestor. When a branch tab is active we only
  // show advisories for nodes that actually ran in that branch, otherwise
  // the banner would flag warnings for nodes the user can't even see on
  // the current tab.
  // NOTE: this useMemo must run on every render (i.e. before any early
  // return below) to preserve React's hook call order.
  const rawMergeWarnings = executionResult?.merge_warnings;
  const branchNodeIdsMemo = executionResult?.branch_node_ids;
  const mergeWarnings = useMemo(() => {
    const all = rawMergeWarnings ?? [];
    if (!activeBranch || !branchNodeIdsMemo || !branchNodeIdsMemo[activeBranch]) {
      return all;
    }
    const branchNodes = new Set(branchNodeIdsMemo[activeBranch]);
    return all.filter((w) => branchNodes.has(w.node_id));
  }, [rawMergeWarnings, activeBranch, branchNodeIdsMemo]);

  // Per-branch advisory counts so the user can see at-a-glance which other
  // branch tabs have warnings without having to click through each one.
  // The banner above is filtered to the active branch only, so without this
  // badge there is no signal that e.g. branch B has 4 advisories while
  // branch A has 0.
  const branchAdvisoryCounts = useMemo<Record<string, number>>(() => {
    const counts: Record<string, number> = {};
    const all = rawMergeWarnings ?? [];
    if (all.length === 0 || !branchNodeIdsMemo) return counts;
    for (const branch of branchLabels) {
      const ids = branchNodeIdsMemo[branch];
      if (!ids) continue;
      const set = new Set(ids);
      const n = all.filter((w) => set.has(w.node_id)).length;
      if (n > 0) counts[branch] = n;
    }
    return counts;
  }, [rawMergeWarnings, branchNodeIdsMemo, branchLabels]);

  const selectNode = (nodeId: string): void => {
    onNodesChange(
      canvasNodes.map((node) => ({
        id: node.id,
        type: 'select',
        selected: node.id === nodeId,
      })),
    );
  };

  const showSummary = validationIssues.length > 0 || lastRunError !== null;
  if (!executionResult && !showSummary) return null;

  const currentRows = executionResult && (effectiveTab && datasets[effectiveTab]) ? datasets[effectiveTab] : [];
  // Real dataset size for the active tab; falls back to the preview row
  // count when the backend didn't ship a total (older response, or single
  // list payload registered under the synthetic `_total` key).
  const currentTotal = executionResult && effectiveTab
    ? (totals[effectiveTab] ?? totals._total ?? currentRows.length)
    : 0;
  const columns = currentRows.length > 0 ? Object.keys(currentRows[0] ?? {}) : [];
  // When viewing a specific branch, restrict the applied-steps pills to nodes
  // that actually ran in that branch (otherwise every tab shows every node).
  const allNodeIds = executionResult?.node_results ? Object.keys(executionResult.node_results) : [];
  const branchNodeIds = executionResult?.branch_node_ids;
  const applied_steps = (branchNodeIds && activeBranch && branchNodeIds[activeBranch])
    ? branchNodeIds[activeBranch]
    : allNodeIds;

  const validationBanner = validationIssues.length > 0 && (
    <section
      className="m-3 rounded-md border border-red-200 bg-red-50 p-3 text-sm text-red-900 dark:border-red-900/40 dark:bg-red-950/20 dark:text-red-100"
      aria-labelledby={validationHeadingId}
    >
      <div className="flex items-start gap-2">
        <AlertTriangle className="mt-0.5 h-4 w-4 shrink-0 text-red-600 dark:text-red-400" aria-hidden="true" />
        <div className="min-w-0 flex-1">
          <p id={validationHeadingId} className="font-semibold">Validation issues</p>
          <p className="mt-0.5 text-xs text-red-700 dark:text-red-200">
            Fix one of the items below, then run preview again.
          </p>
          {/* Only the count is announced: the list recomputes on every graph
              edit, so a live region around it would re-read every issue on
              each keystroke. */}
          <p className="sr-only" role="status" aria-atomic="true">
            {validationIssues.length === 1
              ? '1 validation issue blocking preview'
              : `${validationIssues.length} validation issues blocking preview`}
          </p>
          <ul className="mt-3 space-y-2">
            {validationIssues.map((issue) => (
              <li key={`${issue.nodeId}-${issue.category}-${issue.message}`}>
                <button
                  type="button"
                  onClick={() => selectNode(issue.nodeId)}
                  className="w-full rounded-md border border-red-200 bg-white/80 px-3 py-2 text-left transition-colors hover:bg-red-100 dark:border-red-900/40 dark:bg-slate-950/30 dark:hover:bg-red-950/40"
                >
                  <div className="flex flex-wrap items-center gap-2 text-[11px] uppercase tracking-wide text-red-700 dark:text-red-300">
                    <span className="rounded bg-red-100 px-1.5 py-0.5 dark:bg-red-950/50">{issue.category}</span>
                    <span className="font-semibold normal-case tracking-normal">{issue.nodeLabel}</span>
                  </div>
                  <p className="mt-1 text-sm text-slate-800 dark:text-slate-100">{issue.message}</p>
                </button>
              </li>
            ))}
          </ul>
        </div>
      </div>
    </section>
  );

  const runErrorBanner = lastRunError && (
    <section
      className="m-3 rounded-md border border-red-200 bg-red-50 p-3 text-sm text-red-900 dark:border-red-900/40 dark:bg-red-950/20 dark:text-red-100"
      role="alert"
      aria-atomic="true"
    >
      <div className="flex items-start gap-2">
        <XCircle className="mt-0.5 h-4 w-4 shrink-0 text-red-600 dark:text-red-400" aria-hidden="true" />
        <div className="min-w-0">
          <p className="font-semibold">Last preview run failed</p>
          <p className="mt-0.5 text-sm text-slate-800 dark:text-slate-100">{lastRunError}</p>
        </div>
      </div>
    </section>
  );

  const issueCount =
    validationIssues.length + (lastRunError ? 1 : 0) + (executionResult ? mergeWarnings.length : 0);

  // Land on whichever pane has something to act on: a failed or blocked run
  // has no table worth showing, so default to Issues in that case.
  const defaultPane: ResultsPane =
    validationIssues.length > 0 || lastRunError ? 'issues' : 'data';
  const activePane = pane ?? defaultPane;

  const paneTabs: { id: ResultsPane; label: string; count?: number }[] = [
    { id: 'data', label: 'Data' },
    { id: 'issues', label: 'Issues', count: issueCount },
    { id: 'steps', label: 'Steps', count: applied_steps.length },
  ];

  return (
    <div
      className={`absolute bottom-0 left-0 right-0 bg-background border-t shadow-[0_-4px_6px_-1px_rgba(0,0,0,0.1)] transition-all duration-300 z-20 flex flex-col ${
        !isResultsPanelExpanded ? 'h-10' : isMaximized ? 'top-0' : 'h-96'
      }`}
    >
      {/* Header */}
      <div
        className="flex items-center justify-between gap-2 px-4 py-2 bg-muted/10 cursor-pointer hover:bg-muted/20 border-b select-none"
        {...clickableProps(() => setResultsPanelExpanded(!isResultsPanelExpanded))}
      >
        <div className="flex items-center gap-2 min-w-0">
          <Table className="w-4 h-4 text-primary shrink-0" />
          <span className="font-semibold text-sm shrink-0">Preview Results</span>
          {executionResult && (
            <span className="text-xs text-muted-foreground truncate">
              {currentRows.length === currentTotal
                ? `${currentTotal} rows`
                : `${currentRows.length} of ${currentTotal} rows shown`}
              {branchLabels.length > 0 ? ` · ${branchLabels.length} branches` : ''}
            </span>
          )}
          {executionResult?.status === 'failed' && (
            <span className="text-xs text-red-600 font-bold shrink-0">(Failed)</span>
          )}
          {issueCount > 0 && (
            <span className="shrink-0 text-[11px] font-medium px-1.5 py-0.5 rounded bg-amber-100 text-amber-800 dark:bg-amber-950/40 dark:text-amber-300">
              {issueCount} {issueCount === 1 ? 'issue' : 'issues'}
            </span>
          )}
        </div>
        <div className="flex items-center gap-1 shrink-0">
          {isResultsPanelExpanded && (
            <button
              type="button"
              className="p-1 hover:bg-muted rounded"
              aria-label={isMaximized ? 'Restore results panel' : 'Maximize results panel'}
              onClick={(e) => {
                e.stopPropagation();
                setIsMaximized((v) => !v);
              }}
            >
              {isMaximized ? <Minimize2 className="w-4 h-4" /> : <Maximize2 className="w-4 h-4" />}
            </button>
          )}
          <button type="button" className="p-1 hover:bg-muted rounded" aria-label={isResultsPanelExpanded ? 'Collapse results panel' : 'Expand results panel'}>
            {isResultsPanelExpanded ? <ChevronDown className="w-4 h-4" /> : <ChevronUp className="w-4 h-4" />}
          </button>
        </div>
      </div>

      {/* Content */}
      {isResultsPanelExpanded && (
        <div className="flex-1 overflow-hidden flex flex-col">
          {/* Pane tabs keep advisories and step pills from pushing the table
              off-screen — each lives in its own pane instead of stacking. */}
          <div className="flex items-center gap-1 px-2 border-b bg-muted/5 shrink-0" role="tablist">
            {paneTabs.map((tab) => (
              <button
                key={tab.id}
                type="button"
                role="tab"
                aria-selected={activePane === tab.id}
                onClick={() => setPane(tab.id)}
                className={`px-3 py-1.5 text-xs font-medium border-b-2 -mb-px transition-colors ${
                  activePane === tab.id
                    ? 'border-primary text-foreground'
                    : 'border-transparent text-muted-foreground hover:text-foreground'
                }`}
              >
                {tab.label}
                {tab.count ? (
                  <span className="ml-1.5 text-[10px] px-1 py-0.5 rounded bg-muted text-muted-foreground">
                    {tab.count}
                  </span>
                ) : null}
              </button>
            ))}
          </div>

          {activePane === 'data' && (
            <div className="flex-1 overflow-hidden flex flex-col">
              {executionResult && branchLabels.length > 0 && (
                <BranchTabs
                  branchLabels={branchLabels}
                  activeBranch={activeBranch}
                  setActiveBranch={setActiveBranch}
                  branchColors={branchColors}
                  branchAdvisoryCounts={branchAdvisoryCounts}
                />
              )}
              {executionResult && tabNames.length > 1 && (
                <SplitTabs
                  tabNames={tabNames}
                  datasets={datasets}
                  totals={totals}
                  effectiveTab={effectiveTab}
                  setActiveTab={setActiveTab}
                />
              )}
              {executionResult ? (
                <ResultsTable columns={columns} currentRows={currentRows} effectiveTab={effectiveTab} />
              ) : (
                <p className="p-4 text-sm text-muted-foreground">
                  Run a preview to see the resulting rows here.
                </p>
              )}
            </div>
          )}

          {activePane === 'issues' && (
            <div className="flex-1 overflow-y-auto">
              {validationBanner}
              {runErrorBanner}
              {executionResult && mergeWarnings.length > 0 && (
                <MergeWarningsBanner
                  mergeWarnings={mergeWarnings}
                  mergeWarningsOpen={mergeWarningsOpen}
                  setMergeWarningsOpen={setMergeWarningsOpen}
                  nodeLabelMap={nodeLabelMap}
                  confirm={confirm}
                  chainSiblings={chainSiblings}
                />
              )}
              {issueCount === 0 && (
                <p className="p-4 text-sm text-muted-foreground">
                  No validation issues, run errors, or merge advisories.
                </p>
              )}
            </div>
          )}

          {activePane === 'steps' && (
            <div className="flex-1 overflow-y-auto p-3">
              {applied_steps.length > 0 && executionResult?.status !== 'failed' ? (
                <div className="flex flex-wrap gap-2">
                  {applied_steps.map((step: string) => (
                    <span
                      key={step}
                      className="text-xs text-blue-800 dark:text-blue-200 bg-blue-100 dark:bg-blue-900/40 px-2 py-1 rounded border border-blue-200 dark:border-blue-800"
                    >
                      {nodeLabelMap[step] ?? step}
                    </span>
                  ))}
                </div>
              ) : (
                <p className="text-sm text-muted-foreground">
                  No steps ran. Run a preview to see which nodes executed.
                </p>
              )}
            </div>
          )}
        </div>
      )}
    </div>
  );
};
