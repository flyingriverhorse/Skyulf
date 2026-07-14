import React, { useState, useMemo } from 'react';
import { useGraphStore } from '../../core/store/useGraphStore';
import { useViewStore } from '../../core/store/useViewStore';
import { ChevronUp, ChevronDown, Table } from 'lucide-react';
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

export const ResultsPanel: React.FC = () => {
  const executionResult = useGraphStore((state) => state.executionResult);
  const canvasNodes = useGraphStore((state) => state.nodes);
  const chainSiblings = useGraphStore((state) => state.chainSiblings);
  const confirm = useConfirm();
  const { isResultsPanelExpanded, setResultsPanelExpanded } = useViewStore();
  const [activeBranch, setActiveBranch] = useState<string | null>(null);
  const [activeTab, setActiveTab] = useState<string | null>(null);
  const [mergeWarningsOpen, setMergeWarningsOpen] = useState<boolean>(false);

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

  if (!executionResult) return null;

  const currentRows = (effectiveTab && datasets[effectiveTab]) ? datasets[effectiveTab] : [];
  // Real dataset size for the active tab; falls back to the preview row
  // count when the backend didn't ship a total (older response, or single
  // list payload registered under the synthetic `_total` key).
  const currentTotal = effectiveTab
    ? (totals[effectiveTab] ?? totals._total ?? currentRows.length)
    : 0;
  const columns = currentRows.length > 0 ? Object.keys(currentRows[0] ?? {}) : [];
  // When viewing a specific branch, restrict the applied-steps pills to nodes
  // that actually ran in that branch (otherwise every tab shows every node).
  const allNodeIds = executionResult.node_results ? Object.keys(executionResult.node_results) : [];
  const branchNodeIds = executionResult.branch_node_ids;
  const applied_steps = (branchNodeIds && activeBranch && branchNodeIds[activeBranch])
    ? branchNodeIds[activeBranch]
    : allNodeIds;

  return (
    <div
      className={`absolute bottom-0 left-0 right-0 bg-background border-t shadow-[0_-4px_6px_-1px_rgba(0,0,0,0.1)] transition-all duration-300 z-20 flex flex-col ${
        isResultsPanelExpanded ? 'h-96' : 'h-10'
      }`}
    >
      {/* Header */}
      <div
        className="flex items-center justify-between px-4 py-2 bg-muted/10 cursor-pointer hover:bg-muted/20 border-b select-none"
        {...clickableProps(() => setResultsPanelExpanded(!isResultsPanelExpanded))}
      >
        <div className="flex items-center gap-2">
          <Table className="w-4 h-4 text-primary" />
          <span className="font-semibold text-sm">Preview Results</span>
          <span className="text-xs text-muted-foreground ml-2">
            {currentRows.length === currentTotal
              ? `${currentTotal} rows`
              : `${currentRows.length} of ${currentTotal} rows shown`}
          </span>
          {branchLabels.length > 0 && (
            <span className="text-xs text-muted-foreground ml-2">
              · {branchLabels.length} branches
            </span>
          )}
          {executionResult.status === 'failed' && (
            <span className="text-xs text-red-600 font-bold ml-2">
              (Failed)
            </span>
          )}
        </div>
        <button className="p-1 hover:bg-muted rounded">
          {isResultsPanelExpanded ? <ChevronDown className="w-4 h-4" /> : <ChevronUp className="w-4 h-4" />}
        </button>
      </div>

      {/* Content */}
      {isResultsPanelExpanded && (
        <div className="flex-1 overflow-hidden flex flex-col">

          {/* Branch Tabs (multi-branch parallel runs only) */}
          {branchLabels.length > 0 && (
            <BranchTabs
              branchLabels={branchLabels}
              activeBranch={activeBranch}
              setActiveBranch={setActiveBranch}
              branchColors={branchColors}
              branchAdvisoryCounts={branchAdvisoryCounts}
            />
          )}

          {/* Split Tabs (train / test / X / y …) */}
          {tabNames.length > 1 && (
            <SplitTabs
              tabNames={tabNames}
              datasets={datasets}
              totals={totals}
              effectiveTab={effectiveTab}
              setActiveTab={setActiveTab}
            />
          )}

          {/* Merge advisories (sibling fan-in etc.) — engine-emitted.
              Collapsed by default to a one-line summary; click to expand
              full per-warning detail (inputs, overlap columns, winner). */}
          {mergeWarnings.length > 0 && (
            <MergeWarningsBanner
              mergeWarnings={mergeWarnings}
              mergeWarningsOpen={mergeWarningsOpen}
              setMergeWarningsOpen={setMergeWarningsOpen}
              nodeLabelMap={nodeLabelMap}
              confirm={confirm}
              chainSiblings={chainSiblings}
            />
          )}

          {/* Signals / Warnings */}
          {applied_steps.length > 0 && executionResult.status !== 'failed' && (
             <div className="p-2 bg-blue-50 dark:bg-blue-950/20 border-b dark:border-blue-900/30 flex gap-2 overflow-x-auto">
                {applied_steps.map((step: string, idx: number) => (
                  <div key={idx} className="text-xs text-blue-800 dark:text-blue-200 bg-blue-100 dark:bg-blue-900/40 px-2 py-1 rounded border border-blue-200 dark:border-blue-800 whitespace-nowrap">
                    {nodeLabelMap[step] ?? step}
                  </div>
                ))}
             </div>
          )}

          {/* Data Table */}
          <ResultsTable columns={columns} currentRows={currentRows} effectiveTab={effectiveTab} />
        </div>
      )}
    </div>
  );
};
