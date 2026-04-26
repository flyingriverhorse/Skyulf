import React, { useState, useMemo } from 'react';
import { useGraphStore } from '../../core/store/useGraphStore';
import { useViewStore } from '../../core/store/useViewStore';
import { ChevronUp, ChevronDown, ChevronRight, Table, Layers, GitBranch, AlertTriangle, Wand2 } from 'lucide-react';
import type { NodeExecutionResult, PreviewDataRows, PreviewData } from '../../core/api/client';
import { generateBranchColors } from '../../core/hooks/useBranchColors';
import { clickableProps } from '../../core/utils/a11y';
import { useConfirm } from '../shared';
import { toast } from '../../core/toast';

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

  const branchColors = useMemo(
    () => generateBranchColors(branchLabels.length),
    [branchLabels.length],
  );

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

  // Check for errors
  const nodeResults = executionResult.node_results || {};
  const errorNodeId = Object.keys(nodeResults).find((nodeId) => {
    const result: NodeExecutionResult | undefined = nodeResults[nodeId];
    return result?.status === 'failed';
  });
  const error = errorNodeId ? nodeResults[errorNodeId]?.error : null;

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
          {/* Error Message */}
          {executionResult.status === 'failed' && (
             <div className="p-4 bg-red-50 border-b text-red-800 text-sm overflow-auto max-h-32">
                <div className="font-bold mb-1">Pipeline Execution Failed</div>
                <div className="font-mono text-xs whitespace-pre-wrap">{error || 'Unknown error occurred during execution.'}</div>
             </div>
          )}

          {/* Branch Tabs (multi-branch parallel runs only) */}
          {branchLabels.length > 0 && (
            <div className="flex items-center gap-1 px-2 pt-2 border-b bg-muted/10">
              <GitBranch className="w-3 h-3 text-muted-foreground mr-1" />
              {branchLabels.map((label, idx) => {
                const isActive = activeBranch === label;
                const advisoryCount = branchAdvisoryCounts[label] ?? 0;
                return (
                  <button
                    key={label}
                    onClick={() => { setActiveBranch(label); }}
                    className={`flex items-center gap-1.5 px-3 py-1 text-xs font-medium rounded-t-md border-t border-l border-r transition-colors ${
                      isActive
                        ? 'bg-background text-foreground border-b-background translate-y-[1px]'
                        : 'bg-muted/30 text-muted-foreground hover:bg-muted/50 border-transparent'
                    }`}
                    title={
                      advisoryCount > 0
                        ? `${advisoryCount} merge advisor${advisoryCount === 1 ? 'y' : 'ies'} in this branch`
                        : undefined
                    }
                  >
                    <span
                      className="inline-block w-2.5 h-2.5 rounded-full"
                      style={{ backgroundColor: branchColors[idx] }}
                    />
                    {label}
                    {advisoryCount > 0 && (
                      <span
                        className="ml-1 inline-flex items-center gap-0.5 px-1.5 py-0.5 rounded-full bg-amber-100 text-amber-900 dark:bg-amber-900/40 dark:text-amber-200 text-[10px] font-semibold leading-none"
                        aria-label={`${advisoryCount} merge advisories`}
                      >
                        <AlertTriangle className="w-2.5 h-2.5" />
                        {advisoryCount}
                      </span>
                    )}
                  </button>
                );
              })}
            </div>
          )}

          {/* Split Tabs (train / test / X / y …) */}
          {tabNames.length > 1 && (
            <div className="flex items-center gap-1 px-2 pt-2 border-b bg-muted/5">
              <Layers className="w-3 h-3 text-muted-foreground mr-1" />
              {tabNames.map(name => {
                const rows = datasets[name];
                const previewCount = Array.isArray(rows) ? rows.length : 0;
                const total = totals[name] ?? totals._total ?? previewCount;
                const truncated = total > previewCount;
                return (
                  <button
                    key={name}
                    onClick={() => { setActiveTab(name); }}
                    className={`flex items-center gap-1.5 px-3 py-1 text-xs font-medium rounded-t-md border-t border-l border-r transition-colors ${
                      effectiveTab === name
                        ? 'bg-background text-primary border-b-background translate-y-[1px]'
                        : 'bg-muted/30 text-muted-foreground hover:bg-muted/50 border-transparent'
                    }`}
                    title={
                      truncated
                        ? `${total} row${total === 1 ? '' : 's'} in ${name} (${previewCount} shown in preview)`
                        : `${total} row${total === 1 ? '' : 's'} in ${name}`
                    }
                  >
                    {name}
                    <span
                      className={`inline-flex items-center px-1.5 py-0.5 rounded-full text-[10px] font-semibold leading-none ${
                        effectiveTab === name
                          ? 'bg-primary/15 text-primary'
                          : 'bg-muted/60 text-muted-foreground'
                      }`}
                    >
                      {total}
                    </span>
                  </button>
                );
              })}
            </div>
          )}

          {/* Merge advisories (sibling fan-in etc.) — engine-emitted.
              Collapsed by default to a one-line summary; click to expand
              full per-warning detail (inputs, overlap columns, winner). */}
          {mergeWarnings.length > 0 && (
            <div className="bg-amber-50 dark:bg-amber-950/20 border-b dark:border-amber-900/30">
              <button
                type="button"
                onClick={() => setMergeWarningsOpen((v) => !v)}
                className="w-full flex items-center gap-2 px-2 py-1.5 text-xs text-amber-900 dark:text-amber-200 hover:bg-amber-100/60 dark:hover:bg-amber-900/30 transition-colors"
                aria-expanded={mergeWarningsOpen}
              >
                {mergeWarningsOpen ? (
                  <ChevronDown className="w-3.5 h-3.5 flex-shrink-0" />
                ) : (
                  <ChevronRight className="w-3.5 h-3.5 flex-shrink-0" />
                )}
                <AlertTriangle className="w-3.5 h-3.5 flex-shrink-0" />
                <span className="font-medium">
                  {mergeWarnings.length} merge advisor{mergeWarnings.length === 1 ? 'y' : 'ies'}
                </span>
                <span className="opacity-80 truncate">
                  — {mergeWarnings
                    .map((w) => nodeLabelMap[w.node_id] ?? w.node_id)
                    .slice(0, 3)
                    .join(', ')}
                  {mergeWarnings.length > 3 ? `, +${mergeWarnings.length - 3} more` : ''}
                </span>
              </button>
              {mergeWarningsOpen && (
                <div className="px-2 pb-2 space-y-1.5">
                  {mergeWarnings.map((w, idx) => {
                    // Row-wise merge dropped non-shared columns: render a
                    // simpler advisory (no inputs / overlap to show).
                    if (w.kind === 'row_concat_drop') {
                      const dropped = w.dropped_columns ?? [];
                      return (
                        <div key={idx} className="flex items-start gap-2 text-xs text-amber-900 dark:text-amber-200 pl-5">
                          <AlertTriangle className="w-3.5 h-3.5 mt-0.5 flex-shrink-0" />
                          <div className="flex-1">
                            <span className="font-medium">
                              {nodeLabelMap[w.node_id] ?? w.node_id}
                            </span>{' '}
                            row-wise merge{w.part ? ` (${w.part})` : ''} dropped{' '}
                            {dropped.length} non-shared column
                            {dropped.length === 1 ? '' : 's'}:{' '}
                            <span className="font-mono bg-amber-100 dark:bg-amber-900/40 px-1 rounded">
                              {dropped.slice(0, 6).join(', ')}
                              {dropped.length > 6 ? `, +${dropped.length - 6} more` : ''}
                            </span>
                            . Only columns present in every input are kept when row counts differ.
                          </div>
                        </div>
                      );
                    }
                    const inputs = w.inputs ?? [];
                    const inputLabels = inputs.map((i) => nodeLabelMap[i] ?? i);
                    const winner = w.winner_input
                      ? (nodeLabelMap[w.winner_input] ?? w.winner_input)
                      : inputLabels[inputLabels.length - 1];
                    const overlap = w.overlap_columns ?? [];
                    const canAutoChain = overlap.length > 0 && inputs.length >= 2;
                    return (
                      <div key={idx} className="flex items-start gap-2 text-xs text-amber-900 dark:text-amber-200 pl-5">
                        <AlertTriangle className="w-3.5 h-3.5 mt-0.5 flex-shrink-0" />
                        <div className="flex-1">
                          <span className="font-medium">
                            {nodeLabelMap[w.node_id] ?? w.node_id}
                          </span>{' '}
                          merges {inputs.length} parallel branches:{' '}
                          {inputLabels.map((label, i) => (
                            <span key={i}>
                              <span className="font-mono bg-amber-100 dark:bg-amber-900/40 px-1 rounded">
                                {label}
                              </span>
                              {i < inputLabels.length - 1 && ' + '}
                            </span>
                          ))}
                          .{' '}
                          {overlap.length > 0 ? (
                            <>
                              {overlap.length} overlapping column
                              {overlap.length === 1 ? '' : 's'}{' '}
                              (<span className="font-mono">
                                {overlap.slice(0, 4).join(', ')}
                                {overlap.length > 4 ? `, +${overlap.length - 4} more` : ''}
                              </span>){' '}
                              take values from{' '}
                              <span className="font-mono font-semibold">{winner}</span>;
                              unique columns from the others are kept as-is.
                            </>
                          ) : (
                            <>
                              No column overlap — all columns from all branches are kept.
                            </>
                          )}{' '}
                          For sequential application, chain them instead.
                          {canAutoChain && (
                            <div className="mt-1.5">
                              <button
                                type="button"
                                onClick={() => {
                                  const consumerLabel = nodeLabelMap[w.node_id] ?? w.node_id;
                                  const chainStr = [...inputLabels, consumerLabel].join(' → ');
                                  void (async () => {
                                    const ok = await confirm({
                                      title: 'Rewire as a linear chain?',
                                      message: (
                                        <span>
                                          {chainStr}
                                          <br />
                                          <br />
                                          <span className="text-xs text-slate-500">Use Ctrl+Z to undo.</span>
                                        </span>
                                      ),
                                      confirmLabel: 'Rewire',
                                    });
                                    if (!ok) return;
                                    const success = chainSiblings(w.node_id, inputs);
                                    if (!success) {
                                      toast.error('Auto-chain failed', 'Re-run preview and try again.');
                                    }
                                  })();
                                }}
                                className="inline-flex items-center gap-1 px-2 py-0.5 text-[11px] font-medium rounded border border-amber-400 dark:border-amber-600 bg-amber-100 dark:bg-amber-900/40 hover:bg-amber-200 dark:hover:bg-amber-900/60 text-amber-900 dark:text-amber-100 transition-colors"
                                title="Rewire fan-in into a linear chain (no fan-in, no overwrite)"
                              >
                                <Wand2 className="w-3 h-3" />
                                Chain instead
                              </button>
                            </div>
                          )}
                        </div>
                      </div>
                    );
                  })}
                </div>
              )}
            </div>
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
          <div className="flex-1 overflow-auto">
            <table className="w-full text-sm text-left border-collapse">
              <thead className="text-xs text-muted-foreground uppercase bg-muted sticky top-0 z-10 shadow-sm">
                <tr>
                  {columns.map((col) => (
                    <th key={col} className="px-4 py-2 font-medium border-b whitespace-nowrap">
                      {col}
                    </th>
                  ))}
                </tr>
              </thead>
              <tbody>
                {currentRows.map((row: unknown, idx: number) => (
                  <tr key={idx} className="border-b hover:bg-muted/10">
                    {columns.map((col) => (
                      <td key={`${idx}-${col}`} className="px-4 py-2 whitespace-nowrap font-mono text-xs">
                        {(row as Record<string, unknown>)[col] !== null ? String((row as Record<string, unknown>)[col]) : <span className="text-muted-foreground italic">null</span>}
                      </td>
                    ))}
                  </tr>
                ))}
              </tbody>
            </table>
            {currentRows.length === 0 && (
              <div className="flex items-center justify-center h-32 text-muted-foreground text-sm">
                No preview data available {effectiveTab ? `for ${effectiveTab}` : ''}
              </div>
            )}
          </div>
        </div>
      )}
    </div>
  );
};
