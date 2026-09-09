import React, { useState, useMemo } from 'react';
import { collectGraphValidationIssues, useGraphStore, type GraphValidationIssue } from '../../core/store/useGraphStore';
import { FOCUS_CANVAS_EVENT, FOCUS_NODE_EVENT } from '../../core/hooks/useKeyboardShortcuts';
import { getReadOnlyMode } from '../../core/hooks/useReadOnlyMode';
import { useViewStore } from '../../core/store/useViewStore';
import { generateBranchColors } from '../../core/hooks/useBranchColors';
import { useConfirm } from '../shared';
import { PanelHeader } from './resultsPanel/presentation/PanelHeader';
import { PanelResizeHandle } from './resultsPanel/presentation/PanelResizeHandle';
import { PanelContent, type ResultsPane } from './resultsPanel/presentation/PanelPanes';
import {
  toDatasetMap,
  getCurrentDataset,
  getAppliedSteps,
  getPaneSummary,
  getPanelHeight,
} from './resultsPanel/presentation/previewPresentation';

/** Shows preview results alongside canvas validation and run failure summaries. */
export const ResultsPanel: React.FC<{ maxHeight?: number }> = ({ maxHeight = 720 }) => {
  const executionResult = useGraphStore((state) => state.executionResult);
  const setExecutionResult = useGraphStore((state) => state.setExecutionResult);
  const setLastRunError = useGraphStore((state) => state.setLastRunError);
  const canvasNodes = useGraphStore((state) => state.nodes);
  const canvasEdges = useGraphStore((state) => state.edges);
  const lastRunError = useGraphStore((state) => state.lastRunError);
  const selectNode = useGraphStore((state) => state.selectNode);
  const chainSiblings = useGraphStore((state) => state.chainSiblings);
  const confirm = useConfirm();
  const {
    isResultsPanelExpanded,
    setResultsPanelExpanded,
    isResultsPanelMaximized: isMaximized,
    setResultsPanelMaximized: setIsMaximized,
    isResultsPanelDismissed: dismissed,
    setResultsPanelDismissed: setDismissed,
    resultsPanelHeight,
    setResultsPanelHeight,
  } = useViewStore();
  const panelId = React.useId();
  const dragStart = React.useRef<{ y: number; height: number } | null>(null);
  const panelHeight = Math.min(resultsPanelHeight, maxHeight);
  const resizeTo = (height: number) => setResultsPanelHeight(Math.max(200, Math.min(maxHeight, height)));
  const stopResizing = () => { dragStart.current = null; };
  const [activeBranch, setActiveBranch] = useState<string | null>(null);
  const [activeTab, setActiveTab] = useState<string | null>(null);
  const [mergeWarningsOpen, setMergeWarningsOpen] = useState<boolean>(false);
  const [pane, setPane] = useState<ResultsPane | null>(null);
  const validationHeadingId = React.useId();
  const validationIssues = useMemo(
    () => collectGraphValidationIssues(canvasNodes, canvasEdges).filter(issue => issue.category !== 'leakage'),
    [canvasNodes, canvasEdges],
  );
  const validationIssueKey = JSON.stringify(validationIssues);

  // After the user closes the panel with X it stays hidden until something
  // new happens: fresh preview/run results arrive, or the set of validation
  // issues changes (edited graph → new problem to look at).
  React.useEffect(() => {
    if (executionResult) setDismissed(false);
  }, [executionResult, setDismissed]);
  React.useEffect(() => {
    if (lastRunError) setDismissed(false);
  }, [lastRunError, setDismissed]);
  React.useEffect(() => {
    setDismissed(false);
  }, [validationIssueKey, setDismissed]);

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

  const openIssue = (issue: GraphValidationIssue): void => {
    if (!selectNode(issue.nodeId)) return;
    const readOnly = getReadOnlyMode();
    if (!readOnly) useViewStore.getState().requestValidationFocus(issue);
    window.dispatchEvent(new CustomEvent(FOCUS_NODE_EVENT, { detail: { id: issue.nodeId, focusWrapper: readOnly } }));
  };

  const showSummary = validationIssues.length > 0 || lastRunError !== null;
  if (!executionResult && !showSummary) return null;
  if (dismissed) return null;

  const { currentRows, currentTotal, columns } = getCurrentDataset(executionResult, effectiveTab, datasets, totals);
  const applied_steps = getAppliedSteps(executionResult, activeBranch);
  const { issueCount, activePane } = getPaneSummary(
    pane, validationIssues.length, lastRunError, executionResult, mergeWarnings.length,
  );
  const closePanel = () => {
    setExecutionResult(null);
    setLastRunError(null);
    setIsMaximized(false);
    setMergeWarningsOpen(false);
    setDismissed(true);
    window.dispatchEvent(new Event(FOCUS_CANVAS_EVENT));
  };

  return (
    <div
      id={panelId}
      role="region"
      aria-label="Preview results"
      style={{ height: getPanelHeight(isResultsPanelExpanded, isMaximized, panelHeight) }}
      className="absolute bottom-0 left-0 right-0 bg-background border-t shadow-[0_-4px_6px_-1px_rgba(0,0,0,0.1)] z-20 flex flex-col"
    >
      {isResultsPanelExpanded && !isMaximized && (
        <PanelResizeHandle
          panelId={panelId}
          panelHeight={panelHeight}
          maxHeight={maxHeight}
          dragStart={dragStart}
          resizeTo={resizeTo}
          setResultsPanelHeight={setResultsPanelHeight}
          stopResizing={stopResizing}
        />
      )}
      <PanelHeader
        executionResult={executionResult}
        lastRunError={lastRunError}
        validationCount={validationIssues.length}
        previewCount={currentRows.length}
        currentTotal={currentTotal}
        branchCount={branchLabels.length}
        issueCount={issueCount}
        isResultsPanelExpanded={isResultsPanelExpanded}
        setResultsPanelExpanded={setResultsPanelExpanded}
        isMaximized={isMaximized}
        setIsMaximized={setIsMaximized}
        onClose={closePanel}
      />
      {isResultsPanelExpanded && (
        <PanelContent
          activePane={activePane}
          setPane={setPane}
          issueCount={issueCount}
          stepCount={applied_steps.length}
          data={{
            hasResults: Boolean(executionResult),
            branchTabs: { branchLabels, activeBranch, setActiveBranch, branchColors, branchAdvisoryCounts },
            splitTabs: { tabNames, datasets, totals, effectiveTab, setActiveTab },
            table: { columns, currentRows, effectiveTab },
          }}
          issues={{
            validationIssues, validationHeadingId, openIssue, lastRunError, hasResults: Boolean(executionResult), issueCount,
            advisories: { mergeWarnings, mergeWarningsOpen, setMergeWarningsOpen, nodeLabelMap, confirm, chainSiblings },
          }}
          steps={{ applied_steps, status: executionResult?.status, nodeLabelMap }}
        />
      )}
    </div>
  );
};
