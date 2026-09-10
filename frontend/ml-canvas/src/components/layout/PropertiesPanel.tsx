import React from 'react';
import { useGraphStore } from '../../core/store/useGraphStore';
import { useViewStore } from '../../core/store/useViewStore';
import { useSidebarOpen } from '../../core/hooks/useSidebarOpen';
import { FOCUS_CANVAS_EVENT, FOCUS_NODE_EVENT } from '../../core/hooks/useKeyboardShortcuts';
import { registry } from '../../core/registry/NodeRegistry';
import { ValidationNavigation, useValidationReveal } from '../shared/ValidationField';
import { NodeDetails } from './NodeDetails';
import { NodeInspectionPanel } from './NodeInspectionPanel';
import {
  ExecutionMode,
  getExecutionMode,
  isAutoParallelType,
  supportsExecutionModeToggle,
} from '../../core/types/executionMode';
import { getMergeStrategy, type MergeStrategy } from '../../core/types/nodeData';
import { predictMergeConflict } from '../../core/utils/predictMergeConflict';
import { X, Maximize2, Minimize2, Settings2, Merge, LocateFixed } from 'lucide-react';
import { Node } from '@xyflow/react';

export const PropertiesPanel: React.FC = () => {
  const nodes = useGraphStore((state) => state.nodes);
  const {
    isPropertiesPanelExpanded, setPropertiesPanelExpanded,
    propertiesPanelWidth, setPropertiesPanelWidth,
  } = useViewStore();
  const isSidebarOpen = useSidebarOpen();
  const panelRef = React.useRef<HTMLElement>(null);
  const panelId = React.useId();
  const dragStart = React.useRef<{ x: number; width: number } | null>(null);
  const [isResizing, setIsResizing] = React.useState(false);
  const [workspaceWidth, setWorkspaceWidth] = React.useState(0);

  React.useEffect(() => {
    const workspace = panelRef.current?.parentElement;
    if (!workspace) return;
    const observer = new ResizeObserver(([entry]) => {
      if (entry && entry.contentRect.width > 0) setWorkspaceWidth(entry.contentRect.width);
    });
    observer.observe(workspace);
    return () => observer.disconnect();
  }, []);

  // Reserve canvas space when docked; keep the preferred width so it returns
  // when the window grows or the component library closes.
  const maxWidth = Math.max(320, Math.min(720, workspaceWidth - (isSidebarOpen ? 256 : 0) - 400));
  const panelWidth = Math.min(propertiesPanelWidth, maxWidth);
  const resizeTo = (width: number) => setPropertiesPanelWidth(Math.max(320, Math.min(maxWidth, width)));

  const stopResizing = () => {
    dragStart.current = null;
    setIsResizing(false);
  };

  // Find the currently selected node
  const selectedNode = nodes.find((n) => n.selected);

  // Reset expansion when selection clears (optional, but good UX)
  React.useEffect(() => {
    if (!selectedNode) setPropertiesPanelExpanded(false);
  }, [selectedNode, setPropertiesPanelExpanded]);

  // Calculate width based on sidebar state.
  // Chrome consumed to the left of this panel on /canvas:
  //   - global app nav (`components/Layout.tsx`'s `<aside>`) is always
  //     `w-16` (64px) here — `isCollapsed` there is driven purely by
  //     `location.pathname === '/canvas'`, not a user toggle, so this
  //     never changes while this panel is visible.
  //   - the Components drag-and-drop `Sidebar` is `w-64` (256px) when open,
  //     or a floating overlay button (0px in the flex flow) when closed.
  // These must stay in sync with those two components' width classes —
  // a mismatch here causes this panel to overflow the viewport (verified:
  // the previous 300/50 constants were 20px/14px short, clipping the
  // right edge of wide layouts, e.g. BasicTrainingSettings' "Customize"
  // checkbox). +8px extra buffer beyond the exact 320/64 chrome width
  // guards against a stray scrollbar/border consuming a few more px.
  const expandedWidth = isSidebarOpen ? 'w-[calc(100vw-328px)]' : 'w-[calc(100vw-72px)]';

  return (
    <aside
      ref={panelRef}
      id={panelId}
      aria-label="Node settings"
      style={selectedNode && !isPropertiesPanelExpanded ? { width: panelWidth } : undefined}
      className={`relative border-l bg-background shrink-0 overflow-hidden ${isResizing ? '' : 'transition-[width,opacity] duration-300 motion-reduce:transition-none'} ${
        panelVisibilityClass(Boolean(selectedNode), isPropertiesPanelExpanded, expandedWidth)
      }`}
    >
      {selectedNode && !isPropertiesPanelExpanded && (
        // eslint-disable-next-line jsx-a11y/no-noninteractive-element-interactions, jsx-a11y/no-noninteractive-tabindex -- A focusable ARIA separator is an interactive pane-resize widget.
        <div role="separator" tabIndex={0}
          aria-label="Resize settings panel"
          aria-orientation="vertical"
          aria-controls={panelId}
          aria-valuemin={320}
          aria-valuemax={maxWidth}
          aria-valuenow={panelWidth}
          aria-valuetext={`${panelWidth} pixels wide`}
          title="Drag to resize. Left arrow widens; Right arrow narrows. Home resets; End maximizes."
          className="absolute inset-y-0 left-0 z-20 w-2 cursor-col-resize touch-none hover:bg-primary/20 focus-visible:bg-primary/30 focus-visible:outline-none focus-visible:ring-2 focus-visible:ring-inset focus-visible:ring-primary"
          onPointerDown={(event) => {
            if (event.button !== 0) return;
            event.preventDefault();
            event.currentTarget.focus();
            event.currentTarget.setPointerCapture(event.pointerId);
            dragStart.current = { x: event.clientX, width: panelWidth };
            setIsResizing(true);
          }}
          onPointerMove={(event) => {
            if (dragStart.current) resizeTo(dragStart.current.width + dragStart.current.x - event.clientX);
          }}
          onPointerUp={stopResizing}
          onPointerCancel={stopResizing}
          onLostPointerCapture={stopResizing}
          onKeyDown={(event) => {
            const widths: Record<string, number> = {
              ArrowLeft: panelWidth + 20,
              ArrowRight: panelWidth - 20,
              Home: 320,
              End: maxWidth,
            };
            const width = widths[event.key];
            if (width === undefined) return;
            event.preventDefault();
            resizeTo(width);
          }}
        >
          <span aria-hidden="true" className="absolute left-1/2 top-1/2 h-8 w-1 -translate-x-1/2 -translate-y-1/2 rounded-full bg-muted-foreground/40" />
        </div>
      )}
      {selectedNode && (
        <ValidationNavigation nodeId={selectedNode.id}>
        <PropertiesContent
          selectedNode={selectedNode}
          isExpanded={isPropertiesPanelExpanded}
          toggleExpand={() => setPropertiesPanelExpanded(!isPropertiesPanelExpanded)}
        />
        </ValidationNavigation>
      )}
    </aside>
  );
};

const PropertiesContent: React.FC<{
  selectedNode: Node;
  isExpanded: boolean;
  toggleExpand: () => void;
}> = ({ selectedNode, isExpanded, toggleExpand }) => {
  const updateNodeData = useGraphStore((state) => state.updateNodeData);
  const onNodesChange = useGraphStore((state) => state.onNodesChange);
  const tabs = ['Settings', 'Input', 'Output'] as const;
  const [activeTab, setActiveTab] = React.useState<typeof tabs[number]>('Settings');
  const tabsId = React.useId();
  const tabRefs = React.useRef<(HTMLButtonElement | null)[]>([]);
  useValidationReveal(() => setActiveTab('Settings'));

  const handleClose = () => {
    onNodesChange([{ id: selectedNode.id, type: 'select', selected: false }]);
    window.dispatchEvent(new Event(FOCUS_CANVAS_EVENT));
  };

  const definitionType = selectedNode.data.definitionType as string;
  const definition = registry.get(definitionType);

  if (!definition) {
    return (
      <div className="p-4">
        <div className="text-destructive flex items-center gap-2">
          <X className="w-4 h-4" />
          Error: Node definition &apos;{definitionType}&apos; not found.
        </div>
        <NodeDetails key={`info-${selectedNode.id}`} nodeId={selectedNode.id} />
      </div>
    );
  }

  const SettingsComponent = definition.settings;

  return (
    <div className="flex-1 min-h-0 flex flex-col">
      <div className="p-4 border-b flex items-center justify-between gap-2 bg-muted/30">
        <div className="flex items-center gap-2 min-w-0">
          <NodeDetails key={`info-${selectedNode.id}`} nodeId={selectedNode.id} />
          <div className="min-w-0">
            <h2 className="font-semibold text-sm truncate" title={String(selectedNode.data.label || definition.label)}>{String(selectedNode.data.label || definition.label)}</h2>
          </div>
        </div>
        <div className="flex items-center gap-1 shrink-0">
          <button
            type="button"
            onClick={() => window.dispatchEvent(new CustomEvent(FOCUS_NODE_EVENT, { detail: { id: selectedNode.id } }))}
            aria-label="Show node on canvas"
            title="Show node on canvas"
            className="p-1.5 hover:bg-accent rounded-md text-muted-foreground hover:text-foreground transition-colors focus-visible:outline-none focus-visible:ring-2 focus-visible:ring-primary"
          >
            <LocateFixed className="w-4 h-4" />
          </button>
          <button
            type="button"
            onClick={toggleExpand}
            aria-label={isExpanded ? 'Collapse settings panel' : 'Expand settings panel'}
            title={isExpanded ? 'Collapse settings panel' : 'Expand settings panel'}
            className="p-1.5 hover:bg-accent rounded-md text-muted-foreground hover:text-foreground transition-colors focus-ring"
          >
            {isExpanded ? <Minimize2 className="w-4 h-4" /> : <Maximize2 className="w-4 h-4" />}
          </button>
          <button
            type="button"
            onClick={handleClose}
            aria-label="Close settings panel"
            title="Close settings panel"
            className="p-1.5 hover:bg-accent rounded-md text-muted-foreground hover:text-foreground transition-colors focus-ring"
          >
            <X className="w-4 h-4" />
          </button>
        </div>
      </div>

      <div role="tablist" aria-label="Node properties" className="flex shrink-0 border-b px-4">
        {tabs.map((tab, index) => (
          <button key={tab} ref={(element) => { tabRefs.current[index] = element; }} type="button"
            role="tab" id={`${tabsId}-${tab}-tab`} aria-controls={`${tabsId}-${tab === 'Settings' ? 'Settings' : 'inspection'}-panel`}
            aria-selected={activeTab === tab} tabIndex={activeTab === tab ? 0 : -1}
            onClick={() => setActiveTab(tab)}
            onKeyDown={(event) => {
              const nextIndex = event.key === 'ArrowRight' ? (index + 1) % tabs.length
                : event.key === 'ArrowLeft' ? (index + tabs.length - 1) % tabs.length
                  : event.key === 'Home' ? 0 : event.key === 'End' ? tabs.length - 1 : null;
              if (nextIndex === null) return;
              event.preventDefault();
              setActiveTab(tabs[nextIndex]!);
              tabRefs.current[nextIndex]?.focus();
            }}
            className={`flex-1 border-b-2 px-2 py-2.5 text-xs font-medium focus-visible:outline-none focus-visible:ring-2 focus-visible:ring-inset focus-visible:ring-primary ${activeTab === tab
              ? 'border-primary text-primary' : 'border-transparent text-muted-foreground hover:text-foreground'}`}
          >{tab}</button>
        ))}
      </div>

      <div role="tabpanel" id={`${tabsId}-Settings-panel`} aria-labelledby={`${tabsId}-Settings-tab`}
        hidden={activeTab !== 'Settings'} className="flex-1 min-h-0 overflow-y-auto p-4">
        <div className="space-y-6">
          <SettingsComponent
            key={`settings-${selectedNode.id}`}
            config={selectedNode.data}
            onChange={(data: unknown) => updateNodeData(selectedNode.id, data)}
            nodeId={selectedNode.id}
            isExpanded={isExpanded}
          />
          <MultiInputModeSection selectedNode={selectedNode} />
          <MergeStrategySection selectedNode={selectedNode} />
        </div>
      </div>
      <div role="tabpanel" id={`${tabsId}-inspection-panel`} aria-labelledby={`${tabsId}-${activeTab === 'Input' ? 'Input' : 'Output'}-tab`}
        hidden={activeTab === 'Settings'} className="flex-1 min-h-0 min-w-0 overflow-y-auto p-4">
        {activeTab !== 'Settings' && <NodeInspectionPanel nodeId={selectedNode.id} side={activeTab === 'Input' ? 'input' : 'output'} />}
      </div>
    </div>
  );
};

/**
 * Modeling nodes can either merge their multiple upstream inputs into one
 * dataset or fan out and run each input as a separate experiment. This toggle
 * lives in the properties panel so it sits right above the related Merge
 * Strategy dropdown instead of being buried in each settings panel's footer.
 */
const MultiInputModeSection: React.FC<{ selectedNode: Node }> = ({ selectedNode }) => {
  const nodes = useGraphStore((state) => state.nodes);
  const edges = useGraphStore((state) => state.edges);
  const setExecutionMode = useGraphStore((state) => state.setExecutionMode);

  const definitionType = selectedNode.data.definitionType as string;

  // Only modeling nodes opt in to this toggle today.
  const supportsToggle = supportsExecutionModeToggle(definitionType);
  const incomingSourceIds = edges
    .filter((e) => e.target === selectedNode.id)
    .map((e) => e.source)
    .filter((id, index, arr) => arr.indexOf(id) === index);
  const incomingSourceCount = incomingSourceIds.length;

  if (!supportsToggle || incomingSourceCount < 2) return null;

  const current: ExecutionMode = getExecutionMode(selectedNode.data);
  const lastSource = incomingSourceIds[incomingSourceIds.length - 1]!;
  const lastLabel =
    (nodes.find((n) => n.id === lastSource)?.data.label as string | undefined) ?? lastSource;

  return (
    <div className="border-t pt-4">
      <div className="flex items-center gap-2 mb-2">
        <Settings2 className="w-4 h-4 text-muted-foreground" />
        <h3 className="text-sm font-semibold">Multi-Input Mode</h3>
      </div>
      <p className="text-xs text-muted-foreground mb-2">
        Merge combines all inputs into one dataset. Parallel runs each input as a separate experiment.
      </p>
      <div className="flex rounded-md overflow-hidden border border-slate-300 dark:border-slate-600 text-xs font-medium w-fit">
        <button
          type="button"
          aria-pressed={current === 'merge'}
          onClick={() => setExecutionMode(selectedNode.id, 'merge')}
          className={`px-3 py-1.5 transition-colors focus-visible:outline-none focus-visible:ring-2 focus-visible:ring-inset focus-visible:ring-primary ${
            current === 'merge'
              ? 'bg-primary/15 text-primary'
              : 'bg-white dark:bg-slate-700 text-slate-500 dark:text-slate-400 hover:bg-slate-100 dark:hover:bg-slate-600'
          }`}
        >
          Merge
        </button>
        <button
          type="button"
          aria-pressed={current === 'parallel'}
          onClick={() => setExecutionMode(selectedNode.id, 'parallel')}
          className={`px-3 py-1.5 transition-colors focus-visible:outline-none focus-visible:ring-2 focus-visible:ring-inset focus-visible:ring-primary ${
            current === 'parallel'
              ? 'bg-primary/15 text-primary'
              : 'bg-white dark:bg-slate-700 text-slate-500 dark:text-slate-400 hover:bg-slate-100 dark:hover:bg-slate-600'
          }`}
        >
          Parallel
        </button>
      </div>
      {current === 'merge' && (
        <p className="text-xs text-muted-foreground mt-2">
          If two branches carry the same column, the last connected branch (
          <span className="font-medium text-foreground">{lastLabel}</span>) wins it by default.
          After a run detects a conflict, the Merge Strategy dropdown lets you pick the winner.
        </p>
      )}
    </div>
  );
};

const MergeStrategySection: React.FC<{ selectedNode: Node }> = ({ selectedNode }) => {
  const fieldId = React.useId();
  const nodes = useGraphStore((state) => state.nodes);
  const edges = useGraphStore((state) => state.edges);
  const updateNodeData = useGraphStore((state) => state.updateNodeData);
  const executionResult = useGraphStore((state) => state.executionResult);

  if (!nodeMergesColumns(selectedNode, edges)) return null;

  // Branches editing different columns have an unambiguous owner per column,
  // so the engine never needs a tiebreak and this setting would do nothing.
  // A run's `sibling_fan_in` advisory is authoritative (it diffs real values);
  // before the first run we fall back to a config-time prediction so the
  // control is discoverable while wiring, not only after a run.
  const conflict = mergeConflictDetails(selectedNode.id, nodes, edges, executionResult);
  if (!conflict) return null;
  const { advisory, columns: contestedColumns, inputs: contestingInputs } = conflict;
  const current = getMergeStrategy(selectedNode.data);

  const { firstLabel, lastLabel, selectedLabel } = mergeBranchLabels(selectedNode, nodes, contestingInputs);

  return (
    <div className="border-t pt-4">
      <div className="flex items-center gap-2 mb-2">
        <Merge className="w-4 h-4 text-muted-foreground" />
        <h3 className="text-sm font-semibold"><label htmlFor={fieldId}>Merge Strategy</label></h3>
        {!advisory && (
          <span className="text-[10px] uppercase tracking-wide px-1.5 py-0.5 rounded bg-muted text-muted-foreground">
            Predicted
          </span>
        )}
      </div>
      <p className="text-xs text-muted-foreground mb-1">
        At <span className="font-medium text-foreground">{selectedLabel}</span>,{' '}
        {contestedColumns.length} column
        {contestedColumns.length === 1 ? ' is' : 's are'} changed by both{' '}
        <span className="font-medium text-foreground">{firstLabel}</span> and{' '}
        <span className="font-medium text-foreground">{lastLabel}</span>
        {contestedColumns.length > 0 ? `: ${contestedColumns.slice(0, 4).join(', ')}` : ''}
        {contestedColumns.length > 4 ? `, +${contestedColumns.length - 4} more` : ''}.
      </p>
      <p className="text-xs text-muted-foreground mb-2">
        {advisory
          ? 'Pick which branch’s version of those columns to keep. Every other column is unaffected.'
          : 'Based on the current node settings. Run a preview to confirm which columns actually collide.'}
      </p>
      <select
        id={fieldId}
        value={current}
        onChange={(e) =>
          updateNodeData(selectedNode.id, { merge_strategy: e.target.value as MergeStrategy })
        }
        className="w-full px-2 py-1.5 text-sm bg-background border rounded-md"
      >
        <option value="last_wins">Keep {lastLabel} (last connected, default)</option>
        <option value="first_wins">Keep {firstLabel} (first connected)</option>
      </select>
      <p className="text-xs text-muted-foreground mt-2">
        After a Split, ownership doesn’t apply — the winning branch takes every overlapping
        column. Keep post-split branches disjoint or fully numeric, or training fails on leftover
        string columns.
      </p>
    </div>
  );
};

function panelVisibilityClass(hasSelection: boolean, expanded: boolean, expandedWidth: string) {
  return hasSelection ? (expanded ? `${expandedWidth} opacity-100` : 'opacity-100') : 'w-0 opacity-0';
}

/** Match winner options to the first and last connected branch, including unnamed nodes. */
function mergeBranchLabels(selectedNode: Node, nodes: Node[], contestingInputs: string[]) {
  const labelOf = (nodeId: string) => {
    const node = nodes.find((n) => n.id === nodeId);
    return (node?.data.label as string | undefined) ?? nodeId;
  };
  const firstLabel = contestingInputs.length ? labelOf(contestingInputs[0]!) : 'the first branch';
  const lastLabel = contestingInputs.length
    ? labelOf(contestingInputs[contestingInputs.length - 1]!)
    : 'the last branch';
  const selectedLabel = (selectedNode.data.label as string | undefined) ?? selectedNode.id;
  return { firstLabel, lastLabel, selectedLabel };
}

/** Exclude terminals and execution modes whose fan-in never merges columns. */
function nodeMergesColumns(selectedNode: Node, edges: ReturnType<typeof useGraphStore.getState>['edges']) {
  const definitionType = selectedNode.data.definitionType as string;
  const definition = registry.get(definitionType);
  const canMerge = (definition?.inputs?.length ?? 0) > 0;
  const incomingSourceCount = new Set(
    edges.filter((e) => e.target === selectedNode.id).map((e) => e.source)
  ).size;

  // Auto-parallel terminals (data_preview) render each input in its own
  // tab instead of merging columns, so the merge-strategy dropdown is
  // meaningless for them. Sourced from `core/types/executionMode` so the
  // canvas / engine / UI all agree on which types are auto-parallel.
  const isAutoParallel = isAutoParallelType(definitionType);

  // The Ensemble node's fan-in is one dataset edge plus N model-spec edges
  // (base learners), not a column merge — so the "resolve overlapping columns"
  // strategy is meaningless here. The pipeline converter separates these inputs
  // by source type and never column-merges them.
  const isEnsemble = definitionType === 'EnsembleNode';

  // Modeling nodes expose an explicit Multi-Input Mode toggle (merge / parallel).
  // When the user picks "parallel", merging is skipped at runtime, so the
  // strategy dropdown would be misleading. Hide it in that case.
  const isParallelMode = getExecutionMode(selectedNode.data) === 'parallel';

  // Only expose the strategy when the node actually merges: multi-input
  // node with 2+ distinct upstream sources, not an auto-parallel terminal,
  // not an ensemble (model-spec fan-in), and not explicitly set to parallel.
  return canMerge && incomingSourceCount >= 2 && !isAutoParallel && !isEnsemble && !isParallelMode;
}

/** Runtime evidence takes precedence over the configuration-time prediction. */
function mergeConflictDetails(nodeId: string, nodes: Node[], edges: ReturnType<typeof useGraphStore.getState>['edges'],
  executionResult: ReturnType<typeof useGraphStore.getState>['executionResult']) {
  const advisory = (executionResult?.merge_warnings ?? []).find(
    (w) => w.kind === 'sibling_fan_in' && w.node_id === nodeId
  );
  const predicted = advisory ? null : predictMergeConflict(nodeId, nodes, edges);
  if (!advisory && !predicted) return null;

  const columns = advisory ? (advisory.overlap_columns ?? []) : predicted!.columns;
  const inputs = advisory ? (advisory.inputs ?? []) : predicted!.branchIds;
  return { advisory, columns, inputs };
}
