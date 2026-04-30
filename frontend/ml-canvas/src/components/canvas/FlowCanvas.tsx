import React, { useCallback, useEffect, useMemo, useRef } from 'react';
import { 
  ReactFlow, 
  Background, 
  Controls, 
  ReactFlowProvider,
  useReactFlow
} from '@xyflow/react';
import { useShallow } from 'zustand/react/shallow';
import '@xyflow/react/dist/style.css';

import { useGraphStore } from '../../core/store/useGraphStore';
import { useViewStore } from '../../core/store/useViewStore';
import { useClipboard } from '../../core/hooks/useClipboard';
import { useBranchColors } from '../../core/hooks/useBranchColors';
import { useReadOnlyMode } from '../../core/hooks/useReadOnlyMode';
import { useNodeJobSummaries } from '../../core/hooks/useNodeJobSummaries';
import { CustomNodeWrapper } from './CustomNodeWrapper';
import { CustomEdge } from './CustomEdge';
import { useConfirm } from '../shared';
import { Sparkles } from 'lucide-react';
import { SHOW_TEMPLATES_EVENT } from '../../core/hooks/useKeyboardShortcuts';
import { PerfOverlayLegend } from './PerfOverlayLegend';

const nodeTypes = {
  custom: CustomNodeWrapper
};

const edgeTypes = {
  custom: CustomEdge
};

const FlowCanvasContent: React.FC = () => {
  const reactFlowWrapper = useRef<HTMLDivElement>(null);
  const { screenToFlowPosition, fitView } = useReactFlow();
  
  const { 
    nodes, 
    edges, 
    onNodesChange, 
    onEdgesChange, 
    onConnect,
    addNode,
    executionResult
  } = useGraphStore(
    useShallow((state) => ({
      nodes: state.nodes,
      edges: state.edges,
      onNodesChange: state.onNodesChange,
      onEdgesChange: state.onEdgesChange,
      onConnect: state.onConnect,
      addNode: state.addNode,
      executionResult: state.executionResult
    }))
  );

  const { isResultsPanelExpanded, perfOverlayEnabled, setPerfOverlayEnabled } = useViewStore();
  const readOnly = useReadOnlyMode();

  const confirm = useConfirm();

  // Gate Backspace/Delete behind a real confirmation modal. React Flow
  // calls `onBeforeDelete` synchronously and awaits the returned promise;
  // resolving `false` cancels the delete. Single-node deletes are still
  // undoable via Ctrl+Z, but a misclick on a complex pipeline should
  // not silently wipe state.
  const onBeforeDelete = useCallback(
    async ({ nodes: toDelete }: { nodes: typeof nodes; edges: typeof edges }) => {
      if (toDelete.length === 0) return true;
      const count = toDelete.length;
      return confirm({
        title: count === 1 ? 'Delete node?' : `Delete ${count} nodes?`,
        message:
          count === 1
            ? 'The selected node and any connected edges will be removed. You can undo with Ctrl+Z.'
            : `${count} nodes and any connected edges will be removed. You can undo with Ctrl+Z.`,
        confirmLabel: 'Delete',
        variant: 'danger',
      });
    },
    [confirm],
  );

  useClipboard();

  // Keep trainer/tuner card summaries fresh from the backend even
  // though their jobs run via Celery (so the inline preview path
  // never populates `executionResult.node_results` for them).
  useNodeJobSummaries();

  const branchColorMap = useBranchColors(nodes, edges);

  // Mirror the per-edge Path label into the global store so trainer
  // cards (rendered inside individual nodes) can show the exact same
  // "Path B · Xgboost" letters the user sees on the canvas, instead
  // of re-deriving them from `branch_index` (which can drift across
  // multi-terminal partitions).
  const setBranchEdgeLabels = useGraphStore((s) => s.setBranchEdgeLabels);
  useEffect(() => {
    const labels: Record<string, string> = {};
    for (const [edgeId, info] of branchColorMap) {
      if (info.label) labels[edgeId] = info.label;
    }
    setBranchEdgeLabels(labels);
  }, [branchColorMap, setBranchEdgeLabels]);

  // Edges that "won" a merge: for every sibling_fan_in advisory, mark the
  // edge from winner_input -> advisory.node_id so the canvas can highlight
  // which branch's values survived the last-wins / first-wins tiebreak.
  const winnerEdgeIds = useMemo(() => {
    const ids = new Set<string>();
    const warnings = executionResult?.merge_warnings ?? [];
    for (const w of warnings) {
      if (!w.winner_input || !w.node_id) continue;
      const edge = edges.find(
        (e) => e.source === w.winner_input && e.target === w.node_id
      );
      if (edge) ids.add(edge.id);
    }
    return ids;
  }, [edges, executionResult]);

  const coloredEdges = useMemo(() => {
    return edges.map(edge => {
      const info = branchColorMap.get(edge.id);
      const isMergeWinner = winnerEdgeIds.has(edge.id);
      if (info === undefined && !isMergeWinner) return edge;
      return {
        ...edge,
        data: {
          ...edge.data,
          branchColor: info?.color,
          branchLabel: info?.label,
          branchShared: info?.shared,
          isMergeWinner,
        },
      };
    });
  }, [edges, branchColorMap, winnerEdgeIds]);

  const onDragOver = useCallback((event: React.DragEvent) => {
    event.preventDefault();
    event.dataTransfer.dropEffect = 'move';
  }, []);

  // F or Ctrl/Cmd+0 → fit view. Lives here (not in the global keyboard
  // hook) because `fitView` is only available inside ReactFlowProvider.
  // Ctrl/Cmd+0 mirrors the browser-standard "reset zoom" gesture users
  // already have muscle memory for. Skip when typing in form fields so
  // it doesn't fight with text input.
  useEffect(() => {
    const handler = (e: KeyboardEvent): void => {
      const t = e.target as HTMLElement | null;
      const tag = t?.tagName;
      if (
        tag === 'INPUT' ||
        tag === 'TEXTAREA' ||
        tag === 'SELECT' ||
        t?.isContentEditable === true
      ) {
        return;
      }
      const mod = e.ctrlKey || e.metaKey;
      const isPlainF = (e.key === 'f' || e.key === 'F') && !mod && !e.altKey;
      const isModZero = mod && e.key === '0';
      if (!isPlainF && !isModZero) return;
      e.preventDefault();
      fitView({ duration: 250, padding: 0.15 });
    };
    window.addEventListener('keydown', handler);
    return () => window.removeEventListener('keydown', handler);
  }, [fitView]);

  // Command palette → drop selected node at the canvas viewport center.
  // Lives here (not in the palette itself) because screenToFlowPosition
  // requires being inside <ReactFlowProvider>.
  useEffect(() => {
    const handler = (e: Event): void => {
      const detail = (e as CustomEvent<{ type: string }>).detail;
      if (!detail?.type) return;
      const wrapper = reactFlowWrapper.current;
      const rect = wrapper?.getBoundingClientRect();
      const cx = rect ? rect.left + rect.width / 2 : window.innerWidth / 2;
      const cy = rect ? rect.top + rect.height / 2 : window.innerHeight / 2;
      const position = screenToFlowPosition({ x: cx, y: cy });
      addNode(detail.type, position);
    };
    window.addEventListener('skyulf:add-node-at-center', handler);
    return () => window.removeEventListener('skyulf:add-node-at-center', handler);
  }, [screenToFlowPosition, addNode]);

  const onDrop = useCallback(
    (event: React.DragEvent) => {
      event.preventDefault();

      const type = event.dataTransfer.getData('application/reactflow');

      // check if the dropped element is valid
      if (typeof type === 'undefined' || !type) {
        return;
      }

      const position = screenToFlowPosition({
        x: event.clientX,
        y: event.clientY,
      });

      addNode(type, position);
    },
    [screenToFlowPosition, addNode],
  );

  return (
    <div 
      className="w-full h-full outline-none relative" 
      ref={reactFlowWrapper}
      // eslint-disable-next-line jsx-a11y/no-noninteractive-tabindex -- canvas wrapper must be focusable to capture keyboard shortcuts
      tabIndex={0}
    >
      <ReactFlow
        nodes={nodes}
        edges={coloredEdges}
        onNodesChange={onNodesChange}
        onEdgesChange={onEdgesChange}
        onConnect={onConnect}
        nodeTypes={nodeTypes}
        onDragOver={onDragOver}
        onDrop={onDrop}
        // Read-only mode (auto on tablet/mobile, or user toggle): keep
        // pan/zoom/selection so the user can still inspect a pipeline,
        // but disable all mutations and the delete-key handler.
        nodesDraggable={!readOnly}
        nodesConnectable={!readOnly}
        edgesReconnectable={!readOnly}
        elementsSelectable
        deleteKeyCode={readOnly ? null : ['Backspace', 'Delete']}
        fitView
        proOptions={{ hideAttribution: true }}
        defaultEdgeOptions={{
          type: 'custom',
          style: {
            // Solid indigo stroke with the React Flow CSS dash animation
            // (`animated: true`). The previous default referenced an SVG
            // `<linearGradient id="edge-gradient">` which collapsed to an
            // invisible stroke on some near-collinear paths once the glow
            // filter was removed. Solid stroke avoids that without losing
            // the animated-flow affordance. The real perf killer was the
            // `feGaussianBlur` glow filter recomputed every frame; the
            // dash-offset CSS keyframe is cheap.
            stroke: '#6366f1',
            strokeWidth: 2,
            strokeDasharray: '8 6',
          },
          animated: true,
        }}
        edgeTypes={edgeTypes}
        onBeforeDelete={onBeforeDelete}
      >
        <Background />
        <Controls 
          position="bottom-left" 
          style={
            (executionResult && !isResultsPanelExpanded) 
              ? { marginBottom: '40px', transition: 'margin-bottom 0.3s ease-in-out' } 
              : { transition: 'margin-bottom 0.3s ease-in-out' }
          }
        />
      </ReactFlow>
      {perfOverlayEnabled && nodes.length > 0 && (
        // Floating mini-legend so the colored rings on each node are
        // self-explanatory without opening the Toolbar's legend popover.
        // Draggable + collapsible; "hide" also flips the global toggle
        // off so the toolbar Perf button stays in sync.
        <PerfOverlayLegend onHide={() => setPerfOverlayEnabled(false)} />
      )}
      {!readOnly && nodes.length === 0 && (
        // Cold-start helper. Floats above the empty React Flow viewport;
        // pointer-events stay scoped to the inner card so users can still
        // pan/zoom the empty canvas around it. Dispatches to the Toolbar's
        // gallery modal via the same event-bridge pattern used for the
        // command palette and shortcuts overlay.
        <div className="pointer-events-none absolute inset-0 flex items-center justify-center z-10">
          <div className="pointer-events-auto rounded-lg border border-dashed border-border bg-background/80 backdrop-blur px-6 py-5 max-w-sm text-center shadow-sm">
            <div className="text-sm font-medium mb-1">Empty canvas</div>
            <p className="text-xs text-muted-foreground mb-3 leading-relaxed">
              Drag a node from the sidebar, or start from a curated
              pipeline template.
            </p>
            <button
              type="button"
              onClick={() => window.dispatchEvent(new CustomEvent(SHOW_TEMPLATES_EVENT))}
              data-testid="empty-state-templates"
              className="inline-flex items-center gap-1.5 px-3 py-1.5 rounded-md text-xs font-medium bg-primary text-primary-foreground hover:bg-primary/90"
            >
              <Sparkles className="w-3.5 h-3.5" />
              Browse templates
            </button>
          </div>
        </div>
      )}
    </div>
  );
};

export const FlowCanvas: React.FC = () => {
  return (
    <ReactFlowProvider>
      <FlowCanvasContent />
    </ReactFlowProvider>
  );
};
