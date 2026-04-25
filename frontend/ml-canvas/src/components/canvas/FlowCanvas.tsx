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
import { CustomNodeWrapper } from './CustomNodeWrapper';
import { CustomEdge } from './CustomEdge';
import { useConfirm } from '../shared';

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

  const { isResultsPanelExpanded } = useViewStore();

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

  const branchColorMap = useBranchColors(nodes, edges);

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

  // F key → fit view. Lives here (not in the global keyboard hook)
  // because `fitView` is only available inside ReactFlowProvider.
  // Skip when typing in form fields so it doesn't fight with text input.
  useEffect(() => {
    const handler = (e: KeyboardEvent): void => {
      if (e.key !== 'f' && e.key !== 'F') return;
      if (e.ctrlKey || e.metaKey || e.altKey) return;
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
      e.preventDefault();
      fitView({ duration: 250, padding: 0.15 });
    };
    window.addEventListener('keydown', handler);
    return () => window.removeEventListener('keydown', handler);
  }, [fitView]);

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
      className="w-full h-full outline-none" 
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
        deleteKeyCode={['Backspace', 'Delete']}
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
