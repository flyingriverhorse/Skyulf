import React, { useCallback, useMemo, useRef } from 'react';
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

const nodeTypes = {
  custom: CustomNodeWrapper
};

const edgeTypes = {
  custom: CustomEdge
};

const FlowCanvasContent: React.FC = () => {
  const reactFlowWrapper = useRef<HTMLDivElement>(null);
  const { screenToFlowPosition } = useReactFlow();
  
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
