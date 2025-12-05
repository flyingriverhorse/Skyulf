import React, { useCallback, useRef } from 'react';
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
    addNode 
  } = useGraphStore(
    useShallow((state) => ({
      nodes: state.nodes,
      edges: state.edges,
      onNodesChange: state.onNodesChange,
      onEdgesChange: state.onEdgesChange,
      onConnect: state.onConnect,
      addNode: state.addNode,
    }))
  );

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
      tabIndex={0} // Make the container focusable to capture key events
    >
      <ReactFlow
        nodes={nodes}
        edges={edges}
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
            stroke: 'url(#edge-gradient)', 
            strokeWidth: 3,
            strokeDasharray: '8 6',
            filter: 'url(#glow)'
          },
          animated: true,
        }}
        edgeTypes={edgeTypes}
        deleteKeyCode={['Backspace', 'Delete']}
      >
        <svg style={{ position: 'absolute', top: 0, left: 0, width: 0, height: 0, pointerEvents: 'none' }}>
          <defs>
            <linearGradient id="edge-gradient">
              <stop offset="0%" stopColor="#38bdf8" />
              <stop offset="50%" stopColor="#6366f1" />
              <stop offset="100%" stopColor="#a855f7" />
            </linearGradient>
            <filter id="glow" x="-50%" y="-50%" width="200%" height="200%">
              <feGaussianBlur stdDeviation="3" result="coloredBlur"/>
              <feMerge>
                <feMergeNode in="coloredBlur"/>
                <feMergeNode in="SourceGraphic"/>
              </feMerge>
            </filter>
          </defs>
        </svg>
        <Background />
        <Controls position="top-left" />
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
