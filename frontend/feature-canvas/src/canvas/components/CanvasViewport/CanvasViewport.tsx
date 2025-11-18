// @ts-nocheck
import React from 'react';
import type { MutableRefObject } from 'react';
import ReactFlow, {
  Background,
  Connection,
  ConnectionMode,
  Controls,
  Edge,
  EdgeTypes,
  Node,
  NodeTypes,
  OnEdgesChange,
  OnNodesChange,
  ReactFlowInstance,
} from 'react-flow-renderer';
import ConnectionLine from '../../../components/edges/ConnectionLine';

 type CanvasViewportProps = {
   nodes: Node[];
   edges: Edge[];
   nodeTypes: NodeTypes;
   edgeTypes: EdgeTypes;
   onNodesChange: OnNodesChange;
   onEdgesChange: OnEdgesChange;
   onConnect: (params: Edge | Connection) => void;
   isValidConnection: (connection: Connection) => boolean;
   onNodeClick: (event: React.MouseEvent, node: Node) => void;
   reactFlowInstanceRef: MutableRefObject<ReactFlowInstance | null>;
   canvasViewportRef: MutableRefObject<HTMLDivElement | null>;
   handleResetAllNodes: () => void;
   openCatalog: () => void;
 };
 
 export const CanvasViewport: React.FC<CanvasViewportProps> = ({
   nodes,
   edges,
   nodeTypes,
   edgeTypes,
   onNodesChange,
   onEdgesChange,
   onConnect,
   isValidConnection,
   onNodeClick,
   reactFlowInstanceRef,
   canvasViewportRef,
   handleResetAllNodes,
   openCatalog,
 }) => (
   <div
     className="canvas-stage__viewport"
     ref={canvasViewportRef}
   >
     <ReactFlow
       style={{ width: '100%', height: '100%' }}
       nodes={nodes}
       edges={edges}
       nodeTypes={nodeTypes}
       edgeTypes={edgeTypes}
       onNodesChange={onNodesChange}
       onEdgesChange={onEdgesChange}
       onConnect={onConnect}
       isValidConnection={isValidConnection}
       onNodeClick={onNodeClick}
       onInit={(instance) => {
         reactFlowInstanceRef.current = instance;
         requestAnimationFrame(() => instance.fitView({ padding: 0.25 }));
       }}
       minZoom={0.5}
       maxZoom={1.75}
       connectionRadius={180}
       connectionMode={ConnectionMode.Loose}
       connectOnClick
       nodeDragHandle=".feature-node__drag-handle"
       defaultEdgeOptions={{
         type: 'animatedEdge',
         animated: true,
         style: { strokeWidth: 3 },
       }}
       connectionLineComponent={ConnectionLine}
       elevateEdgesOnSelect={true}
       edgesUpdatable={false}
     >
      <Controls position="bottom-left" />
       <Background gap={16} />
     </ReactFlow>
 
     <button
       type="button"
       className="canvas-fab canvas-fab--reset"
       onClick={handleResetAllNodes}
       aria-label="Reset preprocessing nodes"
     >
       ↺
     </button>
 
     <button
       type="button"
       className="canvas-fab"
       onClick={openCatalog}
       aria-label="Open node catalog"
     >
       +
     </button>
   </div>
 );
 
 export default CanvasViewport;
