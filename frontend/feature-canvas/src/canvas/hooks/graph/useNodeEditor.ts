import { useCallback, useRef } from 'react';
import type { Dispatch, MutableRefObject, SetStateAction } from 'react';
import type { Edge, Node, ReactFlowInstance } from 'react-flow-renderer';
import type { FeatureNodeCatalogEntry } from '../../../api';
import { cloneConfig, sanitizeDefaultConfigForNode, PENDING_CONFIRMATION_FLAG } from '../../services/configSanitizer';
import { resolveDropPosition } from '../../services/layout';
import { registerNodeInteractions, createNewNode, isResettableCatalogEntry } from '../../services/nodeFactory';
import type { FeatureNodeData } from '../../types/nodes';

 type SetNodesFn = Dispatch<SetStateAction<Node[]>>;
 type SetEdgesFn = Dispatch<SetStateAction<Edge[]>>;
 type SetStringStateFn = Dispatch<SetStateAction<string | null>>;
 type SetBooleanStateFn = Dispatch<SetStateAction<boolean>>;
 
 type UseNodeEditorOptions = {
   setNodes: SetNodesFn;
   setEdges: SetEdgesFn;
   setSelectedNodeId: SetStringStateFn;
   setIsSettingsModalOpen: SetBooleanStateFn;
   catalogEntryMapRef: MutableRefObject<Map<string, FeatureNodeCatalogEntry>>;
   handleOpenSettings: (nodeId: string) => void;
   reactFlowInstanceRef: MutableRefObject<ReactFlowInstance | null>;
   canvasViewportRef: MutableRefObject<HTMLDivElement | null>;
   scheduleFitView: () => void;
 };
 
 type UseNodeEditorResult = {
   registerNode: (node: Node) => Node;
   createNode: (catalogNode?: FeatureNodeCatalogEntry | null, position?: { x: number; y: number }) => Node;
   handleAddNode: (catalogNode: FeatureNodeCatalogEntry) => void;
   handleRemoveNode: (nodeId: string) => void;
   handleUpdateNodeConfig: (nodeId: string, nextConfig: Record<string, any>) => void;
   handleUpdateNodeData: (nodeId: string, dataUpdates: Partial<FeatureNodeData>) => void;
   handleResetNodeConfig: (nodeId: string, template?: Record<string, any> | null) => void;
   handleResetAllNodes: () => void;
   updateNodeCounter: (list: Node[]) => void;
 };
 
 export const useNodeEditor = ({
   setNodes,
   setEdges,
   setSelectedNodeId,
   setIsSettingsModalOpen,
   catalogEntryMapRef,
   handleOpenSettings,
   reactFlowInstanceRef,
   canvasViewportRef,
   scheduleFitView,
 }: UseNodeEditorOptions): UseNodeEditorResult => {
   const nodeIdRef = useRef(0);
 
   const handleRemoveNode = useCallback(
     (nodeId: string) => {
       if (nodeId === 'dataset-source') {
         return;
       }
 
       setNodes((current) => current.filter((node) => node.id !== nodeId));
       setEdges((current) => current.filter((edge) => edge.source !== nodeId && edge.target !== nodeId));
       setSelectedNodeId((currentSelected) => {
         if (currentSelected === nodeId) {
           setIsSettingsModalOpen(false);
           return null;
         }
         return currentSelected;
       });
     },
     [setEdges, setIsSettingsModalOpen, setNodes, setSelectedNodeId]
   );
 
   const registerNode = useCallback(
     (node: Node) =>
       registerNodeInteractions(node, {
         handleOpenSettings,
         handleRemoveNode,
         catalogEntryMap: catalogEntryMapRef.current,
       }),
     [catalogEntryMapRef, handleOpenSettings, handleRemoveNode]
   );
 
   const updateNodeCounter = useCallback((list: Node[]) => {
     const highestId = list.reduce((max, node) => {
       if (typeof node.id === 'string' && node.id.startsWith('node-')) {
         const parsed = Number(node.id.replace('node-', ''));
         if (!Number.isNaN(parsed)) {
           return Math.max(max, parsed);
         }
       }
       return max;
     }, 0);
 
     if (highestId > nodeIdRef.current) {
       nodeIdRef.current = highestId;
     }
   }, []);
 
   const createNode = useCallback(
     (catalogNode?: FeatureNodeCatalogEntry | null, position?: { x: number; y: number }) => {
       const nextNumericId = nodeIdRef.current + 1;
       const nodeId = `node-${nextNumericId}`;
       nodeIdRef.current = nextNumericId;
 
       const basePosition =
         position ?? {
           x: 160 + ((nextNumericId - 1) % 4) * 220,
           y: 40 + Math.floor((nextNumericId - 1) / 4) * 160,
         };
 
       return registerNode(
         createNewNode(catalogNode ?? null, nodeId, basePosition, `Step ${nextNumericId}`, {
           handleOpenSettings,
           handleRemoveNode,
           catalogEntryMap: catalogEntryMapRef.current,
         })
       );
     },
     [catalogEntryMapRef, handleOpenSettings, handleRemoveNode, registerNode]
   );
 
   const handleAddNode = useCallback(
     (catalogNode: FeatureNodeCatalogEntry) => {
       setNodes((current) => {
         const dropPosition = resolveDropPosition(current, reactFlowInstanceRef.current, canvasViewportRef.current);
         const newNode = createNode(catalogNode, dropPosition);
         return [...current, newNode];
       });
       scheduleFitView();
     },
     [canvasViewportRef, createNode, reactFlowInstanceRef, scheduleFitView, setNodes]
   );
 
   const handleUpdateNodeConfig = useCallback(
     (nodeId: string, nextConfig: Record<string, any>) => {
       setNodes((current) =>
         current.map((node) => {
           if (node.id !== nodeId) {
             return node;
           }
 
           const sanitizedConfig = cloneConfig(nextConfig);
           if (sanitizedConfig && typeof sanitizedConfig === 'object') {
             delete (sanitizedConfig as Record<string, any>)[PENDING_CONFIRMATION_FLAG];
           }
 
           const baseData = {
             ...(node.data ?? {}),
             config: sanitizedConfig,
             isConfigured: true,
           };
 
           return registerNode({
             ...node,
             data: baseData,
           });
         })
       );
     },
     [registerNode, setNodes]
   );
 
   const handleUpdateNodeData = useCallback(
     (nodeId: string, dataUpdates: Partial<FeatureNodeData>) => {
       setNodes((current) =>
         current.map((node) => {
           if (node.id !== nodeId) {
             return node;
           }
 
           const baseData = {
             ...(node.data ?? {}),
             ...dataUpdates,
           };
 
           return registerNode({
             ...node,
             data: baseData,
           });
         })
       );
     },
     [registerNode, setNodes]
   );
 
   const handleResetNodeConfig = useCallback(
     (nodeId: string, template?: Record<string, any> | null) => {
       setNodes((current) =>
         current.map((node) => {
           if (node.id !== nodeId) {
             return node;
           }
 
           const catalogType = node?.data?.catalogType;
           const catalogEntry = catalogType ? catalogEntryMapRef.current.get(catalogType) ?? null : null;
           const resolvedTemplate =
             template && typeof template === 'object'
               ? cloneConfig(template)
               : sanitizeDefaultConfigForNode(catalogEntry ?? null);
 
           const baseData = {
             ...(node.data ?? {}),
             config: resolvedTemplate,
             isConfigured: false,
             backgroundExecutionStatus: 'idle',
           };
 
           return registerNode({
             ...node,
             data: baseData,
           });
         })
       );
     },
     [catalogEntryMapRef, registerNode, setNodes]
   );
 
   const handleResetAllNodes = useCallback(() => {
     setNodes((current) =>
       current.map((node) => {
         if (!isResettableCatalogEntry(node?.data?.catalogType, catalogEntryMapRef.current)) {
           return node;
         }
 
         const catalogType = node?.data?.catalogType;
         const catalogEntry = catalogType ? catalogEntryMapRef.current.get(catalogType) ?? null : null;
         const sanitizedConfig = sanitizeDefaultConfigForNode(catalogEntry ?? null);
 
         const baseData = {
           ...(node.data ?? {}),
           config: sanitizedConfig,
           isConfigured: false,
           backgroundExecutionStatus: 'idle',
         };
 
         return registerNode({
           ...node,
           data: baseData,
         });
       })
     );
   }, [catalogEntryMapRef, registerNode, setNodes]);
 
   return {
     registerNode,
     createNode,
     handleAddNode,
     handleRemoveNode,
     handleUpdateNodeConfig,
     handleUpdateNodeData,
     handleResetNodeConfig,
     handleResetAllNodes,
     updateNodeCounter,
   };
 };
