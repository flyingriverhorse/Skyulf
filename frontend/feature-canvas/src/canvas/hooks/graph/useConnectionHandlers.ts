import { useCallback } from 'react';
import type { Dispatch, SetStateAction } from 'react';
import type { Connection, Edge, Node } from 'react-flow-renderer';
import { addEdge } from 'react-flow-renderer';
import {
  CONNECTION_ACCEPT_MATCHERS,
  NODE_HANDLE_CONFIG,
  extractHandleKey,
} from '../../constants/nodeHandles';

 type SetEdgesFn = Dispatch<SetStateAction<Edge[]>>;
 
 type UseConnectionHandlersOptions = {
   nodes: Node[];
   setEdges: SetEdgesFn;
   scheduleNodeInternalsUpdate: (nodeIds: string | string[]) => void;
 };
 
 type UseConnectionHandlersResult = {
   isValidConnection: (connection: Connection) => boolean;
   onConnect: (params: Edge | Connection) => void;
 };
 
 export const useConnectionHandlers = ({
   nodes,
   setEdges,
   scheduleNodeInternalsUpdate,
 }: UseConnectionHandlersOptions): UseConnectionHandlersResult => {
   const isValidConnection = useCallback(
     (connection: Connection) => {
       const { source, target, sourceHandle, targetHandle } = connection;
 
       if (!source || !target || !targetHandle) {
         return false;
       }
 
       if (source === target) {
         return false;
       }
 
       const targetNode = nodes.find((node) => node.id === target);
       if (!targetNode) {
         return false;
       }
 
       const targetCatalogType = targetNode?.data?.catalogType ?? '';
       const handleConfig = targetCatalogType ? NODE_HANDLE_CONFIG[targetCatalogType] : undefined;
 
       if (handleConfig?.inputs?.length) {
         const handleKey = extractHandleKey(target, targetHandle);
         if (!handleKey) {
           return false;
         }
 
         const inputDefinition = handleConfig.inputs.find((definition) => definition.key === handleKey);
         if (!inputDefinition) {
           return false;
         }
 
         if (!sourceHandle) {
           return false;
         }
 
         if (inputDefinition.accepts && inputDefinition.accepts.length > 0) {
           return inputDefinition.accepts.some((matcherKey) => {
             const matcher = CONNECTION_ACCEPT_MATCHERS[matcherKey];
             return matcher ? matcher(sourceHandle) : false;
           });
         }
 
         return true;
       }
 
       return true;
     },
     [nodes]
   );
 
   const onConnect = useCallback(
     (params: Edge | Connection) => {
       setEdges((eds) =>
         addEdge(
           {
             ...params,
             type: 'animatedEdge',
             animated: true,
           },
           eds
         )
       );
 
       const impactedNodes = [params.source as string | undefined, params.target as string | undefined].filter(
         (value): value is string => Boolean(value)
       );
       if (impactedNodes.length) {
         scheduleNodeInternalsUpdate(impactedNodes);
       }
     },
     [scheduleNodeInternalsUpdate, setEdges]
   );
 
   return {
     isValidConnection,
     onConnect,
   };
 };
