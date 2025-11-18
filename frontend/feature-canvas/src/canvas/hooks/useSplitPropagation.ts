import { useEffect } from 'react';
import type { Dispatch, SetStateAction } from 'react';
import type { Edge, Node } from 'react-flow-renderer';
import {
  computeActiveSplitMap,
  computeSplitConnectionMap,
  checkNodeConnectionStatus,
} from '../services/splitPropagation';
import {
  areSplitArraysEqual,
  sanitizeSplitList,
  SPLIT_TYPE_ORDER,
} from '../constants/splits';
import type { FeatureNodeData } from '../types/nodes';

 type SetNodesFn = Dispatch<SetStateAction<Node[]>>;
 
 type UseSplitPropagationOptions = {
   edges: Edge[];
   setNodes: SetNodesFn;
   scheduleNodeInternalsUpdate: (nodeIds: string | string[]) => void;
 };
 
 export const useSplitPropagation = ({
   edges,
   setNodes,
   scheduleNodeInternalsUpdate,
 }: UseSplitPropagationOptions): void => {
   useEffect(() => {
     const nodesWithHandleChanges: string[] = [];
 
     setNodes((currentNodes) => {
       const activeSplitMap = computeActiveSplitMap(currentNodes, edges);
       const splitConnectionMap = computeSplitConnectionMap(edges);
       let didChange = false;
 
       const nextNodes = currentNodes.map((node) => {
         const isSplitNodeType = node?.data?.catalogType === 'train_test_split';
         const desiredSplits = isSplitNodeType ? [...SPLIT_TYPE_ORDER] : activeSplitMap.get(node.id) ?? [];
         const currentSplits = sanitizeSplitList(node.data?.activeSplits);
         const desiredConnections = sanitizeSplitList(splitConnectionMap.get(node.id));
         const currentConnections = sanitizeSplitList(node.data?.connectedSplits);
 
         const catalogType = node?.data?.catalogType ?? '';
         const hasRequiredConnections = checkNodeConnectionStatus(node.id, catalogType, edges);
         const currentHasRequiredConnections = node.data?.hasRequiredConnections ?? true;
 
         const splitsChanged = !areSplitArraysEqual(currentSplits, desiredSplits);
         const connectionsChanged = !areSplitArraysEqual(currentConnections, desiredConnections);
         const connectionStatusChanged = hasRequiredConnections !== currentHasRequiredConnections;
 
         if (!splitsChanged && !connectionsChanged && !connectionStatusChanged) {
           return node;
         }
 
         const nextData = {
           ...node.data,
           hasRequiredConnections,
         } as FeatureNodeData;
 
         if (desiredSplits.length) {
           nextData.activeSplits = desiredSplits;
         } else {
           delete nextData.activeSplits;
         }
 
         if (desiredConnections.length) {
           nextData.connectedSplits = desiredConnections;
         } else {
           delete nextData.connectedSplits;
         }
 
         if (splitsChanged || connectionsChanged) {
           nodesWithHandleChanges.push(node.id);
         }
 
         didChange = true;
         return {
           ...node,
           data: nextData,
         };
       });
 
       return didChange ? nextNodes : currentNodes;
     });
 
     if (nodesWithHandleChanges.length) {
       scheduleNodeInternalsUpdate(nodesWithHandleChanges);
     }
   }, [edges, scheduleNodeInternalsUpdate, setNodes]);
 };
