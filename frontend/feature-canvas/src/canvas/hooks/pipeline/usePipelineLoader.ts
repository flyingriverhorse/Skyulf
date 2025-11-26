import { useEffect, useRef } from 'react';
import type { Dispatch, SetStateAction } from 'react';
import type { Edge, Node } from 'react-flow-renderer';
import { useQuery } from '@tanstack/react-query';
import { fetchPipeline } from '../../../api';
import { getDefaultNodes, getSamplePipelineGraph } from '../../services/layout';
import type { CanvasShellProps } from '../../types/pipeline';

 type SetNodesFn = Dispatch<SetStateAction<Node[]>>;
 type SetEdgesFn = Dispatch<SetStateAction<Edge[]>>;
 type PrepareNodesFn = (nodes: Node[]) => Node[];
 
 type UsePipelineLoaderOptions = {
   sourceId?: string | null;
   datasetDisplayLabel: string;
   registerNode: (node: Node) => Node;
   prepareNodes: PrepareNodesFn;
   setNodes: SetNodesFn;
   setEdges: SetEdgesFn;
   updateNodeCounter: (nodes: Node[]) => void;
   scheduleFitView: () => void;
   onPipelineHydrated?: CanvasShellProps['onPipelineHydrated'];
   onPipelineError?: CanvasShellProps['onPipelineError'];
 };
 
 type UsePipelineLoaderResult = {
   isPipelineLoading: boolean;
 };
 
 export const usePipelineLoader = ({
   sourceId,
   datasetDisplayLabel,
   registerNode,
   prepareNodes,
   setNodes,
   setEdges,
   updateNodeCounter,
   scheduleFitView,
   onPipelineHydrated,
   onPipelineError,
 }: UsePipelineLoaderOptions): UsePipelineLoaderResult => {
   const hasInitialSampleHydratedRef = useRef(false);
 
   const pipelineQuery = useQuery({
     queryKey: ['feature-canvas', 'pipeline', sourceId],
     queryFn: () => fetchPipeline(sourceId as string),
     enabled: Boolean(sourceId),
     staleTime: 60 * 1000,
     retry: 1,
   });
 
   useEffect(() => {
     const datasetNodeLabel = `Dataset input\n(${datasetDisplayLabel})`;
 
     if (!sourceId) {
       if (!hasInitialSampleHydratedRef.current) {
         const sample = getSamplePipelineGraph(datasetDisplayLabel);
         const preparedNodes = prepareNodes(sample.nodes);
         hasInitialSampleHydratedRef.current = true;
         setNodes(preparedNodes);
         setEdges(sample.edges);
         updateNodeCounter(preparedNodes);
         scheduleFitView();
         onPipelineHydrated?.({ nodes: preparedNodes, edges: sample.edges, pipeline: null, context: 'sample' });
       } else {
         setNodes((existing) => {
           let changed = false;
           const next = existing.map((node) => {
             if (node.id !== 'dataset-source') {
               return node;
             }
             const currentLabel = node?.data?.label ?? '';
             if (currentLabel === datasetNodeLabel) {
               return node;
             }
             changed = true;
             return registerNode({
               ...node,
               data: {
                 ...(node.data ?? {}),
                 label: datasetNodeLabel,
                 isDataset: true,
                 isRemovable: false,
               },
             });
           });
           return changed ? next : existing;
         });
       }
 
       return;
     }
 
     hasInitialSampleHydratedRef.current = false;
 
     if (pipelineQuery.isLoading) {
       setNodes((existing) =>
         existing.map((node) =>
           node.id === 'dataset-source'
             ? registerNode({
                 ...node,
                 data: {
                   ...(node.data ?? {}),
                   label: datasetNodeLabel,
                   isDataset: true,
                   isRemovable: false,
                 },
               })
             : node
         )
       );
       return;
     }
 
     if (pipelineQuery.isError) {
       const pipelineError = (pipelineQuery.error as Error) ?? new Error('Failed to load saved pipeline');
       console.error('Failed to load saved pipeline', pipelineError);
       onPipelineError?.(pipelineError);
       const defaultNodes = getDefaultNodes().map((n) =>
         n.id === 'dataset-source'
           ? {
               ...n,
               data: {
                 ...(n.data ?? {}),
                 label: datasetNodeLabel,
                 isDataset: true,
                 isRemovable: false,
               },
             }
           : n
       );
       const preparedNodes = prepareNodes(defaultNodes);
       setNodes(preparedNodes);
       setEdges([]);
       updateNodeCounter(preparedNodes);
       scheduleFitView();
       onPipelineHydrated?.({ nodes: preparedNodes, edges: [], pipeline: null, context: 'reset' });
       return;
     }
 
     if (pipelineQuery.data) {
       const graph = pipelineQuery.data.graph ?? {};
       const rawNodes = Array.isArray(graph?.nodes) && graph.nodes.length ? (graph.nodes as Node[]) : getDefaultNodes();
       const hydratedNodes = prepareNodes(rawNodes);
       const rawEdges = Array.isArray(graph?.edges) ? graph.edges : [];
       const hydratedEdges = rawEdges.map((edge: any) => {
         const existingType = edge?.type;
         const type = !existingType || existingType === 'smoothstep' || existingType === 'default' ? 'animatedEdge' : existingType;
         const sourceNodeId = edge?.source;
         const targetNodeId = edge?.target;
         const targetNode = hydratedNodes.find((node) => node.id === targetNodeId);
         const normalizedSourceHandle = edge?.sourceHandle ?? (sourceNodeId ? `${sourceNodeId}-source` : undefined);
         const normalizedTargetHandle =
           edge?.targetHandle ?? (targetNode?.data?.isDataset ? undefined : targetNodeId ? `${targetNodeId}-target` : undefined);
         return {
           ...edge,
           animated: edge?.animated ?? true,
           type,
           sourceHandle: normalizedSourceHandle,
           targetHandle: normalizedTargetHandle,
         };
       });
 
       setNodes(hydratedNodes);
       setEdges(hydratedEdges);
       updateNodeCounter(hydratedNodes);
       scheduleFitView();
       onPipelineHydrated?.({
         nodes: hydratedNodes,
         edges: hydratedEdges,
         pipeline: pipelineQuery.data ?? null,
         context: 'stored',
       });
       return;
     }
 
     if (pipelineQuery.isFetched && !pipelineQuery.data) {
       const defaultNodes = getDefaultNodes().map((n) =>
         n.id === 'dataset-source'
           ? {
               ...n,
               data: {
                 ...(n.data ?? {}),
                 label: datasetNodeLabel,
                 isDataset: true,
                 isRemovable: false,
               },
             }
           : n
       );
       const preparedNodes = prepareNodes(defaultNodes);
       setNodes(preparedNodes);
       setEdges([]);
       updateNodeCounter(preparedNodes);
       scheduleFitView();
       onPipelineHydrated?.({ nodes: preparedNodes, edges: [], pipeline: null, context: 'sample' });
     }
   }, [
     datasetDisplayLabel,
     onPipelineHydrated,
     onPipelineError,
     pipelineQuery.data,
     pipelineQuery.error,
     pipelineQuery.isError,
     pipelineQuery.isFetched,
     pipelineQuery.isLoading,
     prepareNodes,
     registerNode,
     scheduleFitView,
     setEdges,
     setNodes,
     sourceId,
     updateNodeCounter,
   ]);
 
   return {
     isPipelineLoading: pipelineQuery.isLoading || pipelineQuery.isFetching,
   };
 };
