import React, { useEffect, useRef } from 'react';
import { useSearchParams, useLocation, useNavigate } from 'react-router-dom';
import { MainLayout } from '../components/layout/MainLayout';
import { useGraphStore } from '../core/store/useGraphStore';
import type { PipelineVersionEntry } from '../core/api/pipelineVersions';
import { toast } from '../core/toast';
import type { Node, Edge } from '@xyflow/react';

export const CanvasPage: React.FC = () => {
  const addNode = useGraphStore((state) => state.addNode);
  const setGraph = useGraphStore((state) => state.setGraph);
  const nodes = useGraphStore((state) => state.nodes);
  const [searchParams, setSearchParams] = useSearchParams();
  const location = useLocation();
  const navigate = useNavigate();
  const processedRef = useRef(false);
  const restoreProcessedRef = useRef(false);

  // L7: when DataSources -> Versions -> Restore navigates here with a
  // version payload in router state, apply it to the canvas. Runs
  // before the source_id branch so we don't add an extra dataset node
  // on top of the snapshot.
  useEffect(() => {
    const state = location.state as { restoreVersion?: PipelineVersionEntry } | null;
    const entry = state?.restoreVersion;
    if (!entry || restoreProcessedRef.current) return;
    restoreProcessedRef.current = true;

    const graph = entry.graph as { nodes?: Node[]; edges?: Edge[] } | undefined;
    if (graph && Array.isArray(graph.nodes) && Array.isArray(graph.edges)) {
      setGraph(graph.nodes, graph.edges);
      // Mark source_id as processed so the existing dataset-node branch
      // below doesn't fire on top of the restored graph.
      processedRef.current = true;
      const newParams = new URLSearchParams(searchParams);
      newParams.delete('source_id');
      setSearchParams(newParams, { replace: true });
      // Clear router state so a refresh doesn't re-apply.
      navigate(location.pathname + location.search, { replace: true, state: null });
      toast.success(`Restored "${entry.name}"`, `Version v${entry.versionInt}`);
    } else {
      toast.error(
        'Restore failed',
        'Snapshot graph is not in a recognised shape.',
      );
    }
  }, [location, navigate, setGraph, searchParams, setSearchParams]);

  useEffect(() => {
    const sourceId = searchParams.get('source_id');

    if (sourceId) {
      // Prevent double-firing in StrictMode
      if (processedRef.current) return;

      // Check if we already have this dataset on the canvas
      const alreadyExists = nodes.some(n => 
        n.type === 'dataset_node' && 
        (n.data as { datasetId?: string })?.datasetId === sourceId
      );

      if (alreadyExists) {
        // If it exists, just clean the URL
        const newParams = new URLSearchParams(searchParams);
        newParams.delete('source_id');
        setSearchParams(newParams, { replace: true });
        return;
      }

      // Mark as processed to prevent duplicates
      processedRef.current = true;
      
      // Add the node
      addNode('dataset_node', { x: 100, y: 100 }, { datasetId: sourceId });
      
      // Clean up URL
      const newParams = new URLSearchParams(searchParams);
      newParams.delete('source_id');
      setSearchParams(newParams, { replace: true });
    }
  }, [searchParams, addNode, nodes, setSearchParams]);

  return <MainLayout />;
};
