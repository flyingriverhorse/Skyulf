import React, { useEffect, useRef } from 'react';
import { useSearchParams } from 'react-router-dom';
import { MainLayout } from '../components/layout/MainLayout';
import { useGraphStore } from '../core/store/useGraphStore';

export const CanvasPage: React.FC = () => {
  const addNode = useGraphStore((state) => state.addNode);
  const nodes = useGraphStore((state) => state.nodes);
  const [searchParams, setSearchParams] = useSearchParams();
  const processedRef = useRef(false);

  useEffect(() => {
    const sourceId = searchParams.get('source_id');

    if (sourceId) {
      // Prevent double-firing in StrictMode
      if (processedRef.current) return;

      // Check if we already have this dataset on the canvas
      const alreadyExists = nodes.some(n => 
        n.type === 'dataset_node' && 
        (n.data as any)?.datasetId === sourceId
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
