import React from 'react';
import { MainLayout } from '../components/layout/MainLayout';
import { useGraphStore } from '../core/store/useGraphStore';

export const CanvasPage: React.FC = () => {
  const addNode = useGraphStore((state) => state.addNode);

  React.useEffect(() => {
    // Check for source_id in URL query params
    const params = new URLSearchParams(window.location.search);
    const sourceId = params.get('source_id');

    if (sourceId) {
      setTimeout(() => {
        addNode('dataset_node', { x: 100, y: 100 }, { datasetId: sourceId });
      }, 100);
    }
  }, []);

  return <MainLayout />;
};
