import { useEffect } from 'react'
import { MainLayout } from './components/layout/MainLayout'
import { useGraphStore } from './core/store/useGraphStore'

function App() {
  const addNode = useGraphStore((state) => state.addNode);

  useEffect(() => {
    // Check for source_id in URL query params
    const params = new URLSearchParams(window.location.search);
    const sourceId = params.get('source_id');

    if (sourceId) {
      // Add a dataset node automatically
      // We use a timeout to ensure the store is ready and to avoid strict mode double-mount issues
      setTimeout(() => {
        addNode('dataset_node', { x: 100, y: 100 }, { datasetId: sourceId });
      }, 100);
    }
  }, []);

  // Simple dark mode toggle for testing (can be moved to a proper settings UI)
  useEffect(() => {
    // document.documentElement.classList.add('dark'); // Uncomment to force dark mode
  }, []);

  return (
    <MainLayout />
  )
}

export default App
