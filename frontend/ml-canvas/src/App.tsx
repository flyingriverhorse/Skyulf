import { useEffect } from 'react'
import { registry } from './core/registry/NodeRegistry'
import { DebugNode } from './modules/nodes/base/DebugNode'
import { DatasetNode } from './modules/nodes/data/DatasetNode'
import { DropColumnsNode } from './modules/nodes/processing/DropColumnsNode'
import { ImputationNode } from './modules/nodes/processing/ImputationNode'
import { ScalingNode } from './modules/nodes/processing/ScalingNode'
import { OneHotEncodingNode } from './modules/nodes/processing/OneHotEncodingNode'
import { LabelEncodingNode } from './modules/nodes/processing/LabelEncodingNode'
import { ModelTrainingNode } from './modules/nodes/modeling/ModelTrainingNode'
import { EvaluationNode } from './modules/nodes/modeling/EvaluationNode'
import { MainLayout } from './components/layout/MainLayout'
import { useGraphStore } from './core/store/useGraphStore'

// Register nodes (normally this would happen in a bootstrap file)
registry.register(DebugNode);
registry.register(DatasetNode);
registry.register(DropColumnsNode);
registry.register(ImputationNode);
registry.register(ScalingNode);
registry.register(OneHotEncodingNode);
registry.register(LabelEncodingNode);
registry.register(ModelTrainingNode);
registry.register(EvaluationNode);

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
