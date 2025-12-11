import React from 'react';
import { NodeDefinition } from '../../../core/types/nodes';
import { Split } from 'lucide-react';
import { useDatasetSchema } from '../../../core/hooks/useDatasetSchema';
import { useGraphStore } from '../../../core/store/useGraphStore';
import { getIncomers } from '@xyflow/react';

interface FeatureTargetSplitConfig {
  target_column: string;
  datasetId?: string;
}

const FeatureTargetSplitSettings: React.FC<{ config: FeatureTargetSplitConfig; onChange: (c: FeatureTargetSplitConfig) => void; nodeId?: string }> = ({
  config,
  onChange,
  nodeId,
}) => {
  // Recursive search for datasetId
  const nodes = useGraphStore((state) => state.nodes);
  const edges = useGraphStore((state) => state.edges);

  const findUpstreamDatasetId = (currentNodeId: string): string | undefined => {
    const visited = new Set<string>();
    const queue = [currentNodeId];
    
    while (queue.length > 0) {
      const id = queue.shift()!;
      if (visited.has(id)) continue;
      visited.add(id);
      
      const node = nodes.find(n => n.id === id);
      if (!node) continue;
      
      // If this is NOT the current node, check if it has datasetId
      if (id !== currentNodeId && node.data?.datasetId) {
        return node.data.datasetId as string;
      }
      
      const incomers = getIncomers(node, nodes, edges);
      for (const incomer of incomers) {
        queue.push(incomer.id);
      }
    }
    return undefined;
  };

  const upstreamDatasetId = findUpstreamDatasetId(nodeId || '');

  React.useEffect(() => {
    if (upstreamDatasetId && config.datasetId !== upstreamDatasetId) {
      onChange({ ...config, datasetId: upstreamDatasetId });
    }
  }, [upstreamDatasetId, config.datasetId, onChange]);

  const { data: schema } = useDatasetSchema(upstreamDatasetId || config.datasetId);
  const columns = schema ? Object.values(schema.columns).map(c => c.name) : [];

  return (
    <div className="p-4 space-y-4">
      <div className="space-y-2">
        <label className="text-sm font-medium">Target Column</label>
        <select
          className="w-full p-2 border rounded bg-background text-sm"
          value={config.target_column}
          onChange={(e) => onChange({ ...config, target_column: e.target.value })}
        >
          <option value="">-- Select Target --</option>
          {columns.map(col => (
            <option key={col} value={col}>{col}</option>
          ))}
        </select>
        <p className="text-xs text-muted-foreground">
          This column will be separated as the target (y), and all other columns will be features (X).
        </p>
      </div>
    </div>
  );
};

export const FeatureTargetSplitNode: NodeDefinition<FeatureTargetSplitConfig> = {
  type: 'feature_target_split',
  label: 'Feature-Target Split',
  category: 'Preprocessing',
  description: 'Separate features (X) from target (y).',
  icon: Split,
  inputs: [{ id: 'in', label: 'Data', type: 'dataset' }],
  outputs: [
    { id: 'X', label: 'Features (X)', type: 'dataset' },
    { id: 'y', label: 'Target (y)', type: 'dataset' }
  ],
  settings: FeatureTargetSplitSettings,
  validate: (config) => {
    if (!config.target_column) {
      return { isValid: false, message: 'Target column is required.' };
    }
    return { isValid: true };
  },
  getDefaultConfig: () => ({
    target_column: '',
  }),
};
