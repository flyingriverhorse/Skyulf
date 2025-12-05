import React, { useEffect, useState } from 'react';
import { NodeDefinition } from '../../../core/types/nodes';
import { Database } from 'lucide-react';
import { DatasetService } from '../../../core/api/datasets';
import { Dataset } from '../../../core/types/api';

interface DatasetNodeConfig {
  datasetId: string;
  datasetName?: string;
}

const DatasetSettings: React.FC<{ config: DatasetNodeConfig; onChange: (c: DatasetNodeConfig) => void }> = ({
  config,
  onChange,
}) => {
  const [datasets, setDatasets] = useState<Dataset[]>([]);
  const [loading, setLoading] = useState(false);

  useEffect(() => {
    setLoading(true);
    DatasetService.getAll()
      .then(setDatasets)
      .catch(console.error)
      .finally(() => setLoading(false));
  }, []);

  const handleChange = (e: React.ChangeEvent<HTMLSelectElement>) => {
    const selectedId = e.target.value;
    const selectedDataset = datasets.find(d => String(d.id) === selectedId);
    onChange({ 
      ...config, 
      datasetId: selectedId,
      datasetName: selectedDataset?.name 
    });
  };

  return (
    <div className="p-4 space-y-2">
      <label className="block text-sm font-medium">Select Dataset</label>
      {loading ? (
        <div className="text-xs text-muted-foreground">Loading datasets...</div>
      ) : (
        <select
          className="w-full p-2 border rounded bg-background"
          value={config.datasetId || ''}
          onChange={handleChange}
        >
          <option value="">-- Select --</option>
          {datasets.map((d) => (
            <option key={d.id} value={d.id}>
              {d.name}
            </option>
          ))}
        </select>
      )}
    </div>
  );
};

const DatasetNodeComponent: React.FC<{ data: DatasetNodeConfig }> = ({ data }) => {
  // Fallback state for backward compatibility with nodes that only have ID
  const [fetchedName, setFetchedName] = useState<string | undefined>(undefined);

  useEffect(() => {
    // Only fetch if we have an ID but no Name, and haven't fetched yet
    if (data.datasetId && !data.datasetName && fetchedName === undefined) {
      DatasetService.getById(data.datasetId)
        .then(d => setFetchedName(d.name))
        .catch(() => setFetchedName(data.datasetId)); // Fallback to ID on error
    }
  }, [data.datasetId, data.datasetName, fetchedName]);

  const displayName = data.datasetName || fetchedName || data.datasetId;

  return (
    <div className="text-xs">
      {data.datasetId ? (
        <div className="font-medium truncate max-w-[150px]" title={displayName}>
          {displayName || 'Loading...'}
        </div>
      ) : (
        <div className="text-muted-foreground italic">Select a dataset</div>
      )}
    </div>
  );
};

export const DatasetNode: NodeDefinition<DatasetNodeConfig> = {
  type: 'dataset_node',
  label: 'Dataset',
  category: 'Data Source',
  description: 'Load a dataset from the registry.',
  icon: Database,
  inputs: [],
  outputs: [{ id: 'data', label: 'Data', type: 'dataset' }],
  settings: DatasetSettings,
  component: DatasetNodeComponent,
  validate: (config) => {
    return {
      isValid: !!config.datasetId,
      message: !config.datasetId ? 'Dataset is required' : undefined,
    };
  },
  getDefaultConfig: () => ({
    datasetId: '',
  }),
};
