import React, { useEffect, useState } from 'react';
import { NodeDefinition } from '../../../core/types/nodes';
import { Database, TableProperties, Plus } from 'lucide-react';
import { DatasetService } from '../../../core/api/datasets';
import { Dataset } from '../../../core/types/api';
import { useDatasetSchema } from '../../../core/hooks/useDatasetSchema';
import { FileUpload } from './FileUpload';

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
  const [showUpload, setShowUpload] = useState(false);
  const { data: schema, isLoading: isSchemaLoading } = useDatasetSchema(config.datasetId);

  const fetchDatasets = () => {
    setLoading(true);
    DatasetService.getUsable()
      .then(setDatasets)
      .catch(console.error)
      .finally(() => { setLoading(false); });
  };

  useEffect(() => {
    fetchDatasets();
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

  const handleUploadComplete = (newId: string, newName: string) => {
    setShowUpload(false);
    fetchDatasets(); // Refresh list
    onChange({
      ...config,
      datasetId: newId,
      datasetName: newName
    });
  };

  if (showUpload) {
    return <FileUpload onUploadComplete={handleUploadComplete} onCancel={() => { setShowUpload(false); }} />;
  }

  return (
    <div className="p-4 space-y-4">
      <div className="space-y-2">
        <div className="flex justify-between items-center">
          <label className="block text-sm font-medium">Select Dataset</label>
          <button 
            onClick={() => { setShowUpload(true); }}
            className="text-xs flex items-center gap-1 text-blue-600 hover:text-blue-700 font-medium"
          >
            <Plus size={14} />
            New Upload
          </button>
        </div>
        
        {loading ? (
          <div className="text-xs text-muted-foreground">Loading datasets...</div>
        ) : (
          <select
            className="w-full p-2 border rounded bg-background focus:ring-1 focus:ring-primary outline-none"
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

      {/* Schema Preview */}
      {config.datasetId && (
        <div className="space-y-2 pt-2 border-t">
          <div className="flex items-center gap-2 text-sm font-medium text-muted-foreground">
            <TableProperties size={14} />
            <span>Dataset Schema</span>
          </div>
          
          {isSchemaLoading ? (
            <div className="text-xs text-muted-foreground animate-pulse">Loading schema...</div>
          ) : schema ? (
            <>
              <div className="border rounded-md bg-muted/10 max-h-60 overflow-auto">
                <table className="w-full text-xs text-left min-w-max">
                  <thead className="bg-muted/20 sticky top-0 z-10">
                    <tr>
                      <th className="p-2 font-medium border-b bg-muted/20">Column</th>
                      <th className="p-2 font-medium border-b bg-muted/20">Type</th>
                      <th className="p-2 font-medium border-b text-right bg-muted/20">Missing</th>
                      <th className="p-2 font-medium border-b text-right bg-muted/20">Unique</th>
                      <th className="p-2 font-medium border-b text-right bg-muted/20">Min</th>
                      <th className="p-2 font-medium border-b text-right bg-muted/20">Max</th>
                      <th className="p-2 font-medium border-b text-right bg-muted/20">Mean</th>
                    </tr>
                  </thead>
                  <tbody>
                    {Object.values(schema.columns).map((col) => (
                      <tr key={col.name} className="border-b last:border-0 hover:bg-muted/20">
                        <td className="p-2 font-mono font-medium whitespace-nowrap">{col.name}</td>
                        <td className="p-2 text-muted-foreground whitespace-nowrap">{col.dtype}</td>
                        <td className="p-2 text-right text-muted-foreground">{col.missing_count}</td>
                        <td className="p-2 text-right text-muted-foreground">{col.unique_count}</td>
                        <td className="p-2 text-right text-muted-foreground">{col.min_value !== undefined ? Number(col.min_value).toFixed(2) : '-'}</td>
                        <td className="p-2 text-right text-muted-foreground">{col.max_value !== undefined ? Number(col.max_value).toFixed(2) : '-'}</td>
                        <td className="p-2 text-right text-muted-foreground">{col.mean_value !== undefined ? Number(col.mean_value).toFixed(2) : '-'}</td>
                      </tr>
                    ))}
                  </tbody>
                </table>
              </div>
              <p className="text-[10px] text-muted-foreground mt-1 text-center italic">
                Expand node to see full details
              </p>
            </>
          ) : (
            <div className="text-xs text-muted-foreground italic">No schema available.</div>
          )}
        </div>
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
        .then(d => { setFetchedName(d.name); })
        .catch(() => { setFetchedName(data.datasetId); }); // Fallback to ID on error
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
