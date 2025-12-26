import React from 'react';
import { NodeDefinition } from '../../../core/types/nodes';
import { Split } from 'lucide-react';
import { useUpstreamData } from '../../../core/hooks/useUpstreamData';
import { useDatasetSchema } from '../../../core/hooks/useDatasetSchema';

interface TrainTestSplitConfig {
  test_size: number;
  validation_size?: number;
  random_state: number;
  stratify: boolean;
  shuffle: boolean;
  target_column?: string;
  datasetId?: string;
}

const TrainTestSplitSettings: React.FC<{ config: TrainTestSplitConfig; onChange: (c: TrainTestSplitConfig) => void; nodeId?: string }> = ({
  config,
  onChange,
  nodeId,
}) => {
  const upstreamData = useUpstreamData(nodeId || '');
  
  // Find datasetId from upstream data (injected by useUpstreamData hook)
  const upstreamDatasetId = upstreamData.find(d => d.datasetId)?.datasetId as string | undefined;
  
  // Check if an upstream node (like FeatureTargetSplitter) has already defined a target column
  const upstreamTargetColumn = upstreamData.find((d: Record<string, unknown>) => d.target_column)?.target_column as string | undefined;

  React.useEffect(() => {
    const updates: Partial<TrainTestSplitConfig> = {};
    
    // Propagate datasetId
    if (upstreamDatasetId && config.datasetId !== upstreamDatasetId) {
      updates.datasetId = upstreamDatasetId;
    }
    
    // Propagate target_column if upstream has it and we don't (or if we want to enforce it)
    if (upstreamTargetColumn && config.target_column !== upstreamTargetColumn) {
      updates.target_column = upstreamTargetColumn;
    }

    if (Object.keys(updates).length > 0) {
      onChange({ ...config, ...updates });
    }
  }, [upstreamDatasetId, upstreamTargetColumn, config.datasetId, config.target_column, onChange]);

  const { data: schema, isLoading } = useDatasetSchema(upstreamDatasetId || config.datasetId);
  const columns = schema ? Object.values(schema.columns).map(c => c.name) : [];

  const valSize = config.validation_size || 0;
  const trainSize = 1 - config.test_size - valSize;

  // Responsive Layout Logic
  const containerRef = React.useRef<HTMLDivElement>(null);
  const [isWide, setIsWide] = React.useState(false);

  React.useEffect(() => {
    if (!containerRef.current) return;
    const observer = new ResizeObserver((entries) => {
      for (const entry of entries) {
        setIsWide(entry.contentRect.width > 450); // Switch to 2-column layout if wider than 450px
      }
    });
    observer.observe(containerRef.current);
    return () => { observer.disconnect(); };
  }, []);

  return (
    <div ref={containerRef} className={`flex flex-col h-full w-full bg-background ${isWide ? 'overflow-hidden' : 'overflow-y-auto'}`}>
      {/* Top Status Bar (Always Visible) */}
      <div className="shrink-0 p-4 pb-0 space-y-2">
        {!upstreamDatasetId && !config.datasetId && (
          <div className="p-2 bg-yellow-50 text-yellow-800 text-xs rounded border border-yellow-200">
            Connect a dataset node to see available columns.
          </div>
        )}
        
        {isLoading && (
          <div className="text-xs text-muted-foreground animate-pulse">
            Loading schema...
          </div>
        )}
      </div>

      {/* Main Content Area - Responsive Grid/Flex */}
      <div className={`flex-1 min-h-0 p-4 gap-4 ${isWide ? 'grid grid-cols-2' : 'flex flex-col'}`}>
        
        {/* Left Column: Split Ratios & Random State */}
        <div className={`space-y-4 ${isWide ? 'overflow-y-auto pr-2' : 'shrink-0'}`}>
          <div className="space-y-2">
            <label className="text-sm font-medium">Test Size (0.0 - 1.0)</label>
            <input
              type="number"
              step="0.05"
              min="0.05"
              max="0.95"
              className="w-full p-2 border rounded bg-background text-sm"
              value={config.test_size}
              onChange={(e) => onChange({ ...config, test_size: parseFloat(e.target.value) })}
            />
          </div>

          <div className="space-y-2">
            <label className="text-sm font-medium">Validation Size (0.0 - 1.0)</label>
            <input
              type="number"
              step="0.05"
              min="0.00"
              max="0.95"
              className="w-full p-2 border rounded bg-background text-sm"
              value={valSize}
              onChange={(e) => onChange({ ...config, validation_size: parseFloat(e.target.value) })}
            />
            <p className="text-xs text-muted-foreground">
              {Math.round(trainSize * 100)}% Training, {Math.round(valSize * 100)}% Validation, {Math.round(config.test_size * 100)}% Testing
            </p>
            {trainSize <= 0 && (
              <p className="text-xs text-red-500 font-medium">
                Error: Total split size exceeds 100%
              </p>
            )}
          </div>

          <div className="space-y-2">
            <label className="text-sm font-medium">Random State</label>
            <input
              type="number"
              className="w-full p-2 border rounded bg-background text-sm"
              value={config.random_state}
              onChange={(e) => onChange({ ...config, random_state: parseInt(e.target.value) })}
            />
          </div>
        </div>

        {/* Right Column: Options & Stratification */}
        <div className={`space-y-4 ${isWide ? 'overflow-y-auto pl-2 border-l' : 'shrink-0 pt-4 border-t'}`}>
          <div className="flex items-center gap-2">
            <input
              type="checkbox"
              id="shuffle"
              checked={config.shuffle}
              onChange={(e) => onChange({ ...config, shuffle: e.target.checked })}
            />
            <label htmlFor="shuffle" className="text-sm font-medium">
              Shuffle Data
            </label>
          </div>

          <div className="flex items-center gap-2">
            <input
              type="checkbox"
              id="stratify"
              checked={config.stratify}
              onChange={(e) => onChange({ ...config, stratify: e.target.checked })}
            />
            <label htmlFor="stratify" className="text-sm font-medium">
              Stratify by Target
            </label>
          </div>

          <div className="space-y-2 pl-6 border-l-2 border-muted">
            <label className="text-sm font-medium">Target Column</label>
            <select
              className="w-full p-2 border rounded bg-background text-sm"
              value={config.target_column ?? ''}
              onChange={(e) => onChange({ ...config, target_column: e.target.value })}
              disabled={!config.stratify}
            >
              <option value="">-- Select Target --</option>
              {columns.map(col => (
                <option key={col} value={col}>{col}</option>
              ))}
            </select>
            <p className="text-xs text-muted-foreground">
              Required for stratification. If input is already separated (X, y), this is ignored.
            </p>
          </div>
        </div>
      </div>
    </div>
  );
};

export const TrainTestSplitNode: NodeDefinition<TrainTestSplitConfig> = {
  type: 'TrainTestSplitter', // Matches backend registry ID
  label: 'Train-Test Split',
  category: 'Preprocessing',
  description: 'Split data into training and testing sets.',
  icon: Split,
  inputs: [{ id: 'in', label: 'Data', type: 'dataset' }],
  outputs: [
    { id: 'train', label: 'Train', type: 'dataset' },
    { id: 'validation', label: 'Validation', type: 'dataset' },
    { id: 'test', label: 'Test', type: 'dataset' }
  ],
  settings: TrainTestSplitSettings,
  validate: (config) => {
    if (config.test_size <= 0 || config.test_size >= 1) {
      return { isValid: false, message: 'Test size must be between 0 and 1.' };
    }
    const valSize = config.validation_size || 0;
    if (valSize < 0 || valSize >= 1) {
      return { isValid: false, message: 'Validation size must be between 0 and 1.' };
    }
    if (config.test_size + valSize >= 1) {
      return { isValid: false, message: 'Sum of Test and Validation sizes must be less than 1.' };
    }
    return { isValid: true };
  },
  getDefaultConfig: () => ({
    test_size: 0.2,
    validation_size: 0.0,
    random_state: 42,
    stratify: false,
    shuffle: true,
  }),
};
