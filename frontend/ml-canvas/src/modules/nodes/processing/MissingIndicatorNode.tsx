import React, { useMemo } from 'react';
import { NodeDefinition } from '../../../core/types/nodes';
import { Flag, Activity } from 'lucide-react';
import { useUpstreamData } from '../../../core/hooks/useUpstreamData';
import { useDatasetSchema } from '../../../core/hooks/useDatasetSchema';
import { useUpstreamDroppedColumns } from '../../../core/hooks/useUpstreamDroppedColumns';
import { useGraphStore } from '../../../core/store/useGraphStore';
import { ColumnMultiSelect } from '../shared/ColumnMultiSelect';
import { useIsWideContainer } from '../../../core/hooks/useIsWideContainer';

interface MissingIndicatorConfig {
  columns: string[];
  flag_suffix: string;
}

const MissingIndicatorSettings: React.FC<{ config: MissingIndicatorConfig; onChange: (c: MissingIndicatorConfig) => void; nodeId?: string }> = ({
  config,
  onChange,
  nodeId,
}) => {
  const upstreamData = useUpstreamData(nodeId || '');
  const datasetId = upstreamData.find(d => d.datasetId)?.datasetId as string | undefined;
  const { data: schema, isLoading } = useDatasetSchema(datasetId);
  const droppedUpstream = useUpstreamDroppedColumns(nodeId);
  const availableColumns = useMemo(
    () => schema ? Object.values(schema.columns).map(c => c.name).filter(n => !droppedUpstream.has(n)) : [],
    [schema, droppedUpstream]
  );

  const executionResult = useGraphStore((state) => state.executionResult);
  const nodeResult = nodeId ? executionResult?.node_results[nodeId] : null;
  const metrics = (nodeResult?.metrics && typeof nodeResult.metrics === 'object')
    ? (nodeResult.metrics as Record<string, unknown>)
    : null;

  // Responsive layout: switch to a 2-column layout once the panel is wider than 450px.
  const [containerRef, isWide] = useIsWideContainer();

  const indicatorsCreated = metrics?.missing_indicators_created;
  const indicatorColumns = Array.isArray(metrics?.missing_indicators_columns)
    ? (metrics?.missing_indicators_columns as unknown[])
    : [];

  return (
    <div ref={containerRef} className={`flex flex-col h-full w-full bg-background ${isWide ? 'overflow-hidden' : 'overflow-y-auto'}`}>
      {/* Top Status Bar (Always Visible) */}
      <div className="shrink-0 p-4 pb-0 space-y-2">
        {!datasetId && (
          <div className="p-2 bg-yellow-50 text-yellow-800 text-xs rounded border border-yellow-200">
            Connect a dataset node to see available columns.
          </div>
        )}

        {isLoading && !!datasetId && (
          <div className="text-xs text-muted-foreground animate-pulse">
            Loading schema...
          </div>
        )}
      </div>

      {/* Main Content Area - Responsive Grid/Flex */}
      <div className={`flex-1 min-h-0 p-4 gap-4 ${isWide ? 'grid grid-cols-2' : 'flex flex-col'}`}>

        {/* Left Column (Settings) */}
        <div className={`space-y-4 ${isWide ? 'overflow-y-auto pr-2' : 'shrink-0'}`}>
          <div>
            <span className="block text-sm font-medium mb-1">Indicator Suffix</span>
            <input
              type="text"
              className="w-full p-2 border rounded bg-background focus:ring-1 focus:ring-primary outline-none"
              value={config.flag_suffix}
              onChange={(e) => onChange({ ...config, flag_suffix: e.target.value })}
              placeholder="_was_missing"
            />
            <p className="text-xs text-muted-foreground mt-1">
              Suffix added to the new indicator columns (e.g., &quot;Age_was_missing&quot;).
            </p>
          </div>

          {/* Feedback Section */}
          {indicatorsCreated !== undefined && (
            <div className="p-3 bg-muted/30 rounded-md border border-border">
              <div className="flex items-center gap-2 mb-2 text-sm font-semibold text-primary">
                <Activity size={14} />
                <span>Last Run Results</span>
              </div>
              <div className="space-y-1 text-xs">
                <div className="flex justify-between">
                  <span className="text-muted-foreground">Indicators Created:</span>
                  <span className="font-medium text-primary">{String(indicatorsCreated)}</span>
                </div>
                {indicatorColumns.length > 0 && (
                  <div className="pt-1 border-t mt-1">
                    <span className="text-muted-foreground block mb-1">New Columns:</span>
                    <div className="flex flex-wrap gap-1">
                      {indicatorColumns.map((col: unknown) => {
                        const colKey = String(col);
                        return (
                          <span key={colKey} className="px-1.5 py-0.5 bg-background border rounded text-[10px] font-mono">
                            {colKey}
                          </span>
                        );
                      })}
                    </div>
                  </div>
                )}
              </div>
            </div>
          )}
        </div>

        {/* Right Column (Column List) */}
        <div className={`flex flex-col overflow-hidden ${isWide ? 'min-h-0 flex-1' : 'shrink-0'}`}>
          <p className="text-xs text-muted-foreground mb-2">
            Create indicators for these columns. If empty, all columns with missing values are used.
          </p>
          <ColumnMultiSelect
            columns={availableColumns}
            selected={config.columns}
            onChange={(newCols) => { onChange({ ...config, columns: newCols }); }}
            label="Target Columns (Optional)"
            variant="panel"
            isLoading={isLoading}
            fillHeight={isWide}
          />
        </div>
      </div>
    </div>
  );
};

export const MissingIndicatorNode: NodeDefinition<MissingIndicatorConfig> = {
  type: 'MissingIndicator',
  label: 'Missing Indicator',
  category: 'Preprocessing',
  description: 'Create binary indicators for missing values.',
  icon: Flag,
  inputs: [{ id: 'in', label: 'Data', type: 'dataset' }],
  outputs: [{ id: 'out', label: 'Augmented Data', type: 'dataset' }],
  settings: MissingIndicatorSettings,
  bodyPreview: (config) => {
    const cols = config.columns?.length ?? 0;
    if (cols === 0) return null;
    return `${cols} ${cols === 1 ? 'col' : 'cols'} → flags`;
  },
  validate: (_config) => ({ isValid: true }),
  getDefaultConfig: () => ({
    columns: [],
    flag_suffix: '_was_missing',
  }),
};
