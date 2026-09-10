import { validateScaling } from './scaling/validation';
import { ScalingFeedback } from './scaling/ScalingFeedback';
import type { ScalingConfig, ScalingSettingsProps } from './scaling/types';
import { ScalingControls } from './scaling/ScalingControls';
import { useScalingData } from './scaling/useScalingData';
import { ValidationField } from '../../../components/shared/ValidationField';
import React from 'react';
import { NodeDefinition } from '../../../core/types/nodes';
import { Scaling } from 'lucide-react';
import { ColumnMultiSelect } from '../shared/ColumnMultiSelect';
import { useIsWideContainer } from '../../../core/hooks/useIsWideContainer';


const ScalingSettings: React.FC<ScalingSettingsProps> = ({
  config,
  onChange,
  nodeId,
}) => {
  const { datasetId, isLoading, numericColumns, metrics } = useScalingData(nodeId);

  // Responsive layout: switch to a 2-column layout once the panel is wider than 450px.
  const [containerRef, isWide] = useIsWideContainer();


  return (
    <div ref={containerRef} className={`flex flex-col h-full w-full bg-background ${isWide ? 'overflow-hidden' : 'overflow-y-auto'}`}>
      {/* Top Status Bar */}
      <div className="shrink-0 p-4 pb-0 space-y-2">
        {!datasetId && (
          <div className="p-2 bg-yellow-50 dark:bg-yellow-900/20 text-yellow-800 dark:text-yellow-400 text-xs rounded border border-yellow-200 dark:border-yellow-800">
            Connect a dataset node to see available columns.
          </div>
        )}
      </div>

      {/* Main Content */}
      <div className={`flex-1 min-h-0 p-4 gap-4 ${isWide ? 'grid grid-cols-2' : 'flex flex-col'}`}>

        {/* Left Column: Settings */}
        <div className={`space-y-4 ${isWide ? 'overflow-y-auto pr-2' : 'shrink-0'}`}>
          <ScalingControls config={config} onChange={onChange} />

          {/* Feedback Section (Wide) */}
          {isWide && <ScalingFeedback config={config} metrics={metrics} />}
        </div>

        {/* Right Column: Columns */}
        <div className={`flex flex-col overflow-hidden ${isWide ? 'min-h-0 flex-1' : 'shrink-0'}`}>
          <ValidationField field="columns" className={isWide ? "flex min-h-0 flex-1 flex-col" : ""}>
            <ColumnMultiSelect
              columns={numericColumns}
              selected={config.columns}
              onChange={(newCols) => { onChange({ ...config, columns: newCols }); }}
              label="Numeric Columns"
              variant="panel"
              isLoading={isLoading}
              emptyMessage="No numeric columns found"
              fillHeight={isWide}
            />
          </ValidationField>
        </div>

        {/* Feedback Section (Narrow) */}
        {!isWide && <ScalingFeedback config={config} metrics={metrics} />}

      </div>
    </div>
  );
};

export const ScalingNode: NodeDefinition<ScalingConfig> = {
  type: 'scale_numeric_features',
  label: 'Scaling',
  category: 'Preprocessing',
  description: 'Scale numeric features to a standard range.',
  icon: Scaling, // Note: You might need to import a real icon or use a placeholder if 'Scaling' doesn't exist in lucide-react
  inputs: [{ id: 'in', label: 'Data', type: 'dataset' }],
  outputs: [{ id: 'out', label: 'Scaled Data', type: 'dataset' }],
  settings: ScalingSettings,
  bodyPreview: (config) => {
    const cols = config.columns?.length ?? 0;
    const method = config.method ?? 'standard';
    if (cols === 0) return method;
    return `${method} · ${cols} ${cols === 1 ? 'col' : 'cols'}`;
  },
  validate: validateScaling,
  getDefaultConfig: () => ({
    columns: [],
    method: 'standard',
  }),
};
