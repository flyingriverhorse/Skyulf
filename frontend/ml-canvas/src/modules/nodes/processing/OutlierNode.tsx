import { validateOutlier } from './outlier/validation';
import { OutlierRecommendations } from './outlier/OutlierRecommendations';
import { OutlierFeedback } from './outlier/OutlierFeedback';
import type { OutlierSettingsProps } from './outlier/types';
import { OutlierControls } from './outlier/OutlierControls';
import { useOutlierData } from './outlier/useOutlierData';
import { ValidationField } from '../../../components/shared/ValidationField';
import React from 'react';
import { NodeDefinition } from '../../../core/types/nodes';
import { Scissors } from 'lucide-react';
import { ColumnMultiSelect } from '../shared/ColumnMultiSelect';
import { useIsWideContainer } from '../../../core/hooks/useIsWideContainer';


const OutlierSettings: React.FC<OutlierSettingsProps> = ({
  config,
  onChange,
  nodeId,
}) => {
  const { datasetId, isLoading, numericColumns, metrics, nodeResult, backendRecommendations } = useOutlierData(nodeId);

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
          <OutlierControls config={config} onChange={onChange} />

          {/* Feedback Section (Wide) */}
          {isWide && (
            <>
              {<OutlierFeedback config={config} metrics={metrics} />}
              {<OutlierRecommendations config={config} metrics={metrics} hasNodeResult={Boolean(nodeResult)} backendRecommendations={backendRecommendations} />}
            </>
          )}
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
        {!isWide && (
          <>
            {<OutlierFeedback config={config} metrics={metrics} />}
            {<OutlierRecommendations config={config} metrics={metrics} hasNodeResult={Boolean(nodeResult)} backendRecommendations={backendRecommendations} />}
          </>
        )}

      </div>
    </div>
  );
};

export const OutlierNode: NodeDefinition = {
  type: 'outlier',
  label: 'Outlier Removal',
  category: 'Preprocessing',
  description: 'Detect and remove or clip outliers.',
  icon: Scissors,
  inputs: [{ id: 'in', type: 'dataset', label: 'Dataset' }],
  outputs: [{ id: 'out', type: 'dataset', label: 'Cleaned' }],
  settings: OutlierSettings,
  bodyPreview: (config) => {
    const cols = config.columns?.length ?? 0;
    const method = (config.method ?? 'iqr').toUpperCase();
    if (cols === 0) return method;
    return `${method} · ${cols} ${cols === 1 ? 'col' : 'cols'}`;
  },
  validate: validateOutlier,
  getDefaultConfig: () => ({
    method: 'iqr',
    columns: [],
    multiplier: 1.5,
    threshold: 3.0,
    lower_percentile: 5.0,
    upper_percentile: 95.0,
    contamination: 0.01
  })
};
