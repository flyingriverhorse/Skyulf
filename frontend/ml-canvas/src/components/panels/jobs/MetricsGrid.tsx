import React, { useState } from 'react';
import { getMetricDescription } from '../../../core/utils/format';
import { splitOfMetric } from '../../../core/utils/metricMeta';
import { InfoTooltip } from '../../ui/InfoTooltip';

interface Props {
  metrics: Record<string, unknown>;
  /** Result fields that live alongside metrics but aren't metrics (best_params, trials…). */
  excludeKeys?: string[];
}

const formatMetricValue = (key: string, value: number): string => {
  if (key.endsWith('_std')) return value.toFixed(6);
  return value.toFixed(4);
};

/**
 * Tile grid of per-metric values with a toggle for CV-population metrics
 * (`cv_*` and `best_score`), which can dominate the grid for tuned jobs and
 * crowd out the train/test numbers users actually compare.
 */
export const MetricsGrid: React.FC<Props> = ({ metrics, excludeKeys = [] }) => {
  const [showCv, setShowCv] = useState(true);

  const entries = Object.entries(metrics)
    .filter(([k, v]) => !excludeKeys.includes(k) && (typeof v === 'number' || typeof v === 'string'));
  const cvCount = entries.filter(([k]) => splitOfMetric(k) === 'cv').length;
  const visible = showCv ? entries : entries.filter(([k]) => splitOfMetric(k) !== 'cv');

  return (
    <div className="space-y-2">
      {cvCount > 0 && (
        <label className="flex items-center gap-2 cursor-pointer w-fit text-xs text-gray-600 dark:text-gray-400">
          <input
            type="checkbox"
            checked={showCv}
            onChange={e => { setShowCv(e.target.checked); }}
            className="rounded border-gray-300 text-purple-600 focus:ring-purple-500 dark:border-gray-600 dark:bg-gray-700"
          />
          <span>Show CV metrics ({cvCount})</span>
        </label>
      )}
      <div className="grid grid-cols-2 sm:grid-cols-4 gap-3">
        {visible.map(([k, v]) => (
          <div key={k} className={`p-3 border rounded-lg ${splitOfMetric(k) === 'cv' ? 'bg-purple-50 dark:bg-purple-900/20 border-purple-200 dark:border-purple-800' : 'bg-white dark:bg-gray-800 border-gray-200 dark:border-gray-700'}`}>
            <div className="text-xs text-gray-500 dark:text-gray-400 mb-1 capitalize flex items-center gap-1">
              {k.replace(/_/g, ' ')}
              {getMetricDescription(k) && <InfoTooltip size="sm" text={getMetricDescription(k)!} />}
            </div>
            <div className={`font-mono font-medium ${splitOfMetric(k) === 'cv' ? 'text-purple-600 dark:text-purple-400' : 'text-blue-600 dark:text-blue-400'}`}>
              {typeof v === 'number' ? formatMetricValue(k, v) : String(v)}
            </div>
          </div>
        ))}
      </div>
    </div>
  );
};
