import React, { useMemo, useState } from 'react';
import { Loader2, Check, Download } from 'lucide-react';
import {
  ScatterChart,
  Scatter,
  XAxis,
  YAxis,
  CartesianGrid,
  Tooltip,
  ResponsiveContainer,
  ReferenceLine,
} from 'recharts';
import { InfoTooltip } from '../../../ui/InfoTooltip';
import type { ShapExplanationData } from '../types';

interface Props {
  jobId: string;
  modelType: string;
  shapExplanation: ShapExplanationData;
  handleDownload: (elementId: string, fileName: string) => void | Promise<void>;
  downloadingChart: string | null;
  doneChart: string | null;
}

export const ShapDependenceView: React.FC<Props> = ({
  jobId,
  modelType,
  shapExplanation,
  handleDownload,
  downloadingChart,
  doneChart,
}) => {
  const chartId = `shap-dependence-chart-${jobId}`;

  const rankedFeatures = useMemo(
    () => Object.entries(shapExplanation.mean_abs_importance).sort((a, b) => b[1] - a[1]).map(([name]) => name),
    [shapExplanation]
  );

  const [selectedFeature, setSelectedFeature] = useState<string>(rankedFeatures[0] ?? '');
  const activeFeature = rankedFeatures.includes(selectedFeature) ? selectedFeature : (rankedFeatures[0] ?? '');

  const points = useMemo(() => {
    if (!activeFeature) return [];
    return shapExplanation.samples.map(sample => ({
      featureValue: sample.feature_values[activeFeature] ?? 0,
      shapValue: sample.shap_values[activeFeature] ?? 0,
    }));
  }, [shapExplanation, activeFeature]);

  if (rankedFeatures.length === 0) return null;

  return (
    <div className="space-y-6">
      <div className="flex items-center gap-2 flex-wrap">
        <h3 className="text-lg font-medium text-gray-800 dark:text-gray-100">
          SHAP Dependence — {modelType !== 'unknown' ? modelType : jobId.slice(0, 8)}
        </h3>
        <InfoTooltip
          text="Shows how a single feature's value relates to its SHAP contribution across sampled rows. A rising trend means higher feature values push predictions up; a falling trend means the opposite. Non-linear or scattered shapes suggest interactions with other features."
          align="center"
        />
        <select
          className="ml-auto bg-gray-50 dark:bg-gray-900 border border-gray-300 dark:border-gray-600 text-gray-900 dark:text-gray-100 text-sm rounded-lg p-2"
          value={activeFeature}
          onChange={(e) => { setSelectedFeature(e.target.value); }}
        >
          {rankedFeatures.map(f => (
            <option key={f} value={f}>{f}</option>
          ))}
        </select>
      </div>
      <div className="bg-white dark:bg-gray-800 rounded-lg border border-gray-200 dark:border-gray-700 p-6 relative group" id={chartId}>
        <div className="absolute top-4 right-4 z-10 opacity-0 group-hover:opacity-100 transition-opacity" data-export-ignore="true">
          <button
            onClick={() => void handleDownload(chartId, `shap_dependence_${activeFeature}_${jobId.slice(0, 8)}`)}
            disabled={downloadingChart === chartId}
            className="p-1.5 bg-white dark:bg-gray-800 border border-gray-200 dark:border-gray-600 rounded shadow-sm text-gray-500 hover:text-blue-600 disabled:opacity-50"
            title="Download Graph"
          >
            {downloadingChart === chartId ? <Loader2 className="w-3.5 h-3.5 animate-spin" /> : doneChart === chartId ? <Check className="w-3.5 h-3.5 text-green-500" /> : <Download className="w-3.5 h-3.5" />}
          </button>
        </div>
        <div className="h-[500px]">
          <ResponsiveContainer width="100%" height="100%">
            <ScatterChart margin={{ top: 5, right: 30, bottom: 20, left: 20 }}>
              <CartesianGrid strokeDasharray="3 3" opacity={0.1} />
              <XAxis
                type="number"
                dataKey="featureValue"
                name={activeFeature}
                tick={{ fontSize: 12 }}
                label={{ value: activeFeature, position: 'insideBottom', offset: -10, fontSize: 12 }}
              />
              <YAxis
                type="number"
                dataKey="shapValue"
                name="SHAP value"
                tick={{ fontSize: 12 }}
                label={{ value: 'SHAP value', angle: -90, position: 'insideLeft', fontSize: 12 }}
              />
              <ReferenceLine y={0} stroke="#94a3b8" strokeDasharray="3 3" />
              <Tooltip
                contentStyle={{ backgroundColor: 'rgba(0,0,0,0.85)', border: 'none', borderRadius: '8px', color: '#fff' }}
                formatter={(value: number, name: string) => [value.toFixed(3), name]}
              />
              <Scatter data={points} fill="#8884d8" fillOpacity={0.7} />
            </ScatterChart>
          </ResponsiveContainer>
        </div>
        <p className="text-[11px] text-gray-400 dark:text-gray-500 mt-2 text-center">
          Each point is one sampled row · x = {activeFeature} value, y = its SHAP contribution
        </p>
      </div>
    </div>
  );
};
