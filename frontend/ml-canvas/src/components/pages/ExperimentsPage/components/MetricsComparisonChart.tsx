import React from 'react';
import {
  BarChart,
  Bar,
  XAxis,
  YAxis,
  CartesianGrid,
  Tooltip,
  Legend,
  ResponsiveContainer,
} from 'recharts';
import { InfoTooltip } from '../../../ui/InfoTooltip';
import { getMetricDescription } from '../../../../core/utils/format';

interface Props {
  metricsData: Record<string, unknown>[];
  metricGroups: Map<string, string[]>;
  availableMetrics: string[];
  activeMetric: string | null;
  setSelectedMetric: (m: string) => void;
  showTrainMetrics: boolean;
  setShowTrainMetrics: (v: boolean) => void;
  showTestMetrics: boolean;
  setShowTestMetrics: (v: boolean) => void;
  showValMetrics: boolean;
  setShowValMetrics: (v: boolean) => void;
  showCvMetrics: boolean;
  setShowCvMetrics: (v: boolean) => void;
}

const parseMetricKey = (key: string) => {
  if (key === 'best_score') return { type: 'val', base: 'best_score' };
  if (key.startsWith('train_')) return { type: 'train', base: key.replace('train_', '') };
  if (key.startsWith('test_')) return { type: 'test', base: key.replace('test_', '') };
  if (key.startsWith('val_')) return { type: 'val', base: key.replace('val_', '') };
  return { type: 'other', base: key };
};

export const MetricsComparisonChart: React.FC<Props> = ({
  metricsData,
  metricGroups,
  availableMetrics,
  activeMetric,
  setSelectedMetric,
  showTrainMetrics,
  setShowTrainMetrics,
  showTestMetrics,
  setShowTestMetrics,
  showValMetrics,
  setShowValMetrics,
  showCvMetrics,
  setShowCvMetrics,
}) => {
  return (
    <div className="space-y-6">
      <div className="flex justify-between items-center">
        <h3 className="text-lg font-medium text-gray-800 dark:text-gray-100">Metrics Comparison</h3>
        <div className="flex gap-4 text-sm">
          <label className="flex items-center gap-2 cursor-pointer">
            <input
              type="checkbox"
              checked={showTrainMetrics}
              onChange={e => { setShowTrainMetrics(e.target.checked); }}
              className="rounded border-gray-300 text-blue-600 focus:ring-blue-500"
            />
            <span className="text-gray-700 dark:text-gray-300">Train</span>
          </label>
          <label className="flex items-center gap-2 cursor-pointer">
            <input
              type="checkbox"
              checked={showTestMetrics}
              onChange={e => { setShowTestMetrics(e.target.checked); }}
              className="rounded border-gray-300 text-blue-600 focus:ring-blue-500"
            />
            <span className="text-gray-700 dark:text-gray-300">Test</span>
          </label>
          <label className="flex items-center gap-2 cursor-pointer">
            <input
              type="checkbox"
              checked={showValMetrics}
              onChange={e => { setShowValMetrics(e.target.checked); }}
              className="rounded border-gray-300 text-blue-600 focus:ring-blue-500"
            />
            <span className="text-gray-700 dark:text-gray-300">Validation</span>
          </label>
          <label className="flex items-center gap-2 cursor-pointer">
            <input
              type="checkbox"
              checked={showCvMetrics}
              onChange={e => { setShowCvMetrics(e.target.checked); }}
              className="rounded border-gray-300 text-purple-600 focus:ring-purple-500"
            />
            <span className="text-gray-700 dark:text-gray-300">Cross-Validation</span>
          </label>
        </div>
      </div>

      {/* Metric Selector Tabs */}
      {availableMetrics.length > 0 && (
        <div className="flex flex-wrap gap-2 border-b border-gray-200 dark:border-gray-700 pb-2">
          {availableMetrics.map(metric => (
            <button
              key={metric}
              onClick={() => { setSelectedMetric(metric); }}
              className={`px-3 py-1.5 text-xs font-medium rounded-md transition-colors flex items-center gap-1 ${
                activeMetric === metric
                  ? 'bg-blue-100 text-blue-700 dark:bg-blue-900/30 dark:text-blue-400'
                  : 'text-gray-600 dark:text-gray-400 hover:bg-gray-100 dark:hover:bg-gray-800'
              }`}
            >
              {metric.toUpperCase()}
              {getMetricDescription(metric) && <InfoTooltip size="sm" text={getMetricDescription(metric)!} />}
            </button>
          ))}
        </div>
      )}

      {activeMetric && metricGroups.has(activeMetric) && (
        <div className="bg-white dark:bg-gray-800 p-6 rounded-lg shadow-sm border border-gray-200 dark:border-gray-700">
          <div className="h-[300px] w-full">
            <ResponsiveContainer width="100%" height="100%">
              <BarChart data={metricsData} margin={{ top: 5, right: 30, left: 30, bottom: 40 }}>
                <CartesianGrid strokeDasharray="3 3" stroke="#374151" opacity={0.1} />
                <XAxis dataKey="name" stroke="#6B7280" fontSize={12} />
                <YAxis stroke="#6B7280" fontSize={12} />
                <Tooltip
                  shared={false}
                  cursor={{ fill: 'rgba(107, 114, 128, 0.1)' }}
                  contentStyle={{ backgroundColor: '#1F2937', borderColor: '#374151', color: '#F3F4F6' }}
                  itemStyle={{ color: '#F3F4F6' }}
                />
                <Legend />
                {metricGroups.get(activeMetric)?.map((key) => {
                  const { type } = parseMetricKey(key);
                  // Colors: Train=Blue, Test=Green, Val=Orange, Other=Purple
                  let color = '#8884d8';
                  if (type === 'train') color = '#3b82f6';
                  if (type === 'test') color = '#22c55e';
                  if (type === 'val') color = '#f97316';
                  if (type === 'other') color = '#a855f7';
                  return <Bar key={key} dataKey={key} fill={color} name={`${type} (${activeMetric})`} />;
                })}
              </BarChart>
            </ResponsiveContainer>
          </div>
        </div>
      )}
    </div>
  );
};
