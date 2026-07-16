import React, { useMemo, useState } from 'react';
import { Loader2, Check, Download } from 'lucide-react';
import {
  BarChart,
  Bar,
  XAxis,
  YAxis,
  CartesianGrid,
  Tooltip,
  ResponsiveContainer,
  ReferenceLine,
  Cell,
} from 'recharts';
import { InfoTooltip } from '../../../ui/InfoTooltip';
import { useChartTheme } from '../../../../core/hooks/useChartTheme';
import type { ShapExplanationData } from '../types';

interface Props {
  jobId: string;
  modelType: string;
  shapExplanation: ShapExplanationData;
  handleDownload: (elementId: string, fileName: string) => void | Promise<void>;
  downloadingChart: string | null;
  doneChart: string | null;
}

const MAX_FEATURE_ROWS = 10;
const POSITIVE_COLOR = '#ef4444';
const NEGATIVE_COLOR = '#3b82f6';

interface WaterfallRow {
  label: string;
  pad: number;
  size: number;
  delta: number;
  featureValue?: number;
}

export const ShapWaterfallView: React.FC<Props> = ({
  jobId,
  modelType,
  shapExplanation,
  handleDownload,
  downloadingChart,
  doneChart,
}) => {
  const chartId = `shap-waterfall-chart-${jobId}`;
  const chartTheme = useChartTheme();
  const { samples } = shapExplanation;

  const [selectedIndex, setSelectedIndex] = useState(0);
  const activeIndex = selectedIndex < samples.length ? selectedIndex : 0;
  const sample = samples[activeIndex];

  const { rows, baseValue, outputValue } = useMemo(() => {
    if (!sample) return { rows: [] as WaterfallRow[], baseValue: 0, outputValue: 0 };

    const entries = Object.entries(sample.shap_values).sort(
      (a, b) => Math.abs(b[1]) - Math.abs(a[1])
    );
    const top = entries.slice(0, MAX_FEATURE_ROWS);
    const rest = entries.slice(MAX_FEATURE_ROWS);
    const restSum = rest.reduce((sum, [, v]) => sum + v, 0);

    let running = sample.base_value;
    const rows: WaterfallRow[] = top.map(([feature, delta]) => {
      const start = running;
      running += delta;
      return {
        label: feature,
        pad: Math.min(start, running),
        size: Math.abs(delta),
        delta,
        featureValue: sample.feature_values[feature] ?? 0,
      };
    });

    if (rest.length > 0) {
      const start = running;
      running += restSum;
      rows.push({
        label: `Other features (${rest.length})`,
        pad: Math.min(start, running),
        size: Math.abs(restSum),
        delta: restSum,
      });
    }

    return { rows, baseValue: sample.base_value, outputValue: running };
  }, [sample]);

  if (!sample) return null;

  return (
    <div className="space-y-6">
      <div className="flex items-center gap-2 flex-wrap">
        <h3 className="text-lg font-medium text-gray-800 dark:text-gray-100">
          SHAP Waterfall — {modelType !== 'unknown' ? modelType : jobId.slice(0, 8)}
        </h3>
        <InfoTooltip
          text="Breaks down one sampled prediction: starting from the model's base value (average output), each feature pushes the prediction up (red) or down (blue) until reaching the final predicted output for that row."
          align="center"
        />
        <select
          className="ml-auto bg-gray-50 dark:bg-gray-900 border border-gray-300 dark:border-gray-600 text-gray-900 dark:text-gray-100 text-sm rounded-lg p-2"
          value={activeIndex}
          onChange={(e) => { setSelectedIndex(Number(e.target.value)); }}
        >
          {samples.map((s, i) => {
            const output = s.base_value + Object.values(s.shap_values).reduce((a, b) => a + b, 0);
            return (
              <option key={i} value={i}>Row {i + 1} (output ≈ {output.toFixed(3)})</option>
            );
          })}
        </select>
      </div>
      <div className="bg-white dark:bg-gray-800 rounded-lg border border-gray-200 dark:border-gray-700 p-6 relative group" id={chartId}>
        <div className="absolute top-4 right-4 z-10 opacity-0 group-hover:opacity-100 transition-opacity" data-export-ignore="true">
          <button
            onClick={() => void handleDownload(chartId, `shap_waterfall_row${activeIndex + 1}_${jobId.slice(0, 8)}`)}
            disabled={downloadingChart === chartId}
            className="p-1.5 bg-white dark:bg-gray-800 border border-gray-200 dark:border-gray-600 rounded shadow-sm text-gray-500 hover:text-blue-600 disabled:opacity-50"
            title="Download Graph"
          >
            {downloadingChart === chartId ? <Loader2 className="w-3.5 h-3.5 animate-spin" /> : doneChart === chartId ? <Check className="w-3.5 h-3.5 text-green-500" /> : <Download className="w-3.5 h-3.5" />}
          </button>
        </div>
        <div className="h-[500px]">
          <ResponsiveContainer width="100%" height="100%">
            <BarChart data={rows} layout="vertical" margin={{ top: 5, right: 30, bottom: 5, left: 140 }}>
              <CartesianGrid strokeDasharray="3 3" opacity={0.1} stroke={chartTheme.gridColor} />
              <XAxis type="number" tick={{ fontSize: 12, fill: chartTheme.axisColor }} />
              <YAxis type="category" dataKey="label" tick={{ fontSize: 11, fill: chartTheme.axisColor }} width={130} />
              <ReferenceLine x={baseValue} stroke="#94a3b8" strokeDasharray="3 3" label={{ value: 'Base', position: 'top', fontSize: 11, fill: chartTheme.axisColor }} />
              <ReferenceLine x={outputValue} stroke="#6366f1" strokeDasharray="3 3" label={{ value: 'Prediction', position: 'top', fontSize: 11, fill: chartTheme.axisColor }} />
              <Tooltip
                contentStyle={chartTheme.tooltipContentStyle}
                formatter={(_value: number, _name: string, entry) => {
                  const row = entry?.payload as WaterfallRow | undefined;
                  if (!row) return ['', ''];
                  const parts = [row.delta.toFixed(3), 'SHAP contribution'];
                  return parts;
                }}
                labelFormatter={(label: string, entry) => {
                  const row = entry?.[0]?.payload as WaterfallRow | undefined;
                  if (row?.featureValue !== undefined) return `${label} (value: ${row.featureValue.toFixed(3)})`;
                  return label;
                }}
              />
              <Bar dataKey="pad" stackId="waterfall" fill="transparent" isAnimationActive={false} />
              <Bar dataKey="size" stackId="waterfall" isAnimationActive={false} radius={[0, 4, 4, 0]}>
                {rows.map((row, i) => (
                  <Cell key={i} fill={row.delta >= 0 ? POSITIVE_COLOR : NEGATIVE_COLOR} />
                ))}
              </Bar>
            </BarChart>
          </ResponsiveContainer>
        </div>
        <p className="text-[11px] text-gray-400 dark:text-gray-500 mt-2 text-center">
          Base value {baseValue.toFixed(3)} → prediction {outputValue.toFixed(3)} ·
          <span className="text-red-500"> red</span> increases, <span className="text-blue-500">blue</span> decreases
        </p>
      </div>
    </div>
  );
};
