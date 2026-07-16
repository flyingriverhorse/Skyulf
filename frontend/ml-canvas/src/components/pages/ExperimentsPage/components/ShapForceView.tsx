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
import type { TooltipContentProps } from 'recharts';
import type { ValueType, NameType } from 'recharts/types/component/DefaultTooltipContent';
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

const MAX_FEATURE_SEGMENTS = 8;
const POSITIVE_COLOR = '#ef4444';
const NEGATIVE_COLOR = '#3b82f6';

interface ForceSegment {
  key: string;
  label: string;
  value: number;
  featureValue?: number;
  stack: 'pos' | 'neg';
}

// Custom tooltip: the pad/pos/neg stacking trick means every segment shares
// the same category row, so Recharts' default tooltip lumps every bar
// (including the invisible `padPos`/`padNeg` offsets) into one payload. This
// filters those out and sorts the real segments by contribution magnitude.
const ForceTooltip: React.FC<
  TooltipContentProps<ValueType, NameType> & {
    segments: ForceSegment[];
    chartTheme: ReturnType<typeof useChartTheme>;
  }
> = ({ active, payload, segments, chartTheme }) => {
  if (!active || !payload || payload.length === 0) return null;

  const visibleKeys = new Set(payload.map(p => p.dataKey));
  const rows = segments
    .filter(seg => visibleKeys.has(seg.key))
    .sort((a, b) => Math.abs(b.value) - Math.abs(a.value));
  if (rows.length === 0) return null;

  return (
    <div style={chartTheme.tooltipContentStyle} className="px-3 py-2 text-xs">
      {rows.map(seg => (
        <div key={seg.key} style={chartTheme.tooltipItemStyle}>
          {seg.label}
          {seg.featureValue !== undefined ? ` (value: ${seg.featureValue.toFixed(3)})` : ''}: {seg.value.toFixed(3)}
        </div>
      ))}
    </div>
  );
};

// A single stacked bar row per feature, split into a "pos" stack growing
// right from base_value (red, increases the prediction) and a "neg" stack
// growing left from base_value (blue, decreases it) — the classic SHAP
// force-plot layout, compressed from the Waterfall's multi-row staircase
// into one compact row.
export const ShapForceView: React.FC<Props> = ({
  jobId,
  modelType,
  shapExplanation,
  handleDownload,
  downloadingChart,
  doneChart,
}) => {
  const chartId = `shap-force-chart-${jobId}`;
  const chartTheme = useChartTheme();
  const { samples } = shapExplanation;

  const [selectedIndex, setSelectedIndex] = useState(0);
  const activeIndex = selectedIndex < samples.length ? selectedIndex : 0;
  const sample = samples[activeIndex];

  const { segments, baseValue, outputValue, chartData } = useMemo(() => {
    if (!sample) {
      return { segments: [] as ForceSegment[], baseValue: 0, outputValue: 0, chartData: [] as Record<string, number>[] };
    }

    const entries = Object.entries(sample.shap_values).sort(
      (a, b) => Math.abs(b[1]) - Math.abs(a[1])
    );
    const top = entries.slice(0, MAX_FEATURE_SEGMENTS);
    const rest = entries.slice(MAX_FEATURE_SEGMENTS);
    const restSum = rest.reduce((sum, [, v]) => sum + v, 0);

    const segments: ForceSegment[] = top.map(([feature, delta], i) => ({
      key: `f${i}`,
      label: feature,
      value: delta,
      featureValue: sample.feature_values[feature] ?? 0,
      stack: delta >= 0 ? 'pos' : 'neg',
    }));

    if (rest.length > 0 && restSum !== 0) {
      segments.push({
        key: 'other',
        label: `Other features (${rest.length})`,
        value: restSum,
        stack: restSum >= 0 ? 'pos' : 'neg',
      });
    }

    const outputValue = sample.base_value + Object.values(sample.shap_values).reduce((a, b) => a + b, 0);

    const row: Record<string, number> = { padPos: sample.base_value, padNeg: sample.base_value };
    for (const seg of segments) row[seg.key] = seg.value;

    return { segments, baseValue: sample.base_value, outputValue, chartData: [row] };
  }, [sample]);

  if (!sample) return null;

  return (
    <div className="space-y-6">
      <div className="flex items-center gap-2 flex-wrap">
        <h3 className="text-lg font-medium text-gray-800 dark:text-gray-100">
          SHAP Force Plot — {modelType !== 'unknown' ? modelType : jobId.slice(0, 8)}
        </h3>
        <InfoTooltip
          text="Compact, single-row view of one sampled prediction: features push the prediction above the base value (red, right) or below it (blue, left) until reaching the final predicted output."
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
            onClick={() => void handleDownload(chartId, `shap_force_row${activeIndex + 1}_${jobId.slice(0, 8)}`)}
            disabled={downloadingChart === chartId}
            className="p-1.5 bg-white dark:bg-gray-800 border border-gray-200 dark:border-gray-600 rounded shadow-sm text-gray-500 hover:text-blue-600 disabled:opacity-50"
            title="Download Graph"
          >
            {downloadingChart === chartId ? <Loader2 className="w-3.5 h-3.5 animate-spin" /> : doneChart === chartId ? <Check className="w-3.5 h-3.5 text-green-500" /> : <Download className="w-3.5 h-3.5" />}
          </button>
        </div>
        <div className="h-[220px]">
          <ResponsiveContainer width="100%" height="100%">
            <BarChart data={chartData} layout="vertical" margin={{ top: 20, right: 30, bottom: 20, left: 30 }}>
              <CartesianGrid strokeDasharray="3 3" opacity={0.1} stroke={chartTheme.gridColor} />
              <XAxis type="number" tick={{ fontSize: 12, fill: chartTheme.axisColor }} />
              <YAxis type="category" dataKey={() => 'output'} tick={false} width={0} />
              <ReferenceLine x={baseValue} stroke="#94a3b8" strokeDasharray="3 3" label={{ value: 'Base', position: 'top', fontSize: 11, fill: chartTheme.axisColor }} />
              <ReferenceLine x={outputValue} stroke="#6366f1" strokeDasharray="3 3" label={{ value: 'Prediction', position: 'top', fontSize: 11, fill: chartTheme.axisColor }} />
              <Tooltip
                content={(props) => <ForceTooltip {...props} segments={segments} chartTheme={chartTheme} />}
              />
              <Bar dataKey="padPos" stackId="pos" fill="transparent" isAnimationActive={false} />
              <Bar dataKey="padNeg" stackId="neg" fill="transparent" isAnimationActive={false} />
              {segments.filter(s => s.stack === 'pos').map(seg => (
                <Bar key={seg.key} dataKey={seg.key} stackId="pos" isAnimationActive={false} radius={[0, 4, 4, 0]}>
                  <Cell fill={POSITIVE_COLOR} />
                </Bar>
              ))}
              {segments.filter(s => s.stack === 'neg').map(seg => (
                <Bar key={seg.key} dataKey={seg.key} stackId="neg" isAnimationActive={false} radius={[4, 0, 0, 4]}>
                  <Cell fill={NEGATIVE_COLOR} />
                </Bar>
              ))}
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
