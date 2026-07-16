import React, { useMemo } from 'react';
import { Loader2, Check, Download } from 'lucide-react';
import {
  ScatterChart,
  Scatter,
  XAxis,
  YAxis,
  ZAxis,
  CartesianGrid,
  Tooltip,
  ResponsiveContainer,
  ReferenceLine,
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

const MAX_FEATURES = 12;

interface BeeswarmPoint {
  feature: string;
  featureIndex: number;
  y: number;
  shapValue: number;
  featureValue: number;
  color: string;
}

// Simple deterministic pseudo-random jitter (avoids points stacking exactly
// on the same row) — seeded by index so re-renders don't reshuffle points.
const jitterFor = (seed: number): number => {
  const x = Math.sin(seed * 12.9898) * 43758.5453;
  return (x - Math.floor(x)) - 0.5;
};

// Blue (low) -> red (high) diverging colour scale for the normalised feature value.
const colorForNormalized = (t: number): string => {
  const clamped = Math.max(0, Math.min(1, t));
  const low = { r: 0x3b, g: 0x82, b: 0xf6 };
  const high = { r: 0xef, g: 0x44, b: 0x44 };
  const r = Math.round(low.r + (high.r - low.r) * clamped);
  const g = Math.round(low.g + (high.g - low.g) * clamped);
  const b = Math.round(low.b + (high.b - low.b) * clamped);
  return `rgb(${r}, ${g}, ${b})`;
};

interface DotShapeProps {
  cx?: number;
  cy?: number;
  payload?: BeeswarmPoint;
}

const BeeswarmDot: React.FC<DotShapeProps> = ({ cx, cy, payload }) => {
  if (cx === undefined || cy === undefined || !payload) return null;
  return <circle cx={cx} cy={cy} r={3.5} fill={payload.color} fillOpacity={0.75} stroke="none" />;
};

export const ShapBeeswarmView: React.FC<Props> = ({
  jobId,
  modelType,
  shapExplanation,
  handleDownload,
  downloadingChart,
  doneChart,
}) => {
  const chartId = `shap-beeswarm-chart-${jobId}`;
  const chartTheme = useChartTheme();

  const { points, topFeatures } = useMemo(() => {
    const { samples, mean_abs_importance: meanAbs } = shapExplanation;
    const topFeatures = Object.entries(meanAbs)
      .sort((a, b) => b[1] - a[1])
      .slice(0, MAX_FEATURES)
      .map(([name]) => name);

    const points: BeeswarmPoint[] = [];
    topFeatures.forEach((feature, featureIndex) => {
      const values = samples.map(s => s.feature_values[feature] ?? 0);
      const min = Math.min(...values);
      const max = Math.max(...values);
      const range = max - min;

      samples.forEach((sample, sampleIndex) => {
        const featureValue = sample.feature_values[feature] ?? 0;
        const normalized = range > 0 ? (featureValue - min) / range : 0.5;
        points.push({
          feature,
          featureIndex,
          y: featureIndex + jitterFor(sampleIndex * 97 + featureIndex) * 0.6,
          shapValue: sample.shap_values[feature] ?? 0,
          featureValue,
          color: colorForNormalized(normalized),
        });
      });
    });

    return { points, topFeatures };
  }, [shapExplanation]);

  if (topFeatures.length === 0) return null;

  return (
    <div className="space-y-6">
      <div className="flex items-center gap-2">
        <h3 className="text-lg font-medium text-gray-800 dark:text-gray-100">
          SHAP Beeswarm — {modelType !== 'unknown' ? modelType : jobId.slice(0, 8)}
        </h3>
        <InfoTooltip
          text="Each point is one sampled row. Position on the x-axis shows that row's SHAP value (impact on the model's output) for the feature. Colour shows whether that row's own feature value was low (blue) or high (red), revealing whether high/low values of a feature push predictions up or down."
          align="center"
        />
      </div>
      <div className="bg-white dark:bg-gray-800 rounded-lg border border-gray-200 dark:border-gray-700 p-6 relative group" id={chartId}>
        <div className="absolute top-4 right-4 z-10 opacity-0 group-hover:opacity-100 transition-opacity" data-export-ignore="true">
          <button
            onClick={() => void handleDownload(chartId, `shap_beeswarm_${jobId.slice(0, 8)}`)}
            disabled={downloadingChart === chartId}
            className="p-1.5 bg-white dark:bg-gray-800 border border-gray-200 dark:border-gray-600 rounded shadow-sm text-gray-500 hover:text-blue-600 disabled:opacity-50"
            title="Download Graph"
          >
            {downloadingChart === chartId ? <Loader2 className="w-3.5 h-3.5 animate-spin" /> : doneChart === chartId ? <Check className="w-3.5 h-3.5 text-green-500" /> : <Download className="w-3.5 h-3.5" />}
          </button>
        </div>
        <div className="h-[500px]">
          <ResponsiveContainer width="100%" height="100%">
            <ScatterChart margin={{ top: 5, right: 30, bottom: 5, left: 120 }}>
              <CartesianGrid strokeDasharray="3 3" opacity={0.1} stroke={chartTheme.gridColor} />
              <XAxis
                type="number"
                dataKey="shapValue"
                name="SHAP value"
                tick={{ fontSize: 12, fill: chartTheme.axisColor }}
                label={{ value: 'SHAP value (impact on model output)', position: 'insideBottom', offset: -5, fontSize: 11, fill: chartTheme.axisColor }}
              />
              <YAxis
                type="number"
                dataKey="y"
                domain={[-0.75, topFeatures.length - 0.25]}
                ticks={topFeatures.map((_, i) => i)}
                tickFormatter={(v: number) => topFeatures[Math.round(v)] ?? ''}
                tick={{ fontSize: 11, fill: chartTheme.axisColor }}
                width={110}
                reversed
              />
              <ZAxis range={[20, 20]} />
              <ReferenceLine x={0} stroke="#94a3b8" strokeDasharray="3 3" />
              <Tooltip
                contentStyle={chartTheme.tooltipContentStyle}
                itemStyle={chartTheme.tooltipItemStyle}
                labelStyle={chartTheme.tooltipLabelStyle}
                formatter={(value: number, name: string) => [value.toFixed(3), name]}
                labelFormatter={() => ''}
              />
              <Scatter data={points} shape={(props: unknown) => <BeeswarmDot {...(props as DotShapeProps)} />} />
            </ScatterChart>
          </ResponsiveContainer>
        </div>
        <p className="text-[11px] text-gray-400 dark:text-gray-500 mt-2 text-center">
          Top {topFeatures.length} features by mean |SHAP value| · colour: <span className="text-blue-500">low</span> → <span className="text-red-500">high</span> feature value
        </p>
      </div>
    </div>
  );
};
