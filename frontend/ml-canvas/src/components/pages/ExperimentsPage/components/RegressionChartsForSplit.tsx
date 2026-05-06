/**
 * All 8 regression evaluation charts for one split (train/test/val).
 * Extracted from ExperimentsPage to keep that file focused on orchestration.
 *
 * Charts: Actual vs Predicted, Residuals vs Predicted, Residual Histogram,
 * Q-Q Plot, Relative Error Histogram, Scale-Location, Sorted Actual vs Predicted,
 * Residual Lag Plot — plus the absolute-error percentile chips strip.
 */

import React from 'react';
import {
  BarChart, Bar, XAxis, YAxis, CartesianGrid, Tooltip, Legend, ResponsiveContainer,
  ScatterChart, Scatter, Line, ReferenceLine, ComposedChart,
} from 'recharts';
import { Loader2, Check, Download } from 'lucide-react';
import { InfoTooltip } from '../../../ui/InfoTooltip';
import type { EvaluationSplit } from '../types';
import {
  getQQData,
  getErrorPercentiles,
  getRelativeErrorHist,
  getScaleLocationData,
  getSortedActualPred,
  getResidualLagData,
  getResidualHistogram,
} from '../utils/regressionCharts';

interface Props {
  splitName: string;
  splitData: EvaluationSplit;
  handleDownload: (elementId: string, fileName: string) => Promise<void>;
  downloadingChart: string | null;
  doneChart: string | null;
}

export const RegressionChartsForSplit: React.FC<Props> = ({
  splitName,
  splitData,
  handleDownload,
  downloadingChart,
  doneChart,
}) => {
  const data = splitData.y_true.map((y, i) => ({ x: y, y: splitData.y_pred[i] }));

  // Derived datasets for the percentile chips + secondary charts.
  const hist = getResidualHistogram(splitData.y_true as number[], splitData.y_pred as number[]);
  const meanBinLabel = hist
    ? hist.bins.reduce((best, b, i) => (
        Math.abs(parseFloat(b.label) - hist.mean) < Math.abs(parseFloat(hist.bins[best]!.label) - hist.mean) ? i : best
      ), 0)
    : 0;
  const pct = getErrorPercentiles(splitData.y_true as number[], splitData.y_pred as number[]);
  const relHist = getRelativeErrorHist(splitData.y_true as number[], splitData.y_pred as number[]);
  const qqData = getQQData(splitData.y_true as number[], splitData.y_pred as number[]);
  const qqMin = qqData.length ? Math.min(...qqData.map(d => Math.min(d.theoretical, d.sample))) : 0;
  const qqMax = qqData.length ? Math.max(...qqData.map(d => Math.max(d.theoretical, d.sample))) : 1;
  const slData = getScaleLocationData(splitData.y_true as number[], splitData.y_pred as number[]);
  const sortedData = getSortedActualPred(splitData.y_true as number[], splitData.y_pred as number[]);
  const lagData = getResidualLagData(splitData.y_true as number[], splitData.y_pred as number[]);

  /** Tiny renderer for the per-chart download button — one place, no copy/paste. */
  const downloadBtn = (id: string, fileName: string) => (
    <div className="absolute top-0 right-0 z-10 opacity-0 group-hover:opacity-100 transition-opacity" data-export-ignore="true">
      <button
        onClick={() => void handleDownload(id, fileName)}
        disabled={downloadingChart === id}
        className="p-1.5 bg-white dark:bg-gray-800 border border-gray-200 dark:border-gray-600 rounded shadow-sm text-gray-500 hover:text-blue-600 disabled:opacity-50"
        title="Download Graph"
      >
        {downloadingChart === id
          ? <Loader2 className="w-3.5 h-3.5 animate-spin" />
          : doneChart === id
            ? <Check className="w-3.5 h-3.5 text-green-500" />
            : <Download className="w-3.5 h-3.5" />}
      </button>
    </div>
  );

  return (
    <div className="grid grid-cols-1 gap-8">
      {/* 1. Scatter Plot: Actual vs Predicted */}
      <div className="h-[300px] relative group" id={`${splitName}-actual-pred`}>
        {downloadBtn(`${splitName}-actual-pred`, `${splitName}_actual_vs_predicted`)}
        <div className="flex items-center justify-center gap-1.5 mb-2">
          <h5 className="text-xs font-medium text-gray-500 dark:text-gray-400 text-center">Actual vs Predicted</h5>
          <InfoTooltip text="Scatter plot of true target values (X-axis) vs model predictions (Y-axis). The dashed diagonal = perfect prediction. Points above the line = over-predictions; below = under-predictions. Tight clustering along the diagonal means high accuracy." align="center" size="sm" />
        </div>
        <ResponsiveContainer width="100%" height="100%">
          <ScatterChart margin={{ top: 20, right: 30, bottom: 40, left: 30 }}>
            <CartesianGrid strokeDasharray="3 3" opacity={0.2} />
            <XAxis type="number" dataKey="x" name="Actual" unit="" label={{ value: 'Actual Values', position: 'bottom', offset: 0, fontSize: 12 }} tick={{ fontSize: 11 }} />
            <YAxis type="number" dataKey="y" name="Predicted" unit="" label={{ value: 'Predicted Values', angle: -90, position: 'insideLeft', style: { textAnchor: 'middle' }, fontSize: 12 }} tick={{ fontSize: 11 }} />
            <Tooltip
              cursor={{ strokeDasharray: '3 3' }}
              content={({ active, payload }) => {
                if (active && payload && payload.length) {
                  const d = payload[0].payload;
                  return (
                    <div className="bg-white dark:bg-gray-800 p-2 border border-gray-200 dark:border-gray-700 shadow-sm rounded text-xs">
                      <p className="font-medium">Actual: {d.x.toFixed(4)}</p>
                      <p className="font-medium">Predicted: {d.y.toFixed(4)}</p>
                      <p className="text-gray-500">Error: {(d.y - d.x).toFixed(4)}</p>
                    </div>
                  );
                }
                return null;
              }}
            />
            {/* Reference Line for Perfect Prediction (y=x) */}
            <Line dataKey="x" stroke="#ccc" strokeDasharray="3 3" dot={false} activeDot={false} legendType="none" isAnimationActive={false} />
            <Scatter name="Predictions" data={data} fill="#8884d8" fillOpacity={0.6} />
          </ScatterChart>
        </ResponsiveContainer>
      </div>

      {/* 2. Residual Plot: Residuals vs Predicted */}
      <div className="h-[300px] relative group" id={`${splitName}-residuals`}>
        {downloadBtn(`${splitName}-residuals`, `${splitName}_residuals`)}
        <div className="flex items-center justify-center gap-1.5 mb-2">
          <h5 className="text-xs font-medium text-gray-500 dark:text-gray-400 text-center">Residuals vs Predicted</h5>
          <InfoTooltip text="Residual (Actual − Predicted) on Y-axis vs predicted value on X-axis. Ideal: a flat cloud of points centred at 0 with no pattern. A funnel shape = variance grows with prediction (heteroscedasticity). A curve = the relationship is non-linear." align="center" size="sm" />
        </div>
        <ResponsiveContainer width="100%" height="100%">
          <ScatterChart margin={{ top: 20, right: 30, bottom: 40, left: 30 }}>
            <CartesianGrid strokeDasharray="3 3" opacity={0.2} />
            <XAxis type="number" dataKey="y" name="Predicted" unit="" label={{ value: 'Predicted Values', position: 'bottom', offset: 0, fontSize: 12 }} tick={{ fontSize: 11 }} />
            <YAxis type="number" dataKey="residual" name="Residual" unit="" label={{ value: 'Residuals (Actual - Predicted)', angle: -90, position: 'insideLeft', style: { textAnchor: 'middle' }, fontSize: 12 }} tick={{ fontSize: 11 }} />
            <Tooltip
              cursor={{ strokeDasharray: '3 3' }}
              content={({ active, payload }) => {
                if (active && payload && payload.length) {
                  const d = payload[0].payload;
                  return (
                    <div className="bg-white dark:bg-gray-800 p-2 border border-gray-200 dark:border-gray-700 shadow-sm rounded text-xs">
                      <p className="font-medium">Predicted: {d.y.toFixed(4)}</p>
                      <p className="font-medium">Residual: {d.residual.toFixed(4)}</p>
                    </div>
                  );
                }
                return null;
              }}
            />
            {/* Zero Line */}
            <ReferenceLine y={0} stroke="#ccc" strokeDasharray="3 3" />
            <Scatter
              name="Residuals"
              data={data.map(d => ({ ...d, residual: (d.x as number) - (d.y as number) }))}
              fill="#82ca9d"
              fillOpacity={0.6}
            />
          </ScatterChart>
        </ResponsiveContainer>
      </div>

      {/* Absolute-error percentile chips */}
      {pct && (
        <div className="flex items-center gap-3 flex-wrap">
          <span className="text-xs text-gray-400 dark:text-gray-500 font-medium">Absolute error percentiles:</span>
          {([
            ['P50 (median)', pct.p50, 'bg-blue-50 dark:bg-blue-900/30 text-blue-700 dark:text-blue-300 border-blue-200 dark:border-blue-700'],
            ['P90', pct.p90, 'bg-amber-50 dark:bg-amber-900/30 text-amber-700 dark:text-amber-300 border-amber-200 dark:border-amber-700'],
            ['P95', pct.p95, 'bg-red-50 dark:bg-red-900/30 text-red-700 dark:text-red-300 border-red-200 dark:border-red-700'],
          ] as [string, number, string][]).map(([label, val, cls]) => (
            <span key={label} className={`px-2 py-0.5 rounded-full text-xs font-medium border ${cls}`}>
              {label}: <span className="font-mono">{val.toFixed(4)}</span>
            </span>
          ))}
          <InfoTooltip text="Absolute error percentiles show tail behaviour beyond MAE/RMSE.&#10;P50 = median absolute error (robust to outliers).&#10;P90 = 90% of predictions are within this error.&#10;P95 = worst-case error for 95% of samples." align="center" size="sm" />
        </div>
      )}

      {/* 3. Residual Histogram */}
      {hist && (
        <div className="h-[260px] relative group" id={`${splitName}-residual-hist`}>
          {downloadBtn(`${splitName}-residual-hist`, `${splitName}_residual_histogram`)}
          <h5 className="text-xs font-medium text-gray-500 dark:text-gray-400 mb-2 text-center flex items-center justify-center gap-1">
            Residual Distribution
            <InfoTooltip text="Histogram of (Actual − Predicted). A bell-shaped distribution centred at 0 indicates unbiased predictions. A peak offset from 0 signals systematic over- or under-prediction." align="center" size="sm" />
          </h5>
          <ResponsiveContainer width="100%" height="88%">
            <BarChart data={hist.bins} margin={{ top: 5, right: 20, bottom: 28, left: 30 }}>
              <CartesianGrid strokeDasharray="3 3" opacity={0.15} />
              <XAxis dataKey="label" tick={{ fontSize: 10 }} label={{ value: 'Residual (Actual − Predicted)', position: 'insideBottom', offset: -8, fontSize: 11, fill: '#9ca3af' }} />
              <YAxis tick={{ fontSize: 10 }} label={{ value: 'Count', angle: -90, position: 'insideLeft', style: { textAnchor: 'middle' }, fontSize: 11, fill: '#9ca3af' }} />
              <Tooltip
                content={({ active, payload }) => {
                  if (active && payload?.length) {
                    const p = payload[0]!;
                    return (
                      <div className="bg-white dark:bg-gray-800 p-2 border border-gray-200 dark:border-gray-700 shadow-sm rounded text-xs">
                        <p>Bin start: <span className="font-mono">{String(p.payload.label)}</span></p>
                        <p>Count: <span className="font-mono font-semibold">{String(p.value)}</span></p>
                      </div>
                    );
                  }
                  return null;
                }}
              />
              <ReferenceLine x={hist.bins.find(b => parseFloat(b.label) >= 0)?.label ?? ''} stroke="#ef4444" strokeDasharray="3 3" strokeWidth={1.5} label={{ value: '0', position: 'top', fontSize: 10, fill: '#ef4444' }} />
              <ReferenceLine x={hist.bins[meanBinLabel]?.label ?? ''} stroke="#f59e0b" strokeDasharray="4 2" strokeWidth={1.5} label={{ value: `\u03bc=${hist.mean.toFixed(2)}`, position: 'top', fontSize: 10, fill: '#f59e0b' }} />
              <Bar dataKey="count" fill="#6ee7b7" fillOpacity={0.8} isAnimationActive={false} />
            </BarChart>
          </ResponsiveContainer>
        </div>
      )}

      {/* 4. Q-Q Plot */}
      {qqData.length > 0 && (
        <div className="h-[280px] relative group" id={`${splitName}-qq`}>
          {downloadBtn(`${splitName}-qq`, `${splitName}_qq_plot`)}
          <h5 className="text-xs font-medium text-gray-500 dark:text-gray-400 mb-2 text-center flex items-center justify-center gap-1">
            Q-Q Plot (Residuals)
            <InfoTooltip text="Quantile-Quantile plot of residuals vs a normal distribution. Points on the diagonal line = residuals are normally distributed (good). S-curves or heavy tails reveal heteroscedasticity or outliers." align="center" size="sm" />
          </h5>
          <ResponsiveContainer width="100%" height="88%">
            <ScatterChart margin={{ top: 5, right: 20, bottom: 32, left: 40 }}>
              <CartesianGrid strokeDasharray="3 3" opacity={0.15} />
              <XAxis type="number" dataKey="theoretical" name="Theoretical" tickFormatter={(v: number) => v.toFixed(2)} tick={{ fontSize: 10 }} label={{ value: 'Theoretical Quantile', position: 'insideBottom', offset: -10, fontSize: 11, fill: '#9ca3af' }} domain={[qqMin, qqMax]} />
              <YAxis type="number" dataKey="sample" name="Sample" tickFormatter={(v: number) => v.toFixed(2)} tick={{ fontSize: 10 }} label={{ value: 'Sample Quantile', angle: -90, position: 'insideLeft', offset: 10, fontSize: 11, fill: '#9ca3af' }} domain={[qqMin, qqMax]} />
              <Tooltip content={({ active, payload }) => {
                if (active && payload?.length) {
                  const d = payload[0]!.payload as { theoretical: number; sample: number };
                  return <div className="bg-white dark:bg-gray-800 p-2 border border-gray-200 dark:border-gray-700 shadow-sm rounded text-xs"><p>Theoretical: <span className="font-mono">{d.theoretical.toFixed(3)}</span></p><p>Sample: <span className="font-mono">{d.sample.toFixed(3)}</span></p></div>;
                }
                return null;
              }} />
              {/* 45-degree diagonal reference */}
              <Line data={[{ theoretical: qqMin, sample: qqMin }, { theoretical: qqMax, sample: qqMax }]} type="linear" dataKey="sample" stroke="#d1d5db" strokeDasharray="4 3" dot={false} legendType="none" strokeWidth={1} isAnimationActive={false} />
              <Scatter data={qqData} fill="#a78bfa" fillOpacity={0.7} isAnimationActive={false} />
            </ScatterChart>
          </ResponsiveContainer>
        </div>
      )}

      {/* 5. Relative Error Histogram */}
      {relHist && (
        <div className="h-[260px] relative group" id={`${splitName}-rel-err`}>
          {downloadBtn(`${splitName}-rel-err`, `${splitName}_relative_error`)}
          <h5 className="text-xs font-medium text-gray-500 dark:text-gray-400 mb-2 text-center flex items-center justify-center gap-1">
            Relative Error Distribution
            <InfoTooltip text="Histogram of (Predicted − Actual) / |Actual|. Shows proportional error, useful when targets span orders of magnitude. Capped at ±200% for readability. A spike at 0 = accurate; wide spread = inconsistent predictions." align="center" size="sm" />
          </h5>
          <ResponsiveContainer width="100%" height="88%">
            <BarChart data={relHist} margin={{ top: 5, right: 20, bottom: 28, left: 30 }}>
              <CartesianGrid strokeDasharray="3 3" opacity={0.15} />
              <XAxis dataKey="label" tick={{ fontSize: 10 }} label={{ value: 'Relative Error (Predicted − Actual) / |Actual|', position: 'insideBottom', offset: -8, fontSize: 11, fill: '#9ca3af' }} />
              <YAxis tick={{ fontSize: 10 }} label={{ value: 'Count', angle: -90, position: 'insideLeft', style: { textAnchor: 'middle' }, fontSize: 11, fill: '#9ca3af' }} />
              <Tooltip content={({ active, payload }) => {
                if (active && payload?.length) {
                  const p = payload[0]!;
                  return <div className="bg-white dark:bg-gray-800 p-2 border border-gray-200 dark:border-gray-700 shadow-sm rounded text-xs"><p>Bin ≥ <span className="font-mono">{String(p.payload.label)}</span></p><p>Count: <span className="font-mono font-semibold">{String(p.value)}</span></p></div>;
                }
                return null;
              }} />
              <ReferenceLine x={relHist.find(b => parseFloat(b.label) >= 0)?.label ?? ''} stroke="#ef4444" strokeDasharray="3 3" strokeWidth={1.5} label={{ value: '0', position: 'top', fontSize: 10, fill: '#ef4444' }} />
              <Bar dataKey="count" fill="#fbbf24" fillOpacity={0.8} isAnimationActive={false} />
            </BarChart>
          </ResponsiveContainer>
        </div>
      )}

      {/* 6. Scale-Location: sqrt(|residual|) vs predicted */}
      {slData.length >= 3 && (
        <div className="h-[280px] relative group" id={`${splitName}-scale-loc`}>
          {downloadBtn(`${splitName}-scale-loc`, `${splitName}_scale_location`)}
          <div className="flex items-center justify-center gap-1.5 mb-2">
            <h5 className="text-xs font-medium text-gray-500 dark:text-gray-400 text-center">Scale-Location</h5>
            <InfoTooltip text="√|residual| vs predicted value. A flat horizontal band = error variance is constant (homoscedastic — good). A rising trend = error grows as predictions get larger (heteroscedastic). This view amplifies the signal compared to the raw residual plot." align="center" size="sm" />
          </div>
          <ResponsiveContainer width="100%" height="88%">
            <ScatterChart margin={{ top: 10, right: 20, bottom: 35, left: 35 }}>
              <CartesianGrid strokeDasharray="3 3" opacity={0.15} />
              <XAxis type="number" dataKey="predicted" tick={{ fontSize: 10 }} label={{ value: 'Predicted', position: 'insideBottom', offset: -8, fontSize: 11, fill: '#9ca3af' }} />
              <YAxis type="number" dataKey="sqrtAbsRes" tick={{ fontSize: 10 }} label={{ value: '√|Residual|', angle: -90, position: 'insideLeft', style: { textAnchor: 'middle' }, fontSize: 11, fill: '#9ca3af' }} />
              <Tooltip cursor={{ strokeDasharray: '3 3' }} content={({ active, payload }) => {
                if (!active || !payload?.length) return null;
                const d = payload[0]!.payload as { predicted?: number; sqrtAbsRes?: number };
                if (d.predicted == null || d.sqrtAbsRes == null) return null;
                return <div className="bg-white dark:bg-gray-800 p-2 border border-gray-200 dark:border-gray-700 rounded text-xs shadow-sm"><p>Predicted <span className="font-mono">{d.predicted.toFixed(4)}</span></p><p>√|Res| <span className="font-mono text-teal-600 dark:text-teal-400">{d.sqrtAbsRes.toFixed(4)}</span></p></div>;
              }} />
              <Scatter data={slData} fill="#14b8a6" fillOpacity={0.5} shape={(props: unknown) => { const p = props as { cx?: number; cy?: number }; return <circle cx={p.cx} cy={p.cy} r={5} fill="#14b8a6" fillOpacity={0.5} stroke="none" />; }} />
            </ScatterChart>
          </ResponsiveContainer>
        </div>
      )}

      {/* 7. Sorted Actual vs Predicted */}
      {sortedData.length >= 3 && (
        <div className="h-[280px] relative group" id={`${splitName}-sorted-pred`}>
          {downloadBtn(`${splitName}-sorted-pred`, `${splitName}_sorted_actual_pred`)}
          <div className="flex items-center justify-center gap-1.5 mb-2">
            <h5 className="text-xs font-medium text-gray-500 dark:text-gray-400 text-center">Sorted Actual vs Predicted</h5>
            <InfoTooltip text="Samples sorted by true value (ascending). Blue line = actual target; orange dots = predictions. Where the dots track the line closely, the model is accurate. Divergence at the extremes (low or high values) reveals where the model struggles most." align="center" size="sm" />
          </div>
          <ResponsiveContainer width="100%" height="88%">
            <ComposedChart data={sortedData} margin={{ top: 10, right: 20, bottom: 35, left: 35 }}>
              <CartesianGrid strokeDasharray="3 3" opacity={0.15} />
              <XAxis dataKey="idx" tick={{ fontSize: 10 }} label={{ value: 'Sample rank (by actual)', position: 'insideBottom', offset: -8, fontSize: 11, fill: '#9ca3af' }} />
              <YAxis tick={{ fontSize: 10 }} label={{ value: 'Value', angle: -90, position: 'insideLeft', style: { textAnchor: 'middle' }, fontSize: 11, fill: '#9ca3af' }} />
              <Tooltip content={({ active, payload }) => {
                if (!active || !payload?.length) return null;
                const d = payload[0]!.payload as { idx: number; actual?: number; predicted?: number };
                if (d.actual == null || d.predicted == null) return null;
                return <div className="bg-white dark:bg-gray-800 p-2 border border-gray-200 dark:border-gray-700 rounded text-xs shadow-sm"><p>Actual <span className="font-mono text-blue-600 dark:text-blue-400">{d.actual.toFixed(4)}</span></p><p>Predicted <span className="font-mono text-orange-500">{d.predicted.toFixed(4)}</span></p><p className="text-gray-400 text-[10px]">Error: {(d.predicted - d.actual).toFixed(4)}</p></div>;
              }} />
              <Line type="monotone" dataKey="actual" stroke="#3b82f6" strokeWidth={1.5} dot={false} isAnimationActive={false} name="Actual" />
              <Line type="monotone" dataKey="predicted" stroke="#f97316" strokeWidth={0} dot={{ r: 2, fill: '#f97316', fillOpacity: 0.6 }} activeDot={{ r: 4 }} isAnimationActive={false} name="Predicted" />
              <Legend verticalAlign="top" height={18} iconSize={10} formatter={(value) => <span className="text-xs text-gray-600 dark:text-gray-400">{value}</span>} />
            </ComposedChart>
          </ResponsiveContainer>
        </div>
      )}

      {/* 8. Residual Lag Plot */}
      {lagData.length >= 3 && (
        <div className="h-[280px] relative group" id={`${splitName}-lag`}>
          {downloadBtn(`${splitName}-lag`, `${splitName}_residual_lag`)}
          <div className="flex items-center justify-center gap-1.5 mb-2">
            <h5 className="text-xs font-medium text-gray-500 dark:text-gray-400 text-center">Residual Lag Plot</h5>
            <InfoTooltip text="Residual[i] (Y) vs Residual[i-1] (X). A random scatter centred at (0,0) means errors are independent — ideal. A diagonal pattern = the model consistently over- or under-corrects. Curved or fan-shaped clusters indicate serial correlation (common in time-series data)." align="center" size="sm" />
          </div>
          <ResponsiveContainer width="100%" height="88%">
            <ScatterChart margin={{ top: 10, right: 20, bottom: 35, left: 35 }}>
              <CartesianGrid strokeDasharray="3 3" opacity={0.15} />
              <XAxis type="number" dataKey="r0" tick={{ fontSize: 10 }} label={{ value: 'Residual[i-1]', position: 'insideBottom', offset: -8, fontSize: 11, fill: '#9ca3af' }} />
              <YAxis type="number" dataKey="r1" tick={{ fontSize: 10 }} label={{ value: 'Residual[i]', angle: -90, position: 'insideLeft', style: { textAnchor: 'middle' }, fontSize: 11, fill: '#9ca3af' }} />
              <ReferenceLine y={0} stroke="#d1d5db" strokeDasharray="3 3" />
              <ReferenceLine x={0} stroke="#d1d5db" strokeDasharray="3 3" />
              <Tooltip cursor={{ strokeDasharray: '3 3' }} content={({ active, payload }) => {
                if (!active || !payload?.length) return null;
                const d = payload[0]!.payload as { r0: number; r1: number };
                return <div className="bg-white dark:bg-gray-800 p-2 border border-gray-200 dark:border-gray-700 rounded text-xs shadow-sm"><p>Res[i-1] <span className="font-mono">{d.r0.toFixed(4)}</span></p><p>Res[i] <span className="font-mono text-pink-500">{d.r1.toFixed(4)}</span></p></div>;
              }} />
              <Scatter data={lagData} fill="#ec4899" fillOpacity={0.45} shape={(props: unknown) => { const p = props as { cx?: number; cy?: number }; return <circle cx={p.cx} cy={p.cy} r={5} fill="#ec4899" fillOpacity={0.45} stroke="none" />; }} />
            </ScatterChart>
          </ResponsiveContainer>
        </div>
      )}
    </div>
  );
};
