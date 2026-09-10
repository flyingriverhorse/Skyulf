import type { ReactNode } from 'react';
import { ResponsiveContainer, BarChart, CartesianGrid, XAxis, YAxis, Tooltip, ReferenceLine, Bar } from 'recharts';
import { InfoTooltip } from '../../../../ui/InfoTooltip';
import type { getResidualHistogram } from '../../utils/regressionCharts';

/** Signed residual histogram with zero and mean reference markers. */
export function ResidualHistogram({ hist, meanBinLabel, splitName, downloadBtn }: {
  hist: ReturnType<typeof getResidualHistogram>;
  meanBinLabel: number;
  splitName: string;
  downloadBtn: (id: string, filename: string) => ReactNode;
}) {
  return <>
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
                if(active && payload?.length) {
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
            <ReferenceLine x={hist.bins.find(b => Number.parseFloat(b.label) >= 0)?.label ?? ''} stroke="#ef4444" strokeDasharray="3 3" strokeWidth={1.5} label={{ value: '0', position: 'top', fontSize: 10, fill: '#ef4444' }} />
            <ReferenceLine x={hist.bins[meanBinLabel]?.label ?? ''} stroke="#f59e0b" strokeDasharray="4 2" strokeWidth={1.5} label={{ value: `\u03bc=${hist.mean.toFixed(2)}`, position: 'top', fontSize: 10, fill: '#f59e0b' }} />
            <Bar dataKey="count" fill="#6ee7b7" fillOpacity={0.8} isAnimationActive={false} />
          </BarChart>
        </ResponsiveContainer>
      </div>
    )}

  </>;
}
