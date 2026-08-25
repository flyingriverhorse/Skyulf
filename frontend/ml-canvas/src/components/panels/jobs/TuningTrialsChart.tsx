import React from 'react';
import {
  CartesianGrid,
  Legend,
  Line,
  LineChart,
  ResponsiveContainer,
  Tooltip as RechartsTooltip,
  XAxis,
  YAxis,
} from 'recharts';
import { Activity } from 'lucide-react';
import { useChartTheme } from '../../../core/hooks/useChartTheme';
import type { SeriesKind, TrialPoint } from '../../../core/hooks/useTuningTrials';

interface TuningTrialsChartProps {
  points: TrialPoint[];
  /** Scoring metric label shown in the heading (e.g. "accuracy"). */
  metric?: string | undefined;
  /** True while the job is still emitting trials. */
  isLive?: boolean;
  /** Which series is charted — boosting iterations vs tuning trials. */
  kind?: SeriesKind | undefined;
}

/**
 * Per-trial score and best-so-far curves for a tuning run. The same
 * component serves live jobs (WebSocket-fed points) and completed jobs
 * (rebuilt from persisted metrics.trials). Hidden with fewer than two
 * points — fixed runs (n_trials=1) and callback-less strategies have
 * nothing to chart.
 */
export const TuningTrialsChart: React.FC<TuningTrialsChartProps> = ({
  points,
  metric,
  isLive = false,
  kind = 'trial',
}) => {
  const theme = useChartTheme();
  if (points.length < 2) return null;
  const isIterations = kind === 'iteration';

  return (
    <div className="bg-white dark:bg-slate-800 p-6 rounded-lg shadow border dark:border-slate-700 mt-6">
      <h2 className="text-lg font-semibold mb-4 flex items-center gap-2">
        <Activity size={18} /> {isIterations ? 'Boosting Iterations' : 'Tuning Trials'}
        {metric ? (
          <span className="text-xs font-normal text-gray-400 ml-1">({metric})</span>
        ) : null}
        {isLive ? (
          <span className="ml-auto flex items-center gap-1.5 text-xs font-medium text-emerald-500">
            <span className="h-2 w-2 rounded-full bg-emerald-500 animate-pulse" />
            Live
          </span>
        ) : null}
      </h2>
      <div className="h-[240px]">
        <ResponsiveContainer width="100%" height="100%">
          <LineChart data={points} margin={{ top: 5, right: 30, left: 0, bottom: 5 }}>
            <CartesianGrid strokeDasharray="3 3" stroke={theme.gridColor} vertical={false} />
            <XAxis
              dataKey="trial"
              tick={{ fill: theme.axisColor, fontSize: 11 }}
              tickLine={false}
              axisLine={{ stroke: theme.gridColor }}
              label={{ value: isIterations ? 'Iteration' : 'Trial', position: 'insideBottomRight', offset: -2, fill: theme.subTextColor, fontSize: 11 }}
            />
            <YAxis
              tick={{ fill: theme.axisColor, fontSize: 11 }}
              tickLine={false}
              axisLine={false}
              domain={['auto', 'auto']}
            />
            <RechartsTooltip
              contentStyle={theme.tooltipContentStyle}
              itemStyle={theme.tooltipItemStyle}
              labelFormatter={(label) => `${isIterations ? 'Iteration' : 'Trial'} ${label}`}
            />
            <Legend verticalAlign="top" height={36} iconType="circle" />
            <Line
              type="monotone"
              dataKey="score"
              name={isIterations ? 'Iteration score' : 'Trial score'}
              stroke="#6366f1"
              strokeWidth={2}
              dot={{ r: 3, fill: '#6366f1' }}
              isAnimationActive={false}
            />
            <Line
              type="stepAfter"
              dataKey="best"
              name="Best so far"
              stroke="#10b981"
              strokeWidth={2}
              dot={false}
              strokeDasharray="5 5"
              isAnimationActive={false}
            />
          </LineChart>
        </ResponsiveContainer>
      </div>
    </div>
  );
};
