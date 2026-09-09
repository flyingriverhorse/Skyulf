import React from 'react';
import { BarChart, Bar, XAxis, YAxis, Tooltip, ResponsiveContainer, Cell } from 'recharts';
import type { ErrorLogPageState } from './useErrorLogPage';

type ErrorLogOverviewProps = Pick<ErrorLogPageState,
  | 'loading'
  | 'events'
  | 'pipelineLogs'
  | 'mergedTimeline'
>;

export const ErrorLogOverview: React.FC<ErrorLogOverviewProps> = ({
  loading,
  events,
  pipelineLogs,
  mergedTimeline,
}) => {
  if (loading || (events.length === 0 && pipelineLogs.length === 0)) return null;
  const total500 = events.filter(e => e.status_code >= 500).length;
  return (<>
    {/* Stats */}
    <div className="grid grid-cols-3 gap-4 mb-6">
      {[
        { label: 'HTTP events', value: events.length, color: 'text-slate-700 dark:text-slate-200' },
        { label: 'Server errors (5xx)', value: total500, color: 'text-red-600 dark:text-red-400' },
        { label: 'Pipeline failures', value: pipelineLogs.filter(l => l.level === 'error').length, color: 'text-amber-600 dark:text-amber-400' },
      ].map(s => (
        <div key={s.label} className="bg-white dark:bg-slate-800 rounded-xl border border-slate-200 dark:border-slate-700 p-4">
          <p className="text-xs text-slate-500 dark:text-slate-400 mb-1">{s.label}</p>
          <p className={`text-2xl font-bold ${s.color}`}>{s.value}</p>
        </div>
      ))}
    </div>

    {/* Timeline chart */}
    <div className="bg-white dark:bg-slate-800 rounded-xl border border-slate-200 dark:border-slate-700 p-4 mb-6">
      <p className="text-xs font-medium text-slate-500 dark:text-slate-400 mb-3">Errors per hour — last 24 h</p>
      <ResponsiveContainer width="100%" height={80}>
        <BarChart data={mergedTimeline} margin={{ top: 0, right: 0, left: -30, bottom: 0 }}>
          <XAxis
            dataKey="hour"
            tickFormatter={h => h.slice(11, 16)}
            tick={{ fontSize: 10, fill: '#94a3b8' }}
            interval="preserveStartEnd"
            axisLine={false}
            tickLine={false}
          />
          <YAxis allowDecimals={false} tick={{ fontSize: 10, fill: '#94a3b8' }} axisLine={false} tickLine={false} />
          <Tooltip
            formatter={(v: number) => [v, 'errors']}
            labelFormatter={l => `Hour starting ${l}`}
            contentStyle={{ fontSize: 12, borderRadius: 8, border: '1px solid #e2e8f0' }}
          />
          <Bar dataKey="count" radius={[3, 3, 0, 0]}>
            {mergedTimeline.map((t, i) => (
              <Cell key={i} fill={t.count > 0 ? '#ef4444' : '#e2e8f0'} fillOpacity={t.count > 0 ? 0.85 : 0.4} />
            ))}
          </Bar>
        </BarChart>
      </ResponsiveContainer>
    </div>

  </>);
};
