import type { JobInfo } from '../../../../../core/api/jobs';
import { formatMetricName, getMetricDescription } from '../../../../../core/utils/format';
import { getMetricDirection, getMetricSplitLabel, pickBestIndex } from '../../../../../core/utils/metricMeta';
import { InfoTooltip } from '../../../../ui/InfoTooltip';
import { MetricDirectionBadge } from '../../../../ui/MetricDirectionBadge';
import { groupJobsByScoringMetric } from '../../utils/jobMeta';

export function MetricRows({ selectedJobs, metricKeys }: { selectedJobs: JobInfo[]; metricKeys: string[] }) {
  return <>{metricKeys.flatMap(metricKey => {
    const direction = getMetricDirection(metricKey);
    const splitLabel = getMetricSplitLabel(metricKey);
    const allValues = selectedJobs.map(job => {
      const m = (job.metrics || job.result?.metrics || {}) as Record<string, unknown>;
      const raw = m[metricKey];
      return typeof raw === 'number' ? raw : undefined;
    });

    // F-36: best_score is only comparable between runs that
    // optimised the same metric (a basic run silently carries an
    // internal accuracy default while a tuned run carries the
    // user's pick). Expand it into one row per metric — labelled
    // with that row's own metric and starring only within it —
    // instead of one mixed row labelled from the first job.
    const rows = metricKey === 'best_score'
      ? groupJobsByScoringMetric(selectedJobs)
        .map(row => ({
          key: `best_score__${row.metric ?? 'unknown'}`,
          label: `Best Score (${formatMetricName(row.metric) || 'CV'})`,
          values: allValues.map((v, i) => (row.indices.includes(i) ? v : undefined)),
        }))
        .filter(row => row.values.some(v => v !== undefined))
      : [{
        key: metricKey,
        label: metricKey.replace(/_/g, ' ').replace(/\b\w/g, c => c.toUpperCase()),
        values: allValues,
      }];

    return rows.map(row => {
      const bestIndex = pickBestIndex(row.values, direction);

      return (
        <tr key={row.key} className="bg-white dark:bg-gray-800 hover:bg-gray-50 dark:hover:bg-gray-700/50">
          <td className="px-4 py-1.5 text-gray-500 dark:text-gray-400 pl-8">
            <div className="flex items-center gap-1 flex-wrap">
              {row.label}
              <MetricDirectionBadge direction={direction} />
              {splitLabel && (
                <span className="text-[10px] px-1 py-px rounded bg-gray-100 dark:bg-gray-700 text-gray-500 dark:text-gray-400">
                  {splitLabel}
                </span>
              )}
              {getMetricDescription(metricKey) && <InfoTooltip size="sm" text={getMetricDescription(metricKey)!} />}
            </div>
          </td>
          {selectedJobs.map((job, i) => {
            const val = row.values[i];
            const isBest = bestIndex === i;
            return (
              <td
                key={job.job_id}
                className={`px-4 py-1.5 font-mono ${isBest ? 'text-green-600 dark:text-green-400 font-semibold' : 'text-gray-600 dark:text-gray-300'}`}
              >
                {val === undefined
                  ? <span className="text-gray-400" title={`${metricKey} was not reported by this run`}>—</span>
                  : (metricKey.endsWith('_std') ? val.toFixed(6) : val.toFixed(4))}
                {isBest && ' ★'}
              </td>
            );
          })}
        </tr>
      );
    });
  })}</>;
}
