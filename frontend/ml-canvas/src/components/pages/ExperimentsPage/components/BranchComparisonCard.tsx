import React, { useMemo } from 'react';
import { GitBranch, Trophy } from 'lucide-react';
import type { JobInfo } from '../../../../core/api/jobs';
import { formatMetricName } from '../../../../core/utils/format';
import { getMetricDirection, pickBestIndex, type MetricDirection } from '../../../../core/utils/metricMeta';
import { MetricDirectionBadge } from '../../../ui/MetricDirectionBadge';
import { groupJobsByScoringMetric } from '../utils/jobMeta';

interface Props {
  selectedJobs: JobInfo[];
  getDuration: (start: string | null, end: string | null) => string;
}

interface MetricRow {
  key: string;
  label: string;
  values: (number | undefined)[];
  bestIndex: number | null;
  direction: MetricDirection;
}

interface GroupRow {
  parentId: string;
  groupJobs: JobInfo[];
  metricRows: MetricRow[];
}

export const BranchComparisonCard: React.FC<Props> = ({ selectedJobs, getDuration }) => {
  // All of the grouping/sorting/best-value work below scans every selected
  // job and metric, so it's memoised on selectedJobs to avoid recomputing on
  // every unrelated re-render.
  const multiGroups: GroupRow[] = useMemo(() => {
    const branchJobs = selectedJobs.filter(j => j.parent_pipeline_id != null);
    // Group by parent_pipeline_id
    const groups = new Map<string, JobInfo[]>();
    branchJobs.forEach(j => {
      const key = j.parent_pipeline_id!;
      if (!groups.has(key)) groups.set(key, []);
      groups.get(key)!.push(j);
    });
    // Only show groups with 2+ branches
    const rawMultiGroups = Array.from(groups.entries()).filter(([, jobs]) => jobs.length >= 2);

    // Collect all metric keys across branch jobs (test + best_score only)
    const allBranchMetricKeys = Array.from(new Set(
      branchJobs.flatMap(j => {
        const m = (j.metrics || j.result?.metrics || {}) as Record<string, unknown>;
        return Object.keys(m).filter(k => typeof m[k] === 'number');
      })
    )).filter(k => k.startsWith('test_') || k === 'best_score').sort();

    return rawMultiGroups.map(([parentId, groupJobsRaw]) => {
      const groupJobs = [...groupJobsRaw].sort((a, b) => (a.branch_index ?? 0) - (b.branch_index ?? 0));
      // F-36: best_score is only comparable between runs that optimised the
      // same metric, so expand it into one row per metric — labelled from
      // that row's own metric and starred only within it.
      const metricRows: MetricRow[] = allBranchMetricKeys.flatMap(key => {
        const direction = getMetricDirection(key);
        const allValues = groupJobs.map(j => {
          const m = (j.metrics || j.result?.metrics || {}) as Record<string, unknown>;
          const raw = m[key];
          return typeof raw === 'number' ? raw : undefined;
        });
        const rows = key === 'best_score'
          ? groupJobsByScoringMetric(groupJobs)
              .map(row => ({
                key: `best_score__${row.metric ?? 'unknown'}`,
                label: `Best Score (${formatMetricName(row.metric) || 'CV'})`,
                values: allValues.map((v, i) => (row.indices.includes(i) ? v : undefined)),
              }))
              .filter(row => row.values.some(v => v !== undefined))
          : [{
              key,
              label: key.replace(/_/g, ' ').replace(/\b\w/g, c => c.toUpperCase()),
              values: allValues,
            }];
        return rows.map(row => ({ ...row, direction, bestIndex: pickBestIndex(row.values, direction) }));
      });
      return { parentId, groupJobs, metricRows };
    });
  }, [selectedJobs]);

  if (multiGroups.length === 0) return null;

  return (
    <>
      {multiGroups.map(({ parentId, groupJobs, metricRows }) => (
        <div key={parentId} className="bg-gradient-to-r from-purple-50 to-blue-50 dark:from-purple-900/10 dark:to-blue-900/10 rounded-lg border border-purple-200 dark:border-purple-800 p-4">
          <div className="flex items-center gap-2 mb-3">
            <GitBranch className="w-4 h-4 text-purple-600 dark:text-purple-400" />
            <h3 className="text-sm font-semibold text-purple-800 dark:text-purple-300">
              Parallel Run Comparison
            </h3>
            <span className="text-xs text-purple-500 dark:text-purple-400 font-mono">
              {groupJobs.length} paths · run {parentId.replace(/^preview_/, '').replace(/__branch_.*$/, '').slice(0, 8)}
            </span>
          </div>
          <div className="overflow-x-auto">
            <table className="w-full text-xs text-left">
              <thead>
                <tr className="border-b border-purple-200 dark:border-purple-700">
                  <th className="px-3 py-1.5 text-gray-600 dark:text-gray-400 font-medium">Metric</th>
                  {groupJobs.map(j => (
                    <th key={j.job_id} className="px-3 py-1.5 font-medium text-gray-700 dark:text-gray-300">
                      <div className="flex items-center gap-1">
                        <span className="inline-block w-2 h-2 rounded-full" style={{ backgroundColor: `hsl(${(j.branch_index ?? 0) * 120}, 70%, 50%)` }} />
                        Path {String.fromCharCode(65 + (j.branch_index ?? 0))} · {j.model_type}
                        {j.promoted_at && <Trophy className="w-3 h-3 text-amber-500 ml-1" />}
                      </div>
                    </th>
                  ))}
                </tr>
              </thead>
              <tbody>
                {metricRows.map(({ key, label, values, bestIndex, direction }) => (
                  <tr key={key} className="border-b border-purple-100 dark:border-purple-800/50">
                    <td className="px-3 py-1.5 text-gray-600 dark:text-gray-400">
                      <span className="inline-flex items-center gap-1">
                        {label}
                        <MetricDirectionBadge direction={direction} />
                      </span>
                    </td>
                    {groupJobs.map((j, i) => {
                      const val = values[i];
                      const isBest = bestIndex === i;
                      return (
                        <td key={j.job_id} className={`px-3 py-1.5 font-mono ${isBest ? 'text-green-600 dark:text-green-400 font-bold' : 'text-gray-600 dark:text-gray-300'}`}>
                          {val != null
                            ? val.toFixed(4)
                            : <span className="text-gray-400" title="Not reported by this run">—</span>}
                          {isBest && ' ★'}
                        </td>
                      );
                    })}
                  </tr>
                ))}
                <tr className="border-b border-purple-100 dark:border-purple-800/50">
                  <td className="px-3 py-1.5 text-gray-600 dark:text-gray-400">Duration</td>
                  {groupJobs.map(j => (
                    <td key={j.job_id} className="px-3 py-1.5 font-mono text-gray-600 dark:text-gray-300">
                      {getDuration(j.start_time, j.end_time)}
                    </td>
                  ))}
                </tr>
              </tbody>
            </table>
          </div>
        </div>
      ))}
    </>
  );
};
