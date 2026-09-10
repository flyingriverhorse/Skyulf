import React from 'react';
import { Terminal, CheckCircle } from 'lucide-react';
import { JobInfo } from '../../../../core/api/jobs';
import { formatMetricName } from '../../../../core/utils/format';
import { MetricsGrid } from '../MetricsGrid';
import { getScoringMetric } from './scoring';
import { TuningConfiguration } from './TuningConfiguration';

const FeatureImportancesSection: React.FC<{ result: Record<string, unknown> }> = ({ result }) => {
  const metrics = result.metrics as Record<string, unknown> | undefined;
  const raw = (metrics?.feature_importances ?? result.feature_importances) as Record<string, number> | undefined;

  if (!raw || typeof raw !== 'object') return null;

  const sorted = Object.entries(raw).sort(([, a], [, b]) => b - a).slice(0, 5);
  if (sorted.length === 0) return null;
  const maxVal = sorted[0]![1] || 1;

  return (
    <div className="space-y-2">
      <h4 className="text-xs font-medium text-gray-500 dark:text-gray-400 uppercase tracking-wider">Feature Importances</h4>
      <div className="bg-white dark:bg-gray-800 border border-gray-200 dark:border-gray-700 rounded-lg p-3 max-h-64 overflow-y-auto">
        {sorted.map(([feature, importance]) => (
          <div key={feature} className="flex items-center gap-2 py-1">
            <span className="text-xs text-gray-600 dark:text-gray-300 w-32 truncate shrink-0" title={feature}>{feature}</span>
            <div className="flex-1 h-4 bg-gray-100 dark:bg-gray-700 rounded overflow-hidden">
              <div
                className="h-full bg-blue-500 dark:bg-blue-400 rounded"
                style={{ width: `${(importance / maxVal) * 100}%` }}
              />
            </div>
            <span className="text-xs font-mono text-gray-500 dark:text-gray-400 w-14 text-right shrink-0">{importance.toFixed(4)}</span>
          </div>
        ))}
      </div>
    </div>
  );
};

function JobTuningResults({ job }: { job: JobInfo }) {
  return (
    <>
      {job.job_type === 'tuning' && (
        <div className="space-y-4">
          {/* Tuning Configuration */}
          {job.graph && (
            <div className="p-4 bg-gray-50 dark:bg-gray-900/50 rounded-lg border border-gray-100 dark:border-gray-700">
              <h4 className="text-xs font-medium text-gray-500 dark:text-gray-400 uppercase tracking-wider mb-3">Tuning Configuration</h4>
              <div className="grid grid-cols-2 gap-4 text-xs">
                <TuningConfiguration job={job} />
              </div>
            </div>
          )}

          {/* Best Score */}
          {(job.result as Record<string, unknown>).best_score !== undefined && (
            <div className="p-3 bg-white dark:bg-gray-800 border border-gray-200 dark:border-gray-700 rounded-lg w-fit">
              <div className="text-xs text-gray-500 dark:text-gray-400 mb-1">
                Best Score{getScoringMetric(job) ? ` (${formatMetricName(getScoringMetric(job))})` : ''}
              </div>
              <div className="font-mono font-bold text-lg text-purple-600 dark:text-purple-400">
                {Number((job.result as Record<string, unknown>).best_score).toFixed(4)}
              </div>
            </div>
          )}

          {/* Full Metrics (Train/Test/Val) */}
          {!!(job.result as Record<string, unknown>).metrics && (
            <div className="space-y-2">
              <h4 className="text-xs font-medium text-gray-500 dark:text-gray-400 uppercase tracking-wider">Evaluation Metrics</h4>
              <MetricsGrid
                metrics={(job.result as Record<string, unknown>).metrics as Record<string, unknown>}
                excludeKeys={['best_score', 'best_params', 'trials', 'fold_refit_fallback']}
              />
            </div>
          )}

          <FeatureImportancesSection result={job.result as Record<string, unknown>} />

          {/* Best Params */}
          {!!(job.result as Record<string, unknown>).best_params && (
            <div className="bg-gray-900 text-gray-100 p-4 rounded-lg font-mono text-xs overflow-x-auto">
              <div className="text-gray-500 mb-2"># Best Hyperparameters</div>
              <pre>{JSON.stringify((job.result as Record<string, unknown>).best_params, null, 2)}</pre>
            </div>
          )}
        </div>
      )}
    </>
  );
}

export function JobResults({ job }: { job: JobInfo }) {
  return (
    <>
      {job.result && (
        <div className="space-y-4">
          <div className="flex items-center justify-between">
            <h3 className="text-sm font-medium text-gray-800 dark:text-gray-200 flex items-center gap-2">
              <Terminal className="w-4 h-4" />
              Execution Results
            </h3>
            {job.status === 'completed' && (
              <span className="text-xs px-2 py-1 bg-green-100 text-green-700 dark:bg-green-900/30 dark:text-green-400 rounded-full flex items-center gap-1 border border-green-200 dark:border-green-800 font-medium">
                <CheckCircle className="w-3 h-3" /> Model Ready
              </span>
            )}
          </div>

          {job.job_type === 'training' && !!(job.result as Record<string, unknown>).metrics && (
            <div className="space-y-4">
              <MetricsGrid metrics={(job.result as Record<string, unknown>).metrics as Record<string, unknown>} excludeKeys={['fold_refit_fallback']} />
              <FeatureImportancesSection result={job.result as Record<string, unknown>} />
            </div>
          )}

          <JobTuningResults job={job} />
        </div>
      )}
    </>
  );
}
