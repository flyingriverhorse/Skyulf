import { Database, AlertCircle } from 'lucide-react';
import { JobInfo } from '../../../../core/api/jobs';
import { RecordLink } from '../../../shared';
import { JobRecordContext } from './types';
import { JobSafety, LeakageGateTile, RefitAuditTile, ScoreAdvisoryTile } from './JobSafety';

function JobDatasetTile({ job, origin, filters }: JobRecordContext) {
  return (
    <div className="p-4 bg-gray-50 dark:bg-gray-900/50 rounded-lg border border-gray-100 dark:border-gray-700">
      <div className="text-xs text-gray-500 dark:text-gray-400 mb-1">Dataset</div>
      <div className="font-medium text-gray-800 dark:text-gray-200 flex items-center gap-2">
        <Database className="w-3 h-3 text-gray-400" />
        {job.dataset_id ? (
          <RecordLink
            recordRef={{ kind: 'dataset', datasetId: job.dataset_id }}
            label={job.dataset_name || job.dataset_id}
            {...(origin !== undefined ? { origin } : {})}
            {...(filters !== undefined ? { filters } : {})}
          />
        ) : (
          job.dataset_name || 'Unknown'
        )}
      </div>
    </div>
  );
}

export function JobStatusSection({ job, safety, ...context }: JobRecordContext & { safety: JobSafety }) {
  const smGridCols = getStatusGridColumns(safety);
  return (
    <div className={`grid grid-cols-2 ${smGridCols} gap-4`}>
      <div className="p-4 bg-gray-50 dark:bg-gray-900/50 rounded-lg border border-gray-100 dark:border-gray-700">
        <div className="text-xs text-gray-500 dark:text-gray-400 mb-1">Status</div>
        <div className="font-medium capitalize flex items-center gap-2 text-gray-800 dark:text-gray-200">
          {job.status}
        </div>
      </div>
      <JobDatasetTile job={job} {...context} />
      <div className="p-4 bg-gray-50 dark:bg-gray-900/50 rounded-lg border border-gray-100 dark:border-gray-700">
        <div className="text-xs text-gray-500 dark:text-gray-400 mb-1">Duration</div>
        <div className="font-medium text-gray-800 dark:text-gray-200 font-mono">
          {job.start_time && job.end_time
            ? `${Math.round((new Date(job.end_time).getTime() - new Date(job.start_time).getTime()) / 1000)}s`
            : '-'}
        </div>
      </div>
      <div className="p-4 bg-gray-50 dark:bg-gray-900/50 rounded-lg border border-gray-100 dark:border-gray-700">
        <div className="text-xs text-gray-500 dark:text-gray-400 mb-1">Engine</div>
        <div className="font-medium text-gray-800 dark:text-gray-200">
          {/* Legacy jobs predate engine recording and trained on pandas. */}
          {job.engine === 'polars' ? 'Polars' : 'pandas'}
        </div>
      </div>
      <LeakageGateTile safety={safety} />
      <RefitAuditTile safety={safety} />
      <ScoreAdvisoryTile safety={safety} />
    </div>
  );
}

export function JobTimeline({ job }: { job: JobInfo }) {
  return (
    <div className="grid grid-cols-1 sm:grid-cols-3 gap-4 text-xs">
      <div>
        <span className="text-gray-500 dark:text-gray-400">Created:</span>
        <span className="ml-2 font-mono text-gray-700 dark:text-gray-300">
          {job.created_at ? new Date(job.created_at).toLocaleString() : '—'}
        </span>
      </div>
      <div>
        <span className="text-gray-500 dark:text-gray-400">Started:</span>
        <span className="ml-2 font-mono text-gray-700 dark:text-gray-300">
          {job.start_time ? new Date(job.start_time).toLocaleString() : 'Not started'}
        </span>
      </div>
      <div>
        <span className="text-gray-500 dark:text-gray-400">Ended:</span>
        <span className="ml-2 font-mono text-gray-700 dark:text-gray-300">
          {job.end_time ? new Date(job.end_time).toLocaleString() : 'Not finished'}
        </span>
      </div>
    </div>
  );
}

export function JobError({ job }: { job: JobInfo }) {
  return (
    <>
      {job.error && (
        <div className="p-4 bg-red-50 dark:bg-red-900/20 border border-red-100 dark:border-red-900/30 rounded-lg">
          <h3 className="text-sm font-medium text-red-800 dark:text-red-300 mb-2 flex items-center gap-2">
            <AlertCircle className="w-4 h-4" />
            Error Log
          </h3>
          <pre className="text-xs text-red-700 dark:text-red-400 whitespace-pre-wrap font-mono">
            {job.error}
          </pre>
        </div>
      )}
    </>
  );
}

function getStatusGridColumns(safety: JobSafety) {
  const { leakageGate, refitAudit, refitFallback } = safety;
  // 4 base tiles + one column per optional verdict/advisory tile.
  const extraTiles = (leakageGate ? 1 : 0) + (refitAudit ? 1 : 0) + (refitFallback ? 1 : 0);
  const smGridCols =
    extraTiles >= 3 ? 'sm:grid-cols-7' : extraTiles === 2 ? 'sm:grid-cols-6' : extraTiles === 1 ? 'sm:grid-cols-5' : 'sm:grid-cols-4';
  return smGridCols;
}
