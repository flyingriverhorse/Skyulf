import React, { useMemo, useState } from 'react';
import { LayoutDashboard, FileText } from 'lucide-react';
import { JobInfo } from '../../../core/api/jobs';
import { useJobPolling, isTerminalStatus } from '../../../core/hooks/useJobPolling';
import { JobHeader } from './jobDetails/JobHeader';
import { JobLogs, useJobLogs } from './jobDetails/JobLogs';
import { JobChart, useJobChart } from './jobDetails/JobChart';
import { JobSafetyModals, useJobSafety } from './jobDetails/JobSafety';
import { JobStatusSection, JobTimeline, JobError } from './jobDetails/JobOverview';
import { JobRelatedRecords } from './jobDetails/JobRelatedRecords';
import { JobResults } from './jobDetails/JobResults';

interface JobDetailsViewProps {
  job: JobInfo;
  onBack: () => void;
  onClose: () => void;
  /** Route the caller is presenting this view from, carried into related-record links. */
  origin?: string;
  /** List filters to preserve on related-record links (mirrors OPS-007 `OperationalContext.filters`). */
  filters?: Record<string, string>;
}

function JobDetailsTabs({ job, activeTab, setActiveTab }: { job: JobInfo; activeTab: 'overview' | 'logs'; setActiveTab: (tab: 'overview' | 'logs') => void }) {
  return (
    <div className="flex border-b border-gray-200 dark:border-gray-700 px-4">
      <button
        className={`py-3 px-4 text-sm font-medium border-b-2 transition-colors flex items-center gap-2 ${activeTab === 'overview'
          ? 'border-blue-500 text-blue-600 dark:text-blue-400'
          : 'border-transparent text-gray-500 dark:text-gray-400 hover:text-gray-700 dark:hover:text-gray-200'
          }`}
        onClick={() => { setActiveTab('overview'); }}
      >
        <LayoutDashboard className="w-4 h-4" />
        Overview
      </button>
      <button
        className={`py-3 px-4 text-sm font-medium border-b-2 transition-colors flex items-center gap-2 ${activeTab === 'logs'
          ? 'border-blue-500 text-blue-600 dark:text-blue-400'
          : 'border-transparent text-gray-500 dark:text-gray-400 hover:text-gray-700 dark:hover:text-gray-200'
          }`}
        onClick={() => { setActiveTab('logs'); }}
      >
        <FileText className="w-4 h-4" />
        Live Logs
        {job.status === 'running' && <span className="w-2 h-2 rounded-full bg-green-500 animate-pulse" />}
      </button>
    </div>
  );
}

export const JobDetailsView: React.FC<JobDetailsViewProps> = ({ job: initialJob, onBack, onClose, origin, filters }) => {
  const [activeTab, setActiveTab] = useState<'overview' | 'logs'>('overview');
  // Poll the single job until terminal. We feed an empty array once
  // the initial job is already terminal so the hook does no work,
  // and otherwise let it stop itself on the next terminal snapshot.
  const pollIds = useMemo(
    () => (isTerminalStatus(initialJob.status) ? [] : [initialJob.job_id]),
    [initialJob.job_id, initialJob.status],
  );
  const { jobs: polledJobs } = useJobPolling(pollIds, { intervalMs: 2000 });
  const job: JobInfo = polledJobs[initialJob.job_id] ?? initialJob;
  const logControls = useJobLogs(job, activeTab);
  const chart = useJobChart(job);
  const safety = useJobSafety(job);
  const context = {
    ...(origin !== undefined ? { origin } : {}),
    ...(filters !== undefined ? { filters } : {}),
  };
  return (
    <div className="flex flex-col h-full">
      <JobHeader job={job} onBack={onBack} onClose={onClose} />
      <JobDetailsTabs job={job} activeTab={activeTab} setActiveTab={setActiveTab} />
      <div className="flex-1 overflow-y-auto p-6">
        {activeTab === 'overview' ? (
          <div className="space-y-6">
            <JobStatusSection job={job} safety={safety} {...context} />
            <JobChart job={job} chart={chart} />
            <JobTimeline job={job} />
            <JobRelatedRecords job={job} {...context} />
            <JobError job={job} />
            <JobResults job={job} />
          </div>
        ) : <JobLogs job={job} controls={logControls} />}
      </div>
      <JobSafetyModals safety={safety} />
    </div>
  );
};
