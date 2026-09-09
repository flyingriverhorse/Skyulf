import React from 'react';
import { Bug } from 'lucide-react';
import type { GroupedIssue } from '../../core/api/monitoring';
import { EmptyState } from '../../components/shared';
import { relativeTime } from './errorLogFormatting';
import type { PipelineIssue } from './errorLogAggregation';

const GroupedIssueRow: React.FC<{ issue: GroupedIssue; onViewSample: (id: number) => void }> = ({ issue, onViewSample }) => (
  <tr className="border-b border-slate-100 dark:border-slate-800 hover:bg-slate-50 dark:hover:bg-slate-800/50 transition-colors">
    <td className="px-4 py-3">
      <span className="inline-flex items-center justify-center min-w-[1.75rem] h-6 px-1.5 rounded-full bg-red-100 dark:bg-red-900/30 text-red-700 dark:text-red-400 text-xs font-bold">
        {issue.count}
      </span>
    </td>
    <td className="px-4 py-3 font-mono text-sm text-slate-700 dark:text-slate-300 max-w-[200px] truncate">
      {issue.error_type}
    </td>
    <td className="px-4 py-3 text-xs text-slate-500 dark:text-slate-400 font-mono max-w-[220px] truncate">
      {issue.route || '—'}
    </td>
    <td className="px-4 py-3 text-xs text-slate-400 whitespace-nowrap">
      {relativeTime(issue.last_seen)}
    </td>
    <td className="px-4 py-3 text-xs text-slate-400 whitespace-nowrap">
      {relativeTime(issue.first_seen)}
    </td>
    <td className="px-4 py-3 text-right">
      <button
        className="text-xs text-blue-500 hover:underline"
        onClick={() => onViewSample(issue.sample_id)}
      >
        View sample
      </button>
    </td>
  </tr>
);

const PipelineIssueRow: React.FC<{ issue: PipelineIssue }> = ({ issue }) => (
  <tr className="border-b border-slate-100 dark:border-slate-800 hover:bg-slate-50 dark:hover:bg-slate-800/50 transition-colors">
    <td className="px-4 py-3">
      <span className="inline-flex items-center justify-center min-w-[1.75rem] h-6 px-1.5 rounded-full bg-amber-100 dark:bg-amber-900/30 text-amber-700 dark:text-amber-400 text-xs font-bold">
        {issue.count}
      </span>
    </td>
    <td className="px-4 py-3 font-mono text-sm text-slate-700 dark:text-slate-300 max-w-[200px] truncate">
      {issue.node_type}
    </td>
    <td className="px-4 py-3 text-xs text-amber-600 dark:text-amber-400 font-medium">
      pipeline
    </td>
    <td className="px-4 py-3 text-xs text-slate-400 whitespace-nowrap">
      {issue.last_seen ? relativeTime(issue.last_seen) : '\u2014'}
    </td>
    <td className="px-4 py-3 text-xs text-slate-400 whitespace-nowrap">
      {issue.first_seen ? relativeTime(issue.first_seen) : '\u2014'}
    </td>
    <td className="px-4 py-3" />
  </tr>
);

export const ErrorIssueTable: React.FC<{ grouped: GroupedIssue[]; pipelineIssues: PipelineIssue[]; handleViewSample: (id: number) => void }> = ({ grouped, pipelineIssues, handleViewSample }) => (
  grouped.length === 0 && pipelineIssues.length === 0 ? (
    <EmptyState
      icon={<Bug size={40} className="text-slate-300" />}
      title="No open issues"
      description="All errors have been resolved, or none have been recorded yet."
    />
  ) : (
    <div className="bg-white dark:bg-slate-800 rounded-xl border border-slate-200 dark:border-slate-700 overflow-hidden">
      <table className="w-full text-sm">
        <thead>
          <tr className="text-left text-xs text-slate-500 dark:text-slate-400 bg-slate-50 dark:bg-slate-900/50 border-b border-slate-200 dark:border-slate-700">
            <th className="px-4 py-3">Count</th>
            <th className="px-4 py-3">Type</th>
            <th className="px-4 py-3">Route</th>
            <th className="px-4 py-3">Last seen</th>
            <th className="px-4 py-3">First seen</th>
            <th className="px-4 py-3" />
          </tr>
        </thead>
        <tbody>
          {pipelineIssues.map((p, i) => (
            <PipelineIssueRow key={`pi-${i}`} issue={p} />
          ))}
          {grouped.map((g, i) => (
            <GroupedIssueRow key={i} issue={g} onViewSample={handleViewSample} />
          ))}
        </tbody>
      </table>
    </div>
  )

);
