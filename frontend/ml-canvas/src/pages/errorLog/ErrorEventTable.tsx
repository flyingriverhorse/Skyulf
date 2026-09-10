import React from 'react';
import { Bug } from 'lucide-react';
import { EmptyState } from '../../components/shared';
import { ErrorRow, PipelineRow } from './ErrorEventRows';
import type { ErrorLogPageState } from './useErrorLogPage';

type ErrorEventTableProps = Pick<ErrorLogPageState,
  | 'events'
  | 'pipelineLogs'
  | 'hasActiveFilters'
  | 'eventsTotalUnfiltered'
  | 'operationalTimeRange'
  | 'linkFilters'
  | 'setModal'
  | 'handleResolve'
>;

export const ErrorEventTable: React.FC<ErrorEventTableProps> = ({
  events,
  pipelineLogs,
  hasActiveFilters,
  eventsTotalUnfiltered,
  operationalTimeRange,
  linkFilters,
  setModal,
  handleResolve,
}) => (
  events.length === 0 && pipelineLogs.length === 0 ? (
    <EmptyState
      icon={<Bug size={40} className="text-slate-300" />}
      title={hasActiveFilters ? 'No matching errors' : 'No errors recorded'}
      description={
        hasActiveFilters
          ? eventsTotalUnfiltered === 0
            ? 'No error events have been recorded yet — nothing to search or filter.'
            : `No events match the current search/facets out of ${eventsTotalUnfiltered} recorded. Try widening the time range or clearing a facet.`
          : 'Any unhandled 5xx or failed pipeline will appear here automatically.'
      }
    />
  ) : (
    <div className="bg-white dark:bg-slate-800 rounded-xl border border-slate-200 dark:border-slate-700 overflow-hidden">
      <table className="w-full text-sm">
        <thead>
          <tr className="text-left text-xs text-slate-500 dark:text-slate-400 bg-slate-50 dark:bg-slate-900/50 border-b border-slate-200 dark:border-slate-700">
            <th className="px-4 py-3 w-8" />
            <th className="px-4 py-3">Severity</th>
            <th className="px-4 py-3">Code</th>
            <th className="px-4 py-3">Type</th>
            <th className="px-4 py-3">Message</th>
            <th className="px-4 py-3">Target</th>
            <th className="px-4 py-3">When</th>
            <th className="px-4 py-3" />
          </tr>
        </thead>
        <tbody>
          {pipelineLogs.map(l => (
            <PipelineRow
              key={`pl-${l.id}`}
              log={l}
              origin="/errors"
              timeRange={operationalTimeRange}
              filters={linkFilters}
            />
          ))}
          {events.map(e => (
            <ErrorRow
              key={e.id}
              event={e}
              onExpand={setModal}
              onResolve={handleResolve}
              origin="/errors"
              timeRange={operationalTimeRange}
              filters={linkFilters}
            />
          ))}
        </tbody>
      </table>
    </div>
  )
);
