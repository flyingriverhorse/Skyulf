import React from 'react';
import { AlertTriangle } from 'lucide-react';
import { LoadingState } from '../components/shared';
import { useErrorLogPage, type ErrorLogPageState } from './errorLog/useErrorLogPage';
import { ErrorLogHeader, ErrorLogTabs } from './errorLog/ErrorLogHeader';
import { ErrorLogOverview } from './errorLog/ErrorLogOverview';
import { ErrorLogFilters } from './errorLog/ErrorLogFilters';
import { ErrorIssueTable } from './errorLog/ErrorIssueTable';
import { ErrorEventTable } from './errorLog/ErrorEventTable';
import { TracebackModal } from './errorLog/ErrorDiagnostics';

const ErrorLogContent: React.FC<{ page: ErrorLogPageState }> = ({ page }) => {
  if (page.loading) return <LoadingState message="Loading error events…" />;
  if (page.error) return (
    <div className="flex items-center gap-3 p-4 bg-amber-50 dark:bg-amber-900/20 border border-amber-200 dark:border-amber-800 rounded-xl text-amber-700 dark:text-amber-400 text-sm">
      <AlertTriangle size={16} />
      {page.error}
    </div>
  );
  if (page.view === 'issues') return <ErrorIssueTable {...page} />;
  return <ErrorEventTable {...page} />;
};

export const ErrorLogPage: React.FC = () => {
  const page = useErrorLogPage();
  return (
    <div className="p-6 max-w-7xl mx-auto">
      <ErrorLogHeader {...page} />
      <ErrorLogOverview {...page} />
      <ErrorLogTabs {...page} />
      <ErrorLogFilters {...page} />
      <ErrorLogContent page={page} />
      {page.modal && (
        <TracebackModal
          event={page.modal}
          onClose={() => page.setModal(null)}
          origin="/errors"
          timeRange={page.operationalTimeRange}
          filters={page.linkFilters}
        />
      )}
    </div>
  );
};
