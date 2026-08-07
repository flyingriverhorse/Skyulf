import React from 'react';
import { AlertTriangle, RefreshCw } from 'lucide-react';

interface ErrorStateProps {
  error: string;
  onRetry?: () => void;
}

/** Shows an assertive error alert with an optional retry action. */
export const ErrorState: React.FC<ErrorStateProps> = ({ error, onRetry }) => {
  const errorId = React.useId();

  return (
    <div className="flex flex-col items-center justify-center py-12" role="alert" aria-atomic="true">
      <AlertTriangle className="w-8 h-8 text-red-500" aria-hidden="true" />
      <p id={errorId} className="mt-3 text-sm text-red-600 dark:text-red-400">
        {error}
      </p>
      {onRetry && (
        <button
          onClick={onRetry}
          aria-describedby={errorId}
          className="mt-4 inline-flex items-center gap-2 px-4 py-2 text-sm font-medium text-slate-700 dark:text-slate-200 bg-slate-100 dark:bg-slate-700 rounded-md hover:bg-slate-200 dark:hover:bg-slate-600 transition-colors"
        >
          <RefreshCw className="w-4 h-4" aria-hidden="true" />
          Retry
        </button>
      )}
    </div>
  );
};
