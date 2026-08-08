import React from 'react';
import { AlertTriangle, RefreshCw } from 'lucide-react';

interface ErrorStateProps {
  error: string;
  onRetry?: (() => void | Promise<unknown>) | undefined;
}

/** Shows an assertive error alert with an optional retry action. */
export const ErrorState: React.FC<ErrorStateProps> = ({ error, onRetry }) => {
  const errorId = React.useId();
  const [isRetrying, setIsRetrying] = React.useState(false);
  const retryLockRef = React.useRef(false);

  const handleRetry = async () => {
    if (!onRetry || retryLockRef.current) return;

    let result: unknown;
    try {
      result = onRetry();
    } catch (error) {
      console.error('Retry failed', error);
      return;
    }

    if (result && typeof (result as PromiseLike<unknown>).then === 'function') {
      retryLockRef.current = true;
      setIsRetrying(true);
      try {
        await result;
      } catch (error) {
        console.error('Retry failed', error);
      } finally {
        retryLockRef.current = false;
        setIsRetrying(false);
      }
    }
  };

  return (
    <div className="flex flex-col items-center justify-center py-12" role="alert" aria-atomic="true">
      <AlertTriangle className="w-8 h-8 text-red-500" aria-hidden="true" />
      <p id={errorId} className="mt-3 text-sm text-red-600 dark:text-red-400">
        {error}
      </p>
      {onRetry && (
        <button
          type="button"
          onClick={() => { void handleRetry(); }}
          disabled={isRetrying}
          aria-describedby={errorId}
          className="mt-4 inline-flex items-center gap-2 px-4 py-2 text-sm font-medium text-slate-700 dark:text-slate-200 bg-slate-100 dark:bg-slate-700 rounded-md hover:bg-slate-200 dark:hover:bg-slate-600 transition-colors disabled:opacity-50 disabled:cursor-not-allowed"
        >
          <RefreshCw className="w-4 h-4" aria-hidden="true" />
          Retry
        </button>
      )}
    </div>
  );
};
