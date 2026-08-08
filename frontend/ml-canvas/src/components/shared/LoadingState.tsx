import React from 'react';
import { Loader2 } from 'lucide-react';

interface LoadingStateProps {
  message?: string;
}

/** Shows a polite loading status for async content. */
export const LoadingState: React.FC<LoadingStateProps> = ({ message = 'Loading...' }) => {
  return (
    <div className="flex flex-col items-center justify-center py-12" role="status" aria-atomic="true">
      <Loader2 className="w-8 h-8 animate-spin text-blue-500" aria-hidden="true" />
      <p className="mt-3 text-sm text-slate-500 dark:text-slate-400">{message}</p>
    </div>
  );
};
