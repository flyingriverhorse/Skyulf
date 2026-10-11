import React from 'react';
import { LoadingLogo } from './LoadingLogo';

interface LoadingStateProps {
  message?: string;
}

/** Shows a polite loading status for async content. */
export const LoadingState: React.FC<LoadingStateProps> = ({ message = 'Loading...' }) => {
  return (
    <div className="flex flex-col items-center justify-center py-12" role="status" aria-atomic="true">
      <LoadingLogo />
      <p className="mt-3 text-sm text-slate-500 dark:text-slate-400">{message}</p>
    </div>
  );
};
