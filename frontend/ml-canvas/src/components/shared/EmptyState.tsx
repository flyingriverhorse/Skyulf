import React, { ReactNode } from 'react';
import { Inbox } from 'lucide-react';

interface EmptyStateProps {
  icon?: ReactNode;
  title: string;
  description?: string;
  action?: ReactNode;
}

/** Shows a polite empty-state message with a hidden decorative icon. */
export const EmptyState: React.FC<EmptyStateProps> = ({ icon, title, description, action }) => {
  const decorativeIcon = icon ? <span aria-hidden="true">{icon}</span> : <Inbox className="w-12 h-12 text-slate-300 dark:text-slate-600" aria-hidden="true" />;

  return (
    <div className="flex flex-col items-center justify-center py-12">
      <div role="status" aria-atomic="true" className="flex flex-col items-center">
        {decorativeIcon}
        <p className="mt-3 text-lg font-medium text-slate-900 dark:text-slate-100">{title}</p>
        {description && (
          <p className="mt-1 text-sm text-slate-500 dark:text-slate-400">{description}</p>
        )}
      </div>
      {action && <div className="mt-4">{action}</div>}
    </div>
  );
};
