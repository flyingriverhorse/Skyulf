import React from 'react';
import { ChevronDown, RefreshCw } from 'lucide-react';
import type { JobListSidebar } from '../JobListSidebar';
type SidebarProps = React.ComponentProps<typeof JobListSidebar>;

/** Pagination retains its compact icon-only sidebar presentation. */
export function JobPagination({ hasMore, isSidebarCollapsed, isLoading, loadMoreJobs }: Pick<SidebarProps, 'hasMore' | 'isSidebarCollapsed' | 'isLoading' | 'loadMoreJobs'>) {
  return <>
    {hasMore && (
      <div className="p-3 border-t border-gray-100 dark:border-gray-700 bg-gray-50/50 dark:bg-gray-900/30">
        <button
          onClick={() => loadMoreJobs()}
          disabled={isLoading}
          className={`w-full py-2 text-xs font-medium rounded-lg border shadow-sm transition-all duration-200 flex items-center justify-center gap-2 ${isSidebarCollapsed
              ? 'bg-transparent border-transparent text-blue-600 dark:text-blue-400 hover:bg-gray-100 dark:hover:bg-gray-800'
              : 'bg-white dark:bg-gray-800 border-gray-200 dark:border-gray-700 text-gray-700 dark:text-gray-200 hover:border-blue-400 dark:hover:border-blue-500 hover:text-blue-600 dark:hover:text-blue-400'
            } disabled:opacity-50 disabled:cursor-not-allowed`}
          title="Load More Runs"
        >
          {isLoading ? <RefreshCw className="w-3.5 h-3.5 animate-spin" /> : <ChevronDown className="w-3.5 h-3.5" />}
          {!isSidebarCollapsed && 'Load More Runs'}
        </button>
      </div>
    )}</>;
}
