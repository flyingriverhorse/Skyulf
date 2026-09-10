import { Search, Filter } from 'lucide-react';
import type { HistoryFilterValues } from './history';

interface FacetFiltersProps {
  statusFilter: string;
  setStatusFilter: (value: string) => void;
  modelFilter: string;
  setModelFilter: (value: string) => void;
  statuses: string[];
  modelTypes: string[];
}

interface HistoryFiltersProps extends FacetFiltersProps, HistoryFilterValues {
  setSearchQuery: (value: string) => void;
  showFilters: boolean;
  setShowFilters: (value: boolean) => void;
}

/** Select filters clear together while the independently controlled search remains intact. */
function FacetFilters({ statusFilter, setStatusFilter, modelFilter, setModelFilter, statuses, modelTypes }: FacetFiltersProps) {
  return (
    <div className="flex items-center gap-3">
      <div className="flex items-center gap-1.5">
        <span className="text-[10px] font-medium text-gray-500 dark:text-gray-400 uppercase tracking-wider">Status</span>
        <select
          value={statusFilter}
          onChange={(e) => setStatusFilter(e.target.value)}
          className="text-xs bg-gray-50 dark:bg-gray-900 border border-gray-200 dark:border-gray-700 rounded px-2 py-1 text-gray-700 dark:text-gray-200 focus:outline-none focus:ring-1 focus:ring-blue-500"
        >
          <option value="all">All</option>
          {statuses.map(s => (
            <option key={s} value={s}>{s.charAt(0).toUpperCase() + s.slice(1)}</option>
          ))}
        </select>
      </div>
      <div className="flex items-center gap-1.5">
        <span className="text-[10px] font-medium text-gray-500 dark:text-gray-400 uppercase tracking-wider">Model</span>
        <select
          value={modelFilter}
          onChange={(e) => setModelFilter(e.target.value)}
          className="text-xs bg-gray-50 dark:bg-gray-900 border border-gray-200 dark:border-gray-700 rounded px-2 py-1 text-gray-700 dark:text-gray-200 focus:outline-none focus:ring-1 focus:ring-blue-500"
        >
          <option value="all">All</option>
          {modelTypes.map(m => (
            <option key={m} value={m}>{m.replace(/_/g, ' ')}</option>
          ))}
        </select>
      </div>
      {(statusFilter !== 'all' || modelFilter !== 'all') && (
        <button
          onClick={() => { setStatusFilter('all'); setModelFilter('all'); }}
          className="text-[10px] text-blue-500 hover:underline"
        >
          Clear all
        </button>
      )}
    </div>
  );
}

/** Search and facet visibility are controlled by the drawer so detail navigation cannot reset them. */
export function HistoryFilters({ searchQuery, setSearchQuery, statusFilter, setStatusFilter,
  modelFilter, setModelFilter, showFilters, setShowFilters, statuses, modelTypes }: HistoryFiltersProps) {
  const activeFilterCount = (statusFilter !== 'all' ? 1 : 0) + (modelFilter !== 'all' ? 1 : 0);
  return (
    <div className="px-4 py-2 border-b border-gray-200 dark:border-gray-700 bg-white dark:bg-gray-800 space-y-2">
      <div className="flex items-center gap-2">
        <div className="relative flex-1">
          <Search className="absolute left-2.5 top-1/2 -translate-y-1/2 w-3.5 h-3.5 text-gray-400 pointer-events-none z-10" />
          <input
            type="text"
            placeholder="Search by job ID, dataset, or model..."
            value={searchQuery}
            onChange={(e) => setSearchQuery(e.target.value)}
            className="w-full pl-8 pr-3 py-1.5 text-xs bg-gray-50 dark:bg-gray-900 border border-gray-200 dark:border-gray-700 rounded-md focus:outline-none focus:ring-1 focus:ring-blue-500 text-gray-700 dark:text-gray-200 placeholder-gray-400"
          />
        </div>
        <button
          onClick={() => setShowFilters(!showFilters)}
          className={`flex items-center gap-1.5 px-2.5 py-1.5 text-xs rounded-md border transition-colors ${showFilters || activeFilterCount > 0
              ? 'bg-blue-50 dark:bg-blue-900/30 border-blue-300 dark:border-blue-700 text-blue-600 dark:text-blue-400'
              : 'bg-gray-50 dark:bg-gray-900 border-gray-200 dark:border-gray-700 text-gray-500 dark:text-gray-400 hover:bg-gray-100 dark:hover:bg-gray-800'
            }`}
        >
          <Filter className="w-3.5 h-3.5" />
          Filters
          {(activeFilterCount > 0) && (
            <span className="w-4 h-4 flex items-center justify-center bg-blue-500 text-white rounded-full text-[10px] font-bold">
              {activeFilterCount}
            </span>
          )}
        </button>
      </div>
      {showFilters && (
        <FacetFilters statusFilter={statusFilter} setStatusFilter={setStatusFilter}
          modelFilter={modelFilter} setModelFilter={setModelFilter}
          statuses={statuses} modelTypes={modelTypes} />
      )}
    </div>

  );
}
