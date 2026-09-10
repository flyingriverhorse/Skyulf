import React from 'react';
import { Search } from 'lucide-react';
import type { ErrorSeverity } from '../../core/api/monitoring';
import { TIME_RANGES, SEVERITY_LABELS } from './errorLogFormatting';
import type { ErrorLogPageState } from './useErrorLogPage';

const FacetSelect: React.FC<{
  id: string;
  label: string;
  value: string;
  onChange: (v: string) => void;
  options: string[];
  optionLabel?: (v: string) => string;
}> = ({ id, label, value, onChange, options, optionLabel }) => (
  <div className="flex flex-col gap-1">
    <label htmlFor={id} className="sr-only">{label}</label>
    <select
      id={id}
      value={value}
      onChange={e => onChange(e.target.value)}
      className="px-3 py-1.5 text-xs rounded-lg border border-slate-200 dark:border-slate-700 bg-white dark:bg-slate-800 text-slate-600 dark:text-slate-300 focus:outline-none focus:ring-2 focus:ring-blue-500"
    >
      <option value="">{label}</option>
      {options.map(o => (
        <option key={o} value={o}>{optionLabel ? optionLabel(o) : o}</option>
      ))}
    </select>
  </div>
);

type ErrorLogFiltersProps = Pick<ErrorLogPageState,
  | 'view'
  | 'search'
  | 'setSearch'
  | 'timeRange'
  | 'setTimeRange'
  | 'showResolved'
  | 'setShowResolved'
  | 'hasActiveFilters'
  | 'eventsTotalUnfiltered'
  | 'eventsTotal'
  | 'severityFilter'
  | 'setSeverityFilter'
  | 'errorTypeFilter'
  | 'setErrorTypeFilter'
  | 'jobIdFilter'
  | 'setJobIdFilter'
  | 'nodeIdFilter'
  | 'setNodeIdFilter'
  | 'errorFacets'
  | 'pipelineFacets'
>;

export const ErrorLogFilters: React.FC<ErrorLogFiltersProps> = ({
  view,
  search,
  setSearch,
  timeRange,
  setTimeRange,
  showResolved,
  setShowResolved,
  hasActiveFilters,
  eventsTotalUnfiltered,
  eventsTotal,
  severityFilter,
  setSeverityFilter,
  errorTypeFilter,
  setErrorTypeFilter,
  jobIdFilter,
  setJobIdFilter,
  nodeIdFilter,
  setNodeIdFilter,
  errorFacets,
  pipelineFacets,
}) => (
  <div className={`flex flex-col gap-3 mb-4 ${view === 'issues' ? 'hidden' : ''}`}>
    <div className="flex items-center gap-3 flex-wrap">
      <div className="relative flex-1 max-w-sm">
        <Search size={14} className="absolute left-3 top-1/2 -translate-y-1/2 text-slate-400" />
        <input
          type="text"
          placeholder="Search errors, job id, node id…"
          value={search}
          onChange={e => setSearch(e.target.value)}
          className="w-full pl-9 pr-4 py-2 text-sm bg-white dark:bg-slate-800 border border-slate-200 dark:border-slate-700 rounded-lg focus:outline-none focus:ring-2 focus:ring-blue-500 dark:text-slate-200"
        />
      </div>
      {/* Time-range pills */}
      <div className="flex items-center gap-1 bg-slate-100 dark:bg-slate-800 rounded-lg p-1">
        {TIME_RANGES.map(r => (
          <button
            key={r.value}
            onClick={() => setTimeRange(r.value)}
            className={`px-3 py-1 text-xs rounded-md font-medium transition-colors ${timeRange === r.value
                ? 'bg-white dark:bg-slate-700 text-slate-800 dark:text-slate-100 shadow-sm'
                : 'text-slate-500 dark:text-slate-400 hover:text-slate-700 dark:hover:text-slate-200'
              }`}
          >
            {r.label}
          </button>
        ))}
      </div>
      {/* Show resolved toggle */}
      <button
        onClick={() => setShowResolved(v => !v)}
        className={`flex items-center gap-1.5 px-3 py-1.5 text-xs rounded-lg font-medium border transition-colors ${showResolved
            ? 'bg-slate-200 dark:bg-slate-700 border-slate-300 dark:border-slate-600 text-slate-700 dark:text-slate-200'
            : 'bg-white dark:bg-slate-800 border-slate-200 dark:border-slate-700 text-slate-500 dark:text-slate-400 hover:text-slate-700'
          }`}
      >
        {showResolved ? '✓ Showing resolved' : 'Show resolved'}
      </button>
      {hasActiveFilters && (
        <span className="text-xs text-slate-500 dark:text-slate-400">
          {eventsTotalUnfiltered === 0
            ? 'No history recorded yet'
            : `${eventsTotal} of ${eventsTotalUnfiltered} HTTP events match`}
        </span>
      )}
    </div>
    {/* Typed facets — composable, exact-match filters distinct from the generic search above */}
    <div className="flex items-center gap-2 flex-wrap">
      <FacetSelect
        id="facet-severity"
        label="All severities"
        value={severityFilter}
        onChange={v => setSeverityFilter(v as '' | ErrorSeverity)}
        options={errorFacets.severities}
        optionLabel={s => SEVERITY_LABELS[s as ErrorSeverity] ?? s}
      />
      <FacetSelect
        id="facet-error-type"
        label="All error types"
        value={errorTypeFilter}
        onChange={setErrorTypeFilter}
        options={errorFacets.error_types}
      />
      <FacetSelect
        id="facet-job-id"
        label="All job IDs"
        value={jobIdFilter}
        onChange={setJobIdFilter}
        options={errorFacets.job_ids}
      />
      <FacetSelect
        id="facet-node-id"
        label="All node IDs"
        value={nodeIdFilter}
        onChange={setNodeIdFilter}
        options={pipelineFacets.node_ids}
      />
    </div>
  </div>

);
