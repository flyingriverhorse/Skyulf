import { useId, useState } from 'react';
import { AlertTriangle, ArrowRight, Search } from 'lucide-react';
import {
  placementLabels,
  preprocessingPlacementCatalog,
  type PreprocessingPlacement,
} from './preprocessingPlacementCatalog';

const placementClasses: Record<PreprocessingPlacement, string> = {
  before: 'bg-emerald-50 text-emerald-800 dark:bg-emerald-900/30 dark:text-emerald-200',
  conditional: 'bg-amber-50 text-amber-800 dark:bg-amber-900/30 dark:text-amber-200',
  after: 'bg-rose-50 text-rose-800 dark:bg-rose-900/30 dark:text-rose-200',
  time: 'bg-sky-50 text-sky-800 dark:bg-sky-900/30 dark:text-sky-200',
  split: 'bg-slate-100 text-slate-700 dark:bg-slate-800 dark:text-slate-200',
};

/** Searchable placement reference for the full preprocessing registry, including aliases. */
export const PreprocessingPlacementGuide = () => {
  const guideId = useId();
  const [query, setQuery] = useState('');
  const [placement, setPlacement] = useState('all');
  const search = query.trim().toLowerCase();
  const entries = preprocessingPlacementCatalog.filter((entry) =>
    (placement === 'all' || entry.placement === placement) &&
    `${entry.id} ${entry.name} ${entry.category} ${placementLabels[entry.placement]} ${entry.rule}`
      .toLowerCase().includes(search),
  );

  return (
    <div className="text-sm text-slate-600 dark:text-slate-300">
      <section className="p-5 space-y-3 border-b border-slate-100 dark:border-slate-800">
        <h3 className="font-semibold text-slate-900 dark:text-slate-100">Place preprocessing by what it learns</h3>
        <p>
          Fixed rules can run before the row split. Anything that learns statistics, categories,
          selected features or mappings must fit on training rows after the split, then reuse that
          fitted state for validation, test and inference.
        </p>
        <div className="flex flex-wrap items-center gap-2 rounded-lg border border-slate-200 bg-slate-50 p-3 text-xs font-medium dark:border-slate-700 dark:bg-slate-800/50">
          {['Loader', 'Fixed rules', 'Train-Test Split', 'Learned preprocessing', 'Train / Tune'].map((step, index) => (
            <span key={step} className="inline-flex items-center gap-2">
              {index > 0 && <ArrowRight className="h-3 w-3 text-slate-400" aria-hidden="true" />}
              {step}
            </span>
          ))}
        </div>
        <p>
          <strong className="text-slate-900 dark:text-slate-100">Cross-validation adds another boundary.</strong>{' '}
          Fit learned preprocessing again inside each training fold. Fitting it once on the whole
          outer training set does not protect the inner validation folds. Resampling belongs only
          on training rows, never validation or test rows.
        </p>
        <div className="flex gap-2 rounded-lg bg-amber-50 p-3 text-amber-900 dark:bg-amber-900/20 dark:text-amber-200">
          <AlertTriangle className="h-4 w-4 shrink-0 mt-0.5" aria-hidden="true" />
          <p>
            Placement is not a universal leakage guarantee. Fixed formulas can still use the
            target or future information. Check feature provenance, entity independence, time
            ordering and which inputs exist when making each prediction.
          </p>
        </div>
      </section>

      <section className="p-5 space-y-3 border-b border-slate-100 dark:border-slate-800">
        <h3 className="font-semibold text-slate-900 dark:text-slate-100">Read the configured operation</h3>
        <p>
          A node name or registry flag alone is not enough. General Transformation can apply a
          fixed logarithm or fit Yeo-Johnson. Feature Generation can calculate a ratio or learn a
          group lookup. If any operation in a mixed list learns, place the whole node after the split.
        </p>
        <p className="text-xs text-slate-500 dark:text-slate-400">
          Open the Split &amp; Merge tab for split boundaries, column selection, branch examples
          and audit details.
        </p>
      </section>

      <section aria-labelledby={`${guideId}-catalog`}>
        <div className="sticky top-0 z-10 space-y-3 border-b border-slate-200 bg-white p-5 dark:border-slate-700 dark:bg-slate-900">
          <div className="flex flex-wrap items-baseline justify-between gap-2">
            <h3 id={`${guideId}-catalog`} className="font-semibold text-slate-900 dark:text-slate-100">Node placement catalog</h3>
            <p className="text-xs text-slate-500 dark:text-slate-400">60 transformations + 2 row-split registrations</p>
          </div>
          <div className="flex flex-col gap-2 sm:flex-row">
            <div className="relative flex-1">
              <label htmlFor={`${guideId}-search`} className="sr-only">Search preprocessing nodes</label>
              <Search className="pointer-events-none absolute left-3 top-3 h-4 w-4 text-slate-400" aria-hidden="true" />
              <input
                id={`${guideId}-search`}
                type="search"
                value={query}
                onChange={(event) => setQuery(event.target.value)}
                placeholder="Node, registry ID or operation..."
                className="w-full rounded-lg border border-slate-300 bg-white py-2 pl-9 pr-3 text-sm text-slate-900 focus-ring dark:border-slate-600 dark:bg-slate-800 dark:text-slate-100"
              />
            </div>
            <label htmlFor={`${guideId}-placement`} className="sr-only">Filter by placement</label>
            <select
              id={`${guideId}-placement`}
              value={placement}
              onChange={(event) => setPlacement(event.target.value)}
              className="rounded-lg border border-slate-300 bg-white px-3 py-2 text-sm text-slate-900 focus-ring dark:border-slate-600 dark:bg-slate-800 dark:text-slate-100"
            >
              <option value="all">All placements</option>
              {Object.entries(placementLabels).map(([value, label]) => (
                <option key={value} value={value}>{label}</option>
              ))}
            </select>
          </div>
          <p role="status" className="text-xs text-slate-500 dark:text-slate-400">
            {entries.length} of {preprocessingPlacementCatalog.length} registered types. Aliases are listed separately.
          </p>
        </div>
        <div className="p-5 space-y-3">
          {entries.map((entry) => (
            <article key={entry.id} aria-labelledby={`${guideId}-${entry.id}`} className="rounded-lg border border-slate-200 p-4 dark:border-slate-700">
              <div className="flex flex-wrap items-start justify-between gap-2">
                <div className="min-w-0">
                  <h4 id={`${guideId}-${entry.id}`} className="font-semibold text-slate-900 dark:text-slate-100">{entry.name}</h4>
                  <code className="break-all text-xs text-slate-500 dark:text-slate-400">{entry.id}</code>
                </div>
                <span className={`rounded-full px-2 py-1 text-xs font-medium ${placementClasses[entry.placement]}`}>
                  {placementLabels[entry.placement]}
                </span>
              </div>
              <p className="mt-2 leading-relaxed">{entry.rule}</p>
              <p className="mt-2 text-xs text-slate-500 dark:text-slate-400">{entry.category}</p>
            </article>
          ))}
          {entries.length === 0 && (
            <div className="rounded-lg border border-dashed border-slate-300 p-6 text-center dark:border-slate-600">
              <p>No preprocessing nodes match these filters.</p>
              <button
                type="button"
                onClick={() => { setQuery(''); setPlacement('all'); }}
                className="mt-3 rounded px-3 py-1.5 font-medium text-indigo-600 hover:bg-indigo-50 focus-ring dark:text-indigo-400 dark:hover:bg-slate-800"
              >
                Clear filters
              </button>
            </div>
          )}
        </div>
      </section>
    </div>
  );
};
