import type { EvaluationData, EvaluationPopulation } from '../../types';

/** Preserve uncertainty when merged branches have no common original population. */
function populationDescription(coverage: EvaluationPopulation) {
  if (coverage.scored_rows == null) return 'Scored row count unavailable.';
  const scored = coverage.scored_rows.toLocaleString();
  if (coverage.input_rows == null || coverage.excluded_rows == null) {
    return `${scored} rows scored; original population and excluded count unavailable. ${coverage.reason ?? ''}`.trim();
  }
  return `${scored} of ${coverage.input_rows.toLocaleString()} rows scored; ${coverage.excluded_rows.toLocaleString()} excluded.`;
}

/** Explain which held-out rows contribute to the displayed model metrics. */
export function EvaluationCoverage({ data }: { data: EvaluationData | null }) {
  if (!data) return null;
  const splits = Object.entries(data.splits).filter(([name, split]) => name !== 'train' && split.coverage);
  if (splits.length === 0) return null;
  return <div className="rounded border border-slate-200 dark:border-slate-700 p-3 text-sm text-slate-600 dark:text-slate-300">
    <p>Metrics describe rows eligible under the configured preprocessing filters.</p>
    {splits.map(([name, split]) => {
      const coverage = split.coverage!;
      return <p key={name}>
        {name.charAt(0).toUpperCase() + name.slice(1)}: {populationDescription(coverage)}
      </p>;
    })}
  </div>;
}
